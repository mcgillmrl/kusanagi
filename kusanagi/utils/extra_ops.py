from __future__ import absolute_import, division, print_function

import warnings

import pkg_resources
import numpy as np
from numpy.linalg.linalg import LinAlgError

import theano
from theano import Op, config, tensor
from theano.scalar import (bool as bool_t,
                           neq as scalar_neq_op,
                           log as scalar_log_op)
from theano.gof import COp, ParamsType
from theano.gpuarray import GpuArrayType

from theano.gpuarray.basic_ops import(CGpuKernelBase, as_gpuarray_variable,
                                      gpu_contiguous, gpuarray_helper_inc_dir,
                                      infer_context_name)
from theano.gpuarray.elemwise import GpuElemwise
from theano.gpuarray.subtensor import GpuExtractDiag                        
from theano.gpuarray.type import gpu_context_type
from theano.gpuarray.opt import register_opt, register_opt2, op_lifter

try:
    import pygpu
    from pygpu.basic import triu, tril
    pygpu_available = True
except ImportError:
    pygpu_available = False

cusolver_available = False
try:
    import skcuda
    from skcuda import cusolver, linalg as sklinalg
    cusolver_available = True
except (ImportError, OSError, RuntimeError, pkg_resources.DistributionNotFound):
    pass

cublas_available = False
try:
    from skcuda import cublas
    cublas_available = True
except (ImportError, OSError, RuntimeError, pkg_resources.DistributionNotFound):
    pass

if cusolver_available:
    # Add cusolver call as it is missing in skcuda
    # SPOTRS
    cusolver._libcusolver.cusolverDnSpotrs.restype = int
    cusolver._libcusolver.cusolverDnSpotrs.argtypes = [cusolver.ctypes.c_void_p,
                                                       cusolver.ctypes.c_int,
                                                       cusolver.ctypes.c_int,
                                                       cusolver.ctypes.c_int,
                                                       cusolver.ctypes.c_void_p,
                                                       cusolver.ctypes.c_int,
                                                       cusolver.ctypes.c_void_p,
                                                       cusolver.ctypes.c_int,
                                                       cusolver.ctypes.c_void_p]

    def cusolverDnSpotrs(handle, uplo, n, nrhs, A, lda,
                         B, ldb, devInfo):
        """
        Solve real single precision linear system for hermitian matrices.
        References
        ----------
        `cusolverDn<t>potrs <http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-potrs>`_
        """

        status = cusolver._libcusolver.cusolverDnSpotrs(handle, uplo, n, nrhs,
                                                        int(A), lda, int(B),
                                                        ldb, int(devInfo))
        cusolver.cusolverCheckStatus(status)


def attach_cusolver_handle_to_context(ctx):
    handle = getattr(ctx, 'cusolver_handle', None)
    if handle is None:
        with ctx:
            ctx.cusolver_handle = cusolver.cusolverDnCreate()


class GpuLU(Op):
    """
    CUSOLVER GPU LU factorization Op.
    Given a non-singular square matrix, computes its LU
    factorization. Useful for computing the matrix determinant
    Parameters
    ----------
    """
    __props__ = ('inplace', 'check_output')

    def __init__(self, inplace=False, check_output=True):
        self.inplace = inplace
        self.check_output = check_output
        if self.inplace:
            self.destroy_map = {0: [0]}
        super(GpuLU, self).__init__()

    def clone_inplace(self):
        return self.__class__(lower=self.lower, inplace=True)

    def make_node(self, inp):
        if not cusolver_available:
            raise RuntimeError('CUSOLVER is not available and '
                               'GpuLU Op can not be constructed.')
        if skcuda.__version__ <= '0.5.1':
            warnings.warn('The GpuLU op requires scikit-cuda > 0.5.1 to work with CUDA 8')
        if not pygpu_available:
            raise RuntimeError('Missing pygpu or triu/tril functions.'
                               'Install or update libgpuarray.')
        context_name = infer_context_name(inp)

        inp = as_gpuarray_variable(inp, context_name)

        inp = gpu_contiguous(inp)

        # this op can only operate on float32 matrices
        # because of current implementation of triu/tril.
        # TODO: support float64
        assert inp.ndim == 2
        assert inp.dtype == 'float32'

        # outputs LU in a single matrix, and a pivots array
        pivots_type = GpuArrayType('int32',
                                   broadcastable=inp[0].broadcastable,
                                   context_name=context_name)()
        return theano.Apply(self, [inp], [inp.type(), pivots_type])

    def prepare_node(self, node, storage_map, compute_map, impl):
        ctx = node.inputs[0].type.context
        attach_cusolver_handle_to_context(ctx)

    def perform(self, node, inputs, outputs):
        context = inputs[0][0].context

        # Input matrix.
        A = inputs[0]

        l, n = A.shape
        if l != n:
            raise ValueError('A must be a square matrix')

        lda = max(1, n)

        # cusolver operates on F ordered matrices
        if not self.inplace:
            LU = pygpu.array(A, copy=True, order='F')
        else:
            LU = A.T if A.flags['C_CONTIGUOUS'] else A

        LU_ptr = LU.gpudata

        with context:
            workspace_size = cusolver.cusolverDnSgetrf_bufferSize(
                context.cusolver_handle, n, n, LU_ptr, lda)

            workspace = pygpu.zeros(workspace_size, dtype='float32',
                                    context=context)

            pivots = pygpu.zeros(n, dtype='int32', context=context)

            dev_info = pygpu.zeros((1,), dtype='int32', context=context)

            workspace_ptr = workspace.gpudata
            pivots_ptr = pivots.gpudata
            dev_info_ptr = dev_info.gpudata

            cusolver.cusolverDnSgetrf(
                context.cusolver_handle, n, n, LU_ptr, lda, workspace_ptr,
                pivots_ptr, dev_info_ptr)

            if self.check_output:
                val_dev_info = np.asarray(dev_info)[0]
                if val_dev_info > 0:
                    raise LinAlgError('LU decomposition failed')

            outputs[1][0] = pivots

        outputs[0][0] = LU


def gpu_det(A, inplace=False):
    """
        Computes the matrix determinant on the GPU using its LU
        factorization; i.e. if A = PLU then det(A) = (-1)**p*prod(L)*prod(U),
        where p is the number of permuted rows defined by P
        Parameters
        ----------
        A : square matrix
        Returns
        -------
        det : determinant of A
    """
    print("GPU_DET!!!!!!!!!!!!!!!")
    LU, pivots = GpuLU(inplace=inplace)(A)
    idx = theano.tensor.arange(1, A.shape[0] + 1, dtype=pivots.dtype)
    p = GpuElemwise(scalar_neq_op)(pivots, idx).sum().astype(A.dtype)
    diag = GpuExtractDiag(view=True)(LU)
    det = diag.prod() * ((-1)**(p))
    return det


def gpu_slogdet(A, inplace=False):
    """
        Computes the logartihm of the matrix determinant on the GPU using
        its LU factorization; i.e. if A = PLU then
        det(A) = (-1)**p*prod(L)*prod(U),
        where p is the number of permuted rows defined by P
        Parameters
        ----------
        A : square matrix
        Returns
        -------
        s, logabsdet : sign of the determinant and log(|det(A)|)
    """
    LU, pivots = GpuLU(inplace=inplace)(A)
    idx = theano.tensor.arange(1, A.shape[0] + 1, dtype=pivots.dtype)
    p = GpuElemwise(scalar_neq_op)(pivots, idx).sum().astype(A.dtype)
    logabsdet = GpuElemwise(scalar_log_op)(GpuExtractDiag(view=True)(LU)).sum()
    return ((-1)**(p)), logabsdet


# determinant optimizer
@register_opt('fast_compile')
@op_lifter([theano.tensor.nlinalg.Det])
@register_opt2([theano.tensor.nlinalg.Det], 'fast_compile')
def local_gpu_det(op, context_name, inputs, outputs):
    print("GPU_DET OPT !!!!!!!!!!!!!!!")
    if not cusolver_available:
        return
    if inputs[0].dtype not in ['float16', 'float32']:
        return
    if inputs[0].dtype == 'float16':
        return gpu_det(inputs[0].astype('float32')).astype('float16')
    else:
        return gpu_det(inputs[0])

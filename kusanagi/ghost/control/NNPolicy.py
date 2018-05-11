import lasagne
import numpy as np
import theano
import theano.tensor as tt

from kusanagi.ghost.regression import BNN, mlp, dropout_mlp, layers
from kusanagi.ghost.control.saturation import sfunc, tanhSat as sat
from functools import partial


# NN controller
class NNPolicy(BNN):
    def __init__(self, input_dims, maxU=[10], minU=None, angle_dims=[],
                 sat_func=sat, name='NNPolicy', filename=None, **kwargs):
        # policy output noise is not input-dependent by default
        kwargs['heteroscedastic'] = kwargs.get('heteroscedastic', False)
        self.maxU = np.array(maxU, dtype=theano.config.floatX)
        self.minU = (np.array(minU, dtype=theano.config.floatX)
                     if minU is not None else -self.maxU)
        self.angle_dims = angle_dims
        self.D = input_dims + len(self.angle_dims)
        self.E = len(maxU)

        self.sat_func = None
        if callable(sat_func):
            scale = 0.5*(self.maxU - self.minU)
            bias = self.minU
            sat_func = partial(sat_func, e=scale)
            self.sat_func = partial(
                sfunc, scale + bias, sat_func)

        super(NNPolicy, self).__init__(self.D, self.E, name=name,
                                       filename=filename, **kwargs)

    def predict_symbolic(self, mx, Sx=None, **kwargs):
        if self.network_spec is None:
            self.network_spec = dropout_mlp(
                input_dims=self.D,
                output_dims=self.E,
                hidden_dims=[50]*2,
                p=0.1, p_input=0.0,
                nonlinearities=lasagne.nonlinearities.rectify,
                output_nonlinearity=lasagne.nonlinearities.linear,
                dropout_class=layers.DenseDropoutLayer,
                name=self.name)

        if self.network is None:
            params = self.network_params\
                     if self.network_params is not None\
                     else {}
            self.build_network(self.network_spec,
                               params=params,
                               name=self.name)

        if Sx is not None:
            # generate random samples from input (assuming gaussian
            # distributed inputs)
            # standard uniform samples (one sample per network sample)
            z_std = self.m_rng.normal((self.n_samples, self.D))

            # scale and center particles
            Lx = tt.slinalg.cholesky(Sx)
            x = mx + z_std.dot(Lx.T)
        else:
            x = mx[None, :] if mx.ndim == 1 else mx

        # we are going to apply the saturation function
        # after whitening the outputs
        return_samples = kwargs.get('return_samples', True)
        kwargs['return_samples'] = True

        y, sn = super(NNPolicy, self).predict_symbolic(x, None, **kwargs)
        if callable(self.sat_func):
            y = self.sat_func(y)

        if return_samples:
            return y, sn
        else:
            n = tt.cast(y.shape[0], dtype=theano.config.floatX)
            # empirical mean
            M = y.mean(axis=0)
            # empirical covariance
            S = y.T.dot(y)/n - tt.outer(M, M)
            # noise
            S += tt.diag((sn**2).mean(axis=0))
            # empirical input output covariance
            if Sx is not None:
                C = x.T.dot(y)/n - tt.outer(mx, M)
            else:
                C = tt.zeros((self.D, self.E))
            return [M, S, C]

    def evaluate(self, m, s=None, t=None, symbolic=False, **kwargs):
        # by default, sample internal params (e.g. dropout masks)
        # at every evaluation
        kwargs['iid_per_eval'] = kwargs.get('iid_per_eval', True)
        kwargs['whiten_inputs'] = kwargs.get('whiten_inputs', True)
        kwargs['whiten_outputs'] = kwargs.get('whiten_outputs', True)
        if s is None:
            kwargs['return_samples'] = kwargs.get('return_samples', True)
        kwargs['deterministic'] = kwargs.get('deterministic', False)
        if symbolic:
            ret = self.predict_symbolic(m, s, **kwargs)
        else:
            ret = self.predict(m, s, **kwargs)
        return ret

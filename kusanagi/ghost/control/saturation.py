import lasagne
import theano.tensor as tt
import numpy as np
import theano


def gSin(m, v, i=None, e=None):
    D = m.shape[0]
    if i is None:
        i = tt.arange(D)
    if e is None:
        e = tt.ones((D,))
    elif e.__class__ is list:
        e = tt.as_tensor_variable(np.array(e)).flatten()
    elif e.__class__ is np.array:
        e = tt.as_tensor_variable(e).flatten()

    # compute the output mean
    mi = m[i]
    vi = v[i, :][:, i]
    vii = v[i, i]
    exp_vii_h = tt.exp(-vii/2)
    M = exp_vii_h*tt.sin(mi)

    # output covariance
    vii_c = vii.dimshuffle(0, 'x')
    vii_r = vii.dimshuffle('x', 0)
    lq = -0.5*(vii_c+vii_r)
    q = tt.exp(lq)
    exp_lq_p_vi = tt.exp(lq+vi)
    exp_lq_m_vi = tt.exp(lq-vi)
    mi_c = mi.dimshuffle(0, 'x')
    mi_r = mi.dimshuffle('x', 0)
    U1 = (exp_lq_p_vi - q)*(tt.cos(mi_c-mi_r))
    U2 = (exp_lq_m_vi - q)*(tt.cos(mi_c+mi_r))

    V = 0.5*(U1 - U2)

    # inv input covariance dot input output covariance
    C = tt.diag(exp_vii_h*tt.cos(mi))

    # account for the effect of scaling the output
    M = e*M
    V = tt.outer(e, e)*V
    C = e*C

    retvars = [M, V, C]

    return retvars


def gSat(m, v=None, i=None, e=None):
    ''' Reimplementation from the PILCO matlab code. Saturates the input
    signal to -1 to 1 through the function sat(x) = (9*sin(x) +sin(3*x))/8.
    If v is not None, this function returns the output mean, covariance and
    input-output covariance for computing he joint distribution p(input,output)
    as a multivariate Gaussian.'''
    D = m.shape[0]

    if i is None:
        i = tt.arange(D)
    if e is None:
        e = tt.ones((D,))
    elif e.__class__ is list:
        e = tt.as_tensor_variable(np.array(e)).flatten()
    elif e.__class__ is np.array:
        e = tt.as_tensor_variable(e).flatten()
    e = e.astype(m.dtype)

    # if no input variance, return deterministic
    if v is None:
        return e*(9*tt.sin(m) + tt.sin(3*m))/8

    # construct joint distribution of x and 3*x
    Q = tt.vertical_stack(tt.eye(D), 3*tt.eye(D))
    ma = Q.dot(m)
    va = Q.dot(v).dot(Q.T)

    # compute the joint distribution of 9*sin(x)/8 and sin(3*x)/8
    i1 = tt.concatenate([i, i+D])
    e1 = tt.concatenate([9.0*e, e])/8.0
    M2, V2, C2 = gSin(ma, va, i1, e1)
    # get the distribution of (9*sin(x) + sin(3*x))/8
    P = tt.vertical_stack(tt.eye(D), tt.eye(D))
    # mean
    M = M2.dot(P)
    # variance
    V = P.T.dot(V2).dot(P)

    # inv input covariance dot input output covariance
    C = Q.T.dot(C2).dot(P)

    retvars = [M, V, C]

    return retvars


def tanhSat(u, e):
    return e*tt.tanh(u)


def sigmoidSat(u, e):
    return e*(2*tt.nnet.sigmoid(u)-1)


def maxSat(u, e):
    return tt.minimum(theano.tensor.maximum(u, -e), e)

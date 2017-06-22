# pylint: disable=C0103
import theano
import numpy as np
import theano.tensor as tt
from theano.tensor.nlinalg import matrix_inverse, trace
from theano.tensor.slinalg import solve
from theano.tensor.nlinalg import det
from theano.sandbox.linalg import psd
from kusanagi.utils import print_with_stamp,gTrig2, gTrig_np, gTrig

def linear_loss(mx, Sx, target, Q, absolute=True, *args, **kwargs):
    '''
        Linear penalty function c(x) = Q.dot(|x-target|)
    '''
    if Sx is None:
        # deterministic case
        if mx.ndim == 1:
            mx = mx[None, :]
        delta = mx-target
        if absolute:
            delta = tt.abs(delta)
        cost = (delta).dot(Q)
        return cost
    else:
        # stochastic case (moment matching)
        delta = mx-target
        if absolute:
            delta = tt.abs(delta)
        SxQ = Sx.dot(Q)
        m_cost = Q.T.dot(delta)
        s_cost = Q.T.dot(SxQ)
        return m_cost, s_cost

def quadratic_loss(mx, Sx, target, Q, *args, **kwargs):
    '''
        Quadratic penalty function c(x) = (||x-target||_Q)^2
    '''
    if Sx is None:
        # deterministic case
        if mx.ndim == 1:
            mx = mx[None, :]
        delta = mx-target
        deltaQ = delta.T.dot(Q)
        cost = deltaQ.dot(delta)
        return cost
    else:
        # stochastic case (moment matching)
        delta = mx-target
        deltaQ = delta.T.dot(Q)
        SxQ = Sx.dot(Q)
        m_cost = tt.sum(Sx*Q) + deltaQ.dot(delta)
        s_cost = 2*tt.sum(SxQ.dot(SxQ)) + 4*deltaQ.dot(Sx).dot(deltaQ.T)
        return m_cost, s_cost

def quadratic_saturating_loss(mx, Sx, target, Q, *args, **kwargs):
    '''
        Squashing loss penalty function c(x) = ( 1 - e^(-0.5*quadratic_loss(x, target)) )
    '''
    if Sx is None:
        if mx.ndim == 1:
            mx = mx[None, :]
        delta = mx-target[None, :]
        deltaQ = delta.dot(Q)
        cost = 1.0 - tt.exp(-0.5*tt.sum(deltaQ*delta, 1))
        return cost
    else:
        # stochastic case (moment matching)
        delta = mx-target
        SxQ = Sx.dot(Q)
        IpSxQ = tt.eye(mx.shape[0]) + SxQ
        #S1 = Q.dot(matrix_inverse(IpSxQ))   x = Q dot I^-1; x' = I^-1' dot Q'
        S1 = solve(IpSxQ.T, Q.T).T
        m_cost = -tt.exp(-0.5*delta.dot(S1).dot(delta))/tt.sqrt(det(IpSxQ))
        Ip2SxQ = tt.eye(mx.shape[0]) + 2*SxQ
        #S2= Q.dot(matrix_inverse(Ip2SxQ))
        S2 = solve(Ip2SxQ.T, Q.T).T
        s_cost = tt.exp(-delta.dot(S2).dot(delta))/tt.sqrt(det(Ip2SxQ)) - m_cost**2

        return 1.0 + m_cost, s_cost


def generic_loss(mx, Sx, target, Q,
                 cw=[1], expl=None,
                 loss_func=quadratic_saturating_loss,
                 *args, **kwargs):
    '''
        Loss function that depends on a quadratic function of state
    '''
    if not isinstance(cw, list):
        cw = [cw]
    if Sx is None:
        # deterministic case
        cost = []
        # total cost is the sum of costs with different widths
        for c in cw:
            cost_c = loss_func(mx, None, target, Q/c**2, *args, **kwargs)
            cost.append(cost_c)
        return sum(cost)/len(cw)
    else:
        M_cost = []
        S_cost = []
        # total cost is the sum of costs with different widths
        for c in cw:
            m_cost, s_cost = loss_func(mx, Sx, target, Q/c**2, *args, **kwargs)
            # add UCB  exploration term
            if expl is not None and expl != 0.0:
                m_cost += expl*tt.sqrt(s_cost)
            M_cost.append(m_cost)
            S_cost.append(s_cost)
        return sum(M_cost)/len(cw), sum(S_cost)/(len(cw)**2)

def build_loss_func(loss_func, uncertain_inputs=False, name='loss_func', *args, **kwargs):
    '''
        Utility function to compiling a theano graph corresponding to aloss function
    '''
    mx = tt.vector('mx')
    Sx = tt.matrix('Sx') if uncertain_inputs else None
    inputs = [mx, Sx] if uncertain_inputs else [mx]
    outputs = loss_func(mx, Sx, *args, **kwargs)
    return theano.function(inputs, outputs, name=name)

import theano
import numpy as np
import theano.tensor as tt
from theano.tensor.nlinalg import matrix_inverse, trace
from theano.tensor.slinalg import solve
from theano.tensor.nlinalg import det
from theano.sandbox.linalg import psd
from kusanagi.utils import print_with_stamp,gTrig2, gTrig_np, gTrig

def linear_loss(mx,Sx,params,absolute=True):
    # linear penalty function
    Q = tt.constant(params['Q'],dtype=mx.dtype)
    target = tt.constant(params['target'],dtype=mx.dtype)
    delta = mx-target
    SxQ = Sx.dot(Q)
    m_cost = Q.T.dot(delta) 
    s_cost = Q.T.dot(Sx).dot(Q)

    return m_cost, s_cost

def quadratic_loss(mx,Sx,params):
    # Quadratic penalty function
    Q = tt.constant(params['Q'],dtype=mx.dtype) if 'Q' in params else tt.eye(Sx.shape[0])
    target = tt.constant(params['target'],dtype=mx.dtype)

    delta = mx-target
    deltaQ = delta.T.dot(Q)
    SxQ = Sx.dot(Q)  
    m_cost = tt.sum(Sx*Q) + deltaQ.dot(delta)
    s_cost = 2*tt.sum(SxQ.dot(SxQ)) + 4*deltaQ.dot(Sx).dot(deltaQ.T)

    return m_cost, s_cost

def quadratic_saturating_loss(mx,Sx,params):
    # Quadratic penalty function
    Q = tt.constant(params['Q'],dtype=mx.dtype)
    target = tt.constant(params['target'],dtype=mx.dtype)
    delta = mx-target
    deltaQ = delta.T.dot(Q)
    SxQ = Sx.dot(Q)
    IpSxQ = tt.eye(mx.shape[0]) + SxQ
    #S1 = Q.dot(matrix_inverse(IpSxQ))   x = Q dot I^-1; x' = I^-1' dot Q'
    S1 = solve(IpSxQ.T,Q.T).T
    m_cost = -tt.exp (-0.5*delta.dot(S1).dot(delta))/tt.sqrt(det(IpSxQ))
    Ip2SxQ = tt.eye(mx.shape[0]) + 2*SxQ
    #S2= Q.dot(matrix_inverse(Ip2SxQ))
    S2 = solve(Ip2SxQ.T,Q.T).T
    s_cost = tt.exp (-delta.dot(S2).dot(delta))/tt.sqrt(det(Ip2SxQ)) - m_cost**2

    return 1 + m_cost, s_cost

def generic_loss(mx,Sx,params,loss_func,angle_idims=[]):
    if len(angle_idims) > 0:
        mxa,Sxa = gTrig2(mx,Sx,angle_idims,mx.size)
    else:
        mxa = mx; Sxa = Sx
    return loss_func(mxa,Sxa,params)

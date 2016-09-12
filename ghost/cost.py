import theano
import numpy as np
import theano.tensor as T
from theano.tensor.nlinalg import matrix_inverse, trace
from theano.tensor.nlinalg import det
from theano.sandbox.linalg import psd
from utils import print_with_stamp,gTrig2, gTrig_np, gTrig

def linear_loss(mx,Sx,params,absolute=True):
    # linear penalty function
    Q = T.constant(params['Q'],dtype=mx.dtype)
    target = T.constant(params['target'],dtype=mx.dtype)
    delta = mx-target
    SxQ = Sx.dot(Q)
    m_cost = Q.T.dot(delta) 
    s_cost = Q.T.dot(Sx).dot(Q)

    return m_cost, s_cost

def quadratic_loss(mx,Sx,params):
    # Quadratic penalty function
    Q = T.constant(params['Q'],dtype=mx.dtype) if 'Q' in params else T.eye(Sx.shape[0])
    target = T.constant(params['target'],dtype=mx.dtype)

    delta = mx-target
    deltaQ = delta.T.dot(Q)
    SxQ = Sx.dot(Q)  
    m_cost = T.sum(Sx*Q) + deltaQ.dot(delta)
    s_cost = 2*T.sum(SxQ.dot(SxQ)) + 4*deltaQ.dot(Sx).dot(deltaQ.T)

    return m_cost, s_cost

def quadratic_saturating_loss(mx,Sx,params):
    # Quadratic penalty function
    Q = T.constant(params['Q'],dtype=mx.dtype)
    target = T.constant(params['target'],dtype=mx.dtype)
    delta = mx-target
    deltaQ = delta.T.dot(Q)
    SxQ = Sx.dot(Q)
    IpSxQ = T.eye(mx.shape[0]) + SxQ
    S1 = Q.dot(matrix_inverse(IpSxQ))
    m_cost = -T.exp (-0.5*delta.dot(S1).dot(delta))/T.sqrt(det(IpSxQ))
    Ip2SxQ = T.eye(mx.shape[0]) + 2*SxQ
    S2= Q.dot(matrix_inverse(Ip2SxQ))
    s_cost = T.exp (-delta.dot(S2).dot(delta))/T.sqrt(det(Ip2SxQ)) - m_cost**2

    return 1 + m_cost, s_cost

def generic_loss(mx,Sx,params,loss_func,angle_idims=[]):
    if len(angle_idims) > 0:
        mxa,Sxa = gTrig2(mx,Sx,angle_idims,mx.size)
    else:
        mxa = mx; Sxa = Sx
    return loss_func(mxa,Sxa,params)

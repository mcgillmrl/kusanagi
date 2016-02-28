import theano.tensor as T
import numpy as np
import theano

def gSin(m,v,i=None,e=None,derivs=False):
    D = m.shape[0]
    if i is None:
        i = T.arange(D)
    if e is None:
        e = T.ones((D,))
    elif e.__class__ is list:
        e = theano.tensor.as_tensor_variable(np.array(e)).flatten()
    elif e.__class__ is np.array:
        e = theano.tensor.as_tensor_variable(e).flatten()

    Di = i.shape[0]

    # compute the output mean
    mi = m[i]
    vi = (v[i,:][:,i])
    vii = (v[i,i])
    exp_vii_h = T.exp(-vii/2)
    M = exp_vii_h*T.sin(mi)

    # output covariance
    vii_c = vii[:,None]
    vii_r = vii[None,:]
    lq = -0.5*(vii_c+vii_r); q = T.exp(lq)
    exp_lq_p_vi = T.exp(lq+vi)
    exp_lq_m_vi = T.exp(lq-vi)
    mi_c = mi[:,None]
    mi_r = mi[None,:]
    U1 = (exp_lq_p_vi - q)*(T.cos(mi_c-mi_r))
    U2 = (exp_lq_m_vi - q)*(T.cos(mi_c+mi_r))

    V = 0.5*(U1 - U2)

    # inv input covariance dot input output covariance
    C = T.diag(exp_vii_h*T.cos(mi))
    
    # account for the effect of scaling the output
    M = e*M; V = e[:,None].dot(e[None,:])*V; C = e*C

    retvars = [M,V,C]

    # compute derivatives
    if derivs:
        dretvars = []
        for r in retvars:
            dretvars.append( T.jacobian(r.flatten(),m) )
        for r in retvars:
            dretvars.append( T.jacobian(r.flatten(),v) )
        retvars.extend(dretvars)

    return retvars

def gSat(m,v,i=None,e=None,derivs=False):
    D = m.shape[0]

    if i is None:
        i = T.arange(D)
    if e is None:
        e = T.ones((D,))
    elif e.__class__ is list:
        e = theano.tensor.as_tensor_variable(np.array(e)).flatten()
    elif e.__class__ is np.array:
        e = theano.tensor.as_tensor_variable(e).flatten()
    e = T.cast(e,m.dtype)
    # construct joint distribution of x and 3*x
    Q = T.vertical_stack(T.eye(D), 3*T.eye(D))
    ma = Q.dot(m)
    va = Q.dot(v).dot(Q.T)
    
    # compute the joint distribution of 9*sin(x)/8 and sin(3*x)/8
    i1 = T.concatenate([i, i+D]);
    e1 = T.concatenate([9.0*e, e])/8.0;
    M2, V2, C2 = gSin(ma, va, i1, e1, derivs=False);
    # get the distribution of (9*sin(x) + sin(3*x))/8
    P = T.vertical_stack(T.eye(D), T.eye(D))
    # mean
    M = M2.dot(P)
    # variance
    V = P.T.dot(V2).dot(P)

    # inv input covariance dot input output covariance
    C = Q.T.dot(C2).dot(P)
    
    retvars = [M,V,C]

    # compute derivatives
    if derivs:
        dretvars = []
        for r in retvars:
            dretvars.append( T.jacobian(r.flatten(),m) )
        for r in retvars:
            dretvars.append( T.jacobian(r.flatten(),v) )
        retvars.extend(dretvars)

    return retvars

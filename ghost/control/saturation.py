import theano.tensor as T
import numpy as np
import theano

def gSin(m,v,i=None,e=None,derivs=False):
    n = m.shape[0]
    D = m.shape[1]
    if i is None:
        i = T.arange(D)
    if e is None:
        e = T.ones((1,D))
    elif e.__class__ is list:
        e = theano.shared(np.array(e), borrow=True).flatten()[None,:]
    elif e.__class__ is np.array:
        e = theano.shared(e, borrow=True).fltten()[None,:]

    Di = i.shape[0]

    # compute the output mean
    mi = m[:,i]
    vi = (v[:,i,:][:,:,i])
    vii = (v[:,i,i])
    exp_vii_h = T.exp(-vii/2)
    M = exp_vii_h*T.sin(mi)

    # output covariance
    lq = -0.5*(vii[:,:,None]+vii[:,None,:]); q = T.exp(lq)
    exp_lq_p_vi = T.exp(lq+vi)
    exp_lq_m_vi = T.exp(lq-vi)
    U1 = (exp_lq_p_vi - q)*(T.cos(mi[:,:,None]-mi[:,None,:]))
    U2 = (exp_lq_m_vi - q)*(T.cos(mi[:,:,None]+mi[:,None,:]))

    V = 0.5*(U1 - U2)

    # inv input covariance dot input output covariance
    C = exp_vii_h*T.cos(mi)
    C,updts = theano.scan(fn=lambda Ci: T.diag(Ci), sequences=[C])
    
    # account for the effect of scaling the output
    M = e*M; V = e.T.dot(e)*V; C = e*C

    retvars = [M,V,C]

    if derivs:
        dMdm = T.jacobian(M,m)
        dMdv = T.jacobian(M,v).reshape((n,D,D**2))
        dVdm = T.jacobian(V.T.flatten(),m)
        dVdv = T.jacobian(V.T.flatten(),v).reshape((n,D**2, D**2))
        dCdm = T.jacobian(C.T.flatten(),m)
        dCdv = T.jacobian(C.T.flatten(),v).reshape((n,D**2, D**2))

        retvars.append(dMdm,dVdm,dMdv,dVdv,dCdm,dCdv)

    return retvars

def gSat(m,v,i=None,e=None,derivs=False):
    n = m.shape[0]
    D = m.shape[1]

    if i is None:
        i = T.arange(D)
    if e is None:
        e = T.ones((1,D))
    elif e.__class__ is list:
        e = theano.shared(np.array(e), borrow=True).flatten()[None,:]
    elif e.__class__ is np.array:
        e = theano.shared(e, borrow=True).fltten()[None,:]
    
    # construct joint distribution of x and 3*x
    Q = T.vertical_stack(T.eye(D), 3*T.eye(D))
    ma = Q.dot(m.T).T
    va = ((Q[:,:,None]*v[:,None,:,:]).sum(2)).dot(Q.T)
    va = (va + va.transpose(0,2,1))/2
    
    # compute the joint distribution of 9*sin(x)/8 and sin(3*x)/8
    i1 = T.horizontal_stack(i[None,:], i[None,:]+D).flatten();
    e1 = T.horizontal_stack(9.0*e, e)/8.0;
    M2, V2, C2 = gSin(ma, va, i1, e1, derivs=False);
    # get the distribution of (9*sin(x) + sin(3*x))/8
    P = T.vertical_stack(T.eye(D), T.eye(D))
    # mean
    M = M2.dot(P)
    # variance
    V = ((P.T[:,:,None]*V2[:,None,:,:]).sum(2)).dot(P)
    V = (V + V.transpose(0,2,1))/2
    # inv input covariance dot input output covariance
    C = ((Q.T[:,:,None]*C2[:,None,:,:]).sum(2)).dot(P)
    
    retvars = [M,V,C]

    if derivs:
        dMdm = T.jacobian(M,m)
        dMdv = T.jacobian(M,v).reshape((n,D,D**2))
        dVdm = T.jacobian(V.T.flatten(),m)
        dVdv = T.jacobian(V.T.flatten(),v).reshape((n,D**2, D**2))
        dCdm = T.jacobian(C.T.flatten(),m)
        dCdv = T.jacobian(C.T.flatten(),v).reshape((n,D**2, D**2))

        retvars.append(dMdm,dVdm,dMdv,dVdv,dCdm,dCdv)

    return retvars

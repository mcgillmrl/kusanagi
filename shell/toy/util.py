import numpy as np
from itertools import chain
import theano.tensor as T
import theano

def mahalanobis(a, b, Q=None):
    if a.ndim == 1:
        a = a.dimshuffle(('x',0))
    if b.ndim == 1:
        b = b.dimshuffle(('x',0))
    K = []
    if Q is None:
        K = T.sum(a**2,1)[:,None] + T.sum(b**2,1) - 2*a.dot(b.T);
    else:
        aQ = a.dot(Q);
        K = T.sum(aQ*a,1)[:,None] + T.sum(b.dot(Q)*b,1) - 2*aQ.dot(b.T)
    return K

def squared_row_norms(X,Y):
    return np.einsum('ij,ij->i', X, Y)

def squared_euclidean_distance(X, Y):
    X2 = np.einsum('ij,ij->i', X, Y)
    Y2 = []
    if X is Y:
        Y2 = X2[:,None]
    else:
        Y2 = np.einsum('ij,ij->i', X, Y)[:,None]
    distances = np.dot(X,Y.T)
    distances *= -2
    distances += X2
    distances += Y2
    return distances

def squared_mahalannobis_distance(X,Y,M, col=False):
    delta = X - Y
    if col:
        delta = delta.T
    d = np.einsum('nj,jk,nk->n', delta, M, delta)
    return d

# trigonometric state augmentation ( angle dimensions are replaced by a unit vector representation)
def augment(input_vector, angi):
    input_vector  = np.array(input_vector)
    if len(input_vector.shape) < 2:
        input_vector = input_vector[:,None]
    D,n = input_vector.shape
    aug_x = np.zeros((2*len(angi), n))
    aug_x[::2,:] =  np.sin(input_vector[angi,:])
    aug_x[1::2,:] =  np.cos(input_vector[angi,:])

    new_x = np.empty((D + 2*len(angi), n))
    
    new_x[:D,:] = input_vector
    new_x[D:,:] = aug_x

    return new_x

def gTrig(input_vector,variance, angi, derivs=False):
    D = input_vector.shape[0]
    Da = 2*len(angi)
    Is = 2*np.arange(len(angi)); Ic = Is +1;
    M = np.zeros((Da, 1))
    V = np.zeros((Da,Da))
    C = np.zeros((D,Da))
    dMdm=[];dMdv=[];dVdm=[];dVdv=[];dCdm=[];dCdv=[];

    # computed the mean
    mi = input_vector[angi]
    M[::2] =  np.sin(mi)
    M[1::2] =  np.cos(mi)

    # compute the entries in the augmented covariance matrix
    vi = (variance[angi,:][:,angi])
    vii = (variance[angi,angi])[:,None]
    exp_vii_h = np.exp(-vii/2)
    M[::2] = exp_vii_h*M[::2]
    M[1::2] = exp_vii_h*M[1::2]
    
    lq = -0.5*(vii+vii.T); q = np.exp(lq)
    exp_lq_p_vi = np.exp(lq+vi)
    exp_lq_m_vi = np.exp(lq-vi)
    U1 = (exp_lq_p_vi - q)*(np.sin(mi-mi.T))
    U2 = (exp_lq_m_vi - q)*(np.sin(mi+mi.T))
    U3 = (exp_lq_p_vi - q)*(np.cos(mi-mi.T))
    U4 = (exp_lq_m_vi - q)*(np.cos(mi+mi.T))
    
    V[::2,::2] = U3 - U4; V[1::2,1::2] = U3 + U4; V[::2,1::2] = U1 + U2; 
    V[1::2,::2] = V[::2,1::2].T; V = 0.5*V;

    # inv times input output covariance
    C[angi,::2] = np.diag(M[1::2].ravel())
    C[angi,1::2] = np.diag(-M[::2].ravel())

    # compute the derivatives
    if derivs:
        dMdm = C.T
        dVdm = np.zeros((Da,Da,D))
        dCdm = np.zeros((D,Da,D))
        dVdv = np.zeros((Da,Da,D,D))
        dCdv = np.zeros((D,Da,D,D))
        exp_vii = exp_vii_h**2

        for j in xrange(len(angi)):
            u = np.zeros((len(angi),1)); u[j] = 0.5;
            minus_u = (u-u.T); plus_u = (u+u.T)
            dVdm[::2,::2,angi[j]] = -U1*minus_u + U2*plus_u
            dVdm[1::2,1::2,angi[j]] = -U1*minus_u - U2*plus_u
            dVdm[::2,1::2,angi[j]] = U3*minus_u + U4*plus_u
            dVdm[1::2,::2,angi[j]] = dVdm[::2,1::2,angi[j]]
            
            dVdv[Is[j],Is[j],angi[j],angi[j]] = exp_vii[j]*(1+(2*exp_vii[j]-1)*np.cos(2*mi[j]))*0.5
            dVdv[Ic[j],Ic[j],angi[j],angi[j]] = exp_vii[j]*(1-(2*exp_vii[j]-1)*np.cos(2*mi[j]))*0.5
            dVdv[Is[j],Ic[j],angi[j],angi[j]] = exp_vii[j]*(1-2*exp_vii[j])*np.sin(2*mi[j])*0.5
            dVdv[Ic[j],Is[j],angi[j],angi[j]] = dVdv[Is[j],Ic[j],angi[j],angi[j]]

            for k in chain(xrange(j),xrange(j+1,len(angi))):
                dVdv[Is[j],Is[k],angi[j],angi[k]] =  (exp_lq_p_vi[j,k]*np.cos(mi[j]-mi[k]) + exp_lq_m_vi[j,k]*np.cos(mi[j]+mi[k]))*0.5
                dVdv[Is[j],Is[k],angi[j],angi[j]] = -V[Is[j],Is[k]]*0.5
                dVdv[Is[j],Is[k],angi[k],angi[k]] = -V[Is[j],Is[k]]*0.5
                dVdv[Ic[j],Ic[k],angi[j],angi[k]] =  (exp_lq_p_vi[j,k]*np.cos(mi[j]-mi[k]) - exp_lq_m_vi[j,k]*np.cos(mi[j]+mi[k]))*0.5
                dVdv[Ic[j],Ic[k],angi[j],angi[j]] = -V[Ic[j],Ic[k]]*0.5
                dVdv[Ic[j],Ic[k],angi[k],angi[k]] = -V[Ic[j],Ic[k]]*0.5
                dVdv[Ic[j],Is[k],angi[j],angi[k]] = -(exp_lq_p_vi[j,k]*np.sin(mi[j]-mi[k]) + exp_lq_m_vi[j,k]*np.sin(mi[j]+mi[k]))*0.5
                dVdv[Ic[j],Is[k],angi[j],angi[j]] = -V[Ic[j],Is[k]]*0.5
                dVdv[Ic[j],Is[k],angi[k],angi[k]] = -V[Ic[j],Is[k]]*0.5
                dVdv[Is[j],Ic[k],angi[j],angi[k]] =  (exp_lq_p_vi[j,k]*np.sin(mi[j]-mi[k]) - exp_lq_m_vi[j,k]*np.sin(mi[j]+mi[k]))*0.5
                dVdv[Is[j],Ic[k],angi[j],angi[j]] = -V[Is[j],Ic[k]]*0.5
                dVdv[Is[j],Ic[k],angi[k],angi[k]] = -V[Is[j],Ic[k]]*0.5

            dCdm[angi[j],Is[j],angi[j]] = -M[Is[j]] 
            dCdm[angi[j],Ic[j],angi[j]] = -M[Ic[j]] 
            dCdv[angi[j],Is[j],angi[j],angi[j]] = -C[angi[j],Is[j]]*0.5
            dCdv[angi[j],Ic[j],angi[j],angi[j]] = -C[angi[j],Ic[j]]*0.5

        dMdv = dCdm.transpose((1,0,2))*0.5

        dMdv = dMdv.reshape((Da, D**2),order='F')
        dVdv = dVdv.reshape((Da**2, D**2),order='F')
        dVdm = dVdm.reshape((Da**2, D),order='F')
        dCdv = dCdv.reshape((Da*D, D**2),order='F')
        dCdm = dCdm.reshape((Da*D, D),order='F')
    return M, V, C, dMdm, dVdm, dCdm, dMdv, dVdv, dCdv

def gTrig2(m,v,angi,D, derivs=False):
    Da = 2*len(angi)
    Is = 2*np.arange(len(angi)); Ic = Is +1;
    M = theano.shared(np.zeros(Da))
    V = theano.shared(np.zeros((Da,Da)))
    C = theano.shared(np.zeros((D,Da)))

    # compute the mean
    mi = m[angi]
    vi = (v[angi,:][:,angi])
    vii = (v[angi,angi])
    exp_vii_h = T.exp(-vii/2)

    M = T.set_subtensor(M[::2], exp_vii_h*T.sin(mi))
    M = T.set_subtensor(M[1::2], exp_vii_h*T.cos(mi))
    
    # compute the entries in the augmented covariance matrix
    lq = -0.5*(vii[:,None]+vii); q = T.exp(lq)
    exp_lq_p_vi = T.exp(lq+vi)
    exp_lq_m_vi = T.exp(lq-vi)
    U1 = (exp_lq_p_vi - q)*(T.sin(mi[:,None]-mi))
    U2 = (exp_lq_m_vi - q)*(T.sin(mi[:,None]+mi))
    U3 = (exp_lq_p_vi - q)*(T.cos(mi[:,None]-mi))
    U4 = (exp_lq_m_vi - q)*(T.cos(mi[:,None]+mi))
    
    V = T.set_subtensor(V[::2,::2], U3-U4)
    V = T.set_subtensor(V[1::2,1::2], U3+U4)
    V = T.set_subtensor(V[::2,1::2],U1+U2)
    V = T.set_subtensor(V[1::2,::2],V[::2,1::2].T)
    V = 0.5*V

    # inv times input output covariance
    C = T.set_subtensor( C[angi,Is], M[1::2]) 
    C = T.set_subtensor( C[angi,Ic], -M[::2]) 

    retvars = [M,V,C]

    # compute derivatives
    if derivs:
        dMdm = T.jacobian(M,m)
        dMdv =  T.jacobian(M,v).reshape((Da, D**2))
        dVdm = T.jacobian(V.T.flatten(),m)
        dVdv =  T.jacobian(V.T.flatten(),v).reshape((Da**2, D**2))
        dCdm = T.jacobian(C.T.flatten(),m)
        dCdv =  T.jacobian(C.T.flatten(),v).reshape((Da*D, D**2))
        retvars.append(dMdm,dVdm,dCdm,dMdv,dVdv,dCdv)

    return retvars

# Fill in covariance matrix...and derivatives 
def fillIn(S,C,mdm,sdm,Cdm,mds,sds,Cds,Mdm,Sdm,Mds,Sds,i,k,D):
    X = np.arange(D*D).reshape((D,D))                    # vectorized indices
    ii=X[i[:,None],i].ravel(); kk=X[k[:,None],k].ravel(); ik=X.T[i[:,None],k].ravel(1); ki=X.T[k[:,None],i].ravel()

    Mdm[k,:]  = mdm.dot(Mdm[i,:]) + mds.dot(Sdm[ii,:])             # chainrule
    Mds[k,:]  = mdm.dot(Mds[i,:]) + mds.dot(Sds[ii,:])
    Sdm[kk,:] = sdm.dot(Mdm[i,:]) + sds.dot(Sdm[ii,:])
    Sds[kk,:] = sdm.dot(Mds[i,:]) + sds.dot(Sds[ii,:])
    dCdm      = Cdm.dot(Mdm[i,:]) + Cds.dot(Sdm[ii,:])
    dCds      = Cdm.dot(Mds[i,:]) + Cds.dot(Sds[ii,:])

    S[i[:,None],k] = S[i[:,None],i].dot(C); S[k[:,None],i] = S[i[:,None],k].T;                             # off-diagonal
    SS = np.kron(np.eye(len(k)),S[i[:,None],i]); CC = np.kron(C.T,np.eye(len(i)))
    Sdm[ik,:] = SS.dot(dCdm) + CC.dot(Sdm[ii,:]); Sdm[ki,:] = Sdm[ik,:];
    Sds[ik,:] = SS.dot(dCds) + CC.dot(Sds[ii,:]); Sds[ki,:] = Sds[ik,:];

    return S,Mdm,Mds,Sdm,Sds

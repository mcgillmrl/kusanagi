import os,sys
from datetime import datetime

import numpy as np
import theano
import theano.tensor as T

def maha(X1,X2=None,M=None, all_pairs=True):
    ''' Returns the squared Mahalanobis distance'''
    D = []
    deltaM = []
    if X2 is None:
        X2 = X1

    if all_pairs:
        if M is None:
            D, updts = theano.scan(fn=lambda xi,X: T.sum((xi-X)**2,1), sequences=[X1], non_sequences=[X2])
        else:
            D, updts = theano.scan(fn=lambda xi,X,V: T.sum(((xi-X).dot(V))*(xi-X),1), sequences=[X1], non_sequences=[X2,M])
    else:
        # computes the distance  x1i - x2i for each row i
        if X1 is X2:
            # in this case, we don't need to compute anything
            D = T.zeros((X1.shape[0],))
            return D
        delta = X1-X2
        if M is None:
            deltaM = delta
        else:
            deltaM = delta.dot(M)
        D = T.sum(deltaM*delta,1)
        
    return D

def print_with_stamp(message, name=None, same_line=False):
    out_str = ''
    if name is None:
        out_str = '[%s] %s'%(str(datetime.now()),message)
    else:
        out_str = '[%s] %s > %s'%(str(datetime.now()),name,message)
    
    if same_line:
        sys.stdout.write('\r'+out_str)
    else:
        sys.stdout.write(out_str)
        print ''
    sys.stdout.flush()

def kmeans(X,W,learning_rate=0.001):
    #find the closest vector from W to each vector in X
    Xm = X - X.mean(0)
    Wm = W - X.mean(0)
    sq_D = maha(Xm,Wm)
    # get the closest element from Wm for each Xm
    Dmin = T.sqrt(sq_D.min(1))

    n = T.cast(X.shape[0], theano.config.floatX)
    err = T.sum(Dmin)
    # compute updates to the vectors in W
    direction = T.stacklists(T.grad(err,[W])).reshape(W.shape)
    updts = [(W, W - learning_rate*direction)]
    # compute the error (i.e. the total distance from data to centers)
    return err, updts

def get_kmeans_func(W_):
    ''' Compiles the kmeans function for the given data vector'''
    if theano.config.floatX == 'float32':
        X = theano.tensor.fmatrix('X')
        learning_rate = theano.tensor.fscalar('alpha')
    else: 
        X = theano.tensor.dmatrix('X')
        learning_rate = theano.tensor.dscalar('alpha')

    if isinstance(W_,T.sharedvar.TensorSharedVariable):
        W = W_
    else:
        W = theano.shared(W_, name='W', borrow=True)
        
    error, updts = kmeans(X,W,learning_rate)

    km = theano.function(inputs= [X,learning_rate], outputs=error,updates=updts, allow_input_downcast=True)
    return km,W

def get_kmeans_func2(X_,W_):
    if isinstance(W_,T.sharedvar.TensorSharedVariable):
        W = W_
    else:
        W = theano.shared(W_, name='W', borrow=True)
    if isinstance(X_,T.sharedvar.TensorSharedVariable):
        X = X_
    else:
        X = theano.shared(X_, name='X', borrow=True)

    #find the closest vector from W to each vector in X
    Xm = X - X.mean(0)
    Wm = W - X.mean(0)
    sq_D = maha(Xm,Wm)
    # get the closest element from Wm for each Xm
    Dmin = T.sqrt(sq_D.min(1))

    n = T.cast(X.shape[0], theano.config.floatX)
    err = T.sum(Dmin)

    km = theano.function([],[err,T.stacklists(T.grad(err,[W])).flatten()], allow_input_downcast=True)
    return km,W,X

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


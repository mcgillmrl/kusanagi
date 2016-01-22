import os,sys
from datetime import datetime

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.linalg import psd,matrix_inverse

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

def gTrig(x,angi,D):
    Da = 2*len(angi)
    n = x.shape[0]
    xang = T.zeros((n,Da))
    xi = x[:,angi]
    xang = T.set_subtensor(xang[:,::2], T.sin(xi))
    xang = T.set_subtensor(xang[:,1::2], T.cos(xi))

    non_angle_dims = list(set(range(D)).difference(angi))
    xnang = x[:,non_angle_dims]
    m = T.concatenate([xnang,xang],axis=1)
    return m

def gTrig2(m, v, angi, D, derivs=False):
    non_angle_dims = list(set(range(D)).difference(angi))
    Da = 2*len(angi)
    Dna = len(non_angle_dims) 
    n = m.shape[0]
    Ma = T.zeros((n,Da))
    Va = T.zeros((n,Da,Da))
    Ca = T.zeros((n,D,Da))

    # compute the mean
    mi = m[:,angi]
    vi = (v[:,angi,:][:,:,angi])
    vii = (v[:,angi,angi])
    exp_vii_h = T.exp(-vii/2)

    Ma = T.set_subtensor(Ma[:,::2], exp_vii_h*T.sin(mi))
    Ma = T.set_subtensor(Ma[:,1::2], exp_vii_h*T.cos(mi))
    
    # compute the entries in the augmented covariance matrix
    lq = -0.5*(vii[:,:,None]+vii[:,None,:]); q = T.exp(lq)
    exp_lq_p_vi = T.exp(lq+vi)
    exp_lq_m_vi = T.exp(lq-vi)
    U1 = (exp_lq_p_vi - q)*(T.sin(mi[:,:,None]-mi[:,None,:]))
    U2 = (exp_lq_m_vi - q)*(T.sin(mi[:,:,None]+mi[:,None,:]))
    U3 = (exp_lq_p_vi - q)*(T.cos(mi[:,:,None]-mi[:,None,:]))
    U4 = (exp_lq_m_vi - q)*(T.cos(mi[:,:,None]+mi[:,None,:]))
    
    Va = T.set_subtensor(Va[:,::2,::2], U3-U4)
    Va = T.set_subtensor(Va[:,1::2,1::2], U3+U4)
    Va = T.set_subtensor(Va[:,::2,1::2],U1+U2)
    Va = T.set_subtensor(Va[:,1::2,::2],Va[:,::2,1::2].transpose(0,2,1))
    Va = 0.5*Va

    # inv times input output covariance
    Is = 2*np.arange(len(angi)); Ic = Is +1;
    Ca = T.set_subtensor( Ca[:,angi,Is], Ma[:,1::2]) 
    Ca = T.set_subtensor( Ca[:,angi,Ic], -Ma[:,::2]) 

    # construct mean vectors ( non angle dimensions come first, then angle dimensions)
    Mna = m[:, non_angle_dims]
    M = T.concatenate([Mna,Ma],axis=1)

    # construct the corresponding covariance matrices ( just the blocks for the non angle dimensions and the angle dimensions separately)
    V = T.zeros((n,Dna+Da,Dna+Da))
    Vna = v[:,non_angle_dims,:][:,:,non_angle_dims]
    V = T.set_subtensor(V[:,:Dna,:Dna], Vna)
    V = T.set_subtensor(V[:,Dna:,Dna:], Va)

    # fill in the cross covariances
    V = T.set_subtensor(V[:,:Dna,Dna:], (v[:,:,None,:]*Ca[:,:,:,None]).sum(3)[:,non_angle_dims,:] )
    V = T.set_subtensor(V[:,Dna:,:Dna], V[:,:Dna,Dna:].transpose(0,2,1))

    retvars = [M,V]

    # compute derivatives
    if derivs:
        dMdm = T.jacobian(M,m)
        dMdv =  T.jacobian(M,v).reshape((n,Da, D**2))
        dVdm = T.jacobian(V.T.flatten(),m)
        dVdv =  T.jacobian(V.T.flatten(),v).reshape((n,Da**2, D**2))
        retvars.append(dMdm,dVdm,dMdv,dVdv)

    return retvars

def gTrig_np(x,angi):
    D = x.shape[1]
    Da = 2*len(angi)
    n = x.shape[0]
    xang = np.zeros((n,Da))
    xi = x[:,angi]
    xang[:,::2] =  np.sin(xi)
    xang[:,1::2] =  np.cos(xi)

    non_angle_dims = list(set(range(D)).difference(angi))
    xnang = x[:,non_angle_dims]
    m = np.concatenate([xnang,xang],axis=1)

    return m

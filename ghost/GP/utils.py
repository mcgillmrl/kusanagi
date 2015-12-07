import os,sys
from datetime import datetime

import theano
import theano.tensor as T

def maha(X1,X2=None,M_sqrt=None, all_pairs=True):
    ''' Returns the squared Mahalanobis distance'''
    X1M = []
    X2M = []
    D = []
    if M_sqrt is None:
        if X2 is None:
            X1M = X1
            X2M = X1
        else:
            X1M = X1
            X2M = X2
    else:
        if X2 is None:
            X1M = X1.dot(M_sqrt)
            X2M = X1M
        else:
            X1M = X1.dot(M_sqrt)
            X2M = X2.dot(M_sqrt)
    if all_pairs:
        D, updts = theano.scan( lambda x1,x2: T.sum((x1-x2)**2,axis=1), sequences=[X1M],non_sequences=[X2M] )
    else:
        # computes the distance  x1i - x2i for each row i
        # TODO, ensure that x1 and x2 have the same number of elements
        D = T.sum((X1M-X2M)**2,axis=1)
        
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
    D = (Xm**2).sum(1)[:,None] + (Wm**2).sum(1)[None,:] - 2*Xm.dot(W.T)
    #D = maha(Xm,Wm)
    err = T.sum(D)/X.shape[0]
    # compute updates to the vectors in W
    direction = T.stacklists(T.grad(err,[W])).reshape(W.shape)
    updts = [(W, W - learning_rate*direction)]
    # compute the error (i.e. the total distance from data to centers)
    return err, updts

def get_kmeans_func(W_):
    ''' Compiles the kmeans function for the given data vector'''
    X = theano.tensor.matrix('X')
    learning_rate = theano.tensor.scalar('alpha')
    W = []

    if isinstance(W_,T.sharedvar.TensorSharedVariable):
        W = W_
    else:
        W = theano.shared(W_, name='W', borrow=True)
        
    error, updts = kmeans(X,W,learning_rate)

    km = theano.function(inputs= [X,learning_rate], outputs=error,updates=updts, allow_input_downcast=True)
    return (W, km)




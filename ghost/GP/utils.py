import os,sys
from datetime import datetime

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
    return (W, km)




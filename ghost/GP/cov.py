import theano
import theano.tensor as T
from utils import *

def SEard(loghyp,X1,X2=None):
    ''' Squared exponential kernel with diagonal scaling matrix (one lengthscale per dimension)'''
    n = 1; idims = 1
    if(X1.ndim == 2):
        n,idims = X1.shape
    elif(X2.ndim == 2):
        n,idims = X2.shape
    else:
        idims = X1.shape[0]
    D = maha(X1,X2,T.diag(T.exp(-loghyp[:idims])))
    K = T.exp(2*loghyp[idims] - 0.5*D)
    nf = T.cast(n, theano.config.floatX)
    eps = 5e-4 if theano.config.floatX == 'float32' else 1e-4
    return K + eps*T.log(nf)*T.eye(n) if (X1 is X2 or X2 is None) else K

def Noise(loghyp,X1,X2=None,D=None):
    ''' Noise kernel. Takes as an input a distance matrix D and creates a new matrix 
    as Kij = sn2 if Dij == 0 else 0'''
    if D is None:
        X2 = X1 if X2 is None else X2
        D = maha(X1,X2)
    K = T.isclose(D,0)*T.exp(2*loghyp)
    return K

def Sum(loghyp_l, cov_l, X1, X2=None):
    ''' Returns the sum of multiple covariance functions'''
    K = sum([cov_l[i](loghyp_l[i],X1,X2) for i in xrange(len(cov_l)) ] )
    return K

import theano
import theano.tensor as T
from utils import *

def SEard(loghyp,X1,X2=None, all_pairs=True):
    ''' Squared exponential kernel with diagonal scaling matrix (one lengthscale per dimension)'''
    n1 = 1; n2 = 1; idims = 1
    if(X1.ndim == 2):
        n1,idims = X1.shape
    elif(X2.ndim == 2):
        n2,idims = X2.shape
    else:
        idims = X1.shape[0]
    
    if  not all_pairs and X1 is X2 or X2 is None:
        # in this case all the distances are zero
        K = T.tile(T.exp(2*loghyp[idims]),(n1,1))

    D = maha(X1,X2,T.diag(T.exp(-2*loghyp[:idims])),all_pairs=all_pairs)
    K = T.exp(2*loghyp[idims] - 0.5*D)

    if all_pairs and X1 is X2:
        nf = T.cast(n1, theano.config.floatX)
        eps = 5e-4 if theano.config.floatX == 'float32' else 1e-4
        return K + eps*T.log(nf)*T.eye(n) if (X1 is X2 or X2 is None) else K
    else:
        return K

def Noise(loghyp,X1,X2=None,D=None, all_pairs=True):
    ''' Noise kernel. Takes as an input a distance matrix D and creates a new matrix 
    as Kij = sn2 if Dij == 0 else 0'''
    if D is None:
        X2 = X1 if X2 is None else X2
        D = maha(X1,X2,all_pairs=all_pairs)
    K = T.isclose(D,0)*T.exp(2*loghyp)
    return K

def Sum(loghyp_l, cov_l, X1, X2=None, all_pairs=True):
    ''' Returns the sum of multiple covariance functions'''
    K = sum([cov_l[i](loghyp_l[i],X1,X2,all_pairs=all_pairs) for i in xrange(len(cov_l)) ] )
    return K

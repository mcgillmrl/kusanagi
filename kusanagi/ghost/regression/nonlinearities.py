import theano.tensor as tt
from lasagne.nonlinearities import *


def silu(x):
    '''
    aka  SiL, Swish, etc.
    '''
    return x*tt.nnet.sigmoid(x)


def phi(x):
    return 0.5*(tt.erfc(-x/tt.sqrt(2)))


def gelu(x):
    '''
        similar to silu
    '''
    return x*phi(x)


def gelu2(x):
    '''
        similar to silu
    '''
    return x*(tt.erf(x) + 1)


def rbf(x):
    return tt.exp(-x**2)

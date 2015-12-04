from datetime import datetime
import theano
import theano.tensor as T

def maha(X1,X2=None,M_sqrt=None):
    ''' Returns the squared Mahalanobis distance'''
    X1M = []
    X2M = []
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
    D, updts = theano.scan( lambda x1,x2: T.sum((x1-x2)**2,axis=1), sequences=[X1M],non_sequences=[X2M] )
    return D

def print_with_stamp(message, name=None):
    if name is None:
        print '[%s] %s'%(str(datetime.now()),message)
    else:
        print '[%s] %s > %s'%(str(datetime.now()),name,message)

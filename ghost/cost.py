import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nlinalg import matrix_inverse
from theano.tensor.nlinalg import det
from util import augment,gTrig2
import utils

# saturating cost function c(x) =  1 - exp{ -(1/(2*cw**2))*( (x-z)^T . W . (x-z) ) }
# where  
# x ~ Normal ( m, s ) : state with vector with trigonometrically augmented angle dimensions
# z : trigonometrically augmented target state
# Q : corresponding cost matrix
# inputs:   * a Cost object with parameters W, z and angi
#           * symbolic Theano variables for the mean and covariance matrix of x ( m and s )
# returns: symbolic Theano expressions for E{c(x)|m,s} and var{c(x)|m,s} (c and v )
def lossSat(cost,m,s):
    # initialize variables for symbolic computations
    c = T.dscalar('c')
    v = T.dscalar('v')

    # initialize other variables
    D = cost.z.shape[0]
    Q = cost.W/(cost.cw**2)
    target = cost.z

    # define the saturating cost as in Deisenroth2009
    SQ = s.dot(Q)
    SpQ = np.eye(D)+SQ
    iSpQ = T.dot(Q,matrix_inverse(SpQ)) #solve(SpQ.T,Q.T).T  theano does not provide gradients for the result of the solve method ( should be straight forward to do)
    mQ = m.dot(iSpQ)
    delta = m-target
    maha_dist = delta.T.dot(iSpQ).dot(delta) # TODO  is this inefficient?
    c = 1-T.exp( -0.5*(maha_dist) ) / T.sqrt( det(SpQ) )

    # compute the variance (again using the derivation from Deisenroth2009)
    SpQ2 = np.eye(D)+2*SQ
    i2SpQ = T.dot(Q,matrix_inverse(SpQ2)) #solve(SpQ.T,Q.T).T  theano does not provide gradients for the result of the solve method ( should be straight forward to do)

    # var = E{c^2} - (E{c})^2
    v = T.exp( -delta.T.dot(i2SpQ).dot(delta) ) / T.sqrt( det(SpQ2) ) - c**2
    
    return c,v

# Generic cost function which augments angle dimensions before calling an implementation of a loss function
# adds an exploration term when available, evaluated at state x ~ Normal( m, s).
# inputs:   * a Cost object with parameters target, Q and angi
#           * symbolic Theano variables for the mean and covariance matrix of x ( m and s )
# returns: symbolic Theano expressions for E{L(x)|m,s} and var{L(x)|m,s} ( L and S2 )
def lossGeneric(cost,m,s,loss_function=lossSat):
    L = T.dscalar('L')
    S2 = T.dscalar('S2')

    # initialize other variables
    target = cost.target
    Q = cost.Q
    expl = cost.expl
    D0 = cost.target.shape[0]
    angi = cost.angi

    D1 = D0 + 2*len(angi)
    M = theano.shared(np.zeros(D1))
    S = theano.shared(np.zeros((D1,D1)))
    S = T.set_subtensor(S[0:D0,0:D0], s)

    if D1 - D0 > 0: # if there are some angle dimensions, augment the state
        i = np.arange(D0)
        k = np.arange(D0,D1)
        # this augments the state and computes the partial derivatives of the trigonometric functions
        gTrig_vars = gTrig2(m,s,angi,D0)

        # this fills in the mean and covariance
        M = T.set_subtensor(M[k], gTrig_vars[0])
        S = T.set_subtensor(S[k[:,None],k], gTrig_vars[1])
        S = T.set_subtensor(S[i[:,None],k], s.dot(gTrig_vars[2]) )
        S = T.set_subtensor(S[k[:,None],i], S[i[:,None],k].T)

        cost.z = np.squeeze(augment(target,angi))
        cost.W = Q
        assert Q.shape[0] == len(target) + 2*len(angi)  and Q.shape[1] == Q.shape[0]
    else:
        assert Q.shape[0] == len(target) and Q.shape[1] == Q.shape[0]
        cost.z = target
        cost.W = Q
        
    # calculate cost
    L,S2 = loss_function(cost,M,S)
    # add exploration term
    L = L + expl*T.sqrt(S2)

    return L,S2

# This class defines a cost function
class Cost(object):
    def __init__(self, loss_function, target, Q, cw=1, expl = 0, angi= [], gamma = 1.0):
        # init member attributes
        self.name = 'Cost'
        self.Q = Q
        self.target = np.array(target)
        self.angi = np.array(angi)
        self.cw = cw
        self.expl = expl
        self.cw = cw
        self.gamma = gamma
        self.L = None
        self.S2 = None
        self.Lfcn = None
        self.Lgrad = None
        self.loss_function = loss_function

        # init symbolic variables for the mean and covariance of the state
        self.m = T.dvector('m')
        self.s = T.dmatrix('s')
        self.init()

    def init(self):
        # get the symbolic expressions for the mean and the variance of the cost
        if self.L is None or self.S2 is None:
            L,S2 = self.loss_function(self,self.m,self.s)
            self.L = L
            self.S2 = S2

        # compile the loss function if needed
        if self.Lfcn is None:
            utils.print_with_stamp('Compiling cost function',self.name)
            self.Lfcn = theano.compile.function([self.m,self.s],(self.L,self.S2))
            utils.print_with_stamp('Done compiling',self.name)

        # compile the derivatives if needed
        if self.Lgrad is None:
            utils.print_with_stamp('Compiling gradients of cost function',self.name)
            dLdm = T.grad(self.L,self.m)
            dLds = T.grad(self.L,self.s).T.ravel()
            dS2dm = T.grad(self.S2,self.m)
            dS2ds = T.grad(self.S2,self.s).T.ravel()
            self.Lgrad = theano.compile.function([self.m,self.s],(self.L, dLdm,dLds, self.S2))
            utils.print_with_stamp('Done compiling',self.name)

    def evaluate(self,mx,mu=None,Sx=None,Su=None,derivs=False):
        # set a default covariance matrix if needed
        if Sx is None:
            D = len(mx)
            Sx = np.zeros((D,D))

        if not derivs:
            # this will return values for  L and S2 evaluated at m and s
            return self.Lfcn(mx,Sx)

        # add derivatives to the return values if requested
        else:
            # this will return values for dLdm and dLds evaluated at m and s
            return self.Lgrad(mx,Sx)

    def update(target=None,Q=None,cw=None,expl=None):
        # update the variables
        no_changes = True
        if target is not None:
            self.target = np.array(target)
            no_changes = False
        if Q is not None:
            self.Q = Q
            no_changes = False
        if cw is not None:
            self.cw = cw
            no_changes = False
        if expl is not None:
            self.expl = expl
            no_changes = False
        if no_changes:
            return

        # update the symbolic expressions for the loss and variance of loss
        L,S2 = self.loss_function(self,self.m,self.s)
        self.L = L
        self.S2 = S2

        # this will trigger a recompilation during next call of the fcn method
        self.Lfcn = None
        self.Lgrad = None

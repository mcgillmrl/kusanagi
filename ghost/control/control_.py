import numpy as np
from ghost.regression.GPRegressor import RBFGP
from utils import print_with_stamp, gTrig2_np
from ghost.control.saturation import gSat
from functools import partial
import theano

# GP based controller
class RBFPolicy:
    def __init__(self, m0, S0, maxU=[10], n_basis_functions=10, angle_idims=[]):
        self.m0 = np.array(m0)
        self.S0 = np.array(S0)
        self.maxU = np.array(maxU)
        self.n_basis_functions = n_basis_functions
        self.angle_idims = angle_idims
        # set the model to be a RBF with saturated outputs
        sat_func = partial(gSat, e=maxU)
        self.model = RBFGP(idims=self.m0.shape[0], odims=len(maxU), sat_func=sat_func)

        # check if we need to initialize
        params = self.get_params()
        for p in params:
            if p is None or p.size == 0:
                self.initialize_policy_default()
                break

    def initialize_policy_default(self):
        # init policy inputs near the given initial state
        m0 = self.m0
        S0 = self.S0
        if len(self.angle_idims)>0:
            m0, S0 = gTrig2_np(np.array(m0)[None,:], np.array(S0)[None,:,:], self.angle_idims, len(m0))
            m0 = m0.squeeze(); S0 = S0.squeeze();

        #self.inputs = np.random.multivariate_normal(m0,S0,n_basis_functions)
        L_noise = np.linalg.cholesky(S0)
        inputs = np.array([m0 + np.random.randn(S0.shape[1]).dot(L_noise) for i in xrange(self.n_basis_functions)]);
        # init policy targets close to zero
        targets = 0.1*np.random.randn(self.n_basis_functions,self.maxU.size)

        # set the initial inputs and targets
        self.model.set_dataset(inputs,targets)

        # set the initial log hyperparameters (1 for linear dimensions, 0.7 for angular)
        l0 = np.hstack([np.ones(self.m0.size-len(self.angle_idims)),0.7*np.ones(2*len(self.angle_idims)),1,0.01])
        self.model.set_loghyp(np.log(np.tile(l0,(self.maxU.size,1))))
        self.model.init_log_likelihood()
        self.model.init_predict()

    def evaluate(self, t, m, s=None, derivs=False, symbolic=False):
        D = m.shape[0]
        if s is None:
            s = np.zeros((D,D))
        if symbolic:
            ret = self.model.predict_symbolic(m,s)
        else:
            ret = self.model.predict(m,s) if not derivs else self.model.predict_d(m,s)
        return ret 

    def get_params(self, symbolic=False):
        return self.model.get_params(symbolic)

    def set_params(self,params):
        self.model.set_params(params)

    def save(self):
        self.model.save()

# random controller
class RandPolicy:
    def __init__(self, maxU=[10]):
        self.maxU = np.array(maxU)

    def evaluate(self, t, m, s=None, derivs=False):
        ret = ((2*np.random.random(self.maxU.size)-1.0)).reshape(self.maxU.shape)*self.maxU
        return ret

# linear time varying policy
class LocalLinearPolicy:
    def __init__(self, z_nominal, u_nominal, maxU=[10], angle_idims=[]):
        self.maxU = np.array(maxU)
        self.angle_idims = angle_idims
        self.H = len(u_nominal)

        self.u_nominal_ = np.array(u_nominal).squeeze()
        self.z_nominal_ = np.array(z_nominal).squeeze()
        self.b_ = np.zeros(t, u_nominal[0].shape[0])
        self.A_ = np.zeros(t, u_nominal[0].shape[0], z_nominal[0].shape[0])

        self.A = theano.shared(self.A_,borrow=True)
        self.b = theano.shared(self.b_,borrow=True)
        self.u_nominal = theano.shared(self.u_nominal_,borrow=True)
        self.z_nominal = theano.shared(self.z_nominal_,borrow=True)

    def evaluate(self, t, m, s=None, derivs=False, symbolic=False):
        D = m.shape[0]
        if symbolic:
            u_t = self.u_nominal[t]
            z_t = self.z_nominal[t]
            A_t = self.A[t]
            b_t = self.b[t]
            if s is None:
                s = np.zeros((D,D))
            z = np.concatenate([m,s.flatten()])
        else:
            u_t = self.u_nominal_[t]
            z_t = self.z_nominal_[t]
            A_t = self.A_[t]
            b_t = self.b_[t]
            if s is None:
                s = theano.tensor.zeros((D,D))
            z = theano.tensor.concatenate([m,s.flatten()])


        return u_t + b_t + A_t.dot(z - z_t)

    def get_params(self, symbolic=False):
        if symbolic:
            pass
        return self.model.get_params(symbolic)

    def set_params(self,params):
        self.model.set_params(params)

    def save(self):
        self.model.save()

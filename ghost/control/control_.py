import numpy as np
import theano
import sys

from theano.misc.pkl_utils import dump as t_dump, load as t_load
from ghost.regression.GPRegressor import RBFGP
from utils import print_with_stamp, gTrig2_np
from ghost.control.saturation import gSat
from functools import partial

class BaseControl(object):
    '''
        Class that specifies the basic interface for a controller. Any class implementing this interface should provide the  following methods:
        set_default_parameters, evaluate, get_params, set_params
    '''
    def __init__(self, name='BaseControl'):
        # load from disk
        self.load()
        # check if we need to initialize
        params = self.get_params()
        for p in params:
            if p is None or p.size == 0:
                self.set_default_parameters()
                break

    def set_default_parameters(self):
        raise NotImplementedError("You need to implement the set_default_parameters method in your BaseControl subclass.")

    def evaluate(self, m, s=None, derivs=False, symbolic=False):
        raise NotImplementedError("You need to implement evaluate method in your BaseControl subclass.")

    def get_params(self, symbolic=False):
        raise NotImplementedError("You need to implement the get_params method in your BaseControl subclass.")

    def set_params(self,params):
        raise NotImplementedError("You need to implement the set_params method in your BaseControl subclass.")

    def save(self):
        # call get_params and write them to disk
        pass

    def load(self):
        # load params from disk and pass them to set_params
        pass
        

# GP based controller
class RBFPolicy(RBFGP):
    def __init__(self, m0, S0, maxU=[10], n_basis_functions=10, angle_idims=[], name='RBFGP'):
        self.m0 = np.array(m0)
        self.S0 = np.array(S0)
        self.maxU = np.array(maxU)
        self.n_basis_functions = n_basis_functions
        self.angle_idims = angle_idims
        self.name = name
        # set the model to be a RBF with saturated outputs
        sat_func = partial(gSat, e=maxU)

        policy_idims = len(self.m0) + len(self.angle_idims)
        policy_odims = len(self.maxU)
        super(RBFPolicy, self).__init__(idims=policy_idims, odims=policy_odims, sat_func=sat_func, name=self.name)
        
        # check if we need to initialize
        params = self.get_params()
        for p in params:
            if p is None or p.size == 0:
                self.set_default_parameters()
                break

    def set_default_parameters(self):
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
        self.set_dataset(inputs,targets)

        # set the initial log hyperparameters (1 for linear dimensions, 0.7 for angular)
        l0 = np.hstack([np.ones(self.m0.size-len(self.angle_idims)),0.7*np.ones(2*len(self.angle_idims)),1,0.01])
        self.set_loghyp(np.log(np.tile(l0,(self.maxU.size,1))))
        self.init_log_likelihood()
        self.init_predict()

    def evaluate(self, m, s=None, derivs=False, symbolic=False):
        D = m.shape[0]
        if symbolic:
            if s is None:
                s = theano.tensor.zeros((D,D))
            ret = self.predict_symbolic(m,s)
        else:
            if s is None:
                s = np.zeros((D,D))
            ret = self.predict(m,s) if not derivs else self.predict_d(m,s)
        return ret 

# random controller
class RandPolicy:
    def __init__(self, maxU=[10]):
        self.maxU = np.array(maxU)

    def evaluate(self, m, s=None, derivs=False):
        ret = ((2*np.random.random(self.maxU.size)-1.0)).reshape(self.maxU.shape)*self.maxU
        return ret

    def save(self):
        pass # nothing to save

    def load(self):
        pass # nothing to load

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

    def evaluate(self, m, s=None, derivs=False, symbolic=False):
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

class AdjustedPolicy:
    def __init__(self, source_policy):
        # TODO initialize adjustment model
        self.source_policy = source_policy

    def evaluate(self, m, S=None, derivs=False, symbolic=False):
        T = theano.tensor if symbolic else np
        # get the output of the source policy
        ret = self.source_policy.evaluate(t,m,derivs,symbolic)
        
        # initialize the inputs to the policy adjustment function
        adj_input_m = m
        if self.use_control_input:
            adj_input_m = T.concatenate([adj_input_m,ret[0]])
        adj_D = adj_input_m.size
        adj_input_S = T.zeros((adj_D,adj_D))
        # TODO fill the covariance matrix appropriately if S is not None
        adj_ret = self.adjustment_model.evaluate(t,m,S,derivs,symbolic)
        # TODO fill the output covariance correctly
        ret[0] += adj_ret[0]
        return ret

    def get_params(self, symbolic=False):
        if symbolic:
            pass
        return self.adjustment_model.get_params(symbolic)

    def set_params(self,params):
        return self.adjustment_model.set_params(symbolic)

    def save(self):
        self.adjustment_model.save()

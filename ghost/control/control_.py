import numpy as np
import theano
import os, sys
import utils

from theano.misc.pkl_utils import dump as t_dump, load as t_load
from ghost.regression.GPRegressor import RBFGP
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

    def evaluate(self, m, s=None, t=None, derivs=False, symbolic=False):
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
    def __init__(self, m0, S0, maxU=[10], n_basis=10, angle_dims=[], name='RBFGP'):
        self.m0 = np.array(m0)
        self.S0 = np.array(S0)
        self.maxU = np.array(maxU)
        self.n_basis = n_basis
        self.angle_dims = angle_dims
        self.name = name
        # set the model to be a RBF with saturated outputs
        sat_func = partial(gSat, e=maxU)

        policy_idims = len(self.m0) + len(self.angle_dims)
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
        if len(self.angle_dims)>0:
            m0, S0 = utils.gTrig2_np(np.array(m0)[None,:], np.array(S0)[None,:,:], self.angle_dims, len(m0))
            m0 = m0.squeeze(); S0 = S0.squeeze();

        #self.inputs = np.random.multivariate_normal(m0,S0,n_basis)
        L_noise = np.linalg.cholesky(S0)
        inputs = np.array([m0 + np.random.randn(S0.shape[1]).dot(L_noise) for i in xrange(self.n_basis)]);
        # init policy targets close to zero
        teps,argets = 0.1*np.random.randn(self.n_basis,self.maxU.size)

        # set the initial inputs and targets
        self.set_dataset(inputs,targets)

        # set the initial log hyperparameters (1 for linear dimensions, 0.7 for angular)
        l0 = np.hstack([np.ones(self.m0.size-len(self.angle_dims)),0.7*np.ones(2*len(self.angle_dims)),1,0.01])
        self.set_loghyp(np.log(np.tile(l0,(self.maxU.size,1))))
        self.init_log_likelihood()
        self.init_predict()

    def evaluate(self, m, s=None, t=None, derivs=False, symbolic=False):
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

    def evaluate(self, m, s=None, t=None, derivs=False):
        ret = ((2*np.random.random(self.maxU.size)-1.0)).reshape(self.maxU.shape)*self.maxU
        return ret

    def save(self):
        pass # nothing to save

    def load(self):
        pass # nothing to load

# linear time varying policy
class LocalLinearPolicy(object):
    def __init__(self, H, dt, m0, S0=None, maxU=[10], angle_dims=[], name='LocalLinearPolicy'):
        self.maxU = np.array(maxU)
        self.angle_dims = angle_dims
        self.H = H
        self.dt = dt
        self.m0 = m0
        D = len(self.m0)
        self.S0 = S0 if S0 is not None else np.zeros((D,D))
        self.t = 0
        self.name = name
        self.set_default_parameters()

    def set_default_parameters(self):
        H_steps = int(np.ceil(self.H/self.dt))
        # set random (uniform distribution) controls
        self.u_nominal_ = self.maxU*np.random.random((H_steps,len(self.maxU)))
        m0, S0 = utils.gTrig2_np(np.array(self.m0)[None,:], np.array(self.S0)[None,:,:], self.angle_dims, len(self.m0))
        self.filename = self.name+'_'+str(len(self.m0))+'_'+str(len(self.maxU))
        z0 = np.concatenate([m0.flatten(),S0.flatten()])
        self.z_nominal_ = np.tile(z0,(H_steps,1))

        self.b_ = np.ones( (H_steps, len(self.maxU)) )
        self.A_ = np.zeros( (H_steps, len(self.maxU), z0.size) )
        print self.u_nominal_.shape
        print self.z_nominal_.shape
        print self.A_.shape
        print self.b_.shape

        self.A = theano.shared(self.A_,borrow=True)
        self.b = theano.shared(self.b_,borrow=True)
        self.u_nominal = theano.shared(self.u_nominal_,borrow=True)
        self.z_nominal = theano.shared(self.z_nominal_,borrow=True)

    def evaluate(self, m, s=None, t=None, derivs=False, symbolic=False):
        D = m.shape[0]

        if t is not None:
            self.t = t

        if symbolic:
            u_t = self.u_nominal[t]
            z_t = self.z_nominal[t]
            A_t = self.A[t]
            b_t = self.b[t]
            if s is None:
                s = theano.tensor.zeros((D,D))
            z = theano.tensor.concatenate([m,s.flatten()])
        else:
            u_t = self.u_nominal.get_value()[t]
            z_t = self.z_nominal.get_value()[t]
            A_t = self.A.get_value()[t]
            b_t = self.b.get_value()[t]
            if s is None:
                s = np.zeros((D,D))
            z = np.concatenate([m,s.flatten()])
        self.t+=1
        print '==='
        print self.u_nominal_.shape
        print self.z_nominal_.shape
        print self.A_.shape
        print self.b_.shape
        print '---'
        print u_t.shape
        print b_t.shape
        print A_t.shape
        print z_t.shape
        print z.shape
        print u_t + b_t + A_t.dot(z - z_t)
        return u_t + b_t + A_t.dot(z - z_t)

    def get_params(self, symbolic=False):
        if symbolic:
            return (self.u_nominal,self.z_nominal,self.A,self.b)
        else:
            return (self.u_nominal.get_value(),self.z_nominal.get_value(),self.A.get_value(),self.b.get_value())

    def set_params(self,Ain,Bin,uin,zin):
        self.A_= Ain.astype(theano.config.floatX)
        self.A.set_value(self.A_,borrow=True)

        self.B_= Ain.astype(theano.config.floatX)
        self.B.set_value(self.B_,borrow=True)

        self.u_= uin.astype(theano.config.floatX)
        self.u.set_value(self.u_,borrow=True)

        self.z_= zin.astype(theano.config.floatX)
        self.z.set_value(self.z_,borrow=True)

    def load(self):
        # load the parameters of the policy
        path = os.path.join(utils.get_run_output_dir(),self.filename+'.zip')
        with open(path,'rb') as f:
            utils.print_with_stamp('Loading %s from %s.zip'%(self.name, self.filename),self.name)
            self.A, self.b, self.u_nominal, self.z_nominal, self.A_, self.b_, self.u_nominal_, self.z_nominal_ = t_load(f)
        self.state_changed = False
    
    def save(self):
        # save policy and experience separately
        self.policy.save()
        self.experience.save()

        # save learner state
        sys.setrecursionlimit(100000)
        if self.state_changed:
            path = os.path.join(utils.get_run_output_dir(),self.filename+'.zip')
            with open(path,'wb') as f:
                utils.print_with_stamp('Saving learner state to %s.zip'%(self.filename),self.name)
                state = (self.A, self.b, self.u_nominal, self.z_nominal, self.A_, self.b_, self.u_nominal_, self.z_nominal_)
                t_dump(state,f,2)
            self.state_changed = False

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

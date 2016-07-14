import numpy as np
import theano
import os, sys
import utils

from theano.misc.pkl_utils import dump as t_dump, load as t_load
from ghost.regression.GP import RBFGP, SSGP_UI
from ghost.regression.NN import NN
from ghost.control.saturation import gSat
from functools import partial
from utils import gTrig2, gTrig2_np

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
    def __init__(self, m0=None, S0=None, maxU=[10], n_basis=10, angle_dims=[], name='RBFGP', filename=None):
        self.maxU = np.array(maxU)
        self.n_basis = n_basis
        self.angle_dims = angle_dims
        self.name = name
        # set the model to be a RBF with saturated outputs
        sat_func = partial(gSat, e=maxU)

        if filename is not None:
            # try loading from file
            self.uncertain_inputs = True
            self.X_ = None; self.Y_ = None; self.loghyp_=None
            self.filename = filename
            self.sat_func = sat_func
            self.load()
        else:
            self.m0 = np.array(m0)
            self.S0 = np.array(S0)

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
        targets = 0.1*np.random.randn(self.n_basis,self.maxU.size)

        # set the initial inputs and targets
        self.set_dataset(inputs,targets)

        # set the initial log hyperparameters (1 for linear dimensions, 0.7 for angular)
        l0 = np.hstack([np.ones(self.m0.size-len(self.angle_dims)),0.7*np.ones(2*len(self.angle_dims)),1,0.01])
        self.set_loghyp(np.log(np.tile(l0,(self.maxU.size,1))))
        self.init_loss(cache_vars=False)
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
        self.add_noise = True
        self.set_default_parameters()

    def set_default_parameters(self):
        H_steps = int(np.ceil(self.H/self.dt))
        self.state_changed = False
        # set random (uniform distribution) controls
        self.u_nominal_ = self.maxU*(2*np.random.random((H_steps,len(self.maxU))) - 1)
        m0, S0 = utils.gTrig2_np(np.array(self.m0)[None,:], np.array(self.S0)[None,:,:], self.angle_dims, len(self.m0))
        self.filename = self.name+'_'+str(len(self.m0))+'_'+str(len(self.maxU))
        z0 = np.concatenate([m0.flatten(),S0.flatten()])
        self.z_nominal_ = np.tile(z0,(H_steps,1))

        self.b_ = np.zeros( (H_steps, len(self.maxU)) )
        self.A_ = np.zeros( (H_steps, len(self.maxU), z0.size) )

        self.A = theano.shared(self.A_,borrow=True)
        self.b = theano.shared(self.b_,borrow=True)
        self.u_nominal = theano.shared(self.u_nominal_,borrow=True)
        self.z_nominal = theano.shared(self.z_nominal_,borrow=True)

    def evaluate(self, m, s=None, t=None, derivs=False, symbolic=False, alpha = 1, u = None, use_gTrig = False):
        D = m.shape[0]
        if t is not None:
            self.t = t
        t = self.t
        if symbolic:
            u_t = self.u_nominal[t]
            z_t = self.z_nominal[t]
            A_t = self.A[t]
            b_t = self.b[t]
            if s is None:
                s = theano.tensor.zeros((D,D))
            z = theano.tensor.concatenate([m.flatten(),s.flatten()])
            if theano.tensor.neq(z.shape[0],z_t.shape[0]):
                m,s,_ =  gTrig2(m, s, self.angle_dims, len(self.m0))
                z = theano.tensor.concatenate([m.flatten(),s.flatten()])
        else:
            u_t = self.u_nominal.get_value()[t]
            z_t = self.z_nominal.get_value()[t]
            A_t = self.A.get_value()[t]
            b_t = self.b.get_value()[t]
            if s is None:
                s = np.zeros((D,D))
            if use_gTrig:
                m,s,_ = gTrig2_np(m, s, self.angle_dims, D)
            z = np.concatenate([m.flatten(),s.flatten()])
        self.t+=1
        if u is not None:
            u_t = u
        U = u_t.shape[0]
        new_action  = u_t + alpha*(b_t + A_t.dot(z - z_t))
        if symbolic:
            new_action = theano.tensor.maximum(new_action, -self.maxU)
            new_action = theano.tensor.minimum(new_action, self.maxU)
        else:
            new_action[new_action>self.maxU] = self.maxU[new_action>self.maxU]
            new_action[new_action<-self.maxU] = self.maxU[new_action<-self.maxU]
            if self.add_noise:
                new_action = new_action + np.random.multivariate_normal(np.zeros(new_action.shape), 0.01*np.eye(new_action.size))
        return new_action, 0.01*theano.tensor.eye(U), theano.tensor.zeros((D,U))

    def get_params(self, symbolic=False, t=None):
        if symbolic:
            if t is None:
                return (self.u_nominal,self.z_nominal,self.A,self.b)
            else:
                return (self.u_nominal[t],self.z_nominal[t],self.A[t],self.b[t])
        else:
            return (self.u_nominal.get_value(),self.z_nominal.get_value(),self.A.get_value(),self.b.get_value())

    def set_params(self,Ain = None,Bin = None ,uin = None,zin = None):
        if Ain is not None:
            self.A_= Ain.astype(theano.config.floatX)
            self.A.set_value(self.A_,borrow=True)

        if Bin is not None:
            self.B_= Bin.astype(theano.config.floatX)
            self.b.set_value(self.B_,borrow=True)

        if uin is not None:
            self.u_= uin.astype(theano.config.floatX)
            self.u_nominal.set_value(self.u_,borrow=True)

        if zin is not None:
            self.z_= zin.astype(theano.config.floatX)
            self.z_nominal.set_value(self.z_,borrow=True)

    def load(self):
        # load the parameters of the policy
        path = os.path.join(utils.get_run_output_dir(),self.filename+'.zip')
        with open(path,'rb') as f:
            utils.print_with_stamp('Loading %s from %s.zip'%(self.name, self.filename),self.name)
            self.A, self.b, self.u_nominal, self.z_nominal, self.A_, self.b_, self.u_nominal_, self.z_nominal_ = t_load(f)
        self.state_changed = False
    
    def save(self):
        # save policy and experience separately
        # self.policy.save()
        # self.experience.save()

        # save learner state
        sys.setrecursionlimit(100000)
        if self.state_changed:
            path = os.path.join(utils.get_run_output_dir(),self.filename+'.zip')
            with open(path,'wb') as f:
                utils.print_with_stamp('Saving learner state to %s.zip'%(self.filename),self.name)
                state = (self.A, self.b, self.u_nominal, self.z_nominal, self.A_, self.b_, self.u_nominal_, self.z_nominal_)
                t_dump(state,f,2)
            self.state_changed = False

    def get_all_shared_vars(self):
        return [attr for attr in self.__dict__.values() if isinstance(attr,theano.tensor.sharedvar.SharedVariable)]


class AdjustedPolicy:
    def __init__(self, source_policy, maxU=[10], angle_dims=[], name='AdjustedPolicy', adjustment_model_class=SSGP_UI, use_control_input=True):
        self.use_control_input = use_control_input
        self.source_policy = source_policy
        self.angle_dims = angle_dims
        self.name = name
        self.maxU=maxU
        self.adjustment_model = adjustment_model_class(idims=self.source_policy.D, odims=self.source_policy.E) #TODO we may add a saturatinig function here

    def evaluate(self, m, S=None, t=None, derivs=False, symbolic=False):
        T = theano.tensor if symbolic else np
        # get the output of the source policy
        ret = self.source_policy.evaluate(m,S,t,derivs,symbolic)
        
        # initialize the inputs to the policy adjustment function
        adj_input_m = m
        if self.use_control_input:
            adj_input_m = T.concatenate([adj_input_m,ret[0]])
        adj_D = adj_input_m.size
        adj_input_S = T.zeros((adj_D,adj_D))
        # TODO fill the covariance matrix appropriately if S is not None
        if self.adjustment_model.trained == True:
            if symbolic:
                adj_ret = self.adjustment_model.predict_symbolic(adj_input_m,adj_input_S) #TODO change predict symbolic to evaluate
            else:
                adj_ret = self.adjustment_model.predict(adj_input_m,adj_input_S) #TODO change predict symbolic to evaluate
            #adj_ret = self.adjustment_model.evaluate(t,m,S,derivs,symbolic)
            # TODO fill the output covariance correctly
            ret[0] += adj_ret[0]
        return ret

    def get_params(self, symbolic=False):
        if symbolic:
            pass
        return self.adjustment_model.get_params(symbolic)

    def set_params(self,params):
        return self.adjustment_model.set_params(symbolic)

    def load(self):
        self.adjustment_model.load()

    def save(self):
        self.adjustment_model.save()

# GP based controller
class NNPolicy(NN):
    def __init__(self, m0=None, S0=None, maxU=[10], hidden_dims=[20,20,20], angle_dims=[], name='NNPolicy', filename=None):
        self.maxU = np.array(maxU)
        self.angle_dims = angle_dims
        self.name = name
        # set the model to be a RBF with saturated outputs
        sat_func = partial(gSat, e=maxU)

        if filename is not None:
            # try loading from file
            self.uncertain_inputs = True
            self.X_ = None; self.Y_ = None; self.loghyp_=None
            self.filename = filename
            self.sat_func = sat_func
            self.load()
        else:
            self.m0 = np.array(m0)
            self.S0 = np.array(S0)

            policy_idims = len(self.m0) + len(self.angle_dims)
            policy_odims = len(self.maxU)
            super(NNPolicy, self).__init__(idims=policy_idims, hidden_dims=hidden_dims, odims=policy_odims, sat_func=sat_func, name=self.name)
            
            # check if we need to initialize
            params = self.get_params()
            for p in params:
                if p is None or p.size == 0:
                    self.set_default_parameters()
                    break

    def set_default_parameters(self):
        self.init_loss()
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

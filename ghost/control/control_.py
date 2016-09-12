import numpy as np
import theano
import utils

from ghost.regression.GP import RBFGP, SSGP_UI
from ghost.regression.NN import NN
from ghost.control.saturation import gSat
from functools import partial
from utils import gTrig2, gTrig2_np
from base.Loadable import Loadable

# GP based controller
class RBFPolicy(RBFGP):
    def __init__(self, m0=None, S0=None, maxU=[10], n_basis=10, angle_dims=[], name='RBFPolicy', filename=None):
        self.maxU = np.array(maxU)
        self.n_basis = n_basis
        self.angle_dims = angle_dims
        self.name = name
        # set the model to be a RBF with saturated outputs
        sat_func = partial(gSat, e=maxU)

        if filename is not None:
            # try loading from file
            super(RBFPolicy, self).__init__(idims=0, odims=0, sat_func=sat_func, name=self.name, filename=filename)
            self.load()
        else:
            self.m0 = np.array(m0)
            self.S0 = np.array(S0)

            policy_idims = len(self.m0) + len(self.angle_dims)
            policy_odims = len(self.maxU)
            super(RBFPolicy, self).__init__(idims=policy_idims, odims=policy_odims, sat_func=sat_func, name=self.name)
        
        # make sure we always get the parameters in the same order
        self.param_names = ['X','Y','loghyp_full']

    def load(self, output_folder=None,output_filename=None):
        ''' loads the state from file, and initializes additional variables'''
        # load state
        super(RBFGP,self).load(output_folder,output_filename)
        
        # initialize mising variables
        self.loghyp = theano.tensor.concatenate([self.loghyp_full[:,:-2], theano.gradient.disconnected_grad(self.loghyp_full[:,-2:])], axis=1)
        # loghyp is no longer the trainable paramter
        if 'loghyp' in self.param_names: self.param_names.remove('loghyp')

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
        self.set_params( {'loghyp_full': np.log(np.tile(l0,(self.maxU.size,1)))} )
        
        # don't optimize the signal and noise variances
        self.loghyp = theano.tensor.concatenate([self.loghyp_full[:,:-2], theano.gradient.disconnected_grad(self.loghyp_full[:,-2:])], axis=1)
        # loghyp is no longer the trainable paramter
        if 'loghyp' in self.param_names: self.param_names.remove('loghyp')

        self.init_loss(cache_vars=False)
        self.init_predict()

    def evaluate(self, m, s=None, t=None, symbolic=False):
        D = m.shape[0]
        if symbolic:
            if s is None:
                s = theano.tensor.zeros((D,D))
            ret = self.predict_symbolic(m,s)
        else:
            if s is None:
                s = np.zeros((D,D))
            ret = self.predict(m,s)
        return ret 

# random controller
class RandPolicy:
    def __init__(self, maxU=[10], random_walk=False):
        self.maxU = np.array(maxU)
        #self.last_u = np.zeros_like(np.array(maxU))
        self.last_u = 0.5*((2*np.random.random(self.maxU.size)-1.0)).reshape(self.maxU.shape)*self.maxU
        self.random_walk=random_walk
        

    def evaluate(self, m, s=None, t=None, symbolic=False):
        if self.random_walk:
            ret = self.last_u + 0.3*((2*np.random.random(self.maxU.size)-1.0)).reshape(self.maxU.shape)*self.maxU
            ret = np.min ( (ret.flatten(), self.maxU.flatten()), axis=0  ) 
            ret = np.max ( (ret.flatten(), -self.maxU.flatten()), axis=0  ) 
            ret = ret.reshape(self.maxU.shape)
        else:
            ret = ((2*np.random.random(self.maxU.size)-1.0)).reshape(self.maxU.shape)*self.maxU

        self.last_u = ret
        U = len(self.maxU)
        D = m.shape[0]
        return ret, np.zeros((U,U)), np.zeros((D,U))

# linear time varying policy
class LocalLinearPolicy(Loadable):
    def __init__(self, H, dt, m0, S0=None, maxU=[10], angle_dims=[], name='LocalLinearPolicy', **kwargs):
        self.maxU = np.array(maxU)
        self.angle_dims = angle_dims
        self.H = H
        self.dt = dt
        self.m0 = m0
        D = len(self.m0)
        self.S0 = S0 if S0 is not None else np.zeros((D,D))
        self.t = 0
        self.noise = 0
        self.name = name
        self.set_default_parameters()

        Loadable.__init__(self,name=name,filename=self.filename)
        # register theano functions and shared variables for saving
        self.register_types([theano.tensor.sharedvar.SharedVariable, theano.compile.function_module.Function])

    def set_default_parameters(self):
        H_steps = int(np.ceil(self.H/self.dt))
        self.state_changed = False

        # set random (uniform distribution) controls
        u = self.maxU*(2*np.random.random((H_steps,len(self.maxU))) - 1)
        self.u_nominal = theano.shared(u)

        # intialize the nominal states to the appropriate size
        m0, S0 = utils.gTrig2_np(np.array(self.m0)[None,:], np.array(self.S0)[None,:,:], self.angle_dims, len(self.m0))
        self.triu_indices = np.triu_indices(m0.size)
        z0 = np.concatenate([m0.flatten(),S0[0][self.triu_indices]])
        z = np.tile(z0,(H_steps,1))
        self.z_nominal = theano.shared(z)

        # initialize the open loop and feedback matrices 
        I = np.zeros( (H_steps, len(self.maxU)) )
        L = np.zeros( (H_steps, len(self.maxU), z0.size) )
        self.I = theano.shared(I)
        self.L = theano.shared(L)
        
        # set a meaningful filename
        self.filename = self.name+'_'+str(len(self.m0))+'_'+str(len(self.maxU))

    def evaluate(self, m, s=None, t=None, symbolic=False):
        D = m.shape[0]
        if t is not None:
            self.t = t
        t = self.t

        u,z,I,L = self.u_nominal,self.z_nominal,self.I,self.L

        if symbolic:
            T = theano.tensor
        else:
            T = np
            u,z,I,L=u.get_value(),z.get_value(),I.get_value(),L.get_value()
            
        if s is None:
            s = T.zeros((D,D))
        
        # construct flattened state covariance vector
        z_t = T.concatenate([m.flatten(),s[self.triu_indices]])
        # compute control
        u_t = u[t] + I[t] + L[t].dot(z_t - z[t])
        # add random noise if requested (only for non symbolic)
        if not symbolic and self.noise and self.noise > 0:
            u_t += self.noise*T.random.randn(*u_t.shape)

        # limit the controller output
        #u_t = T.maximum(u_t, -self.maxU)
        #u_t = T.minimum(u_t, self.maxU)

        U = u_t.shape[0]
        self.t+=1
        return u_t, T.zeros((U,U)), T.zeros((D,U))

    def get_params(self, symbolic=False, t=None):
        params = [self.u_nominal,self.z_nominal,self.I,self.L]

        if not symbolic:
            params = [ p.get_value() for p in params]
        return params

    def get_all_shared_vars(self):
        return [attr for attr in self.__dict__.values() if isinstance(attr,theano.tensor.sharedvar.SharedVariable)]

class AdjustedPolicy:
    def __init__(self, source_policy, maxU=[10], angle_dims=[], name='AdjustedPolicy', adjustment_model_class=SSGP_UI, use_control_input=True, **kwargs):
        self.use_control_input = use_control_input
        self.angle_dims = angle_dims
        self.name = name
        self.maxU=maxU
        
        self.source_policy = source_policy
        #self.source_policy.init_loss(cache_vars=False)
        #self.source_policy.init_predict()
        
        self.adjustment_model = adjustment_model_class(idims=self.source_policy.D, odims=self.source_policy.E) #TODO we may add a saturatinig function here
    
    def set_default_parameters(self):
        pass

    def evaluate(self, m, S=None, t=None, symbolic=False):
        T = theano.tensor if symbolic else np
        # get the output of the source policy
        ret = self.source_policy.evaluate(m,S,t,symbolic)
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
            print m,ret[0], adj_ret[0]

            ret[0] += adj_ret[0]
        else:
            print m,ret[0]

        return ret

    def get_params(self, symbolic=False):
        if symbolic:
            pass
        return self.adjustment_model.get_params(symbolic)

    def set_params(self,params):
        return self.adjustment_model.set_params(symbolic)

    def load(self, output_folder=None,output_filename=None):
        self.adjustment_model.load(output_folder,output_filename)

    def save(self, output_folder=None,output_filename=None):
        self.adjustment_model.save(output_folder,output_filename)

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

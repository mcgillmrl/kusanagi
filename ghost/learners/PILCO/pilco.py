from matplotlib import pyplot as plt
import numpy as np
import os
import sys
import theano
import utils
import time

from theano.misc.pkl_utils import dump as t_dump, load as t_load
from theano.compile.nanguardmode import NanGuardMode
from ghost.learners.EpisodicLearner import *
from ghost.regression.GP import GP_UI, SPGP_UI, SSGP_UI

class PILCO(EpisodicLearner):
    def __init__(self, params, plant_class, policy_class, cost_func=None, viz_class=None, dynmodel_class=GP_UI, experience = None, async_plant=False, name='PILCO', wrap_angles=False, use_scan=True, filename_prefix=None):
        self.use_scan = use_scan
        self.dynamics_model = None
        self.wrap_angles = wrap_angles
        self.rollout_fn=None
        self.policy_gradient_fn=None
        self.mx0 = np.array(params['x0']).squeeze()
        self.Sx0 = np.array(params['S0']).squeeze()
        self.angle_idims = params['angle_dims']
        self.maxU = params['policy']['maxU']

        # input dimensions to the dynamics model are (state dims - angle dims) + 2*(angle dims) + control dims
        dyn_idims = len(self.mx0) + len(self.angle_idims) + len(self.maxU)
        # output dimensions are state dims
        dyn_odims = len(self.mx0)
        # initialize dynamics model (TODO pass this as argument to constructor)
        if 'dynmodel' not in params:
            params['dynmodel'] = {}
        params['dynmodel']['idims'] = dyn_idims
        params['dynmodel']['odims'] = dyn_odims

        self.dynamics_model = dynmodel_class(**params['dynmodel'])
        self.next_episode = 0

        # initialise parent class
        filename_prefix = name+'_'+self.dynamics_model.name if filename_prefix is None else filename_prefix
        super(PILCO, self).__init__(params, plant_class, policy_class, cost_func,viz_class, experience, async_plant, name, filename_prefix)
    
    def save(self):
        ''' Saves the state of the learner, including the parameters of the policy and the dynamics model'''
        super(PILCO,self).save()
        self.dynamics_model.save()

    def load(self):
        ''' Loads the state of the learner, including the parameters of the policy and the dynamics model'''
        super(PILCO,self).load()
        self.dynamics_model.load()
    
    def set_state(self,state):
        ''' In addition to the EpisodicLearner state variables, saves the values of self.wrap_angles, self.next_episode, self.mx0 and self.Sx0'''
        i = utils.integer_generator(-4)
        self.wrap_angles = state[i.next()]
        self.next_episode = state[i.next()]
        self.mx0 = state[i.next()]
        self.Sx0 = state[i.next()]
        super(PILCO,self).set_state(state[:-4])

    def get_state(self):
        ''' In addition to the EpisodicLearner state variables, loads the values of self.wrap_angles, self.next_episode, self.mx0 and self.Sx0'''
        state = super(PILCO,self).get_state()
        state.append(self.wrap_angles)
        state.append(self.next_episode)
        state.append(self.mx0)
        state.append(self.Sx0)
        return state

    def save_rollout(self):
        ''' Saves the compiled rollout and policy_gradient functions, along with the associated shared variables from the dynamics model and policy. The shared variables from the dynamics model adn the policy will be replaced with whatever is loaded, to ensure that the compiled rollout and policy_gradient functions are consistently updated, when the parameters of the dynamics_model and policy objects are changed. Since we won't store the latest state of these shared variables here, we will copy the values of the policy and dynamics_model parameters into the state of the shared variables. If the policy and dynamics_model parameters have been updated, we will need to load them before calling this function.'''
        sys.setrecursionlimit(100000)
        path = os.path.join(utils.get_output_dir(),self.filename+'_rollout.zip')
        with open(path,'wb') as f:
            utils.print_with_stamp('Saving compiled rollout to %s_rollout.zip'%(self.filename),self.name)
            # this saves the shared variables used by the rollout and policy gradients. This means that the zip file
            # will store the state of those variables from when this function is called
            # TODO save the shared variables form this class
            t_vars = [self.dynamics_model.get_state(), self.policy.get_state(), self.rollout_fn, self.policy_gradient_fn]
            t_dump(t_vars,f,2)

    def load_rollout(self):
        ''' Loads the compiled rollout and policy_gradient functions, along with the associated shared variables from the dynamics model and policy. The shared variables from the dynamics model adn the policy will be replaced with whatever is loaded, to ensure that the compiled rollout and policy_gradient functions are consistently updated, when the parameters of the dynamics_model and policy objects are changed. Since we won't store the latest state of these shared variables here, we will copy the values of the policy and dynamics_model parameters into the state of the shared variables. If the policy and dynamics_model parameters have been updated, we will need to load them before calling this function.'''
        path = os.path.join(utils.get_output_dir(),self.filename+'_rollout.zip')
        with open(path,'rb') as f:
            utils.print_with_stamp('Loading compiled rollout from %s_rollout.zip'%(self.filename),self.name)
            t_vars = t_load(f)
            # here we are loading state variables that are probably outdated, but that are tied to the compiled rollout and policy_gradient functions
            # we need to restore whatever value the dataset and loghyp variables had, which is why we call get_params before replace the state variables
            params = self.dynamics_model.get_params(symbolic=False)
            self.dynamics_model.set_state(t_vars[0])
            self.dynamics_model.set_params(params)

            params = self.policy.get_params(symbolic=False)
            self.policy.set_state(t_vars[1])
            self.policy.set_params(params)
            
            # At this point the dynamics model and policy state variables should be tied to the rollout and policy_graddient function, and contain the up to date values of the
            # parameters
            self.rollout_fn = t_vars[2]
            self.policy_gradient_fn = t_vars[3]

    # define the function for a single propagation step
    def rollout_single_step(self,mx,Sx):
        D=mx.shape[0]

        # convert angles from input distribution to its complex representation
        mxa,Sxa,Ca = utils.gTrig2(mx,Sx,self.angle_idims,self.mx0.size)

        # compute distribution of control signal
        logsn = self.dynamics_model.loghyp[:,-1]
        Sx_ = Sx + theano.tensor.diag(0.5*theano.tensor.exp(2*logsn))# noisy state measurement
        mxa_,Sxa_,Ca_ = utils.gTrig2(mx,Sx_,self.angle_idims,self.mx0.size)
        mu, Su, Cu = self.policy.evaluate(mxa_, Sxa_,symbolic=True)
        
        # compute state control joint distribution
        n = Sxa.shape[0]; Da = Sxa.shape[1]; U = Su.shape[1]
        mxu = theano.tensor.concatenate([mxa,mu])
        q = Sxa.dot(Cu)
        Sxu_up = theano.tensor.concatenate([Sxa,q],axis=1)
        Sxu_lo = theano.tensor.concatenate([q.T,Su],axis=1)
        Sxu = theano.tensor.concatenate([Sxu_up,Sxu_lo],axis=0) # [D+U]x[D+U]

        # state control covariance without angle dimensions
        if Ca is not None:
            na_dims = list(set(range(self.mx0.size)).difference(self.angle_idims))
            Sx_xa = theano.tensor.concatenate([Sx[:,na_dims],Sx.dot(Ca)],axis=1)  # [D] x [Da] 
            Sxu_ =  theano.tensor.concatenate([Sx_xa,Sx_xa.dot(Cu)],axis=1) # [D] x [Da+U]
        else:
            Sxu_ = Sxu[:D,:] # [D] x [D+U]

        #  predict the change in state given current state-action
        # C_deltax = inv (Sxu) dot Sxu_deltax
        m_deltax, S_deltax, C_deltax = self.dynamics_model.predict_symbolic(mxu,Sxu)

        # compute the successor state distribution
        mx_next = mx + m_deltax
        Sx_deltax = Sxu_.dot(C_deltax)
        Sx_next = Sx + S_deltax + Sx_deltax + Sx_deltax.T

        #  get cost:
        mcost, Scost = self.cost_symbolic(mx_next,Sx_next)

        return [mcost,Scost,mx_next,Sx_next]

    def init_rollout(self, derivs=False):
        ''' This compiles the rollout function, which applies the policy and predicts the next state 
            of the system using the learn GP dynamics model. If loading from disk, this should be called ONLY
            after calls to self.dynamics_model.load() and self.policy.load(). (See load_rollout for details)'''
        if self.use_scan:
            self.init_rollout_scan(derivs)
        else:
            self.init_rollout_step(derivs)

    def init_rollout_scan(self, derivs=False):
        ''' This method compiles the rollout function using a Theano scan loop'''
        loaded_from_disk = True
        try:
            self.load_rollout()
            # if after loading from disk, the function we want does not exist
            if (self.rollout is None and not derivs) or (self.policy_gradient is None and derivs):
                self.init_rollout(derivs=derivs)
                loaded_from_disk = False
        except IOError:
            utils.print_with_stamp('Initialising rollout [ Could not open %s_rollout.zip ]'%(self.filename),self.name)
            loaded_from_disk = False

        if loaded_from_disk:
            return

        # define input variables
        utils.print_with_stamp('Computing symbolic expression graph for belief state propagation',self.name)
        mx = theano.tensor.vector('mx')
        Sx = theano.tensor.matrix('Sx')
        H = theano.tensor.iscalar('H')
        gamma = theano.tensor.scalar('gamma')

        # this will generate the list of value and state distributions for each time step
        def get_discounted_reward(mv,Sv,mx,Sx,gamma,*args):
            mv_next, Sv_next, mx_next, Sx_next = self.rollout_single_step(mx,Sx)
            return gamma*mv_next,(gamma**2)*Sv_next, mx_next, Sx_next, gamma*gamma
        
        # this are the initial values for the value and its variance
        mv0, Sv0 = self.cost_symbolic(mx,Sx)

        # these are the shared variables that will be used in the graph, we need to let theano know about these
        # to reduce compilation times
        shared_vars = []
        shared_vars.extend(self.dynamics_model.get_all_shared_vars())
        shared_vars.extend(self.policy.get_all_shared_vars())
        (mV_,SV_,mx_,Sx_,gamma_), updts = theano.scan(fn=get_discounted_reward, 
                                                      outputs_info=[mv0,Sv0,mx,Sx,gamma], 
                                                      non_sequences=shared_vars,
                                                      n_steps=H, 
                                                      strict=True,
                                                      allow_gc=False)
        mV_ = theano.tensor.concatenate([mv0.dimshuffle('x'), mV_])
        SV_ = theano.tensor.concatenate([Sv0.dimshuffle('x'), SV_])
        mx_ = theano.tensor.concatenate([mx.dimshuffle('x',0), mx_])
        Sx_ = theano.tensor.concatenate([Sx.dimshuffle('x',0,1), Sx_])

            
        utils.print_with_stamp('Compiling belief state propagation',self.name)
        self.rollout_fn = theano.function([mx,Sx,H,gamma], (mV_,SV_,mx_,Sx_), allow_input_downcast=True, updates=updts)
        if derivs :
            utils.print_with_stamp('Computing symbolic expression for policy gradients',self.name)
            dretvars = [mV_.sum()]
            params = self.policy.get_params(symbolic=True)
            if not isinstance(params,list):
                params = [params]
            for p in params:
                dretvars.append( theano.tensor.grad(mV_.sum(), p ) ) # we are only interested in the derivative of the sum of expected values
            
            utils.print_with_stamp('Compiling policy gradients',self.name)
            self.policy_gradient_fn = theano.function([mx,Sx,H,gamma], dretvars, allow_input_downcast=True, updates=updts)#,mode=NanGuardMode(nan_is_error=True,inf_is_error=True,big_is_error=False))
        
        utils.print_with_stamp('Done compiling.',self.name)
        #self.save_rollout()

    def init_rollout_step(self, dervis=False):
        ''' This compiles the rollout function, which applies the policy and predicts the next state 
            of the system using the learn GP dynamics model. This version avoids using the theano scan'''
        # define shared state variables
        self.mx = theano.shared(np.array(self.mx0).flatten(), name="mx")
        self.Sx = theano.shared(np.array(self.Sx0).squeeze(), name="Sx")
        self.gamma = theano.shared(np.array(self.discount,dtype=theano.config.floatX),'gamma')
        self.mV = theano.shared(np.array(0.0,theano.config.floatX),'mV')
        
        # define shared variables for storing the gradients
        D = self.mx.get_value().size
        params = self.policy.get_params(symbolic=False)
        n_params = len(params)
        self.dmdp = [ theano.shared(np.zeros((D,params[i].size),dtype=theano.config.floatX), name="dMdp%d"%(i)) for i in xrange(len(params))]
        self.dsdp = [ theano.shared(np.zeros((D*D,params[i].size),dtype=theano.config.floatX), name="dMdp%d"%(i)) for i in xrange(len(params))]
        self.dcdp = [ theano.shared(np.zeros((params[i].shape),dtype=theano.config.floatX), name="dcdp%d"%(i)) for i in xrange(len(params))]

        # create function for clearing/reinitializing shared variables
        mx0 = theano.tensor.vector('mx0')
        Sx0 = theano.tensor.matrix('Sx0')
        discount0 = theano.tensor.scalar('discount0')
        init_updates = [(self.mx,mx0), (self.Sx,Sx0), (self.mV, self.cost_symbolic(mx0,Sx0)[0]), (self.gamma, discount0)]
        zero_val = np.array(0.0,dtype=theano.config.floatX)
        for i in xrange(n_params):
            init_updates.append( (self.dcdp[i], self.dcdp[i].fill(zero_val)) )
            init_updates.append( (self.dmdp[i], self.dmdp[i].fill(zero_val)) )
            init_updates.append( (self.dsdp[i], self.dsdp[i].fill(zero_val)) )
        self.reset_rollout = theano.function([mx0,Sx0,discount0],[],updates=init_updates)

        utils.print_with_stamp('Computing symbolic expression graph for belief state propagation',self.name)
        # apply rollout
        mc_t, Sc_t, mx_next, Sx_next = self.rollout_single_step(self.mx,self.Sx)

        # partial derivatives of next state wrt previous state
        dMdm_t = theano.tensor.jacobian(mx_next,self.mx).flatten(2)
        dMds_t = theano.tensor.jacobian(mx_next,self.Sx).flatten(2)
        dSdm_t = theano.tensor.jacobian(Sx_next.flatten(),self.mx).flatten(2)
        dSds_t = theano.tensor.jacobian(Sx_next.flatten(),self.Sx).flatten(2)
        dcdM_t = theano.tensor.grad(mc_t,mx_next).flatten()[None,:]
        dcdS_t = theano.tensor.grad(mc_t,Sx_next).flatten()[None,:]
        
        params_t = self.policy.get_params(symbolic=True)
        dMdp_t = [[]]*n_params
        dSdp_t = [[]]*n_params
        dcdp_t = [[]]*n_params
        for i in xrange(n_params):
            # partial derivative of next state wrt policy params
            dMdp_t[i] = theano.tensor.jacobian(mx_next,params_t[i]).flatten(2)
            dSdp_t[i] = theano.tensor.jacobian(Sx_next.flatten(),params_t[i]).flatten(2)

            #   total derivative of next state wrt policy params
            dMdp_t[i]  = dMdm_t.dot(self.dmdp[i]) + dMds_t.dot(self.dsdp[i]) + dMdp_t[i]
            dSdp_t[i]  = dSdm_t.dot(self.dmdp[i]) + dSds_t.dot(self.dsdp[i]) + dSdp_t[i]

            dcdp_t[i] = (dcdM_t.dot(dMdp_t[i]) + dcdS_t.dot(dSdp_t[i])).reshape(params_t[i].shape)

        gamma_next= self.gamma*self.gamma

        # update shared variables
        updates = []
        updates.append((self.mx, mx_next))                           # update the rolled out state mean
        updates.append((self.Sx, Sx_next))                           # update the rolled out state covariance
        updates.append((self.gamma, gamma_next))                     # update the discount factor
        updates.append((self.mV, self.mV + gamma_next*mc_t))         # update the expected value of the long term cost V

        utils.print_with_stamp('Compiling belief state propagation',self.name)
        self.rollout_fn = theano.function([], [mc_t, Sc_t, mx_next, Sx_next], allow_input_downcast=True, updates=updates)

        for i in xrange(n_params):
            updates.append((self.dcdp[i], self.dcdp[i] + gamma_next*dcdp_t[i]))    # update the derivative of E{V} wrt the parameters
            updates.append((self.dmdp[i], dMdp_t[i]))              # update the derivative of E{V} wrt the parameters
            updates.append((self.dsdp[i], dSdp_t[i]))              # update the derivative of E{V} wrt the parameters

        utils.print_with_stamp('Compiling policy gradients',self.name)
        self.policy_gradient_fn = theano.function([], [], allow_input_downcast=True, updates=updates)
        
        utils.print_with_stamp('Done compiling.',self.name)

    def rollout(self, mx0, Sx0, H_steps, discount):
        if self.rollout_fn is None:
            # compile the belef state propagation
            self.init_rollout(derivs=False)

        if self.use_scan:
            # the rollout loop is performed inside rollout_fn
            return self.rollout_fn(mx0,Sx0,H_steps,discount)
        else:
            # reset shared vars
            self.reset_rollout(mx0,Sx0,discount)

            # rollout for H_steps and save the intermediate results
            retvars = []
            for i in xrange(H_steps):
                retvars.append(self.rollout_fn())
            
            # organize the return values in the appropriate format
            retvars = map(np.array, zip(*retvars))

            return retvars

    def policy_gradient(self, mx0, Sx0, H_steps, discount):
        if self.policy_gradient_fn is None:
            # compile the belef state propagation
            self.init_rollout(derivs=True)

        if self.use_scan:
            # the rollout loop is performed inside policy_gradient_fn
            return self.policy_gradient_fn(mx0,Sx0,H_steps,discount)
        else:
            # reset shared vars
            self.reset_rollout(mx0,Sx0,discount)

            # rollout for H_steps and save the intermediate results
            for i in xrange(H_steps):
                self.policy_gradient_fn()
            
            # organize the return values in the appropriate format
            retvars = [self.mV.get_value()]
            params = self.policy.get_params(symbolic=False)
            n_params = len(params)
            for i in xrange(n_params):
                retvars.append(self.dcdp[i].get_value())

            return retvars

    def train_dynamics(self):
        ''' Trains a dynamics model using the current experience dataset '''
        utils.print_with_stamp('Training dynamics model',self.name)

        X = []
        Y = []
        x0 = []
        n_episodes = len(self.experience.states)
        
        if n_episodes>0:
            # construct training dataset
            for i in xrange(self.next_episode,n_episodes):
                x = np.array(self.experience.states[i])
                u = np.array(self.experience.actions[i])
                x0.append(x[0])

                # inputs are states, concatenated with actions ( excluding the last entry) 
                x_ = utils.gTrig_np(x, self.angle_idims)
                X.append( np.hstack((x_[:-1],u[:-1])) )
                # outputs are changes in state
                Y.append( x[1:] - x[:-1] )

            self.next_episode = n_episodes 
            X = np.vstack(X)
            Y = np.vstack(Y)
            
            # wrap angles if requested (this might introduce error if the angular velocities are high )
            if self.wrap_angles:
                # wrap angle differences to [-pi,pi]
                Y[:,self.angle_idims] = (Y[:,self.angle_idims] + np.pi) % (2 * np.pi ) - np.pi

            # get distribution of initial states
            x0 = np.array(x0)
            if n_episodes > 1:
                self.mx0 = x0.mean(0)[None,:]
                self.Sx0 = np.cov(x0.T)[None,:,:]
            else:
                self.mx0 = x0[None,:]
                self.Sx0 = 1e-2*np.eye(self.mx0.size)[None,:,:]

            # append data to the dynamics model
            self.dynamics_model.append_dataset(X,Y)
        else:
            self.mx0 = np.array(self.plant.x0).squeeze()
            self.Sx0 = np.array(self.plant.S0).squeeze()

        utils.print_with_stamp('Dataset size:: Inputs: [ %s ], Targets: [ %s ]  '%(self.dynamics_model.X.get_value(borrow=True).shape,self.dynamics_model.Y.get_value(borrow=True).shape),self.name)
        if self.dynamics_model.should_recompile:
            # reinitialize log likelihood
            self.dynamics_model.init_loss()
            if self.rollout is not None:
                # reinitialize rollot and policy gradients
                self.init_rollout(derivs=True)
 
        self.dynamics_model.train()
        utils.print_with_stamp('Done training dynamics model',self.name)

    def value(self, derivs=False):
        '''Returns the value of the current policy by computing long term predictions using a learned dynamics model. If derivs is True, it will also return the policy gradients'''
        # we will perform a rollout with a horizon that is as long as the longest run, but at most self.H
        max_steps = max([len(episode_states) for episode_states in self.experience.states])
        H_steps = int(np.ceil(self.H/self.plant.dt))
        if max_steps > 1:
            H_steps = min(2*max_steps, H_steps)

        # if we have no data to compute the value, return dummy values
        if self.dynamics_model.N < 1:
            return np.zeros((H_steps,)),np.ones((H_steps,))

        # setp initial state
        mx = np.array(self.plant.x0).squeeze()
        Sx = np.array(self.plant.S0).squeeze()
        
        if not derivs:
            # self.H is the number of steps to rollout ( finite horizon )
            ret = self.rollout(mx,Sx,H_steps,self.discount)
            return ret[0].sum()
        else:
            ret = self.policy_gradient(mx,Sx,H_steps,self.discount)
            # first return argument is the value of the policy, second are the gradients wrt the policy params
            return [ret[0], ret[1:]]

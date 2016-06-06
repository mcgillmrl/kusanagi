from matplotlib import pyplot as plt
import numpy as np
import os
import sys
import theano
import utils

from theano.misc.pkl_utils import dump as t_dump, load as t_load

from ghost.learners.EpisodicLearner import *
from ghost.regression.GPRegressor import GP_UI, SPGP_UI, SSGP_UI

class PILCO(EpisodicLearner):
    def __init__(self, plant, policy, cost, angle_idims=None, viz=None, discount=1, experience = None, async_plant = False, name='PILCO', wrap_angles=False):
        self.dynamics_model = None
        self.wrap_angles = wrap_angles
        self.rollout=None
        self.policy_gradient=None

        # input dimensions to the dynamics model are (state dims - angle dims) + 2*(angle dims) + control dims
        dyn_idims = len(plant.x0) + len(angle_idims) + len(policy.maxU)
        # output dimensions are state dims
        dyn_odims = len(plant.x0)
        # initialize dynamics model (TODO pass this as argument to constructor)
        self.dynamics_model = SSGP_UI(idims=dyn_idims,odims=dyn_odims,n_basis=100)
        self.next_episode = 0
        self.mx0 = np.array(plant.x0).squeeze()
        self.Sx0 = np.array(plant.S0).squeeze()

        # initialise parent class
        filename = name+'_'+plant.name+'_'+policy.name+'_'+self.dynamics_model.name
        super(PILCO, self).__init__(plant, policy, cost, angle_idims, viz,  discount, experience, async_plant, name, filename)
    
    def save(self):
        super(PILCO,self).save()
        self.dynamics_model.save()

    def load(self):
        super(PILCO,self).load()
        self.dynamics_model.load()
    
    def set_state(self,state):
        i = utils.integer_generator(-4)
        self.wrap_angles = state[i.next()]
        self.next_episode = state[i.next()]
        self.mx0 = state[i.next()]
        self.Sx0 = state[i.next()]
        super(PILCO,self).set_state(state[:-4])

    def get_state(self):
        state = super(PILCO,self).get_state()
        state.append(self.wrap_angles)
        state.append(self.next_episode)
        state.append(self.mx0)
        state.append(self.Sx0)
        return state

    def save_rollout(self):
        sys.setrecursionlimit(100000)
        path = os.path.join(utils.get_output_dir(),self.filename+'_rollout.zip')
        with open(path,'wb') as f:
            utils.print_with_stamp('Saving compiled rollout to %s_rollout.zip'%(self.filename),self.name)
            # this saves the shared variables used by the rollout and policy gradients. This means that the zip file
            # will store the state of those variables from when this function is called
            t_vars = [self.dynamics_model.get_state(), self.policy.get_state(), self.rollout, self.policy_gradient]
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
            self.rollout = t_vars[2]
            self.policy_gradient = t_vars[3]

    def init_rollout(self, derivs=False):
        ''' This compiles the rollout function, which applies the policy and predicts the next state 
            of the system using the learn GP dynamics model. If loading from disk, this should be called ONLY
            after calls to self.dynamics_model.load() and self.policy.load(). (See load_rollout for details)'''
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

        # define the function for a single propagation step
        def rollout_single_step(mx,Sx):
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
            Sxu = theano.tensor.concatenate([Sxu_up,Sxu_lo],axis=0)

            # state control covariance without angle dimensions
	    if Ca is not None:
	        na_dims = list(set(range(self.mx0.size)).difference(self.angle_idims))
                Sx_xa = theano.tensor.concatenate([Sx[:,na_dims],Sx.dot(Ca)],axis=1)
                Sxu_ =  theano.tensor.concatenate([Sx_xa,Sx_xa.dot(Cu)],axis=1)
	    else:
                Sxu_ = Sxu

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

        # define input variables
        utils.print_with_stamp('Computing symbolic expression graph for belief state propagation',self.name)
        mx = theano.tensor.vector('mx')
        Sx = theano.tensor.matrix('Sx')
        H = theano.tensor.iscalar('H')
        gamma = theano.tensor.scalar('gamma')

        # this will generate the list of value and state distributions for each time step
        def get_discounted_reward(mv,Sv,mx,Sx,gamma,*args):
            mv_next, Sv_next, mx_next, Sx_next = rollout_single_step(mx,Sx)
            return gamma*mv_next,(gamma**2)*Sv_next, mx_next, Sx_next, gamma*gamma
        
        # this are the initial values for the value and its variance
        mv0, Sv0 = self.cost_symbolic(mx,Sx)

        # these are the shared variables that will be used in the graph, we need to let theano know about these
        # to reduce compilation times
        shared_vars = []
        shared_vars.extend(self.dynamics_model.get_params(symbolic=True))
        shared_vars.extend(self.policy.get_params(symbolic=True))
        (mV_,SV_,mx_,Sx_,gamma_), updts = theano.scan(fn=get_discounted_reward, 
                                                      outputs_info=[mv0,Sv0,mx,Sx,gamma], 
                                                      non_sequences=shared_vars,
                                                      n_steps=H, 
                                                      strict=True,
                                                      allow_gc=False)
        mV_ = theano.tensor.concatenate([mv0[None], mV_])
        SV_ = theano.tensor.concatenate([Sv0[None], SV_])
        mx_ = theano.tensor.concatenate([mx[None,:], mx_])
        Sx_ = theano.tensor.concatenate([Sx[None,:,:], Sx_])

        if derivs :
            utils.print_with_stamp('Computing symbolic expression for policy gradients',self.name)
            dretvars = [mV_.sum()]
            params = self.policy.get_params(symbolic=True)
            if not isinstance(params,list):
                params = [params]
            for p in params:
                dretvars.append( theano.tensor.grad(mV_.sum(), p ) ) # we are only interested in the derivative of the sum of expected values
            
            utils.print_with_stamp('Compiling belief state propagation',self.name)
            self.rollout = theano.function([mx,Sx,H,gamma], (mV_,SV_,mx_,Sx_), allow_input_downcast=True, updates=updts)
            utils.print_with_stamp('Compiling policy gradients',self.name)
            self.policy_gradient = theano.function([mx,Sx,H,gamma], dretvars, allow_input_downcast=True, updates=updts)#,mode=NanGuardMode(nan_is_error=True,inf_is_error=True,big_is_error=False))
        else:
            utils.print_with_stamp('Compiling belief state propagation',self.name)
            self.rollout = theano.function([mx,Sx,H,gamma], (mV_,SV_,mx_,Sx_), allow_input_downcast=True, updates=updts)
        utils.print_with_stamp('Done compiling.',self.name)
        #self.save_rollout()

    def train_dynamics(self):
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

        utils.print_with_stamp('Dataset size:: Inputs: [ %s ], Targets: [ %s ]  '%(self.dynamics_model.X_.shape,self.dynamics_model.Y_.shape),self.name)
        if self.dynamics_model.should_recompile:
            # reinitialize log likelihood
            self.dynamics_model.init_log_likelihood()
            # reinitialize rollot and policy gradients
            self.init_rollout(derivs=True)
 
        self.dynamics_model.train()
        utils.print_with_stamp('Done training dynamics model',self.name)

    def value(self, derivs=False):
        # we will perform a rollout with a horizon that is as long as the longest run, but at most self.H
        max_steps = max([len(episode_states) for episode_states in self.experience.states])
        H_steps = np.ceil(self.H/self.plant.dt)
        if max_steps > 1:
            H_steps = min(2*max_steps, H_steps)

        # if we have no data to compute the value, return dummy values
        if self.dynamics_model.N < 1:
            return np.zeros((H_steps,)),np.ones((H_steps,))

        # compile the belef state propagation
        if (self.rollout is None and not derivs) or (self.policy_gradient is None and derivs):
            # try loading the compiled rollout from disk
            self.init_rollout(derivs=derivs)


        # setp initial state
        mx = np.array(self.plant.x0).squeeze()
        Sx = np.array(self.plant.S0).squeeze()

        if False: # TODO make this a debug option
            # plot results
            plt.figure('Current policy rollout')
	    plt.gca().clear()
            ret = self.rollout(mx,Sx,H_steps,self.discount)
	    plt.errorbar(np.arange(0,self.H,self.plant.dt),ret[0],yerr=2*np.sqrt(ret[1]))
	    for i in xrange(len(self.experience.states)):
		cost = np.array(self.experience.immediate_cost[i])[:-1,0]
		plt.plot(np.arange(0,self.H,self.plant.dt),cost)
	    plt.show(False)
	    plt.pause(0.05)
        
        if not derivs:
            # self.H is the number of steps to rollout ( finite horizon )
            ret = self.rollout(mx,Sx,H_steps,self.discount)
            return ret[0].sum()
        else:
            ret = self.policy_gradient(mx,Sx,H_steps,self.discount)
            # first return argument is the value of the policy, second are the gradients wrt the policy params
            return [ret[0], ret[1:]]

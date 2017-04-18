import numpy as np
import os
import sys
import theano
import theano.tensor as tt
import time
import kusanagi.ghost.regression as kreg

from theano.tensor.nlinalg import matrix_inverse
from theano.tensor.slinalg import solve
from theano.misc.pkl_utils import dump as t_dump, load as t_load
from theano.compile.nanguardmode import NanGuardMode
from kusanagi import utils
from kusanagi.ghost.learners.EpisodicLearner import *

class PILCO(EpisodicLearner):
    def __init__(self, params, plant_class, policy_class, cost_func=None, viz_class=None, dynmodel_class=kreg.GP_UI, experience = None, async_plant=False, name='PILCO', filename_prefix=None):
        self.dynamics_model = None
        self.wrap_angles = params['wrap_angles'] if 'wrap_angles' in params else False
        self.use_empirical_x0 = params['use_empirical_x0'] if 'use_empirical_x0' in params else False
        self.rollout_fn=None
        self.policy_gradient_fn=None
        self.angle_idims = params['angle_dims']
        self.maxU = params['policy']['maxU']
        x0 = np.array(params['x0'],dtype=theano.config.floatX).squeeze()
        S0 = np.array(params['S0'],dtype=theano.config.floatX).squeeze()
        self.next_episode = 0
        self.dynmodel_class = dynmodel_class
        dyn_idims, dyn_odims = len(x0) + len(self.angle_idims) + len(self.maxU), len(x0)
        self.dynmodel_params = params['dynmodel'] if 'dynmodel' in params else {}
        self.dynmodel_params['idims'] = dyn_idims
        self.dynmodel_params['odims'] = dyn_odims

        filename_prefix = '%s_%s_%d_%d'%(name, dynmodel_class.__name__, dyn_idims, dyn_odims) if filename_prefix is None else filename_prefix
        super(PILCO, self).__init__(params, plant_class, policy_class, cost_func,viz_class, experience, async_plant, name, filename_prefix)
        
        # create shared variables for rollout input parameters
        self.mx0 = theano.shared(x0.astype(theano.config.floatX))
        self.Sx0 = theano.shared(S0.astype(theano.config.floatX))
        H_steps = int(np.ceil(self.H/self.plant.dt))
        self.H_steps =theano.shared( int(H_steps) )
        self.gamma0 = theano.shared( np.array(self.discount,dtype=theano.config.floatX) )

        self.register(['wrap_angles','next_episode','mx0','Sx0'])
    
    def save(self, output_folder=None,output_filename=None, save_compiled_fns=False):
        ''' Saves the state of the learner, including the parameters of the policy and the dynamics model'''
        super(PILCO,self).save(output_folder,output_filename)
        
        if hasattr(self,'dynamics_model') and self.dynamics_model is not None:
            dynamics_filename = None
            if output_filename is not None:
                dynamics_filename = output_filename + "_dynamics"
            self.dynamics_model.save(output_folder,dynamics_filename)

        if save_compiled_fns:
            self.save_compiled_fns(output_folder,output_filename)

    def load(self, output_folder=None,output_filename=None, load_compiled_fns=False):
        ''' Loads the state of the learner, including the parameters of the policy and the dynamics model, and the compiled rollout functions'''
        super(PILCO,self).load(output_folder,output_filename)
        
        output_filename = self.filename if output_filename is None else output_filename
        dynamics_filename = output_filename + "_dynamics"
        if not hasattr(self,'dynamics_model') or self.dynamics_model is None:
            self.dynamics_model = self.dynmodel_class(filename=dynamics_filename,**self.dynmodel_params)
        else:
            self.dynamics_model.load(output_folder,dynamics_filename)

        if load_compiled_fns:
            self.load_compiled_fns(output_folder,output_filename)
    
    def get_snapshot_content_paths(self, output_folder=None):
        content_paths = super(PILCO,self).get_snapshot_content_paths(output_folder)
        output_folder = utils.get_output_dir() if output_folder is None else output_folder
        content_paths.append( os.path.join(output_folder,self.dynamics_model.filename+'.zip') )
        return content_paths

    def save_compiled_fns(self, output_folder=None,output_filename=None):
        ''' Saves the compiled rollout and policy_gradient functions, along with the associated shared variables from the dynamics model and policy. The shared variables from the dynamics model adn the policy will be replaced with whatever is loaded, to ensure that the compiled rollout and policy_gradient functions are consistently updated, when the parameters of the dynamics_model and policy objects are changed. Since we won't store the latest state of these shared variables here, we will copy the values of the policy and dynamics_model parameters into the state of the shared variables. If the policy and dynamics_model parameters have been updated, we will need to load them before calling this function.'''
        sys.setrecursionlimit(50000)
        output_folder = utils.get_output_dir() if output_folder is None else output_folder
        [output_filename, self.filename] = utils.sync_output_filename(output_filename, self.filename, '_rollout.zip')
        path = os.path.join(output_folder,output_filename)
        # append the zip extension
        if not path.endswith('_rollout.zip'):
            path = path+'_rollout.zip'
        with open(path,'wb') as f:
            utils.print_with_stamp('Saving compiled functions to %s'%(path),self.name)
            # this saves the shared variables used by the rollout and policy gradients. This means that the zip file
            # will store the state of those variables from when this function is called
            t_vars = [self.dynamics_model.get_state(), self.policy.get_state(), self.rollout_fn, self.policy_gradient_fn]
            t_dump(t_vars,f,2)
            utils.print_with_stamp('Done saving.',self.name)

    def load_compiled_fns(self, output_folder=None,output_filename=None):
        ''' Loads the compiled rollout and policy_gradient functions, along with the associated shared variables from the dynamics model and policy. The shared variables from the dynamics model adn the policy will be replaced with whatever is loaded, to ensure that the compiled rollout and policy_gradient functions are consistently updated, when the parameters of the dynamics_model and policy objects are changed. Since we won't store the latest state of these shared variables here, we will copy the values of the policy and dynamics_model parameters into the state of the shared variables. If the policy and dynamics_model parameters have been updated, we will need to load them before calling this function.'''
        sys.setrecursionlimit(50000)
        output_folder = utils.get_output_dir() if output_folder is None else output_folder
        [output_filename, self.filename] = utils.sync_output_filename(output_filename, self.filename, '_rollout.zip')
        path = os.path.join(output_folder,output_filename)
        # append the zip extension
        if not path.endswith('_rollout.zip'):
            path = path+'_rollout.zip'
        with open(path,'rb') as f:
            utils.print_with_stamp('Loading compiled functions from %s'%(path),self.name)
            t_vars = t_load(f)
            # here we are loading state variables that are probably outdated, but that are tied to the compiled rollout and policy_gradient functions
            # we need to restore whatever value the shared variables hold in the latest version
            state = self.dynamics_model.get_state()
            for key in state:
                if key in t_vars[0] and isinstance(state[key],tt.sharedvar.SharedVariable):
                    t_vars[0][key].set_value(state[key].get_value())
            self.dynamics_model.set_state(t_vars[0])

            state = self.policy.get_state()
            for key in state:
                if key in t_vars[1] and isinstance(state[key],tt.sharedvar.SharedVariable):
                    t_vars[1][key].set_value(state[key].get_value())
            self.policy.set_state(t_vars[1])
            
            # At this point the dynamics model and policy state variables should be tied to the rollout and policy_graddient function, and contain the up to date values of the
            # parameters
            self.rollout_fn = t_vars[2]
            self.policy_gradient_fn = t_vars[3]
            utils.print_with_stamp('Done loading.',self.name)

    # define the function for a single propagation step
    def propagate_state(self,mx,Sx,dynmodel=None,policy=None,cost=None):
        ''' Given the input variables mx (tt.vector) and Sx (tt.matrix),
            representing the mean and variance of the system's state x, this function returns
            the next state distribution, and the mean and variance of the immediate cost. This
            is done by 1) evaluating the current policy 2) using the dynamics model to estimate 
            the next state. The immediate cost is returned as a distribution Normal(mcost,Scost),
            since the state is uncertain.
        '''
        if dynmodel is None:
            dynmodel = self.dynamics_model
        if policy is None:
            policy= self.policy
        if cost is None:
            cost= self.cost

        D = mx.shape[0]
        D_ = self.mx0.get_value().size

        # convert angles from input distribution to their complex representation
        mxa,Sxa,Ca = utils.gTrig2(mx,Sx,self.angle_idims,D_)

        # compute distribution of control signal
        sn2 = tt.exp(2*dynmodel.logsn)
        Sx_ = Sx + tt.diag(0.5*sn2)# noisy state measurement
        mxa_,Sxa_,Ca_ = utils.gTrig2(mx,Sx_,self.angle_idims,D_)
        mu, Su, Cu = policy.evaluate(mxa_, Sxa_, symbolic=True)
        
        # compute state control joint distribution
        n = Sxa.shape[0]; U = Su.shape[1]
        mxu = tt.concatenate([mxa,mu])
        q = Sxa.dot(Cu)
        Sxu_up = tt.concatenate([Sxa,q],axis=1)
        Sxu_lo = tt.concatenate([q.T,Su],axis=1)
        Sxu = tt.concatenate([Sxu_up,Sxu_lo],axis=0) # [D+U]x[D+U]

        #  predict the change in state given current state-action
        # C_deltax = inv (Sxu) dot Sxu_deltax
        m_deltax, S_deltax, C_deltax = dynmodel.predict_symbolic(mxu,Sxu)

        # compute the successor state distribution
        mx_next = mx + m_deltax

        # SSGP and BNN return C_delta as the input-output covariance. All the others do it as (input covariance)^-1 dot (input-output covariance)
        if isinstance(dynmodel,kreg.SSGP) or isinstance(dynmodel, kreg.BNN):
            Sxu_deltax = C_deltax
        else:
            Sxu_deltax = Sxu.dot(C_deltax)

        if Ca is not None:
            Da = Sxa.shape[1]; Dna = D-len(self.angle_idims)
            non_angle_dims = list(set(range(D_)).difference(self.angle_idims))
            Sxa_deltax = Sxu_deltax[:Da]        # this contains the covariance between the previous state (with angles as [sin,cos]), and the next state (with angles in radians)
            sxna_deltax = Sxa_deltax[:Dna]      # first come the non angle dimensions  [D-len(angi)] x [D] 
            sxsc_deltax = Sxa_deltax[Dna:]      # then angles as [sin,cos]             [2*len(angi)] x [D]
            #here we undo the [sin,cos] parametrization for the angle dimensions
            Sx_sc = Sx.dot(Ca)[self.angle_idims]                                     # [len(angi)] x [2*len(angi)] 
            Sa = Sxa[Dna:,Dna:]#+1e-12*tt.eye(2*len(self.angle_idims))    # [2*len(angi)] x [2*len(angi)]
            sxa_deltax = Sx_sc.dot(solve(Sa,sxsc_deltax))  # [len(angi] x [D]
            # now we create Sx_deltax and fill it with the appropriate values (i.e. in the correct order)
            Sx_deltax = tt.zeros((D,D))
            Sx_deltax = tt.set_subtensor(Sx_deltax[non_angle_dims,:], sxna_deltax)
            Sx_deltax = tt.set_subtensor(Sx_deltax[self.angle_idims,:], sxa_deltax)
        else:
            Sx_deltax = Sxu_deltax[:D]

        Sx_next = Sx + S_deltax + Sx_deltax + Sx_deltax.T

        #  get cost:
        mcost, Scost = cost(mx_next,Sx_next)

        # check if dynamics model has an updates dictionary
        updates = theano.updates.OrderedUpdates()
        
        return [mcost,Scost,mx_next,Sx_next], updates

    def get_rollout(self, mx0, Sx0, H, gamma0, dynmodel=None, policy=None):
        ''' Given some initial state distribution Normal(mx0,Sx0), and a prediction horizon H 
        (number of timesteps), returns the predicted state distribution and discounted cost for
        every timestep. The discounted cost is returned as a distribution, since the state
        is uncertain.'''
        utils.print_with_stamp('Computing symbolic expression graph for belief state propagation',self.name)
        if dynmodel is None:
            dynmodel = self.dynamics_model
        if policy is None:
            policy= self.policy

        # this defines the loop where the state is propaagated
        def rollout_single_step(mv,Sv,mx,Sx,gamma,*args):
            [mv_next, Sv_next, mx_next, Sx_next], updates = self.propagate_state(mx,Sx,dynmodel,policy)
            gamma0 = args[0]
            return [gamma*mv_next,tt.square(gamma)*Sv_next, mx_next, Sx_next, gamma*gamma0], updates
        
        # this are the initial distribution of the cost
        mv0, Sv0 = self.cost(mx0,Sx0)

        # these are the shared variables that will be used in the graph.
        # we need to pass them as non_sequences here (see: http://deeplearning.net/software/theano/library/scan.html)
        shared_vars = [gamma0]
        shared_vars.extend(dynmodel.get_all_shared_vars())
        shared_vars.extend(policy.get_all_shared_vars())

        # create the nodes that return the result from scan
        rollout_output, updts = theano.scan(fn=rollout_single_step, 
                                            outputs_info=[mv0,Sv0,mx0,Sx0,gamma0], 
                                            non_sequences=shared_vars,
                                            n_steps=H, 
                                            strict=True,
                                            allow_gc=False,
                                            name="%s>rollout_scan"%(self.name))

        mean_costs,var_costs,mean_states,cov_states,gamma_ = rollout_output
        
        # prepend the initial cost distribution
        mean_costs = tt.concatenate([mv0.dimshuffle('x'), mean_costs])
        var_costs = tt.concatenate([Sv0.dimshuffle('x'), var_costs])
        # prepend the initial state distribution
        mean_states = tt.concatenate([mx0.dimshuffle('x',0), mean_states])
        cov_states = tt.concatenate([Sx0.dimshuffle('x',0,1), cov_states])

        mean_costs.name = 'mc_list'
        var_costs.name = 'Sc_list'
        mean_states.name = 'mx_list'
        cov_states.name = 'Sx_list'

        return [mean_costs, var_costs, mean_states, cov_states], updts

    def get_policy_value(self, mx0 = None, Sx0 = None, H = None, gamma0 = None, dynmodel=None, policy=None):
        ''' Returns a symbolic expression (theano tensor variable) for the value of the current policy '''
        mx0 = self.mx0 if mx0 is None else mx0
        Sx0 = self.Sx0 if Sx0 is None else Sx0
        H = self.H_steps if H is None else H
        gamma0 = self.gamma0 if gamma0 is None else gamma0

        [mc_,Sc_,mx_,Sx_], updts = self.get_rollout(mx0,Sx0,H,gamma0,dynmodel,policy)
        return mc_[1:].sum(), updts

    def get_policy_gradients(self, expected_accumulated_cost, params,clip=None):
        ''' Creates the variables representing the policy gradients (theano tensor variables) of
        the provided expected_accumulated_cost, with respect to the policy parameters'''

        utils.print_with_stamp('Computing symbolic expression for policy gradients',self.name)
        
        dJdp = tt.grad(expected_accumulated_cost, params )
        if clip is not None:
            import lasagne
            dJdp = lasagne.updates.total_norm_constraint(dJdp,clip)
        return dJdp

    def compile_rollout(self, mx0 = None, Sx0 = None, H = None, gamma0 = None, dynmodel=None, policy=None):
        ''' Compiles a theano function graph that compute the predicted states and discounted costs for a given
        inital state and prediction horizon.'''
        mx0 = self.mx0 if mx0 is None else mx0
        Sx0 = self.Sx0 if Sx0 is None else Sx0
        H = self.H_steps if H is None else H
        gamma0 = self.gamma0 if gamma0 is None else gamma0

        [mc_,Sc_,mx_,Sx_], updts = self.get_rollout(mx0,Sx0,H,gamma0, dynmodel, policy)

        utils.print_with_stamp('Compiling belief state propagation',self.name)
        rollout_fn = theano.function([], 
                                     [mc_,Sc_,mx_,Sx_], 
                                     allow_input_downcast=True, 
                                     updates=updts,
                                     name='%s>rollout_fn'%(self.name))
        utils.print_with_stamp("Done compiling.",self.name)

        return rollout_fn

    def compile_policy_gradients(self, mx0 = None, Sx0 = None, H = None, gamma0 = None, dynmodel=None, policy=None):
        ''' Compiles a theano function graph that computes the gradients of the expected accumulated a given
        initial state and prediction horizon. The function will return the value of the policy, followed by 
        the policy gradients.'''
        mx0 = self.mx0 if mx0 is None else mx0
        Sx0 = self.Sx0 if Sx0 is None else Sx0
        H = self.H_steps if H is None else H
        gamma0 = self.gamma0 if gamma0 is None else gamma0
        
        [mc_,Sc_,mx_,Sx_], updts = self.get_rollout(mx0,Sx0,H,gamma0, dynmodel, policy)
        expected_accumulated_cost = mc_[1:].sum()
        expected_accumulated_cost.name = 'J'

        # get the policy parameters (as theano tensor variables)
        if policy is None:
            policy= self.policy
        p = policy.get_params(symbolic=True)
        # the policy gradients will have the same shape as the policy parameters (i.e. this returns a list
        # of theano tensor variables with the same dtype and ndim as the parameters in p )
        dJdp = self.get_policy_gradients(expected_accumulated_cost,p)
        retvars = [expected_accumulated_cost]
        retvars.extend(dJdp)

        utils.print_with_stamp('Compiling policy gradients',self.name)
        policy_gradient_fn = theano.function([],
                                             retvars, 
                                             allow_input_downcast=True, 
                                             updates=updts,
                                             name="%s>policy_gradient_fn"%(self.name))
        utils.print_with_stamp("Done compiling.",self.name)
        return policy_gradient_fn

    def rollout(self, mx0, Sx0, H_steps, discount):
        ''' Function that returns trajectory distributions from the dynamics model. Ensures the compiled rollout function is initialised before we can call it'''
        if self.rollout_fn is None:
            self.rollout_fn = self.compile_rollout()

        # update shared vars
        self.mx0.set_value(mx0.astype(theano.config.floatX))
        self.Sx0.set_value(Sx0.astype(theano.config.floatX))
        self.H_steps.set_value(int(H_steps))
        self.gamma0.set_value( np.array(discount,dtype=theano.config.floatX) )

        # call theano function
        return self.rollout_fn()

    def policy_gradient(self, mx0, Sx0, H_steps, discount):
        ''' Function that computes the gradients of the parameters of the current policy. Ensures the compiled policy gradients function is initialised before we can call it'''
        if self.policy_gradient_fn is None:
            self.policy_gradient_fn = self.compile_policy_gradients()
            self.start_time = time.time()

        # update shared vars
        self.mx0.set_value(mx0)
        self.Sx0.set_value(Sx0)
        self.H_steps.set_value(int(H_steps))
        self.gamma0.set_value( np.array(discount,dtype=theano.config.floatX) )
        # call theano function
        return self.policy_gradient_fn()

    def train_dynamics(self,dynmodel=None, dynmodel_class=None, dynmodel_params=None, max_episodes=None):
        ''' Trains a dynamics model using the current experience dataset '''
        utils.print_with_stamp('Training dynamics model',self.name)
        if dynmodel is None:
            if hasattr(self,'dynamics_model'):
                dynmodel = self.dynamics_model

        X = []
        Y = []
        n_episodes = len(self.experience.states)
        
        # get dataset for dynamics model
        episodes = range(self.next_episode,n_episodes) if max_episodes is None or n_episodes < max_episodes else range(max(0,n_episodes-max_episodes),n_episodes)
        self.next_episode = n_episodes 
        X,Y = self.experience.get_dynmodel_dataset(filter_episodes=episodes, angle_dims=self.angle_idims)
        
        # wrap angles if requested (this might introduce error if the angular velocities are high )
        if self.wrap_angles:
            # wrap angle differences to [-pi,pi]
            Y[:,self.angle_idims] = (Y[:,self.angle_idims] + np.pi) % (2 * np.pi ) - np.pi

        # get distribution of initial states
        x0 = np.array([x[0] for x in self.experience.states])
        if self.use_empirical_x0:
            if n_episodes > 2:
                self.mx0.set_value(x0.mean(0).astype(theano.config.floatX))
                self.Sx0.set_value(np.cov(x0.T).astype(theano.config.floatX))
        print x0
        print self.Sx0.get_value()

        if dynmodel is None:
            if dynmodel_class is None: dynmodel_class = self.dynmodel_class
            if dynmodel_params is None: dynmodel_params = self.dynmodel_params
            # initialize dynamics model
            dynamics_filename = self.filename+'_dynamics'
            dynmodel = dynmodel_class(filename=dynamics_filename,**dynmodel_params)
            dynmodel.set_dataset(X,Y)
        else:
            if max_episodes is not None and len(episodes) ==  max_episodes:
                dynmodel.set_dataset(X,Y)
            else:
                # append data to the dynamics model
                dynmodel.append_dataset(X,Y)
        
        #utils.print_with_stamp('%s, \n%s'%(self.mx0.get_value(), self.Sx0.get_value()),self.name)
        utils.print_with_stamp('Dataset size:: Inputs: [ %s ], Targets: [ %s ]  '%(dynmodel.X.get_value(borrow=True).shape,dynmodel.Y.get_value(borrow=True).shape),self.name)
        if dynmodel.should_recompile:
            # reinitialize log likelihood
            dynmodel.init_loss()
            self.should_recompile = True
 
        dynmodel.train()
        utils.print_with_stamp('Done training dynamics model',self.name)
        if self.dynamics_model is None:
            self.dynamics_model = dynmodel
        return dynmodel

    def value(self,return_grads=False):
        '''Returns the value of the current policy by computing long term predictions using a learned dynamics model. If return_grads is True, it will also return the policy gradients'''

        if hasattr(self,'update'):
            # if there are any variable we want to update before computing the value
            self.update()

        # we will perform a rollout with a horizon that is as long as the longest run, but at most self.H
        max_steps = max([len(episode_states) for episode_states in self.experience.states])
        H_steps = int(np.ceil(self.H/self.plant.dt))
        if max_steps > 1:
            H_steps = min(2*max_steps, H_steps)

        # if we have no data to compute the value, return dummy values
        #if self.dynamics_model.N < 1:
        #    return np.zeros((H_steps,)),np.ones((H_steps,))

        # setup initial state
        if self.use_empirical_x0:
            mx = self.mx0.get_value()
            Sx = self.Sx0.get_value()
        else:
            mx = np.array(self.plant.x0, dtype=theano.config.floatX).squeeze()
            Sx = np.array(self.plant.S0, dtype=theano.config.floatX).squeeze()
    
        if return_grads:
            pg = self.policy_gradient(mx,Sx,H_steps,self.discount)
            # first return argument is the value of the policy, second are the gradients wrt the policy params
            ret = [pg[0]]
            ret.append(pg[1:])
            return ret
        else:
            ret = self.rollout(mx,Sx,H_steps,self.discount)
            #self.rollout(mx,Sx,H_steps,self.discount,use_scan=False)
            #self.policy_gradient(mx,Sx,H_steps,self.discount,use_scan=False)
            
            # don't include timestep 0
            return ret[0][1:].sum()

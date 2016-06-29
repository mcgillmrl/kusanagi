import numpy as np
import os
import sys
import theano
import time
import utils

from theano.misc.pkl_utils import dump as t_dump, load as t_load

from scipy.optimize import minimize, basinhopping
from matplotlib import pyplot as plt
from functools import partial

from ghost.learners.ExperienceDataset import ExperienceDataset
from ghost.control import RandPolicy

class EpisodicLearner(object):
    def __init__(self, params, plant_class, policy_class, cost_func=None, viz_class=None, experience = None, async_plant=False, name='EpisodicLearner', filename_prefix=None):
        self.name = name
        # initialize plant
        params['plant']['x0'] = params['x0']
        params['plant']['S0'] = params['S0']
        self.plant = plant_class(**params['plant'])
        # initialize policy
        params['policy']['angle_dims'] = params['angle_dims']
        self.policy = policy_class(**params['policy'])
        # set filename    
        self.filename = self.name+'_'+self.plant.name+'_'+self.policy.name if filename_prefix is None else filename_prefix+'_'+self.plant.name+'_'+self.policy.name
        # initialize cost
        params['cost']['angle_dims'] = params['angle_dims']
        self.cost = partial(cost_func, params=params['cost']) if cost_func is not None else None
        # initialize vizualization
        self.viz = viz_class(self.plant) if viz_class is not None else None
        # initialize experience dataset
        self.experience = ExperienceDataset(filename_prefix=self.filename) if experience is None else experience
        
        # initialize learner state variables
        self.min_method = "L-BFGS-B"
        self.n_episodes = 0
        self.angle_idims = params['angle_dims'] if 'angle_dims' in params else []
        self.H = params['H'] if 'H' in params else 10.0
        self.discount = params['discount'] if 'discout' in params else 1
        self.max_evals = params['max_evals'] if 'max_evals' in params else 150
        self.async_plant = async_plant
        self.learning_iteration = 0;
        self.n_evals = 0

        # try loading from file, initialize from scratch otherwise
        try:
            self.load()   
        except IOError:
            utils.print_with_stamp('Initialising new %s learner [ Could not open %s_state.zip ]'%(self.name, self.filename),self.name)
            if self.cost is not None:
                self.init_cost(self.cost)
        self.state_changed = False

    def load(self):
        # load policy and experience separately
        self.policy.load()
        self.experience.load()
        
        # load learner state
        path = os.path.join(utils.get_run_output_dir(),self.filename+'.zip')
        with open(path,'rb') as f:
            utils.print_with_stamp('Loading learner state from %s.zip'%(self.filename),self.name)
            state = t_load(f)
            self.set_state(state)
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
                t_dump(self.get_state(),f,2)
            self.state_changed = False

    def stop(self):
        ''' Stops the plant, the visualization (if available) and saves the state of the learner'''
        self.plant.stop()
        if self.viz is not None:
            self.viz.stop()
        #self.save()

    def set_state(self,state):
        i = utils.integer_generator()
        self.n_episodes = state[i.next()]
        self.angle_idims = state[i.next()]
        self.async_plant = state[i.next()]
        self.cost = state[i.next()]
        self.cost_symbolic = state[i.next()]
        self.H = state[i.next()]
        self.discount = state[i.next()]
        self.learning_iteration = state[i.next()]
        self.n_evals = state[i.next()]

    def get_state(self):
        return [self.n_episodes,self.angle_idims,self.async_plant,self.cost,self.cost_symbolic,self.H,self.discount,self.learning_iteration,self.n_evals]

    def init_cost(self,cost):
        self.cost_symbolic = cost
        utils.print_with_stamp('Compiling cost function',self.name)
        mx = theano.tensor.vector('mx')
        Sx = theano.tensor.matrix('Sx')
        self.cost = theano.function((mx,Sx),self.cost_symbolic(mx,Sx), allow_input_downcast=True)

    def apply_controller(self,H=None,random_controls=False):
        '''
        Starts the plant and applies the current policy to the plant for a duration specified by H (in seconds). If  H is not set, it will run for self.H seconds. If the random_controls paramter is set to True, the current policy is ignored and random controls between [-self.policy.maxU, self.policy.maxU ] will be sent 
        to the plant

        @param H Horizon for applying controller (in seconds)
        @param random_controls Boolean flag that specifies whether to use the current policy or apply random controls
        '''
        utils.print_with_stamp('Starting data collection run',self.name)
        if H is  None:
            H = self.H
        utils.print_with_stamp('Running for %f seconds'%(H),self.name)

        if random_controls:
            policy = RandPolicy(self.policy.maxU)
        else:
            policy = self.policy

        # mark the start of the episode
        self.experience.new_episode()

        # start robot
        if self.async_plant:
            self.plant.start()
        if self.viz is not None:
            self.viz.start()

        exec_time = time.time()
        x_t, t0 = self.plant.get_state()
        Sx_t = np.zeros((x_t.shape[0],x_t.shape[0]))
        L_noise = np.linalg.cholesky(self.plant.noise)
        if self.plant.noise is not None:
            # randomize state
            x_t = x_t + np.random.randn(x_t.shape[0]).dot(L_noise);
        t = t0
        
        H_steps = int(np.ceil(H/self.plant.dt))
        # do rollout
        #while t <= t0 + H:
        for i in xrange(H_steps):
            # convert input angle dimensions to complex representation
            x_t_ = utils.gTrig_np(x_t[None,:], self.angle_idims).flatten()
            #  get command from policy (this should be fast, or at least account for delays in processing):
            u_t = policy.evaluate(x_t_)[0].flatten()
            #  send command to robot:
            self.plant.apply_control(u_t)
            if self.cost is not None:
                #  get cost:
                c_t = self.cost(x_t, Sx_t)
                # append to experience dataset
                self.experience.add_sample(t,x_t,u_t,c_t)
                # print t,x_t,u_t,c_t[0]
                print u_t
            else:
                # append to experience dataset
                self.experience.append(t,x_t,u_t,0)

            # step the plant if necessary
            if not self.async_plant:
                self.plant.step()

            # sleep to match the desired sample rate
            exec_time = time.time() - exec_time
            if exec_time < self.plant.dt:
                time.sleep(self.plant.dt-exec_time)

            #  get robot state (this should ensure synchronicity by blocking until dt seconds have passed):
            exec_time = time.time()
            x_t, t = self.plant.get_state()
            if self.plant.noise is not None:
                # randomize state
                x_t = x_t + np.random.randn(x_t.shape[0]).dot(L_noise);
            
            if self.plant.done:
                break
        # add last state to experience
        x_t_ = utils.gTrig_np(x_t[None,:], self.angle_idims).flatten()
        u_t = np.zeros_like(u_t)
        if self.cost is not None:
            c_t = self.cost(x_t, Sx_t)
            self.experience.add_sample(t,x_t,u_t,c_t)
            #print t,x_t,u_t,c_t[0]
        else:
            self.experience.append(t,x_t,u_t,0)

        # stop robot
        run_value = np.array(self.experience.immediate_cost[-1][:-1])
        utils.print_with_stamp('Done. Stopping robot. Value of run [%f]'%(run_value.sum()),self.name)

        self.plant.stop()
        if self.viz is not None:
            self.viz.stop()
        self.n_episodes += 1
        return self.experience

    def train_policy(self, H=None):
        if H is not None:
            self.H = H
        # optimize value wrt to the policy parameters
        self.learning_iteration+=1
        self.n_evals=0
        utils.print_with_stamp('Training policy parameters [Iteration %d]'%(self.learning_iteration), self.name)# Starting value [%f]'%(self.value(derivs=False)),self.name)
        utils.print_with_stamp('Initial value estimate [%f]'%(self.value(derivs=True))[0],self.name) 
        p0 = self.policy.get_params(symbolic=False)
        parameter_shapes = [p.shape for p in p0]
        m_loss = utils.MemoizeJac(self.loss)
        try:
            opt_res = minimize(m_loss, utils.wrap_params(p0), jac=m_loss.derivative, args=parameter_shapes, method=self.min_method, tol=1e-12, options={'maxiter': self.max_evals})
        except ValueError:
            print '' 
            print self.policy.get_params(symbolic=False)
            print self.policy.K[0].eval()
            raise
            #utils.print_with_stamp('%s failed after %d evaluations. Switching to CG'%(self.min_method,self.n_evals),self.name)
            #opt_res = minimize(m_loss, utils.wrap_params(p0), jac=m_loss.derivative, args=parameter_shapes, method='CG', tol=1e-12, options={'maxiter': 125})

        self.policy.set_params(utils.unwrap_params(opt_res.x,parameter_shapes))
        print '' 
        #self.policy_gradients.profile.print_summary()
        utils.print_with_stamp('Done training. New value [%f]'%(self.value(derivs=False)),self.name)
        self.state_changed = True

    def loss(self, policy_parameters, parameter_shapes):
        # set policy parameters
        self.policy.set_params( utils.unwrap_params(policy_parameters, parameter_shapes) )

        # compute value + derivatives
        v,dv = self.value(derivs=True)
        dv = utils.wrap_params(dv)
        v,dv = (np.array(v).astype(np.float64),np.array(dv).astype(np.float64))
        self.n_evals+=1
        utils.print_with_stamp('Current value: %s, Total evaluations: %d    '%(str(v),self.n_evals),self.name,True)
        return v,dv


import numpy as np
import theano

from scipy.optimize import minimize, basinhopping
from matplotlib import pyplot as plt
from time import time, sleep

from ghost.control import RandPolicy
from utils import print_with_stamp,gTrig_np,wrap_params,unwrap_params,MemoizeJac

class ExperienceDataset(object):
    def __init__(self):
        self.time_stamps = []
        self.states = []
        self.actions = []
        self.immediate_cost = []
        self.curr_episode = -1

    def add_sample(self,t,x_t=None,u_t=None,c_t=None):
        curr_episode = self.curr_episode
        self.time_stamps[curr_episode].append(t)
        self.states[curr_episode].append(x_t)
        self.actions[curr_episode].append(u_t)
        self.immediate_cost[curr_episode].append(c_t)

    def new_episode(self):
        self.time_stamps.append([])
        self.states.append([])
        self.actions.append([])
        self.immediate_cost.append([])
        self.curr_episode += 1

class EpisodicLearner(object):
    def __init__(self, plant, policy, cost=None, angle_idims=None, discount=1, experience = None, async_plant=True, name='EpisodicLearner'):
        self.name = name
        self.min_method = "L-BFGS-B"
        self.n_episodes = 0
        self.experience = ExperienceDataset() if experience is None else experience
        self.plant = plant
        self.policy = policy
        self.angle_idims = angle_idims
        self.async_plant = async_plant
        self.cost=None
        if cost is not None:
            self.init_cost(cost)
        self.H = 10
        self.discount = 1
        self.learning_iteration = 0;
        self.n_evals = 0

    def save(self):
        self.policy.save()
        # TODO save state of the learner (e.g. learning iteration)

    def init_cost(self,cost):
        self.cost_symbolic = cost
        print_with_stamp('Compiling cost function',self.name)
        mx = theano.tensor.vector('mx')
        Sx = theano.tensor.matrix('Sx')
        self.cost = theano.function((mx,Sx),self.cost_symbolic(mx,Sx), allow_input_downcast=True)

    def apply_controller(self,H=float('inf'),random_controls=False):
        print_with_stamp('Starting data collection run',self.name)
        if H < float('inf'):
            print_with_stamp('Running for %f seconds'%(H),self.name)
            self.H = H

        if random_controls:
            policy =  RandPolicy(self.policy.maxU)
        else:
            policy = self.policy

        # mark the start of the episode
        self.experience.new_episode()

        # start robot
        if self.async_plant:
            self.plant.start()
        exec_time = time()
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
            x_t_ = gTrig_np(x_t[None,:], self.angle_idims).flatten()
            #  get command from policy (this should be fast, or at least account for delays in processing):
            u_t = policy.evaluate(t, x_t_)[0].flatten()
            #  send command to robot:
            self.plant.apply_control(u_t)
            if self.cost is not None:
                #  get cost:
                c_t = self.cost(x_t, Sx_t) 
                # append to experience dataset
                self.experience.add_sample(t,x_t,u_t,c_t)
            else:
                # append to experience dataset
                self.experience.append(t,x_t,u_t,0)

            # step the plant if necessary
            if not self.async_plant:
                self.plant.step()

            # sleep to match the desired sample rate
            exec_time = time() - exec_time
            if exec_time < self.plant.dt:
                sleep(self.plant.dt-exec_time)

            #  get robot state (this should ensure synchronicity by blocking until dt seconds have passed):
            exec_time = time()
            x_t, t = self.plant.get_state()
            if self.plant.noise is not None:
                # randomize state
                x_t = x_t + np.random.randn(x_t.shape[0]).dot(L_noise);
        
        # add last state to experience
        x_t_ = gTrig_np(x_t[None,:], self.angle_idims).flatten()
        u_t = np.zeros_like(u_t)
        if self.cost is not None:
            c_t = self.cost(x_t, Sx_t)
            self.experience.add_sample(t,x_t,u_t,c_t)
        else:
            self.experience.append(t,x_t,u_t,0)

        # stop robot
        run_value = np.array(self.experience.immediate_cost[-1][:-1])
        print_with_stamp('Done. Stopping robot. Value of run [%f]'%(run_value.sum()),self.name)

        self.plant.stop()
        self.n_episodes += 1

    def train_policy(self, H=None):
        if H is not None:
            self.H = H
        # optimize value wrt to the policy parameters
        self.learning_iteration+=1
        self.n_evals=0
        print_with_stamp('Training policy parameters [Iteration %d]'%(self.learning_iteration), self.name)# Starting value [%f]'%(self.value(derivs=False)),self.name)
        print_with_stamp('Initial value estimate [%f]'%(self.value(derivs=True))[0],self.name) 
        p0 = self.policy.get_params(symbolic=False)
        parameter_shapes = [p.shape for p in p0]
        m_loss = MemoizeJac(self.loss)
        try:
            opt_res = minimize(m_loss, wrap_params(p0), jac=m_loss.derivative, args=parameter_shapes, method=self.min_method, tol=1e-9, options={'maxiter': 100})
        except ValueError:
            print '' 
            print_with_stamp('%s failed after %d evaluations. Switching to CG'%(self.min_method,self.n_evals),self.name)
            opt_res = minimize(m_loss, wrap_params(p0), jac=m_loss.derivative, args=parameter_shapes, method='CG', tol=1e-9, options={'maxiter': 100})

        self.policy.set_params(unwrap_params(opt_res.x,parameter_shapes))
        print '' 
        #self.policy_gradients.profile.print_summary()
        print_with_stamp('Done training. New value [%f]'%(self.value(derivs=False)),self.name)
        self.save()

    def loss(self, policy_parameters, parameter_shapes):
        # set policy parameters
        self.policy.set_params( unwrap_params(policy_parameters, parameter_shapes) )

        # compute value + derivatives
        v,dv = self.value(derivs=True)
        dv = wrap_params(dv)
        v,dv = (np.array(v).astype(np.float64),np.array(dv).astype(np.float64))
        self.n_evals+=1
        print_with_stamp('Current value: %s, Total evaluations: %d    '%(str(v),self.n_evals),self.name,True)
        return v,dv


import numpy as np
from utils import print_with_stamp,gTrig_np
from time import time, sleep

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
    def __init__(self, plant, policy, cost=None, angle_idims=None, experience = None, async_plant=True, name='EpisodicLearner'):
        self.name = name
        self.min_method = "L-BFGS-B"
        self.n_episodes = 0
        self.experience = ExperienceDataset() if experience is None else experience
        self.plant = plant
        self.policy = policy
        self.cost = cost
        self.angle_idims = angle_idims
        self.async_plant = async_plant
        pass

    def apply_controller(self,H=float('inf')):
        print_with_stamp('Starting data collection run',self.name)
        if H < float('inf'):
            print_with_stamp('Running for %f seconds'%(H),self.name)

        # mark the start of the episode
        self.experience.new_episode()

        # start robot
        if self.async_plant:
            self.plant.start()
        x_t, t0 = self.plant.get_state()
        L_noise = np.linalg.cholesky(self.plant.noise)
        if self.plant.noise is not None:
            # randomize state
            x_t = x_t + np.random.randn(x_t.shape[0]).dot(L_noise);
        t = t0
        
        H_steps = int(np.ceil(H/self.plant.dt))
        # do rollout
        #while t <= t0 + H:
        for i in xrange(H_steps):
            exec_time = time()
            # convert input angle dimensions to complex representation
            x_t_ = gTrig_np(x_t[None,:], self.angle_idims).flatten()
            #  get command from policy (this should be fast, or at least account for delays in processing):
            u_t = self.policy.evaluate(t, x_t_)[0].flatten()
            #print x_t_,u_t
            #  send command to robot:
            self.plant.apply_control(u_t)
            if self.cost is not None:
                #  get cost:
                c_t = self.cost.evaluate(mx=x_t,mu=u_t) 
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
            sleep(max(self.plant.dt-exec_time,0))

            #  get robot state (this should ensure synchronicity by blocking until dt seconds have passed):
            x_t, t = self.plant.get_state()
            if self.plant.noise is not None:
                # randomize state
                x_t = x_t + np.random.randn(x_t.shape[0]).dot(L_noise);
        
        # add last state to experience
        x_t_ = gTrig_np(x_t[None,:], self.angle_idims).flatten()
        u_t = self.policy.evaluate(t, x_t_)[0].flatten()
        if self.cost is not None:
            c_t = self.cost.evaluate(mx=x_t,mu=u_t) 
            self.experience.add_sample(t,x_t,u_t,c_t)
        else:
            self.experience.append(t,x_t,u_t,0)

        # stop robot
        print_with_stamp('Done. Stopping robot.',self.name)
        self.plant.stop()
        self.n_episodes += 1

    def train_policy(self):
        # optimize value wrt to the policy parameters
        print_with_stamp('Training policy parameters',self.name)
        opt_res = minimize(self.loss, self.policy.get_params(), jac=True, method=self.min_method, tol=1e-9, options={'maxiter': 500})
        print_with_stamp('Done training',self.name)

    def loss(self, policy_parameters):
        # set policy parameters
        self.policy.update(policy_parameters)

        # compute value + derivatives
        v,dv = self.value_d()

        return v,dv

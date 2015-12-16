import numpy as np
from utils import print_with_stamp
from time import time
from ghost.learners import EpisodicLearner

class ExperienceDataset(object):
    def __init__(self):
        self.time_stamps = []
        self.states = []
        self.actions = []
        self.immediate_cost = []
        self.episode_starts = []

    def add_sample(t,x_t=None,u_t=None,c_t=None):
        self.time_stamps.append(t)
        self.states.append(x_t)
        self.actions.append(u_t)
        self.immediate_cost.append(c_t)

    def mark_start(self):
        self.episode_starts.append(len(self.states))

class EpisodicLearner(object):
    def __init__(self, plant, policy, cost=None, experience = None, name='EpisodicLearner'):
        self.name = name
        self.min_method = "L-BFGS-B"
        self.n_episodes = 0
        self.experience = ExperienceDataset() if experience is None else experience
        self.plant = plant
        self.policy = policy
        self.cost = cost
        pass

    def apply_controller(self):
        print_with_stamp('Starting data collection run',self.name)
        # intialize episode specific variables
        t = time()
        self.experience.mark_start()
        # start robot
        plant.start()
        while t < H:
            #  get robot state (this should ensure synchonicity by blocking until dt seconds have passed):
            x_t, t = plant.get_state()
            #  get command from policy (this should be fast, or at least account for delays in processing):
            u_t = policy(t, x_t)
            #  send command to robot:
            plant.apply_control(u_t)
            if cost is not None:
                #  get cost:
                c_t = cost(x_t,u_t) 
                # append to experience dataset
                self.experience.add_sample(t,x_t,u_t,c_t)
            # append to experience dataset
            self.experience.append(x_t,u_t,0)
        
        # stop robot
        print_with_stamp('Done. Stopping robot.',self.name)
        plant.stop()
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

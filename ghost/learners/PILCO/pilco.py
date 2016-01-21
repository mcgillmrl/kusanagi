import numpy as np
from utils import print_with_stamp
from ghost.learners.EpisodicLearner import *
from ghost.regression.GPRegressor import GP_UI

class PILCO(EpisodicLearner):
    def __init__(self, plant, policy, cost, angle_idims=None, experience = None, name='PILCO'):
        super(PILCO, self).__init__(plant, policy, cost, angle_idims, experience, name)
        self.dynamics_model = None

    def train_dynamics(self):
        print_with_stamp('Training dynamics model',self.name)
        x = np.array(self.experience.states)
        u = np.array(self.experience.actions)

        # inputs are states, concatenated with actions (except for the last entry)
        X = np.hstack((x[:-1],u[:-1]))
        # outputs are next states
        Y =  x[1:]
        if self.dynamics_model is None:
            # this will convert angle inputs to cartesian coordinates
            self.dynamics_model = GP_UI(X,Y, angle_idims = self.angle_idims)
        else:
            self.dynamics_model.set_dataset(X,Y)

        self.dynamics_model.train()
        print_with_stamp('Done training dynamics model',self.name)

    def value(self, H, derivs=False):
        if derivs == True:
            return self.value_d(H)
        print_with_stamp('Computing value of current policy',self.name)
	x = np.array(self.experience.states)
	u = np.array(self.experience.actions)
        # get distribution of initial states
        x0 = x[self.experience.episode_starts]
        mx = []; Sx = []
        if x0.shape[0] > 1:
            mx = x0.mean()
            Sx = np.cov(x0.T)
        else:
            mx = x0
            Sx = 1e-2*np.eye(x0.shape[1])
	
	print mx,Sx

        # simulate a rollout using the dynamics model
        dt = self.plant.dt; t = 0
        while t < H:
            # evaluate the policy at current state
            mu, Su, Cu = self.policy.evaluate(t, mx, Sx)
	    
	    print mu,Su

            # fill in the covariance of the state-action vector
            m, S = fillIn(mx,Sx,mu,Su,Vu) 

            #  predict next state given current state-action
            mx, Sx, Cx = self.dynamics_model.predict(m,S)

            #  get cost:
            mc, Sc = cost(mx,mu,Sx,Su)
	    print mc,Sc

        # return value

    def value_d(self,H):
        print_with_stamp('Computing value of current policy, with derivatives',self.name)
	x = np.array(self.experience.states)
	u = np.array(self.experience.actions)
        # get distribution of initial states
        x0 = x[self.experience.episode_starts]
        mx = []; Sx = []
        if x0.shape[0] > 1:
            mx = x0.mean()
            Sx = np.cov(x0.T)
        else:
            mx = x0
            Sx = 1e-2*np.eye(x0.shape[1])

	print mx,Sx

        # simulate a rollout using the dynamics model
        dt = self.plant.dt; t = 0
        while t < H:
            # evaluate the policy at current state
            mu, Su, Cu = self.policy.evaluate(t, mx, Sx)

            # fill in the covariance of the state-action vector
            m, S = fillIn(mx,Sx,mu,Su,Vu) 

            #  predict next state given current state-action
            mx, Sx, Cx = self.dynamics_model.predict(m,S)

            #  get cost:
            mc, Sc = cost(mx,mu,Sx,Su)
	    print mc,Sc

        # return value + derivatives

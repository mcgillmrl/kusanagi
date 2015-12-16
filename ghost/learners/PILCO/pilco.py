import numpy as np
from utils import print_with_stamp
from ghost.learners import EpisodicLearner

class PILCO(EpisodicLearner):
    def __init__(self, plant, policy, cost, experience = None, name='PILCO'):
        super(PILCO, self).__init__(plant, policy, cost, experience, name)
        self.dynamics_model = None

    def train_dynamics(self):
        x = np.array(self.experience.states)
        u = np.array(self.experience.actions)
        # inputs are states, concatenated with actions (except for the last entry)
        X = np.hstack(x[:-1],u[:-1])
        # outputs are next states
        Y =  x[1:]
        if self.dynamics is None:
            self.dynamics_model = GP_UI(X,Y)
        else:
            self.dynamics_model.set_dataset(X,Y)

        self.dynamics_model.train()

    def value(self, derivs=True):
        if derivs == True:
            return self.value_d

        # get distribution of initial states
        x0 = x[self.experience.episode_starts]
        mx = []; Sx = []
        if x0.shape[0] > 1:
            mx = x0.mean()
            Sx = np.cov(x0.T)
        else:
            mx = x0
            Sx = 1e-2*np.eye(x0.shape[1])

        # simulate a rollout using the dynamics model
        dt = self.dt; H = self.H; t = 0
        while t < H:
            # evaluate the policy at current state
            mu, Su, Cu = self.policy.evaluate(t, m0, S0)

            # fill in the covariance of the state-action vector
            m, S = fillIn(mx,Sx,mu,Su,Vu) 

            #  predict next state given current state-action
            mx, Sx, Cx = self.dynamics_model.predict(m,S)

            #  get cost:
            mc, Sc = cost(mx,mu,Sx,Su)

        # return value

    def value_d(self):
        # get distribution of initial states
        x0 = x[self.experience.episode_starts]
        mx = []; Sx = []
        if x0.shape[0] > 1:
            mx = x0.mean()
            Sx = np.cov(x0.T)
        else:
            mx = x0
            Sx = 1e-2*np.eye(x0.shape[1])

        # simulate a rollout using the dynamics model
        dt = self.dt; H = self.H; t = 0
        while t < H:
            # evaluate the policy at current state
            mu, Su, Cu = self.policy.evaluate(t, m0, S0)

            # fill in the covariance of the state-action vector
            m, S = fillIn(mx,Sx,mu,Su,Vu) 

            #  predict next state given current state-action
            mx, Sx, Cx = self.dynamics_model.predict(m,S)

            #  get cost:
            mc, Sc = cost(mx,mu,Sx,Su)

        # return value + derivatives

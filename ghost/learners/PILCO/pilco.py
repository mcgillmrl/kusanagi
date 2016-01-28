import numpy as np
from utils import print_with_stamp, get_compiled_gTrig,gTrig_np
from ghost.learners.EpisodicLearner import *
from ghost.regression.GPRegressor import GP_UI

class PILCO(EpisodicLearner):
    def __init__(self, plant, policy, cost, angle_idims=None, experience = None, name='PILCO', wrap_angles=True):
        super(PILCO, self).__init__(plant, policy, cost, angle_idims, experience, name)
        self.dynamics_model = None
        self.wrap_angles = wrap_angles

    def train_dynamics(self):
        print_with_stamp('Training dynamics model',self.name)
        x = np.array(self.experience.states)
        u = np.array(self.experience.actions)

        # inputs are states, concatenated with actions (except for the last entry)
        x_ = gTrig_np(x, self.angle_idims)
        X = np.hstack((x_[:-1],u[:-1]))
        # outputs are changes in state
        Y =  x[1:] - x[:-1]

        if self.wrap_angles:
            # wrap angle differences to [-pi,pi]
            Y[:,self.angle_idims] = (Y[:,self.angle_idims] + np.pi) % (2 * np.pi ) - np.pi

        if self.dynamics_model is None:
            self.dynamics_model = GP_UI(X,Y)
        else:
            self.dynamics_model.set_dataset(X,Y)

        self.dynamics_model.train()
        print_with_stamp('Done training dynamics model',self.name)

    def value(self, H, derivs=False):
        print_with_stamp('Computing value of current policy',self.name)
	x = np.array(self.experience.states)
	u = np.array(self.experience.actions)

        # compile gTrig
        print_with_stamp('Compiling gTrig',self.name)
        self.gTrig = get_compiled_gTrig(self.angle_idims, x.shape[1], derivs=derivs)

        # get distribution of initial states
        x0 = x[self.experience.episode_starts]
        mx = []; Sx = []
        if x0.shape[0] > 1:
            mx = x0.mean(0)[None,:]
            Sx = np.cov(x0.T)
        else:
            mx = x0
            Sx = 1e-2*np.eye(x0.shape[1])
        mx = mx.reshape((1,mx.shape[1]))
        Sx = Sx.reshape((1,mx.shape[1],mx.shape[1]))
	
        # simulate a rollout using the dynamics model
        dt = self.plant.dt; t = 0
        while t < H:
            print '--- %f:'%(t)
            print Sx
            np.linalg.cholesky(Sx)
            # convert angles from input distribution to its complex representation
            mxa,Sxa = self.gTrig(mx,Sx)
            print Sxa
            np.linalg.cholesky(Sxa)

            # compute distribution of control signal
            Sx_ = Sx + np.diag(0.5*np.exp(self.dynamics_model.loghyp_[:,-1]))# noisy state measurement
            print Sx_
            np.linalg.cholesky(Sx_)
            mxa_,Sxa_ = self.gTrig(mx,Sx_)
            print Sxa_
            np.linalg.cholesky(Sxa_)
            mu, Su, Cu = self.policy.evaluate(t, mxa, Sxa_)
	    
            # compute state control joint distribution
            n = Sxa.shape[0]; D = Sxa.shape[1]; U = Su.shape[1]
            idims = D + U
            q = (Sxa[:,:,:,None]*Cu[:,:,None,:]).sum(1)
            mxu = np.c_[mxa,mu]
            Sxu = np.zeros((n,idims,idims))
            Sxu[:,:D,:D] = Sxa
            Sxu[:,D:,D:] = Su
            Sxu[:,:D,D:] = q
            Sxu[:,D:,:D] = q.transpose(0,2,1)
            print Sxu
            np.linalg.cholesky(Sxu)

            #  predict the change in state given current state-action
            # C_deltax = inv (Sxu) dot Sxu_deltax
            m_deltax, S_deltax, C_deltax = self.dynamics_model.predict(mxu,Sxu)
            print S_deltax
            np.linalg.cholesky(S_deltax)

            # compute the successor state distribution
            mx = mx + m_deltax
            D=mx.shape[1]
            Sx_deltax = (Sxu[:,:,:,None]*C_deltax[:,:,None,:]).sum(1)[:,:D,:]
            print Sx_deltax
            np.linalg.cholesky(Sx_deltax)
            Sx = Sx + Sx_deltax + Sx_deltax.transpose(0,2,1) + S_deltax
            #  get cost:
            #mc, Sc = cost(mx,mu,Sx,Su)
	    #print mc,Sc
	    t += dt

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

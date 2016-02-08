import numpy as np
from utils import print_with_stamp, get_compiled_gTrig,gTrig_np,gTrig2
from ghost.learners.EpisodicLearner import *
from ghost.regression.GPRegressor import GP_UI
import theano
from theano.misc.pkl_utils import dump as t_dump, load as t_load

class PILCO(EpisodicLearner):
    def __init__(self, plant, policy, cost, angle_idims=None, experience = None, async_plant=True, name='PILCO', wrap_angles=True):
        super(PILCO, self).__init__(plant, policy, cost, angle_idims, experience, async_plant, name)
        self.dynamics_model = None
        self.wrap_angles = wrap_angles
        self.propagate=None
        self.propagate_d=None

    def load(self):
        with open(self.filename+'.zip','rb') as f:
            utils.print_with_stamp('Loading compiled GP with %d inputs and %d outputs'%(self.idims,self.odims),self.name)
            state = t_load(f)
            self.set_state(state)
        self.state_changed = False
    
    def save(self):
        sys.setrecursionlimit(100000)
        if self.state_changed:
            with open(self.filename+'.zip','wb') as f:
                utils.print_with_stamp('Saving compiled GP with %d inputs and %d outputs'%(self.idims,self.odims),self.name)
                t_dump(self.get_state(),f,2)
            self.state_changed = False

    def set_state(self,state):
        self.X_shared = state[0]
        self.Y_shared = state[1]
        self.X = state[2]
        self.Y = state[3]
        self.angle_idims = state[4]
        self.angle_odims = state[5]
        self.loghyp = state[6]
        self.K = state[7]
        self.iK = state[8]
        self.beta = state[9]
        self.set_dataset(state[10],state[11])
        self.set_loghyp(state[12])
        self.nlml = state[13]
        self.dnlml = state[14]
        self.predict_ = state[15]
        self.predict_d_ = state[16]

    def get_state(self):
        return (self.X_shared,self.Y_shared,self.X,self.Y,self.angle_idims,self.angle_odims,self.loghyp,self.K,self.iK,self.beta,self.X_,self.Y_,self.loghyp_,self.nlml,self.dnlml,self.predict_,self.predict_d_)


    def init_propagate(self, derivs=False):
        ''' This compiles the propagate function, which applies the policy and predicts the next state 
            of the system using the learn GP dynamics model '''
        
        # define the function for a single propagation step
        def propagate_single_step(mx,Sx):
            D=mx.shape[1]
            # convert angles from input distribution to its complex representation
            mxa,Sxa = gTrig2(mx,Sx,self.angle_idims,self.mx0.size)

            # compute distribution of control signal
            logsn = theano.tensor.stack([ lh[-1] for lh in self.dynamics_model.loghyp ]).flatten()
            Sx_ = Sx + theano.tensor.diag(0.5*theano.tensor.exp(2*logsn))# noisy state measurement
            mxa_,Sxa_ = gTrig2(mx,Sx_,self.angle_idims,self.mx0.size)
            mu, Su, Cu = self.policy.model.predict_symbolic(mxa, Sxa_)
            
            # compute state control joint distribution
            n = Sxa.shape[0]; Da = Sxa.shape[1]; U = Su.shape[1]
            idimsa = Da + U
            q = (Sxa[:,:,:,None]*Cu[:,:,None,:]).sum(1)
            mxu = theano.tensor.horizontal_stack(mxa,mu)
            Sxu = theano.tensor.zeros((n,idimsa,idimsa))
            Sxu = theano.tensor.set_subtensor(Sxu[:,:Da,:Da], Sxa)
            Sxu = theano.tensor.set_subtensor(Sxu[:,Da:,Da:], Su)
            Sxu = theano.tensor.set_subtensor(Sxu[:,:Da,Da:], q)
            Sxu = theano.tensor.set_subtensor(Sxu[:,Da:,:Da], q.transpose(0,2,1))

            #  predict the change in state given current state-action
            # C_deltax = inv (Sxu) dot Sxu_deltax
            m_deltax, S_deltax, C_deltax = self.dynamics_model.predict_symbolic(mxu,Sxu)

            # compute the successor state distribution
            mx = mx + m_deltax
            Sx_deltax = (Sxu[:,:,:,None]*C_deltax[:,:,None,:]).sum(1)[:,:D,:]
            Sx = Sx + S_deltax + Sx_deltax + Sx_deltax.transpose(0,2,1) 

            #  get cost:
            #mc, Sc = cost(mx,mu,Sx,Su)
            #print mc,Sc

            return [mx,Sx]

        # define input variables
        print_with_stamp('Computing symbolic expression graph for belief state propagation',self.name)
        mx = theano.tensor.dmatrix('mx')
        Sx = theano.tensor.dtensor3('Sx')
        mx_next,Sx_next = propagate_single_step(mx,Sx)
        retvars = [mx_next,Sx_next]
        if derivs :
            print_with_stamp('Computing jacobians for belief state propagation',self.name)
            retvars.append( theano.tensor.jacobian( mx_next.flatten(), mx ) )
            retvars.append( theano.tensor.jacobian( Sx_next.flatten(), mx ) )
            retvars.append( theano.tensor.jacobian( mx_next.flatten(), Sx ) )
            retvars.append( theano.tensor.jacobian( Sx_next.flatten(), Sx ) )
            
            # derivatives wrt policy parameters
            params = self.policy.get_params()
            if not isinstance(params,list):
                params = [params]

            dMdp = []; dSdp = []
            #for p in params:
            #    retvars.append( theano.tensor.jacobian( mx_next.flatten(), p ) )
            #    retvars.append( theano.tensor.jacobian( Sx_next.flatten(), p ) )
            #dMdp = theano.tensor.stack(dMdp)
            #dSdp = theano.tensor.stack(dSdp)

            #retvars.append( dMdp )
            #retvars.append( dSdp )
            print_with_stamp('Compiling belief state propagation with derivatives',self.name)
            self.propagate_d = theano.function([mx,Sx], retvars)
        else:
            print_with_stamp('Compiling belief state propagation',self.name)
            self.propagate = theano.function([mx,Sx], retvars)
        


    def train_dynamics(self):
        print_with_stamp('Training dynamics model',self.name)
        
        X = []
        Y = []
        x0 = []
        n_episodes = len(self.experience.states)
        # construct training dataset
        for i in xrange(n_episodes):
            x = np.array(self.experience.states[i])
            u = np.array(self.experience.actions[i])
            x0.append(x[0])

            # inputs are states, concatenated with actions (except for the last entry)
            x_ = gTrig_np(x, self.angle_idims)
            X.append( np.hstack((x_[:-1],u[:-1])) )
            # outputs are changes in state
            Y.append( x[1:] - x[:-1] )

        X = np.vstack(X)
        Y = np.vstack(Y)

        # get distribution of initial states
        x0 = np.array(x0)
        if n_episodes > 1:
            self.mx0 = x0.mean(0)[None,:]
            self.Sx0 = np.cov(x0.T)[None,:,:]
        else:
            self.mx0 = x0[None,:]
            self.Sx0 = 1e-2*np.eye(len(x0))[None,:,:]

        if self.wrap_angles:
            # wrap angle differences to [-pi,pi]
            Y[:,self.angle_idims] = (Y[:,self.angle_idims] + np.pi) % (2 * np.pi ) - np.pi

        if self.dynamics_model is None:
            self.dynamics_model = GP_UI(X,Y)
        else:
            self.dynamics_model.set_dataset(X,Y)
        
        self.dynamics_model.train()
        self.dynamics_model.save()
        print_with_stamp('Done training dynamics model',self.name)

    def value(self, H, derivs=False):
        print_with_stamp('Computing value of current policy',self.name)

        # compile the belef state propagation
        self.init_propagate(derivs=True)
    
        mx = np.array(self.plant.x0)[None,:]
        Sx = np.array(self.plant.S0)[None,:,:]
	
        # simulate a rollout using the dynamics model
        dt = self.plant.dt; t = 0
        while t < H:
            print '--- %f:'%(t)
            print 'mx:'
            print mx
            print Sx
            ret = self.propagate_d(mx,Sx)
            mx = ret[0]
            Sx = ret[1]
            for r in ret:
                print r.shape
	    t += dt
	    
        self.policy.model.save()
        self.dynamics_model.save()
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

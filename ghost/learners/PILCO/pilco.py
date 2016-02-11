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
            D=mx.shape[0]
            # convert angles from input distribution to its complex representation
            mxa,Sxa = gTrig2(mx,Sx,self.angle_idims,self.mx0.size)

            # compute distribution of control signal
            logsn = self.dynamics_model.loghyp[:,-1]
            Sx_ = Sx + theano.tensor.diag(0.5*theano.tensor.exp(2*logsn))# noisy state measurement
            mxa_,Sxa_ = gTrig2(mx,Sx_,self.angle_idims,self.mx0.size)
            mu, Su, Cu = self.policy.model.predict_symbolic(mxa_, Sxa_)
            
            # compute state control joint distribution
            n = Sxa.shape[0]; Da = Sxa.shape[1]; U = Su.shape[1]
            idimsa = Da + U
            q = Sxa.dot(Cu)
            qT = Cu.T.dot(Sxa) # just to guarantee that the jacobian is symmetric( but has no real effect on end result)
            mxu = theano.tensor.concatenate([mxa,mu])
            Sxu = theano.tensor.zeros((idimsa,idimsa))
            Sxu = theano.tensor.set_subtensor(Sxu[:Da,:Da], Sxa)
            Sxu = theano.tensor.set_subtensor(Sxu[Da:,Da:], Su)
            Sxu = theano.tensor.set_subtensor(Sxu[:Da,Da:], q)
            Sxu = theano.tensor.set_subtensor(Sxu[Da:,:Da], qT)

            #  predict the change in state given current state-action
            # C_deltax = inv (Sxu) dot Sxu_deltax
            m_deltax, S_deltax, C_deltax = self.dynamics_model.predict_symbolic(mxu,Sxu)

            # compute the successor state distribution
            mx = mx + m_deltax
            Sx_deltax = Sxu.dot(C_deltax)[:D,:]
            Sx = Sx + S_deltax + Sx_deltax + Sx_deltax.T

            #  get cost:
            #mc, Sc = cost(mx,mu,Sx,Su)
            #print mc,Sc

            return [mx,Sx]

        # define input variables
        print_with_stamp('Computing symbolic expression graph for belief state propagation',self.name)
        mx = theano.tensor.fvector('mx') if theano.config.floatX == 'float32' else theano.tensor.dvector('mx')
        Sx = theano.tensor.fmatrix('Sx') if theano.config.floatX == 'float32' else theano.tensor.dmatrix('Sx')

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
            for p in params:
                retvars.append( theano.tensor.jacobian( mx_next.flatten(), p ) )
                retvars.append( theano.tensor.jacobian( Sx_next.flatten(), p ) )

            print_with_stamp('Compiling belief state propagation with derivatives',self.name)
            self.propagate = theano.function([mx,Sx], retvars, allow_input_downcast=True)
        else:
            print_with_stamp('Compiling belief state propagation',self.name)
            self.propagate = theano.function([mx,Sx], retvars, allow_input_downcast=True)

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
        
        self.dynamics_model.set_loghyp(np.array([[ 5.509997626011774,   5.695797298947197,   5.692015197025450,   5.638201671096178],
                                                [ 2.172712533907143,   4.666315918776204,   4.612552250383755,   4.942292180454226],
                                                [ 5.146257432313829,   2.296298649465123,   2.456701520947445,   2.614396832176411],
                                                [ 2.890900478963427,   0.214158569334003,   0.224852498280747,   0.493471679361221],
                                                [ 1.797786347384082,   1.586977710450885,  -0.058171149174515,  -0.345601213254841],
                                                [ 4.069820471633279,   3.013197670171895,   2.934665512611971,   3.616775778040698],
                                                [-0.984097642284717,   0.137890342258600,   1.299603816102084,  -0.611523812016797],
                                                [-4.115382429098694,  -4.115897287345494,  -3.168863634366222,  -4.247869488525884]]).T)
        #self.dynamics_model.train()

        #self.dynamics_model.save()
        print_with_stamp('Done training dynamics model',self.name)

    def value(self, H, derivs=False):
        print_with_stamp('Computing value of current policy',self.name)

        # compile the belef state propagation
        self.init_propagate(derivs=True)
    
        mx = np.array(self.plant.x0)
        Sx = np.array(self.plant.S0)
	
        # simulate a rollout using the dynamics model
        dt = self.plant.dt; t = 0
        while t < H:
            print_with_stamp('--- %f:'%(t),self.name)
            ret = self.propagate(mx,Sx)
            mx = ret[0]
            Sx = ret[1]
            for r in ret:
                print r
	    t += dt
	    
        self.policy.model.save()
        self.dynamics_model.save()
        # return value

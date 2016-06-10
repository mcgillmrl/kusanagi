import numpy as np
from utils import print_with_stamp, gTrig_np, gTrig2
from ghost.learners.EpisodicLearner import *
from ghost.regression.GPRegressor import GP_UI
import theano
from theano.misc.pkl_utils import dump as t_dump, load as t_load

class PDDP(EpisodicLearner):
    def __init__(self, plant, policy, cost, angle_idims=None, discount=1, experience = None, async_plant=True, name='PILCO', wrap_angles=False):
        super(PDDP, self).__init__(plant, policy, cost, angle_idims, discount, experience, async_plant, name)
        self.dynamics_model = None
        self.wrap_angles = wrap_angles
        self.rollout=None
        self.policy_gradients=None

    def init_rollout(self, derivs=False):
        ''' This compiles the rollout function, which applies the policy and predicts the next state 
            of the system using the learn GP dynamics model '''
        
        # define the function for a single propagation step
        def rollout_single_step(mx,Sx):
            ''' This function takes as an input the mean and covariance of the state at time t, and returns the 
            state and covariance of the next state (at time t+1) and the action at time t (same t as the input)
            '''
            D=mx.shape[0]

            # convert angles from input distribution to its complex representation
            mxa,Sxa,Ca = gTrig2(mx,Sx,self.angle_idims,self.mx0.size)

            # compute the control signal for the next step( This is different from PILCO: the control 
            # is a linear function  of the Gaussian belief vector, so its output is deterministic)
            logsn = self.dynamics_model.loghyp[:,-1]
            Sx_ = Sx + theano.tensor.diag(0.5*theano.tensor.exp(2*logsn))# noisy state measurement
            mxa_,Sxa_,Ca_ = utils.gTrig2(mx,Sx_,self.angle_idims,self.mx0.size)
            u_prev, ~, ~ = self.policy.evaluate(mxa_, Sxa_,symbolic=True)
            
            # compute state control joint distribution
            n = Sxa.shape[0]; Da = Sxa.shape[1]; U = u.size
            idimsa = Da + U
            mxu = theano.tensor.concatenate([mxa,u])
            Sxu = theano.tensor.zeros((idimsa,idimsa))[1]
            Sxu = theano.tensor.set_subtensor(Sxu[:Da,:Da], Sxa)

            # state control covariance without angle dimensions
	    if Ca is not None:
	        na_dims = list(set(range(self.mx0.size)).difference(self.angle_idims))
                Dna = len(na_dims)
                Sx_xa = theano.tensor.zeros((D,Da))
                Sx_xa = theano.tensor.set_subtensor(Sx_xa[:D,:Dna], Sx[:,na_dims])
                Sx_xa = theano.tensor.set_subtensor(Sx_xa[:D,Dna:], Sx.dot(Ca))
                Sxu_ = theano.tensor.zeros((D,Da+U))
                Sxu_ = theano.tensor.set_subtensor(Sxu_[:D,:Da], Sx_xa)
	    else:
                Sxu_ = Sxu

            #  predict the change in state given current state-action
            # C_deltax = inv (Sxu) dot Sxu_deltax
            m_deltax, S_deltax, C_deltax = self.dynamics_model.predict_symbolic(mxu,Sxu)

            # compute the successor state distribution
            mx_next = mx + m_deltax
            Sx_deltax = Sxu_.dot(C_deltax)
            Sx_next = Sx + S_deltax + Sx_deltax + Sx_deltax.T

            return [mx_next,Sx_next,u_prev]

        # define input variables
        print_with_stamp('Computing symbolic expression graph for belief state propagation',self.name)
        mx = theano.tensor.vector('mx')
        Sx = theano.tensor.matrix('Sx')
        H = theano.tensor.iscalar('H')
        gamma = theano.tensor.scalar('gamma')
        

        # For this, you'll need to read how theano.scan and theano.tensor.grad work ( look at pilco.py for an example )
        def forward_dynamics(mx, Sx):
            ''' This computes the Jacobian matrices Fu and Fx which satisfy the first order expansion of the dynamics.
            We use a Gaussian belief augmented state vector to incorporate uncertainty.'''
            # TODO use the rollout single step file to do the forward dynamics
            # this should return the list of all matrices (F_k)^u and (F_k)^x
            # ( Eq. (11) )
            [mx_next,Sx_next,u_prev] = rollout_single_step(mx, Sx)
            deriv1 = theano.tensor.jacobian(mx_next, mx)
            deriv2 = theano.tensor.jacobian(Sx_next, mx)
            deriv3 = theano.tensor.jacobian(mx_next, Sx)
            deriv4 = theano.tensor.jacobian(Sx_next, Sx)
            deriv5 = theano.tensor.jacobian(mx_next, u_prev)
            deriv6 = theano.tensor.jacobian(Sx_next, u_prev)
            Fx = theano.tensor.matrix('Fx')
            Fu = theano.tensor.matrix('Fu')

            deriv1_rows = deriv1.shape[0]; deriv2_rows = deriv2.shape[0]
            Fx_dims = deriv1_rows + deriv2_rows
            Fx = theano.tensor.zeros((Fx_dims,Fx_dims))
            Fx = theano.tensor.set_subtensor(Fx[:deriv1_rows,:deriv1_rows], deriv1)
            Fx = theano.tensor.set_subtensor(Fx[deriv1_rows:,:deriv1_rows], deriv2)
            Fx = theano.tensor.set_subtensor(Fx[:deriv1_rows,deriv1_rows:], deriv3)
            Fx = theano.tensor.set_subtensor(Fx[deriv1_rows:,deriv1_rows:], deriv4)
            
            Fu = theano.tensor.concatenate([deriv5,deriv6])

            z_next = theano.tensor.transpose(theano.tensor.concatenate(mx_next,theano.tensor.flatten(Sx_next)))

            return [Fx, Fu, z_next]

        def backward_propagation(mx,Sx,u,V,Vx,Vxx,Fx,Fu): 
            #TODO compute Q_t and the associated L_t and I_t
            # ( Eq. (17) ). Use self.cost_symbolic(mx,Sx) to get the cost of the current state. We
            # might need to create a new cost function that takes u as an optional input, so we can call self.cost_symbolic(mx,Sx,u)
            # you only need to use the mean of the cost to compute the derivatives ( See the paragraph after Eq 13 )
            # Here C is used in place of script L
            C, _ = self.cost_symbolic(mx,Sx)
            Cx = theano.tensor.jacobian(C, mx)
            Cxx = theano.tensor.jacobian(Cx, mx)
            Cu = theano.tensor.jacobian(C,u)
            Cux = theano.tensor.jacobian(Cu,x)
            Cuu = theano.tensor.jacobian(Cu,u)
            Qx = Cx + Vx*Fx
            Qu = Cu + Vx*Fu
            Qxx = Cxx + theano.tensor.transpose(Fx)*Vxx*Fx
            Qux = Cux + theano.tensor.transpose(Fu)*Vxx*Fx
            Quu = Cuu + theano.tensor.transpose(Fu)*Vxx*Fu

            I = -theano.tensor.nlinalg.MatrixPinv(Quu)*Qu
            L = -theano.tensor.nlinalg.MatrixPinv(Quu)*Qux

            V_prev = V + Qu*I
            Vx_prev = Qx + Qu*L
            Vxx_prev = Qxx + Qxu*L

            return [V_prev, Vx_prev, Vxx_prev, I, L]

        def locally_optimal_control(I, L, u_bar, z_bar, z):
            #Here we represent the calulcation of deltau (Eq. 16) and its usage to compute u = u_bar + delta_u
            delta_z = z - z_bar
            delta_u = I + L*(delta_z)
            u_new = u_bar + delta_u
            return u_new


        def run_DDP():


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
            self.Sx0 = 1e-2*np.eye(self.mx0.size)[None,:,:]

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

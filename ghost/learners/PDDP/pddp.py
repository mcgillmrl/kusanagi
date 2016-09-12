import numpy as np
import utils
from time import time
from ghost.learners.EpisodicLearner import EpisodicLearner
from ghost.learners.PILCO import PILCO
from ghost.learners.ExperienceDataset import ExperienceDataset
from ghost.regression.GP import GP_UI
from ghost.control import LocalLinearPolicy
from theano.tensor.nlinalg import matrix_inverse, pinv
import theano
from theano.misc.pkl_utils import dump as t_dump, load as t_load

class PDDP(PILCO):
    def __init__(self, params, plant_class, policy_class, cost_func=None, viz_class=None, dynmodel_class=GP_UI, experience = None, async_plant=False, name='PDDP', filename_prefix=None):
        params['policy']['H'] = params['H']
        params['policy']['dt'] = params['plant']['dt']
        params['angle_dims'] = []
        super(PDDP, self).__init__(params, plant_class, policy_class, cost_func, viz_class, dynmodel_class, experience, async_plant, name, filename_prefix)
    
    # define the function for a single propagation step
    def propagate_state(self,mx,Sx,t,open_loop=False):
        ''' Given the input variables mx (theano.tensor.vector) and Sx (theano.tensor.matrix),
            representing the mean and variance of the system's state x, this function returns
            the next state distribution, and the mean and variance of the immediate cost. This
            is done by 1) evaluating the current policy 2) using the dynamics model to estimate 
            the next state. The immediate cost is returned as a distribution Normal(mcost,Scost),
            since the state is uncertain.
        '''
        D = mx.shape[0]
        D_ = self.mx0.get_value().size

        # convert angles from input distribution to its complex representation
        mxa,Sxa,Ca = utils.gTrig2(mx,Sx,self.angle_idims,D_)
        
        if open_loop:
            mu = self.policy.u_nominal[t]
            Su = theano.tensor.zeros((mu.size,mu.size))
        else:
            # compute control signal given uncertain state
            sn2 = theano.tensor.exp(2*self.dynamics_model.logsn)
            Sx_ = Sx + theano.tensor.diag(0.5*sn2)# noisy state measurement
            mxa_,Sxa_,Ca_ = utils.gTrig2(mx,Sx_,self.angle_idims,D_)
            mu, Su, Cu = self.policy.evaluate(mxa_, Sxa_, t, symbolic=True)
        
        # compute state control joint distribution ( controls are deterministic, so the u terms in Sxu are set to zero )
        n = Sxa.shape[0]; Da = Sxa.shape[1]; U = Su.shape[1]
        mxu = theano.tensor.concatenate([mxa,mu])
        Sxu = theano.tensor.zeros((Da+U,Da+U))
        Sxu = theano.tensor.set_subtensor(Sxu[:Da,:Da],Sxa)

        # state control covariance without angle dimensions
        if Ca is not None:
            na_dims = list(set(range(D_)).difference(self.angle_idims))
            Sx_xa = theano.tensor.concatenate([Sx[:,na_dims],Sx.dot(Ca)],axis=1)  # [D] x [Da] 
            Sxu_ =  theano.tensor.concatenate([Sx_xa,Sx_xa.dot(Cu)],axis=1) # [D] x [Da+U]
        else:
            Sxu_ = Sxu[:D,:] # [D] x [D+U]

        #  predict the change in state given current state-action
        # C_deltax = inv (Sxu) dot Sxu_deltax
        m_deltax, S_deltax, C_deltax = self.dynamics_model.predict_symbolic(mxu,Sxu)

        # compute the successor state distribution
        mx_next = mx + m_deltax
        Sx_deltax = Sxu_.dot(C_deltax)
        Sx_next = Sx + S_deltax + Sx_deltax + Sx_deltax.T

        #  get cost at previoues time step
        mcost, Scost = self.cost(mx,Sx)
        cost_params = self.cost.keywords['params']
        # add a term for the action
        R = T.constant(cost_params['R'],dtype=mx.dtype) if 'R' in cost_params else theano.tensor.zeros((mu.size,mu.size))
        mcost += mu.dot(R).dot(mu)


        # check if dynamics model has an updates dictionary
        updates = theano.updates.OrderedUpdates()
        if hasattr(self.dynamics_model,'prediction_updates') and self.dynamics_model.prediction_updates is not None:
            updates += self.dynamics_model.prediction_updates

        if hasattr(self.policy,'prediction_updates') and self.policy.prediction_updates is not None:
            updates += self.policy.prediction_updates
        
        return [mcost,Scost,mx_next,Sx_next,mu,Su], updates

    def backward_pass(self,Fx,Fu,lx,lu,lxx,lux,luu,V,Vx,Vxx):
        Qx = lx + Vx.dot(Fx)
        Qu = lu + Vu.dot(Fu)
        Qxx = lxx + Fx.T.dot(Vxx).dot(Fx)
        Qux = lux + Fu.T.dot(Vxx).dot(Fx)
        Quu = luu + Fu.T.dot(Vxx).dot(Fu)
        iQuu = matrix_inverse(Quu)
        I = -iQuu.dot(Qu)
        L = -iQuu.dot(Qux)
        V = V + Qu.dot(I)
        Vx = Vx + Qu.dot(L)
        Vxx = Vxx + Qux.T.dot(L)
        return I,L,V,Vx,Vxx

    def compute_derivs(self,z,t,*args):
        # split z into the mean and covariance of the state
        D = ((theano.tensor.sqrt(8*z.shape[0]+9) - 3)/2).astype('int64')
        D_ = self.mx0.get_value().size
        triu_indices = self.policy.triu_indices

        mx, Sx_triu = z[:D], z[D:]
        Sx = theano.tensor.zeros((D,D))
        Sx = theano.tensor.set_subtensor(Sx[triu_indices],Sx_triu)
        Sx = Sx + Sx.T - theano.tensor.diag(theano.tensor.diag(Sx))

        # compute the next state using the dynamics model
        [mcost,Scost,mx_next,Sx_next,mu,Su], updates = self.propagate_state(mx,Sx,t,open_loop=True)

        z_next = theano.tensor.concatenate([mx_next.flatten(),Sx_next[triu_indices]])

        # compute all the required derivatives
        Fu,Fx = zip(*[theano.tensor.jacobian(z_next[i:i+D_],[mu,z]) for i in xrange(0,D_+len(triu_indices[0]),D_)])
        Fu,Fx = theano.tensor.concatenate(Fu,axis=0), theano.tensor.concatenate(Fx,axis=0)
        lu,lx = theano.tensor.grad(mcost,[mu,mx])
        luu,lux = theano.tensor.jacobian(lu.flatten(),[mu,mx], disconnected_inputs='ignore')
        lxx = theano.tensor.jacobian(lx.flatten(),mx)

        return [Fx,Fu,lx,lu,lxx,lux,luu], updates

    def compile_derivs_func(self):
        utils.print_with_stamp('Computing symbolic Jacobians along trajectory')
        u_nom = self.policy.u_nominal
        z_nom = self.policy.z_nominal
        H=z_nom.shape[0]

        shared_vars = []
        shared_vars.extend(self.dynamics_model.get_all_shared_vars())
        shared_vars.extend(self.policy.get_all_shared_vars())

        (Fx,Fu,lx,lu,lxx,lux,luu), updts = theano.scan(fn=self.compute_derivs,
                                                       sequences=[z_nom,theano.tensor.arange(H)], 
                                                       non_sequences=shared_vars, 
                                                       strict=True)

        utils.print_with_stamp('Compiling function Jacobians along trajectory')
        self.trajectory_jac_fn = theano.function([],
                                                 [Fx,Fu,lx,lu,lxx,lux,luu],
                                                 allow_input_downcast=True,
                                                 updates=updts,
                                                 name='%s>trajectory_jac_fn'%(self.name))
        utils.print_with_stamp('Done')

    def compile_derivs_func2(self):
        utils.print_with_stamp('Computing symbolic Jacobians along trajectory')
        t = theano.tensor.iscalar('t')
        z_t = self.policy.z_nominal[t]

        (Fx,Fu,lx,lu,lxx,lux,luu), updts = self.compute_derivs(z_t,t)

        utils.print_with_stamp('Compiling function Jacobians along trajectory')
        self.trajectory_jac_fn = theano.function([t],
                                                 [Fx,Fu,lx,lu,lxx,lux,luu],
                                                 allow_input_downcast=True,
                                                 updates=updts,
                                                 name='%s>trajectory_jac_fn'%(self.name))
        utils.print_with_stamp('Done')

    def train_policy(self):
        # compute derivatives along nominal trajectory
        converged=False
        self.n_evals=0
        if not hasattr(self,'trajectory_jac_fn'):
            self.compile_derivs_func()
        utils.print_with_stamp('')
        
        grads = []
        times = []
        start = time()
        #for t in xrange(40):
        #    grads.append(self.trajectory_jac_fn(t))
        grads = self.trajectory_jac_fn()
        end = time()-start
        utils.print_with_stamp("Elapsed: %f"%(end))

        #for r in zip(*grads):
        for r in grads:
            print r.shape,
        #for r in self.trajectory_jac_fn():
        #    print r.shape,
        utils.print_with_stamp('')
        
        while not converged and self.n_evals<self.max_evals:
            # initialize V, Vx and Vxx

            # backward pass

            # forward pass
            return


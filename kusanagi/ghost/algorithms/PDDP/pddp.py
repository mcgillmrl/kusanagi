import numpy as np
from kusanagi import utils
from time import time
from kusanagi.ghost.algorithms.EpisodicLearner import EpisodicLearner
from kusanagi.ghost.algorithms.PILCO import PILCO
from kusanagi.ghost.algorithms.ExperienceDataset import ExperienceDataset
import kusanagi.ghost.regression as kreg
from kusanagi.ghost.control import LocalLinearPolicy
from theano.tensor.nlinalg import matrix_inverse, pinv
from theano.tensor.slinalg import solve
import theano
import theano.tensor as tt
from theano.misc.pkl_utils import dump as t_dump, load as t_load


def unrolled_jacobian(f,wrt,D):
    return [tt.stack(df) for df in zip(*map(lambda i: [tt.grad(f[i],x) for x in wrt], range(D)))]

class PDDP(PILCO):
    def __init__(self, params, plant_class, policy_class, cost_func=None, viz_class=None, dynmodel_class=kreg.GP_UI, experience = None, async_plant=False, name='PDDP', filename_prefix=None):
        params['policy']['H'] = params['H']
        params['policy']['dt'] = params['plant']['dt']
        params['angle_dims'] = []
        super(PDDP, self).__init__(params, plant_class, policy_class, cost_func, viz_class, dynmodel_class, experience, async_plant, name, filename_prefix)
    
    # define the function for a single propagation step
    def propagate_state(self,mx,Sx,t,open_loop=False):
        ''' Given the input variables mx (tt.vector) and Sx (tt.matrix),
            representing the mean and variance of the system's state x, this function returns
            the next state distribution, and the mean and variance of the immediate cost. This
            is done by 1) evaluating the current policy 2) using the dynamics model to estimate 
            the next state. The immediate cost is returned as a distribution Normal(mcost,Scost),
            since the state is uncertain.
        '''
        dynmodel = self.dynamics_model
        D = self.mx0.get_value().size
        # convert angles from input distribution to their complex representation
        mxa, Sxa, Ca = utils.gTrig2(mx, Sx, self.angle_idims, D)

        if open_loop:
            mu = self.policy.u_nominal[t]
            Su = tt.zeros((mu.size,mu.size))
            Cu = tt.zeros((D, mu.size))
        else:
            # compute control signal given uncertain state
            sn2 = tt.exp(2*dynmodel.logsn)
            Sx_ = Sx + tt.diag(0.5*sn2)# noisy state measurement
            mxa_,Sxa_,Ca_ = utils.gTrig2(mx, Sx_, self.angle_idims, D)
            mu, Su, Cu = self.policy.evaluate(mxa_, Sxa_, t, symbolic=True)

        # compute state control joint distribution
        mxu = tt.concatenate([mxa, mu])
        q = Sxa.dot(Cu)
        Sxu_up = tt.concatenate([Sxa, q], axis=1)
        Sxu_lo = tt.concatenate([q.T, Su], axis=1)
        Sxu = tt.concatenate([Sxu_up, Sxu_lo], axis=0) # [D+U]x[D+U]
        
        #  predict the change in state given current state-action
        # C_deltax = inv (Sxu) dot Sxu_deltax
        m_deltax, S_deltax, C_deltax = dynmodel.predict_symbolic(mxu,Sxu)

        # compute the successor state distribution
        mx_next = mx + m_deltax

        # SSGP returns C_delta as the input-output covariance. All the others do it as (input covariance)^-1 dot (input-output covariance)
        if isinstance(dynmodel, kreg.SSGP) or isinstance(dynmodel, kreg.BNN):
            Sxu_deltax = C_deltax
        else:
            Sxu_deltax = Sxu.dot(C_deltax)

        if Ca is not None:
            Da = D+len(self.angle_idims); Dna = D-len(self.angle_idims)
            non_angle_dims = list(set(range(D)).difference(self.angle_idims))
            # this contains the covariance between the previous state (with angles as [sin,cos]),
            # and the next state (with angles in radians)
            Sxa_deltax = Sxu_deltax[:Da]
            # first come the non angle dimensions  [D-len(angi)] x [D]
            sxna_deltax = Sxa_deltax[:Dna]
            # then angles as [sin,cos]             [2*len(angi)] x [D]
            sxsc_deltax = Sxa_deltax[Dna:]
            #here we undo the [sin,cos] parametrization for the angle dimensions
            Sx_sc = Sx.dot(Ca)[self.angle_idims]
            Sa = Sxa[Dna:, Dna:]#+1e-12*tt.eye(2*len(self.angle_idims))
            sxa_deltax = Sx_sc.dot(solve(Sa, sxsc_deltax))
            # now we create Sx_deltax and fill it with the appropriate values
            # (i.e. in the correct order)
            Sx_deltax = tt.zeros((D, D))
            Sx_deltax = tt.set_subtensor(Sx_deltax[non_angle_dims, :], sxna_deltax)
            Sx_deltax = tt.set_subtensor(Sx_deltax[self.angle_idims, :], sxa_deltax)
        else:
            Sx_deltax = Sxu_deltax[:D]

        Sx_next = Sx + S_deltax + Sx_deltax + Sx_deltax.T

        #  get cost at previous time step
        mcost, Scost = self.cost(mx,Sx)
        cost_params = self.cost.keywords['params']
        # add a term for the action
        R = T.constant(cost_params['R'],dtype=mx.dtype) if 'R' in cost_params else tt.zeros((mu.size,mu.size))
        mcost += mu.dot(R).dot(mu)

        # check if dynamics model has an updates dictionary
        updates = theano.updates.OrderedUpdates()
        if hasattr(dynmodel,'prediction_updates') and dynmodel.prediction_updates is not None:
            updates += dynmodel.prediction_updates

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
        #D = ((tt.sqrt(8*z.shape[0]+9) - 3)/2).astype('int64')
        D = self.mx0.get_value().size
        triu_indices = np.triu_indices(D)

        mx, Sx_triu = z[:D], z[D:]
        Sx = tt.zeros((D,D))
        Sx = tt.set_subtensor(Sx[triu_indices],Sx_triu)
        Sx = Sx + Sx.T - tt.diag(tt.diag(Sx))

        # compute the next state using the dynamics model
        [mcost,Scost,mx_next,Sx_next,mu,Su], updates = self.propagate_state(mx,Sx,t,open_loop=True)

        z_next = tt.concatenate([mx_next.flatten(),Sx_next[triu_indices]])

        # compute all the required derivatives
        #Fu,Fx = zip(*[tt.jacobian(z_next[i:i+D],[mu,z]) for i in xrange(0,D+len(triu_indices[0]),D)])
        #Fu,Fx = tt.concatenate(Fu,axis=0), tt.concatenate(Fx,axis=0)
        unrolled = True
        if unrolled:
            D_ = D+D*(D+1)/2
            Fu, Fx = unrolled_jacobian(z_next,[mu,z], D_)
            luu,lux = unrolled_jacobian(lu.flatten(),[mu,z])
            lxx = unrolled_jacobian(lx.flatten(),z)
        else:
            Fu, Fx = tt.jacobian(z_next,[mu,z])
            luu,lux = tt.jacobian(lu.flatten(),[mu,z], disconnected_inputs='ignore')
            lxx = tt.jacobian(lx.flatten(),z)
        
        lu,lx = tt.grad(mcost,[mu,z])
        

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
                                                       sequences=[z_nom,tt.arange(H)], 
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
        utils.print_with_stamp('Computing symbolic Jacobians along trajectory (one step)')
        t = tt.iscalar('t')
        z_t = self.policy.z_nominal[t]

        (Fx,Fu,lx,lu,lxx,lux,luu), updts = self.compute_derivs(z_t,t)

        utils.print_with_stamp('Compiling function Jacobians along trajectory (one step)')
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
            self.compile_derivs_func2()
        utils.print_with_stamp('')
        
        grads = []
        times = []
        start = time()
        for t in xrange(40):
            grads.append(self.trajectory_jac_fn(t))
        #grads = self.trajectory_jac_fn()
        end = time()-start
        utils.print_with_stamp("Elapsed: %f"%(end))
        
        while not converged and self.n_evals<self.max_evals:
            # initialize V, Vx and Vxx

            # backward pass

            # forward pass
            return


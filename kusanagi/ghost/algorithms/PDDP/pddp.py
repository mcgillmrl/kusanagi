# pylint: disable=C0103
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


def alternative_Rop(f, x, u):
    v = theano.tensor.ones_like(f)    # Dummy variable v of same type as f
    g = theano.tensor.Lop(f, x, v)    # Jacobian of f left multiplied by v
    return theano.tensor.Lop(g.flatten(), v, u)

def unrolled_jacobian(f, wrt, D):
    return [tt.stack(df) for df in zip(*map(lambda i: [tt.grad(f[i], x) for x in wrt], range(D)))]

def wrap_belief(mx, Sx, triu_indices):
    z_next = tt.concatenate([mx.flatten(), Sx[triu_indices]])
    return z_next

def unwrap_belief(z, D):
    mx, Sx_triu = z[:D], z[D:]
    Sx = tt.zeros((D, D))
    triu_indices = np.triu_indices(D)
    Sx = tt.set_subtensor(Sx[triu_indices], Sx_triu)
    Sx = Sx + Sx.T - tt.diag(tt.diag(Sx))
    return mx, Sx, triu_indices

class PDDP(PILCO):
    def __init__(self, params, plant_class, policy_class, cost_func=None,
                 viz_class=None, dynmodel_class=kreg.GP_UI, experience=None,
                 async_plant=False, name='PDDP', filename_prefix=None):
        params['policy']['H'] = params['H']
        params['policy']['dt'] = params['plant']['dt']
        params['angle_dims'] = []
        self.lop = False
        super(PDDP, self).__init__(params, plant_class, policy_class, cost_func,
                                   viz_class, dynmodel_class, experience, async_plant,
                                   name, filename_prefix)

    # define the function for a single propagation step
    def propagate_state(self, mx, Sx, t, open_loop=False):
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
            Su = tt.zeros((mu.size, mu.size))
            Cu = tt.zeros((D, mu.size))
        else:
            # compute control signal given uncertain state
            sn2 = tt.exp(2*dynmodel.logsn)
            Sx_ = Sx + tt.diag(0.5*sn2)# noisy state measurement
            mxa_, Sxa_, Ca_ = utils.gTrig2(mx, Sx_, self.angle_idims, D)
            mu, Su, Cu = self.policy.evaluate(mxa_, Sxa_, t, symbolic=True)

        # compute state control joint distribution
        mxu = tt.concatenate([mxa, mu])
        q = Sxa.dot(Cu)
        Sxu_up = tt.concatenate([Sxa, q], axis=1)
        Sxu_lo = tt.concatenate([q.T, Su], axis=1)
        Sxu = tt.concatenate([Sxu_up, Sxu_lo], axis=0) # [D+U]x[D+U]

        #  predict the change in state given current state-action
        # C_deltax = inv (Sxu) dot Sxu_deltax
        m_deltax, S_deltax, C_deltax = dynmodel.predict_symbolic(mxu, Sxu)

        # compute the successor state distribution
        mx_next = mx + m_deltax

        # SSGP returns C_delta as the input-output covariance.
        # All the others do it as (input covariance)^-1 dot (input-output covariance)
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

        # check if dynamics model has an updates dictionary
        updates = theano.updates.OrderedUpdates()
        if hasattr(dynmodel, 'prediction_updates')\
           and dynmodel.prediction_updates is not None:
            updates += dynmodel.prediction_updates

        if hasattr(self.policy, 'prediction_updates')\
           and self.policy.prediction_updates is not None:
            updates += self.policy.prediction_updates

        return [mx_next, Sx_next, mu, Su], updates

    def get_cost(self, mx, Sx, u):
        # evaluate state cost function 
        mcost, Scost = self.cost(mx, Sx)

        # add a term for the action cost (assuming deterministic actions)
        cost_params = self.cost.keywords['params']
        R = tt.constant(cost_params['R'], dtype=mx.dtype)\
            if 'R' in cost_params\
            else tt.zeros((u.size, u.size))
        mcost += u.dot(R).dot(u)

        return mcost, Scost

    def backward_pass(self, t, z, V, Vx, Vxx, *args):
        # propragate state forward
        (z_next, u), updates = self.forward_pass(t-1, z, *args)

        # compute variables that depend on jacobian
        if self.lop:
            [VxdotFx, VxdotFu] = theano.tensor.Lop(z_next, [z, u], Vx)
            #cholVxx = tt.slinalg.cholesky(Vxx)
            qVxx, rVxx = tt.nlinalg.qr(Vxx)
            [qVxxTFx, qVxxTFu] = theano.map(lambda v, zn, z, u: theano.tensor.Lop(zn, [z, u], v),
                                            sequences=qVxx.T,
                                            non_sequences=[z_next, z, u])[0]
            rqVxx = rVxx.dot(qVxx)
            qVxxTVxxFx = rqVxx.dot(qVxxTFx)
            FxVxxFx = qVxxTFx.T.dot(qVxxTVxxFx)
            FuVxxFx = qVxxTFu.T.dot(qVxxTVxxFx)
            FuVxxFu = qVxxTFu.T.dot(rqVxx.dot(qVxxTFu))
        else:
            Fu, Fx = tt.jacobian(z_next, [u, z])
            VxdotFx = Vx.dot(Fx)
            VxdotFu = Vx.dot(Fu)
            FxVxxFx = Fx.T.dot(Vxx).dot(Fx)
            FuVxxFx = Fu.T.dot(Vxx).dot(Fx)
            FuVxxFu = Fu.T.dot(Vxx).dot(Fu)

        # get cost at current state
        D = self.mx0.get_value().size
        mx, Sx = unwrap_belief(z, D)[:2]
        l = self.get_cost(mx, Sx, u)[0]

        # compute gradients and jacobians of cost
        lu, lx = tt.grad(l, [u, z])
        luu, lux = tt.jacobian(lu.flatten(), [u, z], disconnected_inputs='ignore')
        lxx = tt.jacobian(lx.flatten(), z)

        # compute value function
        Qx = lx + VxdotFx
        Qu = lu + VxdotFu
        Qxx = lxx + FxVxxFx
        Qux = lux + FuVxxFx
        Quu = luu + FuVxxFu + 0.01*tt.eye(luu.shape[0])
        I = -tt.slinalg.solve(Quu, Qu)
        L = -tt.slinalg.solve(Quu, Qu)
        V = V + Qu.dot(I)
        Vx = Qx + Qu.dot(L)
        Vxx = Qxx + Qux.T.dot(L)
        return [V, Vx, Vxx, I, L] , updates

    def forward_pass(self, t, z, *args):
        # split z into the mean and covariance of the state
        #D = ((tt.sqrt(8*z.shape[0]+9) - 3)/2).astype('int64')
        D = self.mx0.get_value().size
        mx, Sx, triu_indices = unwrap_belief(z, D)

        # compute the next state using the dynamics model
        outs, updates = self.propagate_state(mx, Sx, t, open_loop=True)
        mx_next, Sx_next, mu, Su = outs

        # build belief vector
        z_next = wrap_belief(mx_next, Sx_next, triu_indices)

        return [z_next, mu], updates

    def compile_forward_pass(self):
        utils.print_with_stamp('Computing symbolic forward pass')
        u_nom = self.policy.u_nominal
        z_nom = self.policy.z_nominal
        H = z_nom.shape[0]

        shared_vars = []
        shared_vars.extend(self.dynamics_model.get_all_shared_vars())
        shared_vars.extend(self.policy.get_all_shared_vars())

        forw_out, f_updts = theano.scan(fn=self.forward_pass,
                                        outputs_info=[z_nom[0], u_nom[0]],
                                        sequences=[tt.arange(H)],
                                        non_sequences=shared_vars,
                                        strict=True)
        z_next, u = forw_out
        utils.print_with_stamp('Compiling forward pass')
        self.trajectory_jac_fn = theano.function([],
                                                 [z_next, u],
                                                 allow_input_downcast=True,
                                                 updates=f_updts,
                                                 name='%s>trajectory_jac_fn'%(self.name))

    def compile_backward_pass(self):
        utils.print_with_stamp('Computing symbolic backward pass Lop')
        u_nom = self.policy.u_nominal
        z_nom = self.policy.z_nominal

        # get terminal cost (cost of the las time step in this case)
        D = self.mx0.get_value().size
        H = z_nom.shape[0]
        z_H = self.forward_pass(H-1, z_nom[-1])[0][0]
        mx_H, Sx_H = unwrap_belief(z_H, D)[:2]
        l_H = self.get_cost(mx_H, Sx_H, u_nom[-1])[0]

        # compute gradients and jacobians of terminal cost
        lx_H = tt.grad(l_H, z_H)
        lxx_H = tt.jacobian(lx_H.flatten(), z_H)

        # get shared vars
        shared_vars = []
        shared_vars.extend(self.dynamics_model.get_all_shared_vars())
        shared_vars.extend(self.policy.get_all_shared_vars())

        self.lop = True
        back_out, b_updts = theano.scan(fn=self.backward_pass,
                                        outputs_info=[l_H, lx_H, lxx_H, None, None],
                                        sequences=[tt.arange(H), z_nom[:H]],
                                        non_sequences=shared_vars,
                                        go_backwards=True,
                                        strict=True)
        self.lop = False
        utils.print_with_stamp('Computing symbolic backward pass')
        back_out2, b_updts = theano.scan(fn=self.backward_pass,
                                         outputs_info=[l_H, lx_H, lxx_H, None, None],
                                         sequences=[tt.arange(H), z_nom[:H]],
                                         non_sequences=shared_vars,
                                         go_backwards=True,
                                         strict=True)

        utils.print_with_stamp('Compiliing backward pass Lop')
        self.trajectory_jac_fn2 = theano.function([],
                                                  back_out,
                                                  allow_input_downcast=True,
                                                  updates=b_updts,
                                                  name='%s>trajectory_jac_fn2'%(self.name))
        utils.print_with_stamp('Compiliing backward pass Lop')
        self.trajectory_jac_fn3 = theano.function([],
                                                  back_out2,
                                                  allow_input_downcast=True,
                                                  updates=b_updts,
                                                  name='%s>trajectory_jac_fn2'%(self.name))
        return

    def train_policy(self):
        # compute derivatives along nominal trajectory
        converged = False
        self.n_evals = 0
        if not hasattr(self, 'trajectory_jac_fn'):
            self.compile_forward_pass()
            self.compile_backward_pass()
        utils.print_with_stamp('')

        grads = []
        times = []
        #for t in xrange(40):
        #    grads.append(self.trajectory_jac_fn(t))
        #grads = self.trajectory_jac_fn()
        start = time()
        V, Vx, Vxx, I, L = self.trajectory_jac_fn2()
        end = time()-start
        utils.print_with_stamp("Elapsed: %f"%(end))
        start = time()
        V_, Vx_, Vxx_, I_, L_ = self.trajectory_jac_fn3()
        end = time()-start
        utils.print_with_stamp("Elapsed: %f"%(end))
        #print(grads)
        print(I, I_)
        print(L, L_)

        while not converged and self.n_evals < self.max_evals:
            # initialize V, Vx and Vxx

            # backward pass

            # forward pass
            return
# pylint: disable=C0103
import numpy as np
import theano
import theano.tensor as tt

from kusanagi import utils
from kusanagi.ghost.regression import RBFGP, SSGP_UI
from kusanagi.ghost.control.saturation import gSat

from kusanagi.base.Loadable import Loadable
from functools import partial

floatX = theano.config.floatX


# GP based controller
class RBFPolicy(RBFGP):
    def __init__(self, idims=None, odims=None, sat_func=gSat, state0_dist=None,
                 maxU=[10], minU=None, n_inducing=10, angle_dims=[], name='RBFPolicy',
                 filename=None, max_evals=750, *kwargs):
        self.maxU = np.array(maxU)
        self.minU = np.array(minU) if minU is not None else -self.maxU
        self.n_inducing = n_inducing
        self.angle_dims = angle_dims
        self.name = name
        self.state0_dist = state0_dist

        if callable(sat_func):
            # set the model to be a RBF with saturated outputs
            maxU = self.maxU - self.minU
            sat_func = partial(sat_func, e=0.5*maxU)
            def sfunc(*args, **kwargs):
                return sat_func(*args, **kwargs) + 0.5*maxU + self.minU
            self.sat_func = sfunc

        if filename is not None:
            # try loading from file
            super(RBFPolicy, self).__init__(
                idims=0, odims=0, sat_func=self.sat_func, max_evals=max_evals,
                name=self.name, filename=filename)
            # self.load()
        else:
            if self.state0_dist is None:
                self.state0_dist = utils.distributions.Gaussian(
                    np.zeros((idims, )), 0.01*np.eye(idims))
            idims = self.state0_dist.mean.size
            odims = len(self.maxU)
            super(RBFPolicy, self).__init__(
                idims=idims, odims=odims, sat_func=self.sat_func,
                max_evals=max_evals, name=self.name)
            self.init_params()

        # make sure we always get the parameters in the same order
        self.param_names = ['X', 'Y', 'unconstrained_hyp']

    def load(self, output_folder=None, output_filename=None):
        ''' loads the state from file, and initializes additional variables'''
        # load state
        super(RBFGP, self).load(output_folder, output_filename)

        # don't optimize the signal and noise variances
        self.hyp = tt.concatenate(
            [self.hyp[:, :-2],
             theano.gradient.disconnected_grad(self.hyp[:, -2:])],
            axis=np.array(1, dtype='int64'))

        # hyp is no longer the trainable paramter
        self.predict_fn = None

    def init_params(self, compile_funcs=False):
        utils.print_with_stamp('Initializing parameters', self.name)

        # init inputs
        inputs_ = self.state0_dist.sample(self.n_inducing)
        inputs = utils.gTrig_np(inputs_, self.angle_dims)

        # set the initial log hyperparameters (1 for linear dimensions,
        # 0.7 for angular)
        l0 = np.hstack([np.ones(inputs_.shape[1]-len(self.angle_dims)),
                        0.7*np.ones(2*len(self.angle_dims)), 1, 0.01])

        l0 = np.tile(l0, (self.maxU.size, 1)).astype(floatX)
        l0 = np.log(np.exp(l0, dtype=floatX) - 1.0)

        # init policy targets close to zero
        mu = np.zeros((self.maxU.size, ))
        Su = 0.1*np.eye(self.maxU.size)
        targets = utils.distributions.Gaussian(mu, Su).sample(self.n_inducing)
        targets = targets.reshape((self.n_inducing, self.maxU.size))

        self.trained = False

        # set the parameters
        self.N = inputs.shape[0]
        self.D = inputs.shape[1]
        self.E = targets.shape[1]

        self.set_params({'X': inputs.astype(floatX),
                         'Y': targets.astype(floatX)})
        self.set_params({'unconstrained_hyp': l0.astype(floatX)})
        eps = np.finfo(np.__dict__[floatX]).eps
        self.hyp = tt.nnet.softplus(self.unconstrained_hyp) + eps

        # don't optimize the signal and noise variances
        self.hyp = tt.concatenate(
            [self.hyp[:, :-2],
             theano.gradient.disconnected_grad(self.hyp[:, -2:])],
            axis=np.array(1, dtype='int64'))

        # call init loss to initialize the intermediate shared variables
        super(RBFGP, self).get_loss(cache_intermediate=False)

        # init the prediction function
        self.evaluate(np.zeros((self.D, )))

    def evaluate(self, m, s=None, t=None, symbolic=False, **kwargs):
        if symbolic:
            ret = self.predict_symbolic(m, s)
        else:
            ret = self.predict(m, s)
        return ret


# random controller
class RandPolicy:
    def __init__(self, maxU=[10], minU=None, random_walk=False):
        self.maxU = np.array(maxU)
        self.minU = np.array(minU) if minU is not None else -self.maxU
        # self.last_u = np.zeros_like(np.array(maxU))
        self.random_walk = random_walk
        self.last_u = None

    def evaluate(self, m, s=None, t=None, symbolic=False):
        scale = self.maxU - self.minU
        bias = self.minU
        if self.random_walk:
            new_u = np.random.random(scale.size)
            new_u = new_u.reshape(scale.shape)*self.scale + bias
            r = np.random.binomial(1, 0.3)*0.75
            ret = (new_u if self.last_u is None or t==0
                   else self.last_u + r*(new_u - self.last_u))
            ret = np.min((ret.flatten(), self.maxU.flatten()), axis=0)
            ret = np.max((ret.flatten(), self.minU.flatten()), axis=0)
            ret = ret.reshape(self.maxU.shape)
        else:
            ret = np.random.random(scale.size)
            ret = new_u.reshape(scale.shape)*self.scale + bias

        self.last_u = ret
        U = len(self.maxU)
        D = m.shape[0]
        return ret, np.zeros((U, U)), np.zeros((D, U))


# linear time varying policy
class LocalLinearPolicy(Loadable):
    def __init__(self, H, dt, m0, S0=None, maxU=[10], angle_dims=[],
                 name='LocalLinearPolicy', **kwargs):
        self.maxU = np.array(maxU)
        self.angle_dims = angle_dims
        self.H = H
        self.dt = dt
        self.m0 = m0
        D = len(self.m0)
        self.S0 = S0 if S0 is not None else np.zeros((D, D))
        self.t = 0
        self.noise = 0
        self.name = name
        self.init_params()

        Loadable.__init__(self, name=name, filename=self.filename)
        # register theano functions and shared variables for saving
        self.register_types([tt.sharedvar.SharedVariable,
                             theano.compile.function_module.Function])

    def init_params(self):
        H_steps = int(np.ceil(self.H/self.dt))
        self.state_changed = False

        # set random (uniform distribution) controls
        u = self.maxU*(2*np.random.random((H_steps, len(self.maxU))) - 1)
        self.u_nominal = theano.shared(u)

        # intialize the nominal states to the appropriate size
        m0, S0 = utils.gTrig2_np(np.array(self.m0)[None, :],
                                 np.array(self.S0)[None, :, :],
                                 self.angle_dims,
                                 len(self.m0))
        self.triu_indices = np.triu_indices(m0.size)
        z0 = np.concatenate([m0.flatten(), S0[0][self.triu_indices]])
        z = np.tile(z0, (H_steps, 1))
        self.z_nominal = theano.shared(z)

        # initialize the open loop and feedback matrices
        I = np.zeros((H_steps, len(self.maxU)))
        L = np.zeros((H_steps, len(self.maxU), z0.size))
        self.I = theano.shared(I)
        self.L = theano.shared(L)

        # set a meaningful filename
        self.filename = self.name+'_'+str(len(self.m0))+'_'+str(len(self.maxU))

    def evaluate(self, m, s=None, t=None, symbolic=False):
        D = m.shape[0]
        if t is not None:
            self.t = t
        t = self.t

        u, z, I, L = self.u_nominal, self.z_nominal, self.I, self.L

        if symbolic:
            tt_ = theano.tensor
        else:
            tt_ = np
            u, z = u.get_value(), z.get_value(),
            I, L = I.get_value(), L.get_value()

        if s is None:
            s = tt_.zeros((D, D))

        # construct flattened state covariance vector
        z_t = tt_.concatenate([m.flatten(), s[self.triu_indices]])
        # compute control
        u_t = u[t] + I[t] + L[t].dot(z_t - z[t])
        # add random noise if requested (only for non symbolic)
        if not symbolic and self.noise and self.noise > 0:
            u_t += self.noise*tt_.random.randn(*u_t.shape)

        # limit the controller output
        u_t = tt_.maximum(u_t, -self.maxU)
        u_t = tt_.minimum(u_t, self.maxU)

        U = u_t.shape[0]
        self.t += 1
        return u_t, tt_.zeros((U, U)), tt_.zeros((D, U))

    def get_params(self, symbolic=False, t=None):
        params = [self.u_nominal, self.z_nominal, self.I, self.L]

        if not symbolic:
            params = [p.get_value() for p in params]
        return params

    def get_all_shared_vars(self):
        return [attr for attr in list(self.__dict__.values())
                if isinstance(attr, tt.sharedvar.SharedVariable)]


class AdjustedPolicy:
    def __init__(self, source_policy, maxU=[10], angle_dims=[],
                 name='AdjustedPolicy', adjustment_model_class=SSGP_UI,
                 use_control_input=True, **kwargs):
        self.use_control_input = use_control_input
        self.angle_dims = angle_dims
        self.name = name
        self.maxU = maxU

        self.source_policy = source_policy
        # TODO we may add a saturating function here
        self.adjustment_model = adjustment_model_class(
            idims=self.source_policy.D, odims=self.source_policy.E,
            name='AdjustmentModel', **kwargs)

    def init_params(self):
        # self.source_policy.init_params() TODO
        pass

    def evaluate(self, m, S=None, t=None, symbolic=False):
        tt_ = theano.tensor if symbolic else np
        S = S if S is not None else 1e-9*tt_.eye(m.size)
        # get the output of the source policy
        mu, Su, Cu = self.source_policy.evaluate(m, S, t, symbolic)

        if self.adjustment_model.trained:
            # initialize the inputs to the policy adjustment function
            adj_input_m = m
            adj_input_S = S

            if self.use_control_input:
                adj_input_m = tt_.concatenate([adj_input_m, mu])
                # fill input convariance matrix
                q = adj_input_S.dot(Cu)
                Sxu_up = tt_.concatenate([adj_input_S, q], axis=1)
                Sxu_lo = tt_.concatenate([q.T, Su], axis=1)
                adj_input_S = tt_.concatenate([Sxu_up, Sxu_lo], axis=0)

            if symbolic:
                madj, Sadj, Cadj = self.adjustment_model.predict_symbolic(
                    adj_input_m, adj_input_S)
            else:
                madj, Sadj, Cadj = self.adjustment_model.predict(
                    adj_input_m, adj_input_S)

            # compute the adjusted control distribution
            mu = mu + madj
            Sxu_adj = adj_input_S.dot(Cadj)
            Su_adj = Sxu_adj[m.size:]
            Su = Su + Sadj + Su_adj + Su_adj.T
            if S is not None:
                if symbolic:
                    Cu = Cu + tt_.slinalg.Solve(S, Sxu_adj[:m.size])
                else:
                    Cu = Cu + np.linalg.pinv(S).dot(Sxu_adj[:m.size])

        return mu, Su, Cu

    def get_params(self, symbolic=False):
        return self.adjustment_model.get_params(symbolic)

    def set_params(self, params):
        return self.adjustment_model.set_params(params)

    def get_all_shared_vars(self):
        return (self.source_policy.get_all_shared_vars()
                + self.adjustment_model.get_all_shared_vars())

    def load(self, output_folder=None, output_filename=None):
        self.adjustment_model.load(output_folder, output_filename)

    def save(self, output_folder=None, output_filename=None):
        self.adjustment_model.save(output_folder, output_filename)

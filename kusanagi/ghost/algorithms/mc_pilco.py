import lasagne
import numpy as np
import theano
import theano.tensor as tt
from kusanagi import utils

randint = lasagne.random.get_rng().randint(1, 2147462579)
m_rng = theano.sandbox.rng_mrg.MRG_RandomStreams(randint)
s_rng = theano.tensor.shared_randomstreams.RandomStreams(randint)


def propagate_particles(latent_x, measured_x, pol, dyn, angle_dims=[],
                        iid_per_eval=False, **kwargs):
    ''' Given a set of input states, this function returns predictions for
        the next states. This is done by 1) evaluating the current pol
        2) using the dynamics model to estimate the next state. If x has
        shape [n, D] where n is tthe number of samples and D is the state
        dimension, this function will return x_next with shape [n, D]
        representing the next states and costs with shape [n, 1]
    '''
    # convert angles from input states to their complex representation
    xa1 = utils.gTrig(latent_x, angle_dims)
    xa2 = utils.gTrig(measured_x, angle_dims)

    # compute controls for each sample
    u = pol.evaluate(xa2, symbolic=True,
                     iid_per_eval=iid_per_eval,
                     return_samples=True)

    # build state-control vectors
    xu = tt.concatenate([xa1, u], axis=1)

    # predict the change in state given current state-control for each particle
    delta_x, sn_x = dyn.predict_symbolic(xu, iid_per_eval=iid_per_eval,
                                         return_samples=True)

    # compute the successor states
    x_next = latent_x + delta_x

    return x_next, sn_x


def rollout(x0, H, gamma0,
            pol, dyn, cost,
            angle_dims=[],
            z=None, mm_state=True, mm_cost=True,
            noisy_policy_input=True, noisy_cost_input=True,
            truncate_gradient=-1, extra_shared=[],
            split_H=2, **kwargs):
    ''' Given some initial state particles x0, and a prediction horizon H
    (number of timesteps), returns a set of trajectories sampled from the
    dynamics model and the discounted costs for each step in the
    trajectory.
    '''
    msg = 'Building computation graph for rollout'
    utils.print_with_stamp(msg, 'mc_pilco.rollout')
    msg = 'Moment-matching [state: %s, cost:%s]'
    msg += ', State measurement noise [policy: %s, cost: %s]'
    opts = (mm_state, mm_cost, noisy_policy_input, noisy_cost_input)
    utils.print_with_stamp(msg % opts, 'mc_pilco.rollout')

    # define internal scan computations
    def step_rollout(z1, z2, z2_prev, x, sn, gamma, *args):
        '''
            Single step of rollout.
        '''
        n = x.shape[0]
        n = n.astype(theano.config.floatX)

        # noisy state measruement for control
        xn = x + z2_prev*(0.5*sn) if noisy_policy_input else x

        # get next state distribution
        x_next, sn_next = propagate_particles(
            x, xn, pol, dyn, angle_dims, **kwargs)

        def eval_cost(xn, mxn=None, Sxn=None):
            c = cost(xn, None)
            # moment-matching for cost
            if mm_cost:
                # compute input moments
                if mxn is None:
                    mxn = xn.mean(0)
                if Sxn is None:
                    Sxn = (xn.T.dot(xn)/n
                           - tt.outer(mxn, mxn))
                # propagate gaussian through cost (should be implemented in
                # cost func)
                mc = cost(mxn, Sxn)[0]
            # no moment-matching
            else:
                mc = c.sum()/n
            return mc, c

        # if resampling (moment-matching for state)
        if mm_state:
            mx_next = x_next.mean(0)
            Sx_next = x_next.T.dot(x_next)/n - tt.outer(mx_next, mx_next)
            x_next = mx_next + z1.dot(tt.slinalg.cholesky(Sx_next).T)
            # noisy state measurement for cost
            xn_next = x_next
            if noisy_cost_input:
                xn_next += z2*sn_next
                #  get cost of applying action:
                mc_next, c_next = eval_cost(xn_next)
            else:
                mc_next, c_next = eval_cost(xn_next, mx_next, Sx_next)
        # no moment-matching for state
        else:
            # noisy state measurement for cost
            xn_next = x_next + z2*sn_next if noisy_cost_input else x_next
            #  get cost of applying action:
            mc_next, c_next = eval_cost(xn_next)

        return [gamma*mc_next, gamma*c_next, x_next, sn_next, gamma*gamma0]

    # these are the shared variables that will be used in the scan graph.
    # we need to pass them as non_sequences here
    # see: http://deeplearning.net/software/theano/library/scan.html
    nseq = [gamma0]
    nseq.extend(dyn.get_intermediate_outputs())
    nseq.extend(pol.get_intermediate_outputs())
    nseq.extend(extra_shared)

    # loop over the planning horizon
    mode = theano.compile.mode.get_mode('FAST_RUN')
    mcosts, costs, trajectories = [], [] ,[]
    H_ = tt.ceil(H*1.0/split_H).astype('int32') # if split_H > 1, this results in truncated BPTT
    for i in range(1, split_H+1):
        start_idx = (i-1)*H_ + 1
        end_idx = start_idx + H_
        output = theano.scan(
            fn=step_rollout, sequences=[z[0, start_idx:end_idx], z[1, start_idx:end_idx], z[1, -end_idx:-start_idx]],
            outputs_info=[None, None, x0, 1e-4*tt.ones_like(x0), gamma0],
            non_sequences=nseq, strict=True, allow_gc=False,
            truncate_gradient=truncate_gradient, name="mc_pilco>rollout_scan_%d"%(i),
            mode=mode)

        rollout_output, rollout_updts = output
        mcosts_i, costs_i, trajectories_i = rollout_output[:3]
        mcosts.append(mcosts_i)
        costs.append(costs_i)
        trajectories.append(trajectories_i)
        x0 = trajectories_i[-1, :, :]
        x0 = theano.gradient.disconnected_grad(x0) # this causes truncated backprop
    
    mcosts = tt.concatenate(mcosts)
    costs = tt.concatenate(costs)
    trajectories = tt.concatenate(trajectories)

    trajectories.name = 'trajectories'

    # first axis: batch, second axis: time step
    costs = costs.T
    # first axis; batch, second axis: time step
    trajectories = trajectories.transpose(1, 0, 2)

    return [mcosts, costs, trajectories], rollout_updts


def get_loss(pol, dyn, cost, angle_dims=[], n_samples=50,
             intermediate_outs=False, mm_state=True, mm_cost=True,
             noisy_policy_input=True, noisy_cost_input=True,
             resample_dyn=False, crn=True, average=True,
             truncate_gradient=-1, split_H=1, extra_shared=[], **kwargs):
    '''
        Constructs the computation graph for the value function according to
        the mc-pilco algorithm:
        1) sample x0 from initial state distribution N(mx0,Sx0)
        2) propagate the state particles forward in time
            2a) compute controls for eachc particle
            2b) use dynamics model to predict next states for each particle
            2c) compute cost for each particle
        3) return the expected value of the sum of costs
        @param pol
        @param dyn
        @param cost
        @param angle_dims angle dimensions that should be converted to complex
                          representation
        @param crn wheter to use common random numbers.
        @return Returns a tuple of (outs, inps, updts). These correspond to the
                output variables, input variables and updates dictionary, if
                any.
                By default, the only output variable is the value.
    '''
    # get angle dims from policy, if any
    if len(angle_dims) == 0 and hasattr(pol, 'angle_dims'):
        angle_dims = pol.angle_dims
    # make sure that the dynamics model has the same number of samples
    if hasattr(dyn, 'update'):
        dyn.update(n_samples)
    if hasattr(pol, 'update'):
        pol.update(n_samples)

    # initial state distribution
    mx0 = tt.vector('mx0')
    Sx0 = tt.matrix('Sx0')

    # prediction horizon
    H = tt.iscalar('H')
    # discount factor
    gamma = tt.scalar('gamma')
    # how many times we've done a forward pass
    n_evals = theano.shared(0)
    # new samples with every rollout
    z = m_rng.normal((2, H+1, n_samples, mx0.shape[0]))

    # sample random numbers to be used in the rollout
    updates = theano.updates.OrderedUpdates()
    if crn:
        utils.print_with_stamp(
            "Using common random numbers for moment matching",
            'mc_pilco.rollout')
        # we reuse samples and resamples every 500 iterations
        # resampling is done to avoid getting stuck with bad solutions
        # when we get unlucky.
        z_resampled = z
        z_init = np.random.normal(
            size=(2, 1000, n_samples, dyn.E)).astype(theano.config.floatX)
        z = theano.shared(z_init)
        updates[z] = theano.ifelse.ifelse(
            tt.eq(n_evals % 500, 0), z_resampled, z)
        updates[n_evals] = n_evals + 1

        # now we will make sure that z is has the correct shape
        z = theano.ifelse.ifelse(
            z.shape[1] < H,
            tt.tile(z, (1, tt.ceil(H/z.shape[0]).astype('int64'), 1, 1)),
            z
        )[:H+1]

    # draw initial set of particles
    z0 = m_rng.normal((n_samples, mx0.shape[0]))
    Lx0 = tt.slinalg.cholesky(Sx0)
    x0 = mx0 + z0.dot(Lx0.T)

    # get rollout output
    r_outs, updts = rollout(x0, H, gamma,
                            pol, dyn, cost,
                            angle_dims,
                            z=z,
                            mm_state=mm_state,
                            iid_per_eval=resample_dyn,
                            mm_cost=mm_cost,
                            truncate_gradient=truncate_gradient,
                            split_H=split_H,
                            noisy_policy_input=noisy_policy_input,
                            noisy_cost_input=noisy_cost_input,
                            extra_shared=extra_shared, **kwargs)

    mean_costs, costs, trajectories = r_outs

    # loss is E_{dyns}((1/H)*sum c(x_t))
    #          = (1/H)*sum E_{x_t}(c(x_t))
    loss = mean_costs.mean() if average else mean_costs.sum()

    inps = [mx0, Sx0, H, gamma]
    updates += updts
    if intermediate_outs:
        return [loss, costs, trajectories], inps, updates
    else:
        return loss, inps, updates


def build_rollout(*args, **kwargs):
    kwargs['intermediate_outs'] = True
    outs, inps, updts = get_loss(*args, **kwargs)
    rollout_fn = theano.function(inps, outs, updates=updts,
                                 allow_input_downcast=True)
    return rollout_fn

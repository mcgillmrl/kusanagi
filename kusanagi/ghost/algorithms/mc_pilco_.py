import lasagne
import theano
import theano.tensor as tt

from kusanagi import utils

randint = lasagne.random.get_rng().randint(1, 2147462579)
m_rng = theano.sandbox.rng_mrg.MRG_RandomStreams(randint)

def propagate_particles(x, pol, dyn, D, angle_dims=None, iid_per_eval=False):
    ''' Given a set of input states, this function returns predictions for
        the next states. This is done by 1) evaluating the current pol
        2) using the dynamics model to estimate the next state. If x has
        shape [n, D] where n is tthe number of samples and D is the state
        dimension, this function will return x_next with shape [n, D]
        representing the next states and costs with shape [n, 1]
    '''
    # convert angles from input states to their complex representation
    xa = utils.gTrig(x, angle_dims, D)
    xa.name='xa'

    # compute controls for each sample
    sn = tt.exp(dyn.logsn)
    sn.name = 'sn'
    Lnoise = tt.diag(sn/2)
    Lnoise.name = 'Lnoise'
    x_noisy = x + m_rng.normal(x.shape).dot(Lnoise)
    x_noisy.name='x_noisy'
    xa_noisy = utils.gTrig(x_noisy, angle_dims, D)
    xa_noisy.name = 'xa_noisy'
    u = pol.evaluate(xa_noisy, symbolic=True, iid_per_eval=iid_per_eval, return_samples=True)
    u.name = 'u'
    #u = pol.evaluate(xa, symbolic=True, iid_per_eval=iid_per_eval, return_samples=True)

    # build state-control vectors
    xu = tt.concatenate([xa, u], axis=1)
    xu.name='xu'

    # predict the change in state given current state-control for each particle
    delta_x = dyn.predict_symbolic(xu, iid_per_eval=iid_per_eval, return_samples=True)
    delta_x.name='delta_x'

    # compute the successor states
    x_next = x + delta_x
    x_next.name='x_next'

    return x_next

def rollout(x0, H, gamma0,
            pol, dyn, cost,
            D, angle_dims=None, resample=False,
            **kwargs):
    ''' Given some initial state particles x0, and a prediction horizon H
    (number of timesteps), returns a set of trajectories sampled from the
    dynamics model and the discounted costs for each step in the
    trajectory.
    '''
    utils.print_with_stamp('Building computation graph for state particles propagation',
                           'mc_pilco.rollout')

    # define internal scan computations
    def step_rollout(x, gamma, *args):
        '''
            Single step of rollout.
        '''
        x.name = 'x'
        # get next state distribution
        x_next = propagate_particles(x, pol, dyn, D, angle_dims, **kwargs)
        x_next.name = 'x_next'
        #  get cost of applying action:
        n = x_next.shape[0]
        n = n.astype(theano.config.floatX)
        n.name='n'
        mx_next = x_next.mean(0)
        mx_next.name='mx_next'
        Sx_next = x_next.T.dot(x_next)/n - tt.outer(mx_next, mx_next)
        Sx_next.name='Sx_next'
        # with measurement noise
        sn2 = tt.exp(2*dyn.logsn)
        sn2.name='sn2'
        Sx_next += tt.diag(sn2)
        Sx_next.name = 'Sx_next+noise'
        mc_next = cost(mx_next, Sx_next)[0]
        mc_next.name = 'mc_next'
        c_next = cost(x_next, None)
        c_next.name = 'c_next'
        # resample if requested
        if resample:
            z = m_rng.normal(x.shape)
            z.name = 'z'
            x_next = mx_next + z.dot(tt.slinalg.cholesky(Sx_next).T)
            x_next.name = 'x_next_resampled'
        return [gamma*mc_next, gamma*c_next, x_next, gamma*gamma0]

    # these are the shared variables that will be used in the scan graph.
    # we need to pass them as non_sequences here
    # see: http://deeplearning.net/software/theano/library/scan.html
    shared_vars = []
    shared_vars.extend(dyn.get_all_shared_vars())
    shared_vars.extend(pol.get_all_shared_vars())

    # loop over the planning horizon
    output = theano.scan(fn=step_rollout,
                         outputs_info=[None, None, x0, gamma0],
                         non_sequences=[gamma0]+shared_vars,
                         n_steps=H,
                         strict=True,
                         allow_gc=False,
                         profile=True,
                         name="mc_pilco>rollout_scan")
    rollout_output, rollout_updts = output
    mcosts, costs, trajectories = rollout_output[:3]
    mcosts.name = 'mcosts'
    costs.name = 'costs'
    trajectories.name = 'trajectories'

    # first axis: batch, second axis: time step
    costs = costs.T
    # first axis; batch, second axis: time step
    trajectories = trajectories.transpose(1, 0, 2)

    mcosts.name = 'mc_list'
    costs.name = 'c_list'
    trajectories.name = 'x_list'

    return [mcosts, costs, trajectories], rollout_updts

def get_loss(pol, dyn, cost, D, angle_dims, n_samples=50,
             intermediate_outs=False, resample_particles=False,
             resample_dyn=False, average=True):
    '''
        Constructs the computation graph for the value function according to the
        mcpilco algorithm:
        1) sample x0 from initial state distribution N(mx0,Sx0)
        2) propagate the state particles forward in time
            2a) compute controls for eachc particle
            2b) use dynamics model to predict next states for each particle
            2c) compute cost for each particle
        3) return the expected value of the sum of costs
        @param pol
        @param dyn
        @param cost
        @param D number of state dimensions, must be a python integer
        @param angle_dims angle dimensions that should be converted to complex
                          representation
        @return Returns a tuple of (outs, inps, updts). These correspond to the
                output variables, input variables and updates dictionary, if any.
                By default, the only output variable is the value.
    '''
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


    # draw initial set of particles
    z0 = m_rng.normal((n_samples, mx0.size))
    z0.name = 'z0'
    Lx0 = tt.slinalg.cholesky(Sx0)
    Lx0.name = 'Lx0'
    x0 = mx0 + z0.dot(Lx0.T)
    x0.name = 'x0'

    # get rollout output
    r_outs, updts = rollout(x0, H, gamma,
                            pol, dyn, cost,
                            D, angle_dims,
                            resample=resample_particles,
                            iid_per_eval=resample_dyn)

    mean_costs, costs, trajectories = r_outs
    #mean_costs = costs.mean(0) # mean over particles

    # loss is E_{dyns}((1/H)*sum c(x_t))
    #          = (1/H)*sum E_{x_t}(c(x_t))
    loss = mean_costs.mean() if average else mean_costs.sum()

    inps = [mx0, Sx0, H, gamma]
    if intermediate_outs:
        return [loss, costs, trajectories], inps, updts
    else:
        return loss, inps, updts

def build_rollout(*args, **kwargs):
    kwargs['intermediate_outs'] = True
    outs, inps, updts = get_loss(*args, **kwargs)  
    rollout_fn = theano.function(inps, outs, updates=updts, allow_input_downcast=True)
    return rollout_fn
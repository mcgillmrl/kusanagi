import lasagne
import theano
import theano.tensor as tt

from kusanagi import utils

randint = lasagne.random.get_rng().randint(1, 2147462579)
m_rng = theano.sandbox.rng_mrg.MRG_RandomStreams(randint)

def propagate_particles(x, policy, dynmodel, D, angle_dims=None, resample=True, iid_per_eval=False):
    ''' Given a set of input states, this function returns predictions for
        the next states. This is done by 1) evaluating the current policy
        2) using the dynamics model to estimate the next state. If x has
        shape [n, D] where n is tthe number of samples and D is the state
        dimension, this function will return x_next with shape [n, D]
        representing the next states and costs with shape [n, 1]
    '''
    # resample if requested
    if resample:
        n = x.shape[0]
        n = n.astype(theano.config.floatX)
        mx = x.mean(0)
        Sx = x.T.dot(x)/n - tt.outer(mx, mx)
        z = m_rng.normal(x.shape)
        x = mx + z.dot(tt.slinalg.cholesky(Sx).T)

    # convert angles from input states to their complex representation
    xa = utils.gTrig(x, angle_dims, D)

    # compute controls for each sample
    u = policy.evaluate(xa, symbolic=True, iid_per_eval=iid_per_eval, return_samples=True)

    # build state-control vectors
    xu = tt.concatenate([xa, u], axis=1)

    # predict the change in state given current state-control for each particle
    delta_x = dynmodel.predict_symbolic(xu, iid_per_eval=iid_per_eval, return_samples=True)
    
    # add measurement noise TODO: this should go inside the dynmodel predict_symbolic method
    sn = tt.exp(dynmodel.logsn)
    delta_x += m_rng.normal(x.shape).dot(tt.diag(sn))

    # compute the successor states
    x_next = x + delta_x

    return x_next

def rollout(x0, H, gamma0,
            policy, dynmodel, cost,
            D, angle_dims=None, **kwargs):
    ''' Given some initial state particles x0, and a prediction horizon H
    (number of timesteps), returns a set of trajectories sampled from the
    dynamics model and the discounted costs for each step in the
    trajectory.
    '''
    utils.print_with_stamp('Building computation graph for state particles propagation',
                           'pilco.rollout')

    # define internal scan computations
    def step_rollout(x, gamma, *args):
        '''
            Single step of rollout.
        '''
        # get next state distribution
        x_next = propagate_particles(x, policy, dynmodel, D, angle_dims, **kwargs)

        #  get cost of applying action:
        c_next = cost(x_next, None)
        return [gamma*c_next, x_next, gamma*gamma0]

    # these are the shared variables that will be used in the scan graph.
    # we need to pass them as non_sequences here
    # see: http://deeplearning.net/software/theano/library/scan.html
    shared_vars = []
    shared_vars.extend(dynmodel.get_all_shared_vars())
    shared_vars.extend(policy.get_all_shared_vars())
    # loop over the planning horizon
    output = theano.scan(fn=step_rollout,
                         outputs_info=[None, x0, gamma0],
                         non_sequences=[gamma0]+shared_vars,
                         n_steps=H,
                         strict=True,
                         allow_gc=False,
                         name="mc_pilco>rollout_scan")
    rollout_output, rollout_updts = output
    costs, trajectories = rollout_output[:2]

    # first axis: batch, second axis: time step
    costs = costs.T
    # first axis; batch, second axis: time step
    trajectories = trajectories.transpose(1, 0, 2)

    costs.name = 'c_list'
    trajectories.name = 'x_list'

    return [costs, trajectories], rollout_updts

def get_loss(policy, dynmodel, cost, D, angle_dims, n_samples=10,
             intermediate_outs=False, resample_particles=True,
             resample_dynmodel=False):
    '''
        Constructs the computation graph for the value function according to the
        mcpilco algorithm:
        1) sample x0 from initial state distribution N(mx0,Sx0)
        2) propagate the state particles forward in time
            2a) compute controls for eachc particle
            2b) use dynamics model to predict next states for each particle
            2c) compute cost for each particle
        3) return the expected value of the sum of costs
        @param policy
        @param dynmodel
        @param cost
        @param D number of state dimensions, must be a python integer
        @param angle_dims angle dimensions that should be converted to complex
                          representation
        @return Returns a tuple of (outs, inps, updts). These correspond to the
                output variables, input variables and updates dictionary, if any.
                By default, the only output variable is the value.
    '''
    # initial state distribution
    mx0 = tt.vector('mx0')
    Sx0 = tt.matrix('Sx0')
    # prediction horizon
    H = tt.iscalar('H')
    # discount factor
    gamma0 = tt.scalar('gamma0')

    inps = [mx0, Sx0, H, gamma0]

    # draw initial set of particles
    z0 = m_rng.normal((n_samples, mx0.size))
    Lx0 = tt.slinalg.cholesky(Sx0)
    x0 = mx0 + z0.dot(Lx0.T)

    # get rollout output
    r_outs, updts = rollout(x0, H, gamma0,
                            policy, dynmodel, cost,
                            D, angle_dims,
                            resample=resample_particles,
                            iid_per_eval=resample_dynmodel)

    costs = r_outs[0]
    mean_costs = costs.mean(0) # mean over particles

    # loss is E_{dynmodels}((1/H)*sum c(x_t))
    #          = (1/H)*sum E_{x_t}(c(x_t))
    loss = mean_costs.mean()

    if intermediate_outs:
        return [loss]+list(r_outs), inps, updts
    else:
        return loss, inps, updts
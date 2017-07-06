# pylint: disable=C0103
import theano
import theano.tensor as tt
from kusanagi import utils
from kusanagi.ghost import regression

def propagate_belief(mx, Sx, policy, dynmodel, D, angle_dims=None):
    ''' Given the input variables mx (tt.vector) and Sx (tt.matrix),
        representing the mean and variance of the system's state x, this function returns
        the next state distribution, and the mean and variance of the immediate cost. This
        is done by 1) evaluating the current policy 2) using the dynamics model to estimate
        the next state. The immediate cost is returned as a distribution Normal(mcost,Scost),
        since the state is uncertain.

        This implementation is based on Theano, thus all operations are assumed
        to be symbolic; i.e. we are constructing a computation graph.

        @param mx mean of state distribution
        @param Sx variance of state distribution
        @param dynmodel dynamics model compatible with moment matching
        @param policy Interface to the policy operations, compatible with moment matching
        @param cost cost function, compatible with moment matching
    '''
    if angle_dims is None:
        angle_dims = []

    # convert angles from input distribution to their complex representation
    mxa, Sxa, Ca = utils.gTrig2(mx, Sx, angle_dims, D)

    # compute distribution of control signal
    mu, Su, Cu = policy.evaluate(mxa, Sxa, symbolic=True)

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

    # SSGP and BNN return C_delta as the input-output covariance. All the others do it as
    # (input covariance)^-1 dot (input-output covariance)
    if isinstance(dynmodel, regression.SSGP) or\
       isinstance(dynmodel, regression.BNN):
        Sxu_deltax = C_deltax
    else:
        Sxu_deltax = Sxu.dot(C_deltax)

    if Ca is not None:
        Da = D+len(angle_dims)
        Dna = D-len(angle_dims)
        non_angle_dims = list(set(range(D)).difference(angle_dims))
        # this contains the covariance between the previous state (with angles as [sin,cos]),
        # and the next state (with angles in radians)
        Sxa_deltax = Sxu_deltax[:Da]
        # first come the non angle dimensions  [D-len(angi)] x [D]
        sxna_deltax = Sxa_deltax[:Dna]
        # then angles as [sin,cos]             [2*len(angi)] x [D]
        sxsc_deltax = Sxa_deltax[Dna:]
        #here we undo the [sin,cos] parametrization for the angle dimensions
        Sx_sc = Sx.dot(Ca)[angle_dims]
        Sa = Sxa[Dna:, Dna:]#+1e-12*tt.eye(2*len(angle_idims))
        sxa_deltax = Sx_sc.dot(tt.slinalg.solve(Sa, sxsc_deltax))
        # now we create Sx_deltax and fill it with the appropriate values
        # (i.e. in the correct order)
        Sx_deltax = tt.zeros((D, D))
        Sx_deltax = tt.set_subtensor(Sx_deltax[non_angle_dims, :], sxna_deltax)
        Sx_deltax = tt.set_subtensor(Sx_deltax[angle_dims, :], sxa_deltax)
    else:
        Sx_deltax = Sxu_deltax[:D]

    Sx_next = Sx + S_deltax + Sx_deltax + Sx_deltax.T

    # check if dynamics model has an updates dictionary
    updates = theano.updates.OrderedUpdates()

    return [mx_next, Sx_next], updates

def rollout(mx0, Sx0, H, gamma0,
            policy, dynmodel, cost,
            D, angle_dims=None):
    ''' Given some initial state distribution Normal(mx0,Sx0), and a prediction horizon H
    (number of timesteps), returns the predicted state distribution and discounted cost for
    every timestep. The discounted cost is returned as a distribution, since the state
    is uncertain.'''
    utils.print_with_stamp('Building computation graph for belief state propagation',
                           'pilco.rollout')

    # define internal scan computations
    def step_rollout(mx, Sx, gamma, *args):
        '''
            Single step of rollout.
        '''
        # get next state distribution
        b_out, updates = propagate_belief(mx, Sx, policy, dynmodel, D, angle_dims)
        mx_next, Sx_next = b_out

        #  get cost of applying action:
        mcost, Scost = cost(mx_next, Sx_next)
        gamma0 = args[0]
        next_v = [gamma*mcost, tt.square(gamma)*Scost, mx_next, Sx_next, gamma*gamma0]
        return next_v, updates

    # these are the shared variables that will be used in the graph.
    # we need to pass them as non_sequences here
    # (see: http://deeplearning.net/software/theano/library/scan.html)
    shared_vars = [gamma0]
    shared_vars.extend(dynmodel.get_all_shared_vars())
    shared_vars.extend(policy.get_all_shared_vars())

    # create the nodes that return the result from scan
    rollout_output, updts = theano.scan(fn=step_rollout,
                                        outputs_info=[None, None, mx0, Sx0, gamma0],
                                        non_sequences=shared_vars,
                                        n_steps=H,
                                        strict=True,
                                        allow_gc=False,
                                        name="pilco>rollout_scan")

    mean_costs, var_costs, mean_states, cov_states = rollout_output[:4]

    mean_costs.name = 'mc_list'
    var_costs.name = 'Sc_list'
    mean_states.name = 'mx_list'
    cov_states.name = 'Sx_list'

    return [mean_costs, var_costs, mean_states, cov_states], updts

def get_loss(policy, dynmodel, cost, D, angle_dims, intermediate_outs=False):
    '''
        Constructs the computation graph for the value function according to the
        pilco algorithm:
        1) get initial state distribution N(mx0,Sx0)
        2) propagate the state distribution forward in time
            2a) compute control distribution
            2b) compute next state distribution
            2c) compute cost distribution
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

    # get rollout output
    r_outs, updts = rollout(mx0, Sx0, H, gamma0,
                            policy, dynmodel, cost,
                            D, angle_dims)

    mean_costs = r_outs[0]

    # loss is E_{dynmodels}((1/H)*sum c(x_t))
    #          = (1/H)*sum E_{x_t}(c(x_t))
    loss = mean_costs.mean()

    if intermediate_outs:
        return [loss]+list(r_outs), inps, updts
    else:
        return loss, inps, updts

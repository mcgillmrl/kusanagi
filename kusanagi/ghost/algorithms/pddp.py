# pylint: disable=C0103
import numpy as np
import theano
import theano.tensor as tt
from kusanagi import utils
from kusanagi.ghost import regression


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


def propagate_belief(mx, Sx, u, dynmodel, D, angle_dims=None):
    ''' Given the input variables mx (tt.vector) and Sx (tt.matrix),
        representing the mean and variance of the system's state x, this
        function returns the next state distribution, and the mean and
        variance of the immediate cost. This is done by using the dynamics
        model to estimate the next state.
        The immediate cost is returned as a distribution Normal(mcost,Scost),
        since the state is uncertain.

        This implementation is based on Theano, thus all operations are assumed
        to be symbolic; i.e. we are constructing a computation graph.

        @param mx mean of state distribution
        @param Sx variance of state distribution
        @param u deterministic control action
        @param policy Interface to the policy operations, compatible with
               moment matching
        @param cost cost function, compatible with moment matching
    '''
    if angle_dims is None:
        angle_dims = []
    if isinstance(angle_dims, list) or isinstance(angle_dims, tuple):
        angle_dims = np.array(angle_dims, dtype=np.int32)
    D = mx.size

    # convert angles from input distribution to their complex representation
    mxa, Sxa, Ca = utils.gTrig2(mx, Sx, angle_dims)

    # compute distribution of control signal (note, control is deterministic,
    # the notation is just for convenience)
    mu = u
    Su = tt.zeros((mu.size, mu.size))
    Cu = tt.zeros((D, mu.size))

    # compute state control joint distribution
    mxu = tt.concatenate([mxa, mu])
    q = Sxa.dot(Cu)  # TODO this is a multiplication with a zero matrix
    Sxu_up = tt.concatenate([Sxa, q], axis=1)
    Sxu_lo = tt.concatenate([q.T, Su], axis=1)
    Sxu = tt.concatenate([Sxu_up, Sxu_lo], axis=0)  # [D+U]x[D+U]

    #  predict the change in state given current state-action
    # C_deltax = inv (Sxu) dot Sxu_deltax
    m_deltax, S_deltax, C_deltax = dynmodel.predict(mxu, Sxu)

    # compute the successor state distribution
    mx_next = mx + m_deltax

    # SSGP and BNN return C_delta as the input-output covariance. All the
    # others do it as (input covariance)^-1 dot (input-output covariance)
    if isinstance(dynmodel, regression.SSGP) or\
       isinstance(dynmodel, regression.BNN):
        Sxu_deltax = C_deltax
    else:
        Sxu_deltax = Sxu.dot(C_deltax)

    idx = tt.arange(D)
    non_angle_dims = (1-tt.eq(idx, angle_dims[:, None])).prod(0).nonzero()[0]
    Da = D+angle_dims.size
    Dna = D-angle_dims.size
    # this contains the covariance between the previous state (with angles
    # as [sin,cos]), and the next state (with angles in radians)
    Sxa_deltax = Sxu_deltax[:Da]
    # first come the non angle dimensions  [D-len(angle_dims)] x [D]
    sxna_deltax = Sxa_deltax[:Dna]
    # then angles as [sin,cos]             [2*len(angle_dims)] x [D]
    sxsc_deltax = Sxa_deltax[Dna:]
    # here we undo the [sin,cos] parametrization for the angle dimensions
    Sx_sc = Sx.dot(Ca)[angle_dims]
    Sa = Sxa[Dna:, Dna:]
    sxa_deltax = Sx_sc.dot(tt.slinalg.solve(Sa, sxsc_deltax))
    # now we create Sx_deltax and fill it with the appropriate values
    # (i.e. in the correct order)
    Sx_deltax = tt.zeros((D, D))
    Sx_deltax = tt.set_subtensor(Sx_deltax[non_angle_dims, :], sxna_deltax)
    Sx_deltax = tt.set_subtensor(Sx_deltax[angle_dims, :], sxa_deltax)

    Sx_next = Sx + S_deltax + Sx_deltax + Sx_deltax.T

    # check if dynamics model has an updates dictionary
    updates = theano.updates.OrderedUpdates()

    return [mx_next, Sx_next], updates


def rollout(mx0, Sx0,
            Z_nom, U_nom, I, L,
            dynmodel, cost,
            D, angle_dims=None):
    ''' Given some initial state distribution Normal(mx0,Sx0), and a
    prediction horizon H (number of timesteps), returns the predicted state
    distribution and discounted cost for every timestep. The discounted cost
    is returned as a distribution, since the state is uncertain.'''
    msg = 'Building computation graph for belief state propagation'
    utils.print_with_stamp(msg, 'pilco.rollout')

    # define internal scan computations
    def forward_step(z, z_nom, u_nom, L_, I_, *args):
        '''
            Single step of rollout.
        '''
        # get controls from local linear policy
        u = u_nom + I_ + L_.dot(z-z_nom)

        # split z into the mean and covariance of the state
        mx, Sx, triu_indices = unwrap_belief(z, D)

        # get next state distribution
        b_out, updates = propagate_belief(mx, Sx, u, dynmodel,
                                          D, angle_dims)
        mx_next, Sx_next = b_out

        # build belief vector
        z_next = wrap_belief(mx_next, Sx_next, triu_indices)

        #  get cost of applying action:
        mcost, Scost = cost(mx_next, Sx_next)
        next_v = [z_next, u]
        return next_v, updates

    # these are the shared variables that will be used in the graph.
    # we need to pass them as non_sequences here
    # (see: http://deeplearning.net/software/theano/library/scan.html)
    shared_vars = dynmodel.get_all_shared_vars()

    z0 = wrap_belief(mx0, Sx0, np.triu_indices(D))

    # create the nodes that return the result from scan
    rollout_output, updts = theano.scan(fn=forward_step,
                                        sequences=[Z_nom, U_nom, L, I],
                                        outputs_info=[z0],
                                        non_sequences=shared_vars,
                                        strict=True,
                                        allow_gc=False,
                                        name="pddp>rollout_scan")

    z, u = rollout_output[:2]

    return [z, u], updts


def build_pddp(dynmodel, cost, D, angle_dims=None):
    # initial state distribution
    mx0 = tt.vector('mx0')
    Sx0 = tt.matrix('Sx0')

    # locally linear time varying controller
    # u[t] = u_nom[t] + I[t] + L[t].dot(z - z_nom[t])
    u_nom = tt.matrix('u_nom')
    z_nom = tt.matrix('z_nom')
    I = tt.matrix('I')
    L = tt.tensor3('L')

    [z, u], updts = rollout(mx0, Sx0, z_nom, u_nom, I, L,
                            dynmodel, cost, D, angle_dims)

    return [z, u], updts

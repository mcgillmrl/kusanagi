import numpy as np

from functools import partial
from lasagne import nonlinearities
from matplotlib import pyplot as plt

from kusanagi.ghost import regression, control, optimizers, ExperienceDataset
from kusanagi.shell import cartpole, double_cartpole
from kusanagi import utils


def plot_rollout(rollout_fn, exp, *args, **kwargs):
    fig = kwargs.get('fig')
    axarr = kwargs.get('axarr')
    loss, costs, trajectories = rollout_fn(*args)
    n_samples, T, dims = trajectories.shape

    if fig is None or axarr is None:
        fig, axarr = plt.subplots(dims, sharex=True)
    exp_states = np.array(exp.states)
    for d in range(dims):
        axarr[d].clear()
        st = trajectories[:, :, d]
        # plot predictive distribution
        for i in range(n_samples):
            axarr[d].plot(
                np.arange(T-1), st[i, :-1], color='steelblue', alpha=0.3)
        # for i in range(len(exp.states)):
        #    axarr[d].plot(
        #         np.arange(T-1), exp_states[i,1:,d],
        #         color='orange', alpha=0.3)
        # plot experience
        axarr[d].plot(
            np.arange(T-1), np.array(exp.states[-1])[1:H, d], color='red')
        axarr[d].plot(
            np.arange(T-1), st[:, :-1].mean(0), color='orange')
    plt.show(block=False)
    plt.waitforbuttonpress(0.1)

    return fig, axarr


def check_task_learned(rollout_fn, *args, **kwargs):
    '''
    From Deisenroth's PhD thesis page 61
    '''
    loss, costs, trajectories = rollout_fn(*args)
    return costs.mean(0)[-1] < 0.2


# function to execute before applying policy
def gTrig(state, angle_dims=[]):
    return utils.gTrig_np(state, angle_dims).flatten()


def setup_pilco_experiment(params, pol, dynmodel_class=regression.SSGP_UI):
    # initial state distribution
    p0 = params['state0_dist']
    D = p0.mean.size

    # init policy
    if pol is None:
        pol = control.RBFPolicy(**params['policy'])

    # init dynmodel
    dyn = dynmodel_class(**params['dynamics_model'])

    # create experience dataset
    exp = ExperienceDataset()

    # init policy optimizer
    polopt = optimizers.ScipyOptimizer(**params['optimizer'])

    return p0, D, pol, dyn, exp, polopt


def setup_mc_pilco_experiment(params, pol=None,
                              pol_spec=None, dyn_spec=None):
    # initial state distribution
    p0 = params['state0_dist']
    D = p0.mean.size

    # init policy
    if pol is None:
        pol = control.NNPolicy(p0.mean, **params['policy'])
        if pol_spec is None:
            pol_spec = regression.mlp(
                input_dims=pol.D,
                output_dims=pol.E,
                hidden_dims=[50]*2,
                p=0.05, p_input=0.0,
                nonlinearities=nonlinearities.rectify,
                output_nonlinearity=pol.sat_func,
                dropout_class=regression.DenseDropoutLayer,
                name=pol.name)
        pol.network = pol.build_network(pol_spec)

    # init dynmodel
    dyn = regression.BNN(**params['dynamics_model'])
    if dyn_spec is None:
        dyn_spec = regression.dropout_mlp(
            input_dims=dyn.D,
            output_dims=dyn.E,
            hidden_dims=[200]*2,
            p=0.1, p_input=0.1,
            nonlinearities=nonlinearities.rectify,
            dropout_class=regression.DenseLogNormalDropoutLayer,
            name=dyn.name)
    dyn.network = dyn.build_network(dyn_spec)

    # create experience dataset
    exp = ExperienceDataset()

    # init policy optimizer
    params['optimizer']['min_method'] = 'adam'
    params['optimizer']['max_evals'] = 1000
    polopt = optimizers.SGDOptimizer(**params['optimizer'])

    return p0, D, pol, dyn, exp, polopt


def setup_cartpole_experiment(params=None):
    # get experiment parameters
    if params is None:
        params = cartpole.default_params()

    # init environment
    env = cartpole.Cartpole(**params['plant'])

    # init cost model
    cost = partial(cartpole.cartpole_loss, **params['cost'])

    return env, cost, params


def pilco_cartpole_experiment(params=None, policy=None,
                              dynmodel_class=regression.SSGP_UI):
    # init cartpole specific objects
    env, cost, params = setup_cartpole_experiment(params)

    # init policy and dynamics model
    p0, D, pol, dyn, exp, polopt = setup_pilco_experiment(
        params, policy, dynmodel_class)

    return p0, D, env, pol, dyn, cost, exp, polopt


def mcpilco_cartpole_experiment(params=None, policy=None):
    # init cartpole specific objects
    env, cost, params = setup_cartpole_experiment(params)

    # init policy and dynamics model
    p0, D, pol, dyn, exp, polopt = setup_mc_pilco_experiment(params, policy)

    return p0, D, env, pol, dyn, cost, exp, polopt


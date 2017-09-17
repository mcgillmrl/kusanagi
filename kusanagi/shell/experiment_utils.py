import numpy as np

from functools import partial
from lasagne import nonlinearities
from matplotlib import pyplot as plt

from kusanagi import utils
from kusanagi.ghost import (algorithms, regression, control, optimizers)
from kusanagi.shell import cartpole, double_cartpole
from kusanagi.base import apply_controller, train_dynamics, ExperienceDataset


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


def setup_pilco_experiment(params, pol=None, dyn=None):
    # initial state distribution
    p0 = params['state0_dist']
    D = p0.mean.size

    # init policy
    if pol is None:
        pol = control.RBFPolicy(**params['policy'])

    # init dynmodel
    if dyn is None:
        dynmodel_class = params.get('dynmodel_class', regression.SSGP_UI)
        dyn = dynmodel_class(**params['dynamics_model'])

    # create experience dataset
    exp = ExperienceDataset()

    # init policy optimizer
    polopt = optimizers.ScipyOptimizer(**params['optimizer'])

    # function for building the objective function
    build_loss_fn = algorithms.pilco.get_loss

    return p0, D, pol, dyn, exp, polopt, build_loss_fn


def setup_mc_pilco_experiment(params, pol=None, dyn=None):
    # initial state distribution
    p0 = params['state0_dist']
    D = p0.mean.size
    pol_spec = params.get('pol_spec', None)
    dyn_spec = params.get('dyn_spec', None)

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
    if dyn is None:
        dyn = regression.BNN(**params['dynamics_model'])
        if dyn_spec is None:
            odims = 2*dyn.E if dyn.heteroscedastic else dyn.E
            dyn_spec = regression.dropout_mlp(
                input_dims=dyn.D,
                output_dims=odims,
                hidden_dims=[200]*2,
                p=0.1, p_input=0.1,
                nonlinearities=nonlinearities.rectify,
                dropout_class=regression.DenseLogNormalDropoutLayer,
                name=dyn.name)
        dyn.network = dyn.build_network(dyn_spec)

    # create experience dataset
    exp = ExperienceDataset()

    # init policy optimizer
    polopt = optimizers.SGDOptimizer(**params['optimizer'])

    # function for building the objective function
    build_loss_fn = algorithms.mc_pilco.get_loss

    return p0, D, pol, dyn, exp, polopt, build_loss_fn


def setup_cartpole_experiment(params=None):
    # get experiment parameters
    if params is None:
        params = cartpole.default_params()

    # init environment
    env = cartpole.Cartpole(**params['plant'])

    # init cost model
    cost = partial(cartpole.cartpole_loss, **params['cost'])

    return env, cost, params


def pilco_cartpole_experiment(params=None, policy=None, dynmodel=None):
    # init cartpole specific objects
    env, cost, params = setup_cartpole_experiment(params)

    # init policy and dynamics model
    ret = setup_pilco_experiment(params, policy, dynmodel)
    p0, D, pol, dyn, exp, polopt, build_loss_fn = ret

    return p0, D, env, pol, dyn, cost, exp, polopt, build_loss_fn, params


def mcpilco_cartpole_experiment(params=None, policy=None, dynmodel=None):
    # init cartpole specific objects
    env, cost, params = setup_cartpole_experiment(params)

    # init policy and dynamics model
    ret = setup_mc_pilco_experiment(params, policy, dynmodel)
    p0, D, pol, dyn, exp, polopt, build_loss_fn = ret

    return p0, D, env, pol, dyn, cost, exp, polopt, build_loss_fn, params


def setup_double_cartpole_experiment(params=None):
    # get experiment parameters
    if params is None:
        params = cartpole.default_params()

    # init environment
    env = double_cartpole.DoubleCartpole(**params['plant'])

    # init cost model
    cost = partial(double_cartpole.double_cartpole_loss, **params['cost'])

    return env, cost, params


def pilco_double_cartpole_experiment(params=None, policy=None, dynmodel=None):
    # init cartpole specific objects
    env, cost, params = setup_double_cartpole_experiment(params)

    # init policy and dynamics model
    ret = setup_pilco_experiment(params, policy, dynmodel)
    p0, D, pol, dyn, exp, polopt, build_loss_fn = ret

    return p0, D, env, pol, dyn, cost, exp, polopt, build_loss_fn, params


def mcpilco_double_cartpole_experiment(params=None, policy=None, dynmodel=None):
    # init cartpole specific objects
    env, cost, params = setup_double_cartpole_experiment(params)

    # init policy and dynamics model
    ret = setup_mc_pilco_experiment(params, policy, dynmodel)
    p0, D, pol, dyn, exp, polopt, build_loss_fn = ret

    return p0, D, env, pol, dyn, cost, exp, polopt, build_loss_fn, params


def run_pilco_experiment(exp_setup=mcpilco_cartpole_experiment,
                         params=None, loss_kwargs={}, polopt_kwargs={},
                         extra_inps=[], step_cb=None, polopt_cb=None,
                         learning_iteration_cb=None,
                         render=False):
    # setup experiment
    exp_objs = exp_setup(params)
    p0, D, env, pol, dyn, cost, exp, polopt, build_loss_fn, params = exp_objs
    n_rnd = params.get('n_rnd', 1)
    n_opt = params.get('n_opt', 100)
    return_best = params.get('return_best', False)
    H = params['min_steps']
    gamma = params['discount']
    angle_dims = params['angle_dims']

    # init callbacks
    # callback executed after every call to env.step
    def step_cb_internal(state, action, cost, info):
        exp.add_sample(state, action, cost, info)
        if render:
            env.render()
        if callable(step_cb):
            step_cb(state, action, cost, info)

    def polopt_cb_internal(*args, **kwargs):
        if hasattr(dyn, 'update'):
            dyn.update()
        if hasattr(pol, 'update'):
            pol.update()
        if callable(polopt_cb):
            polopt_cb(*args, **kwargs)

    # function to execute before applying policy
    def gTrig(state):
        return utils.gTrig_np(state, angle_dims).flatten()

    # collect experience with random controls
    randpol = control.RandPolicy(maxU=pol.maxU)
    for i in range(n_rnd):
        exp.new_episode()
        apply_controller(env, randpol, H,
                         preprocess=gTrig,
                         callback=step_cb_internal)

    # 1. train dynamics once
    train_dynamics(dyn, exp, angle_dims=angle_dims)

    # build loss function
    loss, inps, updts = build_loss_fn(
        pol, dyn, cost, D, angle_dims, **loss_kwargs)

    # set objective of policy optimizer
    inps += extra_inps
    polopt.set_objective(loss, pol.get_params(symbolic=True),
                         inps, updts, **polopt_kwargs)

    # initial call so that the user gets the state before
    # the first learrning iteration
    if callable(learning_iteration_cb):
        learning_iteration_cb(exp, dyn, pol, polopt, params)

    for i in range(n_opt):
        total_exp = sum([len(st) for st in exp.states])
        msg = '==== Iteration [%d], experience: [%d steps] ===='
        utils.print_with_stamp(msg % (i+1, total_exp))

        # get initial state distribution (assumed gaussian)
        x0 = np.array([st[0] for st in exp.states])
        m0 = x0.mean(0)
        S0 = np.cov(x0, rowvar=False, ddof=1) +\
            1e-4*np.eye(x0.shape[1]) if len(x0) > 10 else p0.cov

        # 2. optimize policy
        polopt_args = [m0, S0, H, gamma]
        if isinstance(polopt, optimizers.SGDOptimizer):
            # check if we have a learning rate parameter
            lr = params.get('learning_rate', 1e-4)
            if callable(lr):
                lr = lr(i)
            polopt_args.append(lr)
        polopt.minimize(*polopt_args,
                        callback=polopt_cb_internal,
                        return_best=return_best)

        # 3. apply controller
        exp.new_episode(policy_params=pol.get_params(symbolic=False))
        apply_controller(env, pol, H,
                         preprocess=gTrig, callback=step_cb_internal)
        # 4. train dynamics once
        train_dynamics(dyn, exp, angle_dims=angle_dims)

        if callable(learning_iteration_cb):
            # user callback
            learning_iteration_cb(exp, dyn, pol, polopt, params)

    env.close()

def evaluate_policy(env, pol, exp, params, n_tests=100, render=False):
    H = params['min_steps']
    angle_dims = params['angle_dims']

    def gTrig(state):
        return utils.gTrig_np(state, angle_dims).flatten()

    def step_cb(*args, **kwargs):
        env.render()
    
    results = []
    for i,p in enumerate(exp.policy_parameters):
        utils.print_with_stamp('Evaluating policy at iteration %d'%(i))
        if p:
            pol.set_params(p)
        else:
            continue
        results_i = []
        for it in range(n_tests):
            ret = apply_controller(env, pol, H, preprocess=gTrig, callback=step_cb)
            results_i.append(ret)
        results.append(results_i)

    return results        
import numpy as np

from lasagne import nonlinearities
from matplotlib import pyplot as plt

from kusanagi import utils
from kusanagi.ghost import (algorithms, regression, control, optimizers)
from kusanagi.base import apply_controller, train_dynamics, ExperienceDataset


def plot_rollout(rollout_fn, exp, *args, **kwargs):
    fig = kwargs.get('fig')
    axarr = kwargs.get('axarr')
    name = kwargs.get('name', 'Rollout')
    n_exp = kwargs.get('n_exp', 0)
    ret = rollout_fn(*args)
    trajectories = m_states = None
    if len(ret) == 3:
        loss, costs, trajectories = ret
        n_samples, T, dims = trajectories.shape
    else:
        loss, m_costs, s_costs, m_states, s_states = ret
        T, dims = m_states.shape

    if fig is None or axarr is None:
        fig, axarr = plt.subplots(dims, num=name, sharex=True)

    exp_states = np.array(exp.states)
    for d in range(dims):
        axarr[d].clear()
        if trajectories is not None:
            st = trajectories[:, :, d]
            # plot predictive distribution
            for i in range(n_samples):
                axarr[d].plot(
                    np.arange(T), st[i, :], color='steelblue',
                    alpha=10.0/n_samples)
            axarr[d].plot(
                np.arange(T), st[:, :].mean(0), color='blue', linewidth=2)
        if m_states is not None:
            axarr[d].plot(
                np.arange(T), m_states[:, d], color='steelblue',
                alpha=0.3)
            axarr[d].errorbar(
                np.arange(T), m_states[:, d],
                1.96*np.sqrt(s_states[:, d, d]), color='steelblue', alpha=0.3)

        # plot experience
        exp_i = np.array(exp.states[-1])[:, d]
        T_exp = len(exp_i)
        total_exp = len(exp_states)
        for i in range(n_exp):
            axarr[d].plot(
                 np.arange(T_exp), exp_states[total_exp - n_exp + i][:, d],
                 color='orange', alpha=0.3)
        axarr[d].plot(
            np.arange(T_exp), exp_i, color='red')
        axarr[d].figure.canvas.draw()

    plt.show(False)
    plt.waitforbuttonpress(0.5)

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

    # module where get_loss and build_rollout are defined
    # (can also be a class)
    learner = algorithms.pilco

    return p0, pol, dyn, exp, polopt, learner, params


def setup_mc_pilco_experiment(params, pol=None, dyn=None):
    # initial state distribution
    p0 = params['state0_dist']
    D = p0.mean.size
    pol_spec = params.get('pol_spec', None)
    dyn_spec = params.get('dyn_spec', None)

    # init policy
    if pol is None:
        pol = control.NNPolicy(D, **params['policy'])
        if pol_spec is None:
            pol_spec = regression.mlp(
                input_dims=pol.D,
                output_dims=pol.E,
                hidden_dims=[200]*2,
                p=0.1, p_input=0.0,
                nonlinearities=nonlinearities.rectify,
                output_nonlinearity=pol.sat_func,
                dropout_class=regression.layers.DenseDropoutLayer,
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
                dropout_class=regression.layers.DenseLogNormalDropoutLayer,
                name=dyn.name)
        dyn.network = dyn.build_network(dyn_spec)

    # create experience dataset
    exp = ExperienceDataset()

    # init policy optimizer
    polopt = optimizers.SGDOptimizer(**params['optimizer'])

    # module where get_loss and build_rollout are defined
    # (can also be a class)
    learner = algorithms.mc_pilco

    return p0, pol, dyn, exp, polopt, learner, params


def run_pilco_experiment(env, cost, exp_setup=setup_mc_pilco_experiment,
                         params=None, loss_kwargs={}, polopt_kwargs={},
                         extra_inps=[], step_cb=None, minimize_cb=None,
                         learning_iteration_cb=None, max_dataset_size=0,
                         render=False, debug_plot=0):
    # setup experiment
    exp_objs = exp_setup(params)
    p0, pol, dyn, exp, polopt, learner, params = exp_objs
    n_rnd = params.get('n_rnd', 1)
    n_init = params.get('n_init', 0)
    if n_rnd == 0:
        n_init = 1
    n_opt = params.get('n_opt', 100)
    return_best = params.get('return_best', False)
    crn_dropout = params.get('crn_dropout', True)
    H = params.get('min_steps', 100)
    gamma = params.get('discount', 1.0)
    angle_dims = params.get('angle_dims', [])

    # init callbacks
    # callback executed after every call to env.step
    def step_cb_internal(state, action, cost, info):
        exp.add_sample(state, action, cost, info)
        if render:
            env.render()
        if callable(step_cb):
            step_cb(state, action, cost, info)

    def minimize_cb_internal(*args, **kwargs):
        if not crn_dropout:
            if hasattr(dyn, 'update'):
                dyn.update()
            if hasattr(pol, 'update'):
                pol.update()
        if callable(minimize_cb):
            minimize_cb(*args, **kwargs)

    # function to execute before applying policy
    def gTrig(state):
        return utils.gTrig_np(state, angle_dims).flatten()

    # collect experience with random controls
    randpol = control.RandPolicy(maxU=pol.maxU)
    for i in range(n_rnd):
        exp.new_episode()
        utils.print_with_stamp('Executing uniformly-random controls')
        apply_controller(env, randpol, H,
                         preprocess=gTrig,
                         callback=step_cb_internal)

    for i in range(n_init):
        exp.new_episode()
        utils.print_with_stamp('Executing initial policy')
        apply_controller(env, pol, H,
                         preprocess=gTrig,
                         callback=step_cb_internal)

    # 1. train dynamics once
    train_dynamics(
        dyn, exp, angle_dims=angle_dims, max_dataset_size=max_dataset_size)

    # build loss function
    loss, inps, updts = learner.get_loss(
        pol, dyn, cost, angle_dims, **loss_kwargs)

    outs = []
    if isinstance(loss, list):
        loss, outs = loss[0], loss[1:]

    rollout_fn = None
    if debug_plot > 0:
        # build rollout function for plotting
        loss_kwargs['mm_state'] = False
        rollout_fn = learner.build_rollout(
            pol, dyn, cost, angle_dims, **loss_kwargs)
        fig, axarr = None, None

    # set objective of policy optimizer
    inps += extra_inps
    polopt.set_objective(loss, pol.get_params(symbolic=True),
                         inps, updts, outs, **polopt_kwargs)

    # initial call so that the user gets the state before
    # the first learrning iteration
    if callable(learning_iteration_cb):
        learning_iteration_cb(exp, dyn, pol, polopt, params, rollout_fn)

    if crn_dropout:
        utils.print_with_stamp(
            'using common random numbers for dyn and pol', 'experiment_utils')
    else:
        utils.print_with_stamp(
            'resampling weights for dyn and pol', 'experiment_utils')
    for i in range(n_opt):
        total_exp = sum([len(st) for st in exp.states])
        msg = '==== Iteration [%d], experience: [%d steps] ===='
        utils.print_with_stamp(msg % (i+1, total_exp))
        if crn_dropout:
            if hasattr(dyn, 'update'):
                dyn.update()
            if hasattr(pol, 'update'):
                pol.update()

        # get initial state distribution (assumed gaussian)
        x0 = np.array([st[0] for st in exp.states])
        m0 = x0.mean(0)
        S0 = np.cov(x0, rowvar=False, ddof=1) +\
            1e-4*np.eye(x0.shape[1]) if len(x0) > 10 else p0.cov

        # 2. optimize policy
        minimize_args = [m0, S0, H, gamma]
        if isinstance(polopt, optimizers.SGDOptimizer):
            # check if we have a learning rate parameter
            lr = params.get('learning_rate', 1e-4)
            if callable(lr):
                lr = lr(i)
            minimize_args.append(lr)

        polopt.minimize(*minimize_args,
                        callback=minimize_cb_internal,
                        return_best=return_best)

        # 3. apply controller
        exp.new_episode(policy_params=pol.get_params(symbolic=False))
        apply_controller(env, pol, H,
                         preprocess=gTrig, callback=step_cb_internal)
        # 4. train dynamics once
        train_dynamics(dyn, exp, angle_dims=angle_dims,
                       max_dataset_size=max_dataset_size)

        if callable(learning_iteration_cb):
            # user callback
            learning_iteration_cb(exp, dyn, pol, polopt, params, rollout_fn)

        if debug_plot > 0:
            fig, axarr = plot_rollout(
                rollout_fn, exp, m0, S0, H, gamma, fig=fig, axarr=axarr)
    env.close()


def evaluate_policy(env, input_pol, exp, params, n_tests=100, render=False):
    H = params['min_steps']
    angle_dims = params['angle_dims']

    def gTrig(state):
        return utils.gTrig_np(state, angle_dims).flatten()

    def step_cb(*args, **kwargs):
        if render:
            env.render()

    results = []
    for i, p in enumerate(exp.policy_parameters):
        utils.print_with_stamp('Evaluating policy at iteration %d' % i)
        if p:
            input_pol.set_params(p)
            pol = input_pol
        else:
            pol = control.RandPolicy(maxU=input_pol.maxU)

        results_i = []
        for it in range(n_tests):
            ret = apply_controller(
                env, pol, H, preprocess=gTrig, callback=step_cb)
            results_i.append(ret)
        results.append(results_i)

    return results

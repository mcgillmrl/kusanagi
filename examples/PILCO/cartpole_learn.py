'''
Example of how to use the library for learning using the PILCO learner 
on the cartpole tas
'''
# pylint: disable=C0103
import os
import sys
import numpy as np
import theano

from kusanagi.ghost import control
from kusanagi.ghost import regression
from kusanagi.shell import cartpole
from kusanagi.ghost.algorithms import pilco, mc_pilco
from kusanagi.ghost.optimizers import ScipyOptimizer, SGDOptimizer
from kusanagi.base import apply_controller, train_dynamics, ExperienceDataset
from kusanagi import utils
from functools import partial
from matplotlib import pyplot as plt


# np.random.seed(1337)
np.set_printoptions(linewidth=500)


def plot_rollout(rollout_fn, *args, **kwargs):
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
            np.arange(T-1), st[:, :-1].mean(0), color='purple')
    plt.show(block=False)
    plt.waitforbuttonpress(0.1)

    return fig, axarr


if __name__ == '__main__':
    use_bnn_dyn = True
    use_bnn_pol = True

    # setup output directory
    utils.set_output_dir(os.path.join(utils.get_output_dir(), 'cartpole'))

    params = cartpole.default_params()
    n_rnd = 4                           # number of random initial trials
    n_opt = 100                         # learning iterations
    n_samples = 100                      # number of MC samples if bayesian nn
    learning_rate = 1e-3
    polyak_averaging = None
    H = params['max_steps']
    gamma = params['discount']
    angle_dims = params['angle_dims']

    # initial state distribution
    p0 = params['state0_dist']
    D = p0.mean.size

    # init environment
    env = cartpole.Cartpole(**params['plant'])

    # init policy
    pol = control.NNPolicy(p0.mean, **params['policy'])\
        if use_bnn_pol else control.RBFPolicy(**params['policy'])
    randpol = control.RandPolicy(maxU=pol.maxU)

    # init dynmodel
    dyn = regression.BNN(**params['dynamics_model'])\
        if use_bnn_dyn else regression.SSGP_UI(**params['dynamics_model'])

    # init cost model
    cost = partial(cartpole.cartpole_loss, **params['cost'])

    # create experience dataset
    exp = ExperienceDataset()

    # init policy optimizer
    if use_bnn_dyn:
        params['optimizer']['min_method'] = 'adam'
        params['optimizer']['max_evals'] = 1000
        polopt = SGDOptimizer(**params['optimizer'])
    else:
        polopt = ScipyOptimizer(**params['optimizer'])

    # callback executed after every call to env.step
    def step_cb(state, action, cost, info):
        exp.add_sample(state, action, cost, info)
        env.render()

    def polopt_cb(*args, **kwargs):
        if hasattr(dyn, 'update'):
            dyn.update()
        if hasattr(pol, 'update'):
            pol.update()

    # function to execute before applying policy
    def gTrig(state):
        return utils.gTrig_np(state, angle_dims).flatten()

    # during first n_rnd trials, apply randomized controls
    for i in range(n_rnd):
        exp.new_episode()
        apply_controller(env, randpol, H,
                         preprocess=gTrig,
                         callback=step_cb)

    # PILCO loop
    rollout_fn = None
    fig, axarr = None, None
    for i in range(n_opt):
        total_exp = sum([len(st) for st in exp.states])
        msg = '==== Iteration [%d], experience: [%d steps] ===='
        utils.print_with_stamp(msg % (i+1, total_exp))

        # train dynamics model
        train_dynamics(dyn, exp, angle_dims=angle_dims,
                       init_episode=max(0, i-10))

        # initial state distribution
        x0 = np.array([st[0] for st in exp.states])
        m0 = x0.mean(0)
        S0 = np.cov(x0, rowvar=False, ddof=1) +\
            1e-4*np.eye(x0.shape[1]) if len(x0) > 2 else p0.cov

        if fig is not None:
            # plot rollout
            fig, axarr = plot_rollout(
                rollout_fn, m0, S0, H, gamma, fig=fig, axarr=axarr)

        # train policy
        if polopt.loss_fn is None or dyn.should_recompile:
            loss_kwargs = {}
            obj_kwargs = {}
            extra_inps = []
            if use_bnn_dyn:
                # init learning rate parameter
                lr = theano.tensor.scalar('lr')
                extra_inps += [lr]

                # parameters for building loss function
                loss_kwargs['n_samples'] = n_samples
                loss_kwargs['resample_particles'] = True
                obj_kwargs['learning_rate'] = lr
                obj_kwargs['clip'] = 1.0
                obj_kwargs['polyak_averaging'] = polyak_averaging
                learner = mc_pilco
            else:
                learner = pilco

            # build loss function
            loss, inps, updts = learner.get_loss(
                pol, dyn, cost, D, angle_dims, **loss_kwargs)
            inps += extra_inps

            # set objective of policy optimizer
            polopt.set_objective(loss, pol.get_params(symbolic=True),
                                 inps, updts, **obj_kwargs)

            # build rollout function for plotting
            if rollout_fn is None:
                loss_kwargs['resample_particles'] = False
                rollout_fn = learner.build_rollout(
                    pol, dyn, cost, D, angle_dims, **loss_kwargs)

        polopt_args = [m0, S0, H, gamma]
        if use_bnn_dyn:
            polopt_args.append(learning_rate)
        polopt.minimize(*polopt_args,
                        callback=polopt_cb,
                        return_best=False)

        # apply controller
        exp.new_episode(policy_params=pol.get_params())
        apply_controller(env, pol, H,
                         preprocess=gTrig, callback=step_cb)

        # plot rollout
        fig, axarr = plot_rollout(
            rollout_fn, m0, S0, H, gamma, fig=fig, axarr=axarr)
    input('Finished training')
    sys.exit(0)

'''
Example of how to use the library for learning using the PILCO learner 
on the cartpole task
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

if __name__ == '__main__':
    use_bnn_dyn = True
    use_bnn_pol = True

    # setup output directory
    utils.set_output_dir(os.path.join(utils.get_output_dir(), 'cartpole'))

    params = cartpole.default_params()
    n_rnd = 2                           # number of random initial trials
    n_opt = 100                         # learning iterations
    H = 26  # params['max_steps']
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
            1e-7*np.eye(x0.shape[1]) if len(x0) > 2 else p0.cov

        # train policy
        if polopt.loss_fn is None or dyn.should_recompile:
            if use_bnn_dyn:
                # build loss function
                n_samples = 50
                loss, inps, updts = mc_pilco.get_loss(
                    pol, dyn, cost, D, angle_dims, n_samples=n_samples,
                    resample_particles=True, truncate_gradient=-1)

                # set objective of policy optimizer
                lr = theano.tensor.scalar('lr')
                polopt.set_objective(loss, pol.get_params(symbolic=True),
                                     inps+[lr], updts, learning_rate=lr)
            else:
                # build loss function
                loss, inps, updts = pilco.get_loss(
                    pol, dyn, cost, D, angle_dims)

                # set objective of policy optimizer
                polopt.set_objective(loss, pol.get_params(symbolic=True),
                                     inps, updts)
        if use_bnn_dyn:
            polopt.minimize(m0, S0, H, gamma, 1e-3,
                            callback=polopt_cb)
        else:
            polopt.minimize(m0, S0, H, gamma,
                            callback=polopt_cb)

        # apply controller
        exp.new_episode(policy_params=pol.get_params())
        apply_controller(env, pol, H,
                         preprocess=gTrig, callback=step_cb)

    input('Finished training')
    sys.exit(0)

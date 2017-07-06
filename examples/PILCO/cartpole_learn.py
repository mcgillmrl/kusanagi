'''
Example of how to use the library for learning using the PILCO learner on the cartpole task
'''
# pylint: disable=C0103
import os
import numpy as np

from kusanagi.ghost import control
from kusanagi.ghost import regression
from kusanagi.shell import cartpole
from kusanagi.ghost.algorithms import pilco_
from kusanagi.ghost.optimizers import ScipyOptimizer
from kusanagi.base import apply_controller, train_dynamics, ExperienceDataset
from kusanagi import utils
from functools import partial

#np.random.seed(1337)
np.set_printoptions(linewidth=500)

if __name__ == '__main__':
    # setup output directory
    utils.set_output_dir(os.path.join(utils.get_output_dir(), 'cartpole'))

    params = cartpole.default_params()
    n_rnd = 4                                                # number of random initial trials
    n_opt = 100                                              #learning iterations
    H = params['max_steps']
    gamma = params['discount']
    angle_dims = params['angle_dims']

    # initial state distribution
    p0 = params['state0_dist']
    D = p0.mean.size

    # init environment
    env = cartpole.Cartpole(**params['plant'])

    # init policy
    pol = control.RBFPolicy(**params['policy'])
    randpol = control.RandPolicy(maxU=pol.maxU)

    # init dynmodel
    dyn = regression.SSGP_UI(**params['dynamics_model'])

    # init cost model
    cost = partial(cartpole.cartpole_loss, **params['cost'])

    # create experience dataset
    exp = ExperienceDataset()

    # init policy optimizer
    polopt = ScipyOptimizer(**params['optimizer'])

    # callback executed after every call to env.step
    def step_cb(state, action, cost, info):
        exp.add_sample(state, action, cost, info)
        env.render()
    
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
        utils.print_with_stamp('==== Iteration [%d], experience: [%d steps] ===='%(i+1, total_exp))

        # train dynamics model
        train_dynamics(dyn, exp, angle_dims=angle_dims)

        # initial state distribution
        x0 = np.array([st[0] for st in exp.states])
        m0 = x0.mean(0)
        S0 = np.cov(x0.T).T if len(x0) > 2 else p0.cov

        # train policy
        if polopt.loss_fn is None or dyn.should_recompile:
            loss, inps, updts = pilco_.get_loss(pol, dyn, cost, D, angle_dims)
            polopt.set_objective(loss, pol.get_params(symbolic=True), inps, updts)
        polopt.minimize(m0, S0, H, gamma)

        # apply controller
        exp.new_episode(policy_params=pol.get_params())
        apply_controller(env, pol, H,
                         preprocess=gTrig, callback=step_cb)

    input('Finished training')
    sys.exit(0)

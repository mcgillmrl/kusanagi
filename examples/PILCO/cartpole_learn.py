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

np.random.seed(31337)
np.set_printoptions(linewidth=500)

if __name__ == '__main__':
    # setup output directory
    utils.set_output_dir(os.path.join(utils.get_output_dir(), 'cartpole'))

    params = cartpole.default_params()
    n_rnd = 4                                                # number of random initial trials
    n_opt = 100                                              #learning iterations
    max_steps = params['max_steps']
    angle_dims = params['angle_dims']

    # initial state distribution
    p0 = params['state0_dist']
    D = p0.mean.size

    # init environment
    env = cartpole.Cartpole(**params['plant'])

    # init policy
    pol = control.RBFPolicy(**params['policy'])
    randpol = control.RandPolicy(maxU=pol.maxU, random_walk=True)

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
        apply_controller(env, randpol, max_steps,
                         preprocess=gTrig,
                         callback=step_cb)

    # PILCO loop
    for i in range(n_opt):
        total_exp = sum([len(st) for st in exp.states])
        utils.print_with_stamp('Iteration %d, experience: %d steps'%(i+1, total_exp))

        # train dynamics model
        train_dynamics(dyn, exp, angle_dims=angle_dims)
    
        # train policy
        if polopt.loss_fn is None:
            loss, inps, updts = pilco_.get_loss(pol, dyn, cost, D, angle_dims)
            polopt.set_objective(loss, pol.get_params(symbolic=True), inps, updts)
        polopt.minimize(p0.mean, p0.cov, 40, 1)

        # apply controller
        exp.new_episode(policy_params=pol.get_params())
        apply_controller(env, pol, max_steps,
                         preprocess=gTrig, callback=step_cb)

    input('Finished training')
    sys.exit(0)

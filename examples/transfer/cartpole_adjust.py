import atexit
import argparse
import os, sys
import numpy as np
from functools import partial

import ghost
from ghost.regression import GP
from ghost import control
from ghost.learners.ExperienceDataset import ExperienceDataset
from ghost.transfer.trajectory_matching import TrajectoryMatching
from ghost.control import RBFPolicy
from ghost.control import AdjustedPolicy

from shell.cartpole import Cartpole, CartpoleDraw, cartpole_loss
import shell.cartpole

from shell.plant import SerialPlant

import utils
#np.random.seed(31337)
np.set_printoptions(linewidth=500)

if __name__ == '__main__':
    N = 100
    J = 100
    simulation = False
    base_dir = os.path.dirname(ghost.__file__).rsplit('/',1)[0]
    #source_dir = os.path.join(base_dir,'examples/learned_policies/cartpole')
    source_dir = os.path.join(base_dir,'/home/juancamilog/.kusanagi/output/cartpole_serial')
    target_dir = os.path.join(base_dir,'examples/learned_policies/target_90g_run_1')
    # SOURCE DOMAIN 
    utils.set_output_dir(source_dir)
    # load source experience
    #source_experience = ExperienceDataset(filename='PILCO_SSGP_UI_Cartpole_RBFPolicy_sat_dataset')
    source_experience = ExperienceDataset(filename='PILCO_SSGP_UI_SerialPlant_RBFPolicy_sat_dataset')
    #load source policy
    source_policy = RBFPolicy(filename='RBFPolicy_sat_5_1_cpu_float64')
    
    # TARGET DOMAIN
    utils.set_output_dir(target_dir)
    target_params = shell.cartpole.default_params()
    target_params['params']['max_evals'] = 125
    # policy
    target_params['dynmodel_class'] = GP.SSGP_UI
    target_params['invdynmodel_class'] = GP.GP_UI
    target_params['policy_class'] = AdjustedPolicy
    target_params['params']['policy']['adjustment_model_class'] = GP.GP
    #target_params['params']['policy']['adjustment_model_class'] = control.RBFPolicy
    #target_params['params']['policy']['n_basis'] = 20
    target_params['params']['policy']['sat_func'] = None # this is because we probably need bigger controls for heavier pendulums
    target_params['params']['policy']['max_evals'] = 5000
    target_params['params']['policy']['m0'] = np.zeros(source_policy.D+source_policy.E)
    target_params['params']['policy']['S0'] = 1e-2*np.eye(source_policy.E)

    # initialize target plant
    if not simulation:
        target_params['plant_class'] = SerialPlant
        target_params['params']['plant']['maxU'] = target_params['params']['policy']['maxU']
        target_params['params']['plant']['state_indices'] = [0,2,3,1]
        target_params['params']['plant']['baud_rate'] = 4000000
        target_params['params']['plant']['port'] = '/dev/ttyACM0'
    else:
        # TODO get these as command line arguments
        target_params['params']['plant']['params'] = {'l': 0.5, 'm': 1.5, 'M': 1.5, 'b': 0.1, 'g': 9.82}
        target_params['params']['cost']['pendulum_length'] = target_params['params']['plant']['params']['l']

    target_params['params']['source_policy'] = source_policy
    target_params['params']['source_experience'] = source_experience

    # initialize trajectory matcher
    tm = TrajectoryMatching(**target_params)
    atexit.register(tm.stop)
    draw_cp = CartpoleDraw(tm.plant)
    draw_cp.start()
    atexit.register(draw_cp.stop)

    for i in xrange(N):
        # sample target trajecotry
        tm.plant.reset_state()
        tm.apply_controller()
        
        if i <= J:
            # training the adjustment (supervised)
            tm.train_inverse_dynamics()
            tm.train_adjustment()

        n_episodes = len(tm.experience.states)
        if n_episodes > J:
            # fine tuning the adjustment (RL)
            tm.train_dynamics()
            tm.train_policy()

        tm.save()

    sys.exit(0)

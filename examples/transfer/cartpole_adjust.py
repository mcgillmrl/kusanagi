import atexit
import argparse
import os, sys
import numpy as np
from functools import partial

from kusanagi import ghost
from kusanagi.ghost import regression as kreg
from kusanagi.ghost import control
from kusanagi.ghost.learners.ExperienceDataset import ExperienceDataset
from kusanagi.ghost.transfer.trajectory_matching import TrajectoryMatching
from kusanagi.ghost.control import RBFPolicy
from kusanagi.ghost.control import AdjustedPolicy

from kusanagi.shell.cartpole import Cartpole, CartpoleDraw, cartpole_loss, default_params

from kusanagi.shell.plant import SerialPlant

from kusanagi import utils
#np.random.seed(31337)
np.set_printoptions(linewidth=500)

if __name__ == '__main__':
    N = 100
    J = 100
    simulation = True
    base_dir = os.path.dirname(ghost.__file__).rsplit('/',1)[0]
    #source_dir = os.path.join(base_dir,'examples/learned_policies/cartpole')
    source_dir = os.path.join(base_dir,'/home/juancamilog/.kusanagi/output/cartpole_0.5m/')
    #source_dir = os.path.join(base_dir,'examples/learned_policies/cartpole_serial')
    target_dir = os.path.join(base_dir,'examples/learned_policies/target_sim2robot')
    # SOURCE DOMAIN 
    utils.set_output_dir(source_dir)
    # load source experience
    source_experience = ExperienceDataset(filename='PILCO_SSGP_UI_Cartpole_RBFPolicy_sat_dataset')
    #source_experience = ExperienceDataset(filename='PILCO_SSGP_UI_SerialPlant_RBFPolicy_sat_dataset')
    #load source policy
    source_policy = RBFPolicy(filename='RBFPolicy_sat_5_1_cpu_float64')
    
    # TARGET DOMAIN
    utils.set_output_dir(target_dir)
    target_params = default_params()
    target_params['params']['H'] = 5.0                                               # control horizon
    target_params['params']['max_evals'] = 125
    # policy
    target_params['dynmodel_class'] = kreg.SSGP_UI
    target_params['invdynmodel_class'] = kreg.GP_UI
    target_params['params']['invdynmodel'] = {}
    target_params['params']['invdynmodel']['max_evals'] = 1000
    target_params['policy_class'] = AdjustedPolicy
    target_params['params']['policy']['adjustment_model_class'] = kreg.GP
    #target_params['params']['policy']['adjustment_model_class'] = control.RBFPolicy
    #target_params['params']['policy']['n_inducing'] = 20
    target_params['params']['policy']['sat_func'] = None # this is because we probably need bigger controls for heavier pendulums
    target_params['params']['policy']['max_evals'] = 5000
    target_params['params']['policy']['m0'] = np.zeros(source_policy.D+source_policy.E)
    target_params['params']['policy']['S0'] = 1e-2*np.eye(source_policy.E)

    # initialize target plant
    if not simulation:
        target_params['plant_class'] = SerialPlant
        target_params['params']['plant']['maxU'] = np.array(target_params['params']['policy']['maxU'])*(1.0/0.4)
        target_params['params']['plant']['state_indices'] = [0,2,3,1]
        target_params['params']['plant']['baud_rate'] = 4000000
        target_params['params']['plant']['port'] = '/dev/ttyACM0'
    else:
        # TODO get these as command line arguments
        target_params['params']['plant']['params'] = {'l': 0.6, 'm': 1.5, 'M': 1.5, 'b': 0.1, 'g': 9.82}
        target_params['params']['cost']['pendulum_length'] = target_params['params']['plant']['params']['l']

    target_params['params']['source_policy'] = source_policy
    target_params['params']['source_experience'] = source_experience

    # initialize trajectory matcher
    tm = TrajectoryMatching(**target_params)
    atexit.register(tm.stop)
    draw_cp = CartpoleDraw(tm.plant)
    draw_cp.start()
    atexit.register(draw_cp.stop)

    for i in range(N):
        # sample target trajecotry
        tm.plant.reset_state()
        tm.apply_controller()
        
        if i <= J:
            # training the adjustment (supervised)
            tm.train_inverse_dynamics()
            tm.train_adjustment()

        n_episodes = len(tm.experience.states)
        #if n_episodes > J:
            # fine tuning the adjustment (RL)
        #    tm.train_dynamics()
        #    tm.train_policy()

        tm.save()

    sys.exit(0)

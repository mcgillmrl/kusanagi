import os, sys
import numpy as np
from functools import partial

from ghost.regression.GPRegressor import SSGP_UI,SSGP
from ghost.learners.ExperienceDataset import ExperienceDataset
from ghost.transfer.trajectory_matching import TrajectoryMatching
from ghost.control import RBFPolicy
from ghost.control import AdjustedPolicy

from shell.cartpole import Cartpole, CartpoleDraw, cartpole_loss
from shell.plant import SerialPlant

import utils
#np.random.seed(31337)
np.set_printoptions(linewidth=500)

if __name__ == '__main__':
    N = 100
    # SOURCE DOMAIN 
    # load source experience
    utils.set_run_output_dir(os.path.join(utils.get_output_dir(),'cartpole_serial'))
    source_experience = ExperienceDataset(filename='PILCO_SSGP_UI_SerialPlant_RBFGP_sat_dataset')
    
    #load source policy
    source_policy = RBFPolicy(filename='RBFGP_sat_5_1_cpu_float64')
    
    # TARGET DOMAIN
    utils.set_run_output_dir(os.path.join(utils.get_output_dir(),'target_cartpole'))
    target_params = {}
    target_params['x0'] = [0,0,0,0]                                        # initial state mean
    target_params['S0'] = np.eye(4)*(0.1**2)                               # initial state covariance
    target_params['angle_dims'] = [3]                                      # angle dimensions
    target_params['H'] = 4.0                                               # control horizon
    target_params['discount'] = 1.0                                        # discoutn factor
    # policy
    policy_params = {}
    policy_params['maxU'] = [10]
    policy_params['adjustment_model_class'] = SSGP
    # initialize target plant
    plant_params = {}
    plant_params['dt'] = 0.1
    plant_params['params'] = {'l': 0.5, 'm': 0.5, 'M': 0.5, 'b': 0.1, 'g': 9.82}
    plant_params['noise'] = np.diag(np.ones(len(target_params['x0']))*0.01**2)   # model measurement noise (randomizes the output of the plant)
    #plant_params['maxU'] = policy_params['maxU']
    #plant_params['state_indices'] = [0,2,3,1]
    #plant_params['baud_rate'] = 4000000
    #plant_params['port'] = '/dev/ttyACM0'
    # dynamics model
    dynmodel_params = {}
    dynmodel_params['n_basis'] = 100
    # cost function
    cost_params = {}
    cost_params['target'] = [0,0,0,np.pi]
    cost_params['width'] = 0.25
    cost_params['expl'] = 0.0
    cost_params['pendulum_length'] = plant_params['params']['l']

    target_params['source_policy'] = source_policy
    target_params['source_experience'] = source_experience
    target_params['plant'] = plant_params
    target_params['policy'] = policy_params
    target_params['dynmodel'] = dynmodel_params
    target_params['cost'] = cost_params

    # initialize trajectory matcher
    #TODO let the trajectory matcher receive a plant as input instead of the source experrience
    tm = TrajectoryMatching(target_params, Cartpole, AdjustedPolicy, cartpole_loss, dynmodel_class=SSGP, viz_class=CartpoleDraw)

    # sample source trajectories (or load experience data)
    #tm.sample_trajectory_source(10)

    for i in xrange(N):
        # sample target trajecotry
        tm.apply_controller()

        tm.train_inverse_dynamics()
        tm.train_adjustment()

    sys.exit(0)

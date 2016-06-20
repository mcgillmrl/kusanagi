import atexit
import signal,sys
#sys.path.append('/home/adaptation/achatr/Desktop/Summer2016/PILCO_clone/kusanagi')
import numpy as np
from functools import partial
from ghost.regression.GPRegressor import SSGP_UI
from ghost.learners.PILCO import PILCO
from shell.plant import SerialPlant
from shell.cartpole import Cartpole, CartpoleDraw, cartpole_loss
from ghost.control import RBFPolicy
from utils import plot_results
#np.random.seed(31337)
np.set_printoptions(linewidth=500)

if __name__ == '__main__':
    # setup learner parameters
    # general parameters
    J = 100                                                                 # number of random initial trials
    maxU = [10]
    learner_params = {}
    learner_params['x0'] = [0,0,0,0]                                        # initial state mean
    learner_params['S0'] = np.eye(4)*(0.1**2)                               # initial state covariance
    learner_params['angle_dims'] = [3]                                      # angle dimensions
    learner_params['H'] = 1000.0                                               # control horizon
    learner_params['discount'] = 1.0                                        # discoutn factor
    # plant
    plant_params = {}
    plant_params['dt'] = 0.1
    plant_params['params'] = {'l': 0.5, 'm': 0.5, 'M': 0.5, 'b': 0.1, 'g': 9.82}
    plant_params['noise'] = np.diag(np.ones(len(learner_params['x0']))*0.01**2)   # model measurement noise (randomizes the output of the plant)
    plant_params['maxU'] = maxU
    plant_params['state_indices'] = [0,2,3,1]
    plant_params['baud_rate'] = 4000000
    plant_params['port'] = '/dev/ttyACM0'
    # policy
    policy_params = {}
    policy_params['m0'] = learner_params['x0']
    policy_params['S0'] = learner_params['S0']
    policy_params['n_basis'] = 10
    policy_params['maxU'] = maxU
    # dynamics model
    dynmodel_params = {}
    dynmodel_params['n_basis'] = 100
    # cost function
    cost_params = {}
    cost_params['target'] = [0,0,0,np.pi]
    cost_params['width'] = 0.25
    cost_params['expl'] = 0.0
    cost_params['pendulum_length'] = plant_params['params']['l']

    learner_params['plant'] = plant_params
    learner_params['policy'] = policy_params
    learner_params['dynmodel'] = dynmodel_params
    learner_params['cost'] = cost_params

    # initialize learner
    learner = PILCO(learner_params, SerialPlant, RBFPolicy, cartpole_loss, dynmodel_class=SSGP_UI)#,viz=CartpoleDraw)
    atexit.register(learner.stop)

    # gather data with random trials
    for i in xrange(J):
        learner.plant.reset_state()
        learner.apply_controller(learner_params['H'])
    
    sys.exit(0)

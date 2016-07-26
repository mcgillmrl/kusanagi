import atexit
import signal,sys,os
import numpy as np
import utils
from functools import partial
from ghost.regression.GP import SPGP_UI,SSGP_UI,GP_UI
from ghost.regression.NN import NN
from ghost.learners.PILCO import PILCO
from shell.cartpole import Cartpole, CartpoleDraw, cartpole_loss
from ghost.control import RBFPolicy, NNPolicy
from utils import plot_results
#np.random.seed(31337)
np.set_printoptions(linewidth=500)

def default_params():
    # setup output directory
    #utils.set_run_output_dir(os.path.join(utils.get_output_dir(),'cartpole'))
    # setup learner parameters
    # general parameters
    J = 4                                                                   # number of random initial trials
    N = 100                                                                 # learning iterations
    learner_params = {}
    learner_params['x0'] = [0,0,0,0]                                        # initial state mean
    learner_params['S0'] = np.eye(4)*(0.1**2)                               # initial state covariance
    learner_params['angle_dims'] = [3]                                      # angle dimensions
    learner_params['H'] = 4.0                                               # control horizon
    learner_params['discount'] = 1.0                                        # discoutn factor
    # plant
    plant_params = {}
    plant_params['dt'] = 0.1
    plant_params['params'] = {'l': 0.5, 'm': 0.5, 'M': 0.5, 'b': 0.1, 'g': 9.82}
    plant_params['noise'] = np.diag(np.ones(len(learner_params['x0']))*0.01**2)   # model measurement noise (randomizes the output of the plant)
    # policy
    policy_params = {}
    policy_params['m0'] = learner_params['x0']
    policy_params['S0'] = learner_params['S0']
    policy_params['n_basis'] = 10
    #policy_params['hidden_dims'] = [50,50,50]
    policy_params['maxU'] = [10]
    # dynamics model
    dynmodel_params = {}
    #dynmodel_params['n_basis'] = 100
    #dynmodel_params['hidden_dims'] = [50,50,50]
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

    return {'params': learner_params, 'plant_class': Cartpole, 'policy_class': RBFPolicy, 'cost_func': cartpole_loss, 'dynmodel_class': SSGP_UI}


if __name__ == '__main__':
    J = 4                                                                   # number of random initial trials
    N = 100                                                                 # learning iterations
    learner_params = default_params()
    # initialize learner
    learner = PILCO(**learner_params)#, viz_class=CartpoleDraw)
    #learner = PILCO(learner_params, Cartpole, NNPolicy, cartpole_loss, dynmodel_class=NN)#, viz_class=CartpoleDraw)
    atexit.register(learner.stop)

    if learner.experience.n_samples() == 0: #if we have no prior data
        # gather data with random trials
        for i in xrange(J):
            learner.plant.reset_state()
            learner.apply_controller(random_controls=True)
    else:
        # learner.plant.reset_state()
        # learner.apply_controller()
        
        # plot results
        learner.init_rollout(derivs=False)
        plot_results(learner)

    # learning loop
    for i in xrange(N):
        # train the dynamics models given the collected data
        learner.train_dynamics()

        # train policy
        learner.train_policy()

        # execute it on the robot
        learner.plant.reset_state()
        learner.apply_controller()

        # plot results
        plot_results(learner)

        # save latest state of the learner
        learner.save()
    
    sys.exit(0)

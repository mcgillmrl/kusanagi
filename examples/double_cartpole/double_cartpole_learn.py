import atexit
import signal,sys
import numpy as np
from functools import partial
from utils import plot_results
from ghost.regression.GPRegressor import SSGP_UI
from ghost.learners.PILCO import PILCO
from shell.double_cartpole import DoubleCartpole, DoubleCartpoleDraw, double_cartpole_loss
from ghost.control import RBFPolicy
#np.random.seed(31337)
np.set_printoptions(linewidth=500)

if __name__ == '__main__':
    # setup learner parameters
    # general parameters
    J = 2                                                                   # number of random initial trials
    N = 100                                                                 # learning iterations
    learner_params = {}
    learner_params['x0'] = [0,0,0,0,np.pi,np.pi]                            # initial state mean ( x, dx, dtheta1, dtheta2, theta1, theta2
    learner_params['S0'] = np.eye(6)*(0.1**2)                               # initial state covariance
    learner_params['angle_dims'] = [4,5]                                    # angle dimensions
    learner_params['H'] = 5.0                                               # control horizon
    learner_params['discount'] = 1.0                                        # discoutn factor
    # plant
    plant_params = {}
    plant_params['dt'] = 0.05
    plant_params['params'] = {'m1': 0.5, 'm2': 0.5, 'm3': 0.5, 'l2': 0.6, 'l3': 0.6, 'b': 0.1, 'g': 9.82}
    plant_params['noise'] = np.diag(np.ones(len(learner_params['x0']))*0.01**2)   # model measurement noise (randomizes the output of the plant)
    # policy
    policy_params = {}
    policy_params['m0'] = learner_params['x0']
    policy_params['S0'] = learner_params['S0']
    policy_params['n_basis'] = 200
    policy_params['maxU'] = [20]
    # dynamics model
    dynmodel_params = {}
    dynmodel_params['n_basis'] = 100
    # cost function
    cost_params = {}
    cost_params['target'] = [0,0,0,0,0,0]
    cost_params['width'] = 0.5
    cost_params['expl'] = 0.0
    cost_params['pendulum_lengths'] = [ plant_params['params']['l2'], plant_params['params']['l3'] ]

    learner_params['plant'] = plant_params
    learner_params['policy'] = policy_params
    learner_params['dynmodel'] = dynmodel_params
    learner_params['cost'] = cost_params

    # initialize learner
    learner = PILCO(learner_params, DoubleCartpole, RBFPolicy, double_cartpole_loss, dynmodel_class=SSGP_UI)#,viz=DoubleCartpoleDraw)
    atexit.register(learner.stop)

    if learner.experience.n_samples() == 0: #if we have no prior data
        # gather data with random trials
        for i in xrange(J):
            learner.plant.reset_state()
            learner.apply_controller(random_controls=True)
    else:
        learner.plant.reset_state()
        experience_data = learner.apply_controller()
        
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
        experience_data = learner.apply_controller()

        # plot results
        plot_results(learner)

        # save latest state of the learner
        learner.save()
    
    sys.exit(0)

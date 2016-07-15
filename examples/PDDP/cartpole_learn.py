import atexit
import signal,sys
import numpy as np
from functools import partial
from ghost.regression.GP import SSGP_UI, GP_UI
from ghost.learners.PDDP import PDDP
from ghost.cost import quadratic_loss
from shell.cartpole import Cartpole, CartpoleDraw, cartpole_loss
from ghost.control import LocalLinearPolicy
from utils import plot_results, gTrig2_np, gTrig_np
#np.random.seed(31347)
np.set_printoptions(linewidth=500)

if __name__ == '__main__':
    # setup learner parameters
    # general parameters
    J = 5                                                                   # number of random initial trials
    N = 100                                                                 # learning iterations
    learner_params = {}
    learner_params['x0'] = [0,0,0,0]                                        # initial state mean
    learner_params['S0'] = np.eye(4)*(0.1**2)                               # initial state covariance
    learner_params['angle_dims'] = []                                      # angle dimensions
    x0 = np.array(learner_params['x0'])[None,:]
    S0 = np.array(learner_params['S0'])[None,:,:]
    x0, S0 = gTrig2_np(x0,S0,np.array([3]),len(learner_params['x0']))
    learner_params['x0'], learner_params['S0'] = x0[0], S0[0]
    learner_params['H'] = 4.0                                             # control horizon
    learner_params['discount'] = 1.0                                        # discoutn factor
    learner_params['max_evals'] = 200
    # plant
    plant_params = {}
    plant_params['x0'] = [0,0,0,0]   
    plant_params['S0'] = np.eye(4)*(0.1**2)
    plant_params['dt'] = 0.1
    plant_params['angle_dims'] = [3]
    plant_params['params'] = {'l': 0.5, 'm': 0.5, 'M': 0.5, 'b': 0.1, 'g': 9.82}
    plant_params['noise'] = np.diag(np.ones(len(learner_params['x0']))*0.01**2)   # model measurement noise (randomizes the output of the plant)
    # policy
    policy_params = {}
    policy_params['maxU'] = [10]
    policy_params['m0'] = learner_params['x0']
    policy_params['H'] = learner_params['H']
    policy_params['dt'] = plant_params['dt']
    # dynamics model
    dynmodel_params = {}
    #dynmodel_params['n_basis'] = 100
    # cost function
    cost_params = {}
    cost_params['target'] = gTrig_np(np.array([0,0,0,np.pi])[None,:], [3])[0]
    cost_params['angi'] = learner_params['angle_dims']
    cost_params['D'] = len(cost_params['target'])
    Q = np.zeros((5,5))
    Q[0,0] = 1; Q[0,-2] = plant_params['params']['l']; Q[-2,0] = plant_params['params']['l']; Q[-2,-2] = plant_params['params']['l']**2; Q[-1,-1]=plant_params['params']['l']**2
    cost_params['Q'] = Q
    cost_params['width'] = 0.25
    cost_params['expl'] = 0.0
    cost_params['pendulum_length'] = plant_params['params']['l']

    learner_params['plant'] = plant_params
    learner_params['policy'] = policy_params
    learner_params['dynmodel'] = dynmodel_params
    learner_params['cost'] = cost_params

    # initialize learner
    learner = PDDP(learner_params, Cartpole, LocalLinearPolicy, quadratic_loss, dynmodel_class=GP_UI, viz_class=CartpoleDraw)
    atexit.register(learner.stop)

    learner.experience.reset()
    # gather data with random trials
    print "RUNNING RANDOM TRIALS"
    for i in xrange(J):
        learner.policy.t = 0
        learner.plant.reset_state()
        learner.apply_controller(random_controls=True)

    # learning loop
    for i in xrange(N):
        
        print str(i) + " iterations done. Paused program, press ENTER to continue."
        #raw_input()

        #train the dynamics models given the collected data
        learner.train_dynamics()

        # train policy
        learner.train_policy()

        # execute it on the robot
        learner.plant.reset_state()

        learner.save()        
        #DEBUGGING
        # u,z,a,b = learner.policy.get_params()
        # print u
        # print z
        # print a
        # print b
        #raw_input()
        learner.experience.reset()
        for i in xrange(J):
            learner.policy.t = 0
            experience_data = learner.apply_controller()

        # plot results
        #plot_results(learner) # TODO this does not work with PDDP

        # save latest state of the learner
        learner.save()
    
    sys.exit(0)

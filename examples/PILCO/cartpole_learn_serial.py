import atexit
import signal,sys,os
import numpy as np
import utils
from shell.cartpole import default_params
from shell.plant import SerialPlant
from ghost.learners.PILCO import PILCO
from ghost.regression.GP import SPGP_UI,SSGP_UI
from ghost.regression.NN import NN
from ghost.control import NNPolicy
from utils import plot_results
#np.random.seed(31337)
np.set_printoptions(linewidth=500)

if __name__ == '__main__':
    # setup output directory
    utils.set_run_output_dir(os.path.join(utils.get_output_dir(),'cartpole_serial'))

    J = 4                                                                   # number of random initial trials
    N = 100                                                                 # learning iterations
    learner_params = default_params()
    
    # initialize learner
    learner_params['dynmodel_class'] = SSGP_UI
    learner_params['params']['dynmodel']['n_basis'] = 100
    learner_params['plant_class'] = SerialPlant
    learner_params['params']['plant']['maxU'] = learner_params['params']['policy']['maxU']
    learner_params['params']['plant']['state_indices'] = [0,2,3,1]
    learner_params['params']['plant']['baud_rate'] = 4000000
    learner_params['params']['plant']['port'] = '/dev/ttyACM0'
    #learner_params['min_method'] = 'ADAM'
    #learner_params['dynmodel_class'] = NN
    #learner_params['params']['dynmodel']['hidden_dims'] = [100,100,100]
    learner = PILCO(**learner_params)
    try:
        learner.load(load_compiled_fns=True)
        save_compiled_fns = False
    except:
        utils.print_with_stamp('Unable to load compiled fns','main')
        save_compiled_fns = True

    atexit.register(learner.stop)

    if learner.experience.n_samples() == 0: #if we have no prior data
        # gather data with random trials
        for i in xrange(J-1):
            learner.plant.reset_state()
            learner.apply_controller(random_controls=True)
        learner.plant.reset_state()
        learner.apply_controller()
    else:
        learner.plant.reset_state()
        learner.apply_controller()
        
        # plot results
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
        learner.save(save_compiled_fns=save_compiled_fns)
        save_compiled_fns = False  # only need to save the compiled functions once
    
    sys.exit(0)

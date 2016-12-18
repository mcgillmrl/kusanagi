import atexit
import signal,sys,os
import numpy as np
from kusanagi import utils
from kusanagi.shell.plant import SerialPlant
from kusanagi.shell.cartpole import default_params
from kusanagi.ghost.learners.PILCO import PILCO
from kusanagi.ghost.regression import GP
from kusanagi.ghost.regression.NN import NN
from kusanagi.ghost.control import NNPolicy
from kusanagi.utils import plot_results
#np.random.seed(31337)
np.set_printoptions(linewidth=500)

if __name__ == '__main__':
    # setup output directory
    print utils.get_output_dir()
    print os.path.join(utils.get_output_dir(),'cartpole_serial')
    utils.set_output_dir(os.path.join(utils.get_output_dir(),'cartpole_serial'))

    J = 4                                                                   # number of random initial trials
    N = 100                                                                 # learning iterations
    learner_params = default_params()
    # initialize learner
    learner_params['dynmodel_class'] = GP.SSGP_UI
    learner_params['params']['dynmodel']['n_basis'] = 100
    learner_params['plant_class'] = SerialPlant
    learner_params['params']['plant']['maxU'] = np.array(learner_params['params']['policy']['maxU'])*1.0/0.4
    learner_params['params']['plant']['state_indices'] = [0,2,3,1]
    learner_params['params']['plant']['baud_rate'] = 4000000
    learner_params['params']['plant']['port'] = '/dev/ttyACM0'
    #learner_params['min_method'] = 'ADAM'
    #learner_params['dynmodel_class'] = NN
    #learner_params['params']['dynmodel']['hidden_dims'] = [100,100,100]
    learner = PILCO(**learner_params)
    learner.load()
    atexit.register(learner.stop)

    # gather data with random trials
    for i in xrange(J):
        learner.plant.reset_state()
        learner.apply_controller(10*learner_params['params']['H'])
    
    sys.exit(0)

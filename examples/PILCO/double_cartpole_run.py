import atexit
import signal,sys
import numpy as np

from kusanagi import utils
from kusanagi.shell.double_cartpole import default_params, DoubleCartpoleDraw
from kusanagi.ghost.learners.PILCO import PILCO
from kusanagi.ghost.regression import GP
from kusanagi.ghost.regression.NN import NN
from kusanagi.ghost.control import NNPolicy
from kusanagi.utils import plot_results

if __name__ == '__main__':
    #utils.set_output_dir(os.path.join(utils.get_output_dir(),'double_cartpole'))
    J = 2                                                                   # number of random initial trials
    N = 100                                                                 # learning iterations
    # setup learner parameters
    learner_params = default_params()
    # initialize learner
    learner_params['params']['use_empirical_x0'] = True
    learner_params['dynmodel_class'] = GP.SSGP_UI
    learner_params['params']['dynmodel']['n_basis'] = 105
    #learner_params['min_method'] = 'ADAM'
    #learner_params['dynmodel_class'] = NN
    #learner_params['params']['dynmodel']['hidden_dims'] = [100,100,100]
    learner = PILCO(**learner_params)
    learner.load(load_compiled_fns=False)
    atexit.register(learner.stop)
    draw_cdp = DoubleCartpoleDraw(learner.plant)
    draw_cdp.start()
    atexit.register(draw_cdp.stop)

    for i in xrange(N):
        # execute it on the robot
        learner.plant.reset_state()
        learner.apply_controller()

    sys.exit(0)

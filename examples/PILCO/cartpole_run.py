import atexit
import signal,sys,os
import numpy as np

from kusanagi import utils
from kusanagi.ghost.regression import GP, SSGP_UI
from kusanagi.ghost.algorithms.PILCO import PILCO
from kusanagi.shell.cartpole import CartpoleDraw, default_params
from kusanagi.utils import plot_results
from kusanagi.ghost.control import NNPolicy

from kusanagi.base.helpers import apply_controller
#np.random.seed(31337)
np.set_printoptions(linewidth=500)

if __name__ == '__main__':
    # setup output directory
    utils.set_output_dir(os.path.join(utils.get_output_dir(),'cartpole'))
    N = 100                                                                 # learning iterations
    learner_params = default_params()
    learner_params['params']['plant']['params']['l'] = .6
    learner_params['params']['cost']['pendulum_length'] = .6
    learner_params['policy_class'] = NNPolicy
    # initialize learner
    learner = PILCO(**learner_params)
    learner.policy.load()
    #learner.load(load_compiled_fns=False)
    atexit.register(learner.stop)
    #draw_cp = CartpoleDraw(learner.plant)
    #draw_cp.start()
    #atexit.register(draw_cp.stop)
    #learner.policy.predict_fn = None
    for i in range(N):
        apply_controller(learner.plant, learner)
    
    sys.exit(0)

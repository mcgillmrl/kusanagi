import atexit
import signal,sys,os
import numpy as np

from kusanagi import utils
from kusanagi.ghost.regression import GP, SSGP_UI
from kusanagi.ghost.algorithms.PILCO import PILCO
from kusanagi.ghost.control import NNPolicy
from kusanagi.shell.double_cartpole import default_params, DoubleCartpoleDraw
from kusanagi.utils import plot_results
#np.random.seed(31337)
np.set_printoptions(linewidth=500)

if __name__ == '__main__':
    # setup output directory
    utils.set_output_dir(os.path.join(utils.get_output_dir(),'double_cartpole'))
    N = 100                                                                 # learning iterations
    learner_params = default_params()
    learner_params['params']['H'] = 4.0
    learner_params['params']['plant']['dt'] = 0.05
    learner_params['policy_class'] = NNPolicy
    # initialize learner
    learner = PILCO(**learner_params)
    learner.load(load_compiled_fns=False)
    atexit.register(learner.stop)
    draw_cdp = DoubleCartpoleDraw(learner.plant)
    draw_cdp.start()
    atexit.register(draw_cdp.stop)

    for i in range(N):
        learner.plant.reset_state()
        learner.apply_controller()

    sys.exit(0)

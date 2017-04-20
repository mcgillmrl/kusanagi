import atexit,os,sys
import numpy as np
from kusanagi.ghost.regression import GP,GP_UI,SSGP,SSGP_UI
from kusanagi.ghost.learners.PDDP import PDDP
from kusanagi.ghost.cost import quadratic_loss
from kusanagi.ghost.control import LocalLinearPolicy
from kusanagi.shell.cartpole import default_params,CartpoleDraw
from kusanagi import utils
#np.random.seed(31347)
np.set_printoptions(linewidth=500)

if __name__ == '__main__':
    # setup output directory
    utils.set_output_dir(os.path.join(utils.get_output_dir(),'pddp_cartpole'))

    J = 1                                                                   # number of random initial trials
    N = 100                                                                 # learning iterations
    learner_params = default_params()
    learner_params['policy_class'] = LocalLinearPolicy
    learner_params['dynmodel_class'] = GP_UI
    # initialize learner
    learner = PDDP(**learner_params)
    try:
        learner.load(load_compiled_fns=True)
        save_compiled_fns = False
    except:
        utils.print_with_stamp('Unable to load compiled fns','main')
        save_compiled_fns = True

    atexit.register(learner.stop)
    draw_cp = CartpoleDraw(learner.plant)
    draw_cp.start()
    atexit.register(draw_cp.stop)
    
    # apply random controls to obtain initial experience
    for i in range(J):
        learner.policy.t = 0
        learner.plant.reset_state()
        learner.apply_controller(random_controls=True)

    # learning loop
    for i in range(N):
        #train the dynamics models given the collected data
        learner.train_dynamics()

        # train policy
        learner.train_policy()
        
        # execute it on the robot
        # once to obtain a new nominal trajectory
        learner.policy.t = 0
        learner.policy.noise = 0
        learner.plant.reset_state()
        learner.apply_controller()
        # J-1 times to obtain randomized data
        for i in range(J-1):
            learner.policy.t = 0
            learner.policy.noise = 1e-1*np.array(learner.policy.maxU)
            learner.plant.reset_state()
            learner.apply_controller()

        # save latest state of the learner
        #learner.save()
    
    sys.exit(0)

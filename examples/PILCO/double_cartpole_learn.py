import atexit
import signal,sys,os
import numpy as np
import kusanagi.ghost.regression as kreg

from kusanagi import utils
from kusanagi.shell.double_cartpole import default_params, DoubleCartpoleDraw
from kusanagi.ghost.learners.PILCO import PILCO, MC_PILCO
from kusanagi.ghost.control import BNNPolicy
from kusanagi.utils import plot_results

#np.random.seed(31337)
np.set_printoptions(linewidth=500)

if __name__ == '__main__':
    # setup output directory
    utils.set_output_dir(os.path.join(utils.get_output_dir(),'double_cartpole'))
    
    use_bnn = True
    J = 2                                                                  # number of random initial trials
    N = 100                                                                 # learning iterations
    learner_params = default_params()
    # initialize learner
    learner_params['params']['use_empirical_x0'] = True
    learner_params['params']['realtime'] = False
    learner_params['params']['H'] = 4.0
    learner_params['params']['plant']['dt'] = 0.1
    learner_params['params']['plant']['params']['l'] = .5
    learner_params['params']['cost']['pendulum_length'] = .5

    if not use_bnn:
        # gp based PILCO
        learner_params['dynmodel_class'] = kreg.SSGP_UI
        learner_params['params']['dynmodel']['n_inducing'] = 100
        learner = PILCO(**learner_params)
    else:
        # dropout network (BNN) based PILCO
        learner_params['params']['min_method'] = 'ADAM'
        learner_params['params']['learning_rate'] = 5e-3
        learner_params['params']['max_evals'] = 1000
        learner_params['dynmodel_class'] = kreg.BNN

        learner = MC_PILCO(**learner_params)

    try:
        learner.load(load_compiled_fns=True)
        save_compiled_fns = False
    except:
        utils.print_with_stamp('Unable to load compiled fns','main')
        save_compiled_fns = True

    atexit.register(learner.stop)
    #draw_cdp = DoubleCartpoleDraw(learner.plant)
    #draw_cdp.start()
    #atexit.register(draw_cdp.stop)

    if learner.experience.n_samples() == 0: #if we have no prior data
        # gather data with random trials
        for i in xrange(J):
            learner.plant.reset_state()
            learner.apply_controller(random_controls=True)
        #learner.plant.reset_state()
        #learner.apply_controller()
    else:
        last_pp = learner.experience.policy_parameters[-1]
        current_pp = learner.policy.get_params(symbolic=False)
        should_run = True
        for lastp,curp in zip(last_pp,current_pp):
            should_run = should_run and not np.allclose(lastp,curp)

        if should_run:
            learner.plant.reset_state()
            learner.apply_controller()
        
        # plot results
        plot_results(learner)

    # learning loop
    for i in xrange(N):
        # train the dynamics models given the collected data
        if use_bnn:
            learner.train_dynamics(max_episodes=10)
        else:
            learner.train_dynamics()

        # plot results with new dynamics
        plot_results(learner)

        # train policy
        learner.train_policy()

        # execute it on the robot
        learner.plant.reset_state()
        learner.apply_controller()

        # plot results with new policy
        plot_results(learner)

        # save latest state of the learner
        learner.save(save_compiled_fns=save_compiled_fns)
        save_compiled_fns = False  # only need to save the compiled functions once
    
    sys.exit(0)

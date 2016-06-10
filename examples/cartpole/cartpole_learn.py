import atexit
import signal,sys
import numpy as np
from functools import partial
from ghost.learners.PILCO import PILCO
from shell.cartpole import Cartpole, CartpoleDraw, cartpole_loss
from ghost.control import RBFPolicy
from utils import plot_results

if __name__ == '__main__':
    #np.random.seed(31337)
    np.set_printoptions(linewidth=500)
    # initliaze plant
    dt = 0.1                                                         # simulation time step
    model_parameters ={}                                             # simulation parameters
    model_parameters['l'] = 0.5
    model_parameters['m'] = 0.5
    model_parameters['M'] = 0.5
    model_parameters['b'] = 0.1
    model_parameters['g'] = 9.82
    x0 = [0,0,0,0]                                                   # initial state mean
    S0 = np.eye(4)*(0.1**2)                                          # initial state covariance
    maxU = [10]
    measurement_noise = np.diag(np.ones(len(x0))*0.01**2)            # model measurement noise (randomizes the output of the plant)
    plant = Cartpole(model_parameters,x0,S0,dt,measurement_noise)
    #draw_cp = CartpoleDraw(plant,0.033)                              # initializes visualization
    #draw_cp.start()
    #atexit.register(draw_cp.stop)

    # initialize policy
    angle_dims = [3]
    policy = RBFPolicy(x0,S0,maxU,10, angle_dims)

    # initialize cost function
    cost_parameters = {}
    cost_parameters['angle_dims'] = angle_dims
    cost_parameters['target'] = [0,0,0,np.pi]
    cost_parameters['width'] = 0.25
    cost_parameters['expl'] = 0.0
    cost_parameters['pendulum_length'] = model_parameters['l']
    cost = partial(cartpole_loss, params=cost_parameters)

    # initialize learner
    H = 4.0                                                          # controller horizon
    H_steps = np.ceil(H/dt)
    J = 4                                                            # number of random initial trials
    N = 100                                                           # learning iterations
    learner = PILCO(plant, policy, cost, angle_dims)#,viz=CartpoleDraw)
    atexit.register(learner.stop)

    if learner.dynamics_model.X_ is None: #if we have no prior data
        # gather data with random trials
        for i in xrange(J):
            plant.reset_state()
            learner.apply_controller(H=H,random_controls=True)
    else:
        plant.reset_state()
        experience_data = learner.apply_controller(H=H)
        
        # plot results
        learner.init_rollout(derivs=False)
        plot_results(learner)

    # learning loop
    for i in xrange(N):
        # train the dynamics models given the collected data
        learner.train_dynamics()

        # train policy
        learner.train_policy(H=H)

        # execute it on the robot
        plant.reset_state()
        experience_data = learner.apply_controller(H=H)

        # plot results
        plot_results(learner)

        # save latest state of the learner
        learner.save()
    
    sys.exit(0)

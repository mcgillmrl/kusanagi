import signal,sys
import numpy as np
from functools import partial
from ghost.learners.PILCO import PILCO
from shell.double_cartpole import DoubleCartpole, DoubleCartpoleDraw, double_cartpole_loss
from ghost.control import RBFPolicy

if __name__ == '__main__':
    #np.random.seed(31337)
    np.set_printoptions(linewidth=500)
    # initliaze plant
    dt = 0.05                                                         # simulation time step
    model_parameters ={}                                             # simulation parameters
    model_parameters['m1'] = 0.5
    model_parameters['m2'] = 0.5
    model_parameters['m3'] = 0.5
    model_parameters['l2'] = 0.6
    model_parameters['l3'] = 0.6
    model_parameters['b'] = 0.1
    model_parameters['g'] = 9.82
    x0 = [0,0,0,0,np.pi,np.pi]                                               # initial state mean ( x, dx, dtheta1, dtheta2, theta1, theta2
    S0 = np.eye(6)*(0.1**2)                                          # initial state covariance
    measurement_noise = np.diag(np.ones(len(x0))*0.01**2)            # model measurement noise (randomizes the output of the plant)
    plant = DoubleCartpole(model_parameters,x0,S0,dt,measurement_noise)
    draw_dcp = DoubleCartpoleDraw(plant,0.033)                              # initializes visualization
    draw_dcp.start()

    # initialize policy
    angle_dims = [4,5]
    policy = RBFPolicy(x0,S0,[20],100, angle_dims)

    # initialize cost function
    cost_parameters = {}
    cost_parameters['angle_dims'] = angle_dims
    cost_parameters['target'] = [0,0,0,0,0,0]
    cost_parameters['width'] = 0.5
    cost_parameters['expl'] = 0.0
    cost_parameters['pendulum_lengths'] = [model_parameters['l2'],model_parameters['l3']]
    cost = partial(double_cartpole_loss, params=cost_parameters)

    # initialize learner
    T = 100.0                                                          # controller horizon
    J = 500                                                            # number of random initial trials
    N = 15                                                           # learning iterations
    learner = PILCO(plant, policy, cost, angle_dims, async_plant=False)

    def signal_handler(signal, frame):                               # initialize signal handler to capture ctrl-c
        print 'Caught CTRL-C!'
        draw_dcp.stop()
        plant.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    # gather data with random trials
    for i in xrange(J):
        plant.reset_state()
        learner.apply_controller(H=T)

    draw_dcp.stop()

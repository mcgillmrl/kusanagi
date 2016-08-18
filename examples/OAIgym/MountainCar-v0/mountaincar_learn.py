import signal,sys
sys.path.append('/home/adaptation/achatr/Desktop/Summer2016/PILCO_clone/kusanagi')
import gym
import numpy as np
from functools import partial
from ghost.learners.PILCO import PILCO
from shell.cartpole import Cartpole, CartpoleDraw, cartpole_loss
from shell.plant import SerialPlant
from ghost.control import RBFPolicy
from examples.OAIgym.new_plant import Plant, OAIPlant

if __name__ == '__main__':
    #np.random.seed(31337)
    np.set_printoptions(linewidth=500)
    # initliaze plant
    dt = 0.02                                                         # simulation time step
    model_parameters ={}                                             # simulation parameters
    model_parameters['l'] = 0.5                                      
    model_parameters['m'] = 0.1
    model_parameters['M'] = 1
    model_parameters['b'] = 0.1
    model_parameters['g'] = 9.8
    x0 = [0,0,0,0]
    S0 = np.eye(4)*(0.1**2)                                          # initial state covariance
    maxU = [10]
    measurement_noise = np.diag(np.ones(len(x0))*0.01**2)            # model measurement noise (randomizes the output of the plant)


    env = gym.make('CartPole-v0')                                   # creates the cartpole visualization and environment
    discrete = False
    env.render()
    plant = OAIPlant(discrete, model_parameters,x0,S0, dt, measurement_noise)
    plant.setEnv(env)
    x0, _ = plant.get_plant_state()
    # initialize policy
    angle_dims = [3]                                                
    policy = RBFPolicy(x0,S0,maxU,10, angle_dims)

    # initialize cost function
    cost_parameters = {}
    cost_parameters['angle_dims'] = angle_dims
    cost_parameters['target'] = [0,0,0,0]                           
    cost_parameters['width'] = 0.25
    cost_parameters['expl'] = 0.0
    cost_parameters['pendulum_length'] = model_parameters['l']
    cost = partial(cartpole_loss, params=cost_parameters)

    # initialize learner
    T = 4.0                                                          # controller horizon
    J = 2                                                            # number of random initial trials
    N = 100                                                           # learning iterations
    learner = PILCO(plant, policy, cost, angle_dims, async_plant=False)
    
    def signal_handler(signal, frame):                               # initialize signal handler to capture ctrl-c
        print 'Caught CTRL-C!'
        #draw_cp.stop()
        #plant.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    if learner.dynamics_model.X_ is None: #if we have no prior data
        # gather data with random trials
        for i in xrange(J):
            plant.reset_state()
            learner.apply_controller(H=T,random_controls=True)
 #   else:
 #       plant.reset_state()
 
    plant.reset_state()
    learner.apply_controller(H=T)
        
    for i in xrange(N):
        # train the dynamics models given the collected data
        learner.train_dynamics()

        # train policy
        learner.train_policy(H=T)

        # execute it on the robot564,
        plant.reset_state()
        learner.apply_controller(H=T)

        # save latest state of the learner
        learner.save()
    
    draw_cp.stop()

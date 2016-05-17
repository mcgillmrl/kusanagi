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
from shell.double_cartpole import DoubleCartpole, DoubleCartpoleDraw, double_cartpole_loss_openAI as double_cartpole_loss

def plot_results(learner):
    # plot last run cost vs predicted cost
    plt.figure('Cost of last run and Predicted cost')
    plt.gca().clear()
    cost = np.array(learner.experience.immediate_cost[-1])[1:,0]
    rollout_ =  learner.rollout(x0,S0,H_steps,1)
    plt.errorbar(np.arange(0,T,dt),rollout_[0],yerr=2*np.sqrt(rollout_[1]))
    plt.plot(np.arange(0,T,dt),cost)

    states = np.array(learner.experience.states[-1])[1:]
    predicted_means = np.array(rollout_[2])
    predicted_vars = np.array(rollout_[3])
    
    for d in xrange(learner.mx0.size):
        plt.figure('Last run vs Predicted rollout %d'%(d))
        plt.gca().clear()
        plt.errorbar(np.arange(0,T,dt),predicted_means[:,d],yerr=2*np.sqrt(predicted_vars[:,d,d]))
        plt.plot(np.arange(0,T,dt),states[:,d])

    plt.show(False)
    plt.pause(0.05)

if __name__ == '__main__':
    #np.random.seed(31337)
    np.set_printoptions(linewidth=500)
    # initliaze plant
    dt = 0.02                                                         # simulation time step
    model_parameters ={}                                             # simulation parameters
    model_parameters['m1'] = 0.5
    model_parameters['m2'] = 0.5
    model_parameters['m3'] = 0.5
    model_parameters['l2'] = 0.6#
    model_parameters['l3'] = 0.6#
    model_parameters['b'] = 0.1
    model_parameters['g'] = 9.82
    x0 = [0,1,0,1,0,0,0,0,0,0,0] #                                              # initial state mean ( x, dx, dtheta1, dtheta2, theta1, theta2
    S0 = np.eye(11)*(0.1**2) 
    measurement_noise = np.diag(np.ones(len(x0))*0.01**2) 

    env = gym.make('InvertedDoublePendulum-v1')                                   # creates the cartpole visualization and environment
                                           
    discrete = False
    env.render()  

    plant = OAIPlant(discrete, model_parameters,x0,S0,dt,measurement_noise)
    plant.setEnv(env)
    x0, _ = plant.get_state()
    # initialize policy
    angle_dims = [] #
    policy = RBFPolicy(x0,S0,[1],200, angle_dims)

    # initialize cost function
    cost_parameters = {}
    cost_parameters['angle_dims'] = angle_dims
    cost_parameters['target'] = [0,1,0,1,0,0,0,0,0,0,0] #
    cost_parameters['width'] = 0.5
    cost_parameters['expl'] = 0.0
    cost_parameters['pendulum_lengths'] = [model_parameters['l2'],model_parameters['l3']]
    cost = partial(double_cartpole_loss, params=cost_parameters)

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
        experience_data = learner.apply_controller(H=T)

        # plot results
        plot_results(learner)

        # save latest state of the learner
        learner.save()
    
    sys.exit(0)
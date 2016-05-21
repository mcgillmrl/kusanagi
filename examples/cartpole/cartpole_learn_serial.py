import atexit
import signal,sys
import numpy as np
from matplotlib import pyplot as plt
from functools import partial
from ghost.learners.PILCO import PILCO
from shell.cartpole import Cartpole, CartpoleDraw, cartpole_loss
from shell.plant import SerialPlant
from ghost.control import RBFPolicy

def plot_results(learner):
    # plot last run cost vs predicted cost
    plt.figure('Cost of last run and Predicted cost')
    plt.gca().clear()
    T_range = np.arange(0,T+dt,dt)
    cost = np.array(learner.experience.immediate_cost[-1])[:,0]
    rollout_ =  learner.rollout(x0,S0,H_steps,1)
    plt.errorbar(T_range,rollout_[0],yerr=2*np.sqrt(rollout_[1]))
    plt.plot(T_range,cost)

    states = np.array(learner.experience.states[-1])
    predicted_means = np.array(rollout_[2])
    predicted_vars = np.array(rollout_[3])
    
    for d in xrange(learner.mx0.size):
        plt.figure('Last run vs Predicted rollout %d'%(d))
        plt.gca().clear()
        plt.errorbar(T_range,predicted_means[:,d],yerr=2*np.sqrt(predicted_vars[:,d,d]))
        plt.plot(T_range,states[:,d])

    plt.show(False)
    plt.pause(0.05)

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
    #plant = Cartpole(model_parameters,x0,S0,dt,measurement_noise)
    plant = SerialPlant(model_parameters,x0,S0,dt,measurement_noise,state_indices=[0,2,3,1],maxU=maxU,baud_rate=4000000,port='/dev/ttyACM0')
    draw_cp = CartpoleDraw(plant,0.033)                              # initializes visualization
    draw_cp.start()

    atexit.register(plant.stop)
    atexit.register(draw_cp.stop)

    # initialize policy
    angle_dims = [3]
    policy = RBFPolicy(x0,S0,maxU,20, angle_dims)

    # initialize cost function
    cost_parameters = {}
    cost_parameters['angle_dims'] = angle_dims
    cost_parameters['target'] = [0,0,0,np.pi]
    cost_parameters['width'] = 0.25
    cost_parameters['expl'] = 0.0
    cost_parameters['pendulum_length'] = model_parameters['l']
    cost = partial(cartpole_loss, params=cost_parameters)

    # initialize learner
    T = 4.0                                                          # controller horizon
    H_steps = np.ceil(T/dt)
    J = 4                                                            # number of random initial trials
    N = 100                                                          # learning iterations
    learner = PILCO(plant, policy, cost, angle_dims, async_plant=False)

    if learner.dynamics_model.X_ is None: #if we have no prior data
        # gather data with random trials
        for i in xrange(J):
            plant.reset_state()
            learner.apply_controller(H=T,random_controls=True)
    else:
        #TODO make this an option when running the script from the command line
        plant.reset_state()
        learner.apply_controller(H=T)
        
        # plot results
        learner.init_rollout(derivs=False)
        plot_results(learner)
        
    for i in xrange(N):
        # train the dynamics models given the collected data
        learner.train_dynamics()

        # train policy
        learner.train_policy(H=T)

        # execute it on the robot
        plant.reset_state()
        learner.apply_controller(H=T)
        
        # plot results
        plot_results(learner)

        # save latest state of the learner
        learner.save()
    
    draw_cp.stop()

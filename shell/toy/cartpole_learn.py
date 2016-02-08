from ghost.learners.PILCO import PILCO
from shell.toy.cartpole import Cartpole, CartpoleDraw, CartpoleCost
from ghost.control import RandPolicy, RBFPolicy
import numpy as np
from threading import Thread
from time import sleep,time
from util import augment
import utils
from utils import print_with_stamp, get_compiled_gTrig,gTrig_np
import sys
from scipy.io import savemat

if __name__ == '__main__':
    np.random.seed(31337)
    np.set_printoptions(linewidth=200, precision=16, suppress=True)
    dt = 0.1
    T = 4.0
    J = 5
    model_parameters ={}
    model_parameters['l'] = 0.5
    model_parameters['m'] = 0.5
    model_parameters['M'] = 0.5
    model_parameters['b'] = 0.1
    model_parameters['g'] = 9.82

    x0 = [0,0,0,0]
    S0 = np.eye(4)*(0.1**2)
    target = [0,0,0,np.pi]
    angle_dims = [3]
    measurement_noise = np.diag(np.ones(len(x0))*0.01**2)

    # initialize policy
    p1 = RandPolicy([10])
    p2 = RBFPolicy(x0,S0,[10],10, angle_dims)
    p2.model.set_loghyp(np.log(np.tile(np.array([1,1,1,0.7,0.7,1,0.01]),(p2.maxU.size,1))))
    # initialize cost
    cost = CartpoleCost(target,model_parameters['l'], angle_dims)
    # init plant and setup handler for ctrl-c
    plant = Cartpole(model_parameters,x0,S0,dt,measurement_noise)
    draw_cp = CartpoleDraw(plant,0.033)
    draw_cp.start()
    import signal
    def signal_handler(signal, frame):
        draw_cp.stop()
        plant.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    # initialize learner
    learner = PILCO(plant, p2, cost, angle_dims, async_plant=False)
    
    # gather data by applying the controller
    for i in xrange(J):
        plant.reset_state()
        learner.apply_controller(H=T)
    
    # train the dynamics models given the collected data
    #learner.policy = p2
    learner.train_dynamics()

    # estimate the value of the current policy
    learner.value(H=4)
    #learner.train_policy()
    
    # saving dataset for external tests
    x = np.array(learner.experience.states)
    u = np.array(learner.experience.actions)

    np.savez('experience.npz',x=x,u=u)

    savemat('experience.mat',{'x':x,'u':u})

    draw_cp.stop()

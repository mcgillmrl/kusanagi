import sys
import numpy as np
from functools import partial

from ghost.transfer.trajectory_matching import TrajectoryMatching
from ghost.control import RBFPolicy

from shell.cartpole import Cartpole, CartpoleDraw, cartpole_loss
from shell.plant import SerialPlant

from utils import print_with_stamp

if __name__ == '__main__':
    # SOURCE DOMAIN PARAMETERS
    # initialize plant
    source_dt = 0.1 
    source_parameters ={}
    source_parameters['l'] = 0.5
    source_parameters['m'] = 0.5
    source_parameters['M'z] = 0.5
    source_parameters['b'] = 0.1
    source_parameters['g'] = 9.82
    source_x0 = [0,0,0,0]                                                   # initial state mean
    source_S0 = np.eye(4)*(0.1**2)                                          # initial state covariance
    source_maxU = [10]
    source_measurement_noise = np.diag(np.ones(len(x0))*0.01**2)            # model measurement noise (randomizes the output of the plant)
    source_plant = Cartpole(source_parameters,source_x0,source_S0,source_dt,source_measurement_noise)

    # load learned source policy
    angle_dims = [3]
    source_policy = RBFPolicy(x0,S0,maxU,10, angle_dims, name='cartpole_source') #TODO remove unnecessary parameters from the input arguments
    if not source_policy.model.trained:
        print_with_stamp('Need a trained policy to start transfer','main')
        sys.exit(0)
    
    # initialize cost function
    cost_parameters = {}
    cost_parameters['angle_dims'] = angle_dims
    cost_parameters['target'] = [0,0,0,np.pi]
    cost_parameters['width'] = 0.25
    cost_parameters['expl'] = 0.0
    cost_parameters['pendulum_length'] = 0.5
    cost = partial(cartpole_loss, params=cost_parameters)

    # TARGET DOMAIN PARAMETERS
    dt = 0.1                                                         # simulation time step
    model_parameters ={}                                             # simulation parameters
    model_parameters['l'] = 0.5
    model_parameters['m'] = 0.5
    model_parameters['M'] = 0.5
    model_parameters['b'] = 0.1
    model_parameters['g'] = 9.82
    measurement_noise = np.diag(np.ones(len(x0))*0.01**2)            # model measurement noise (randomizes the output of the plant)
    target_plant = SerialPlant(model_parameters,x0,S0,dt,measurement_noise,state_indices=[0,2,3,1],maxU=maxU,baud_rate=4000000,port='/dev/ttyACM0')
   
    # initialize trajectory matcher
    tm = TrajectoryMatching(target_plant=target_plant,source_policy=source_policy, cost=cost) # TODO give the option of using other adjusttment models

    # TODO Load source experience (or use the plant to directly sample trajectories?

    # sample source trajectories (or load experience data)
    tm.sample_trajectory_source(10)

    for i in xrange(N):
        # sample target trajecotry
        tm.sample_trajectory_target(1)


    sys.exit(0)
    '''

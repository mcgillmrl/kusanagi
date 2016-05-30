import numpy as np
from ghost.control import RBFPolicy

if __name__ == '__main__':
    N = 10
    x0 = [0,0,0,0]                                                   # initial state mean
    S0 = np.eye(4)*(0.1**2)                                          # initial state covariance
    maxU = [10]
    angle_dims = [3]
    # load learned source policy
    angle_dims = [3]
    policy = RBFPolicy(x0,S0,maxU,10, angle_dims, name='cartpole_source') #TODO remove unnecessary parameters from the input arguments
    # if no source policy available, learn it
    print policy.model.trained
    
    # TODO
    '''
    # initialize trajectory matcher
    tm = TrajecotryMatching(source_policy=source_policy) # TODO give the option of using other adjusttment models
    
    # sample source trajectories (or load experience data)
    tm.sample_trajectory_source(10)

    for i in xrange(N):
        # sample target trajecotry
        tm.sample_trajectory_target(1)


    sys.exit(0)
    '''

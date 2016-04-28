import signal,sys
import numpy as np
from time import time,sleep
from functools import partial
from ghost.learners.PILCO import PILCO
from shell.plant import SerialPlant, LivePlot
from ghost.control import RBFPolicy
from utils import gTrig_np

if __name__ == '__main__':
    #np.random.seed(31337)
    np.set_printoptions(linewidth=500)
    # initliaze plant
    dt = 0.02                                                      # simulation time step
    model_parameters ={}                                             # simulation parameters
    x0 = [0,0,0,0]                                                   # initial state mean
    S0 = np.eye(4)*(0.1**2)                                          # initial state covariance
    maxU = [10]
    measurement_noise = np.diag(np.ones(len(x0))*0.01**2)            # model measurement noise (randomizes the output of the plant)
    plant = SerialPlant(model_parameters,x0,S0,dt,measurement_noise,state_indices=[0,2,3,1],maxU=maxU,baud_rate=4000000,port='/dev/ttyACM0')
    plot = LivePlot(plant,H=100,angi=[0,3])
    plot.start()

    def signal_handler(signal, frame):                               # initialize signal handler to capture ctrl-c
        print 'Caught CTRL-C!'
        plot.stop()
        plant.stop()
        sys.exit(0)

    try:
        w = 2*np.pi*1.5;
        A = 0;
        plant.reset_state()
        while True:
            exec_time = time()
            #u_t = A*w*np.cos(w*(time()+0.5*dt))[None]
            u_t = A*np.cos(w*time())[None]
            #if u_t >= 0:
            #    u_t[0] = A
            #else:
            #    u_t[0] = -A
            plant.apply_control(u_t)
            plant.step()
            exec_time = time() - exec_time
            sleep(max(dt-exec_time,0))
    except:
        print 'Caught exception!'
        plot.stop()
        plant.stop()
        sys.exit(0)


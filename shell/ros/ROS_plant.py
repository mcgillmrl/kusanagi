import numpy as np
import theano
import sys
import serial
import struct
from enum import Enum
from scipy.optimize import minimize, basinhopping
from matplotlib import pyplot as plt
from time import time, sleep
from matplotlib.widgets import Cursor
from matplotlib.colors import cnames
from scipy.integrate import ode
from time import time, sleep
from threading import Thread, Lock
from multiprocessing import Process
from multiprocessing.managers import BaseManager
from ghost.control import RandPolicy
from utils import print_with_stamp,gTrig_np,wrap_params,unwrap_params,MemoizeJac

plant_name = 'ROSPlant'

def apply_control(u):
    u = np.array(u,dtype=np.float64)
    if len(u.shape) < 2:
        u = u[:,None]

def get_state():
    #TODO: Get the plant's state

def start(self):
    #TODO: Start it

def stop(self):
    #TODO: Stop robot

def reset_state(self):
    raise NotImplementedError("You need to implement the reset_state method in your Plant subclass.")


def apply_controller(name = 'EpisodicLearner', policy, cost=None, angle_idims=None, experience = None, async_plant=True, H=float('inf') ,random_controls=False):
    print_with_stamp('Starting data collection run',name)
    if H < float('inf'):
        print_with_stamp('Running for %f seconds'%(H),name)

    if random_controls:
        policy =  RandPolicy(self.policy.maxU)

    # mark the start of the episode
    experience.new_episode()                                ##TODO: DEAL WITH EXPERIENCE DATA HANDLING

    # start robot
    if async_plant:
        plant.start()                                       ##TODO: PLANT METHODS
    exec_time = time()
    x_t, t0 = plant.get_state()                             ##TODO: PLANT METHODS
    Sx_t = np.zeros((x_t.shape[0],x_t.shape[0]))
    L_noise = np.linalg.cholesky(plant.noise)               ##TODO: PLANT METHODS
    if plant.noise is not None:
        # randomize state
        x_t = x_t + np.random.randn(x_t.shape[0]).dot(L_noise);
    t = t0
    
    H_steps = int(np.ceil(H/plant.dt))
    # do rollout
    #while t <= t0 + H:
    for i in xrange(H_steps):
        # convert input angle dimensions to complex representation
        x_t_ = gTrig_np(x_t[None,:],angle_idims).flatten()
        #  get command from policy (this should be fast, or at least account for delays in processing):
        u_t = policy.evaluate(t, x_t_)[0].flatten()
        #  send command to robot:
        plant.apply_control(u_t)                            ##TODO: PLANT METHODS
        if cost is not None:
            #  get cost:
            c_t = cost(x_t, Sx_t)
            # append to experience dataset
            experience.add_sample(t,x_t,u_t,c_t)            ##TODO: DEAL WITH EXPERIENCE DATA HANDLING
        else:
            # append to experience dataset
            experience.append(t,x_t,u_t,0)                  ##TODO: DEAL WITH EXPERIENCE DATA HANDLING

        # step the plant if necessary
        # if not async_plant:
        #     plant.step()

        # sleep to match the desired sample rate
        exec_time = time() - exec_time
        if exec_time < plant.dt:
            sleep(plant.dt-exec_time)

        #  get robot state (this should ensure synchronicity by blocking until dt seconds have passed):
        exec_time = time()
        x_t, t = plant.get_state()                          ##TODO: PLANT METHODS
        if plant.noise is not None:                         ##TODO: PLANT METHODS
            # randomize state
            x_t = x_t + np.random.randn(x_t.shape[0]).dot(L_noise);
        
    # add last state to experience
    x_t_ = gTrig_np(x_t[None,:], angle_idims).flatten()
    u_t = np.zeros_like(u_t)
    if cost is not None:
        c_t = cost(x_t, Sx_t)
        experience.add_sample(t,x_t,u_t,c_t)                ##TODO: DEAL WITH EXPERIENCE DATA HANDLING
    else:
        experience.append(t,x_t,u_t,0)                      ##TODO: DEAL WITH EXPERIENCE DATA HANDLING

    # stop robot
    run_value = np.array(experience.immediate_cost[-1][:-1])        ##TODO: DEAL WITH EXPERIENCE DATA HANDLING
    print_with_stamp('Done. Stopping robot. Value of run [%f]'%(run_value.sum()),name)

    plant.stop()
    n_episodes += 1             ##TODO: DEAL WITH EXPERIENCE DATA HANDLING
    return experience
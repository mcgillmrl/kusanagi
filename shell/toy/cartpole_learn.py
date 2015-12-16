from ghost.learners.PILCO import PILCO
from shell.toy.cartpole import Cartpole, CartpoleDraw, CartpoleCost
from ghost.control import conRand
import numpy as np
from threading import Thread
from time import sleep,time
from util import augment
import utils

def test_sim():
    np.set_printoptions(linewidth=200, precision=6, suppress=True)
    # setup plant parameters
    dt = 0.005
    model_parameters ={}
    model_parameters['l'] = 0.5
    model_parameters['m'] = 0.5
    model_parameters['M'] = 0.5
    model_parameters['b'] = 0.1
    model_parameters['g'] = 9.82

    x0 = [0,0,0,np.pi]
    target = [0,0,0,np.pi]
    angle_dims = [3]

    plant = Cartpole(dt,model_parameters,x0)
    cost = CartpoleCost(target,model_parameters['l'],angle_dims)
    policy = conRand([1])

    running = True
    def simulation_loop():
        t=0
        start_time = time()
        s = np.zeros((4,4))
        avg_exec_time = 0
        n = 0
        utils.print_with_stamp('Starting simulation loop','cartpole_learn')
        while running:
            exec_time = time()
            #u = policy.fcn(plant.x)
            plant.fcn(plant.x)
            c = cost.fcn(plant.x)[0]
            t = t + dt
            #print "%f\t%s\t%f"%(t, plant.x.T, c)
            exec_time = time() - exec_time
            avg_exec_time += exec_time
            n += 1
            sleep(max(dt-exec_time,0))
        print avg_exec_time/n

    draw_cp = CartpoleDraw(plant,16)

    simulation_thread = Thread(target=simulation_loop)
    simulation_thread.start()
    try:
        utils.print_with_stamp('Starting drawing loop','cartpole_learn')
        draw_cp.start()
    except KeyboardInterrupt:
        running = False

if __name__ == '__main__':
    dt = 0.005
    model_parameters ={}
    model_parameters['l'] = 0.5
    model_parameters['m'] = 0.5
    model_parameters['M'] = 0.5
    model_parameters['b'] = 0.1
    model_parameters['g'] = 9.82

    x0 = [0,0,0,np.pi]
    target = [0,0,0,np.pi]
    angle_dims = [3]

    plant = Cartpole(dt,model_parameters,x0)
    cost = CartpoleCost(target,model_parameters['l'])
    policy = conRand([1])

    learner = PILCO(plant, policy, cost, angle_dims)
    #test_sim()

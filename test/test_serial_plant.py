import atexit
import signal,sys
import numpy as np
from time import time,sleep
from functools import partial
from kusanagi.shell.arduino import SerialPlant
from kusanagi.shell.plant import LivePlot
from kusanagi.shell.cartpole import  CartpoleDraw
from kusanagi.ghost.control import RBFPolicy
from kusanagi.utils import gTrig_np

if __name__ == '__main__':
    #np.random.seed(31337)
    np.set_printoptions(linewidth=500)
    # initliaze plant
    plant_params = dict(dt=0.1, l=0.5, m=0.5, M=0.5, b=0.1, g=9.82)
    maxU = [10]
    plant = SerialPlant(state_indices=[0,2,3,1], maxU=maxU, baud_rate=4000000,port='/dev/ttyACM0', **plant_params)
    
    # initializes visualization
    plot = LivePlot(plant,0.001,H=5,angi=[0,3])
    plot.init_ui()
    draw_cp = CartpoleDraw(plant, 0.001)
    draw_cp.init_ui()
    def _render(self, mode='human', close=False, *args, **kwargs):
        state = self.get_state(noisy=False)
        draw_cp.update(*state)
        plot.update(*state)
    plant.set_render_func(_render)
    
    atexit.register(plant.stop)
    atexit.register(plot.close)
    atexit.register(draw_cp.close)

    w = 2*np.pi*0.5
    A = 10.0
    plant.reset()
    while True:
        exec_time = time()
        #u_t = A*w*np.cos(w*(time()+0.5*dt))[None]
        u_t = A*np.cos(w*time())[None]
        #if u_t >= 0:
        #    u_t[0] = A
        #else:
        #    u_t[0] = -A
        plant.step(u_t)
        plant.render()
        exec_time = time() - exec_time
        sleep(max(plant.dt-exec_time,0))


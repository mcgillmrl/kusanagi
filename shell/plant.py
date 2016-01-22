import numpy as np
import sys

from matplotlib import pyplot as plt
from matplotlib.widgets import Cursor
from scipy.integrate import ode
from time import time, sleep
from threading import Thread

from utils import print_with_stamp

class ODEPlant(object):
    def __init__(self, params, x0, dt=0.01, name='Plant', integrator='dopri5', atol=1e-12, rtol=1e-12):
        self.solver = ode(self.dynamics).set_integrator(integrator,atol=atol,rtol=rtol)
        self.name = name
        self.x = None
        self.u = None
        self.t = 0
        self.dt = dt
        self.params = params
        self.set_state(x0)

    def apply_control(self,u):
        self.u = np.array(u)[:,None]

    def set_state(self, x):
        if (self.x is None or np.linalg.norm(x-self.x) > 1e-12):
            self.x = np.array(x)[:,None]
            self.solver = self.solver.set_initial_value(x)

    def get_state(self):
        return self.x.flatten(),self.solver.t

    def step(self,dt):
        t1 = self.solver.t + dt
        while self.solver.successful and self.solver.t < t1:
            self.solver.integrate(self.solver.t+ dt)
        self.x = np.array(self.solver.y)
        self.t = self.solver.t
        return self.x
    
    def run(self):
        start_time = time()
        print_with_stamp('Starting simulation loop',self.name)
        while self.running:
            exec_time = time()
            self.step(self.dt)
            #print_with_stamp('%f, %s'%(self.t,self.x),self.name)
            exec_time = time() - exec_time
            sleep(max(self.dt-exec_time,0))
        print_with_stamp('Stopping simulation loop',self.name)

    def start(self):
        self.sim_thread = Thread(target=self.run)
        self.running = True
        self.sim_thread.start()
    
    def stop(self):
        self.running = False

    def dynamics(self):
        print "You need to implement the function dynamics() in your plant class."

    def fcn(self,x,u=None,x_cov=None,u_cov=None):
        print "You need to implement the function fcn(x,u,x_cov,u_cov) in your plant class."

class PlantDraw(object):
    def __init__(self, plant, refresh_period=(1.0/60), name='PlantDraw'):
        super(PlantDraw,self).__init__()
        self.name = name
        self.plant = plant

        self.dt = refresh_period
        self.scale =  150 # pixels per meter

        self.center_x = 0
        self.center_y = 0

    def init_ui(self):
        self.fig = plt.figure(self.name)
        plt.xlim([-2,2])
        plt.ylim([-2,2])
        plt.ion()
        plt.show()
        self.ax = plt.gca()
        self.ax.set_aspect('equal','datalim')
        self.bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.init_artists()
        self.fig.canvas.draw()
        self.cursor = Cursor(self.ax, useblit=True, color='red', linewidth=2 )
        plt.show()
        pass

    def run(self):
        # start the matplotlib plotting
        self.init_ui()

        while self.running:
            exec_time = time()
            # update the drawing from the plant state
            state,t = self.plant.get_state()
            updts = self.update(state,t)
            self.fig.canvas.restore_region(self.bg)
            for artist in updts:
                self.ax.draw_artist(artist)
            self.fig.canvas.blit(self.ax.bbox)

            # sleep to guarantee the desired frame rate
            exec_time = time() - exec_time
            sleep(max(self.dt-exec_time,0))

        # close the matplotlib windows, clean up
        plt.ioff()
        plt.close('all')

    def start(self):
        print_with_stamp('Starting drawing loop',self.name)
        self.sim_thread = Thread(target=self.run)
        self.running = True
        self.sim_thread.start()
    
    def stop(self):
        print_with_stamp('Stopping drawing loop',self.name)
        self.running = False
    
    def update(self):
        print "You need to implement the self.update(qp) function in your PlantDraw class."

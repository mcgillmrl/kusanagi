import numpy as np
import sys
import serial
import struct
from enum import Enum

from matplotlib import pyplot as plt
from matplotlib.widgets import Cursor
from scipy.integrate import ode
from time import time, sleep
from threading import Thread
from utils import print_with_stamp

class Plant(object):
    def __init__(self, params, x0, S0=None, dt=0.01, noise=None, name='Plant'):
        self.name = name
        self.params = params
        self.x0 = x0
        self.S0= S0
        self.x = np.array(x0,dtype=np.float64)
        self.u = None
        self.t = 0
        self.dt = dt
        self.noise = noise
        self.async = False
        self.plant_thread = Thread(target=self.run)
    
    def apply_control(self,u):
        self.u = np.array(u,dtype=np.float64)
        if len(self.u.shape) < 2:
            self.u = self.u[:,None]

    def get_state(self):
        return self.x.flatten(),self.t
    
    def run(self):
        start_time = time()
        print_with_stamp('Starting simulation loop',self.name)
        while self.async:
            exec_time = time()
            self.step(self.dt)
            #print_with_stamp('%f, %s'%(self.t,self.x),self.name)
            exec_time = time() - exec_time
            sleep(max(self.dt-exec_time,0))

    def start(self):
        if self.plant_thread.is_alive():
            print_with_stamp('Waiting until robot stops',self.name)
            while self.plant_thread.is_alive():
                sleep(1.0)
        
        self.async = True
        self.plant_thread.start()
    
    def stop(self):
        self.async = False
        if self.plant_thread.is_alive():
            # wait until thread stops
            self.plant_thread.join(10)
            # create new thread object, since python threads can only be started once
            self.plant_thread = Thread(target=self.run)
        print_with_stamp('Stopped simulation loop',self.name)

    def step(self):
        raise NotImplementedError("You need to implement the step method in your Plant subclass.")

    def reset_state(self):
        raise NotImplementedError("You need to implement the reset_state method in your Plant subclass.")

class ODEPlant(Plant):
    def __init__(self, params, x0, S0=None, dt=0.01, noise=None, name='ODEPlant', integrator='dopri5', atol=1e-12, rtol=1e-12):
        super(ODEPlant,self).__init__(params, x0, S0, dt, noise, name)
        self.solver = ode(self.dynamics).set_integrator(integrator,atol=atol,rtol=rtol)
        self.set_state(self.x0)

    def set_state(self, x):
        if (self.x is None or np.linalg.norm(x-self.x) > 1e-12):
            self.x = np.array(x,dtype=np.float64).flatten()[:,None]
            self.solver = self.solver.set_initial_value(x)
    
    def reset_state(self):
        print_with_stamp('Reset to inital state',self.name)
        if self.S0 is None:
            self.set_state(self.x0)
        else:
            #self.set_state(np.random.multivariate_normal(self.x0,self.S0))
            L_noise = np.linalg.cholesky(self.S0)
            start = self.x0 + np.random.randn(self.S0.shape[1]).dot(L_noise)
            self.set_state( start );

    def step(self,dt=None):
        if dt is None:
            dt = self.dt
        t1 = self.solver.t + dt
        while self.solver.successful and self.solver.t < t1:
            self.solver.integrate(self.solver.t+ dt)
        self.x = np.array(self.solver.y)
        self.t = self.solver.t
        return self.x

    def dynamics(self):
        raise NotImplementedError("You need to implement the dynamics method in your ODEPlant subclass.")

class SerialPlant(Plant):
    cmds = ['RESET_STATE','GET_STATE','APPLY_CONTROL','CMD_OK','STATE']
    cmds = dict(zip(cmds,[str(i) for i in xrange(len(cmds))]))

    def __init__(self, params, x0, S0=None, dt=0.01, noise=None, name='SerialPlant', baud_rate='115200', port='/dev/ttyACM3', state_indices=None, maxU=None):
        super(SerialPlant,self).__init__(params, x0, S0, dt, noise, name)
        self.port = port
        self.baud_rate = baud_rate
        self.serial = serial.Serial(self.port,self.baud_rate)
        self.state_indices = state_indices if state_indices is not None else range(len(x0))
        self.maxU = np.array(maxU)/2048.0
    
    def apply_control(self,u):
        self.u = np.array(u,dtype=np.float64)
        if len(self.u.shape) < 2:
            self.u = self.u[:,None]
        if self.maxU is not None:
            self.u /= self.maxU;

        u_array = self.u.flatten().tolist()
        u_array.append(self.dt)
        u_string = ','.join([ str(ui) for ui in u_array ] ) #TODO pack as binary
        cmd = self.cmds['APPLY_CONTROL']+','+u_string+";"
        self.serial.write(cmd)

    def step(self,dt=None):
        if dt is None:
            dt = self.dt
        t1 = self.t + dt
        while self.t < t1:
            self.x,self.t= self.state_from_serial()
        return self.x

    def state_from_serial(self):
        self.serial.write(self.cmds['GET_STATE']+";")
        c = self.serial.read()
        buf = c
        while buf != self.cmds['STATE']+',': # TODO timeout this loop
            c = self.serial.read()
            buf = buf[-1]+c
        buf = []
        res = []
        escaped = False
        while True: # TODO timeout this loop
            c = self.serial.read()
            if not escaped:
                if c == '/':
                    escaped = True
                    continue
                elif c == ',':
                    res.append(''.join(buf))
                    buf = []
                    continue
                elif c == ';':
                    res.append(''.join(buf))
                    buf = []
                    break
            buf.append(c)
            escaped = False
        #print res
        #if sum([len(i) for i in res]) != 40:
        #    for r in res:
        #        print len(r),r
        #    raw_input()
        res = np.array([struct.unpack('<d',ri) for ri in res])
        return res[self.state_indices],res[-1]

    def reset_state(self):
        print_with_stamp('Please reset your plant to its initial state and hit Enter',self.name)
        raw_input()
        self.serial.write(self.cmds['RESET_STATE']+";")

class PlantDraw(object):
    def __init__(self, plant, refresh_period=(1.0/60), name='PlantDraw'):
        super(PlantDraw,self).__init__()
        self.name = name
        self.plant = plant

        self.dt = refresh_period
        self.scale =  150 # pixels per meter

        self.center_x = 0
        self.center_y = 0
        self.running = False

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
        self.plant_thread = Thread(target=self.run)
        self.running = True
        self.plant_thread.start()
    
    def stop(self):
        print_with_stamp('Stopping drawing loop',self.name)
        self.running = False
    
    def update(self):
        print "You need to implement the self.update(qp) function in your PlantDraw class."

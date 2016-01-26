import numpy as np
from scipy.integrate import ode
from shell.plant import ODEPlant, PlantDraw
from ghost.cost import Cost, lossGeneric
from util import augment
from threading import Thread
import thirdparty.mtTkinter as tk
from utils import print_with_stamp
from matplotlib import pyplot as plt

class Cartpole(ODEPlant):
    def __init__(self, params, x0, S0=None, dt=0.01, name='Cartpole', integrator='dopri5', atol=1e-12, rtol=1e-12):
        super(Cartpole, self).__init__(params, x0, S0, dt=dt, name=name, integrator=integrator, atol=atol, rtol=rtol)

    def dynamics(self,t,z):
        l = self.params['l']
        m = self.params['m']
        M = self.params['M']
        b = self.params['b']
        g = self.params['g']
        f = self.u if self.u is not None else np.array([0])

        sz = np.sin(z[3]); cz = np.cos(z[3]); cz2 = cz*cz;
        a0 = m*l*z[2]*z[2]*sz
        a1 = g*sz
        a2 = f[0] - b*z[1];
        a3 = 4*(M+m) - 3*m*cz2

        dz = np.zeros((4,1))
        dz[0] = z[1]
        dz[1] = (  2*a0 + 3*m*a1*cz + 4*a2 )/ ( a3 )
        dz[2] = -3*( a0*cz + 2*( (M+m)*a1 + a2*cz ) )/( l*a3 ) 
        dz[3] = z[2]

        return dz

class CartpoleCost(Cost):
    def __init__(self, target, pendulum_length, angi):
        D0 = len(target)
        D1 = D0 + 2*len(angi)
        Q = self.getQ(pendulum_length,D0,D1)
        super(CartpoleCost,self).__init__(lossGeneric,target=target,Q=Q,angi=angi)

    def getQ(self, pendulum_length, D0, D1):
        Q = np.zeros((D1,D1))
        i = np.array([0,D0])[:,None]; 
        v = np.array([[1,pendulum_length]])
        Q[i,i.T] = v.T.dot(v)
        Q[D0+1,D0+1] = pendulum_length**2
        return Q

class CartpoleDraw(PlantDraw):
    def __init__(self, cartpole_plant, refresh_period=100, name='CartpoleDraw'):
        super(CartpoleDraw, self).__init__(cartpole_plant, refresh_period,name)
        l = self.plant.params['l']
        m = self.plant.params['m']
        M = self.plant.params['M']
        b = self.plant.params['b']
        g = self.plant.params['g']
        self.mass_r = 0.05*np.sqrt( m ) # distance to corner of bounding box
        self.body_h = 0.5*np.sqrt( M )

        self.center_x = 0
        self.center_y = 0

        # initialize the patches to draw the cartpole
        self.body_rect = plt.Rectangle( (self.center_x-0.5*self.body_h, self.center_y-0.125*self.body_h), self.body_h, 0.25*self.body_h, facecolor='black')
        self.pole_line = plt.Line2D((self.center_x, 0), (self.center_y, l), lw=2, c='r')
        self.mass_circle = plt.Circle((0, l), self.mass_r, fc='y')

    def init_artists(self):
        self.ax.add_patch(self.body_rect)
        self.ax.add_patch(self.mass_circle)
        self.ax.add_line(self.pole_line)

    def update(self, state, t):
        l = self.plant.params['l']

        body_x = self.center_x + state[0]
        body_y = self.center_y
        mass_x = l*np.sin(state[3]) + body_x
        mass_y = -l*np.cos(state[3]) + body_y

        self.body_rect.set_xy((body_x-0.5*self.body_h,body_y-0.125*self.body_h))
        self.pole_line.set_xdata(np.array([body_x,mass_x]))
        self.pole_line.set_ydata(np.array([body_y,mass_y]))
        self.mass_circle.center = (mass_x,mass_y)

        return (self.body_rect,self.pole_line, self.mass_circle)

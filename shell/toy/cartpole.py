import numpy as np
from scipy.integrate import ode
from shell.plant import Plant, PlantDraw
from ghost.cost import Cost, lossGeneric
from util import augment
from threading import Thread
import thirdparty.mtTkinter as tk

class Cartpole(Plant):
    def __init__(self, dt, model_parameters, initial_state, integrator='dopri5', atol=1e-12, rtol=1e-12):
        super(Cartpole, self).__init__(self.dynamics, integrator, atol, rtol)
        self.f = 0
        self.set_state(np.array(initial_state))
        self.dt = dt
        self.l = model_parameters['l']
        self.m = model_parameters['m']
        self.M = model_parameters['M']
        self.b = model_parameters['b']
        self.g = model_parameters['g']
    
    def set_force(self,f):
        self.f = f

    def dynamics(self,t,z):
        l = self.l
        m = self.m
        M = self.M
        b = self.b
        g = self.g

        sz3 = np.sin(z[3])
        cz3 = np.cos(z[3])
        a = m*l*(z[2]**2)*sz3
        b = self.f - b*z[1];
        c = 4*(M+m) - 3*m*(cz3**2)

        dz = np.zeros((4,1))
        dz[0] = z[1]
        dz[1] = (  2*a + 3*m*g*sz3*cz3 + 4*b )/c
        dz[2] = ( -3*a*cz3 - 6*(M+m)*g*sz3 - 6*b )/( l*c ) 
        dz[3] = z[2]

        return dz

    def fcn(self,x,u=None,x_cov=None,u_cov=None):
        self.set_state(x)
        if u is not None:
          self.set_force(u)
        return self.step(self.dt)

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
    def __init__(self, cartpole_plant, refresh_period=100, master=None):
        super(CartpoleDraw, self).__init__(cartpole_plant, refresh_period, master)
        self.mass_r = self.scale*np.sqrt(self.plant.m/400 ) # distance to corner of bounding box
        self.body_h = self.scale*np.sqrt( self.plant.M/20 )

        # draw the body
        self.body = self.canvas.create_rectangle(-self.body_h, -0.5*self.body_h, self.body_h, 0.5*self.body_h,fill="black")
        self.pole = self.canvas.create_line(0,0,0,self.plant.l,fill="red",width=5)
        self.pole_mass = self.canvas.create_oval(-self.mass_r,-self.mass_r,self.mass_r,self.mass_r,fill="yellow")
        
    def draw(self):
        state = np.array(self.plant.x).squeeze()
        body_x = self.center_x + self.scale*state[0]
        body_y = self.center_y

        self.canvas.coords(self.body, (body_x-self.body_h, body_y-0.25*self.body_h, body_x+self.body_h, body_y + 0.25*self.body_h))
        mass_x = self.scale*self.plant.l*np.sin(state[3]) + body_x
        mass_y = self.scale*self.plant.l*np.cos(state[3]) + body_y
        self.canvas.coords(self.pole, (body_x, body_y, mass_x, mass_y))
        self.canvas.coords(self.pole_mass, (-self.mass_r+mass_x,-self.mass_r+mass_y,self.mass_r+mass_x,self.mass_r+mass_y))

import numpy as np
from scipy.integrate import ode
import thirdparty.mtTkinter as tk

from utils import print_with_stamp

class ODEPlant(object):
    def __init__(self, dynamics, dt=0.01, name='Plant', integrator='dopri5', atol=1e-12, rtol=1e-12):
        self.solver = ode(dynamics).set_integrator(integrator,atol=atol,rtol=rtol)
        self.name = name
        self.x = None
        self.u = None
        self.t = 0
        self.dt = dt

    def set_state(self, x):
        if (self.x is None or np.linalg.norm(x-self.x) > 1e-12):
            self.x = np.array(x)[:,None]
            self.solver = self.solver.set_initial_value(x)

    def set_control(self,u):
        print "You need to implement the function fcn(x,u,x_cov,u_cov) in your plant class."

    def step(self,dt):
        t1 = self.solver.t + dt
        while self.solver.successful and self.solver.t < t1:
            self.solver.integrate(self.solver.t+ dt)
        self.x = np.array(self.solver.y)
        self.t = self.solver.t
        return self.x

    def fcn(self,x,u,x_cov,u_cov):
        print "You need to implement the function fcn(x,u,x_cov,u_cov) in your plant class."

class PlantDraw(object):
    def __init__(self, plant, refresh_period=100, master=None):
        self.plant = plant

        # init tkinter window
        if master is None:
            self.master = tk.Tk()
        else:
            self.master = master

        self.master.wm_title("Cartpole!")
        self.master.geometry('512x512')
        self.master.aspect(1,1,1,1)

        # creathe the frame for the content
        self.frame = tk.Frame(self.master)

        # create the canvas for drawing the state of the cartpole
        self.canvas = tk.Canvas(self.frame)

        # place canvas in frame, frame in window
        self.canvas.pack(fill = "both", expand = 1)
        self.frame.pack(fill = "both", expand = 1)

         #configure resize event
        self.canvas.bind('<Configure>', self.resize)

        # init variables
        self.height =  self.canvas.winfo_height()
        self.width =  self.canvas.winfo_width()
        self.center_x = self.width/2.0
        self.center_y = self.height/2.0

        self.refresh_period = refresh_period
        self.scale =  150 # pixels per meter

    def resize(self,event):
        if event.width == self.width and event.height == self.height:
            return
        # update internal variables
        self.width = event.width
        self.height = event.height
        self.center_x = self.width/2.0
        self.center_y = self.height/2.0

    def start(self):
        self.master.after(self.refresh_period, self.update)
        self.master.mainloop()

    def update(self):
        self.draw()
        self.master.after(self.refresh_period, self.update)

    def draw(self):
        print "You need to implement the draw() function  in your PlantDraw class."



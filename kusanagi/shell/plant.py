# pylint: disable=C0103
import gym
import numpy as np
import types

from matplotlib import pyplot as plt
from matplotlib.widgets import Cursor
from matplotlib.colors import cnames
from scipy.integrate import ode
from time import time, sleep
from threading import Thread
from multiprocessing import Process, Pipe, Event
from kusanagi.utils import print_with_stamp, gTrig_np

color_generator = iter(cnames.items())


class Plant(gym.Env):
    metadata = {
        'render.modes': ['human']
    }
    def __init__(self, dt=0.1, noise_dist=None,
                 angle_dims=[], name='Plant',
                 *args, **kwargs):
        self.name = name
        self.dt = dt
        self.noise_dist = noise_dist
        self.angle_dims = angle_dims
        self.state = None
        self.u = None
        self.t = 0
        self.done = False
        self.renderer = None

        # initialize loss_func
        self.loss_func = None

    def apply_control(self, u):
        self.u = np.array(u, dtype=np.float64)
        if len(self.u.shape) < 2:
            self.u = self.u[:, None]

    def get_state(self, noisy=True):
        state = self.state

        if noisy and self.noise_dist is not None:
            # noisy state measurement
            state += self.noise_dist.sample(1).flatten()

        if self.angle_dims:
            # convert angle dimensions to complex representation
            state = gTrig_np(state, self.angle_dims).flatten()
        return state.flatten(), self.t

    def set_state(self, state):
        self.state = state

    def stop(self):
        print_with_stamp('Stopping robot', self.name)
        # self._close()

    def _step(self, action):
        msg = "You need to implement self._step in your Plant subclass."
        raise NotImplementedError(msg)

    def _reset(self):
        msg = "You need to implement self._reset in your Plant subclass."
        raise NotImplementedError(msg)

    def set_render_func(self, render_func):
        self._render = types.MethodType(render_func, self)


class ODEPlant(Plant):
    def __init__(self, name='ODEPlant', integrator='dopri5',
                 atol=1e-12, rtol=1e-12,
                 *args, **kwargs):
        super(ODEPlant, self).__init__(name=name, *args, **kwargs)
        integrator = kwargs.get('integrator', 'dopri5')
        atol = kwargs.get('atol', 1e-12)
        rtol = kwargs.get('rtol', 1e-12)

        # initialize ode solver
        self.solver = ode(self.dynamics).set_integrator(integrator,
                                                        atol=atol,
                                                        rtol=rtol)

    def set_state(self, state):
        if self.state is None or\
           np.linalg.norm(np.array(state)-np.array(self.state)) > 1e-12:
            # float64 required for the ode integrator
            self.state = np.array(state, dtype=np.float64).flatten()
        # set solver internal state
        self.solver = self.solver.set_initial_value(self.state)
        # get time from solver
        self.t = self.solver.t

    def _step(self, action):
        self.apply_control(action)
        dt = self.dt
        t1 = self.solver.t + dt
        while self.solver.successful and self.solver.t < t1:
            self.solver.integrate(self.solver.t + dt)
        self.state = np.array(self.solver.y)
        self.t = self.solver.t
        cost = None
        if self.loss_func is not None:
            cost = self.loss_func(np.array(self.state)[None, :])
        state, t = self.get_state()
        return state, cost, False, dict(t=t)

    def dynamics(self, *args, **kwargs):
        msg = "You need to implement self.dynamics in the ODEPlant subclass."
        raise NotImplementedError(msg)



class PlantDraw(object):
    def __init__(self, plant, refresh_period=(1.0/240),
                 name='PlantDraw', *args, **kwargs):
        super(PlantDraw, self).__init__()
        self.name = name
        self.plant = plant
        self.drawing_thread = None
        self.polling_thread = None

        self.dt = refresh_period
        self.exec_time = time()
        self.scale = 150  # pixels per meter

        self.center_x = 0
        self.center_y = 0
        self.running = Event()

        self.polling_pipe, self.drawing_pipe = Pipe()

    def init_ui(self):
        self.fig = plt.figure(self.name)
        self.ax = plt.gca()
        self.ax.set_xlim([-1.5, 1.5])
        self.ax.set_ylim([-1.5, 1.5])
        self.ax.set_aspect('equal', 'datalim')
        self.ax.grid(True)
        self.fig.canvas.draw()
        self.bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.cursor = Cursor(self.ax, useblit=True, color='red', linewidth=2)
        self.init_artists()
        #plt.ion()
        plt.show(False)

    def drawing_loop(self, drawing_pipe):
        # start the matplotlib plotting
        self.init_ui()

        while self.running.is_set():
            exec_time = time()
            # get any data from the polling loop
            updts = None
            while drawing_pipe.poll():
                data_from_plant = drawing_pipe.recv()
                if data_from_plant is None:
                    self.running.clear()
                    break

                # get the visuzlization updates from the latest state
                state, t = data_from_plant
                updts = self.update(state, t)
                self.update_canvas(updts)

            # sleep to guarantee the desired frame rate
            exec_time = time() - exec_time
            plt.waitforbuttonpress(max(self.dt-exec_time, 1e-9))
        self.close()

    def close(self):
        # close the matplotlib windows, clean up
        #plt.ioff()
        plt.close(self.fig)

    def update(self, *args, **kwargs):
        plt.figure(self.name)
        updts = self._update(*args, **kwargs)
        self.update_canvas(updts)

    def _update(self, *args, **kwargs):
        msg = "You need to implement the self._update() method in your\
 PlantDraw class."
        raise NotImplementedError(msg)

    def init_artists(self, *args, **kwargs):
        msg = "You need to implement the self.init_artists() method in your\
 PlantDraw class."
        raise NotImplementedError(msg)

    def update_canvas(self, updts):
        if updts is not None:
            # update the drawing from the plant state
            self.fig.canvas.restore_region(self.bg)
            for artist in updts:
                self.ax.draw_artist(artist)
            self.fig.canvas.update()
            # sleep to guarantee the desired frame rate
            exec_time = time() - self.exec_time
            plt.waitforbuttonpress(max(self.dt-exec_time, 1e-9))
        self.exec_time = time()

    def polling_loop(self, polling_pipe):
        current_t = -1
        while self.running.is_set():
            exec_time = time()
            state, t = self.plant.get_state(noisy=False)
            if t != current_t:
                polling_pipe.send((state, t))

            # sleep to guarantee the desired frame rate
            exec_time = time() - exec_time
            sleep(max(self.dt-exec_time, 0))

    def start(self):
        print_with_stamp('Starting drawing loop', self.name)
        self.drawing_thread = Process(target=self.drawing_loop,
                                      args=(self.drawing_pipe, ))
        self.drawing_thread.daemon = True
        self.polling_thread = Thread(target=self.polling_loop,
                                     args=(self.polling_pipe, ))
        self.polling_thread.daemon = True
        # self.drawing_thread = Process(target=self.run)
        self.running.set()
        self.polling_thread.start()
        self.drawing_thread.start()

    def stop(self):
        self.running.clear()

        if self.drawing_thread is not None and self.drawing_thread.is_alive():
            # wait until thread stops
            self.drawing_thread.join(10)

        if self.polling_thread is not None and self.polling_thread.is_alive():
            # wait until thread stops
            self.polling_thread.join(10)

        print_with_stamp('Stopped drawing loop', self.name)


# an example that plots lines
class LivePlot(PlantDraw):
    def __init__(self, plant, refresh_period=1.0,
                 name='Serial Data', H=5.0, angi=[]):
        super(LivePlot, self).__init__(plant, refresh_period, name)
        self.H = H
        self.angi = angi
        # get first measurement
        state, t = plant.get_state(noisy=False)
        self.data = np.array([state])
        self.t_labels = np.array([t])

        # keep track of latest time stamp and state
        self.current_t = t
        self.previous_update_time = time()
        self.update_period = refresh_period

    def init_artists(self):
        plt.figure(self.name)
        self.lines = [plt.Line2D(self.t_labels, self.data[:, i],
                                 c=next(color_generator)[0])
                      for i in range(self.data.shape[1])]
        self.ax.set_aspect('auto', 'datalim')
        for line in self.lines:
            self.ax.add_line(line)
        self.previous_update_time = time()

    def _update(self, state, t):
        if t != self.current_t:
            if len(self.data) <= 1:
                self.data = np.array([state]*2)
                self.t_labels = np.array([t]*2)

            if len(self.angi) > 0:
                state[self.angi] = (state[self.angi]+np.pi) % (2*np.pi) - np.pi

            self.current_t = t
            # only keep enough data points to fill the window to avoid using
            # up too much memory
            curr_time = time()
            self.update_period = 0.95*self.update_period + \
                0.05*(curr_time - self.previous_update_time)
            self.previous_update_time = curr_time
            history_size = int(1.5*self.H/self.update_period)
            self.data = np.vstack((self.data, state))[-history_size:, :]
            self.t_labels = np.append(self.t_labels, t)[-history_size:]

            # update the lines
            for i in range(len(self.lines)):
                self.lines[i].set_data(self.t_labels, self.data[:, i])

            # update the plot limits
            plt.xlim([self.t_labels.min(), self.t_labels.max()])
            plt.xlim([t-self.H, t])
            mm = self.data.mean()
            ll = 1.05*np.abs(self.data[:, :]).max()
            plt.ylim([mm-ll, mm+ll])
            self.ax.autoscale_view(tight=True, scalex=True, scaley=True)

        return self.lines

# pylint: disable=C0103
'''
Contains the Cartpole envionment, along with default parameters and
a rendering class
'''
import numpy as np

from gym import spaces
from matplotlib import pyplot as plt

from kusanagi.shell import plant
from kusanagi.ghost import cost
from kusanagi import utils


def default_params():
    # setup learner parameters
    angi = [3]
    x0 = np.array([0, 0, 0, 0])
    S0 = np.eye(len(x0))*(0.2**2)
    p0 = utils.distributions.Gaussian(x0, S0)
    x0a, S0a = utils.gTrig2_np(x0[None, :], np.array(S0)[None, :, :],
                               angi, len(x0))

    # plant parameters
    plant_params = {}
    plant_params['dt'] = 0.1
    plant_params['pole_length'] = 0.6
    plant_params['pole_mass'] = 0.5
    plant_params['cart_mass'] = 0.5
    plant_params['friction'] = 0.1
    plant_params['gravity'] = 9.82
    plant_params['state0_dist'] = p0
    plant_params['noise_dist'] = utils.distributions.Gaussian(
        np.zeros((p0.dim,)),
        np.eye(p0.dim)*0.01**2)

    # policy parameters
    policy_params = {}
    policy_params['state0_dist'] = p0
    policy_params['angle_dims'] = angi
    policy_params['n_inducing'] = 30
    policy_params['maxU'] = [10]

    # dynamics model parameters
    dynmodel_params = {}
    dynmodel_params['idims'] = x0a.size + len(policy_params['maxU'])
    dynmodel_params['odims'] = x0.size
    dynmodel_params['n_inducing'] = 100

    # cost function parameters
    cost_params = {}
    cost_params['angle_dims'] = angi
    cost_params['target'] = [0, 0, 0, np.pi]
    cost_params['cw'] = 0.25
    cost_params['expl'] = -0.1
    cost_params['pole_length'] = plant_params['pole_length']
    cost_params['loss_func'] = cost.quadratic_saturating_loss

    # optimizer params
    opt_params = {}
    opt_params['max_evals'] = 100
    opt_params['conv_thr'] = 1e-16
    opt_params['min_method'] = 'L-BFGS-B'

    # general parameters
    params = {}
    params['state0_dist'] = p0
    params['angle_dims'] = angi
    params['min_steps'] = int(2.5/plant_params['dt'])  # control horizon
    params['max_steps'] = int(4.0/plant_params['dt'])  # control horizon
    params['discount'] = 1.0                           # discount factor
    params['plant'] = plant_params
    params['policy'] = policy_params
    params['dynamics_model'] = dynmodel_params
    params['cost'] = cost_params
    params['optimizer'] = opt_params

    return params


def cartpole_loss(mx, Sx,
                  angle_dims=[3],
                  pole_length=0.5,
                  cw=[0.25],
                  target=np.array([0, 0, 0, np.pi]),
                  *args, **kwargs):
    target = np.array(target)
    D = target.size

    # convert angle dimensions
    targeta = utils.gTrig_np(target, angle_dims).flatten()
    Da = targeta.size
    if Sx is None:
        flatten = False
        if mx.ndim == 1:
            flatten = True
            mx = mx[None, :]
        mxa = utils.gTrig(mx, angle_dims, D)
        if flatten:
            # since we are dealing with one input vector at a time
            mxa = mxa.flatten()
        Sxa = None
    else:
        # angle dimensions are removed, and their complex representation
        # is appended
        mxa, Sxa = utils.gTrig2(mx, Sx, angle_dims, D)[:2]

    # build cost scaling function
    Q = np.zeros((Da, Da))
    Q[0, 0] = 1
    Q[0, -2] = pole_length
    Q[-2, 0] = pole_length
    Q[-2, -2] = pole_length**2
    Q[-1, -1] = pole_length**2

    return cost.generic_loss(mxa, Sxa, targeta, Q, cw, *args, **kwargs)


class Cartpole(plant.ODEPlant):

    metadata = {
        'render.modes': ['human']
    }

    def __init__(self, pole_length=0.5, pole_mass=0.5,
                 cart_mass=0.5, friction=0.1, gravity=9.82,
                 state0_dist=None,
                 loss_func=None,
                 name='Cartpole',
                 *args, **kwargs):
        super(Cartpole, self).__init__(name=name, *args, **kwargs)
        # cartpole system parameters
        self.l = pole_length
        self.m = pole_mass
        self.M = cart_mass
        self.b = friction
        self.g = gravity

        # initial state
        if state0_dist is None:
            self.state0_dist = utils.distributions.Gaussian(
                [0, 0, 0, 0], (0.1**2)*np.eye(4))
        else:
            self.state0_dist = state0_dist

        # reward/loss function
        if loss_func is None:
            self.loss_func = cost.build_loss_func(cartpole_loss, False,
                                                  'cartpole_loss')
        else:
            self.loss_func = cost.build_loss_func(loss_func, False,
                                                  'cartpole_loss')

        # pointer to the class that will draw the state of the carpotle system
        self.renderer = None

        # 4 state dims (x ,x_dot, theta_dot, theta)
        o_lims = np.array([np.finfo(np.float).max for i in range(4)])
        self.observation_space = spaces.Box(-o_lims, o_lims)
        # 1 action dim (x_force)
        a_lims = np.array([np.finfo(np.float).max for i in range(1)])
        self.action_space = spaces.Box(-a_lims, a_lims)

    def dynamics(self, t, z):
        l, m, M, b, g = self.l, self.m, self.M, self.b, self.g
        f = self.u if self.u is not None else np.array([0])

        sz, cz = np.sin(z[3]), np.cos(z[3])
        cz2 = cz*cz
        a0 = m*l*z[2]*z[2]*sz
        a1 = g*sz
        a2 = f[0] - b*z[1]
        a3 = 4*(M+m) - 3*m*cz2

        dz = np.zeros((4, 1))
        dz[0] = z[1]                                      # x
        dz[1] = (2*a0 + 3*m*a1*cz + 4*a2)/a3              # dx/dt
        dz[2] = -3*(a0*cz + 2*((M+m)*a1 + a2*cz))/(l*a3)  # dtheta/dt
        dz[3] = z[2]                                      # theta

        return dz

    def _reset(self):
        state0 = self.state0_dist.sample()
        self.set_state(state0)
        return self.state

    def _render(self, mode='human', close=False):
        if self.renderer is None:
            self.renderer = CartpoleDraw(self)
            self.renderer.init_ui()
        self.renderer.update(*self.get_state(noisy=False))

    def _close(self):
        if self.renderer is not None:
            self.renderer.close()


class CartpoleDraw(plant.PlantDraw):
    def __init__(self, cartpole_plant, refresh_period=(1.0/240),
                 name='CartpoleDraw'):
        super(CartpoleDraw, self).__init__(cartpole_plant,
                                           refresh_period, name)
        l = self.plant.l
        m = self.plant.m
        M = self.plant.M

        self.mass_r = 0.05*np.sqrt(m)  # distance to corner of bounding box
        self.cart_h = 0.5*np.sqrt(M)

        self.center_x = 0
        self.center_y = 0

        # initialize the patches to draw the cartpole
        cart_xy = (self.center_x-0.5*self.cart_h,
                   self.center_y-0.125*self.cart_h)
        self.cart_rect = plt.Rectangle(cart_xy, self.cart_h,
                                       0.25*self.cart_h, facecolor='black')
        self.pole_line = plt.Line2D((self.center_x, 0), (self.center_y, l),
                                    lw=2, c='r')
        self.mass_circle = plt.Circle((0, l), self.mass_r, fc='y')

    def init_artists(self):
        self.ax.add_patch(self.cart_rect)
        self.ax.add_patch(self.mass_circle)
        self.ax.add_line(self.pole_line)

    def _update(self, state, t, *args, **kwargs):
        l = self.plant.l

        cart_x = self.center_x + state[0]
        cart_y = self.center_y
        if self.plant.angle_dims:
            mass_x = l*state[3] + cart_x
            mass_y = -l*state[4] + cart_y
        else:
            mass_x = l*np.sin(state[3]) + cart_x
            mass_y = -l*np.cos(state[3]) + cart_y

        self.cart_rect.set_xy((cart_x-0.5*self.cart_h,
                               cart_y-0.125*self.cart_h))
        self.pole_line.set_xdata(np.array([cart_x, mass_x]))
        self.pole_line.set_ydata(np.array([cart_y, mass_y]))
        self.mass_circle.center = (mass_x, mass_y)

        return (self.cart_rect, self.pole_line, self.mass_circle)

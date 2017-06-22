# pylint: disable=C0103
'''
Contains the Cartpole envionment, along with default parameters and a rendering class
'''
import numpy as np
import theano
import theano.tensor as tt

from gym import spaces
from matplotlib import pyplot as plt

from kusanagi.shell.plant import ODEPlant, PlantDraw
from kusanagi.ghost.cost import generic_loss, build_loss_func
from kusanagi.utils import gTrig2
from kusanagi.ghost.control import RBFPolicy
from kusanagi.ghost.regression import GP_UI
from kusanagi import utils

def default_params():
    # setup learner parameters
    # general parameters
    learner_params = {}
    learner_params['x0'] = [0, 0, 0, 0]       # initial state mean
    learner_params['S0'] = np.eye(4)*(0.1**2) # initial state covariance
    learner_params['angle_dims'] = [3]        # angle dimensions
    learner_params['H'] = 4.0                 # control horizon
    learner_params['discount'] = 1.0          # discount factor
    # plant
    plant_params = {}
    plant_params['dt'] = 0.1
    plant_params['pole_length'] = 0.5
    plant_params['pole_mass'] = 0.5
    plant_params['cart_mass'] = 0.5
    plant_params['friction'] = 0.1
    plant_params['gravity'] = 9.82
    # model measurement noise (randomizes the output of the plant)
    plant_params['noise'] = np.diag(np.ones(len(learner_params['x0']))*0.01**2)

    # policy
    policy_params = {}
    policy_params['m0'] = learner_params['x0']
    policy_params['S0'] = learner_params['S0']
    policy_params['n_inducing'] = 30
    policy_params['maxU'] = [10]
    # dynamics model
    dynmodel_params = {}
    dynmodel_params['n_inducing'] = 100
    # cost function
    cost_params = {}
    cost_params['target'] = [0, 0, 0, np.pi]
    cost_params['cw'] = 0.25
    cost_params['expl'] = 0.0
    cost_params['pole_length'] = plant_params['pole_length']

    learner_params['max_evals'] = 150
    learner_params['conv_thr'] = 1e-12
    learner_params['min_method'] = 'BFGS'#utils.fmin_lbfgs
    learner_params['realtime'] = True

    learner_params['plant'] = plant_params
    learner_params['policy'] = policy_params
    learner_params['dynmodel'] = dynmodel_params
    learner_params['cost'] = cost_params

    return {'params': learner_params,
            'plant_class': Cartpole,
            'policy_class': RBFPolicy,
            'cost_func': cartpole_loss,
            'dynmodel_class': GP_UI}

def cartpole_loss(mx, Sx,
                  angle_dims=[3],
                  pole_length=0.5,
                  target=np.array([0, 0, 0, np.pi]),
                  *args, **kwargs):
    target = np.array(target)
    D = target.size

    #convert angle dimensions
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
        # angle dimensions are removed, and their complex representation is appended
        mxa, Sxa = gTrig2(mx, Sx, angle_dims, D)[:2]

    # build cost scaling function
    Q = np.zeros((Da, Da))
    Q[0, 0] = 1
    Q[0, -2] = pole_length
    Q[-2, 0] = pole_length
    Q[-2, -2] = pole_length**2
    Q[-1, -1] = pole_length**2

    return generic_loss(mxa, Sxa, targeta, Q, *args, **kwargs)

class Cartpole(ODEPlant):
    metadata = {
        'render.modes': ['human']
    }
    def __init__(self, pole_length=0.5, pole_mass=0.5,
                 cart_mass=0.5, friction=0.1, gravity=9.82,
                 state0=np.array([0.0, 0.0, 0.0, np.pi]), cov0=None,
                 loss_func=build_loss_func(cartpole_loss, False, 'cartpole_loss'),
                 *args, **kwargs):
        super(Cartpole, self).__init__(*args, **kwargs)
        # cartpole system parameters
        self.l = pole_length
        self.m = pole_mass
        self.M = cart_mass
        self.b = friction
        self.g = gravity
        # initial state
        self.state0 = state0
        self.cov0 = cov0
        # pointer to the class that will draw the state of the carpotle system
        self.renderer = None

        # 4 state dims (x ,x_dot, theta_dot, theta)
        o_lims = np.array([np.finfo(np.float).max for i in range(4)])
        self.observation_space = spaces.Box(-o_lims, o_lims)
        # 1 action dim (x_force)
        a_lims = np.array([np.finfo(np.float).max for i in range(1)])
        self.action_space = spaces.Box(-a_lims, a_lims)

        # reward/loss function
        self.loss_func = loss_func

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
        dz[0] = z[1]                                     # x
        dz[1] = (2*a0 + 3*m*a1*cz + 4*a2)/a3             # dx/dt
        dz[2] = -3*(a0*cz + 2*((M+m)*a1 + a2*cz))/(l*a3) # dtheta/dt
        dz[3] = z[2]                                     # theta

        return dz

    def _reset(self):
        state0 = self.state0
        if self.cov0 is not None:
            L0 = np.linalg.cholesky(self.cov0)
            state0 += np.random.randn(state0.size).dot(L0)
        self.set_state(state0)

    def _render(self, mode='human', close=False):
        if self.renderer is None:
            self.renderer = CartpoleDraw(self)
            self.renderer.init_ui()
        updts = self.renderer.update(*self.get_state())

    def _close(self):
        self.renderer.close()

class CartpoleDraw(PlantDraw):
    def __init__(self, cartpole_plant, refresh_period=(1.0/24), name='CartpoleDraw'):
        super(CartpoleDraw, self).__init__(cartpole_plant, refresh_period, name)
        l = self.plant.l
        m = self.plant.m
        M = self.plant.M

        self.mass_r = 0.05*np.sqrt(m) # distance to corner of bounding box
        self.cart_h = 0.5*np.sqrt(M)

        self.center_x = 0
        self.center_y = 0

        # initialize the patches to draw the cartpole
        cart_xy = (self.center_x-0.5*self.cart_h, self.center_y-0.125*self.cart_h)
        self.cart_rect = plt.Rectangle(cart_xy, self.cart_h, 0.25*self.cart_h, facecolor='black')
        self.pole_line = plt.Line2D((self.center_x, 0), (self.center_y, l), lw=2, c='r')
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

        self.cart_rect.set_xy((cart_x-0.5*self.cart_h, cart_y-0.125*self.cart_h))
        self.pole_line.set_xdata(np.array([cart_x, mass_x]))
        self.pole_line.set_ydata(np.array([cart_y, mass_y]))
        self.mass_circle.center = (mass_x, mass_y)

        return (self.cart_rect, self.pole_line, self.mass_circle)

# pylint: disable=C0103
'''
Contains the DoubleCartpole envionment, along with default parameters and
a rendering class
'''
import numpy as np

from gym import spaces
from matplotlib import pyplot as plt

from kusanagi.shell import plant
from kusanagi.shell import cost
from kusanagi import utils


def default_params():
    # setup learner parameters
    # initial state mean ( x, dx, dtheta1, dtheta2, theta1, theta2)
    x0 = np.array([0, 0, 0, 0, np.pi, np.pi])
    S0 = np.eye(len(x0))*(0.1**2)
    p0 = utils.distributions.Gaussian(x0, S0)
    angi = [4, 5]
    x0a, S0a = utils.gTrig2_np(x0[None, :], np.array(S0)[None, :, :],
                               angi, len(x0))

    # plant parameters
    plant_params = {}
    plant_params['dt'] = 0.1
    plant_params['link1_length'] = 0.6
    plant_params['link2_length'] = 0.6
    plant_params['link1_mass'] = 0.5
    plant_params['link2_mass'] = 0.5
    plant_params['cart_mass'] = 0.5
    plant_params['friction'] = 0.1
    plant_params['gravity'] = 9.82
    plant_params['state0_dist'] = p0
    plant_params['noise_dist'] = utils.distributions.Gaussian(
        np.zeros((p0.dim, )),
        np.eye(p0.dim)*0.01**2)

    # policy parameters
    policy_params = {}
    policy_params['state0_dist'] = p0
    policy_params['angle_dims'] = angi
    policy_params['n_inducing'] = 100
    policy_params['maxU'] = [20]

    # dynamics model parameters
    dynmodel_params = {}
    dynmodel_params['idims'] = x0a.size + len(policy_params['maxU'])
    dynmodel_params['odims'] = x0.size
    dynmodel_params['n_inducing'] = 100

    # cost function parameters
    cost_params = {}
    cost_params['angle_dims'] = angi
    cost_params['target'] = [0, 0, 0, 0, 0, 0]
    cost_params['cw'] = 0.5
    cost_params['expl'] = 0.0
    cost_params['link1_length'] = plant_params['link1_length']
    cost_params['link2_length'] = plant_params['link2_length']
    cost_params['loss_func'] = cost.quadratic_saturating_loss

    # optimizer params
    opt_params = {}
    opt_params['max_evals'] = 100
    opt_params['conv_thr'] = 1e-12
    opt_params['min_method'] = 'L-BFGS-B'

    # general parameters
    params = {}
    params['state0_dist'] = p0
    params['angle_dims'] = angi
    params['min_steps'] = int(3.0/plant_params['dt'])   # control horizon
    params['max_steps'] = int(7.5/plant_params['dt'])  # control horizon
    params['discount'] = 1.0                            # discount factor
    params['plant'] = plant_params
    params['policy'] = policy_params
    params['dynamics_model'] = dynmodel_params
    params['cost'] = cost_params
    params['optimizer'] = opt_params

    return params


def double_cartpole_loss(mx, Sx,
                         target=np.array([0, 0, 0, 0, 0, 0]),
                         angle_dims=[4, 5],
                         link1_length=0.5,
                         link2_length=0.5,
                         cw=[0.5],
                         *args, **kwargs):

    # size of target vector (and mx) after replacing angles with their
    # (sin, cos) representation:
    # [x1,x2,..., angle,...,xn] -> [x1,x2,...,xn, sin(angle), cos(angle)]
    Da = np.array(target).size + len(angle_dims)

    # build cost scaling function
    Q = np.zeros((Da, Da))
    # these are the dimensions used to compute the cost
    # (x, sin(theta1), cos(theta1), sin(theta2), cos(theta2))
    cost_dims = np.hstack([0, np.arange(Da-2*len(angle_dims), Da)])[:, None]
    C = np.array([[1, -link1_length, 0, -link2_length, 0],
                  [0, 0, link1_length, 0, link2_length]])
    Q[cost_dims, cost_dims.T] = C.T.dot(C)

    return cost.distance_based_cost(
        mx, Sx, target, Q, cw, angle_dims=angle_dims, *args, **kwargs)


class DoubleCartpole(plant.ODEPlant):

    metadata = {
        'render.modes': ['human']
    }

    def __init__(self,
                 link1_length=0.5, link1_mass=0.5,
                 link2_length=0.5, link2_mass=0.5,
                 cart_mass=0.5, friction=0.1, gravity=9.82,
                 state0_dist=None,
                 loss_func=None,
                 name='DoubleCartpole',
                 *args, **kwargs):
        super(DoubleCartpole, self).__init__(name=name, loss_func=loss_func,  *args, **kwargs)
        # double cartpole system parameters
        self.l1 = link1_length
        self.l2 = link2_length
        self.m1 = link1_mass
        self.m2 = link2_mass
        self.M = cart_mass
        self.b = friction
        self.g = gravity

        # initial state
        if state0_dist is None:
            m0, s0 = [0, 0, 0, 0, np.pi, np.pi], (0.1**2)*np.eye(6)
            self.state0_dist = utils.distributions.Gaussian(m0, s0)
        else:
            self.state0_dist = state0_dist

        # pointer to the class that will draw the state of the carpotle system
        self.renderer = None

        # 6 state dims (x, dx, dtheta1, dtheta2, theta1, theta2)
        o_lims = np.array([np.finfo(np.float).max for i in range(6)])
        self.observation_space = spaces.Box(-o_lims, o_lims)
        # 1 action dim (x_force)
        a_lims = np.array([np.finfo(np.float).max for i in range(1)])
        self.action_space = spaces.Box(-a_lims, a_lims)

    def dynamics(self, t, z):
        m1, m2, M, l1, l2, b, g = self.m1, self.m2, self.M,\
                                  self.l1, self.l2, self.b,\
                                  self.g

        f = self.u if self.u is not None else np.array([0])
        f = f.flatten()

        sz4 = np.sin(z[4])
        cz4 = np.cos(z[4])
        sz5 = np.sin(z[5])
        cz5 = np.cos(z[5])
        cz4m5 = np.cos(z[4] - z[5])
        sz4m5 = np.sin(z[4] - z[5])
        a0 = m2+2*M
        a1 = M*l2
        a2 = l1*(z[2]*z[2])
        a3 = a1*(z[3]*z[3])

        A = np.array([[2*(m1+m2+M), -a0*l1*cz4,     -a1*cz5],
                      [-3*a0*cz4,    (2*a0+2*M)*l1,  3*a1*cz4m5],
                      [-3*cz5,       3*l1*cz4m5,     2*l2]])
        b = np.array([2*f[0]-2*b*z[1]-a0*a2*sz4-a3*sz5,
                      3*a0*g*sz4 - 3*a3*sz4m5,
                      3*a2*sz4m5 + 3*g*sz5]).flatten()
        
        x = np.linalg.solve(A, b)

        dz = np.zeros((6,))
        dz[0] = z[1]
        dz[1] = x[0]
        dz[2] = x[1]
        dz[3] = x[2]
        dz[4] = z[2]
        dz[5] = z[3]

        return dz

    def reset(self):
        state0 = self.state0_dist()
        self.set_state(state0)
        return self.state

    def render(self, mode='human', close=False):
        if self.renderer is None:
            self.renderer = DoubleCartpoleDraw(self)
            self.renderer.init_ui()
        self.renderer.update(*self.get_state(noisy=False))

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


class DoubleCartpoleDraw(plant.PlantDraw):
    def __init__(self, double_cartpole_plant, refresh_period=(1.0/240),
                 name='DoubleCartpoleDraw'):
        super(DoubleCartpoleDraw, self).__init__(double_cartpole_plant,
                                                 refresh_period, name)
        m1 = self.plant.m1
        m2 = self.plant.m2
        M = self.plant.M
        l1 = self.plant.l1
        l2 = self.plant.l2

        self.body_h = 0.5*np.sqrt(m1)
        self.mass_r1 = 0.05*np.sqrt(m2)  # distance to corner of bounding box
        self.mass_r2 = 0.05*np.sqrt(M)   # distance to corner of bounding box

        self.center_x = 0
        self.center_y = 0

        # initialize the patches to draw the cartpole
        self.body_rect = plt.Rectangle((self.center_x - 0.5*self.body_h,
                                       self.center_y - 0.125*self.body_h),
                                       self.body_h, 0.25*self.body_h,
                                       facecolor='black')
        self.pole_line1 = plt.Line2D((self.center_x, 0),
                                     (self.center_y, l1), lw=2, c='r')
        self.mass_circle1 = plt.Circle((0, l1), self.mass_r1, fc='y')
        self.pole_line2 = plt.Line2D((self.center_x, 0),
                                     (l1, l2), lw=2, c='r')
        self.mass_circle2 = plt.Circle((0, l1+l2), self.mass_r2, fc='y')

    def init_artists(self):
        self.ax.add_patch(self.body_rect)
        self.ax.add_patch(self.mass_circle1)
        self.ax.add_line(self.pole_line1)
        self.ax.add_patch(self.mass_circle2)
        self.ax.add_line(self.pole_line2)

    def _update(self, state, t, *args, **kwargs):
        l1 = self.plant.l1
        l2 = self.plant.l2

        body_x = self.center_x + state[0]
        body_y = self.center_y
        mass1_x = -l1*np.sin(state[4]) + body_x
        mass1_y = l1*np.cos(state[4]) + body_y
        mass2_x = -l2*np.sin(state[5]) + mass1_x
        mass2_y = l2*np.cos(state[5]) + mass1_y

        self.body_rect.set_xy((body_x-0.5*self.body_h,
                               body_y-0.125*self.body_h))
        self.pole_line1.set_xdata(np.array([body_x, mass1_x]))
        self.pole_line1.set_ydata(np.array([body_y, mass1_y]))
        self.pole_line2.set_xdata(np.array([mass1_x, mass2_x]))
        self.pole_line2.set_ydata(np.array([mass1_y, mass2_y]))
        self.mass_circle1.center = (mass1_x, mass1_y)
        self.mass_circle2.center = (mass2_x, mass2_y)

        return (self.body_rect, self.pole_line1, self.mass_circle1,
                self.pole_line2, self.mass_circle2)

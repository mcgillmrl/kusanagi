import numpy as np
import theano
from shell.plant import ODEPlant, PlantDraw
from ghost.cost import quadratic_saturating_loss
from utils import print_with_stamp, gTrig_np, gTrig2

def default_params():
    # setup learner parameters
    # general parameters
    J = 4                                                                   # number of random initial trials
    N = 100                                                                 # learning iterations
    learner_params = {}
    learner_params['x0'] = [0,0,0,0]                                        # initial state mean
    learner_params['S0'] = np.eye(4)*(0.1**2)                               # initial state covariance
    learner_params['angle_dims'] = [3]                                      # angle dimensions
    learner_params['H'] = 4.0                                               # control horizon
    learner_params['discount'] = 1.0                                        # discoutn factor
    # plant
    plant_params = {}
    plant_params['dt'] = 0.1
    plant_params['params'] = {'l': 0.5, 'm': 0.5, 'M': 0.5, 'b': 0.1, 'g': 9.82}
    plant_params['noise'] = np.diag(np.ones(len(learner_params['x0']))*0.01**2)   # model measurement noise (randomizes the output of the plant)
    # policy
    policy_params = {}
    policy_params['m0'] = learner_params['x0']
    policy_params['S0'] = learner_params['S0']
    policy_params['n_basis'] = 10
    #policy_params['hidden_dims'] = [50,50,50]
    policy_params['maxU'] = [10]
    # dynamics model
    dynmodel_params = {}
    # cost function
    cost_params = {}
    cost_params['target'] = [0,0,0,np.pi]
    cost_params['width'] = 0.25
    cost_params['expl'] = 0.0
    cost_params['pendulum_length'] = plant_params['params']['l']

    learner_params['max_evals'] = 125
    learner_params['conv_thr'] = 1e-10
    learner_params['min_method'] = 'L-BFGS-B'
    learner_params['realtime'] = True

    learner_params['plant'] = plant_params
    learner_params['policy'] = policy_params
    learner_params['dynmodel'] = dynmodel_params
    learner_params['cost'] = cost_params
    from ghost.control import RBFPolicy
    from ghost.regression.GP import GP_UI

    return {'params': learner_params, 'plant_class': Cartpole, 'policy_class': RBFPolicy, 'cost_func': cartpole_loss, 'dynmodel_class': GP_UI}

def cartpole_loss(mx,Sx,params, loss_func=quadratic_saturating_loss, u=None):
    angle_dims = params['angle_dims']
    cw = params['width']
    if type(cw) is not list:
        cw = [cw]
    b = params['expl']
    ell = params['pendulum_length']
    target = np.array(params['target'])
    D = target.size
    
    #convert angle dimensions
    targeta = gTrig_np(target,angle_dims).flatten()
    Da = targeta.size
    mxa,Sxa,Ca = gTrig2(mx,Sx,angle_dims,D) # angle dimensions are removed, and their complex representation is appended
    # build cost scaling function
    Q = np.zeros((Da,Da))
    Q[0,0] = 1; Q[0,-2] = ell; Q[-2,0] = ell; Q[-2,-2] = ell**2; Q[-1,-1]=ell**2
    
    M_cost = [] ; S_cost = []
    
    # total cost is the sum of costs with different widths
    for c in cw:
        loss_params = {}
        loss_params['target'] = targeta
        loss_params['Q'] = Q/c**2
        m_cost, s_cost = loss_func(mxa,Sxa,loss_params)
        if b is not None and b != 0.0:
            m_cost += b*theano.tensor.sqrt(s_cost) # UCB  exploration term
        M_cost.append(m_cost)
        S_cost.append(s_cost)
    
    return sum(M_cost), sum(S_cost)

class Cartpole(ODEPlant):
    def __init__(self, params, x0, S0=None, dt=0.01, noise=None, name='Cartpole', integrator='dopri5', atol=1e-12, rtol=1e-12, angle_dims = []):
        super(Cartpole, self).__init__(params, x0, S0, dt=dt, noise=noise, name=name, integrator=integrator, atol=atol, rtol=rtol, angle_dims = angle_dims)

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
        dz[0] = z[1]                                                    # x
        dz[1] = (  2*a0 + 3*m*a1*cz + 4*a2 )/ ( a3 )                    # dx/dt
        dz[2] = -3*( a0*cz + 2*( (M+m)*a1 + a2*cz ) )/( l*a3 )          # dtheta/dt
        dz[3] = z[2]                                                    # theta

        return dz

    def dynamics_no_angles(self,t,z,u):
        l = self.params['l']
        m = self.params['m']
        M = self.params['M']
        b = self.params['b']
        g = self.params['g']
        f = u if u is not None else np.array([0])

        sz = z[3]; cz = z[4]; cz2 = cz*cz;
        a0 = m*l*z[2]*z[2]*sz
        a1 = g*sz
        a2 = f[0] - b*z[1];
        a3 = 4*(M+m) - 3*m*cz2
                                
        dz0 = z[1]                                                      # x
        dz1 = (  2*a0 + 3*m*a1*cz + 4*a2 )/ ( a3 )                      # dx/dt
        dz2 = -3*( a0*cz + 2*( (M+m)*a1 + a2*cz ) )/( l*a3 )            # dtheta/dt
        dz3 = cz*z[2]                                   # sin(theta)
        dz4 = -sz*z[2]                                   # cos(theta)
        dz = theano.tensor.stack([dz0,dz1,dz2,dz3,dz4])

        return dz*self.dt

class CartpoleDraw(PlantDraw):
    def __init__(self, cartpole_plant, refresh_period=(1.0/24), name='CartpoleDraw'):
        super(CartpoleDraw, self).__init__(cartpole_plant, refresh_period,name)
        if self.plant.params is not None:
            l = self.plant.params['l']
            m = self.plant.params['m']
            M = self.plant.params['M']
        else:
            l = 0.5
            m = 0.5
            M = 0.5

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
        if self.plant.angle_dims:
            mass_x = l*state[3] + body_x
            mass_y = -l*state[4] + body_y
        else:
            mass_x = l*np.sin(state[3]) + body_x
            mass_y = -l*np.cos(state[3]) + body_y

        self.body_rect.set_xy((body_x-0.5*self.body_h,body_y-0.125*self.body_h))
        self.pole_line.set_xdata(np.array([body_x,mass_x]))
        self.pole_line.set_ydata(np.array([body_y,mass_y]))
        self.mass_circle.center = (mass_x,mass_y)

        return (self.body_rect,self.pole_line, self.mass_circle)

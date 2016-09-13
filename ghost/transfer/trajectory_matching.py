import numpy as np 
import utils
from ghost.regression import GP
from ghost.learners.EpisodicLearner import EpisodicLearner
from ghost.learners.PILCO import PILCO

class TrajectoryMatching(PILCO):
    def __init__(self, params, plant_class, policy_class, cost_func=None, viz_class=None, dynmodel_class=GP.GP_UI, experience = None, async_plant=False, name='TrajectoryMatching', wrap_angles=False, filename_prefix=None):
        self.mx0 = np.array(params['x0']).squeeze()
        self.Sx0 = np.array(params['S0']).squeeze()
        self.angle_idims = params['angle_dims']
        self.maxU = params['policy']['maxU']
        # initialize source policy
        params['policy']['source_policy'] = params['source_policy']
        # initialize source policy
        self.source_experience = params['source_experience']
        
        # initialize dynamics model
        # input dimensions to the dynamics model are (state dims - angle dims) + 2*(angle dims) + control dims
        dyn_idims = len(self.mx0) + len(self.angle_idims) + len(self.maxU)
        # output dimensions are state dims
        dyn_odims = len(self.mx0)
        # initialize dynamics model (TODO pass this as argument to constructor)
        if 'inv_dynmodel' not in params:
            params['inv_dynmodel'] = {}
        params['inv_dynmodel']['idims'] = 2*dyn_odims
        params['inv_dynmodel']['odims'] = len(self.maxU)#dyn_odims

        self.inverse_dynamics_model = dynmodel_class(**params['inv_dynmodel'])
        self.next_episode = 0
        super(TrajectoryMatching, self).__init__(params, plant_class, policy_class, cost_func,viz_class, dynmodel_class,  experience, async_plant, name, filename_prefix)
    
    def set_source_domain(self):
        # TODO Set the source dynamics
        pass

    def set_target_domain(self):
        # TODO Set the source dynamics
        pass

    def train_inverse_dynamics(self):
        utils.print_with_stamp('Training inverse dynamics model',self.name)

        X = []
        Y = []
        x0 = []
        n_episodes = len(self.experience.states)
        
        if n_episodes>0:
            # construct training dataset
            for i in xrange(self.next_episode,n_episodes):
                x = np.array(self.experience.states[i])
                u = np.array(self.experience.actions[i])
                x0.append(x[0])

                # inputs are pairs of consecutive states < x_{t}, x_{t+1} >
                x_ = utils.gTrig_np(x, self.angle_idims)
                #X.append( np.hstack((x_[:-1],x[1:])) )
                X.append( np.hstack((x_[:-1],x_[1:])) )
                # outputs are the actions that produced the input state transition
                Y.append( u[:-1] )

            self.next_episode = n_episodes 
            X = np.vstack(X)
            Y = np.vstack(Y)
            
            # get distribution of initial states
            x0 = np.array(x0)
            if n_episodes > 1:
                self.mx0 = x0.mean(0)[None,:]
                self.Sx0 = np.cov(x0.T)[None,:,:]
            else:
                self.mx0 = x0[None,:]
                self.Sx0 = 1e-2*np.eye(self.mx0.size)[None,:,:]

            # append data to the dynamics model
            self.inverse_dynamics_model.append_dataset(X,Y)
        else:
            self.mx0 = np.array(self.plant.x0).squeeze()
            self.Sx0 = np.array(self.plant.S0).squeeze()

        utils.print_with_stamp('Dataset size:: Inputs: [ %s ], Targets: [ %s ]  '%(self.inverse_dynamics_model.X.get_value().shape,self.inverse_dynamics_model.Y.get_value().shape),self.name)
        if self.inverse_dynamics_model.should_recompile:
            # reinitialize log likelihood
            self.inverse_dynamics_model.init_loss()
 
        self.inverse_dynamics_model.train()
        utils.print_with_stamp('Done training inverse dynamics model',self.name)

    def train_adjustment(self):
        n_trajectories=5
        total_trajectories=len(self.source_experience.states)
        X = []
        Y = []
        Y_var = []
        
        for i in xrange(max(0,total_trajectories-n_trajectories),total_trajectories):
            # for every state transition in the source experience, use the target inverse dynamics
            # to find an action that would produce the desired transition
            x = np.array(self.source_experience.states[i])
            x_s = utils.gTrig_np(x, self.angle_idims)
            # source transitions
            t_s = np.hstack((x_s[:-1],x_s[1:])) 
            # source actions
            u_s = np.array(self.source_experience.actions[i])
            # target actions
            u_t = []
            Su_t = []
            for t_s_i in t_s:
                # get prediction from inverse dynamics model
                u, Su, Cu = self.inverse_dynamics_model.predict(t_s_i)
                u_t.append(np.random.multivariate_normal(u,Su))
                #u_t.append(u)
                Su_t.append(np.diag(np.maximum(Su,1e-9)))

            u_t = np.stack(u_t)
            Su_t = np.stack(Su_t)
            
            if self.policy.use_control_input:
                X.append( np.hstack( [x_s[:-1] , u_s[:-1]] ))
            else:
                X.append( x_s[:-1] )

            Y.append( u_t-u_s[:-1] )
            Y_var.append( Su_t )

        X = np.vstack(X)
        Y = np.vstack(Y)
        Y_var = np.vstack(Y_var)

        self.policy.adjustment_model.set_dataset(X,Y,Y_var=Y_var)
        self.policy.adjustment_model.train()

    def sample_trajectory_source(self, N=1):
        # TODO gather data using the source dynamics model or plant
        pass

    def sample_trajectory_target(self, N=1):
        # TODO gather data in the target domain
        pass



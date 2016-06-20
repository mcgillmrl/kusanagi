import numpy as np 
import utils
from ghost.regression.GPRegressor import GP_UI
from ghost.learners.EpisodicLearner import EpisodicLearner

class TrajectoryMatching(EpisodicLearner):
    def __init__(self, params, plant_class, policy_class, cost_func=None, viz_class=None, dynmodel_class=GP_UI, experience = None, async_plant=False, name='TrajectoryMatching', wrap_angles=False, filename_prefix=None):
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
        if 'dynmodel' not in params:
            params['dynmodel'] = {}
        params['dynmodel']['idims'] = dyn_idims
        params['dynmodel']['odims'] = dyn_odims

        self.inverse_dynamics_model = dynmodel_class(**params['dynmodel'])
        self.next_episode = 0
        super(TrajectoryMatching, self).__init__(params, plant_class, policy_class, cost_func,viz_class, experience, async_plant, name, filename_prefix)
    
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

        utils.print_with_stamp('Dataset size:: Inputs: [ %s ], Targets: [ %s ]  '%(self.inverse_dynamics_model.X_.shape,self.inverse_dynamics_model.Y_.shape),self.name)
        if self.inverse_dynamics_model.should_recompile:
            # reinitialize log likelihood
            self.inverse_dynamics_model.init_log_likelihood()
            # reinitialize rollot and policy gradients
            self.init_rollout(derivs=True)
 
        self.inverse_dynamics_model.train()
        utils.print_with_stamp('Done training inverse dynamics model',self.name)

    def train_adjustment(self):
        n_trajectories=10
        total_trajectories=len(self.source_experience.states)
        X = []
        Y = []
        
        for i in xrange(max(0,total_trajectories-n_trajectories),total_trajectories):
            x = np.array(self.source_experience.states[i])
            x_ = utils.gTrig_np(x, self.angle_idims)
            Xi = np.hstack((x_[:-1],x_[1:])) 
            ui = np.array(self.source_experience.actions[i])

            u = np.stack([ self.inverse_dynamics_model.predict(xi)[0] for xi in Xi ])
            
            if self.policy.use_control_input:
                X.append( np.hstack( [x_[:-1] , ui[:-1]] ))
            else:
                X.append( x_[:-1] )

            Y.append( u )

        X = np.vstack(X)
        Y = np.vstack(Y)

        self.policy.adjustment_model.set_dataset(X,Y)
        self.policy.adjustment_model.train()

    def sample_trajectory_source(self, N=1):
        # TODO gather data using the source dynamics model or plant
        pass

    def sample_trajectory_target(self, N=1):
        # TODO gather data in the target domain
        pass



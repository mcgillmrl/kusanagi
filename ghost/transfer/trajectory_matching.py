from ghost.regression.GPRegressor import GP_UI
from ghost.learners.EpisodicLearner import EpisodicLearner

class TrajectoryMatching(EpisodicLearner):
    def __init__(self, params, plant_class, policy_class, cost_func=None, viz_class=None, dynmodel_class=GP_UI, experience = None, async_plant=False, name='TrajectoryMatching', wrap_angles=False, filename_prefix=None):
        # initialize source policy
        params['policy']['source_policy'] = params['source_policy']
        super(TrajectoryMatching, self).__init__(params, plant_class, policy_class, cost_func,viz_class, experience, async_plant, name, filename_prefix)
    
    def set_source_domain(self):
        # TODO Set the source dynamics
        pass

    def set_target_domain(self):
        # TODO Set the source dynamics
        pass

    def train_inverse_dynamics(self):
        # TODO Use the dataset to train the inverse dynamics
        pass
    
    def train_adjustment(self):
        # TODO use the experience data
        pass

    def sample_trajectory_source(self, N=1):
        # TODO gather data using the source dynamics model or plant
        pass

    def sample_trajectory_target(self, N=1):
        # TODO gather data in the target domain
        pass



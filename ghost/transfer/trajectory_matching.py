from ghost.learners.EpisodicLearner import EpisodicLearner
from ghost.control import AdjustedPolicy

class TrajectoryMatching(EpisodicLearner):
    def __init__(self, target_plant, source_plant, source_policy, cost, angle_idims=None, discount=1, experience = None, async_plant=True, name='TrajectoryMatching'):
        super(TrajectoryMatching, self).__init__(target_plant, source_policy, cost, angle_idims, discount, experience, async_plant, name)
        # initialize source policy
        self.source_plant = source_plant
        self.source_policy = source_policy

        # initialize adjusted policy
        self.adjusted_policy = AdjustedPolicy(source_policy)
    
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



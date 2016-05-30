

class TrajectoryMatching(EpisodicLearner):
    def __init__(self, plant, source_policy, cost, angle_idims=None, discount=1, experience = None, async_plant=True, name='PILCO', wrap_angles=False):
        super(PILCO, self).__init__(plant, source_policy, cost, angle_idims, discount, experience, async_plant, name)
        # initialize source policy
        self.source_policy = source_policy

        # initialize adjusted policy
        self.adjusted_policy = AdjustedPolicy(source_policy)
    
    def set_source_domain(self):
        # TODO Set the source dynamics

    def set_target_domain(self):
        # TODO Set the source dynamics

    def train_inverse_dynamics(self):
        # TODO Use the dataset to train the inverse dynamics
    
    def train_adjustment(self):
        # TODO use the experience data

    def sample_trajectory_source(self, N=1):
        # TODO gather data using the source dynamics model or plant

    def sample_trajectory_target(self, N=1):
        # TODO gather data in the target domain



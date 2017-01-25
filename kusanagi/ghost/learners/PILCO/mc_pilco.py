from kusanagi.learners.PILCO import *

class MC_PILCO(PILCO):
    def __init__(self, params, plant_class, policy_class, cost_func=None, viz_class=None, dynmodel_class=GP.GP_UI, experience = None, async_plant=False, name='MC_PILCO', filename_prefix=None):
        super(MC_PILCO, self).__init__(params, plant_class, policy_class, cost_func,viz_class,, dynmodel_class, experience, async_plant, name, filename_prefix)

    def train_dynamics(self):

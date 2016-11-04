from ghost.learners.PILCO import PILCO


class PILCONN(PILCO):
    def __init__(self, params, plant_class, policy_class, cost_func=None, viz_class=None, dynmodel_class=NN, experience = None, async_plant=False, name='PILCONN', wrap_angles=False, filename_prefix=None):
        super(PILCONN,self)__init__(self, params, plant_class, policy_class, cost_func=None, viz_class=None, dynmodel_class=NN, experience = None, async_plant=False, name='PILCONN', wrap_angles=False, filename_prefix=None):
        

import numpy as np
from GP import GP2d

class conCat:
    def __init__(self, policy, saturating_function):
        self.con = policy
        self.sat= saturating_function
        self.GP = GP()
        pass


class conGP:
    def __init__(self, m0, S0, max_control=[10], n_basis_functions=10):
        self.maxU = np.array(maxU)
        # init policy inputs and targets
        self.inputs = np.random.multivariate_normal(m0,S0,n_basis_functions)
        self.targets = 0.1*np.random.random((n_basis_functions,len(max_control)))
        self.model = GP2d(self.inputs,self.targets)

        self.model.init()
        self.model.init_predict()

    def fcn(self,m,s=None):
        D = m[None,:].shape[-1]
        if s is None:
            s = np.zeros((D,D))
        M,S,C = self.model.predict(m,s)
        pass

# random controller
class conRand:
    def __init__(self,maxU=[10]):
        self.maxU = np.array(maxU)

    def fcn(self,m,s=None):
        return (2*np.random.random(self.maxU.shape)-1.0)*self.maxU

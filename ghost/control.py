import numpy as np
from regression.GPRegressor import RBFGP
from utils import print_with_stamp,gTrig_np

# GP based controller
class RBFPolicy:
    def __init__(self, m0, S0, maxU=[10], n_basis_functions=10, angle_idims=[]):
        self.maxU = np.array(maxU)
        self.angle_idims = angle_idims
        # init policy inputs  near the given initial state
        self.inputs = np.random.multivariate_normal(m0,S0,n_basis_functions)
        if len(self.angle_idims)>0:
            self.inputs = gTrig_np(self.inputs, self.angle_idims)

        # init policy targets close to zero
        self.targets = 0.1*np.random.random((n_basis_functions,len(maxU)))

        self.model = RBFGP(self.inputs,self.targets)

    def evaluate(self, t, m, s=None, derivs=False):
        D = m[None,:].shape[-1]
        if s is None:
            s = np.zeros((D,D))
        return self.model.predict(m,s) if not derivs else self.model.predict_d(m,s)

# random controller
class RandPolicy:
    def __init__(self, maxU=[10]):
        self.maxU = np.array(maxU)

    def evaluate(self, t, m, s=None, derivs=False):
        return (2*np.random.random(self.maxU.shape)-1.0)*self.maxU

import numpy as np
from ghost.regression.GPRegressor import RBFGP,GP_UI
from utils import print_with_stamp,gTrig_np, gTrig2_np
from saturation import gSat
from functools import partial

# GP based controller
class RBFPolicy:
    def __init__(self, m0, S0, maxU=[10], n_basis_functions=10, angle_idims=[]):
        self.maxU = np.array(maxU)
        self.angle_idims = angle_idims
        # init policy inputs  near the given initial state
        if len(self.angle_idims)>0:
            m0, S0 = gTrig2_np(np.array(m0)[None,:], np.array(S0)[None,:,:], self.angle_idims, len(m0))
            m0 = m0.squeeze(); S0 = S0.squeeze();

        #self.inputs = np.random.multivariate_normal(m0,S0,n_basis_functions)
        L_noise = np.linalg.cholesky(S0)
        self.inputs = np.array([m0 + np.random.randn(S0.shape[1]).dot(L_noise) for i in xrange(n_basis_functions)]);
        # init policy targets close to zero
        self.targets = 0.1*np.random.randn(n_basis_functions,len(maxU))
        sat_func = partial(gSat, e=maxU)
        self.model = RBFGP(self.inputs,self.targets,sat_func=sat_func)
        self.model.save()

    def evaluate(self, t, m, s=None, derivs=False):
        D = m.shape[0]
        if s is None:
            s = np.zeros((D,D))
        ret = self.model.predict(m,s) if not derivs else self.model.predict_d(m,s)
        return ret 

    def get_params(self, symbolic=True):
        return self.model.get_params(symbolic)

# random controller
class RandPolicy:
    def __init__(self, maxU=[10]):
        self.maxU = np.array(maxU)

    def evaluate(self, t, m, s=None, derivs=False):
        return ((2*np.random.random(self.maxU.size)-1.0)*self.maxU).reshape(self.maxU.shape)


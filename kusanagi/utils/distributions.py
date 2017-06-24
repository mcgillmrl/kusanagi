import numpy as np
class Distribution(object):
    '''
        Base class for distribution. Useful for estimating and sampling initial state distributions
    '''
    def fit(data):
        raise NotImplementedError

    def sample():
        raise NotImplementedError

    @property
    def dim(self):
        return self.__dim

    @dim.setter
    def dim(self, dim):
        self__dim = dim


class Gaussian(Distribution):
    def __init__(self, mean, cov):
        self.mean = np.array(mean)
        self.cov = np.array(cov)
        self.dim = mean.size

    @property
    def cov(self):
        return self.__cov

    @cov.setter
    def cov(self, cov):
        self.__cov = cov
        if cov is not None:
            assert cov.shape[0] == cov.shape[1]
            self.cov_chol = np.linalg.cholesky(cov)

    def sample(self, n_samples=1):
        return np.random.randn((n_samples,self.mean.size)).dot(self.cov).flatten()

import lasagne
import numpy as np
import theano

from kusanagi.ghost.regression import BNN
from kusanagi.ghost.control.saturation import tanhSat as sat
from functools import partial


# NN controller
class NNPolicy(BNN):
    def __init__(self, m0, maxU=[10], angle_dims=[], sat_func=sat,
                 name='NNPolicy', filename=None, **kwargs):
        self.maxU = np.array(maxU, dtype=theano.config.floatX)
        self.D = np.array(m0).size + len(angle_dims)
        self.E = len(maxU)

        if sat_func:
            self.sat_func = partial(sat_func, e=self.maxU)

        print(type(self), isinstance(self, NNPolicy))
        super(NNPolicy, self).__init__(self.D, self.E, name=name,
                                       filename=filename, **kwargs)

    def get_params(self, symbolic=True):
        if symbolic:
            return lasagne.layers.get_all_params(self.network,
                                                 trainable=True)
        else:
            return lasagne.layers.get_all_param_values(self.network,
                                                       trainable=True)

    def set_params(self, params):
        lasagne.layers.set_all_param_values(self.network, params,
                                            trainable=True)

    def predict_symbolic(self, mx, Sx=None, **kwargs):
        if self.network_spec is None:
            self.network_spec = self.get_default_network_spec(
                input_dims=self.D,
                output_dims=self.E,
                hidden_dims=[100]*2,
                nonlinearities=lasagne.nonlinearities.rectify,
                output_nonlinearity=self.sat_func,
                p=0.05, name=self.name)

        if self.network is None:
            params = self.network_params\
                     if self.network_params is not None\
                     else {}

            self.network = self.build_network(self.network_spec,
                                              params=params,
                                              name=self.name)

        ret = super(NNPolicy, self).predict_symbolic(mx, Sx, **kwargs)

        if Sx is None:
            if isinstance(ret, list) or isinstance(ret, tuple):
                ret = ret[0]
            M = ret
            return M
        else:
            M, S, V = ret
            return M, S, V

    def evaluate(self, m, s=None, t=None, symbolic=False, **kwargs):
        # by default, sample internal params (e.g. dropout masks)
        # at every evaluation
        kwargs['iid_per_eval'] = kwargs.get('iid_per_eval', True)
        kwargs['return_samples'] = kwargs.get('return_samples', True)
        kwargs['deterministic'] = kwargs.get('deterministic', False)
        if symbolic:
            ret = self.predict_symbolic(m, s, **kwargs)
        else:
            ret = self.predict(m, s)
        return ret

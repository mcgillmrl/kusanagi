import lasagne
import numpy as np
import theano

from kusanagi.ghost.regression import BNN, mlp, dropout_mlp, layers
from kusanagi.ghost.control.saturation import tanhSat as sat
from functools import partial


# NN controller
class NNPolicy(BNN):
    def __init__(self, input_dims, maxU=[10], angle_dims=[], sat_func=sat,
                 name='NNPolicy', filename=None, **kwargs):
        self.maxU = np.array(maxU, dtype=theano.config.floatX)
        self.angle_dims = angle_dims
        self.D = input_dims + len(self.angle_dims)
        self.E = len(maxU)

        if sat_func:
            self.sat_func = partial(sat_func, e=self.maxU)
        network_spec = kwargs.pop('network_spec', None)
        if type(network_spec) is dict:
            network_spec['output_nonlinearity'] = self.sat_func
        kwargs['network_spec'] = network_spec

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
            self.network_spec = dropout_mlp(
                input_dims=self.D,
                output_dims=self.E,
                hidden_dims=[50]*2,
                p=0.1, p_input=0.0,
                nonlinearities=lasagne.nonlinearities.rectify,
                output_nonlinearity=self.sat_func,
                dropout_class=layers.DenseDropoutLayer,
                name=self.name)

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
        kwargs['whiten_inputs'] = kwargs.get('whiten_inputs', False)
        kwargs['whiten_outputs'] = kwargs.get('whiten_outputs', False)
        if s is None:
            kwargs['return_samples'] = kwargs.get('return_samples', True)
        kwargs['deterministic'] = kwargs.get('deterministic', False)
        if symbolic:
            ret = self.predict_symbolic(m, s, **kwargs)
        else:
            ret = self.predict(m, s, **kwargs)
        return ret

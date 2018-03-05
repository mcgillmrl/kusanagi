import lasagne
import numpy as np
import theano

from kusanagi.ghost.regression import BNN, mlp, dropout_mlp, layers
from kusanagi.ghost.control.saturation import tanhSat as sat
from functools import partial


# NN controller
class NNPolicy(BNN):
    def __init__(self, input_dims, maxU=[10], minU=None, angle_dims=[], sat_func=sat,
                 name='NNPolicy', filename=None, **kwargs):
        self.maxU = np.array(maxU, dtype=theano.config.floatX)
        self.minU = (np.array(minU, dtype=theano.config.floatX)
                     if minU is not None else -self.maxU)
        self.angle_dims = angle_dims
        self.D = input_dims + len(self.angle_dims)
        self.E = len(maxU)

        if callable(sat_func):
            # set the model to be a RBF with saturated outputs
            maxU = self.maxU - self.minU
            sat_func = partial(sat_func, e=0.5*maxU)
            def sfunc(*args, **kwargs):
                return sat_func(*args, **kwargs) + 0.5*maxU + self.minU
            self.sat_func = sfunc

        network_spec = kwargs.pop('network_spec', None)
        if type(network_spec) is dict:
            network_spec['output_nonlinearity'] = self.sat_func
        kwargs['network_spec'] = network_spec

        super(NNPolicy, self).__init__(self.D, self.E, name=name,
                                       filename=filename, **kwargs)
   
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
            self.build_network(self.network_spec,
                               params=params,
                               name=self.name)

        return super(NNPolicy, self).predict_symbolic(mx, Sx, **kwargs)

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

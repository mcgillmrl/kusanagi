import lasagne
import numpy as np
import theano
import theano.tensor as tt

from kusanagi.ghost.regression import BNN
from functools import partial

# NN controller
class NNPolicy(BNN):
    def __init__(self, m0, maxU=[10], angle_dims=[], sat_func=None, name='NNPolicy', filename=None, **kwargs):
        self.maxU = np.array(maxU, dtype=theano.config.floatX)
        self.D = np.array(m0).size + len(angle_dims)
        self.E = len(maxU)
        def sat_func(u,e):
            from lasagne.nonlinearities import rectify, sigmoid, tanh, elu, linear, ScaledTanh
            return e*tanh(u)

        if sat_func:
            # set the model to be a RBF with saturated outputs
            self.sat_func = partial(sat_func, e=self.maxU)
        else:
            self.sat_func = None
        print(type(self), isinstance(self,NNPolicy))
        super(NNPolicy,self).__init__(self.D, self.E, name=name, filename=filename, **kwargs)

    def get_params(self, symbolic=True):
        if symbolic:
            return lasagne.layers.get_all_params(self.network,trainable=True)
        else:
            return lasagne.layers.get_all_param_values(self.network,trainable=True)

    def set_params(self,params):
        lasagne.layers.set_all_param_values(self.network,params,trainable=True)

    def get_network_spec(self,batchsize=None, input_dims=None, output_dims=None, hidden_dims=[200,200], p=0.05, p_input=0.0, name=None):
        from lasagne.layers import InputLayer, DenseLayer
        from kusanagi.ghost.regression.layers import DropoutLayer, relu, selu
        from lasagne.nonlinearities import rectify, sigmoid, tanh, elu, linear, ScaledTanh
        if name is None:
            name = self.name
        if input_dims is None:
            input_dims = self.D
        if output_dims is None:
            output_dims = self.E
        if not isinstance(p, list):
            p = [p]*len(hidden_dims)
        network_spec = []

        # input layer
        input_shape = [batchsize, input_dims]
        network_spec.append((InputLayer, dict(shape=input_shape, name=name+'_input')))
        if p_input > 0:
            network_spec.append((DropoutLayer,
                                 dict(p=p_input,
                                      rescale=False,
                                      name=name+'_drop_input',
                                      dropout_samples=self.dropout_samples.get_value()
                                     )
                                )
                               )
        # hidden layers
        for i in range(len(hidden_dims)):
            network_spec.append((DenseLayer,
                                 dict(num_units=hidden_dims[i],
                                      nonlinearity=elu,
                                      W=lasagne.init.HeNormal(gain=1),
                                      name=name+'_fc%d'%(i)
                                     )
                                )
                               )
            if p[i] > 0:
                network_spec.append((DropoutLayer, dict(p=p[i], rescale=False, name=name+'_drop%d'%(i), dropout_samples=self.dropout_samples.get_value())))
        # output layer
        network_spec.append((DenseLayer, dict(num_units=output_dims, nonlinearity=linear, W=lasagne.init.HeUniform(gain=1.0), name=name+'_output')))

        return network_spec

    def predict_symbolic(self,mx,Sx=None,**kwargs):
        if self.network_spec is None:
            self.network_spec = self.get_network_spec(input_dims=self.D,
                                                      output_dims=self.E,
                                                      hidden_dims=[50],
                                                      p=0.0, name=self.name)

        if self.network is None:
            params = self.network_params if self.network_params is not None else {}
            self.network = self.build_network(self.network_spec,
                                              params=params,
                                              name=self.name)

        ret = super(NNPolicy, self).predict_symbolic(mx, Sx, **kwargs)

        if Sx is None:
            if isinstance(ret, list):
                ret=ret[0]
            M = ret
            if self.sat_func is not None:
                # saturate the output
                M = self.sat_func(M)
            return M
        else:
            M, S, V = ret
            # apply saturating function to the output if available
            if self.sat_func is not None:
                # saturate the output
                M,S,U = self.sat_func(M,S)
                # compute the joint input output covariance
                V = V.dot(U)
            return M, S, V

    def evaluate(self, m, s=None, t=None, symbolic=False, **kwargs):
        D = m.shape[0]
        if symbolic:
            # by default, sample internal params (e.g. dropout masks) at every evaluation
            kwargs['iid_per_eval'] = kwargs.get('iid_per_eval',True)
            
            ret = self.predict_symbolic(m,s,**kwargs)
            theano.printing.Print('ret')(ret)
        else:
            ret = self.predict(m,s)
        return ret 

#!/usr/bin/env python2
import theano
import theano.tensor as tt
import lasagne
import numpy as np

from lasagne.layers import InputLayer, DenseLayer
from kusanagi.ghost.regression.layers import (
        DenseDropoutLayer, DenseGaussianDropoutLayer,
        DenseAdditiveGaussianDropoutLayer, DenseLogNormalDropoutLayer)
from lasagne import nonlinearities
from lasagne.random import get_rng
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from kusanagi import utils
from kusanagi.ghost.optimizers import SGDOptimizer
from kusanagi.ghost.regression import BaseRegressor
from kusanagi.ghost.regression import objectives

floatX = theano.config.floatX


def mlp(input_dims, output_dims, hidden_dims=[200]*4, batchsize=None,
        nonlinearities=nonlinearities.rectify,
        output_nonlinearity=nonlinearities.linear,
        W_init=lasagne.init.Orthogonal(),
        b_init=lasagne.init.Constant(0.),
        name='mlp', **kwargs):
    if not isinstance(nonlinearities, list):
        nonlinearities = [nonlinearities]*len(hidden_dims)
    if not isinstance(W_init, list):
        W_init = [W_init]*(len(hidden_dims)+1)
    if not isinstance(b_init, list):
        b_init = [b_init]*(len(hidden_dims)+1)
    network_spec = []
    # input layer
    input_shape = (batchsize, input_dims)
    network_spec.append((InputLayer,
                         dict(shape=input_shape,
                              name=name+'_input')))

    # hidden layers
    for i in range(len(hidden_dims)):
        layer_type = DenseLayer
        network_spec.append((layer_type,
                             dict(num_units=hidden_dims[i],
                                  nonlinearity=nonlinearities[i],
                                  W=W_init[i],
                                  b=b_init[i],
                                  name=name+'_fc%d' % (i))))
    # output layer
    layer_type = DenseLayer
    network_spec.append((layer_type,
                         dict(num_units=output_dims,
                              nonlinearity=output_nonlinearity,
                              W=W_init[-1],
                              b=b_init[-1],
                              name=name+'_output')))
    return network_spec


def dropout_mlp(input_dims, output_dims, hidden_dims=[200]*4, batchsize=None,
                nonlinearities=nonlinearities.rectify,
                output_nonlinearity=nonlinearities.linear,
                W_init=lasagne.init.Orthogonal(),
                b_init=lasagne.init.Constant(0.),
                p=0.5, p_input=0.2,
                dropout_class=DenseDropoutLayer,
                name='dropout_mlp'):
    if not isinstance(p, list):
        p = [p]*(len(hidden_dims))
    p = [p_input] + p

    network_spec = mlp(input_dims, output_dims, hidden_dims, batchsize,
                       nonlinearities, output_nonlinearity, W_init, b_init,
                       name)

    # first layer is input layer, so we skip that
    for i in range(len(p)):
        layer_class, layer_args = network_spec[i+1]
        if layer_class == DenseLayer and p[i] != 0:
            layer_args['p'] = p[i]
            network_spec[i+1] = (dropout_class, layer_args)
    return network_spec


class BNN(BaseRegressor):
    ''' Bayesian neural net regressor '''
    def __init__(self, idims, odims, n_samples=100,
                 heteroscedastic=False, name='BNN',
                 filename=None, **kwargs):
        self.D = idims
        self.E = odims
        self.name = name
        self.should_recompile = False
        self.trained = False

        sn = (np.ones((self.E,))*1e-3).astype(floatX)
        sn = np.log(np.exp(sn)-1)
        self.unconstrained_sn = theano.shared(sn, name='%s_sn' % (self.name))
        eps = np.finfo(np.__dict__[floatX]).eps
        self.sn = tt.nnet.softplus(self.unconstrained_sn) + eps

        self.network = None
        self.network_spec = None
        self.network_params = None

        sn = (np.ones((self.E,))*1e-3).astype(floatX)
        sn = np.log(np.exp(sn)-1)
        self.unconstrained_sn = theano.shared(sn, name='%s_sn' % (self.name))
        eps = np.finfo(np.__dict__[floatX]).eps
        self.sn = tt.nnet.softplus(self.unconstrained_sn) + eps
        samples = np.array(n_samples).astype('int32')
        samples_name = "%s>n_samples" % (self.name)
        self.n_samples = theano.shared(samples, name=samples_name)
        self.m_rng = RandomStreams(get_rng().randint(1, 2147462579))

        self.X = None
        self.Y = None
        self.Xm = None
        self.iXs = None
        self.Ym = None
        self.Ys = None

        self.heteroscedastic = heteroscedastic

        # filename for saving
        fname = '%s_%d_%d_%s_%s' % (self.name, self.D, self.E,
                                    theano.config.device,
                                    theano.config.floatX)
        self.filename = fname if filename is None else filename
        BaseRegressor.__init__(self, name=name, filename=self.filename)
        if filename is not None:
            self.load()

        # optimizer options
        max_evals = kwargs['max_evals'] if 'max_evals' in kwargs else 5000
        conv_thr = kwargs['conv_thr'] if 'conv_thr' in kwargs else 1e-12
        min_method = kwargs['min_method'] if 'min_method' in kwargs else 'ADAM'
        self.optimizer = SGDOptimizer(min_method, max_evals,
                                      conv_thr, name=self.name+'_opt')

        # register theano shared variables for saving
        self.register_types([tt.sharedvar.SharedVariable])
        self.register(['sn', 'network_params', 'network_spec'])

    def load(self, output_folder=None, output_filename=None):
        n_samples = self.n_samples.get_value()
        super(BNN, self).load(output_folder, output_filename)
        self.n_samples.set_value(n_samples)

        if self.network_params is None:
            self.network_params = {}

        if self.network_spec is not None:
            self.network = self.build_network(self.network_spec,
                                              params=self.network_params,
                                              name=self.name)
        if hasattr(self, 'unconstrained_sn'):
            eps = np.finfo(np.__dict__[floatX]).eps
            self.sn = tt.nnet.softplus(self.unconstrained_sn) + eps

    def save(self, output_folder=None, output_filename=None):
        # store references to the network shared variables, so we can save
        # and load them correctly
        self.network_params = []
        for layer in lasagne.layers.get_all_layers(self.network):
            layer_params = dict([(p.name.split('>')[-1], p)
                                 for p in layer.get_params()])
            self.network_params.append((layer.name, layer_params))
        self.network_params = dict(self.network_params)
        super(BNN, self).save(output_folder, output_filename)

    def get_intermediate_outputs(self):
        ret = super(BNN, self).get_intermediate_outputs()
        ret += lasagne.layers.get_all_params(self.network, unwrap_shared=True)
        for l in lasagne.layers.get_all_layers(self.network):
            if hasattr(l, 'get_intermediate_outputs'):
                ret += l.get_intermediate_outputs()
        return list(set(ret))

    def set_dataset(self, X_dataset, Y_dataset, **kwargs):
        # set dataset
        super(BNN, self).set_dataset(X_dataset.astype(floatX),
                                     Y_dataset.astype(floatX))

        # extra operations when setting the dataset (specific to this class)
        self.update_dataset_statistics(X_dataset, Y_dataset)

        if not self.trained:
            # default log of measurement noise variance is set to 2.5% of
            # dataset variation
            s = (0.05*Y_dataset.std(0)).astype(floatX)
            s = np.log(np.exp(s, dtype=floatX)-1.0)
            self.unconstrained_sn.set_value(s)

    def append_dataset(self, X_dataset, Y_dataset):
        # set dataset
        super(BNN, self).append_dataset(X_dataset.astype(floatX),
                                        Y_dataset.astype(floatX))

        # extra operations when setting the dataset (specific to this class)
        self.update_dataset_statistics(X_dataset, Y_dataset)

    def update_dataset_statistics(self, X_dataset, Y_dataset):
        Xm = X_dataset.mean(0).astype(floatX)
        Xc = np.cov(X_dataset, rowvar=False, ddof=1).astype(floatX)
        iXs = np.linalg.cholesky(
            np.linalg.inv(np.atleast_2d(Xc))).astype(floatX)
        if self.Xm is None:
            self.Xm = theano.shared(Xm, name='%s>Xm' % (self.name))
            self.iXs = theano.shared(iXs, name='%s>Xs' % (self.name))
        else:
            self.Xm.set_value(Xm)
            self.iXs.set_value(iXs)

        Ym = Y_dataset.mean(0).astype(floatX)
        Yc = np.cov(Y_dataset, rowvar=False, ddof=1).astype(floatX)
        Ys = np.linalg.cholesky(np.atleast_2d((Yc))).T.astype(floatX)
        if self.Ym is None:
            self.Ym = theano.shared(Ym, name='%s>Ym' % (self.name))
            self.Ys = theano.shared(Ys, name='%s>Ys' % (self.name))
        else:
            self.Ym.set_value(Ym)
            self.Ys.set_value(Ys)

    def build_network(self, network_spec=None, input_shape=None,
                      params={}, name=None):
        ''' Builds a network according to the specification in the
        network_spec argument. network_spec should be a list containing
        tuples where the first element is a class in lasagne.layers and the
        second element is a dictionary with jeyword arguments for the class
        constructor; i.e. [(layer_class_1, constructor_argss_1), ... ,
        (layer_class_N, constructor_argss_N)].
        Optionally, you can also pass a dictionary of parameters where the
        keys are 'layer_name' and the values are dictionaries with the
        trainable parameters indexed by names (e.g. W or b). The trainable
        parameter values should be numpy arrays, theano shared variables, or
        theano expressions to set the trainable parameters of the lasagne
        layers. e.g params = {'layer_name_1': {'W': theano_shared_1,
                                               'b': some_np_array}}
        would set the weights of layer 1 to be the shared variable
        theano_shared_1 and the biases to be initialized with the values of
        some_np_array.'''
        # set default values
        if name is None:
            name = self.name
        if network_spec is None:
            idims = self.D
            odims = self.E*2 if self.heteroscedastic else self.E
            network_spec = dropout_mlp(
                idims, odims, hidden_dims=[200]*3,
                p=0.1, p_input=0.0,
                dropout_class=DenseLogNormalDropoutLayer)
        utils.print_with_stamp('Building network', self.name)
        self.network_spec = network_spec

        # create input layer
        assert network_spec[0][0] is lasagne.layers.InputLayer\
            or input_shape is not None
        if network_spec[0][0] is lasagne.layers.InputLayer:
            if input_shape is not None:
                # change the input shape
                network_spec[0][1]['shape'] = input_shape
            layer_class, layer_args = network_spec[0]
            print(layer_class.__name__, layer_args)
            network = layer_class(**layer_args)
            network_spec = network_spec[1:]
        else:
            network = lasagne.layers.InputLayer(input_shape,
                                                name=name+'_input')

        # create hidden layers
        for layer_class, layer_args in network_spec:
            layer_name = layer_args['name']

            if layer_name in params:
                layer_args.update(params[layer_name])
            print(layer_class.__name__, layer_args)
            network = layer_class(network, **layer_args)
            # change the periods in variable names
            for p in network.get_params():
                p.name = p.name.replace('.', '>')

        # force rebuilding the prediction functions, as they will be
        # out of date
        self.predict_fn = None
        self.predict_ic_fn = None

        return network

    def get_loss(self):
        ''' initializes the loss function for training '''
        # build the network
        if self.network is None:
            params = self.network_params\
                     if self.network_params is not None\
                     else {}
            self.network = self.build_network(self.network_spec,
                                              params=params,
                                              name=self.name)

        utils.print_with_stamp('Initialising loss function', self.name)

        # Input variables
        input_lengthscale = tt.scalar('%s>input_lengthscale' % (self.name))
        hidden_lengthscale = tt.scalar('%s>hidden_lengthscale' % (self.name))
        train_inputs = tt.matrix('%s>train_inputs' % (self.name))
        train_targets = tt.matrix('%s>train_targets' % (self.name))

        # evaluate nework output for batch
        train_predictions, sn = self.predict_symbolic(
            train_inputs, None, deterministic=False,
            iid_per_eval=True, return_samples=True)

        # build the dropout loss function ( See Gal and Ghahramani 2015)
        M = train_targets.shape[0].astype(theano.config.floatX)
        N = self.X.shape[0].astype(theano.config.floatX)

        # compute negative log likelihood
        # note that if we have sn_std be a 1xD vector, broadcasting
        # rules apply
        lml = objectives.gaussian_log_likelihood(
            train_targets, train_predictions, sn)
        loss = -lml.sum()

        # compute regularization term
        # this is only for binary dropout layers
        input_ls = tt.minimum(M/N, input_lengthscale)
        hidden_ls = tt.minimum(M/N, hidden_lengthscale)
        loss += objectives.dropout_gp_kl(
            self.network, input_ls, hidden_ls)
        # this is only for gaussian dropout layers
        loss += objectives.gaussian_dropout_kl(
            self.network, input_ls, hidden_ls)
        # this is only for log normal dropout layers
        loss += objectives.log_normal_kl(
            self.network, input_ls, hidden_ls)

        inputs = [train_inputs, train_targets,
                  input_lengthscale, hidden_lengthscale]
        updates = theano.updates.OrderedUpdates()

        # get trainable network parameters
        params = lasagne.layers.get_all_params(self.network, trainable=True)

        # if we are learning the noise
        if not self.heteroscedastic:
            params.append(self.unconstrained_sn)
        self.set_params(dict([(p.name, p) for p in params]))
        return loss, inputs, updates

    def get_updates(self, network=None):
        ''' returns an updates dictionary, collected from layers in the
        networks that provide the get_updates method'''
        if network is None:
            network = self.network
        layer_updates = theano.updates.OrderedUpdates()
        for l in lasagne.layers.get_all_layers(network):
            if hasattr(l, 'get_updates'):
                layer_updates += l.get_updates()
        return layer_updates

    def predict_symbolic(self, mx, Sx=None, deterministic=False,
                         iid_per_eval=False, return_samples=False,
                         whiten_inputs=True, whiten_outputs=True):
        ''' returns symbolic expressions for the evaluations of this objects
        neural network. If Sx is specified, the output will correspond to the
        mean, covariance and input-output covariance of the network
        predictions'''
        if Sx is not None:
            # generate random samples from input (assuming gaussian
            # distributed inputs)
            # standard uniform samples (one sample per network sample)
            z_std = self.m_rng.normal((self.n_samples, self.D))

            # scale and center particles
            Lx = tt.slinalg.cholesky(Sx)
            x = mx + z_std.dot(Lx.T)
        else:
            x = mx[None, :] if mx.ndim == 1 else mx

        if (whiten_inputs and hasattr(self, 'Xm') and self.Xm is not None):
            # standardize inputs
            x = (x - self.Xm).dot(self.iXs)

        # unless we set the shared_axes parameter on the dropout layers,
        # the noise samples should be different per input sample
        ret = lasagne.layers.get_output(self.network, x,
                                        deterministic=deterministic,
                                        fixed_noise_samples=not iid_per_eval)
        y = ret[:, :self.E]
        sn = (tt.nnet.softplus(ret[:, self.E:])
              if self.heteroscedastic
              else tt.tile(self.sn, (y.shape[0], 1)))
        # fudge factor
        sn += 1e-6

        if whiten_outputs and hasattr(self, 'Ym') and self.Ym is not None:
            # scale and center outputs
            y = y.dot(self.Ys) + self.Ym
            # rescale variances (I don't think this is necessary)
            # sn = sn.dot(self.Ys)

        y.name = '%s>output_samples' % (self.name)
        if return_samples:
            # nothing else to do!
            return y, sn

        n = tt.cast(y.shape[0], dtype=theano.config.floatX)
        # empirical mean
        M = y.mean(axis=0)
        # empirical covariance
        S = y.T.dot(y)/n - tt.outer(M, M)
        # noise
        S += tt.diag((sn**2).mean(axis=0))

        # empirical input output covariance
        if Sx is not None:
            C = x.T.dot(y)/n - tt.outer(mx, M)
        else:
            C = tt.zeros((self.D, self.E))

        return [M, S, C]

    def update(self, n_samples=None):
        ''' Updates the dropout masks'''
        if n_samples is not None:
            if isinstance(n_samples, tt.sharedvar.SharedVariable):
                self.n_samples = n_samples
                self.update_fn = None
            else:
                self.n_samples.set_value(n_samples)

            # increase the size of the masks
            updts = self.get_updates()
            for v, updt in updts.items():
                v_shape = v.shape.eval()
                if v_shape[0] != n_samples:
                    v_shape[0] = n_samples
                    v.set_value(np.zeros(v_shape, dtype=v.dtype))

        if not hasattr(self, 'update_fn') or self.update_fn is None:
            # get prediction with non deterministic samples
            mx = tt.zeros((self.n_samples, self.D))
            self.predict_symbolic(mx, iid_per_eval=False)

            # create a function to update the masks manually. Here the dropout
            # masks should be shared variables
            updts = self.get_updates()
            self.update_fn = theano.function([], [], updates=updts,
                                             allow_input_downcast=True)

        # draw samples from the networks
        self.update_fn()

    def train(self, batch_size=100,
              input_ls=None, hidden_ls=None, lr=1e-4,
              optimizer=None, callback=None):
        if optimizer is None:
            optimizer = self.optimizer

        if optimizer.loss_fn is None or self.should_recompile:
            loss, inps, updts = self.get_loss()
            # we pass the learning rate as an input, and as a parameter to the
            # updates method
            learning_rate = theano.tensor.scalar('lr')
            inps.append(learning_rate)
            optimizer.set_objective(loss, self.get_params(symbolic=True),
                                    inps, updts, learning_rate=learning_rate)
        if input_ls is None:
            # by default, be less strict with the input layer
            input_ls = 1.0

        if hidden_ls is None:
            hidden_ls = 1.0

        optimizer.minibatch_minimize(self.X.get_value(), self.Y.get_value(),
                                     input_ls, hidden_ls, lr,
                                     batch_size=batch_size)
        self.trained = True
        self.update()

#!/usr/bin/env python2
import theano
import theano.tensor as tt
import lasagne
import numpy as np
import kusanagi

from collections import OrderedDict
from lasagne.random import get_rng
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from kusanagi import utils
from kusanagi.ghost.optimizers import SGDOptimizer
from kusanagi.ghost.regression import BaseRegressor

floatX = theano.config.floatX


class BNN(BaseRegressor):
    ''' Inefficient implementation of the dropout idea by Gal and Gharammani,
     with Gaussian distributed inputs'''
    def __init__(self, idims, odims, n_samples=25,
                 heteroscedastic=True, name='BNN',
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
        self.Xs = None
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
        max_evals = kwargs['max_evals'] if 'max_evals' in kwargs else 1000
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
        if self.Xm is None:
            Xm = X_dataset.mean(0).astype(floatX)
            self.Xm = theano.shared(Xm, name='%s>Xm' % (self.name))
            Xs = X_dataset.std(0).astype(floatX)
            self.Xs = theano.shared(Xs, name='%s>Xs' % (self.name))
        else:
            self.Xm.set_value(X_dataset.mean(0).astype(floatX))
            self.Xs.set_value(X_dataset.std(0).astype(floatX))

        if self.Ym is None:
            Ym = Y_dataset.mean(0).astype(floatX)
            self.Ym = theano.shared(Ym, name='%s>Ym' % (self.name))
            Ys = Y_dataset.std(0).astype(floatX)
            self.Ys = theano.shared(Ys, name='%s>Ys' % (self.name))
        else:
            self.Ym.set_value(Y_dataset.mean(0).astype(floatX))
            self.Ys.set_value(Y_dataset.std(0).astype(floatX))

    def get_default_network_spec(self, batchsize=None, input_dims=None,
                                 output_dims=None,
                                 hidden_dims=[200, 200],
                                 p=0.05, p_input=0.0,
                                 nonlinearities=lasagne.nonlinearities.elu,
                                 name=None):
        from lasagne.layers import InputLayer, DenseLayer
        from kusanagi.ghost.regression.layers import DropoutLayer
        from lasagne.nonlinearities import linear
        if name is None:
            name = self.name
        if input_dims is None:
            input_dims = self.D
        if output_dims is None:
            output_dims = 2*self.E if self.heteroscedastic else self.E
        if not isinstance(p, list):
            p = [p]*len(hidden_dims)
        if not isinstance(nonlinearities, list):
            nonlinearities = [nonlinearities]*len(hidden_dims)
        n_samples = self.n_samples.get_value()
        network_spec = []
        # input layer
        input_shape = (batchsize, input_dims)
        network_spec.append((InputLayer,
                             dict(shape=input_shape,
                                  name=name+'_input')))
        if p_input > 0:
            network_spec.append((DropoutLayer,
                                 dict(p=p_input,
                                      rescale=False,
                                      name=name+'_drop_input',
                                      n_samples=n_samples)))
        # hidden layers
        for i in range(len(hidden_dims)):
            network_spec.append((DenseLayer,
                                 dict(num_units=hidden_dims[i],
                                      nonlinearity=nonlinearities[i],
                                      name=name+'_fc%d' % (i))))
            if p[i] > 0:
                network_spec.append((DropoutLayer,
                                     dict(p=p[i],
                                          rescale=False,
                                          name=name+'_drop%d' % (i),
                                          n_samples=n_samples)))
        # output layer
        network_spec.append((DenseLayer,
                             dict(num_units=output_dims,
                                  nonlinearity=linear,
                                  name=name+'_output')))

        return network_spec

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
            network_spec = self.get_default_network_spec()
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
            iid_per_eval=True, return_samples=True,
            with_measurement_noise=True)

        # build the dropout loss function ( See Gal and Ghahramani 2015)
        deltay = train_predictions - train_targets
        N, E = train_targets.shape
        N = N.astype(theano.config.floatX)
        E = E.astype(theano.config.floatX)

        # compute negative log likelihood
        # note that if we have sn_std be a 1xD vector, broadcasting
        # rules apply
        nlml = 0.5*tt.square(deltay/sn).sum(-1) + 2*tt.log(sn).sum(-1)
        loss = nlml.mean()

        # compute regularization term
        loss += self.get_regularization_term(input_lengthscale,
                                             hidden_lengthscale)/N

        inputs = [train_inputs, train_targets,
                  input_lengthscale, hidden_lengthscale]
        updates = {}

        # get trainable network parameters
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        # if we are learning the noise
        if not self.heteroscedastic:
            params.append(self.unconstrained_sn)
        self.set_params(dict([(p.name, p) for p in params]))
        return loss, inputs, updates

    def get_regularization_term(self, input_lengthscale=1,
                                hidden_lengthscale=1):
        reg = 0
        layers = lasagne.layers.get_all_layers(self.network)

        for i in range(1, len(layers)):
            reg_weight = 1/2.0
            # apply different regularization weigths to the input,
            # and the hidden dimension
            is_dropout = isinstance(layers[i-1],
                                    lasagne.layers.DropoutLayer)
            ind = i if not (i > 1 and is_dropout) else i-1

            is_input = isinstance(layers[ind].input_layer,
                                  lasagne.layers.InputLayer)

            if hasattr(layers[ind], 'input_layer') and is_input:
                reg_weight *= input_lengthscale**2
            else:
                reg_weight *= hidden_lengthscale**2

            # if this layer has a weight layer and the previous layer
            # is a DropoutLayer
            if hasattr(layers[i], 'W'):
                if i > 1 and is_dropout:
                    p = layers[i-1].p
                    if p > 0:
                        reg_weight *= (1-p)
                reg += reg_weight*lasagne.regularization.l2(layers[i].W)

            if hasattr(layers[i], 'b'):
                reg += reg_weight*lasagne.regularization.l2(layers[i].b)

        return reg

    def get_updates(self, network=None):
        ''' returns a dictionary where the keys are lasagne layer instances
            and the values are their corresponding dropout masks'''
        if network is None:
            network = self.network
        mask_updates = []
        for l in lasagne.layers.get_all_layers(network):
            if isinstance(l, kusanagi.ghost.regression.layers.DropoutLayer)\
               and l.mask_updates is not None:
                mask_updates.append((l.mask, l.mask_updates))

        return mask_updates

    def predict_symbolic(self, mx, Sx=None, deterministic=False,
                         iid_per_eval=False, return_samples=False,
                         with_measurement_noise=True):
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

        if hasattr(self, 'Xm') and self.Xm is not None:
            # standardize inputs
            x = (x - self.Xm)/self.Xs
        # unless we set the shared_axes parameter on the droput layers,
        # the dropout masks should be different per sample
        ret = lasagne.layers.get_output(self.network, x,
                                        deterministic=deterministic,
                                        fixed_dropout_masks=not iid_per_eval)
        y = ret[:, :self.E]
        sn = (tt.nnet.softplus(ret[:, self.E:])
              if self.heteroscedastic
              else tt.tile(self.sn, (y.shape[0], 1)))

        if hasattr(self, 'Ym') and self.Ym is not None:
            # scale and center outputs
            y = y*self.Ys + self.Ym
            # rescale noise variance
            sn = sn*self.Ys

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
        if with_measurement_noise:
            S += tt.diag(sn.mean(axis=0)**2)

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
            for mask, updt in updts:
                mask_shape = mask.shape.eval()
                if mask_shape[0] != n_samples:
                    mask_shape[0] = n_samples
                    mask.set_value(np.zeros(mask_shape, dtype=mask.dtype))

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
              input_ls=None, hidden_ls=None, lr=1e-3,
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
            # set to some proportion of the standard deviation
            # (inputs are scaled and centered to N(0,1) )
            input_ls = 0.05

        if hidden_ls is None:
            hidden_ls = input_ls

        optimizer.minibatch_minimize(self.X.get_value(), self.Y.get_value(),
                                     input_ls, hidden_ls, lr,
                                     batch_size=batch_size)
        self.trained = True


def build_mlp(input_dims,
              output_dims,
              batchsize=None,
              hidden_dims=[200, 200],
              nonlin=lasagne.nonlinearities.elu,
              W_init=lasagne.init.GlorotUniform(),
              b_init=lasagne.init.Constant(0.),
              output_nonlin=lasagne.nonlinearities.linear,
              name=None):
    if name is None:
        name = 'mlp'
    if not isinstance(input_dims, list):
        input_dims = [input_dims]
    if not isinstance(nonlin, list):
        nonlin = [nonlin]*len(hidden_dims)
    if not isinstance(W_init, list):
        W_init = [W_init]*(len(hidden_dims)+1)
    if not isinstance(b_init, list):
        b_init = [b_init]*(len(hidden_dims)+1)

    # input layer
    net = lasagne.layers.InputLayer(
        [batchsize, *input_dims], name=name+'_input')

    # hidden layers
    for i in range(len(hidden_dims)):
        net = lasagne.layers.DenseLayer(
            net, hidden_dims[i], W_init[i], b_init[i], nonlin[i],
            name=name+"_fc%d" % i)

    # output layer
    net = lasagne.layers.DenseLayer(
        net, output_dims, W_init[-1], b_init[-1], output_nonlin,
        name=name+"_output")

    return net


def whiten(x, mean, inv_cov):
    '''
        Transforms input data to have zero mean, and identity covariance.
    '''
    if inv_cov.ndim != 2:
        return (x - mean)*inv_cov
    else:
        return (x - mean).dot(inv_cov)


def gaussian_log_likelihood(inputs, targets, 
                            pred_mean, pred_std=None):
    ''' Computes the eempirical expected value of the log likelihood,
    for a gaussian distributed predictions. This assumes diagonal covariances
    '''
    delta = pred_mean - targets
    # note that if we have nois be a 1xD vector, broadcasting
    # rules apply
    if pred_std:
        lml = tt.square(delta/pred_std).sum(-1) + tt.log(pred_std).sum(-1)
    else:
        lml = tt.square(delta).sum(-1)

    E_lml = -0.5*lml.mean()

    return E_lml


class NormalInit(object):
    def __init__(self, mean=0, std=1,
                 init_mean=lasagne.init.GlorotUniform(),
                 rng=RandomStreams(get_rng().randint(1, 2147462579)),
                 name='Normal'):
        self.mean = mean
        self.std = std
        self.rng = rng
        self.init_mean = init_mean
        self.updates = OrderedDict()
        self.name = name
        self.count = 0

    def __call__(self, shape):
        # create random variable
        z = theano.shared(
            floatX(np.random.normal(self.mean, self.std, size=shape)),
            name='%s_z_%d' % (self.name, self.count))
        z_updt = self.rng.normal(shape)
        # we will use this to control exactly when we should draw new samples
        self.updates[z] = z_updt

        # create params for mean and covariance
        # mean initialized around 0
        W_mean = theano.shared(
            self.init_mean(shape), name='%s_mean_%d' % (self.name, self.count))
        # variances initialized around 1
        W_logstd = theano.shared(
            np.random.normal(1, 0.05, size=shape),
            name='%s_std_%d' % (self.name, self.count))

        self.count += 1

        return W_mean + tt.nnet.softplus(W_logstd)*z


class BBB_NN(BaseRegressor):
    def __init__(self, idims, odims, network_init=build_mlp, n_samples=25,
                 heteroscedastic=False, name='BNN',
                 filename=None, **kwargs):
        self.D = idims
        self.E = odims
        utils.print_with_stamp("Initializing network", name)
        self.network = network_init(
            W_init=NormalInit(name=self.name+'_W'))

        # number of samples for the monte carlo integration
        samples = np.array(n_samples).astype('int32')
        samples_name = "%s>n_samples" % (self.name)
        self.n_samples = theano.shared(samples, name=samples_name)

        self.heteroscedastic = heteroscedastic
        self.name = name
        self.filename = filename

        # output measurement noise
        if not self.heteroscedastic:
            sn = (np.ones((self.E,))*1e-3).astype(floatX)
            sn = np.log(np.exp(sn)-1)
            self.unconstrained_sn = theano.shared(
                sn, name='%s_sn' % (self.name))
            # contrained to positive values
            eps = np.finfo(np.__dict__[floatX]).eps
            self.sn = tt.nnet.softplus(self.unconstrained_sn) + eps

        # random number generator
        seed = lasagne.random.get_rng().randint(1, 2147462579)
        self.m_rng = theano.sandbox.rng_mrg.MRG_RandomStreams(seed)

        # filename for saving
        fname = '%s_%d_%d_%s_%s' % (self.name, self.D, self.E,
                                    theano.config.device,
                                    theano.config.floatX)
        self.filename = fname if filename is None else filename
        BaseRegressor.__init__(self, name=name, filename=self.filename)
        if filename is not None:
            self.load()

    def set_dataset(self, X_dataset, Y_dataset, **kwargs):
        # set dataset
        super(BNN, self).set_dataset(X_dataset.astype(floatX),
                                     Y_dataset.astype(floatX))

        # extra operations when setting the dataset (specific to this class)
        self.update_dataset_statistics(X_dataset, Y_dataset)

        if not self.heteroscedastic and not self.trained:
            # default log of measurement noise std is set to 5% of
            # dataset variation
            s = (0.05*Y_dataset.std(0)).astype(floatX)
            s = np.log(np.exp(s, dtype=floatX)-1.0)
            self.unconstrained_sn.set_value(s)

    def get_loss(self):
        ''' initializes the loss function for training '''
        # build the network
        utils.print_with_stamp('Initialising loss function', self.name)

        # Input variables
        input_lengthscale = tt.scalar('%s>input_lengthscale' % (self.name))
        hidden_lengthscale = tt.scalar('%s>hidden_lengthscale' % (self.name))
        train_inputs = tt.matrix('%s>train_inputs' % (self.name))
        train_targets = tt.matrix('%s>train_targets' % (self.name))

        # standardize network inputs and outputs
        train_inputs_std = (train_inputs - self.Xm)/self.Xs
        train_targets_std = (train_targets - self.Ym)/self.Ys

        # evaluate nework output for batch
        if self.heteroscedastic:
            # this assumes that self.sn is obtained from the network output
            train_predictions, sn = lasagne.layers.get_output(
                [self.network], train_inputs_std)
        else:
            train_predictions = lasagne.layers.get_output(
                self.network, train_inputs_std, deterministic=False)
            sn = self.sn

        # scale sn since output network output is standardized
        sn_std = sn/self.Ys

        # build the dropout loss function ( See Gal and Ghahramani 2015)
        deltay = train_predictions-train_targets_std
        N, E = train_targets.shape
        N = N.astype(theano.config.floatX)
        E = E.astype(theano.config.floatX)

        # compute negative log likelihood
        # note that if we have sn_std be a 1xD vector, broadcasting
        # rules apply
        nlml = tt.square(deltay/sn_std).sum(-1) + tt.log(sn_std).sum(-1)
        loss = 0.5*nlml.mean()

        # compute regularization term
        loss += self.get_regularization_term(input_lengthscale,
                                             hidden_lengthscale)/N

        inputs = [train_inputs, train_targets,
                  input_lengthscale, hidden_lengthscale]
        updates = {}

        # get trainable network parameters
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        # if we are learning the noise
        if not self.heteroscedastic:
            params.append(self.unconstrained_sn)
        self.set_params(dict([(p.name, p) for p in params]))
        return loss, inputs, updates

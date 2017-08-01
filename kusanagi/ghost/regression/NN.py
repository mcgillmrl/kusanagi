#!/usr/bin/env python2
import theano
import theano.tensor as tt
import lasagne
import numpy as np
import kusanagi
from kusanagi import utils
from kusanagi.ghost.optimizers import SGDOptimizer
from kusanagi.ghost.regression import BaseRegressor


class BNN(BaseRegressor):
    ''' Inefficient implementation of the dropout idea by Gal and Gharammani,
     with Gaussian distributed inputs'''
    def __init__(self, idims, odims, dropout_samples=25, learn_noise=True,
                 heteroscedastic=False, name='BNN', profile=False,
                 filename=None, **kwargs):
        self.D = idims
        self.E = odims
        self.name = name
        self.should_recompile = False
        self.trained = False

        logsn = (np.ones((self.E,))*np.log(1e-3)).astype(theano.config.floatX)
        self.logsn = theano.shared(logsn, name='%s_logsn' % (self.name))

        self.network = None
        self.network_spec = None
        self.network_params = None
        self.sample_network_fn = None
        self.predict_fn = None
        self.prediction_updates = None
        samples = np.array(dropout_samples).astype('int32')
        samples_name = "%s>dropout_samples" % (self.name)
        self.dropout_samples = theano.shared(samples, name=samples_name)
        seed = lasagne.random.get_rng().randint(1, 2147462579)
        self.m_rng = theano.sandbox.rng_mrg.MRG_RandomStreams(seed)

        self.X = None
        self.Y = None
        self.Xm = None
        self.Xs = None
        self.Ym = None
        self.Ys = None

        self.profile = profile
        self.compile_mode = theano.compile.get_default_mode()

        self.learn_noise = learn_noise
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
        self.register(['logsn', 'network_params', 'network_spec'])

    def load(self, output_folder=None, output_filename=None):
        dropout_samples = self.dropout_samples.get_value()
        super(BNN, self).load(output_folder, output_filename)
        self.dropout_samples.set_value(dropout_samples)

        if self.network_params is None:
            self.network_params = {}

        if self.network_spec is not None:
            self.network = self.build_network(self.network_spec,
                                              params=self.network_params,
                                              name=self.name)

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

    def get_all_shared_vars(self, as_dict=False):
        if as_dict:
            v = [(attr_name, self.__dict__[attr_name])
                 for attr_name in list(self.__dict__.keys())
                 if isinstance(self.__dict__[attr_name],
                               tt.sharedvar.SharedVariable)]
            v += [(p.name, p)
                  for p in lasagne.layers.get_all_params(self.network,
                                                         unwrap_shared=True)]
            return dict(v)
        else:
            v = [attr
                 for attr in list(self.__dict__.values())
                 if isinstance(attr, tt.sharedvar.SharedVariable)]
            v += lasagne.layers.get_all_params(self.network,
                                               unwrap_shared=True)
            return v
    
    def set_dataset(self, X_dataset, Y_dataset, **kwargs):
        # set dataset
        super(BNN, self).set_dataset(X_dataset.astype(theano.config.floatX),
                                     Y_dataset.astype(theano.config.floatX))

        # extra operations when setting the dataset (specific to this class)
        self.update_dataset_statistics(X_dataset, Y_dataset)
        
        if self.learn_noise:
            # default log of measurement noise variance is set to 10% of
            # dataset variation
            logs = np.log((0.05*Y_dataset.std(0)).astype(theano.config.floatX))
            self.logsn.set_value(logs)

    def append_dataset(self, X_dataset, Y_dataset):
        # set dataset
        super(BNN, self).append_dataset(X_dataset.astype(theano.config.floatX),
                                        Y_dataset.astype(theano.config.floatX))

        # extra operations when setting the dataset (specific to this class)
        self.update_dataset_statistics(X_dataset, Y_dataset)

    def update_dataset_statistics(self, X_dataset, Y_dataset):
        if self.Xm is None:
            Xm = X_dataset.mean(0).astype(theano.config.floatX)
            self.Xm = theano.shared(Xm, name='%s>Xm' % (self.name))
            Xs = X_dataset.std(0).astype(theano.config.floatX)
            self.Xs = theano.shared(Xs, name='%s>Xs' % (self.name))
        else:
            self.Xm.set_value(X_dataset.mean(0).astype(theano.config.floatX))
            self.Xs.set_value(X_dataset.std(0).astype(theano.config.floatX))

        if self.Ym is None:
            Ym = Y_dataset.mean(0).astype(theano.config.floatX)
            self.Ym = theano.shared(Ym, name='%s>Ym' % (self.name))
            Ys = Y_dataset.std(0).astype(theano.config.floatX)
            self.Ys = theano.shared(Ys, name='%s>Ys' % (self.name))
        else:
            self.Ym.set_value(Y_dataset.mean(0).astype(theano.config.floatX))
            self.Ys.set_value(Y_dataset.std(0).astype(theano.config.floatX))

    def get_default_network_spec(self, batchsize=None, input_dims=None,
                                 output_dims=None,
                                 hidden_dims=[200, 200, 200, 200],
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
            output_dims = self.E
        if not isinstance(p, list):
            p = [p]*len(hidden_dims)
        if not isinstance(nonlinearities, list):
            nonlinearities = [nonlinearities]*len(hidden_dims)
        n_samples = self.dropout_samples.get_value()
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
                                      dropout_samples=n_samples)))
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
                                          dropout_samples=n_samples)))
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
        assert network_spec[0][0] is lasagne.layers.InputLayer or input_shape is not None
        if network_spec[0][0] is lasagne.layers.InputLayer:
            if input_shape is not None:
                # change the input shape
                network_spec[0][1]['shape'] = input_shape
            layer_class, layer_args = network_spec[0]
            print(layer_class.__name__, layer_args)
            network = layer_class(**layer_args)
            network_spec = network_spec[1:]
        else:
            network = lasagne.layers.InputLayer(input_shape, name=name+'_input')
        
        # create hidden layers
        for layer_class, layer_args in network_spec:
            layer_name = layer_args['name']
            
            if layer_name in params:
                layer_args.update(params[layer_name])
            print(layer_class.__name__, layer_args)
            network = layer_class(network, **layer_args)
            # change the periods in variable names
            for p in network.get_params():
                p.name = p.name.replace('.','>')

        # force rebuilding the prediction functions, as they will be out of date
        self.predict_fn = None
        self.predict_ic_fn = None

        return network
    
    def get_loss(self):
        ''' initializes the loss function for training '''
        # build the network
        if self.network is None:
            params = self.network_params if self.network_params is not None else {}
            self.network = self.build_network( self.network_spec, params=params, name=self.name)

        utils.print_with_stamp('Initialising loss function',self.name)

        # Input variables
        input_lengthscale = tt.scalar('%s>input_lengthscale'%(self.name))
        hidden_lengthscale = tt.scalar('%s>hidden_lengthscale'%(self.name))
        train_inputs = tt.matrix('%s>train_inputs'%(self.name))
        train_targets = tt.matrix('%s>train_targets'%(self.name))

        # standardize network inputs and outputs
        train_inputs_std = (train_inputs - self.Xm)/self.Xs
        train_targets_std = (train_targets - self.Ym)/self.Ys

        # evaluate nework output for batch
        if self.heteroscedastic:
            # this assumes that self.logsn is obtained from the network output
            train_predictions, logsn = lasagne.layers.get_output([self.network,self.logsn], train_inputs_std, deterministic=False)
        else:
            train_predictions = lasagne.layers.get_output(self.network, train_inputs_std, deterministic=False)
            logsn = self.logsn

        # scale logsn since output network output is standardized
        logsn_std = logsn - tt.log(self.Ys)

        # build the dropout loss function ( See Gal and Ghahramani 2015)
        delta_y = train_predictions-train_targets_std
        N, E = train_targets.shape
        N = N.astype(theano.config.floatX)
        E = E.astype(theano.config.floatX)

        # compute negative log likelihood
        nlml = tt.square(delta_y*tt.exp(-logsn_std)).sum(-1) + logsn_std.sum(-1) # note that if we have logsn_std be a 1xD vector, broadcasting rules apply
        loss = 0.5*nlml.mean()

        # compute regularization term
        loss += self.get_regularization_term(input_lengthscale, hidden_lengthscale)/N

        inputs = [train_inputs, train_targets, 
                  input_lengthscale, hidden_lengthscale]
        updates = {}

        # get trainable network parameters
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        # if we are learning the noise
        if self.learn_noise and not self.heteroscedastic:
            params.append(logsn)
        self.set_params(dict([(p.name, p) for p in params]))
        return loss, inputs, updates

    def get_regularization_term(self, input_lengthscale, hidden_lengthscale):
        reg = 0
        layers = lasagne.layers.get_all_layers(self.network)
        
        for i in range(1,len(layers)):
            reg_weight = 1/2.0
            # apply different regularization weigths to the input, and the hidden dimension
            ind = i if not (i >1 and isinstance(layers[i-1],lasagne.layers.DropoutLayer)) else i-1
            if hasattr(layers[ind],'input_layer') and isinstance(layers[ind].input_layer,lasagne.layers.InputLayer):
                reg_weight *= input_lengthscale**2
            else:
                reg_weight *= hidden_lengthscale**2

            # if this layer has a weight layer and the previous layer is a DropoutLayer
            if hasattr(layers[i],'W'):
                if i >1 and isinstance(layers[i-1],lasagne.layers.DropoutLayer):
                    p = layers[i-1].p
                    if p>0:
                        reg_weight *= (1-p)
                reg += reg_weight*lasagne.regularization.l2(layers[i].W)

            if hasattr(layers[i],'b'):
                reg += reg_weight*lasagne.regularization.l2(layers[i].b)

        return reg

    def get_updates(self,network=None):
        ''' returns a dictionary where the keys are lasagne layer instances and the values are their corresponding dropout masks'''
        if network is None:
            network = self.network
        mask_updates = []
        for l in lasagne.layers.get_all_layers(network):
            if isinstance(l,kusanagi.ghost.regression.layers.DropoutLayer) and l.mask_updates is not None:
                mask_updates.append((l.mask,l.mask_updates))

        return mask_updates

    def predict_symbolic(self,mx,Sx=None, deterministic=False, iid_per_eval=False, return_samples=False, with_measurement_noise=True):
        ''' returns symbolic expressions for the evaluations of this objects neural network. If Sx is specified, the
        output will correspond to the mean, covariance and input-output covariance of the network predictions'''
        if Sx is not None:
            # generate random samples from input (assuming gaussian distributed inputs)
            # standard uniform samples (one sample per network sample)
            z_std = self.m_rng.normal((self.dropout_samples,self.D))

            # scale and center particles
            Lx = tt.slinalg.cholesky(Sx)
            x = mx + z_std.dot(Lx.T)
        else:
            x = mx[None,:] if mx.ndim == 1 else mx

        if hasattr(self,'Xm') and self.Xm is not None:
            # standardize inputs
            x = (x - self.Xm)/self.Xs
        # unless we set the shared_axes parameter on the droput layers, the dropout masks should be different per sample
        y = lasagne.layers.get_output(self.network, x, deterministic=deterministic, fixed_dropout_masks=not iid_per_eval)

        if hasattr(self,'Ym') and self.Ym is not None:
            # scale and center outputs
            y = y*self.Ys + self.Ym
        y.name='%s>output_samples'%(self.name)
        if return_samples:
            # nothing else to do!
            return y
        
        n = tt.cast(y.shape[0], dtype=theano.config.floatX)
        # empirical mean
        M = y.mean(axis=0)
        # empirical covariance TODO emprical mean of logsn for heteroscedastic noise
        S = y.T.dot(y)/n - tt.outer(M,M)
        if with_measurement_noise:
            S += tt.diag(tt.exp(2*self.logsn))

        # empirical input output covariance
        if Sx is not None:
            C = x.T.dot(y)/n - tt.outer(mx,M)
        else:
            C = tt.zeros((self.D,self.E))
            
        return [M,S,C]

    def update(self, n_samples=None):
        ''' Updates the dropout masks'''
        if n_samples is not None:
            if isinstance(n_samples, tt.sharedvar.SharedVariable):
                self.dropout_samples = n_samples
                update_fn = None
            else:
                self.dropout_samples.set_value(n_samples)

            # increase the size of the masks
            updts = self.get_updates()
            for mask,updt in updts:
                mask_shape = mask.shape.eval()
                if mask_shape[0] != n_samples:
                    mask_shape[0] = n_samples
                    mask.set_value(np.zeros(mask_shape, dtype=mask.dtype))

        if not hasattr(self,'update_fn') or self.update_fn is None:
            # get prediction with non deterministic samples
            mx = tt.zeros((self.dropout_samples,self.D))
            output_vars = self.predict_symbolic(mx,iid_per_eval=False)

            # create a function to update the masks manually. Here the dropout masks should be shared variables
            updts = self.get_updates()
            self.update_fn = theano.function([],[], updates = updts, allow_input_downcast=True)
            
        # draw samples from the network
        self.update_fn()

    def train(self, batch_size=100,
              input_ls=None, hidden_ls=None, lr=1e-3,
              optimizer=None, callback=None):
        if optimizer is None:
            optimizer = self.optimizer

        if optimizer.loss_fn is None or self.should_recompile:
            loss, inps, updts = self.get_loss()
            # we pass the learning rate as an input, and as a parameter to the updates method
            learning_rate = theano.tensor.scalar('lr')
            inps.append(learning_rate)
            optimizer.set_objective(loss, self.get_params(symbolic=True),
                                    inps, updts, learning_rate=learning_rate)
        if input_ls is None:
            # set to some proportion of the standard deviation
            # (inputs are scaled and centered to N(0,1) )
            input_ls = 0.1

        if hidden_ls is None:
            hidden_ls = input_ls
        
        optimizer.minibatch_minimize(self.X.get_value(), self.Y.get_value(),
                                     input_ls, hidden_ls, lr,
                                     batch_size=batch_size)
        self.trained = True

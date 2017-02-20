#!/usr/bin/env python2
import theano
import theano.tensor as tt
import lasagne
import numpy as np
import time
import kusanagi
from collections import OrderedDict
from kusanagi import utils
from kusanagi.ghost.regression import BaseRegressor

class BNN(BaseRegressor):
    ''' Inefficient implementation of the dropout idea by Gal and Gharammani, with Gaussian distributed inputs'''
    def __init__(self,idims, odims,  dropout_samples=20, learn_noise=True,  heteroscedastic = False, name='BNN', profile=False, filename=None, **kwargs):
        self.D = idims
        self.E = odims
        self.name=name
        self.should_recompile = False
        
        self.logsn = theano.shared((np.ones((self.E,))*np.log(1e-3)).astype(theano.config.floatX), name='%s_logsn'%(self.name))
        self.lengthscale = 1e-2

        self.network = None
        self.network_spec = None
        self.network_params = None
        self.sample_network_fn = None
        self.loss_fn = None
        self.train_fn = None
        self.predict_fn = None
        self.prediction_updates = None
        self.dropout_samples = theano.shared(np.array(dropout_samples).astype('int32'), name="%s>dropout_samples"%(self.name) ) 
        self.m_rng = theano.sandbox.rng_mrg.MRG_RandomStreams(lasagne.random.get_rng().randint(1,2147462579))

        self.X = None; self.Y = None
        self.Xm = None; self.Xs = None
        self.Ym = None; self.Ys = None

        self.profile = profile
        self.compile_mode = theano.compile.get_default_mode()#.excluding('scanOp_pushout_seqs_ops')

        self.learn_noise = learn_noise
        self.heteroscedastic = heteroscedastic
        
        # filename for saving
        self.filename = '%s_%d_%d_%s_%s'%(self.name,self.D,self.E,theano.config.device,theano.config.floatX) if filename is None else filename
        BaseRegressor.__init__(self,name=name,filename=self.filename)
        if filename is not None:
            self.load()
        
        # register theanno functions and shared variables for saving
        #self.register_types([tt.sharedvar.SharedVariable, theano.compile.function_module.Function])
        self.register_types([tt.sharedvar.SharedVariable])
        self.register(['logsn','network_params','network_spec'])
    
    def load(self, output_folder=None,output_filename=None):
        dropout_samples = self.dropout_samples.get_value()
        super(BNN,self).load(output_folder,output_filename)
        self.dropout_samples.set_value(dropout_samples)
        if self.network_spec is not None:
            self.network = self.build_network( self.network_spec, params=self.network_params, name=self.name)
        
    def save(self, output_folder=None,output_filename=None):
        # store references to the network shared variables, so we can save and load them correctly
        self.network_params = []
        for layer in lasagne.layers.get_all_layers(self.network):
            layer_params = dict([(p.name.split('>')[-1],p) for p in layer.get_params()])
            self.network_params.append((layer.name,layer_params))
        self.network_params = dict(self.network_params)
        super(BNN,self).save(output_folder,output_filename)
    
    def get_all_shared_vars(self, as_dict=False):
        if as_dict:
            v = [(attr_name,self.__dict__[attr_name]) for attr_name in self.__dict__.keys() if isinstance(self.__dict__[attr_name],tt.sharedvar.SharedVariable)]
            v += [(p.name,p) for p in lasagne.layers.get_all_params(self.network, unwrap_shared=True) ]
            return dict(v)
        else:
            v = [attr for attr in self.__dict__.values() if isinstance(attr,tt.sharedvar.SharedVariable)]
            v += lasagne.layers.get_all_params(self.network, unwrap_shared=True)
            return v
    
    def set_dataset(self,X_dataset,Y_dataset):
        # set dataset
        super(BNN,self).set_dataset(X_dataset.astype(theano.config.floatX),Y_dataset.astype(theano.config.floatX))

        # extra operations when setting the dataset (specific to this class)
        self.update_dataset_statistics()
        
        if self.learn_noise:
            # default log of measurement noise variance is set to 1% of dataset variation
            self.logsn.set_value(np.log((0.05*Y_dataset.std(0)).astype(theano.config.floatX)))

    def apppend_dataset(self,X_dataset,Y_dataset):
        # set dataset
        super(BNN,self).append_dataset(X_dataset.astype(theano.config.floatX),Y_dataset.astype(theano.config.floatX))

        # extra operations when setting the dataset (specific to this class)
        self.update_dataset_statistics()

    def update_dataset_statistics(self):
        X_dataset,Y_dataset = self.X.get_value(), self.Y.get_value()
        if self.Xm is None:
            self.Xm = theano.shared(X_dataset.mean(0).astype(theano.config.floatX),name='%s>Xm'%(self.name),borrow=True)
            self.Xs = theano.shared(X_dataset.std(0).astype(theano.config.floatX),name='%s>Xs'%(self.name),borrow=True)
        else:
            self.Xm.set_value(X_dataset.mean(0).astype(theano.config.floatX),borrow=True)
            self.Xs.set_value(X_dataset.std(0).astype(theano.config.floatX),borrow=True)

        if self.Ym is None:
            self.Ym = theano.shared(Y_dataset.mean(0).astype(theano.config.floatX),name='%s>Ym'%(self.name),borrow=True)
            self.Ys = theano.shared(Y_dataset.std(0).astype(theano.config.floatX),name='%s>Ys'%(self.name),borrow=True)
        else:
            self.Ym.set_value(Y_dataset.mean(0).astype(theano.config.floatX),borrow=True)
            self.Ys.set_value(Y_dataset.std(0).astype(theano.config.floatX),borrow=True)

        print self.Xm.get_value()

    def get_default_network_spec(self,batchsize=None, input_dims=None, output_dims=None, hidden_dims=[200,200], p=0.05, name=None):
        from lasagne.layers import InputLayer, DenseLayer
        from kusanagi.ghost.regression.layers import DropoutLayer
        from lasagne.nonlinearities import rectify, sigmoid, tanh, elu, linear, ScaledTanh
        if name is None:
            name = self.name
        if input_dims is None:
            input_dims = self.D
        if output_dims is None:
            output_dims = self.E
        if type(p) is not list:
            p = [p]*len(hidden_dims)
        network_spec = []

        # input layer
        input_shape = [batchsize,input_dims]
        network_spec.append( (InputLayer, dict(shape=input_shape, name=name+'_input') ) )
        # hidden layers
        for i in range(len(hidden_dims)):
            network_spec.append( (DenseLayer, dict(num_units=hidden_dims[i], nonlinearity=sigmoid, name=name+'_fc%d'%(i)) ) )
            network_spec.append( (DropoutLayer, dict(p=p[i], rescale=False, name=name+'_drop%d'%(i), dropout_samples=self.dropout_samples.get_value()) ) )
        # output layer
        network_spec.append( (DenseLayer, dict(num_units=output_dims, nonlinearity=linear,name=name+'_output')) )

        return network_spec

    def build_network(self, network_spec=None, input_shape=None, params={}, name=None):
        ''' Builds a network according to the specification in the network_spec argument.
        network_spec should be a list containing tuples where the first element is a class in lasagne.layers
        and the second element is a dictionary with jeyword arguments for the class constructor;
        i.e. [(layer_class_1, constructor_argss_1), ... , (layer_class_N, constructor_argss_N),  ].
        Optionally, you can also pass a dictionary of parameters where the keys are 'layer_name' and the values 
        are dictionaries with the trainable parameters indexed by names (e.g. W or b). The trainable parameter
        values should be numpy arrays, theano shared variables, or theano expressions to set the trainable
        parameters of the lasagne layers. e.g params = {'layer_name_1': {'W': theano_shared_1, 'b': some_np_array}}, would
        set the weights of layer 1 to be the shared variable theano_shared_1 and the biases to be initialized with the values of 
        some_np_array.'''
        # set default values
        if name is None:
            name = self.name
        if network_spec is None:
            network_spec = self.get_default_network_spec()
        utils.print_with_stamp('Building network',self.name)
        self.network_spec = network_spec
        
        # create input layer 
        assert network_spec[0][0] is lasagne.layers.InputLayer or input_shape is not None
        if network_spec[0][0] is lasagne.layers.InputLayer:
            if input_shape is not None:
                # change the input shape
                network_spec[0][1]['shape'] = input_shape
            layer_class, layer_args = network_spec[0]
            print layer_class.__name__, layer_args
            network = layer_class(**layer_args)
            network_spec = network_spec[1:]
        else:
            network = lasagne.layers.InputLayer(input_shape, name=name+'_input')
        
        # create hidden layers
        for layer_class, layer_args in network_spec:
            layer_name = layer_args['name']
            
            if layer_name in params:
                layer_args.update(params[layer_name])

            print layer_class.__name__, layer_args
            network = layer_class(network, **layer_args)
            # change the periods in variable names
            for p in network.get_params():
                p.name = p.name.replace('.','>')
        return network
    
    def init_loss(self):
        ''' initializes the loss function for training '''
        # build the network
        if self.network is None:
            self.network = self.build_network()

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
            train_predictions, logsn = lasagne.layers.get_output([self.network,self.logsn], train_inputs_std, deterministic=False)#, batch_norm_update_averages=True, batch_norm_use_averages=True)
        else:
            train_predictions = lasagne.layers.get_output(self.network, train_inputs_std, deterministic=False)#, batch_norm_update_averages=True, batch_norm_use_averages=True)
            logsn = self.logsn

        # scale logsn since ouotput network output is standardized
        logsn_std = logsn - tt.log(self.Ys)

        # build the dropout loss function ( See Gal and Ghahramani 2015)
        delta_y = train_predictions-train_targets_std
        N,E = train_targets.shape
        N = N.astype(theano.config.floatX)
        E = E.astype(theano.config.floatX)
        # compute negative log likelihood
        error = tt.square(delta_y*tt.exp(-logsn_std)).sum(1)
        loss = 0.5*error.mean() + logsn_std.mean() 

        # compute regularization term
        reg = 0
        layers = lasagne.layers.get_all_layers(self.network)
        
        for i in range(len(layers)):
            reg_weight = 1/(2.0*N)
            # this assumes that the inputs at every input layer have the same prior lengthscale
            if hasattr(layers[i],'input_layer') and isinstance(layers[i].input_layer,lasagne.layers.InputLayer):
                reg_weight *= input_lengthscale**2
            else:
                reg_weight *= hidden_lengthscale**2

            # if this layer has a weight layer and the next layer is a dropout layer
            if hasattr(layers[i],'W'):
                if i < len(layers)-1 and isinstance(layers[i+1],lasagne.layers.DropoutLayer):
                    p = layers[i+1].p
                    if p>0:
                        reg_weight *= (1-p)
                reg += reg_weight*lasagne.regularization.l2(layers[i].W)

            if hasattr(layers[i],'b'):
                reg += reg_weight*lasagne.regularization.l2(layers[i].b)

        loss += reg

        # build the updates dictionary ( sets the optimization algorithm for the network parameters)
        params = lasagne.layers.get_all_params(self.network, trainable=True)

        # SGD trainer
        updates = lasagne.updates.adam(loss,params,learning_rate=1e-3)
        # if we are learning the noise
        if self.learn_noise and not self.heteroscedastic:
            updates.update(lasagne.updates.adam(loss,[logsn],learning_rate=1e-4))
        
        # compile the training function
        l2_error = tt.square(delta_y).sum(1).mean()
        utils.print_with_stamp('Compiling training and  loss functions',self.name)
        self.train_fn = theano.function([train_inputs,train_targets,input_lengthscale,hidden_lengthscale],[loss,l2_error],updates=updates,allow_input_downcast=True)
        self.loss_fn = theano.function([train_inputs,train_targets,input_lengthscale,hidden_lengthscale],[loss,l2_error],allow_input_downcast=True)
        utils.print_with_stamp('Done compiling',self.name)

    def get_dropout_masks(self,network=None):
        ''' returns a dictionary where the keys are lasagne layer instances and the values are their corresponding dropout masks'''
        if network is None:
            network = self.network
        mask_updates = []
        for l in lasagne.layers.get_all_layers(network):
            if isinstance(l,kusanagi.ghost.regression.layers.DropoutLayer):
                mask_updates.append((l.mask,l.mask_updates))

        return mask_updates


    def predict_symbolic(self,mx,Sx=None, reinit_network=False, iid_per_eval=False, return_samples=False, with_measurement_noise=True):
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

        # standardize inputs
        x = (x - self.Xm)/self.Xs

        # unless we set the shared_axes parameter on the droput layers, the dropout masks should be different per sample
        y = lasagne.layers.get_output(self.network, x, deterministic=False, fixed_dropout_masks=not iid_per_eval)

        # scale and center outputs
        y = y*self.Ys + self.Ym

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

    def draw_network_samples(self, n_samples=10):
        ''' Draws realizations of the neural network dropout masks.'''
        if dropout_samples is not None:
            if isinstance(dropout_samples, tt.sharedvar.SharedVariable):
                self.dropout_samples = n_samples
            else:
                self.dropout_samples.set_value(n_samples)

        if self.sample_network_fn is None:
            # get prediction with non deterministic samples
            mx = tt.zeros((self.dropout_samples,self.D))
            output_vars = self.predict_symbolic(mx,iid_per_eval=False)

            # create a function to update the masks manually. Here the dropout masks should be shared variables
            dropout_mask_updts = self.get_dropout_masks()
            self.sample_network_fn = theano.function([],[], updates = dropout_mask_updts, allow_input_downcast=True)

            # call it once to initialize the masks
            self.draw_network_samples()
            
        # draw samples from the network
        self.sample_network_fn()

    def train(self, batchsize=100, maxiters=5000, input_ls=None, hidden_ls=None):
        if input_ls is None:
            # set to some proportion of the standard deviation (inputs are scaled and centered to N(0,1) )
            input_ls = 0.1
        if hidden_ls is None:
            hidden_ls = input_ls

        if self.train_fn is None:
            self.init_loss()

        # go through the dataset
        batch_size = min(batchsize,self.N)
        iters = 1
        while True:
            start_time = time.time()
            should_exit=False
            for x,y in utils.iterate_minibatches(self.X.get_value(borrow=True), self.Y.get_value(borrow=True), batch_size, shuffle=True):
                ret = self.train_fn(x,y,input_ls,hidden_ls)
                iters+=1
                if iters > maxiters:
                    should_exit=True
                    break
            elapsed_time = time.time() - start_time
            utils.print_with_stamp('iter: %d, loss: %E, error: %E, elapsed: %E, sn2: %s'%(iters,ret[0],ret[1],elapsed_time, np.exp(2*self.logsn.get_value())),self.name,True)
            if should_exit:
                break
        print ''

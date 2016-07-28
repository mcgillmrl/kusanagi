#!/usr/bin/env python2
import theano
import lasagne
import numpy as np
import time
import utils

class NN(object):
    def __init__(self,idims, hidden_dims, odims, uncertain_inputs=True, name='NN', profile=False):
        ''' Constructs a Bayessian Neural Network regressor.
        '''
        self.D = idims
        self.hidden_dims = hidden_dims
        self.E = odims
        self.name=name
        self.uncertain_inputs = uncertain_inputs
        self.should_recompile = False
        
        ls2 = np.ones((self.E,))*np.log(1e-3)
        self.logsn2 = theano.shared(np.array(ls2,dtype=theano.config.floatX))
        self.lscale2 = 10

        self.learning_params = {'iters': 10000, 'batch_size': 200}
        self.network = None
        self.loss_fn = None
        self.train_fn = None
        self.predict_fn = None
        self.predict_updts = None
        self.drop_input = None
        self.drop_hidden = 0.1
        self.drop_output = 0.1
        self.dropout_samples = 50 
        self.n_particles = 10
        self.m_rng = theano.sandbox.rng_mrg.MRG_RandomStreams(lasagne.random.get_rng().randint(1,2147462579))

        self.X = None; self.Y = None

        self.profile = profile
        self.compile_mode = theano.compile.get_default_mode()#.excluding('scanOp_pushout_seqs_ops')
    
    def get_all_shared_vars(self, as_dict=False):
        if as_dict:
            # TODO get all the params from the lasagne neural net model
            return [(attr_name,self.__dict__[attr_name]) for attr_name in self.__dict__.keys() if isinstance(self.__dict__[attr_name],theano.tensor.sharedvar.SharedVariable)]
        else:
            v= [attr for attr in self.__dict__.values() if isinstance(attr,theano.tensor.sharedvar.SharedVariable)]
            v.extend(lasagne.layers.get_all_params(self.network))
            return v
    
    def set_dataset(self,X_dataset,Y_dataset):
        # ensure we don't change the number of input and output dimensions ( the number of samples can change)
        assert X_dataset.shape[0] == Y_dataset.shape[0], "X_dataset and Y_dataset must have the same number of rows"
        if self.X is not None:
            assert self.X.get_value(borrow=True).shape[1] == X_dataset.shape[1]
        if self.Y is not None:
            assert self.Y.get_value(borrow=True).shape[1] == Y_dataset.shape[1]
        
        # first, convert numpy arrays to appropriate type
        X_dataset = X_dataset.astype( theano.config.floatX )
        Y_dataset = Y_dataset.astype( theano.config.floatX )
        
        # dims = non_angle_dims + 2*angle_dims
        self.N = X_dataset.shape[0]
        self.D = X_dataset.shape[1]
        self.E = Y_dataset.shape[1]

        # now we create symbolic shared variables
        if self.X is None:
            self.X = theano.shared(X_dataset,name='%s>X'%(self.name),borrow=True)
        else:
            self.X.set_value(X_dataset,borrow=True)
        if self.Y is None:
            self.Y = theano.shared(Y_dataset,name='%s>Y'%(self.name),borrow=True)
        else:
            self.Y.set_value(Y_dataset,borrow=True)

        self.lscale2 = 0.001*self.X.get_value(borrow=True).var(0).sum()
        self.logsn2.set_value(np.log(0.001*self.Y.get_value(borrow=True).var(0)).astype(theano.config.floatX))

    def append_dataset(self,X_dataset,Y_dataset):
        if self.X is None:
            self.set_dataset(X_dataset,Y_dataset)
        else:
            X_ = np.vstack((self.X.get_value(), X_dataset.astype(theano.config.floatX)))
            Y_ = np.vstack((self.Y.get_value(), Y_dataset.astype(theano.config.floatX)))
            self.set_dataset(X_,Y_)

    def init_params(self,reinit=False):
        pass

    def set_params(self, params):
        lasagne.layers.set_all_param_values(network, params)

    def get_params(self, symbolic= True):
        if symbolic:
            return lasagne.layers.get_all_params(self.network, trainable=True)
        return lasagne.layers.get_all_param_values(self.network, trainable=True)
    
    def build_network(self):
        utils.print_with_stamp('Building network: Inputs  %s , Hidden  %s , Outputs  %s '%(str(self.D),str(self.hidden_dims),str(self.E)),self.name)
        # create input layer (first dimension is batch size N, second dimension is input data dimension D)
        input_layer_dims = [None]
        input_layer_dims.append(self.D)
        network = lasagne.layers.InputLayer(input_layer_dims, name=self.name+'_input')

        # add dropout to the input layer
        if self.drop_input:
            network = lasagne.layers.DropoutLayer(network, p=self.drop_input)
        
        # create hidden layers
        for i in xrange(len(self.hidden_dims)):
            network = lasagne.layers.DenseLayer(network, num_units=self.hidden_dims[i], nonlinearity=lasagne.nonlinearities.rectify, name=self.name+'_h%d'%(i))
            # add dropout to the hidden layer
            if self.drop_hidden:
                network = lasagne.layers.DropoutLayer(network, p=self.drop_hidden)

        self.network = lasagne.layers.DenseLayer(network, num_units=self.E, nonlinearity=lasagne.nonlinearities.elu, name=self.name+'_output')
        # add dropout to the output layer
        if self.drop_output:
            self.network = lasagne.layers.DropoutLayer(self.network, p=self.drop_output)

    def init_loss(self):
        ''' initializes the loss function for training '''
        # build the network
        if self.network is None or reinit_network:
            self.build_network()

        utils.print_with_stamp('Initialising loss function',self.name)
        # evaluate the output in mini_batches
        train_inputs = theano.tensor.matrix('%s>train_inputs'%(self.name))
        train_targets = theano.tensor.matrix('%s>train_targets'%(self.name))
        train_predictions = lasagne.layers.get_output(self.network, train_inputs, deterministic=False)

        # build the dropout loss function ( See Gal and Gharamani 2015)
        #loss = theano.tensor.mean(lasagne.objectives.squared_error(train_predictions,train_targets))
        delta_y = train_predictions-train_targets
        N,E = train_targets.shape
        error = ((delta_y*theano.tensor.exp(-0.5*self.logsn2))**2).sum(1)
        l2_penalty = lasagne.regularization.regularize_network_params(self.network, lasagne.regularization.l2)
        l2_weight = self.lscale2*(1-self.drop_hidden)/N
        loss = 0.5*error.mean() + l2_weight*l2_penalty + 0.5*self.logsn2.sum()

        # build the updates dictionary ( sets the optimization algorithm for the network parameters)
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        params.append(self.logsn2)
        #updates = lasagne.updates.nesterov_momentum(loss,params,learning_rate=self.learning_params['rate'],momentum=self.learning_params['momentum'])
        updates = lasagne.updates.adadelta(loss,params)
        
        # compile the training function
        utils.print_with_stamp('Compiling training and  loss functions',self.name)
        self.train_fn = theano.function([train_inputs,train_targets],loss,updates=updates,allow_input_downcast=True)
        self.loss_fn = theano.function([train_inputs,train_targets],loss,allow_input_downcast=True)

    def init_predict(self):
        mx = theano.tensor.vector('mx')
        Sx = theano.tensor.matrix('Sx') if self.uncertain_inputs else None
        dropout_samples = theano.tensor.iscalar('dropout_samples')

        # initialize variable for input covariance 
        input_vars = [mx] if not self.uncertain_inputs else [mx,Sx]
        input_vars.append(dropout_samples)
        
        # get prediction
        output_vars = self.predict_symbolic(mx,Sx,dropout_samples=dropout_samples)

        prediction = []
        for o in output_vars:
            if o is not None:
                prediction.append(o)

        # compile prediction
        utils.print_with_stamp('Compiling mean and variance of prediction',self.name)
        self.predict_fn = theano.function(input_vars,prediction, 
                                            updates=self.predict_updts,
                                            name='%s>predict_'%(self.name),
                                            profile=self.profile,
                                            mode=self.compile_mode,
                                            allow_input_downcast=True)
        self.state_changed = True # for saving

    def predict_symbolic(self,mx,Sx=None, reinit_network=False, deterministic=False, rescale=True, dropout_samples=None):
        if self.network is None or reinit_network:
            self.build_network()
        if dropout_samples is None:
            dropout_samples = self.dropout_samples

        if Sx is not None:
            # generate random samples from input (assuming gaussian distributed inputs)
            # standard uniform samples
            self.z_std = self.m_rng.normal((self.n_particles,self.D))

            # transform to multivariate normal
            Lx = theano.tensor.slinalg.cholesky(Sx + 1e-9*theano.tensor.eye(Sx.shape[0]))
            x = mx + self.z_std.dot(Lx.T)
        else:
            x = mx[None,:]

        if dropout_samples:
            def sample_network(x_):
                # sample from gaussian
                y_ = lasagne.layers.get_output(self.network, x_, deterministic=False)
                return x_,y_

            (x,y), self.predict_updts = theano.scan(fn=sample_network,
                                                non_sequences=[x],
                                                n_steps=dropout_samples, 
                                                allow_gc=False)

            x = x.transpose(2,0,1).flatten(2).T
            y = y.transpose(2,0,1).flatten(2).T
        else:
            y = lasagne.layers.get_output(self.network, x, deterministic=deterministic)

        # empirical mean
        M = y.mean(axis=0)
        # empirical covariance
        S = theano.tensor.diag(theano.tensor.exp(self.logsn2)*theano.tensor.ones((self.E,))) + y.T.dot(y)/y.shape[0] - theano.tensor.outer(M,M)
        # Sx^-1 times empirical input output covariance
        if Sx is not None:
            C = x.T.dot(y)/y.shape[0] - theano.tensor.outer(mx,M)
            C = theano.tensor.slinalg.solve_lower_triangular(Lx,C)
            C = theano.tensor.slinalg.solve_upper_triangular(Lx.T,C)
        else:
            C = theano.tensor.zeros((self.D,self.E))

        return [M,S,C]

    def predict(self,mx,Sx = None, dropout_samples=None):
        predict = None
        if dropout_samples is None:
            dropout_samples = self.dropout_samples

        if self.predict_fn is None:
            self.init_predict()
        predict = self.predict_fn

        odims = self.E
        idims = self.D
        res = None
        if self.uncertain_inputs:
            if Sx is None:
                Sx = np.zeros((idims,idims))
            res = predict(mx, Sx, dropout_samples)
        else:
            res = predict(mx,dropout_samples)
        return res

    def train(self):
        if self.train_fn is None:
            self.init_loss()

        # go through the dataset
        batch_size = min(self.learning_params['batch_size'],self.N)
        for i in xrange(self.learning_params['iters']):
            start_time = time.time()
            for x,y in utils.iterate_minibatches(self.X.get_value(borrow=True), self.Y.get_value(borrow=True), batch_size, shuffle=True):
                ret = self.train_fn(x,y)
            elapsed_time = time.time() - start_time
            utils.print_with_stamp('iter: %d, loss: %E, elapsed: %E, sigma2: %s    '%(i,ret,elapsed_time, np.exp(self.logsn2.get_value())),self.name,True)
        print ''

    def load(self):
        pass

    def save(self):
        pass

    def set_state(self,state):
        pass

    def get_state(self):
        pass

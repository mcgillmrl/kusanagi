#!/usr/bin/env python2
import theano
import lasagne
import numpy as np
import time
import utils

class NN(object):
    def __init__(self,idims, hidden_dims, odims, sat_func=None, name='NN', profile=False):
        ''' Constructs a Bayessian Neural Network regressor.
        '''
        self.D = idims
        self.hidden_dims = hidden_dims
        self.E = odims
        self.name=name
        self.uncertain_inputs = True
        self.should_recompile = False

        self.sigma_n = 1e-3
        self.loghyp = np.array([[self.sigma_n]],dtype=theano.config.floatX)
        self.lscale2 = 10

        self.learning_params = {'rate': 1e-3, 'momentum': 0.9, 'iters': 10000, 'batch_size': 100}
        self.network = None
        self.loss_fn = None
        self.train_fn = None
        self.predict_fn = None
        self.predict_updts = None
        self.predict_d_fn = None
        self.drop_input = None
        self.drop_hidden = 0.05
        self.drop_output = 0.05
        self.dropout_samples = 15
        self.m_rng = theano.sandbox.rng_mrg.MRG_RandomStreams(lasagne.random.get_rng().randint(1,2147462579))

        self.X_ = None
        self.Y_ = None
        self.sat_func = sat_func

        self.profile = profile
        self.compile_mode = theano.compile.get_default_mode()#.excluding('scanOp_pushout_seqs_ops')
    
    def set_dataset(self,X_dataset,Y_dataset):
        #utils.print_with_stamp('Updating GP dataset',self.name)
        # ensure we don't change the number of input and output dimensions ( the number of samples can change)
        assert X_dataset.shape[0] == Y_dataset.shape[0], "X_dataset and Y_dataset must have the same number of rows"
        if self.X_ is not None:
            assert self.X_.shape[1] == X_dataset.shape[1]
        if self.Y_ is not None:
            assert self.Y_.shape[1] == Y_dataset.shape[1]
        
        # first, assign the numpy arrays to class members
        self.X_ = X_dataset.astype( theano.config.floatX )
        self.Y_ = Y_dataset.astype( theano.config.floatX )
        # dims = non_angle_dims + 2*angle_dims
        self.N = self.X_.shape[0]
        self.D = X_dataset.shape[1]
        self.E = Y_dataset.shape[1]

        self.lscale2 = self.X_.var(0).sum()
    
    def append_dataset(self,X_dataset,Y_dataset):
        if self.X_ is None:
            self.set_dataset(X_dataset,Y_dataset)
        else:
            self.X_ = np.vstack((self.X_, X_dataset.astype(theano.config.floatX)))
            self.Y_ = np.vstack((self.Y_, Y_dataset.astype(theano.config.floatX)))
            self.N = self.X_.shape[0]
            self.D = X_dataset.shape[1]
            self.E = Y_dataset.shape[1]
            self.lscale2 = self.X_.var(0).sum()

    def init_params(self,reinit=False):
        pass

    def set_params(self, params):
        lasagne.layers.set_all_param_values(network, params)

    def get_params(self, symbolic= True):
        if symbolic:
            return lasagne.layers.get_all_params(self.network, trainable=True)
        return lasagne.layers.get_all_param_values(self.network, trainable=True)

    def init_loss(self):
        ''' initializes the loss function for training '''
        utils.print_with_stamp('Initialising loss function',self.name)
        # evaluate the output in mini_batches
        train_inputs = theano.tensor.matrix('%s>X'%(self.name))
        train_targets = theano.tensor.matrix('%s>Y'%(self.name))
        l2_weight = theano.tensor.scalar('%s>l2_weight'%(self.name))
        train_predictions = self.predict_symbolic(train_inputs, deterministic=False, dropout_samples=1)
        
        # build the dropout loss function ( See Gal and Gharamani 2015)
        loss = theano.tensor.mean(lasagne.objectives.squared_error(train_predictions,train_targets))
        l2_penalty = lasagne.regularization.regularize_network_params(self.network, lasagne.regularization.l2)
        loss = loss + l2_weight*l2_penalty

        # build the updates dictionary ( sets the optimization algorithm for the network parameters)
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        updates = lasagne.updates.nesterov_momentum(loss,params,learning_rate=self.learning_params['rate'],momentum=self.learning_params['momentum'])
        
        # compile the training function
        utils.print_with_stamp('Compiling training and  loss functions',self.name)
        self.train_fn = theano.function([train_inputs,train_targets,l2_weight],loss,updates=updates,allow_input_downcast=True)
        self.loss_fn = theano.function([train_inputs,train_targets,l2_weight],loss,allow_input_downcast=True)

    def init_predict(self, derivs=False):
        mx = theano.tensor.vector('mx')
        Sx = theano.tensor.matrix('Sx') if self.uncertain_inputs else None
        # initialize variable for input covariance 
        input_vars = [mx] if not self.uncertain_inputs else [mx,Sx]
        
        # get prediction
        output_vars = self.predict_symbolic(mx,Sx)

        prediction = []
        for o in output_vars:
            if o is not None:
                prediction.append(o)

        if not derivs:
            # compile prediction
            utils.print_with_stamp('Compiling mean and variance of prediction',self.name)
            self.predict_fn = theano.function(input_vars,prediction, 
                                              updates=self.predict_updts,
                                              name='%s>predict_'%(self.name),
                                              profile=self.profile,
                                              mode=self.compile_mode,
                                              allow_input_downcast=True)
            self.state_changed = True # for saving
        else:
            # compute the derivatives wrt the input vector ( or input mean in the case of uncertain inputs)
            prediction_derivatives = list(prediction)
            for p in prediction:
                prediction_derivatives.append( theano.tensor.jacobian( p.flatten(), mx ).flatten(2) )
            if self.uncertain_inputs:
                # compute the derivatives wrt the input covariance
                for p in prediction:
                    prediction_derivatives.append( theano.tensor.jacobian( p.flatten(), Sx ).flatten(2) )
            if self.hyperparameter_gradients:
                # compute the derivatives wrt the hyperparameters
                for p in prediction:
                    prediction_derivatives.append( theano.tensor.jacobian( p.flatten(), self.loghyp ) )
                    prediction_derivatives.append( theano.tensor.jacobian( p.flatten(), self.X ) )
                    prediction_derivatives.append( theano.tensor.jacobian( p.flatten(), self.Y ) )
            #prediction_derivatives contains  [p1, p2, ..., pn, dp1/dmx, dp2/dmx, ..., dpn/dmx, dp1/dSx, dp2/dSx, ..., dpn/dSx, dp1/dloghyp, dp2/dloghyp, ..., dpn/loghyp]
            utils.print_with_stamp('Compiling mean and variance of prediction with jacobians',self.name)
            self.predict_d_fn = theano.function(input_vars,
                                                prediction_derivatives,
                                                updates=self.predict_updts,
                                                name='%s>predict_d_'%(self.name),
                                                profile=self.profile,
                                                mode=self.compile_mode,
                                                allow_input_downcast=True)
            self.state_changed = True # for saving

    def predict_symbolic(self,mx,Sx=None, reinit_network=False, deterministic=False, rescale=True, dropout_samples=None):
        if not dropout_samples:
            dropout_samples = self.dropout_samples

        if self.network is None or reinit_network:
            utils.print_with_stamp('Building network: Inputs  %s , Hidden  %s , Outputs  %s '%(str(self.D),str(self.hidden_dims),str(self.E)),self.name)
            # create input layer (first dimension is batch size N, second dimension is input data dimension D)
            input_layer_dims = [None]
            input_layer_dims.append(self.D)
            network = lasagne.layers.InputLayer(input_layer_dims, name=self.name+'_input')

            # add dropout to the input layer
            if self.drop_input:
                network = lasagne.layers.DropoutLayer(network, p=self.drop_input, rescale=rescale)
            
            # create hidden layers
            for i in xrange(len(self.hidden_dims)):
                network = lasagne.layers.DenseLayer(network, num_units=self.hidden_dims[i], nonlinearity=lasagne.nonlinearities.elu, name=self.name+'_h%d'%(i))
                # add dropout to the hidden layer
                if self.drop_hidden:
                    network = lasagne.layers.DropoutLayer(network, p=self.drop_hidden, rescale=rescale)

            self.network = lasagne.layers.DenseLayer(network, num_units=self.E, nonlinearity=lasagne.nonlinearities.elu, name=self.name+'_output')
            # add dropout to the output layer
            if self.drop_output:
                self.network = lasagne.layers.DropoutLayer(self.network, p=self.drop_output, rescale=rescale)
        
        if dropout_samples > 1:
            if Sx is not None:
                if Sx is not None:
                    Wx,Vx = theano.tensor.nlinalg.eigh(Sx) 
                    Lx = Vx.dot(theano.tensor.sqrt(Wx))
                else:
                    Lx = None

                def sample_network(mx,Lx):
                    # sample from gaussian
                    x = mx + Lx.dot(self.m_rng.normal((self.D,)).flatten())
                    y = lasagne.layers.get_output(self.network, x, deterministic=False)

                    return x,y
                
                (x,y), self.predict_updts = theano.scan(fn=sample_network,
                                                        non_sequences=[mx,Lx],
                                                        n_steps=dropout_samples, 
                                                        allow_gc=False)
                new_xshape = [dropout_samples,self.D]
                x = x.reshape(new_xshape)
            else:
                y, self.predict_updts = theano.scan(fn=lambda x: lasagne.layers.get_output(self.network, x, deterministic=False),
                                                                 non_sequences=[mx],
                                                                 n_steps=dropout_samples, 
                                                                 allow_gc=False)
            new_yshape = [dropout_samples,self.E]
            y = y.reshape(new_yshape)
            
            a = theano.tensor.cast(1.0/dropout_samples,theano.config.floatX)
            # emprical mean
            M = y.sum(axis=0)*a
            # empirical covariance
            S = theano.tensor.diag(self.sigma_n*theano.tensor.ones((self.E,))) + y.T.dot(y)*a - theano.tensor.outer(M,M)
            # Sx^-1 times empirical input output covariance
            if Sx is not None:
                C = x.T.dot(y)*a - theano.tensor.outer(mx,M)
            else:
                C = theano.tensor.zeros((self.D,self.E))

            # apply saturating function to the output if available
            if self.sat_func is not None:
                # compute the joint input output covariance
                M,S,U = self.sat_func(M,S)
                C = C.dot(U)

            return [M,S,C]
        else:
            # apply saturating function to the output if available
            if self.sat_func is not None:
                return self.sat_func(lasagne.layers.get_output(self.network, mx, deterministic=deterministic))
            else:
                return lasagne.layers.get_output(self.network, mx, deterministic=deterministic)

    def predict(self,mx,Sx = None, derivs=False):
        predict = None
        if not derivs:
            if self.predict_fn is None:
                self.init_predict(derivs=derivs)
            predict = self.predict_fn
        else:
            if self.predict_d_fn is None:
                self.init_predict(derivs=derivs)
            predict = self.predict_d_fn

        odims = self.E
        idims = self.D
        res = None
        if self.uncertain_inputs:
            if Sx is None:
                Sx = np.zeros((idims,idims))
            res = predict(mx, Sx)
        else:
            res = predict(mx)
        return res

    def train(self):
        if self.train_fn is None:
            self.init_loss()

        # go through the dataset
        batch_size = min(self.learning_params['batch_size'],self.N)
        self.regularization_weight = self.lscale2*(1-self.drop_hidden)*self.sigma_n/(2*batch_size)
        utils.print_with_stamp('L2 regularization weight: %s'%(str(self.regularization_weight)),self.name)
        for i in xrange(self.learning_params['iters']):
            start_time = time.time()
            for x,y in utils.iterate_minibatches(self.X_, self.Y_, batch_size, shuffle=True):
                ret = self.train_fn(x,y,self.regularization_weight)
            elapsed_time = time.time() - start_time
            utils.print_with_stamp('iter: %d, loss: %f, elapsed: %f'%(i,ret,elapsed_time),self.name,True)
        print ''

    def load(self):
        pass

    def save(self):
        pass

    def set_state(self,state):
        pass

    def get_state(self):
        pass

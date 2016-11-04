#!/usr/bin/env python2
import theano
import lasagne
import numpy as np
import time
from kusanagi import utils
from kusanagi.base.Loadable import Loadable

class NN(Loadable):
    ''' Inefficient implementation of the dropout idea by Gal and Gharammani, with Gaussian distributed inputs'''
    def __init__(self,idims, odims, hidden_dims=[128,128], dropout_samples=250, uncertain_inputs=True, name='NN', profile=False):
        self.D = idims
        self.hidden_dims = hidden_dims
        self.E = odims
        self.name=name
        self.uncertain_inputs = uncertain_inputs
        self.should_recompile = False
        
        ls2 = np.ones((self.E,))*np.log(1e-3)
        self.logsn = theano.shared(np.array(ls2,dtype=theano.config.floatX))
        self.lscale2 = 10

        self.learning_params = {'iters': 25000, 'batch_size': 500}
        self.network = None
        self.loss_fn = None
        self.train_fn = None
        self.predict_fn = None
        self.prediction_updates = None
        self.drop_hidden = 0.01
        self.dropout_samples = theano.shared(dropout_samples, name="%s>dropout_samples"%(self.name)) 
        self.m_rng = theano.sandbox.rng_mrg.MRG_RandomStreams(lasagne.random.get_rng().randint(1,2147462579))

        self.X = None; self.Y = None
        self.Ym = None; self.Ys = None

        self.profile = profile
        self.compile_mode = theano.compile.get_default_mode()#.excluding('scanOp_pushout_seqs_ops')

        self.learn_noise = True
        
        # filename for saving
        self.filename = '%s_%d_%d_%s_%s'%(self.name,self.D,self.E,theano.config.device,theano.config.floatX)
        Loadable.__init__(self,name=name,filename=self.filename)
        # register theanno functions and shared variables for saving
        self.register_types([theano.tensor.sharedvar.SharedVariable, theano.compile.function_module.Function])
    
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
            self.Ym = theano.shared(Y_dataset.mean(0),name='%s>Ym'%(self.name),borrow=True)
            self.Ys = theano.shared(3*Y_dataset.std(0),name='%s>Ys'%(self.name),borrow=True)
        else:
            self.Y.set_value(Y_dataset,borrow=True)
            self.Ym.set_value(Y_dataset.mean(0),borrow=True)
            self.Ys.set_value(3*Y_dataset.std(0),borrow=True)

        self.lscale2 = 1e-7*self.X.get_value(borrow=True).var(0).sum()
        self.logsn.set_value(np.log(1e-2*self.Y.get_value(borrow=True).std(0)).astype(theano.config.floatX))

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

        # create hidden layers
        for i in xrange(len(self.hidden_dims)):
            network = lasagne.layers.DenseLayer(network, num_units=self.hidden_dims[i], nonlinearity=lasagne.nonlinearities.rectify, name=self.name+'_h%d'%(i))
            # add batch normalization
            #network = lasagne.layers.batch_norm(network)
            # add dropout to the hidden layer
            if self.drop_hidden and self.drop_hidden>0:
                network = lasagne.layers.DropoutLayer(network, p=self.drop_hidden)

        self.network = lasagne.layers.DenseLayer(network, num_units=self.E, nonlinearity=lasagne.nonlinearities.linear, name=self.name+'_output')

    def init_loss(self):
        ''' initializes the loss function for training '''
        # build the network
        if self.network is None or reinit_network:
            self.build_network()

        utils.print_with_stamp('Initialising loss function',self.name)
        # evaluate the output in mini_batches
        train_inputs = theano.tensor.fmatrix('%s>train_inputs'%(self.name))
        train_targets = theano.tensor.fmatrix('%s>train_targets'%(self.name))
        train_predictions = lasagne.layers.get_output(self.network, train_inputs, deterministic=False)#, batch_norm_update_averages=True, batch_norm_use_averages=True)
        train_predictions = train_predictions*self.Ys + self.Ym

        # build the dropout loss function ( See Gal and Gharamani 2015)
        #loss = theano.tensor.mean(lasagne.objectives.squared_error(train_predictions,train_targets))
        delta_y = train_predictions-train_targets
        N,E = train_targets.shape
        l2_penalty = lasagne.regularization.regularize_network_params(self.network, lasagne.regularization.l2)
        l2_weight = self.lscale2*(1.0-self.drop_hidden)/(2.0*N)
        if self.learn_noise:
            error = ((delta_y*theano.tensor.exp(-self.logsn))**2).sum(1)
            loss = 0.5*error.mean() + l2_weight*l2_penalty + self.logsn.sum()
        else:
            error = (delta_y**2).sum(1)
            loss = error.mean() + l2_weight*l2_penalty 


        # build the updates dictionary ( sets the optimization algorithm for the network parameters)
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        if self.learn_noise:
            params.append(self.logsn)
        #updates = lasagne.updates.nesterov_momentum(loss,params,learning_rate=self.learning_params['rate'],momentum=self.learning_params['momentum'])
        #updates = lasagne.updates.adadelta(loss,params,learning_rate=1e-1)
        updates = lasagne.updates.adam(loss,params,learning_rate=1e-3)
        
        # compile the training function
        utils.print_with_stamp('Compiling training and  loss functions',self.name)
        self.train_fn = theano.function([train_inputs,train_targets],loss,updates=updates,allow_input_downcast=True)
        self.loss_fn = theano.function([train_inputs,train_targets],loss,allow_input_downcast=True)
        utils.print_with_stamp('Done compiling',self.name)
        #theano.printing.pydotprint(self.train_fn,outfile='train_fn_%s.png'%(theano.config.device))
        #theano.printing.pydotprint(self.loss_fn,outfile='loss_fn_%s.png'%(theano.config.device))
        #with open('loss_fn_%s.txt'%(theano.config.device),'w') as f:
        #    theano.printing.debugprint(self.loss_fn,file=f,print_type=True)
        #with open('train_fn_%s.txt'%(theano.config.device),'w') as f:
        #    theano.printing.debugprint(self.train_fn,file=f,print_type=True)


    def init_predict(self):
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

        # compile prediction
        utils.print_with_stamp('Compiling mean and variance of prediction',self.name)
        self.predict_fn = theano.function(input_vars,prediction, 
                                            updates=self.prediction_updates,
                                            name='%s>predict_'%(self.name),
                                            profile=self.profile,
                                            mode=self.compile_mode,
                                            allow_input_downcast=True)
        utils.print_with_stamp('Done compiling',self.name)
        #theano.printing.pydotprint(self.predict_fn,outfile='predict_fn_%s.pdf'%(theano.config.device))
        #with open('predict_fn_%s.png'%(theano.config.device),'w') as f:
        #    theano.printing.debugprint(self.predict_fn,file=f,print_type=True)
        self.state_changed = True # for saving

    def predict_symbolic(self,mx,Sx=None, reinit_network=False):
        if self.network is None or reinit_network:
            self.build_network()
        
        mx = mx.astype('float64')
        if Sx is not None:
            Sx = Sx.astype('float64')
            # generate random samples from input (assuming gaussian distributed inputs)
            # standard uniform samples (one sample per network sample)
            z_std = self.m_rng.normal((self.dropout_samples,self.D))

            # transform to multivariate normal
            Lx = theano.tensor.slinalg.cholesky(Sx)
            x = mx + z_std.dot(Lx.T)
        else:
            x = mx[None,:]
        
        # force the input data to be represented with single precision floats
        x = x.astype('float32')
        def sample_network(x_):
            # sample from gaussian
            y_ = lasagne.layers.get_output(self.network, x_, deterministic=False)#, batch_norm_update_averages=False, batch_norm_use_averages=True)
            return y_

        y, self.prediction_updates = theano.scan(fn=sample_network,
                                            sequences=[x],
                                            #n_steps=self.dropout_samples, 
                                            allow_gc=False)

        y = y.transpose(2,0,1).flatten(2).T
        # convert back to whatever precision is set by default
        y = y*self.Ys + self.Ym
        #y = theano.printing.Print('dropout_sample')(y)
        y = y.astype('float64')
        
        n = theano.tensor.cast(y.shape[0], dtype='float64')
        # empirical mean
        M = y.mean(axis=0)
        # empirical covariance
        S = theano.tensor.diag(theano.tensor.exp(2*self.logsn)) + y.T.dot(y)/n - theano.tensor.outer(M,M)
        # Sx^-1 times empirical input output covariance
        if Sx is not None:
            C = x.T.dot(y)/n - theano.tensor.outer(mx,M)
            C = theano.tensor.slinalg.solve_lower_triangular(Lx,C)
            C = theano.tensor.slinalg.solve_upper_triangular(Lx.T,C)
        else:
            C = theano.tensor.zeros((self.D,self.E))
        
        return [M,S,C]

    def predict(self,mx,Sx = None, dropout_samples=None):
        predict = None

        if self.predict_fn is None:
            self.init_predict()
        predict = self.predict_fn

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
        for i in xrange(self.learning_params['iters']):
            start_time = time.time()
            for x,y in utils.iterate_minibatches(self.X.get_value(borrow=True), self.Y.get_value(borrow=True), batch_size, shuffle=True):
                ret = self.train_fn(x,y)
            elapsed_time = time.time() - start_time
            utils.print_with_stamp('iter: %d, loss: %E, elapsed: %E, sn2: %s'%(i,ret,elapsed_time, np.exp(self.logsn.get_value())),self.name,True)
        print ''

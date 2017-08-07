import numpy as np
import theano
import theano.tensor as tt
from theano import function as F, shared as S
from kusanagi.base.Loadable import Loadable
from kusanagi import utils


class BaseRegressor(Loadable):
    ''' Class that implements a regression model. This base class implements
    the logic for getting and setting parameters (as theano shared variables)
    '''
    def __init__(self, name, filename, *args, **kwargs):
        self.param_names = []
        self.fixed_params = []
        Loadable.__init__(self, name=name, filename=self.filename)
        self.register(['param_names', 'fixed_params'])

        # compiled functions
        self.predict_fn = None
        self.predict_d_fn = None

    def init_params(self, **kwargs):
        pass

    def set_params(self, params):
        ''' Adds a a new parameter to the class instance. Every parameter will
        be stored as a Theano shared variable. This function exists so that we
        do not end up with different compiled functions referencing different
        shared variables in memory; which can be a problem when loading pickled
        compiled theano functions
        '''
        if isinstance(params, list):
            params = dict(list(zip(self.param_names, params)))
        for pname in list(params.keys()):
            # if the parameter that was passed here is a shared variable
            if isinstance(params[pname], tt.sharedvar.SharedVariable):
                p = params[pname]
                self.__dict__[pname] = p
                if pname not in self.param_names:
                    self.param_names.append(pname)
            # if the parameter that was passed here is NOT a shared variable
            else:
                # create shared variable if it doesn't exist
                if pname not in self.__dict__ or self.__dict__[pname] is None:
                    p = S(params[pname],name='%s>%s'%(self.name,pname),borrow=True)

                    self.__dict__[pname] = p
                    if pname not in self.param_names:
                        self.param_names.append(pname)
                # otherwise, update the value of the shared variable
                else:
                    p = self.__dict__[pname]
                    pv = params[pname].reshape(p.get_value(borrow=True).shape)
                    p.set_value(pv,borrow=True)

    def get_params(self, symbolic=False, as_dict=False, ignore_fixed=True):
        ''' Returns the parameters of this regressor (theano shared variables).
        '''
        if ignore_fixed:
            params = [ self.__dict__[pname] for pname in self.param_names if (pname in self.__dict__ and self.__dict__[pname] and not pname in self.fixed_params) ]
        else:
            params = [ self.__dict__[pname] for pname in self.param_names if (pname in self.__dict__ and self.__dict__[pname]) ]

        if not symbolic:
            params = [ p.get_value() for p in params]
        if as_dict:
            params = dict(list(zip(self.param_names,params)))
        return params
    
    def set_dataset(self,X_dataset,Y_dataset,**kwargs):
        # ensure we don't change the number of input and output dimensions ( the number of samples can change)
        assert X_dataset.shape[0] == Y_dataset.shape[0], "X_dataset and Y_dataset must have the same number of rows"
        if self.X is not None:
            assert self.X.get_value(borrow=True).shape[1] == X_dataset.shape[1]
        if self.Y is not None:
            assert self.Y.get_value(borrow=True).shape[1] == Y_dataset.shape[1]
        
        # first, convert numpy arrays to appropriate type
        X_dataset = X_dataset.astype( theano.config.floatX )
        Y_dataset = Y_dataset.astype( theano.config.floatX )
        
        # dims
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

    def append_dataset(self,X_dataset,Y_dataset,**kwargs):
        if self.X is None:
            self.set_dataset(X_dataset,Y_dataset,**kwargs)
        else:
            X_ = np.vstack((self.X.get_value(), X_dataset.astype(theano.config.floatX)))
            Y_ = np.vstack((self.Y.get_value(), Y_dataset.astype(theano.config.floatX)))
            self.set_dataset(X_,Y_,**kwargs)

    def get_dataset(self):
        return self.X.get_value(), self.Y.get_value()
    
    def init_predict(self, input_covariance=False, batch_predict=False, *args, **kwargs):
        ''' Compiles a prediction function for the operation specified in self.predict_symbolic'''
        # input variables
        mx = tt.vector('mx')
        Sx = tt.matrix('Sx') if input_covariance else None
    
        # initialize variable for input covariance 
        input_vars = [mx] if not input_covariance else [mx,Sx]

        # get prediction
        utils.print_with_stamp('Initialising expression graph for prediction',self.name)
        output_vars = self.predict_symbolic(mx, Sx, *args, **kwargs)
        
        # outputs
        if not any([isinstance(output_vars, cl) for cl in [tuple, list]]):
            output_vars = [output_vars]
        prediction = [o for o in output_vars if o is not None]

        # compile prediction
        utils.print_with_stamp('Compiling mean and variance of prediction',self.name)

        fn_name = '%s>predict_ui'%(self.name) if input_covariance else '%s>predict'%(self.name)
        predict_fn = theano.function(input_vars,prediction,
                                     on_unused_input='ignore',
                                     name=fn_name,
                                     profile=self.profile,
                                     mode=self.compile_mode,
                                     allow_input_downcast=True)

        utils.print_with_stamp('Done compiling',self.name)

        return predict_fn

    def get_updates(self):
        return theano.updates.OrderedUpdates()

    def predict(self,mx,Sx=None, *args, **kwargs):
        # check if we need to compile the prediction functions
        if Sx is None:
            if not hasattr(self,'predict_fn') or self.predict_fn is None:
                self.predict_fn = self.init_predict(input_covariance=False, *args, **kwargs)
                self.state_changed = True # for saving
            predict = self.predict_fn
        else:
            if not hasattr(self,'predict_ic_fn') or self.predict_ic_fn is None:
                self.predict_ic_fn = self.init_predict(input_covariance=True, *args, **kwargs)
                self.state_changed = True # for saving
            predict = self.predict_ic_fn
        
        # call the predict function with appropriate inputs
        input_vars=[mx]
        if len([p for p in predict.input_storage if p.data is None]) == 2:
            if Sx is None:
                Sx = np.zeros((self.D,self.D))
            input_vars.append(Sx)
        
        res = predict(*input_vars)
        return res
    

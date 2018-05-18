import numpy as np
import theano
import theano.tensor as tt
from theano import shared as S
from kusanagi.base.Loadable import Loadable
from kusanagi import utils
floatX = theano.config.floatX


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
        self.predict_ic_fn = None

    def get_all_shared_vars(self, as_dict=False):
        '''
            Returns all shared variables that are members of this class
        '''
        if as_dict:
            return [(attr_name, self.__dict__[attr_name])
                    for attr_name in list(self.__dict__.keys())
                    if isinstance(self.__dict__[attr_name],
                                  tt.sharedvar.SharedVariable)]
        else:
            return [attr for attr in list(self.__dict__.values())
                    if isinstance(attr, tt.sharedvar.SharedVariable)]

    def get_intermediate_outputs(self):
        '''
            Returns all theano variables that are members of this class
            as they are assumed to be intermediate outputs for the loss or
            prediction functions
        '''
        s = set([attr for attr in list(self.__dict__.values())
                 if isinstance(attr, theano.gof.Variable)])
        return list(s)

    def init_params(self, **kwargs):
        pass

    def set_params(self, params, trainable=False):
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
                    p = S(params[pname],
                          name='%s>%s' % (self.name, pname))

                    self.__dict__[pname] = p
                    if pname not in self.param_names:
                        self.param_names.append(pname)
                # otherwise, update the value of the shared variable
                else:
                    p = self.__dict__[pname]
                    pv = params[pname].reshape(p.get_value().shape)
                    p.set_value(pv)

    def get_params(self, symbolic=False, as_dict=False, ignore_fixed=True):
        ''' Returns the parameters of this regressor (theano shared variables).
        '''
        if ignore_fixed:
            params = [self.__dict__[pname]
                      for pname in self.param_names
                      if (pname in self.__dict__ and self.__dict__[pname]
                      and pname not in self.fixed_params)]
        else:
            params = [self.__dict__[pname]
                      for pname in self.param_names
                      if (pname in self.__dict__
                      and self.__dict__[pname])]

        if not symbolic:
            params = [p.get_value() for p in params]
        if as_dict:
            params = dict(list(zip(self.param_names, params)))
        return params
    
    def remove_params(self, names):
        for pname in names:
            self.__dict__.pop(pname)

    def set_dataset(self, X_dataset, Y_dataset, **kwargs):
        # ensure we don't change the number of input and output dimensions
        # (the number of samples can change)
        assert X_dataset.shape[0] == Y_dataset.shape[0],\
               "X_dataset and Y_dataset must have the same number of rows"
        if self.X is not None:
            assert self.X.get_value(borrow=True).shape[1] == X_dataset.shape[1]
        if self.Y is not None:
            assert self.Y.get_value(borrow=True).shape[1] == Y_dataset.shape[1]

        # first, convert numpy arrays to appropriate type
        X_dataset = X_dataset.astype(floatX)
        Y_dataset = Y_dataset.astype(floatX)

        # dims
        self.N = X_dataset.shape[0]
        self.D = X_dataset.shape[1]
        self.E = Y_dataset.shape[1]

        # now we create symbolic shared variables
        if self.X is None:
            self.X = theano.shared(
                X_dataset, name='%s>X' % (self.name), borrow=True)
        else:
            self.X.set_value(X_dataset, borrow=True)
        if self.Y is None:
            self.Y = theano.shared(
                Y_dataset, name='%s>Y' % (self.name), borrow=True)
        else:
            self.Y.set_value(Y_dataset, borrow=True)

    def append_dataset(self, X_dataset, Y_dataset, **kwargs):
        if self.X is None:
            self.set_dataset(X_dataset, Y_dataset, **kwargs)
        else:
            X_ = np.vstack((self.X.get_value(), X_dataset.astype(floatX)))
            Y_ = np.vstack((self.Y.get_value(), Y_dataset.astype(floatX)))
            self.set_dataset(X_, Y_, **kwargs)

    def update_dataset_statistics(self, X_dataset, Y_dataset):
        """
        Computes the mean and standard deviation of the regression dataset
        inputs and outputs. Useful for standardizing the dataset.
        """
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

    def get_dataset(self):
        return self.X.get_value(), self.Y.get_value()

    def init_predict(self, input_covariance=False, input_ndim=1,
                     *args, **kwargs):
        ''' Compiles a prediction function for the operation specified in
        self.predict'''
        # input variables
        mx = tt.TensorType(floatX, (False,)*input_ndim)('mx')
        Sx = tt.matrix('Sx') if input_covariance else None

        # initialize variable for input covariance
        input_vars = [mx] if not input_covariance else [mx, Sx]

        # get prediction
        utils.print_with_stamp(
            'Initialising expression graph for prediction', self.name)
        output_vars = self.predict(mx, Sx, *args, **kwargs)

        # outputs
        if not any([isinstance(output_vars, cl) for cl in [tuple, list]]):
            output_vars = [output_vars]
        prediction = [o for o in output_vars if o is not None]

        # compile prediction
        utils.print_with_stamp(
            'Compiling mean and variance of prediction', self.name)

        fn_name = ('%s>predict_ui' % (self.name)
                   if input_covariance else '%s>predict' % (self.name))
        if len(prediction) == 1:
            prediction = prediction[0]
        predict_fn = theano.function(input_vars, prediction,
                                     on_unused_input='ignore',
                                     name=fn_name,
                                     allow_input_downcast=True)

        utils.print_with_stamp('Done compiling', self.name)

        return predict_fn

    def get_updates(self):
        return theano.updates.OrderedUpdates()

    def __call__(self, mx, Sx=None, *args, **kwargs):
        # check if we need to compile the prediction functions
        if Sx is None:
            if not hasattr(self, 'predict_fn') or self.predict_fn is None:
                self.predict_fn = self.init_predict(
                    input_covariance=False, input_ndim=mx.ndim,
                    *args, **kwargs)
                self.state_changed = True  # for saving
            predict = self.predict_fn
        else:
            if not hasattr(self, 'predict_ic_fn') or self.predict_ic_fn is None:
                self.predict_ic_fn = self.init_predict(
                    input_covariance=True, input_ndim=mx.ndim,
                    *args, **kwargs)
                self.state_changed = True  # for saving
            predict = self.predict_ic_fn

        # call the predict function with appropriate inputs
        input_vars = [mx]
        if len([p for p in predict.input_storage if p.data is None]) == 2:
            if Sx is None:
                Sx = np.zeros((self.D, self.D))
            input_vars.append(Sx)

        res = predict(*input_vars)
        return res

# pylint: disable=C0103
import numpy as np
import theano
import time
from kusanagi import utils
from scipy.optimize import minimize
from theano.updates import OrderedUpdates
import traceback

SCIPY_MIN_METHODS = ['L-BFGS-B', 'TNC', 'BFGS', 'SLSQP', 'CG']


class ScipyOptimizer(object):
    def __init__(self, min_method='L-BFGS-B',
                 max_evals=150,
                 conv_thr=1e-12,
                 name='ScipyOptimizer'):
        self.min_method = min_method
        self.max_evals = max_evals
        self.conv_thr = conv_thr
        self.name = name

        self.loss_fn = None
        self.grads_fn = None
        self.n_evals = 0
        self.start_time = 0
        self.iter_time = 0
        self.best_p = [None, None, self.n_evals]
        self.params = None
        self.callback = None

    @property
    def min_method(self):
        return self.__min_method

    @min_method.setter
    def min_method(self, min_method):
        self.__min_method = min_method.upper()

        self.alt_min_methods = [min_method]

        # setup alternative minimization methods (in case the one selected
        # fails)
        for method in SCIPY_MIN_METHODS:
            if method not in self.alt_min_methods:
                self.alt_min_methods.append(method)

    def set_objective(self, loss, params, inputs=None, updts=None, grads=None,
                      diff_mode=0, **kwargs):
        '''
            Changes the objective function to be optimized
            @param loss theano graph representing the loss to be optimized
            @param params theano shared variables representing the parameters
                          to be optimized
            @param inputs theano variables representing the inputs required to
                          compute the loss, other than params
            @param updts dictionary of list of theano updates to be applied
                         after every evaluation of the loss function
            @param grads gradients of the loss function. If not provided, will
                         be computed here
        '''
        if inputs is None:
            inputs = []

        if updts is not None:
            updts = OrderedUpdates(updts)

        if grads is None:
            utils.print_with_stamp('Building computation graph for gradients',
                                   self.name)
            grads = theano.grad(loss, params)

        utils.print_with_stamp('Compiling function for loss', self.name)
        self.loss_fn = theano.function(
            inputs, loss, updates=updts, allow_input_downcast=True)
        utils.print_with_stamp('Compiling function for loss+gradients',
                               self.name)
        self.grads_fn = theano.function(
            inputs, [loss, ]+grads, updates=updts, allow_input_downcast=True)

        self.n_evals = 0
        self.start_time = 0
        self.iter_time = 0
        self.params = params

    def loss_wrapper(self, p, p_shapes, *inputs):
        '''
            Loss function wrapper compatible with scipy optimize
            @param p numpy array with the current evaluation point for the loss
            @param p_shapes array with the shapes of every parameter
        '''
        # transform flattened parameter vector into array of parameters
        p = utils.unwrap_params(p, p_shapes)

        # set new parameter values
        for i in range(len(self.params)):
            self.params[i].set_value(p[i])

        # compute value + derivatives
        ret = self.grads_fn(*inputs)
        loss, dloss = ret[0], ret[1:]

        # flatten gradients
        dloss = utils.wrap_params(dloss)

        # cast value and gradients as double precision floats
        # (required by fmin_l_bfgs_b)
        loss, dloss = (np.array(loss).astype(np.float64),
                       np.array(dloss).astype(np.float64))

        # update internal state variables
        self.n_evals += 1
        if loss < self.best_p[0]:
            self.best_p = [loss, p, self.n_evals]
        end_time = time.time()
        iter_time_upt = ((end_time - self.start_time) - self.iter_time)
        iter_time_upt /= self.n_evals
        self.iter_time += iter_time_upt
        msg = 'Current loss: %s, Total evaluations: %d'
        msg += ', Avg. time per call: %f\t'
        utils.print_with_stamp(msg % (str(loss), self.n_evals, self.iter_time),
                               self.name, True)
        self.start_time = time.time()

        if callable(self.callback):
            self.callback(p, loss, dloss)

        # return loss+gradients
        return loss, dloss

    def minimize(self, *inputs, **kwargs):
        '''
            @param inputs python variables to pass as inputs to the compiled
                   theano functions for the loss and gradients
        '''
        self.callback = kwargs.get('callback')
        utils.print_with_stamp('Optimizing parameters', self.name)

        # set initial loss and parameters
        loss0 = self.loss_fn(*inputs)
        utils.print_with_stamp('Initial loss [%s]' % (loss0), self.name)
        p0 = [p.get_value() for p in self.params]
        self.best_p = [loss0, p0, 0]

        # get parameter shapes
        p_shapes = [p.shape for p in p0]
        mloss = utils.MemoizeJac(self.loss_wrapper,
                                 args=(p_shapes,)+inputs)

        # keep on trying to optimize with all the methods, until one succeeds,
        # or we go through all of them
        self.iter_time = 0
        self.start_time = time.time()
        self.n_evals = 0
        for min_method in self.alt_min_methods:
            try:
                utils.print_with_stamp("Using %s optimizer" % (min_method),
                                       self.name)
                p0_wrapped = utils.wrap_params(p0)
                opts = {'maxiter': self.max_evals,
                        'ftol': 1e5*np.finfo(float).eps,
                        'gtol': 1.0e-7}
                if min_method.lower() == 'l-bfgs-b':
                    opts['maxfun'] = self.max_evals
                    opts['maxcor'] = min(100, p0_wrapped.size)
                    opts['maxls'] = 30

                opt_res = minimize(mloss, p0_wrapped,
                                   jac=mloss.derivative,
                                   method=min_method,
                                   tol=self.conv_thr,
                                   options=opts)
                # set params to new values
                popt = utils.unwrap_params(opt_res.x, p_shapes)
                for i in range(len(self.params)):
                    self.params[i].set_value(popt[i])
                # break the loop since we succeeded
                break
            except (ValueError, np.linalg.LinAlgError):
                print('')
                traceback.print_exc()
                traceback.print_stack()
                msg = "Optimization with %s failed"
                utils.print_with_stamp(msg % (self.min_method),
                                       self.name)
                loss, popt = self.best_p[:2]
                for i in range(len(self.params)):
                    self.params[i].set_value(popt[i])
        print('')
        v, p, i = self.best_p
        for sp_i, p_i in zip(self.params, p):
            sp_i.set_value(p_i)
        v = self.loss_fn(*inputs)
        msg = 'Done training. New loss [%f] iter: [%d]'
        utils.print_with_stamp(msg % (v, i), self.name)

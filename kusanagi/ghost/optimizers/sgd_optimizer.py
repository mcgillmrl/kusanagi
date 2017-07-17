# pylint: disable=C0103
import lasagne
import numpy as np
import theano
import time
import traceback
from theano.updates import OrderedUpdates
from kusanagi import utils
from scipy.optimize import minimize

LASAGNE_MIN_METHODS = {'sgd': lasagne.updates.sgd,
                       'momentum': lasagne.updates.momentum,
                       'nesterov': lasagne.updates.nesterov_momentum,
                       'nesterov_momentum': lasagne.updates.nesterov_momentum,
                       'adagrad': lasagne.updates.adagrad,
                       'rmsprop': lasagne.updates.rmsprop,
                       'adadelta': lasagne.updates.adadelta,
                       'adam': lasagne.updates.adam,
                       'nadam': utils.updates.nadam,
                       'adamax': lasagne.updates.adamax
                      }

class SGDOptimizer(object):
    def __init__(self, min_method='ADAM',
                 max_evals=1000,
                 conv_thr=1e-12,
                 name='SGDOptimizer', **kwargs):
        self.min_method = min_method
        self.max_evals = max_evals
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
        self.__min_method = min_method.lower()

    def set_objective(self, loss, params, inputs=None, updts=None, grads=None, clip=None, **kwargs):
        '''
            Changes the objective function to be optimized
            @param loss theano graph representing the loss to be optimized
            @param params theano shared variables representing the parameters to be optimized
            @param inputs theano variables representing the inputs required to compute the loss,
                          other than params
            @param updts dictionary of list of theano updates to be applied after every evaluation
                         of the loss function
            @param grads gradients of the loss function. If not provided, will be computed here
            @param kwargs arguments to pass to the lasagne.updates function
        '''
        if inputs is None:
            inputs = []

        if updts is not None:
            updts = OrderedUpdates(updts)

        if grads is None:
            utils.print_with_stamp('Building computation graph for gradients',
                                   self.name)
            grads = theano.grad(loss, params)
            if clip is not None:
                grads = lasagne.updates.total_norm_constraint(grads, clip)

        
        utils.print_with_stamp("Computing parameter update rules", self.name)
        min_method_updt = LASAGNE_MIN_METHODS[self.min_method]
        grad_updates = min_method_updt(grads, params, **kwargs)

        utils.print_with_stamp('Compiling function for loss', self.name)
        self.loss_fn = theano.function(inputs, loss, updates=updts,
                                       on_unused_input='ignore',
                                       allow_input_downcast=True, profile=True)

        utils.print_with_stamp("Compiling parameter updates", self.name)
        self.update_params_fn = theano.function(inputs, [loss]+grads,
                                                updates=grad_updates+updts,
                                                on_unused_input='ignore',
                                                allow_input_downcast=True, profile=True)

        self.n_evals = 0
        self.start_time = 0
        self.iter_time = 0
        self.params = params

    def minibatch_minimize(self, X, Y, *inputs, **kwargs):
        callback = kwargs.get('callback')
        batch_size = kwargs.get('batch_size', 100)
        batch_size = min(batch_size, X.shape[0])
        self.iter_time = 0
        self.start_time = time.time()
        self.n_evals = 0
        utils.print_with_stamp('Optimizing parameters via mini batches', self.name)
        # set initial loss and parameters
        loss0 = self.loss_fn(X[-batch_size:], Y[-batch_size:], *inputs)
        utils.print_with_stamp('Initial loss [%s]'%(loss0), self.name)
        p = [p.get_value() for p in self.params]
        self.best_p = [loss0, p, 0]

        # go through the dataset
        while True:
            start_time = time.time()
            should_exit=False
            for x, y in utils.iterate_minibatches(X, Y, batch_size, shuffle=True):
                # mini batch update
                ret = self.update_params_fn(x, y, *inputs)
                # the returned loss corresponds to the parameters BEFORE the update
                loss, dloss = ret[0], ret[1:]
                p = [p.get_value() for p in self.params]
                if loss < self.best_p[0]:
                    self.best_p = [loss, p, self.n_evals]
                if callable(callback):
                    callback(p, loss, dloss)
                self.n_evals+=1
                if self.n_evals > self.max_evals:
                    should_exit = True
                    break
                end_time = time.time()
                self.iter_time += ((end_time - start_time) - self.iter_time)/self.n_evals
                out_str = 'Current value: %E, Total evaluations: %d, Avg. time per updt: %f'
                utils.print_with_stamp(out_str%(self.loss_fn(x, y, *inputs), self.n_evals, self.iter_time),
                                       self.name,True)
            if should_exit:
                break
        print('')

    def minimize(self, *inputs, **kwargs):
        '''
            @param inputs python variables to pass as inputs to the compiled theano functions
                          for the loss and gradients
        '''
        callback = kwargs.get('callback')
        self.iter_time = 0
        self.start_time = time.time()
        self.n_evals = 0
        utils.print_with_stamp('Optimizing parameters', self.name)

        # set initial loss and parameters
        loss0 = self.loss_fn(*inputs)
        utils.print_with_stamp('Initial loss [%s]'%(loss0), self.name)
        p = [p.get_value() for p in self.params]
        self.best_p = [loss0, p, 0]

        # training loop
        for i in range(self.max_evals):
            # evaluate current policy and update parameters
            start_time = time.time()
            ret = self.update_params_fn(*inputs)
            # the returned loss corresponds to the parameters BEFORE the update
            loss, dloss = ret[0], ret[1:]
            p = [p.get_value() for p in self.params]
            if loss < self.best_p[0]:
                self.best_p = [loss, p, i]
            if callable(callback):
                callback(p, loss, dloss)
            self.n_evals+=1
            end_time = time.time()
            self.iter_time += ((end_time - start_time) - self.iter_time)/self.n_evals
            out_str = 'Current value: %E, Total evaluations: %d, Avg. time per updt: %f'
            utils.print_with_stamp(out_str%(self.loss_fn(*inputs), self.n_evals, self.iter_time),
                                   self.name,True)

        print('') 
        v, p, i = self.best_p
        utils.print_with_stamp('Done training. New value [%f] iter: [%d]'%(v, i), self.name)

# pylint: disable=C0103
import lasagne
import numpy as np
import theano
import time

from theano.updates import OrderedUpdates
from kusanagi import utils

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

    def set_objective(self, loss, params, inputs=None, updts=None, grads=None,
                      clip=None, **kwargs):
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
        # converts inputs to shared variables to avoid repeated gpu transfers
        self.shared_inpts = [theano.shared(np.empty([1]*inp.ndim, 
                                           dtype=inp.dtype),
                                           name=inp.name) for inp in inputs]
        givens_dict = dict(zip(inputs, self.shared_inpts))                                       
        self.loss_fn = theano.function([], loss, updates=updts,
                                       on_unused_input='ignore',
                                       allow_input_downcast=True,
                                       givens=givens_dict)

        utils.print_with_stamp("Compiling parameter updates", self.name)
        self.update_params_fn = theano.function([], [loss]+grads,
                                                updates=grad_updates+updts,
                                                on_unused_input='ignore',
                                                allow_input_downcast=True,
                                                givens=givens_dict)

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
        utils.print_with_stamp('Optimizing parameters via mini batches',
                               self.name)
        # set values for shared inputs
        self.shared_inpts[0].set_value(X[-batch_size:])
        self.shared_inpts[1].set_value(Y[-batch_size:])
        for s,i in zip(self.shared_inpts[2:], inputs):
            s.set_value(np.array(i).astype(s.dtype))

        # set initial loss and parameters
        loss0 = self.loss_fn()
        utils.print_with_stamp('Initial loss [%s]' % (loss0), self.name)
        p = [p.get_value() for p in self.params]
        self.best_p = [loss0, p, 0]

        # go through the dataset
        out_str = 'Curr loss: %E, n_evals: %d, Avg. time per updt: %f'
        while True:
            start_time = time.time()
            should_exit = False
            b_iter = utils.iterate_minibatches(X, Y, batch_size, shuffle=True)
            for x, y in b_iter:
                # mini batch update
                self.shared_inpts[0].set_value(x)
                self.shared_inpts[1].set_value(y)
                ret = self.update_params_fn()
                # the returned loss corresponds to the parameters
                # BEFORE the update
                loss, dloss = ret[0], ret[1:]
                p = [p.get_value() for p in self.params]
                if loss < self.best_p[0]:
                    self.best_p = [loss, p, self.n_evals]
                if callable(callback):
                    callback(p, loss, dloss)
                self.n_evals += 1
                if self.n_evals > self.max_evals:
                    should_exit = True
                    break

                end_time = time.time()
                dt = end_time - start_time
                it_updt = (dt - self.iter_time)/self.n_evals
                self.iter_time += it_updt
                str_params = (loss, self.n_evals, self.iter_time)
                utils.print_with_stamp(out_str % str_params, self.name, True)
            if should_exit:
                break
        print('')

    def minimize(self, *inputs, **kwargs):
        '''
            @param inputs python variables to pass as inputs to the compiled
                          theano functions for the loss and gradients
        '''
        callback = kwargs.get('callback')
        self.iter_time = 0
        self.start_time = time.time()
        self.n_evals = 0
        utils.print_with_stamp('Optimizing parameters', self.name)
        # set values for shared inputs
        for s,i in zip(self.shared_inpts, inputs):
            s.set_value(np.array(i).astype(s.dtype))

        # set initial loss and parameters
        loss0 = self.loss_fn()
        utils.print_with_stamp('Initial loss [%s]' % (loss0), self.name)
        p = [p.get_value() for p in self.params]
        self.best_p = [loss0, p, 0]

        # training loop
        for i in range(self.max_evals):
            # evaluate current policy and update parameters
            start_time = time.time()
            ret = self.update_params_fn()
            # the returned loss corresponds to the parameters BEFORE the update
            loss, dloss = ret[0], ret[1:]

            if loss < self.best_p[0]:
                p = [p.get_value() for p in self.params]
                self.best_p = [loss, p, i]
            if callable(callback):
                callback(loss, dloss)
            self.n_evals += 1

            end_time = time.time()
            dt = end_time - start_time
            it_updt = (dt - self.iter_time)/self.n_evals
            self.iter_time += it_updt
            out_str = 'Curr loss: %E, n_evals: %d, Avg. time per updt: %f'
            str_params = (loss, self.n_evals, self.iter_time)
            utils.print_with_stamp(out_str % str_params, self.name, True)

        print('') 
        v, p, i = self.best_p
        utils.print_with_stamp('Done training. New loss [%f] iter: [%d]' % (v, i), self.name)

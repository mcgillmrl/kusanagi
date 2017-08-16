# pylint: disable=C0103
from __future__ import print_function
import lasagne
import numpy as np
import theano
import time

from theano.updates import OrderedUpdates
from kusanagi import utils
floatX = theano.config.floatX

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
                      polyak_averaging=0.99, clip=None, trust_input=True,
                      **kwargs):
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

        outputs = [loss]+grads
        grad_updates = grad_updates+updts
        if polyak_averaging and polyak_averaging > 0.0:
            # create copy of params
            params_avg = [
                theano.shared(p.get_value(borrow=False,
                                          return_internal_type=True),
                              name=p.name+'_copy')
                for p in params]

            # prepare updates for polyak averaging
            prev_loss = theano.shared(
                np.array(0, dtype=floatX))
            t = theano.shared(np.array(1, dtype=floatX))
            b = polyak_averaging

            loss_avg = ((b-b**t)*prev_loss + (1-b)*loss)/(1-b**t)

            grad_updates[prev_loss] = loss_avg
            grad_updates[t] = t+1
            for p, pp in zip(params, params_avg):
                grad_updates[pp] = ((b-b**t)*pp
                                    + (1-b)*grad_updates[p])/(1-b**t)

            outputs[0] = loss_avg
            self.params_avg = params_avg
            self.prev_loss = prev_loss
        else:
            if hasattr(self, 'params_avg'):
                delattr(self, 'params_avg')

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
        self.loss_fn.trust_input = trust_input

        utils.print_with_stamp("Compiling parameter updates", self.name)

        self.update_params_fn = theano.function(
            [], outputs,
            updates=grad_updates,
            on_unused_input='ignore',
            allow_input_downcast=True,
            givens=givens_dict)
        self.update_params_fn.trust_input = trust_input

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
        for s, i in zip(self.shared_inpts[2:], inputs):
            s.set_value(np.array(i).astype(s.dtype))

        # set initial loss and parameters
        params = (self.params
                  if not hasattr(self, 'params_avg')
                  else self.params)
        ret = self.update_params_fn()
        loss0 = self.loss_fn()
        utils.print_with_stamp('Initial loss [%s]' % (loss0), self.name)
        p = [p.get_value(return_internal_type=True, borrow=False)
             for p in params]
        self.best_p = [loss0, p, 0]

        if hasattr(self, 'params_avg'):
            for p, pp in zip(self.params, self.params_avg):
                pp.set_value(p.get_value(borrow=False))
            self.prev_loss.set_value(loss0)

        # go through the dataset
        out_str = 'Curr loss: %E, n_evals: %d, Avg. time per updt: %f'
        while True:
            start_time = time.time()
            should_exit = False
            b_iter = utils.iterate_minibatches(X, Y, batch_size, shuffle=True)
            for x, y in b_iter:
                start_time = time.time()
                # get previous params
                p = [p.get_value(return_internal_type=True, borrow=False)
                     for p in params]

                # mini batch update
                self.shared_inpts[0].set_value(x)
                self.shared_inpts[1].set_value(y)
                ret = self.update_params_fn()

                # the returned loss and gradients correspond to the parameters
                # BEFORE the update
                loss, dloss = ret[0], ret[1:]

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
        v, p, i = self.best_p
        for sp_i, p_i in zip(params, p):
            sp_i.set_value(p_i)
        v = self.loss_fn()
        msg = 'Done training. New loss [%f] iter: [%d]'
        utils.print_with_stamp(msg % (v, i), self.name)

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
        for s, i in zip(self.shared_inpts, inputs):
            s.set_value(np.array(i).astype(s.dtype))
        # set initial loss and parameters
        params = (self.params
                  if not hasattr(self, 'params_avg')
                  else self.params)
        ret = self.update_params_fn()
        loss0 = self.loss_fn()
        utils.print_with_stamp('Initial loss [%s]' % (loss0), self.name)
        p = [p.get_value(return_internal_type=True, borrow=False)
             for p in params]
        self.best_p = [loss0, p, 0]

        if hasattr(self, 'params_avg'):
            for p, pp in zip(self.params, self.params_avg):
                pp.set_value(p.get_value(borrow=False))
            self.prev_loss.set_value(loss0)

        # training loop
        for i in range(self.max_evals):
            start_time = time.time()

            # get previous params
            p = [p.get_value(return_internal_type=True, borrow=False)
                 for p in params]

            # evaluate current policy and update parameters
            ret = self.update_params_fn()
            # the returned loss corresponds to the parameters BEFORE the update
            loss, dloss = ret[0], ret[1:]

            if loss < self.best_p[0]:
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
        for sp_i, p_i in zip(params, p):
            sp_i.set_value(p_i)
        v = self.loss_fn()
        msg = 'Done training. New loss [%f] iter: [%d]'
        utils.print_with_stamp(msg % (v, i), self.name)

# pylint: disable=C0103
import numpy as np
import theano
import time
from kusanagi import utils
from scipy.optimize import minimize
import traceback

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
                 name='ScipyOptimizer'):
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

    def set_objective(self, loss, params, inputs=None, updts=None, grads=None, **kwargs):
        '''
            Changes the objective function to be optimized
            @param loss theano graph representing the loss to be optimized
            @param params theano shared variables representing the parameters to be optimized
            @param inputs theano variables representing the inputs required to compute the loss,
                          other than params
            @param updts dictionary of list of theano updates to be applied after every evaluation
                         of the loss function
            @param grads gradients of the loss function. If not provided, will be computed here
            @param kwargs
        '''
        if inputs is None:
            inputs = []

        min_method_updt = STOCHASTIC_MIN_METHODS[min_method]
        if grads is None:
            utils.print_with_stamp('Building computation graph for gradients',
                                   self.name)
            grads = theano.grad(loss, params)

        utils.print_with_stamp("Computing gradient update rules", self.name)
        grad_updates = min_method_updt(grads, params)

        utils.print_with_stamp("Compiling optimizer", self.name)
        self.update_params_fn = theano.function(inputs, [loss]+grads, updates=grad_updates+updts)

        self.n_evals = 0
        self.start_time = 0
        self.iter_time = 0
        self.params = params

    def minimize(self, *inputs, **kwargs):
        '''
            @param inputs python variables to pass as inputs to the compiled theano functions
                          for the loss and gradients
        '''
        callback = kwargs.get('callback')
        lr = kwargs.get('lr', 1e-4)
        self.iter_time = 0
        self.start_time = time.time()
        self.n_evals = 0
        utils.print_with_stamp('Optimizing parameters', self.name)
        
        self.best_p = [loss0, p0, 0]
        # training loop
        for i in range(self.max_evals):
            # evaluate current policy and update parameters
            ret = self.update_fn()     # v corresponds to the parameters before the training update
            loss, dloss = ret[0], ret[1:]
            if loss < self.best_p[0]:
                self.best_p = [v,p,i]
            self.n_evals+=1
            gmag = [np.sqrt((p**2).sum()) for p in dJdp]
            gmax = [p.max() for p in dJdp]
            out_str = 'Current value: %E, Total evaluations: %d, gm: %s, lr: %f'
            utils.print_with_stamp(out_str%(v, self.n_evals, gmag, lr), self.name,True)

        print('') 
        v, p, i = self.best_p
        utils.print_with_stamp('Done training. New value [%f] iter: [%d]'%(v, i), self.name)

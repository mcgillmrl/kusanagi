import lasagne
import numpy as np
import theano
import theano.tensor as tt
from collections import OrderedDict

def polyak_averaging(updates, params,alpha=0.999):
    if type(params) is not list:
        params = [params]
    
    t_prev = theano.shared(lasagne.utils.floatX(0.))
    t = t_prev+1
    one = tt.constant(1)
    for param in params:
        value = param.get_value(borrow=True)
        # parameter estimate without averaging
        param_est = theano.shared(np.zeros(value.shape), dtype=value.dtype, broadcastable=param.broadcastable)
        old_updt = updates[param]
        updates[param] = updates_param*alpha + params_avg*(1-alpha)
        updates[param_est] = old_updt

def nadam(loss_or_grads, params, learning_rate=0.001, beta1=0.9,
         beta2=0.999, epsilon=1e-8):
    all_grads = lasagne.updates.get_or_compute_grads(loss_or_grads, params)
    t_prev = theano.shared(lasagne.utils.floatX(0.))
    updates = OrderedDict()

    # Using theano constant to prevent upcasting of float32
    one = tt.constant(1)

    t = t_prev + 1
    a_t = learning_rate*tt.sqrt(one-beta2**t)/(one-beta1**t)

    for param, g_t in zip(params, all_grads):
        value = param.get_value(borrow=True)
        m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)
        v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)

        m_t = beta1*m_prev + (one-beta1)*g_t
        v_t = beta2*v_prev + (one-beta2)*g_t**2

        m_t_hat = beta1*m_t*((one-beta1**t)/(one-beta1**(t+1))) + (1-beta1)*g_t
        v_t_hat = beta2*v_t

        step = a_t*m_t_hat/(tt.sqrt(v_t_hat) + epsilon)

        updates[m_prev] = m_t
        updates[v_prev] = v_t
        updates[param] = param - step

    updates[t_prev] = t
    return updates


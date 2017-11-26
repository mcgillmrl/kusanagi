import lasagne
import numpy as np
import theano
import theano.tensor as tt
from kusanagi.ghost.regression.layers import (GaussianDropoutLayer,
                                              DenseDropoutLayer,
                                              DenseGaussianDropoutLayer,
                                              DenseLogNormalDropoutLayer)


def gaussian_log_likelihood(targets, pred_mean, pred_std=None):
    ''' Computes the log likelihood for gaussian distributed predictions.
        This assumes diagonal covariances
    '''
    delta = pred_mean - targets
    # note that if we have nois be a 1xD vector, broadcasting
    # rules apply
    if pred_std:
        # sum over output dimensions
        lml = -tt.square(delta/pred_std).sum(-1)*0.5 - tt.log(pred_std).sum(-1)
    else:
        # sum ove output dimensions
        lml = -tt.square(delta).sum(-1)*0.5

    # sum over all examples
    return lml.sum()


def dropout_gp_kl(output_layer, input_lengthscale=1.0, hidden_lengthscale=1.0):
    '''
        KL divergence approximation for the dropout uncertainty model from
        Gal and Ghahrammani, 2015
    '''
    eps = np.finfo(np.__dict__[theano.config.floatX]).eps
    layers = lasagne.layers.get_all_layers(output_layer)

    reg = []
    for i in range(1, len(layers)):
        reg_weight = 0.5
        # apply different regularization weigths to the input,
        # and the hidden dimension
        is_dropout_a = isinstance(
            layers[i-1], lasagne.layers.DropoutLayer)
        is_dropout_b = isinstance(
            layers[i], DenseDropoutLayer) and\
            not isinstance(layers[i], DenseGaussianDropoutLayer) and\
            not isinstance(layers[i], DenseLogNormalDropoutLayer)

        is_dropout = is_dropout_a or is_dropout_b

        ind = i if not (i > 1 and is_dropout_a) else i-1

        is_input = isinstance(layers[ind].input_layer,
                              lasagne.layers.InputLayer)

        if hasattr(layers[ind], 'input_layer') and is_input:
            reg_weight *= input_lengthscale**2
        else:
            reg_weight *= hidden_lengthscale**2

        # if this layer has a weight layer and the previous layer
        # is a DropoutLayer
        if hasattr(layers[i], 'W'):
            # this p is the parameter of bernoulli (not dropout prob)
            p = 1.0
            if i > 1 and is_dropout:
                if layers[ind].p != 0:
                    p = 1 - layers[ind].p
            W = layers[i].W
            W_reg = reg_weight*tt.sum(p*W*W)
            p_reg = -tt.sum(p*tt.log(p + eps))
            reg.append(W_reg + p_reg)

        if hasattr(layers[i], 'b') and layers[i].b is not None:
            b = layers[i].b
            reg.append(reg_weight*tt.sum(b**2))

    return sum(reg)


def gaussian_dropout_kl(output_layer, input_lengthscale=1.0,
                        hidden_lengthscale=1.0):
    '''
        KL divergence approximation from :
         "Variational Dropout Sparsifies Deep Neural Networks"
         Molchanov et al, 2017
    '''
    layers = lasagne.layers.get_all_layers(output_layer)
    k1, k2, k3 = 0.63576, 1.8732, 1.48695
    C = -0.20452104900969109
    # C= -k1
    reg = []
    sigmoid = tt.nnet.sigmoid
    for i in range(1, len(layers)):
        # check if this is a dropout layer
        is_dropout_a = isinstance(layers[i], GaussianDropoutLayer)
        is_dropout_b = isinstance(layers[i], DenseGaussianDropoutLayer)
        if is_dropout_a or is_dropout_b:
            log_alpha = layers[i].log_alpha
            # there should be one log_alpha per weight
            log_alpha_shape = tuple(log_alpha.shape.eval())
            W_shape = tuple(layers[i].W.get_value().shape)
            if log_alpha_shape != W_shape:
                # we assume that if alpha does not have the same shape as W
                # (i.e. one alpha parameter per weight) there's either one per
                # output or per layer
                # TODO make this compatible with conv layers
                log_alpha = (log_alpha*tt.ones_like(layers[i].W.T)).T
            kl = -(k1*sigmoid(k2+k3*log_alpha)
                   - 0.5*tt.log1p(tt.exp(-log_alpha))
                   + C)
            is_input = isinstance(layers[i].input_layer,
                                  lasagne.layers.InputLayer)
            rw = input_lengthscale if is_input else hidden_lengthscale
            reg.append(rw*kl.sum())

    return sum(reg)


def soft_orthogonality_constraint(output_layer, rw=1.0):
    layers = lasagne.layers.get_all_layers(output_layer)
    reg = []
    for i in range(len(layers)):
        if hasattr(layers[i], 'W'):
            W = layers[i].W
            norm = lasagne.regularization.l2(W.T.dot(W) - tt.eye(W.shape[1]))
            reg.append(rw*norm)
    return sum(reg)


def Phi(x):
    return 0.5*(tt.erfc(-x/tt.sqrt(2)))


def phi(x):
    return tt.exp(-0.5*x**2)/tt.sqrt(2*np.pi)


def log_normal_kl(output_layer, input_lengthscale=1.0, hidden_lengthscale=1.0):
    layers = lasagne.layers.get_all_layers(output_layer)
    reg = []
    for i in range(1, len(layers)):
        is_dropout = isinstance(layers[i], DenseLogNormalDropoutLayer)
        if is_dropout:
            a, b = layers[i].interval
            mu = layers[i].mu
            sigma = layers[i].sigma
            alpha = layers[i].alpha
            beta = layers[i].beta
            Z = layers[i].Z
            kl = (tt.log(b-a) - tt.log(tt.sqrt(2*np.pi)*sigma) - tt.log(Z)
                  - ((alpha*phi(alpha) - beta*phi(beta))/sigma)/(2*Z))

            is_input = isinstance(layers[i].input_layer,
                                  lasagne.layers.InputLayer)
            rw = input_lengthscale if is_input else hidden_lengthscale
            reg.append(rw*kl.sum())
    return sum(reg)

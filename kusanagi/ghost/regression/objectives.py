import lasagne
import theano
import theano.tensor as tt
from kusanagi.ghost.regression.layers import (GaussianDropoutLayer,
                                              DenseDropoutLayer,
                                              DenseGaussianDropoutLayer)


def gaussian_log_likelihood(targets, pred_mean, pred_std=None):
    ''' Computes the eempirical expected value of the log likelihood,
    for a gaussian distributed predictions. This assumes diagonal covariances
    '''
    delta = pred_mean - targets
    # note that if we have nois be a 1xD vector, broadcasting
    # rules apply
    if pred_std:
        lml = tt.square(delta/pred_std).sum(-1) + tt.log(pred_std).sum(-1)
    else:
        lml = tt.square(delta).sum(-1)

    lml = -0.5*lml

    return lml


def dropout_gp_kl(output_layer, input_lengthscale=1.0, hidden_lengthscale=1.0):
    '''
        KL divergence approximation for the dropout uncertainty model from
        Gal and Ghahrammani, 2015
    '''
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
            not isinstance(layers[i], DenseGaussianDropoutLayer)

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
            if i > 1 and is_dropout:
                p = layers[ind].p
                if p > 0:
                    reg_weight *= (1-p)
            reg.append(reg_weight*lasagne.regularization.l2(layers[i].W))

        if hasattr(layers[i], 'b') and layers[i].b is not None:
            reg.append(reg_weight*lasagne.regularization.l2(layers[i].b))

    return sum(reg)


def gaussian_dropout_kl(output_layer, input_lengthscale=1.0, hidden_lengthscale=1.0):
    '''
        KL divergence approximation from :
         "Variational Dropout Sparsifies Deep Neural Networks"
         Molchanov et al, 2017
    '''
    layers = lasagne.layers.get_all_layers(output_layer)
    k1, k2, k3 = 0.63576, 1.8732, 1.48695
    C = -0.20452104900969109
    #C= -k1
    reg = []
    sigmoid = tt.nnet.sigmoid
    for i in range(1, len(layers)):
        # check if this is a dropout layer
        is_dropout_a = isinstance(layers[i], GaussianDropoutLayer)
        is_dropout_b = isinstance(layers[i], DenseGaussianDropoutLayer)
        if is_dropout_a or is_dropout_b:
            log_alpha = layers[i].log_alpha
            kl = -(k1*sigmoid(k2+k3*log_alpha)
                   - 0.5*tt.log1p(tt.exp(-log_alpha))
                   + C)
            is_input = isinstance(layers[i].input_layer,
                                  lasagne.layers.InputLayer)
            rw = input_lengthscale if is_input else hidden_lengthscale
            reg.append(rw*kl.sum())

    return sum(reg)

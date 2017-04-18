import lasagne
import theano
import numpy as np
import theano.tensor as tt

from lasagne.layers.base import Layer
from lasagne.random import get_rng

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


class DropoutLayer(lasagne.layers.noise.DropoutLayer):
    '''
    Modification to the dropout layer from lasagne, that give control over 
    when to sample the masks
    '''
    def __init__(self,incoming,p=0.5,rescale=True,mask=None, dropout_samples=None,**kwargs):
        super(DropoutLayer,self).__init__(incoming,p,rescale,**kwargs)
        if mask is None:
            if dropout_samples is not None:
                # we are going to create a shared variable for storing the mask (with the appropriate number of dimensions)
                # and provide an update expression that the user can use later on to update the mask; i.e. the dropout mask will
                # be fixed unless the user updates it
                name = 'mask' if self.name is None else self.name+'>mask'
                mask_shape = [s if not s is None else dropout_samples for s in self.input_shape]
                mask = theano.shared(np.random.binomial(1,1-p,mask_shape).astype(theano.config.floatX),name=name)
        self.mask = mask
        # initalize dropout mask parameter
        if self.mask is not None:
            self.mask = self.add_param(self.mask, self.mask.get_value().shape, self.mask.name, trainable=False,  regularizable=False)
        self.mask_updates = None

    def get_output_for(self, input, deterministic=False, fixed_dropout_masks=False, **kwargs):
        if deterministic or self.p == 0:
            return input
        else:
            # Using theano constant to prevent upcasting
            one = tt.constant(1)

            retain_prob = one - self.p
            if self.rescale:
                input /= retain_prob
            
            # use nonsymbolic shape for dropout mask if possible
            mask_shape = self.input_shape
            if any(s is None for s in mask_shape):
                mask_shape = input.shape

            # apply dropout, respecting shared axes
            if self.shared_axes:
                shared_axes = tuple(a if a >= 0 else a + input.ndim
                                    for a in self.shared_axes)
                mask_shape = tuple(1 if a in shared_axes else s
                                for a, s in enumerate(mask_shape))
            mask = self._srng.binomial(mask_shape, p=retain_prob,
                                    dtype=input.dtype)
            if self.shared_axes:
                bcast = tuple(bool(s == 1) for s in mask_shape)
                mask = tt.patternbroadcast(mask, bcast)
            
            if self.mask is not None and fixed_dropout_masks:
                # the user may update the shared mask value however they want, but here we provide an update expression
                # note that if the batch size changes, the update will only have an effect at the next call causing a shape
                # mis-match in the elementwise product. To avoid this, the user should update the masks before performing
                # a forward pass on this layer.
                self.mask_updates = mask

                # make sure that we use the local shared variable as the mask
                mask = self.mask

            return input * mask 

def relu(x):
    return tt.maximum(x, 0)

class WeightNormLayer(lasagne.layers.Layer):
    '''
	Weight Normalization implementation by Tim Sallimans as described in the paper:

        Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks
	Tim Salimans, Diederik P. Kingma

    '''
    def __init__(self, incoming, b=lasagne.init.Constant(0.), g=lasagne.init.Constant(1.),
                 W=lasagne.init.Normal(0.05), nonlinearity=relu, **kwargs):
        super(WeightNormLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = nonlinearity
        k = self.input_shape[1]
        if b is not None:
            self.b = self.add_param(b, (k,), name="b", regularizable=False)
        if g is not None:
            self.g = self.add_param(g, (k,), name="g")
        if len(self.input_shape)==4:
            self.axes_to_sum = (0,2,3)
            self.dimshuffle_args = ['x',0,'x','x']
        else:
            self.axes_to_sum = 0
            self.dimshuffle_args = ['x',0]
        
        # scale weights in layer below
        incoming.W_param = incoming.W
        incoming.W_param.set_value(W.sample(incoming.W_param.get_value().shape))
        if incoming.W_param.ndim==4:
            W_axes_to_sum = (1,2,3)
            W_dimshuffle_args = [0,'x','x','x']
        else:
            W_axes_to_sum = 0
            W_dimshuffle_args = ['x',0]
        if g is not None:
            incoming.W = incoming.W_param * (self.g/tt.sqrt(tt.sum(tt.square(incoming.W_param),axis=W_axes_to_sum))).dimshuffle(*W_dimshuffle_args)
        else:
            incoming.W = incoming.W_param / tt.sqrt(tt.sum(tt.square(incoming.W_param),axis=W_axes_to_sum,keepdims=True))        

    def get_output_for(self, input, init=False, **kwargs):
        if init:
            m = tt.mean(input, self.axes_to_sum)
            input -= m.dimshuffle(*self.dimshuffle_args)
            stdv = tt.sqrt(tt.mean(tt.square(input),axis=self.axes_to_sum))
            input /= stdv.dimshuffle(*self.dimshuffle_args)
            self.init_updates = [(self.b, -m/stdv), (self.g, self.g/stdv)]
        elif hasattr(self,'b'):
            input += self.b.dimshuffle(*self.dimshuffle_args)
            
        return self.nonlinearity(input)
        
def weight_norm(layer, **kwargs):
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = lasagne.nonlinearities.identity
    if hasattr(layer, 'b'):
        del layer.params[layer.b]
        layer.b = None
    return WeightNormLayer(layer, nonlinearity=nonlinearity, **kwargs)

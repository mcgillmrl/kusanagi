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
    def __init__(self,incoming,p=0.5,rescale=True,**kwargs):
        super(DropoutLayer,self).__init__(incoming,p,rescale,**kwargs)
        self.mask = None
        self.mask_update = None

    def get_output_for(self, input, deterministic=False, dropout_masks=None, **kwargs):
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

            if dropout_masks is not None and self in dropout_masks:
                # we are going to create a shared variable for storing the mask (with the appropriate number of dimensions)
                # and provide an update expression that the user can use later on to update the mask; i.e. the dropout mask will
                # be fixed unless the user updates it
                # First, create the shared variable 
                self.mask = theano.shared(np.ones([1]*mask.ndim)) if dropout_masks[self] is None else dropout_masks[self]

                # the user may update the shared mask value however they want, but here we provide an update expression
                self.mask_update = mask

                # finally, make sure that we use the shared variable as the mask
                mask = self.mask

            return input * mask

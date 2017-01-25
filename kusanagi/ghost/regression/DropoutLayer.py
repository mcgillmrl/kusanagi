import lasagne
import theano
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

    def get_output_for(self, input, deterministic=False, dropout_masks=None, **kwargs):
	"""
	Parameters
	----------
	input : tensor
	    output from the previous layer
	deterministic : bool
	    If true dropout and scaling is disabled, see notes
	"""

	if deterministic or self.p == 0:
	    return input
	else:
            # Using theano constant to prevent upcasting
            one = tt.constant(1)

            retain_prob = one - self.p
            if self.rescale:
                input /= retain_prob

            if isinstance(dropout_masks,dict) and self.name in dropout_masks:
	        mask = dropout_masks[self.name]
	    else:
                # use nonsymbolic shape for dropout mask if possible
                input_shape = self.input_shape
                if any(s is None for s in input_shape):
                    input_shape = input.shape
	        mask = self._srng.binomial(input_shape, p=retain_prob, dtype=input.dtype)

	    return input * mask

import lasagne
import theano
import numpy as np
import theano.tensor as tt

floatX = theano.config.floatX


class DropoutLayer(lasagne.layers.noise.DropoutLayer):
    '''
    Modification to the dropout layer from lasagne, that give control over
    when to sample the masks
    '''
    def __init__(self, incoming, p=0.5, rescale=True, mask=None,
                 n_samples=None, **kwargs):
        super(DropoutLayer, self).__init__(incoming, p, rescale, **kwargs)
        if mask is None:
            if n_samples is not None:
                # we are going to create a shared variable for storing the
                # mask (with the appropriate number of dimensions) and provide
                # an update expression that the user can use later on, to
                # update the mask; i.e. the dropout mask will be fixed unless
                # the user updates it
                name = 'mask' if self.name is None else self.name+'>mask'
                mask_shape = [s if s is not None else n_samples
                              for s in self.input_shape]
                mask = theano.shared(
                    np.random.binomial(1, 1-p, mask_shape).astype(floatX),
                    name=name)
        self.mask = mask
        # initalize dropout mask parameter
        if self.mask is not None:
            self.mask = self.add_param(
                self.mask, self.mask.get_value().shape, self.mask.name,
                trainable=False,  regularizable=False)
        self.mask_updates = None

    def get_output_for(self, input, deterministic=False,
                       fixed_dropout_masks=False, **kwargs):
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
                # the user may update the shared mask value however they want,
                # but here we provide an update expression. note that if the
                # batch size changes, the update will only have an effect at
                # the next call causing a shape mis-match in the elementwise
                # product. To avoid this, the user should update the masks
                # before performing a forward pass on this layer.
                self.mask_updates = mask

                # make sure that we use the local shared variable as the mask
                mask = self.mask

            return input * mask

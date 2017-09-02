import lasagne
import theano
import numpy as np
import theano.tensor as tt
from numbers import Number

from lasagne.random import get_rng
from lasagne import init, nonlinearities
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

floatX = theano.config.floatX


class DropoutLayer(lasagne.layers.noise.DropoutLayer):
    '''
    Modification to the dropout layer from lasagne, that give control over
    when to sample the masks
    '''
    def __init__(self, incoming, p=0.5, rescale=False, mask=None, **kwargs):
        super(DropoutLayer, self).__init__(incoming, p, rescale, **kwargs)
        mask_name = 'mask' if self.name is None else self.name+'>mask'
        if mask is None:
            # we are going to create a shared variable for storing the
            # mask (with the appropriate number of dimensions) and provide
            # an update expression that the user can use later on, to
            # update the mask; i.e. the dropout mask will be fixed unless
            # the user updates it
            mask_shape = [s if s is not None else 2
                          for s in self.input_shape]
            mask = theano.shared(
                np.random.binomial(1, 0.5, mask_shape).astype(floatX),
                name=mask_name)

        # initalize dropout mask parameter
        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask_shape = mask.shape
            elif isinstance(mask, tt.sharedvar.SharedVariable):
                mask_shape = mask.get_value().shape
            else:
                mask_shape = [s if s is not None else 2
                              for s in self.input_shape]

            self.mask = self.add_param(
                mask, mask_shape, mask_name,
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


class GaussianDropoutLayer(lasagne.layers.Layer):
    '''
        Puts a gaussian prior on the weights of the previous layer
    '''
    def __init__(self, incoming, p=lasagne.init.Constant(-10), log_alpha=None,
                 mask=None, n_samples=None, shared_axes=(), **kwargs):
        super(GaussianDropoutLayer, self).__init__(
            incoming, **kwargs)

        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.shared_axes = tuple(shared_axes)

        if log_alpha is None:
            if isinstance(p, Number):
                p = np.atleast_1d(p)
            if callable(p):
                p_shape = self.input_shape[1:]
            else:
                p_shape = p.shape
            p = lasagne.utils.create_param(p, p_shape, name='p')
            p = p.get_value()
            log_alpha = np.log(p/(1-p))

        # add log_alpha as trainable parameter
        if isinstance(log_alpha, Number):
            log_alpha = np.atleast_1d(log_alpha)
        if callable(log_alpha):
            log_alpha_shape = self.input_shape[1:]
        elif isinstance(log_alpha, tt.sharedvar.SharedVariable):
            log_alpha_shape = log_alpha.get_value().shape
        else:
            log_alpha_shape = log_alpha.shape

        self.log_alpha = self.add_param(
            log_alpha, log_alpha_shape, name='log_alpha', regularizable=False)

        # init mask to shape compatible with log_alpha
        mask_shape = [2] + list(self.input_shape[1:])
        # the mask should be drawn from a normal (1, alpha) distribution
        sq_alpha = np.exp(0.5*self.log_alpha.get_value())
        mask = sq_alpha*np.random.normal(1, 1, mask_shape).astype(floatX)

        self.mask = self.add_param(
            mask, mask_shape, name='mask', trainable=False, regularzable=False)
        self.mask_updates = None

    def get_output_for(self, input, deterministic=False,
                       fixed_dropout_masks=False, **kwargs):
        if deterministic:
            return input
        else:
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

            mask = self._srng.normal(
                mask_shape, avg=0, std=1,
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
            sq_alpha = tt.exp(0.5*self.log_alpha)
            return input * (1 + sq_alpha * mask)


class DenseDropoutLayer(lasagne.layers.DenseLayer):
    '''
        Dense layer with dropout regularization
    '''
    def __init__(self, incoming, num_units, W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 num_leading_axes=1, p=0.5, shared_axes=(), noise_samples=None,
                 **kwargs):
        super(DenseDropoutLayer, self).__init__(
            incoming, num_units, W, b, nonlinearity,
            num_leading_axes, **kwargs)

        self.p = p
        self.shared_axes = shared_axes

        # init randon number generator
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))

        # initialize noise samples
        self.noise = self.init_noise(noise_samples)

    def init_noise(self, noise):
        # initalize noise param
        if isinstance(noise, np.ndarray):
            noise_shape = noise.shape
        elif isinstance(noise, tt.sharedvar.SharedVariable):
            noise_shape = noise.get_value().shape
        elif noise is None:
            # init noise samples with appropriate shape
            noise_shape = tuple(s if s is not None else 2
                                for s in self.input_shape)
            noise = theano.shared(np.ones(noise_shape).astype(floatX))

        noise = self.add_param(
            noise, noise_shape, 'noise_samples',
            trainable=False,  regularizable=False)
        self.updates = theano.updates.OrderedUpdates()

        return noise

    def sample_noise(self, input):
        # get noise_shape
        noise_shape = self.input_shape
        if any(s is None for s in noise_shape):
            noise_shape = input.shape

        # respect shared axes
        if self.shared_axes:
            shared_axes = tuple(a if a >= 0 else a + input.ndim
                                for a in self.shared_axes)
            noise_shape = tuple(1 if a in shared_axes else s
                                for a, s in enumerate(noise_shape))

        one = tt.constant(1)
        retain_prob = one - self.p
        noise = self._srng.binomial(noise_shape, p=retain_prob,
                                    dtype=input.dtype)

        if self.shared_axes:
            bcast = tuple(bool(s == 1) for s in noise_shape)
            noise = tt.patternbroadcast(noise, bcast)

        return noise

    def apply_noise(self, input, noise):
        return input * noise

    def get_output_for(self, input, deterministic=False,
                       fixed_noise_samples=False, **kwargs):
        num_leading_axes = self.num_leading_axes
        if num_leading_axes < 0:
            num_leading_axes += input.ndim
        if input.ndim > num_leading_axes + 1:
            # flatten trailing axes (into (n+1)-tensor for num_leading_axes=n)
            input = input.flatten(num_leading_axes + 1)
        if not deterministic:
            # get expression for getting new noise samples
            noise = self.sample_noise(input)

            if fixed_noise_samples:
                # Use the shared noise variable as the dropout mask. The user
                # will take  care of sampling new values using the updates
                # dictionary.

                # store updates so we can control when to get new samples
                self.updates[self.noise] = noise
                noise = self.noise

            input = self.apply_noise(input, noise)

        # apply forward pass
        activation = tt.dot(input, self.W)
        if self.b is not None:
            activation = activation + self.b

        return self.nonlinearity(activation)

    def get_updates(self):
        return self.updates


class DenseGaussianDropoutLayer(DenseDropoutLayer):
    '''
        Dense layer with multiplicative gaussian noise regularization
        as described in
        "Variational Dropout and the local reparametrization trick"
        by Kingma et. al, 2015
    '''
    def __init__(self, incoming, num_units, W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 num_leading_axes=1, p=0.5, alpha=None, shared_axes=(),
                 noise_samples=None, **kwargs):
        super(DenseGaussianDropoutLayer, self).__init__(
            incoming, num_units, W, b, nonlinearity,
            num_leading_axes, p, shared_axes=(), noise_samples=None,
            **kwargs)

        if alpha is None:
            if isinstance(p, Number):
                p = np.atleast_1d(p)
            if callable(p):
                p_shape = self.input_shape[1:]
            else:
                p_shape = p.shape
            p = lasagne.utils.create_param(p, p_shape, name='p')
            p = p.get_value() + 1e-9
            # we will constrain to positive values using the softplus function
            alpha = np.log(np.exp(p/(1-p))-1)

        # add alpha as trainable parameter
        if isinstance(alpha, Number):
            alpha = np.atleast_1d(alpha)
        if callable(alpha):
            alpha_shape = self.input_shape[1:]
        elif isinstance(alpha, tt.sharedvar.SharedVariable):
            alpha_shape = alpha.get_value().shape
        else:
            alpha_shape = alpha.shape

        self.alpha = self.add_param(
            alpha, alpha_shape, name='alpha', regularizable=False)
        
        self.log_alpha = tt.log(tt.nnet.softplus(self.alpha))

    def sample_noise(self, input, mean=0, std=1):
        # get noise_shape
        noise_shape = input.shape

        # respect shared axes
        if self.shared_axes:
            shared_axes = tuple(a if a >= 0 else a + input.ndim
                                for a in self.shared_axes)
            noise_shape = tuple(1 if a in shared_axes else s
                                for a, s in enumerate(noise_shape))

        noise = self._srng.normal(
            noise_shape, avg=mean, std=std, dtype=input.dtype)

        if self.shared_axes:
            bcast = tuple(bool(s == 1) for s in noise_shape)
            noise = tt.patternbroadcast(noise, bcast)

        return noise

    def apply_noise(self, input, noise):
        # scale noise by alpha.
        # alpha is shared across mini batch samples
        sq_alpha = tt.nnet.softplus(self.alpha)**0.5
        return input*(1 + sq_alpha * noise)

    def get_updates(self):
        return self.updates


class DenseAdditiveGaussianDropoutLayer(DenseGaussianDropoutLayer):
    '''
    Dense layer with additive gaussian noise regularization as
    described in:
    "Variational Dropout Sparsifies Deep Neural Networks"
    Molchanov et al, 2017
    '''
    def __init__(self, incoming, num_units, W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 num_leading_axes=1, p=0.5, log_sigma2=None, shared_axes=(),
                 noise_samples=None, **kwargs):
        super(DenseGaussianDropoutLayer, self).__init__(
              incoming, num_units, W, b, nonlinearity,
              num_leading_axes, p, shared_axes=(), noise_samples=None,
              **kwargs)
        self.p = p
        if log_sigma2 is None:
            if isinstance(p, Number):
                p = np.atleast_1d(p)
            if callable(p):
                p_shape = self.input_shape[1:]
            else:
                p_shape = p.shape
            p = lasagne.utils.create_param(p, p_shape, name='p')
            p = p.get_value() + 1e-8
            alpha = p/(1-p)
            log_sigma2 = np.log(alpha*(self.W.get_value()**2))

        # add log_sigma2 as trainable parameter
        if isinstance(log_sigma2, Number):
            log_sigma2 = np.atleast_1d(log_sigma2)
        if callable(log_sigma2):
            log_sigma2_shape = self.input_shape[1:]
        elif isinstance(log_sigma2, tt.sharedvar.SharedVariable):
            log_sigma2_shape = log_sigma2.get_value().shape
        else:
            log_sigma2_shape = log_sigma2.shape
        
        self.log_sigma2 = self.add_param(
            log_sigma2, log_sigma2_shape, name='log_sigma2',
            regularizable=False)
        self.log_alpha = self.log_sigma2 - tt.log(self.W**2)
        s2 = tt.exp(self.log_sigma2)
        alpha = s2/(self.W**2)
        p = alpha/(1+alpha)
        p = p.eval()
        print((p.min(), p.mean(), p.max()))

    def init_noise(self, noise):
        # initalize noise param
        if isinstance(noise, np.ndarray):
            noise_shape = noise.shape
        elif isinstance(noise, tt.sharedvar.SharedVariable):
            noise_shape = noise.get_value().shape
        elif noise is None:
            # init noise samples with appropriate shape
            input_shape = tuple(s if s is not None else 2
                                for s in self.input_shape)
            noise_shape = np.dot(
                np.ones(input_shape), self.W.get_value()).shape
            noise = np.ones(noise_shape).astype(floatX)

        noise = self.add_param(
            noise, noise_shape, 'noise_samples',
            trainable=False,  regularizable=False)
        self.updates = theano.updates.OrderedUpdates()

        return noise

    def get_output_for(self, input, deterministic=False,
                       fixed_noise_samples=False, **kwargs):
        num_leading_axes = self.num_leading_axes
        if num_leading_axes < 0:
            num_leading_axes += input.ndim
        if input.ndim > num_leading_axes + 1:
            # flatten trailing axes (into (n+1)-tensor for num_leading_axes=n)
            input = input.flatten(num_leading_axes + 1)

        if deterministic or self.p == 0:
            activation = tt.dot(input, self.W)
        else:

            m_act = tt.dot(input, self.W)
            S_act = tt.dot(input**2, tt.exp(self.log_sigma2))

            # get expression for getting new noise samples
            noise = self.sample_noise(m_act)

            if fixed_noise_samples:
                # Use the shared noise variable as the dropout mask. The user
                # will take  care of sampling new values using the updates
                # dictionary.

                # store updates so we can control when to get new samples
                self.updates[self.noise] = noise
                noise = self.noise

            activation = m_act + noise*tt.sqrt(S_act)

        # apply forward pass
        if self.b is not None:
            activation = activation + self.b

        return self.nonlinearity(activation)

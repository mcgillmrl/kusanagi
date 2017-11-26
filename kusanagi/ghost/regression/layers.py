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
                                    dtype=floatX)

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
                 num_leading_axes=1, p=0.5, logit_alpha=None, shared_axes=(),
                 noise_samples=None, max_alpha=0.5, **kwargs):
        super(DenseGaussianDropoutLayer, self).__init__(
            incoming, num_units, W, b, nonlinearity,
            num_leading_axes, p, shared_axes=(), noise_samples=None,
            **kwargs)
        self.max_alpha = max_alpha
        self.logit_alpha = logit_alpha
        self.p = p
        self.init_params()

    def init_params(self):
        p = self.p
        logit_alpha = self.logit_alpha
        max_alpha = self.max_alpha

        if logit_alpha is None:
            if isinstance(p, Number):
                p = np.atleast_1d(p)
            if callable(p):
                p_shape = self.input_shape[1:]
            else:
                p_shape = p.shape
            p = lasagne.utils.create_param(p, p_shape, name='p')
            p = p.get_value()
            # we will constrain log_alpha between [-Inf, log(max_alpha))
            alpha = (p/(1-p))
            alpha = np.clip(alpha, 1e-6, max_alpha-1e-6)
            logit_alpha = -np.log(max_alpha/alpha - 1).astype(floatX)

        # add alpha as trainable parameter
        if isinstance(logit_alpha, Number):
            logit_alpha = np.atleast_1d(logit_alpha)
        if callable(logit_alpha):
            logit_alpha_shape = self.input_shape[1:]
        elif isinstance(logit_alpha, tt.sharedvar.SharedVariable):
            logit_alpha_shape = logit_alpha.get_value().shape
        else:
            logit_alpha_shape = logit_alpha.shape

        self.logit_alpha = self.add_param(
            logit_alpha, logit_alpha_shape, name='logit_alpha',
            regularizable=False)
        alpha = self.max_alpha*tt.nnet.sigmoid(self.logit_alpha)
        self.log_alpha = tt.log(alpha)

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
            noise_shape, avg=mean, std=std, dtype=floatX)

        if self.shared_axes:
            bcast = tuple(bool(s == 1) for s in noise_shape)
            noise = tt.patternbroadcast(noise, bcast)

        return noise

    def apply_noise(self, input, noise):
        # scale noise by alpha.
        # alpha is shared across mini batch samples
        sq_alpha = (tt.exp(0.5*self.log_alpha))
        return input*(1 + sq_alpha * noise)


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
        self.log_sigma2 = log_sigma2
        self.init_params()

    def init_params(self):
        # delete tthe logit param from the superclass
        if hasattr(self, 'logit_alpha'):
            if self.logit_alpha in self.params:
                del self.params[self.logit_alpha]
            self.logit_alpha = None

        log_sigma2 = self.log_sigma2
        p = self.p
        if log_sigma2 is None:
            if isinstance(p, Number):
                p = np.atleast_1d(p)
            if callable(p):
                p_shape = self.input_shape[1:]
            else:
                p_shape = p.shape
            p = lasagne.utils.create_param(p, p_shape, name='p')
            p = p.get_value() + 1e-6
            alpha = p/(1-p)
            log_sigma2 = np.log(
                alpha*(self.W.get_value().T**2)).T.astype(floatX)

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
        self.log_alpha = self.log_sigma2 + np.log(1e-6) - tt.log(self.W**2+1e-6)

        s2 = tt.exp(self.log_sigma2)
        a = s2.eval()
        print((a.min(), a.mean(), a.max()))
        a = (self.W.get_value()+1e-6)**2
        print((a.min(), a.mean(), a.max()))
        alpha = s2/(self.W**2)
        a = alpha.eval()
        print((a.min(), a.mean(), a.max()))
        p = alpha/(1+alpha)
        p = p.eval()
        print((p.min(), p.mean(), p.max()))
        print(self.params)

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
            W = self.W + 1e-6
            m_act = tt.dot(input, W)
            S_act = tt.dot(input**2, tt.exp(self.log_sigma2)+1e-6)

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


def phi(x):
    return 0.5*(tt.erfc(-x/tt.sqrt(2)))


def inv_phi(y):
    return -tt.sqrt(2)*tt.erfcinv(2*y)


class DenseLogNormalDropoutLayer(DenseDropoutLayer):
    def __init__(self, incoming, num_units, W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 num_leading_axes=1, logit_posterior_mean=None,
                 logit_posterior_std=None, interval=[-5.0, 5.0],
                 shared_axes=(), noise_samples=None,
                 **kwargs):
        super(DenseLogNormalDropoutLayer, self).__init__(
            incoming, num_units, W, b, nonlinearity,
            num_leading_axes, shared_axes=(), noise_samples=None,
            **kwargs)
        self.logit_posterior_mean = logit_posterior_mean
        self.logit_posterior_std = logit_posterior_std
        self.interval = interval
        self.init_params()

    def init_params(self):
        logit_posterior_mean = self.logit_posterior_mean
        logit_posterior_std = self.logit_posterior_std
        a, b = self.interval
        uniform_std = np.sqrt(((b-a)**2)/12.0)
        s_interval = [1e-4, uniform_std]
        s_min, s_max = np.array(s_interval).astype(floatX).tolist()
        self.s_interval = [s_min, s_max]

        if logit_posterior_mean is None:
            # initialize close to 0 (weights close to 1), but within range (a, b)
            mu0 = max(a + 1e-2*(b-a), 0) + min(b - 1e-2*(b-a), 0)
            logit_mu0 = -np.log((b-a)/(mu0 - a) - 1).astype(floatX)
            logit_posterior_mean = lasagne.init.Constant(logit_mu0)

        if logit_posterior_std is None:
            # set posterior_std close to the minimum
            logit_posterior_std = lasagne.init.Uniform((-3.0, -2.0))

        # add the posterior parameters as trainable parameters
        if isinstance(logit_posterior_mean, Number):
            logit_posterior_mean = np.atleast_1d(logit_posterior_mean)
        if callable(logit_posterior_mean):
            logit_posterior_mean_shape = self.input_shape[1:]
        elif isinstance(logit_posterior_mean, tt.sharedvar.SharedVariable):
            logit_posterior_mean_shape = logit_posterior_mean.get_value().shape
        else:
            logit_posterior_mean_shape = logit_posterior_mean.shape

        self.logit_posterior_mean = self.add_param(
            logit_posterior_mean, logit_posterior_mean_shape, 
            name='logit_posterior_mean', regularizable=False)

        if isinstance(logit_posterior_std, Number):
            logit_posterior_std = np.atleast_1d(logit_posterior_std)
        if callable(logit_posterior_std):
            logit_posterior_std_shape = self.input_shape[1:]
        elif isinstance(logit_posterior_std, tt.sharedvar.SharedVariable):
            logit_posterior_std_shape = logit_posterior_std.get_value().shape
        else:
            logit_posterior_std_shape = logit_posterior_std.shape

        self.logit_posterior_std = self.add_param(
            logit_posterior_std, logit_posterior_std_shape,
            name='logit_posterior_std',
            regularizable=False)

        self.init_intermediate_vars()

    def init_intermediate_vars(self):
        a, b = self.interval
        s_min, s_max = self.s_interval
        sigmoid = tt.nnet.sigmoid

        # posterior params
        self.mu = (b-a)*sigmoid(self.logit_posterior_mean) + a
        self.sigma = (s_max-s_min)*sigmoid(self.logit_posterior_std) + s_min
        print((self.mu.min().eval(), self.mu.max().eval()))
        print((self.sigma.min().eval(), self.sigma.max().eval()))

        # transform noise  to truncated lognormal samples
        self.alpha = (a - self.mu)/self.sigma
        self.beta = (b - self.mu)/self.sigma
        self.phi_alpha = phi(self.alpha)
        self.Z = phi(self.beta) - self.phi_alpha

        # compute SNR
        #Z1 = phi(sigma-alpha) - phi(sigma-beta)
        #Z2 = phi(2*sigma-alpha) - phi(2*sigma-beta)
        #Enoise = (Z1)/tt.sqrt(Z)
        #Varnoise = tt.sqrt(tt.exp(sigma**2)*Z2 - Z1**2)
        #snr = Enoise/Varnoise

    def get_intermediate_outputs(self):
        ''' returns variables that do not depend on the input;
            i.e. are a function of the parameters only. This is done
            so that we can pass these intermediate variable to calls
            to scan (so they're not recomputed at every loop iteration
        '''
        return [self.mu, self.sigma, self.alpha, self.beta, 
                self.phi_alpha, self.Z]

    def sample_noise(self, input, a=1e-5, b=1-1e-5):
        # get noise_shape
        noise_shape = input.shape

        # respect shared axes
        if self.shared_axes:
            shared_axes = tuple(a if a >= 0 else a + input.ndim
                                for a in self.shared_axes)
            noise_shape = tuple(1 if a in shared_axes else s
                                for a, s in enumerate(noise_shape))

        noise = self._srng.uniform(
            noise_shape, low=a, high=b, dtype=floatX)

        if self.shared_axes:
            bcast = tuple(bool(s == 1) for s in noise_shape)
            noise = tt.patternbroadcast(noise, bcast)

        return noise

    def apply_noise(self, input, noise):
        # noise should come from a U[0, 1] distribution
        # get posterior params
        mu = self.mu
        sigma = self.sigma

        # transform noise  to truncated lognormal samples
        p = self.phi_alpha + self.Z*noise
        iphi = inv_phi(p)
        noise = tt.exp(mu + sigma*iphi)

        # only keep neurons with high signal to noise ratio
        return input*noise


class DenseConcreteDropoutLayer(DenseDropoutLayer):
    '''
        Dense layer with concrete binary dropout
        "Concrete Dropout"
        by Gal et. al, 2017
    '''
    def __init__(self, incoming, num_units, W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 num_leading_axes=1, p=0.5, logit_p=None, temp=0.1, shared_axes=(),
                 noise_samples=None, **kwargs):
        super(DenseConcreteDropoutLayer, self).__init__(
            incoming, num_units, W, b, nonlinearity,
            num_leading_axes, p, shared_axes=(), noise_samples=None,
            **kwargs)

        self.temp = temp
        self.logit_p = logit_p
        self.init_params()

    def init_params(self):
        p = self.p
        logit_p = self.logit_p

        if logit_p is None:
            if isinstance(p, Number):
                p = np.atleast_1d(p)
            if callable(p):
                p_shape = self.input_shape[1:]
            else:
                p_shape = p.shape
            p = lasagne.utils.create_param(p, p_shape, name='p')
            p = p.get_value()
            # we will constrain p between [0, 1]
            logit_p = -np.log(1.0/p - 1.0).astype(floatX)

        # add alpha as trainable parameter
        if isinstance(logit_p, Number):
            logit_p = np.atleast_1d(logit_p)
        if callable(logit_p):
            logit_p_shape = self.input_shape[1:]
        elif isinstance(logit_p, tt.sharedvar.SharedVariable):
            logit_p_shape = logit_p.get_value().shape
        else:
            logit_p_shape = logit_p.shape

        self.logit_p = self.add_param(
            logit_p, logit_p_shape, name='logit_p',
            regularizable=False)

        # p is the dropout probability ( 1-p_bernoulli)
        self.p = 1-tt.nnet.sigmoid(self.logit_p)
        eps = np.finfo(np.__dict__[floatX]).eps
        self.logp = tt.log(self.p + eps)
        self.log1mp = tt.log(1.0 - self.p + eps)

    def sample_noise(self, input, a=0, b=1):
        # get noise_shape
        noise_shape = input.shape

        # respect shared axes
        if self.shared_axes:
            shared_axes = tuple(a if a >= 0 else a + input.ndim
                                for a in self.shared_axes)
            noise_shape = tuple(1 if a in shared_axes else s
                                for a, s in enumerate(noise_shape))

        noise = self._srng.uniform(
            noise_shape, low=a, high=b, dtype=floatX)

        if self.shared_axes:
            bcast = tuple(bool(s == 1) for s in noise_shape)
            noise = tt.patternbroadcast(noise, bcast)

        return noise

    def apply_noise(self, input, noise):
        concrete_p = self.logp - self.log1mp + tt.log(noise) - tt.log(1-noise)
        concrete_noise = tt.nnet.sigmoid(concrete_p/self.temp)
        return input * concrete_noise
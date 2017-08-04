import numpy as np
import theano
import theano.tensor as tt

from functools import partial
from kusanagi.ghost.optimizers import ScipyOptimizer
from theano import function as F, shared as S
from theano.tensor.nlinalg import matrix_dot, det
from theano.tensor.slinalg import (solve_lower_triangular,
                                   solve_upper_triangular,
                                   solve, Cholesky)

from . import cov
from . import SNRpenalty
from kusanagi import utils
from kusanagi.ghost.regression import BaseRegressor


class GP(BaseRegressor):
    def __init__(self, X_dataset=None, Y_dataset=None, name='GP', idims=None,
                 odims=None, profile=theano.config.profile,
                 snr_penalty=SNRpenalty.SEard, filename=None, **kwargs):
        # theano options
        self.profile = profile
        self.compile_mode = theano.compile.get_default_mode()

        # GP options
        self.state_changed = True
        self.should_recompile = False
        self.trained = False
        self.snr_penalty = snr_penalty
        self.covs = (cov.SEard, cov.Noise)
        self.uncertain_inputs = True

        # dimension related variables
        self.N = 0
        if X_dataset is None:
            if idims is None:
                raise ValueError('You need to either provide X_dataset (n x idims numpy array) or a\
                                   value for idims')
            self.D = idims
        else:
            self.D = X_dataset.shape[1]

        if Y_dataset is None:
            if odims is None:
                raise ValueError('You need to either provide Y_dataset (n x odims numpy array) or a\
                                   value for odims')
            self.E = odims
        else:
            self.E = Y_dataset.shape[1]

        # symbolic varianbles
        self.loghyp = None
        self.logsn = None
        self.X = None
        self.Y = None
        self.iK = None
        self.L = None
        self.beta = None
        self.nigp = None
        self.Y_var = None
        self.X_cov = None
        self.kernel_func = None
        self.loss_fn = None
        self.dloss_fn = None

        # compiled functions
        self.predict_fn = None
        self.predict_d_fn = None

        # name of this class for printing command line output and saving
        self.name = name
        # filename for saving
        self.filename = filename if filename else '%s_%d_%d_%s_%s' % (
            self.name, self.D, self.E, theano.config.device,
            theano.config.floatX)
        BaseRegressor.__init__(self, name=name, filename=self.filename)
        if filename is not None:
            self.load()

        # optimizer options
        max_evals = kwargs.get('max_evals', 300)
        conv_thr = kwargs.get('conv_thr', 1e-12)
        min_method = kwargs.get('min_method', 'L-BFGS-B')
        self.optimizer = ScipyOptimizer(min_method, max_evals,
                                        conv_thr, name=self.name+'_opt')

        # register theanno functions and shared variables for saving
        self.register_types([tt.sharedvar.SharedVariable])
        # register additional variables for saving
        self.register(['trained'])

        # initialize the class if no pickled version is available
        if X_dataset is not None and Y_dataset is not None:
            utils.print_with_stamp('Initialising new GP regressor', self.name)
            self.set_dataset(X_dataset, Y_dataset)
            utils.print_with_stamp('Done initialising GP regressor', self.name)

        self.ready = False
        self.predict_fn = None

    def load(self, output_folder=None, output_filename=None):
        ''' loads the state from file, and initializes additional variables'''
        # load state
        super(GP, self).load(output_folder, output_filename)

        # initialize missing variables
        if hasattr(self, 'X') and self.X:
            self.N = self.X.get_value(borrow=True).shape[0]
            self.D = self.X.get_value(borrow=True).shape[1]
        if hasattr(self, 'Y') and self.Y:
            self.E = self.Y.get_value(borrow=True).shape[1]
        if hasattr(self, 'loghyp') and self.loghyp:
            self.logsn = self.loghyp[:, -1]

    def set_dataset(self, X_dataset, Y_dataset, X_cov=None, Y_var=None):
        # set dataset
        super(GP, self).set_dataset(X_dataset, Y_dataset)

        # extra operations when setting the dataset (specific to this class)
        if X_cov is not None:
            self.X_cov = X_cov
            self.nigp = S(np.zeros((self.E, self.N)),
                          name="%s>nigp" % (self.name))
        if Y_var is not None:
            if self.Y_var is None:
                self.Y_var = S(Y_var, name='%s>Y_var' % (self.name),
                               borrow=True)
            else:
                self.Y_var.set_value(Y_var, borrow=True)

        if not self.trained:
            # init log hyperparameters and intermediate variables
            self.init_params()

        # we should be saving, since we updated the trianing dataset
        self.state_changed = True
        if self.N > 0:
            self.ready = True

    def append_dataset(self, X_dataset, Y_dataset, X_cov=None, Y_var=None):
        # overrides append_dataset from BaseRegressor
        if self.X is None:
            self.set_dataset(X_dataset, Y_dataset, X_cov, Y_var)
        else:
            X_ = np.vstack((self.X.get_value(),
                            X_dataset.astype(self.X.dtype)))
            Y_ = np.vstack((self.Y.get_value(),
                            Y_dataset.astype(self.Y.dtype)))
            X_cov_ = None
            if X_cov is not None and hasattr(self, 'X_cov') and self.X_cov:
                X_cov_ = np.vstack((self.X_cov,
                                    X_cov.astype(self.X_cov.dtype)))
            Y_var_ = None
            if Y_var is not None and hasattr(self, 'Y_var'):
                Y_var_ = np.vstack((self.Y_var.get_value(),
                                    Y_var.astype(self.Y_var.dtype)))

            self.set_dataset(X_, Y_, X_cov_, Y_var_)

    def init_params(self):
        utils.print_with_stamp('Initialising parameters', self.name)
        idims = self.D
        odims = self.E
        # initialize the loghyperparameters of the gp
        # this code supports squared exponential only, at the moment
        X = self.X.get_value()
        Y = self.Y.get_value()
        loghyp = np.zeros((odims, idims+2))
        loghyp[:, :idims] = X.std(0, ddof=1)
        loghyp[:, idims] = Y.std(0, ddof=1)
        loghyp[:, idims+1] = 0.1*loghyp[:, idims]
        loghyp = np.log(loghyp)

        # set params will either create the loghyp attribute, or update
        # its value
        self.set_params({'loghyp': loghyp})
        # create logsn (used in PILCO)
        if self.logsn is None:
            self.logsn = self.loghyp[:, -1]

    def get_all_shared_vars(self, as_dict=False):
        if as_dict:
            return [(attr_name, self.__dict__[attr_name])
                    for attr_name in list(self.__dict__.keys())
                    if isinstance(self.__dict__[attr_name],
                                  tt.sharedvar.SharedVariable)]
        else:
            return [attr for attr in list(self.__dict__.values())
                    if isinstance(attr, tt.sharedvar.SharedVariable)]

    def nigp_updates(self):
        idims = self.D
        msg = 'Compiling derivative of mean function at training inputs'
        utils.print_with_stamp(msg, self.name)

        # we need to evaluate the derivative of the mean function at the
        # training inputs
        def dM2_f_i(mx, beta, loghyp, X):
            loghyps = (loghyp[:idims+1], loghyp[idims+1])
            kernel_func = partial(cov.Sum, loghyps, self.covs)
            k = kernel_func(mx[None, :], X).flatten()
            mean = k.dot(beta)
            dmean = tt.jacobian(mean.flatten(), mx)
            return tt.square(dmean.flatten())

        def dM2_f(beta, loghyp, X):
            # iterate over training inputs
            dM2_o, updts = theano.scan(fn=dM2_f_i, sequences=[X],
                                       non_sequences=[beta, loghyp, X],
                                       allow_gc=False)
            return dM2_o

        # iterate over output dimensions
        dM2, updts = theano.scan(fn=dM2_f, sequences=[self.beta, self.loghyp],
                                 non_sequences=[self.X], allow_gc=False)

        # update the nigp parameter using the derivative of the mean function
        nigp = ((dM2[:, :, :, None]*self.X_cov[None]).sum(2)*dM2).sum(-1)
        nigp_updts = updts + (self.nigp, nigp)

        return nigp_updts

    def get_loss(self, cache_intermediate=True):
        msg = 'Building full GP loss'
        utils.print_with_stamp(msg, self.name)
        idims = self.D
        N = self.X.shape[0].astype(theano.config.floatX)

        def nlml(Y, loghyp, i, X, EyeN, nigp=None, y_var=None):
            # initialise the (before compilation) kernel function
            loghyps = (loghyp[:idims+1], loghyp[idims+1])
            kernel_func = partial(cov.Sum, loghyps, self.covs)

            # We initialise the kernel matrices (one for each output dimension)
            K = kernel_func(X)
            # add the contribution from the input noise
            if nigp:
                K += tt.diag(nigp[i])
            # add the contribution from the output uncertainty (acts as weight)
            if y_var:
                K += tt.diag(y_var[i])

            # compute chol(K) and (K^-1)dot(y)
            L = Cholesky()(K)
            iK = solve_upper_triangular(L.T, solve_lower_triangular(L, EyeN))
            Yc = solve_lower_triangular(L, Y)
            beta = solve_upper_triangular(L.T, Yc)

            # And finally, the negative log marginal likelihood
            loss = Yc.T.dot(Yc)
            loss += 2*tt.sum(tt.log(tt.ExtractDiag(view=True)(L)))
            loss += N*tt.log(2*np.pi)
            loss *= 0.5

            return loss, iK, L, beta

        nseq = [self.X, tt.eye(self.X.shape[0])]
        if self.nigp:
            nseq.append(self.nigp)
        if self.Y_var:
            nseq.append(self.Y_var.T)

        seq = [self.Y.T, self.loghyp, tt.arange(self.X.shape[0])]
        (loss, iK, L, beta), updts = theano.scan(
            fn=nlml, sequences=seq, non_sequences=nseq, allow_gc=False,
            strict=True, name="%s>logL_scan" % (self.name), return_list=True)

        if cache_intermediate:
            # we are going to save the intermediate results in the following
            # shared variables, so we can use them during prediction without
            # having to recompute them
            N, E = self.N, self.E
            eyeNnp = np.eye(N, dtype=theano.config.floatX)
            if self.iK is None:
                self.iK = S(np.tile(eyeNnp, (E, 1, 1)),
                            name="%s>iK" % (self.name))
            if self.L is None:
                self.L = S(np.tile(np.eye(N), (E, 1, 1)),
                           name="%s>L" % (self.name))
            if self.beta is None:
                self.beta = S(np.ones((E, N), dtype=theano.config.floatX),
                              name="%s>beta" % (self.name))
            updts = [(self.iK, iK), (self.L, L), (self.beta, beta)]
        else:
            # save intermediate graphs (in case we require grads wrt params)
            self.iK, self.L, self.beta = iK, L, beta
            updts = None

        # we add some penalty to avoid having parameters that are too large
        if self.snr_penalty is not None:
            penalty_params = {'log_snr': np.log(1000),
                              'log_ls': np.log(100),
                              'log_std': tt.log(self.X.std(0)*(N/(N-1.0))),
                              'p': 30}
            loss += self.snr_penalty(self.loghyp, **penalty_params)
        inps = []
        self.state_changed = True  # for saving
        return loss.sum(), inps, updts

    def predict_symbolic(self, mx, Sx):
        idims = self.D

        # compute the mean and variance for each output dimension
        def predict_odim(L, beta, loghyp, X, mx):
            loghyps = (loghyp[:idims+1], loghyp[idims+1])
            kernel_func = partial(cov.Sum, loghyps, self.covs)

            k = kernel_func(mx[None, :], X)
            mean = k.dot(beta)
            kc = solve_lower_triangular(L, k.flatten())
            variance = kernel_func(mx[None, :], all_pairs=False) - kc.dot(kc)

            return mean, variance

        (M, S), updts = theano.scan(fn=predict_odim,
                                    sequences=[self.L, self.beta, self.loghyp],
                                    non_sequences=[self.X, mx],
                                    allow_gc=False,
                                    name='%s>predict_scan' % (self.name))

        # reshape output variables
        M = M.flatten()
        S = tt.diag(S.flatten())
        V = tt.zeros((self.D, self.E))

        return M, S, V

    def train(self, optimizer=None, callback=None):
        if optimizer is None:
            optimizer = self.optimizer

        if optimizer.loss_fn is None or self.should_recompile:
            loss, inps, updts = self.get_loss()
            optimizer.set_objective(loss, self.get_params(symbolic=True),
                                    inps, updts)

        if self.X_cov and not hasattr(self, 'nigp_fn'):
            nigp_updts = self.nigp_updates()
            self.nigp_fn = F([], [], updates=updts+nigp_updts,
                             name='%s>dM2' % (self.name),
                             profile=self.profile,
                             mode=self.compile_mode,
                             allow_input_downcast=True)

            def nigp_cb(*args, **kwargs):
                # update the nigp parameter using the derivative of the
                # mean function
                self.nigp_fn()

            if callable(callback):
                # create a new callback that calls nigp_cb and the input
                # callback
                in_cb = callback

                def combined_cb(*args, **kwargs):
                    nigp_cb(*args, **kwargs)
                    in_cb(*args, **kwargs)

                callback = combined_cb
            else:
                callback = nigp_cb

        optimizer.minimize(callback=callback)
        self.trained = True


class GP_UI(GP):
    ''' Gaussian process with uncertain inputs (Deisenroth et al  2009)'''
    def __init__(self, X_dataset=None, Y_dataset=None, name='GP_UI',
                 idims=None, odims=None, profile=False, **kwargs):
        super(GP_UI, self).__init__(
            X_dataset, Y_dataset, name=name, idims=idims, odims=odims,
            profile=profile, **kwargs)

    def predict_symbolic(self, mx, Sx):
        idims = self.D
        odims = self.E

        # centralize inputs
        zeta = self.X - mx

        # initialize some variables
        sf2 = tt.exp(2*self.loghyp[:, idims])
        eyeE = tt.tile(tt.eye(idims), (odims, 1, 1))
        lscales = tt.exp(self.loghyp[:, :idims])
        iL = eyeE/lscales.dimshuffle(0, 1, 'x')

        # predictive mean
        inp = iL.dot(zeta.T).transpose(0, 2, 1)
        iLdotSx = iL.dot(Sx)
        # TODO vectorize this
        B = tt.stack([iLdotSx[i].dot(iL[i])
                      for i in range(odims)]) + tt.eye(idims)
        t = tt.stack([solve(B[i].T, inp[i].T).T for i in range(odims)])
        c = sf2/tt.sqrt(tt.stack([det(B[i]) for i in range(odims)]))
        l = tt.exp(-0.5*tt.sum(inp*t, 2))
        lb = l*self.beta  # E x N dot E x N
        M = tt.sum(lb, 1)*c

        # input output covariance
        tiL = (t[:, :, None, :]*iL[:, None, :, :]).sum(-1)
        # tiL = tt.stack([t[i].dot(iL[i]) for i in range(odims)])
        V = tt.stack([tiL[i].T.dot(lb[i]) for i in range(odims)]).T*c

        # predictive covariance
        logk = 2*self.loghyp[:, None, idims] - 0.5*tt.sum(inp*inp, 2)
        logk_r = logk.dimshuffle(0, 'x', 1)
        logk_c = logk.dimshuffle(0, 1, 'x')
        Lambda = tt.square(iL)
        LL = (Lambda.dimshuffle(0, 'x', 1, 2) + Lambda).transpose(0, 1, 3, 2)
        R = tt.dot(LL, Sx).transpose(0, 1, 3, 2) + tt.eye(idims)
        z_ = Lambda.dot(zeta.T).transpose(0, 2, 1)

        M2 = tt.zeros((odims, odims))
        # initialize indices
        indices = [tt.as_index_variable(idx) for idx in np.triu_indices(odims)]

        def second_moments(i, j, M2, beta, iK, sf2, R, logk_c, logk_r, z_, Sx):
            # This comes from Deisenroth's thesis ( Eqs 2.51- 2.54 )
            Rij = R[i, j]
            n2 = logk_c[i] + logk_r[j]
            n2 += utils.maha(z_[i], -z_[j], 0.5*solve(Rij, Sx))

            Q = tt.exp(n2)/tt.sqrt(det(Rij))

            # Eq 2.55
            m2 = matrix_dot(beta[i], Q, beta[j])

            m2 = theano.ifelse.ifelse(
                tt.eq(i, j), m2 - tt.sum(iK[i]*Q) + sf2[i], m2)
            M2 = tt.set_subtensor(M2[i, j], m2)
            M2 = theano.ifelse.ifelse(
                tt.eq(i, j), M2, tt.set_subtensor(M2[j, i], m2))
            return M2

        nseq = [self.beta, self.iK, sf2, R, logk_c, logk_r, z_, Sx]
        M2_, updts = theano.scan(fn=second_moments,
                                 sequences=indices,
                                 outputs_info=[M2],
                                 non_sequences=nseq,
                                 allow_gc=False,
                                 name="%s>M2_scan" % (self.name))
        M2 = M2_[-1]
        S = M2 - tt.outer(M, M)

        return M, S, V


class RBFGP(GP_UI):
    ''' RBF network (GP with uncertain inputs/deterministic outputs)'''
    def __init__(self, X_dataset=None, Y_dataset=None, idims=None, odims=None,
                 sat_func=None, name='RBFGP', profile=False, **kwargs):
        self.sat_func = sat_func
        if self.sat_func is not None:
            name += '_sat'
        self.loghyp_full = None
        super(RBFGP, self).__init__(X_dataset, Y_dataset, idims=idims,
                                    odims=odims, name=name, profile=profile,
                                    **kwargs)

        # register additional variables for saving
        self.register(['sat_func'])
        self.register(['iK', 'beta', 'L'])

    def predict_symbolic(self, mx, Sx=None):
        idims = self.D
        odims = self.E

        # initialize some variables
        sf2 = tt.exp(2*self.loghyp[:, idims])
        eyeE = tt.tile(tt.eye(idims), (odims, 1, 1))
        lscales = tt.exp(self.loghyp[:, :idims])
        iL = eyeE/lscales.dimshuffle(0, 1, 'x')

        if Sx is None:
            # first check if we received a vector [D] or a matrix [nxD]
            if mx.ndim == 1:
                mx = mx[None, :]
            # centralize inputs
            zeta = self.X[:, None, :] - mx[None, :, :]

            # predictive mean ( we don't need to do the rest )
            inp = (iL[:, None, :, None, :]*zeta[:, None, :, :]).sum(2)
            l = tt.exp(-0.5*tt.sum(inp**2, -1))
            lb = l*self.beta[:, :, None]  # E x N
            M = tt.sum(lb, 1).T*sf2
            # apply saturating function to the output if available
            if self.sat_func is not None:
                # saturate the output
                M = self.sat_func(M)

            return M

        # centralize inputs
        zeta = self.X - mx

        # predictive mean
        inp = iL.dot(zeta.T).transpose(0, 2, 1)
        iLdotSx = iL.dot(Sx)
        B = tt.stack([iLdotSx[i].dot(iL[i])
                      for i in range(odims)]) + tt.eye(idims)
        t = tt.stack([solve(B[i].T, inp[i].T).T for i in range(odims)])
        c = sf2/tt.sqrt(tt.stack([det(B[i]) for i in range(odims)]))
        l = tt.exp(-0.5*tt.sum(inp*t, 2))
        lb = l*self.beta
        M = tt.sum(lb, 1)*c

        # input output covariance
        tiL = tt.stack([t[i].dot(iL[i]) for i in range(odims)])
        V = tt.stack([tiL[i].T.dot(lb[i]) for i in range(odims)]).T*c

        # predictive covariance
        logk = 2*self.loghyp[:, None, idims] - 0.5*tt.sum(inp*inp, 2)
        logk_r = logk.dimshuffle(0, 'x', 1)
        logk_c = logk.dimshuffle(0, 1, 'x')
        Lambda = tt.square(iL)
        LL = (Lambda.dimshuffle(0, 'x', 1, 2) + Lambda).transpose(0, 1, 3, 2)
        R = tt.dot(LL, Sx).transpose(0, 1, 3, 2) + tt.eye(idims)
        z_ = Lambda.dot(zeta.T).transpose(0, 2, 1)

        M2 = tt.zeros((odims, odims))
        # initialize indices
        indices = [tt.as_index_variable(idx) for idx in np.triu_indices(odims)]

        def second_moments(i, j, M2, beta, R, logk_c, logk_r, z_, Sx):
            # This comes from Deisenroth's thesis ( Eqs 2.51- 2.54 )
            Rij = R[i, j]
            n2 = logk_c[i] + logk_r[j]
            n2 += utils.maha(z_[i], -z_[j], 0.5*solve(Rij, Sx))
            Q = tt.exp(n2)/tt.sqrt(det(Rij))

            # Eq 2.55
            m2 = matrix_dot(beta[i], Q, beta[j])

            m2 = theano.ifelse.ifelse(tt.eq(i, j), m2 + 1e-6, m2)
            M2 = tt.set_subtensor(M2[i, j], m2)
            M2 = theano.ifelse.ifelse(
                tt.eq(i, j), M2, tt.set_subtensor(M2[j, i], m2))
            return M2

        nseqs = [self.beta, R, logk_c, logk_r, z_, Sx]
        M2_, updts = theano.scan(fn=second_moments,
                                 sequences=indices,
                                 outputs_info=[M2],
                                 non_sequences=nseqs,
                                 allow_gc=False,
                                 strict=True,
                                 name="%s>M2_scan" % (self.name))
        M2 = M2_[-1]

        S = M2 - tt.outer(M, M)

        # apply saturating function to the output if available
        if self.sat_func is not None:
            # saturate the output
            M, S, U = self.sat_func(M, S)
            # compute the joint input output covariance
            V = V.dot(U)

        return M, S, V

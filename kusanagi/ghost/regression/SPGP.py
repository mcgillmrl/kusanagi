import numpy as np
import theano
import theano.tensor as tt

from functools import partial
from theano import shared as S
from theano.tensor.nlinalg import matrix_dot, det
from theano.tensor.slinalg import (solve_lower_triangular,
                                   solve_upper_triangular,
                                   solve,
                                   Cholesky)

from kusanagi import utils
from kusanagi.ghost.regression import cov
from kusanagi.ghost.regression.GP import GP, GP_UI
from scipy.cluster.vq import kmeans


class SPGP(GP):
    '''Sparse Pseudo Input FITC approximation Snelson and Gharammani 2005'''
    def __init__(self, X_dataset=None, Y_dataset=None, name='SPGP', idims=None,
                 odims=None, profile=False, n_inducing=100, **kwargs):
        self.X_sp = None  # inducing inputs (symbolic variable)
        self.loss_sp_fn = None
        self.dloss_sp_fn = None
        self.beta_sp = None
        self.iKmm = None
        self.iBmm = None
        self.Lmm = None
        self.Amm = None
        self.should_recompile = False
        self.n_inducing = n_inducing
        # intialize parent class params
        GP.__init__(self, X_dataset, Y_dataset, name=name, idims=idims,
                    odims=odims, profile=profile, **kwargs)

    def init_params(self):
        super(SPGP, self).init_params()

    def init_pseudo_inputs(self):
        msg = "Dataset must have more than n_inducing [ %n ] to enable"
        msg += " inference with sparse pseudo inputs"
        assert self.N >= self.n_inducing, msg % (self.n_inducing)
        self.should_recompile = True
        # pick initial cluster centers from dataset
        X = self.X.get_value()
        X_sp_ = utils.kmeanspp(X, self.n_inducing)

        # perform kmeans to get initial cluster centers
        utils.print_with_stamp('Initialising pseudo inputs', self.name)
        X_sp_, dist = kmeans(X, X_sp_, iter=200, thresh=1e-9)
        # initialize symbolic tensor variable if necessary
        # (this will create the self.X_sp atttribute)
        self.set_params({'X_sp': X_sp_})

    def set_dataset(self, X_dataset, Y_dataset):
        # set the dataset on the parent class
        super(SPGP, self).set_dataset(X_dataset, Y_dataset)
        if self.N < self.n_inducing:
            msg = "Dataset must have more than n_inducing [ %n ] to enable"
            msg + " inference with sparse pseudo inputs"
            utils.print_with_stamp(msg, self.name)
            self.X_sp = None
            self.loss_sp_fn = None
            self.dloss_sp_fn = None
            self.beta_sp = None
            self.Lmm = None
            self.Amm = None
            self.should_recompile = False

        if self.N >= self.n_inducing and self.X_sp is None:
            msg = 'Dataset is large enough for using pseudo inputs. You should'
            msg += ' reinitiialise the training loss function and predictions.'
            utils.print_with_stamp(msg, self.name)
            # init the shared variable for the pseudo inputs
            self.init_pseudo_inputs()
            self.should_recompile = True

    def get_loss(self, cache_intermediate=True):
        if self.N < self.n_inducing:
            # initialize the training loss function of the GP class
            return super(SPGP, self).get_loss(cache_intermediate)
        else:
            utils.print_with_stamp('Building FITC loss', self.name)
            self.should_recompile = False
            odims = self.E
            idims = self.D
            N = self.X.shape[0].astype(theano.config.floatX)

            # initialize the training loss function of the sparse FITC
            # approximation
            def nlml(Y, hyp, X, X_sp, EyeM):
                # TODO allow for different pseudo inputs for each dimension
                # initialise the (before compilation) kernel function
                hyps = [hyp[:idims+1], hyp[idims+1]]
                kernel_func = partial(cov.Sum, hyps, self.covs)

                sf2 = tt.exp(2*hyp[idims])
                sn2 = tt.exp(2*hyp[idims+1])
                N = X.shape[0].astype(theano.config.floatX)

                ridge = 1e-6
                Kmm = kernel_func(X_sp) + ridge*EyeM
                Kmn = kernel_func(X_sp, X)
                Lmm = Cholesky()(Kmm)
                iKmm = solve_upper_triangular(
                    Lmm.T, solve_lower_triangular(Lmm, EyeM))
                Lmn = solve_lower_triangular(Lmm, Kmn)
                diagQnn = (Lmn**2).sum(0)

                # Gamma = diag(Knn - Qnn) + sn2*I
                Gamma = sf2 + sn2 - diagQnn
                Gamma_inv = 1.0/Gamma

                # these operations are done to avoid inverting
                # K_sp = (Qnn+Gamma)
                sqrtGamma_inv = tt.sqrt(Gamma_inv)
                Lmn_ = Lmn*sqrtGamma_inv                      # Kmn_*Gamma^-.5
                Yi = Y*(sqrtGamma_inv)                        # Gamma^-.5* Y
                Bmm = tt.eye(Kmm.shape[0]) + (Lmn_).dot(Lmn_.T)
                Amm = Cholesky()(Bmm)
                LAmm = Lmm.dot(Amm)
                Lmn_dotYi = Lmn_.dot(Yi)
                rhs = tt.concatenate([EyeM, Lmn_dotYi[:, None]], axis=1)
                sol = solve_upper_triangular(
                    LAmm.T, solve_lower_triangular(LAmm, rhs))
                iBmm = sol[:, :-1]
                beta_sp = sol[:, -1]

                log_det_K_sp = tt.sum(tt.log(Gamma))
                log_det_K_sp += 2*tt.sum(tt.log(tt.diag(Amm)))

                loss_sp = Yi.dot(Yi) - Lmn_dotYi.dot(beta_sp)
                loss_sp += log_det_K_sp + N*np.log(2*np.pi)
                loss_sp *= 0.5

                return loss_sp, iKmm, Lmm, Amm, iBmm, beta_sp

            r_outs, updts = theano.scan(
                fn=nlml, sequences=[self.Y.T, self.hyp],
                non_sequences=[self.X, self.X_sp, tt.eye(self.X_sp.shape[0])],
                allow_gc=False, return_list=True)
            (loss_sp, iKmm, Lmm, Amm, iBmm, beta_sp) = r_outs

            if cache_intermediate:
                # we are going to save the intermediate results in the
                # following shared variables,
                # so we can use them during prediction without having to
                # recompute them
                # initialize shared variables
                kk = self.n_inducing
                self.iKmm = S(np.tile(np.eye(kk), (odims, 1, 1)),
                              name="%s>iKmm" % (self.name))
                self.Lmm = S(np.tile(np.eye(kk), (odims, 1, 1)),
                             name="%s>Lmm" % (self.name))
                self.Amm = S(np.tile(np.eye(kk), (odims, 1, 1)),
                             name="%s>Amm" % (self.name))
                self.iBmm = S(np.tile(np.eye(kk), (odims, 1, 1)),
                              name="%s>iBmm" % (self.name))
                self.beta_sp = S(np.ones((self.E, kk)),
                                 name="%s>beta_sp" % (self.name))
                updts = [(self.iKmm, iKmm), (self.Lmm, Lmm), (self.Amm, Amm),
                         (self.iBmm, iBmm), (self.beta_sp, beta_sp)]
            else:
                self.iKmm, self.Lmm, self.Amm, self.iBmm, self.beta_sp = iKmm, Lmm, Amm, iBmm, beta_sp
                updts = None

            # we add some penalty to avoid having parameters that are too large
            if self.snr_penalty is not None:
                penalty_params = {'log_snr': np.log(1000),
                                  'log_ls': np.log(100),
                                  'log_std': tt.log(self.X_sp.std(0)*(N/(N-1.0))),
                                  'p': 30}
                loss_sp += self.snr_penalty(self.hyp, **penalty_params)

            inps = []
            self.state_changed = True  # for saving
            return loss_sp.sum(), inps, updts

    def predict_symbolic(self, mx, Sx):
        if self.N < self.n_inducing:
            # stick with the full GP
            return super(SPGP, self).predict_symbolic(mx, Sx)

        idims = self.D
        odims = self.E

        # compute the mean and variance for each output dimension
        def predict_odim(Lmm, Amm, beta_sp, hyp, X_sp, mx):
            hyps = (hyp[:idims+1], hyp[idims+1])
            kernel_func = partial(cov.Sum, hyps, self.covs)

            k = kernel_func(mx[None, :], X_sp).flatten()
            mean = k.dot(beta_sp)
            kL = solve_lower_triangular(Lmm, k)
            kA = solve_lower_triangular(Amm, Lmm.T.dot(k))
            variance = kernel_func(mx[None, :], all_pairs=False)
            variance += -(kL.dot(kL) + kA.dot(kA))
            variance = tt.largest(variance, 0.0) + 1e-3

            return mean, variance
        seq = [self.Lmm, self.Amm, self.beta_sp, self.hyp]
        nseq = [self.X_sp, mx]
        (M, S), updts = theano.scan(
            fn=predict_odim, sequences=seq, non_sequences=nseq,allow_gc=False)

        # reshape output variables
        M = M.flatten()
        S = tt.diag(S.flatten())
        V = tt.zeros((idims, odims))

        return M, S, V

    def train(self):
        # if dataset is big enough, recompile optimizer
        if self.should_recompile:
            self.optimizer.loss_fn = None
            self.optimizer.grads_fn = None
        super(SPGP, self).train()


class SPGP_UI(SPGP, GP_UI):
    def __init__(self, X_dataset=None, Y_dataset=None, name='SPGP_UI',
                 idims=None, odims=None, profile=False, n_inducing=100,
                 **kwargs):
        SPGP.__init__(self, X_dataset, Y_dataset, name=name, idims=idims,
                      odims=odims, profile=profile, n_inducing=n_inducing,
                      **kwargs)

    def predict_symbolic(self, mx, Sx):
        if self.N < self.n_inducing:
            # stick with the full GP
            return GP_UI.predict_symbolic(self, mx, Sx)

        idims = self.D
        odims = self.E

        # centralize inputs
        zeta = self.X_sp - mx

        # initialize some variables
        sf2 = tt.exp(2*self.hyp[:, idims])
        eyeE = tt.tile(tt.eye(idims), (odims, 1, 1))
        lscales = tt.exp(self.hyp[:, :idims])
        iL = eyeE/lscales.dimshuffle(0, 1, 'x')

        # predictive mean
        inp = iL.dot(zeta.T).transpose(0, 2, 1)
        iLdotSx = iL.dot(Sx)
        B = (iLdotSx[:, :, None, :]*iL[:, None, :, :]).sum(-1) + tt.eye(idims)
        t = tt.stack([solve(B[i].T, inp[i].T).T for i in range(odims)])
        c = sf2/tt.sqrt(tt.stack([det(B[i]) for i in range(odims)]))
        l = tt.exp(-0.5*tt.sum(inp*t, 2))
        lb = l*self.beta_sp
        M = tt.sum(lb, 1)*c

        # input output covariance
        tiL = tt.stack([t[i].dot(iL[i]) for i in range(odims)])
        V = tt.stack([tiL[i].T.dot(lb[i]) for i in range(odims)]).T*c

        # predictive covariance
        logk = 2*self.hyp[:, None, idims] - 0.5*tt.sum(inp*inp, 2)
        logk_r = logk.dimshuffle(0, 'x', 1)
        logk_c = logk.dimshuffle(0, 1, 'x')
        Lambda = tt.square(iL)
        LL = (Lambda.dimshuffle(0, 'x', 1, 2) + Lambda).transpose(0, 1, 3, 2)
        R = tt.dot(LL, Sx.T).transpose(0, 1, 3, 2) + tt.eye(idims)
        z_ = Lambda.dot(zeta.T).transpose(0, 2, 1)

        M2 = tt.zeros((odims, odims))

        # initialize indices
        triu_indices = np.triu_indices(odims)
        indices = [tt.as_index_variable(idx) for idx in triu_indices]

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
                tt.eq(i, j), M2 + 1e-6, tt.set_subtensor(M2[j, i], m2))
            return M2

        nseq = [self.beta_sp, (self.iKmm - self.iBmm), sf2,
                R, logk_c, logk_r, z_, Sx]
        M2_, updts = theano.scan(
            fn=second_moments, sequences=indices, outputs_info=[M2],
            non_sequences=nseq, allow_gc=False)
        M2 = M2_[-1]
        S = M2 - tt.outer(M, M)

        return M, S, V

import numpy as np
import theano
import theano.tensor as tt

from theano import shared as S
from theano.tensor.nlinalg import matrix_dot
from theano.tensor.slinalg import (solve_lower_triangular,
                                   solve_upper_triangular,
                                   cholesky)

from kusanagi import utils
from kusanagi.ghost.regression.GP import GP, GP_UI
floatX = theano.config.floatX


class SSGP(GP):
    ''' Sparse Spectrum Gaussian Process Regression Lazaro-Gredilla
    et al 2010'''
    def __init__(self, X_dataset=None, Y_dataset=None, name='SSGP', idims=None,
                 odims=None, n_inducing=100, **kwargs):
        self.w = None
        self.sr = None
        self.Lmm = None
        self.iA = None
        self.beta_ss = None
        self.loss_ss_fn = None
        self.dloss_ss_fn = None
        self.n_inducing = n_inducing
        GP.__init__(self, X_dataset, Y_dataset,
                    name=name, idims=idims, odims=odims,
                    **kwargs)

    def init_params(self):
        super(SSGP, self).init_params()
        # sample initial unscaled spectrum points
        self.set_ss_samples()

    def set_ss_samples(self, w=None):
        idims = self.D
        odims = self.E
        if w is None:
            w = np.random.randn(self.n_inducing, odims, idims)
            w = w.astype(floatX)
        else:
            w = w.reshape((self.n_inducing, odims, idims))
        self.set_params({'w': w})
        if self.sr is None:
            self.sr = self.w/(self.hyp[:, :idims])
            self.sr = self.sr.transpose(1, 0, 2)

    def get_loss(self, unroll_scan=False, cache_intermediate=True):
        utils.print_with_stamp('Building Sparse Spectrum loss', self.name)
        idims = self.D

        if self.sr is None:
            self.sr = self.w/(self.hyp[:, :idims])
            self.sr = self.sr.transpose(1, 0, 2)

        # init variables
        N = self.X.shape[0].astype(floatX)
        M = self.sr.shape[1].astype(floatX)
        Mi = 2*self.sr.shape[1]
        EyeM = tt.eye(Mi)
        sf2 = self.hyp[:, idims]**2
        sf2M = (sf2/M).dimshuffle(0, 'x', 'x')
        sn2 = (self.hyp[:, idims+1]**2).dimshuffle(0, 'x', 'x')
        srdotX = self.sr.dot(self.X.T)

        phi_f = tt.concatenate([tt.sin(srdotX), tt.cos(srdotX)], axis=1)
        Phi_f = tt.batched_dot(phi_f, phi_f.transpose(0, 2, 1))
        A = sf2M*Phi_f
        A += (sn2 + 1e-6)*EyeM
        phi_f_dotY = tt.batched_dot(phi_f, self.Y.T)

        def nlml(A, phidotY, EyeM):
            Lmm = cholesky(A)
            rhs = tt.concatenate([EyeM, phidotY[:, None]], axis=1)
            sol = solve_upper_triangular(
                Lmm.T, solve_lower_triangular(Lmm, rhs))
            iA = sol[:, :-1]
            beta_ss = sol[:, -1]

            return iA, Lmm, beta_ss

        seq = [A, phi_f_dotY]
        nseq = [EyeM]

        if unroll_scan:
            from lasagne.utils import unroll_scan
            [iA, Lmm, beta_ss] = unroll_scan(nlml, seq, [], nseq, self.E)
            updts = {}
        else:
            (iA, Lmm, beta_ss), updts = theano.scan(
                fn=nlml, sequences=seq, non_sequences=nseq,
                allow_gc=False, return_list=True,
                name='%s>logL_ss' % (self.name))

        # scale beta_ss
        beta_ss *= sf2M[:, :, 0]

        # And finally, the negative log marginal likelihood
        YdotY = tt.sum(self.Y**2, 0)
        Ydotphidotbeta = tt.sum(phi_f_dotY*beta_ss, -1)
        loss_ss = 0.5*(YdotY - Ydotphidotbeta)/sn2
        idx = [theano.tensor.arange(Lmm.shape[i]) for i in [1, 2]]
        loss_ss += tt.sum(tt.log(Lmm[:, idx[0], idx[1]]), 1)
        loss_ss += (0.5*N - M)*tt.log(sn2)
        loss_ss += 0.5*N*np.log(2*np.pi, dtype=floatX)

        if cache_intermediate:
            # we are going to save the intermediate results in the following
            # shared variables, so we can use them during prediction without
            # having to recompute them
            kk = 2*self.n_inducing
            N, E = self.N, self.E
            if type(self.iA) is not tt.sharedvar.SharedVariable:
                self.iA = S(np.tile(np.eye(kk, dtype=floatX), (E, 1, 1)),
                            name="%s>iA" % (self.name))
            if type(self.Lmm) is not tt.sharedvar.SharedVariable:
                self.Lmm = S(np.tile(np.eye(kk, dtype=floatX), (E, 1, 1)),
                             name="%s>Lmm" % (self.name))
            if type(self.beta_ss) is not tt.sharedvar.SharedVariable:
                self.beta_ss = S(np.ones((E, kk), dtype=floatX),
                                 name="%s>beta_ss" % (self.name))
            updts = [(self.iA, iA), (self.Lmm, Lmm), (self.beta_ss, beta_ss)]
        else:
            self.iA, self.Lmm, self.beta_ss = iA, Lmm, beta_ss
            updts = None

        # we add some penalty to avoid having parameters that are too large
        if self.snr_penalty is not None:
            penalty_params = {'log_snr': np.log(1000, dtype=floatX),
                              'log_ls': np.log(100, dtype=floatX),
                              'log_std': tt.log(self.X.std(0)*(N/(N-1.0))),
                              'p': 30}
            loss_ss += self.snr_penalty(tt.log(self.hyp), **penalty_params)

        # add a penalty for high frequencies
        freq_penalty = tt.square(self.w).sum(-1).mean(0)
        loss_ss = loss_ss + freq_penalty

        inps = []
        self.state_changed = True  # for saving
        return loss_ss.sum(), inps, updts

    def pretrain_full(self):
        if not hasattr(self, 'full_optimizer'):
            import copy
            self.full_optimizer = copy.copy(self.optimizer)
            self.full_optimizer.name = self.name+'_fullopt'
            self.full_optimizer.max_evals = self.optimizer.max_evals
            self.full_optimizer.loss_fn = None

        if self.full_optimizer.loss_fn is None or self.should_recompile:
            loss, inps, updts = GP.get_loss(self)
            self.full_optimizer.set_objective(
                loss, self.get_params(symbolic=True)[:-1], inps, updts)

        # train the full GP ( if dataset too large, take a random subsample)
        X_full = None
        Y_full = None
        n_subsample = 2048
        X = self.X.get_value()
        if X.shape[0] > n_subsample:
            msg = 'Training full gp with random subsample of size %d'
            utils.print_with_stamp(msg % (n_subsample),
                                   self.name)
            idx = np.arange(X.shape[0])
            np.random.shuffle(idx)
            idx = idx[:n_subsample]
            X_full = X
            Y_full = self.Y.get_value()
            self.set_dataset(X_full[idx], Y_full[idx])

        super(SSGP, self).train(self.full_optimizer)

        if X_full is not None:
            # restore full dataset for SSGP training
            utils.print_with_stamp('Restoring full dataset', self.name)
            self.set_dataset(X_full, Y_full)

    def resample_ss(self, iters=100):
        self.set_ss_samples()
        if self.optimizer.loss_fn is not None:
            loss_fn = self.optimizer.loss_fn
            loss = loss_fn()
            best_w = self.w.get_value()
            # try a couple spectrum samples and pick the one with the
            # lowest loss
            # TODO do this test per output dimension
            for i in range(iters):
                self.set_ss_samples()
                loss_i = loss_fn()
                if loss_i < loss:
                    loss = loss_i
                    best_w = self.w.get_value()
            self.set_ss_samples(best_w)

    def train(self, pretrain_full=False):
        if pretrain_full:
            self.pretrain_full()
        self.resample_ss(100)
        super(SSGP, self).train()

    def predict(self, mx, Sx):
        odims = self.E
        idims = self.D

        # compute the mean and variance for each output dimension
        mean = [[]]*odims
        variance = [[]]*odims
        for i in range(odims):
            sr = self.sr[i]
            M = sr.shape[0].astype(floatX)
            sf2 = self.hyp[i, idims]**2
            sn2 = self.hyp[i, idims+1]**2
            # sr.T.dot(x) for all sr and X. size n_inducing x N
            srdotX = sr.dot(mx)
            # convert to sin cos
            phi_x = tt.concatenate([tt.sin(srdotX), tt.cos(srdotX)])

            mean[i] = phi_x.T.dot(self.beta_ss[i])
            phi_x_L = solve_lower_triangular(self.Lmm[i], phi_x)
            variance[i] = sn2*(1 + (sf2/M)*phi_x_L.dot(phi_x_L)) + 1e-6

        # reshape output variables
        M = tt.stack(mean).T.flatten()
        S = tt.diag(tt.stack(variance).T.flatten())
        V = tt.zeros((self.D, self.E))

        return M, S, V


class SSGP_UI(SSGP, GP_UI):
    ''' Sparse Spectrum Gaussian Process Regression with Uncertain Inputs.
    '''
    def __init__(self, X_dataset=None, Y_dataset=None, name='SSGP_UI',
                 idims=None, odims=None, n_inducing=100,
                 **kwargs):
        SSGP.__init__(self, X_dataset, Y_dataset, name=name, idims=idims,
                      odims=odims, n_inducing=n_inducing,
                      **kwargs)

    def predict(self, mx, Sx, unroll_scan=False):
        idims = self.D
        odims = self.E

        Ms = self.sr.shape[1]
        sf2M = (self.hyp[:, idims]**2)/tt.cast(Ms, floatX)
        sn2 = self.hyp[:, idims+1]**2

        # TODO this should just fallback to the method from the SSGP class
        if Sx is None:
            # first check if we received a vector [D] or a matrix [nxD]
            if mx.ndim == 1:
                mx = mx[None, :]

            srdotx = self.sr.dot(self.X.T).transpose(0,2,1)
            phi_x = tt.concatenate([tt.sin(srdotx), tt.cos(srdotx)], 2)
            M = (phi_x*self.beta_ss[:, None, :]).sum(-1)
            phi_x_L = tt.stack([
                solve_lower_triangular(self.Lmm[i], phi_x[i].T)
                for i in range(odims)])
            S = sn2[:, None]*(1 + (sf2M[:, None])*(phi_x_L**2).sum(-2)) + 1e-6

            return M, S

        # precompute some variables
        srdotx = self.sr.dot(mx)
        srdotSx = self.sr.dot(Sx)
        srdotSxdotsr = tt.sum(srdotSx*self.sr, 2)
        e = tt.exp(-0.5*srdotSxdotsr)
        cos_srdotx = tt.cos(srdotx)
        sin_srdotx = tt.sin(srdotx)
        cos_srdotx_e = cos_srdotx*e
        sin_srdotx_e = sin_srdotx*e

        # compute the mean vector
        mphi = tt.horizontal_stack(sin_srdotx_e, cos_srdotx_e)  # E x 2*Ms
        M = tt.sum(mphi*self.beta_ss, 1)

        # input output covariance
        mx_c = mx.dimshuffle(0, 'x')
        sin_srdotx_e_r = sin_srdotx_e.dimshuffle(0, 'x', 1)
        cos_srdotx_e_r = cos_srdotx_e.dimshuffle(0, 'x', 1)
        srdotSx_tr = srdotSx.transpose(0, 2, 1)
        c = tt.concatenate([mx_c*sin_srdotx_e_r + srdotSx_tr*cos_srdotx_e_r,
                            mx_c*cos_srdotx_e_r - srdotSx_tr*sin_srdotx_e_r],
                           axis=2)  # E x D x 2*Ms
        beta_ss_r = self.beta_ss.dimshuffle(0, 'x', 1)

        # input output covariance (notice this is not premultiplied by the
        # input covariance inverse)
        V = tt.sum(c*beta_ss_r, 2).T - tt.outer(mx, M)

        srdotSxdotsr_c = srdotSxdotsr.dimshuffle(0, 1, 'x')
        srdotSxdotsr_r = srdotSxdotsr.dimshuffle(0, 'x', 1)
        M2 = tt.zeros((odims, odims))

        # initialize indices
        triu_indices = np.triu_indices(odims)
        indices = [tt.as_index_variable(idx) for idx in triu_indices]

        def second_moments(i, j, M2, beta, iA, sn2, sf2M, sr, srdotSx,
                           srdotSxdotsr_c, srdotSxdotsr_r,
                           sin_srdotx, cos_srdotx, *args):
            # compute the second moments of the spectrum feature vectors
            siSxsj = srdotSx[i].dot(sr[j].T)  # Ms x Ms
            sijSxsij = -0.5*(srdotSxdotsr_c[i] + srdotSxdotsr_r[j])
            em = tt.exp(sijSxsij+siSxsj)      # MsxMs
            ep = tt.exp(sijSxsij-siSxsj)     # MsxMs
            si = sin_srdotx[i]       # Msx1
            ci = cos_srdotx[i]       # Msx1
            sj = sin_srdotx[j]       # Msx1
            cj = cos_srdotx[j]       # Msx1
            sicj = tt.outer(si, cj)  # MsxMs
            cisj = tt.outer(ci, sj)  # MsxMs
            sisj = tt.outer(si, sj)  # MsxMs
            cicj = tt.outer(ci, cj)  # MsxMs
            sm = (sicj-cisj)*em
            sp = (sicj+cisj)*ep
            cm = (sisj+cicj)*em
            cp = (cicj-sisj)*ep

            # Populate the second moment matrix of the feature vector
            Q_up = tt.concatenate([cm-cp, sm+sp], axis=1)
            Q_lo = tt.concatenate([sp-sm, cm+cp], axis=1)
            Q = tt.concatenate([Q_up, Q_lo], axis=0)

            # Compute the second moment of the output
            m2 = 0.5*matrix_dot(beta[i], Q, beta[j].T)

            m2 = theano.ifelse.ifelse(
                tt.eq(i, j),
                m2 + sn2[i]*(1.0 + sf2M[i]*tt.sum(self.iA[i]*Q)) + 1e-6,
                m2)
            M2 = tt.set_subtensor(M2[i, j], m2)
            return M2

        nseq = [self.beta_ss, self.iA, sn2, sf2M, self.sr, srdotSx,
                srdotSxdotsr_c, srdotSxdotsr_r, sin_srdotx, cos_srdotx,
                self.Lmm]

        if unroll_scan:
            from lasagne.utils import unroll_scan
            [M2_] = unroll_scan(second_moments, indices,
                                [M2], nseq, len(triu_indices[0]))
            updts = {}
        else:
            M2_, updts = theano.scan(fn=second_moments,
                                     sequences=indices,
                                     outputs_info=[M2],
                                     non_sequences=nseq,
                                     allow_gc=False,
                                     name="%s>M2_scan" % (self.name))

        M2 = M2_[-1]
        M2 = M2 + tt.triu(M2, k=1).T
        S = M2 - tt.outer(M, M)

        return M, S, V

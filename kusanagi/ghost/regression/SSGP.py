from GP import *

class SSGP(GP):
    ''' Sparse Spectral Gaussian Process Regression LAzaro-GRedilla et al 2010'''
    def __init__(self, X_dataset=None, Y_dataset=None, name='SSGP', idims=None, odims=None, profile=False, n_inducing=100,  uncertain_inputs=False, **kwargs):
        self.w = None
        self.sr = None
        self.Lmm = None
        self.iA = None
        self.beta_ss = None
        self.loss_ss_fn = None
        self.dloss_ss_fn = None
        self.n_inducing = n_inducing
        GP.__init__(self,X_dataset,Y_dataset,name=name,idims=idims,odims=odims,profile=profile,uncertain_inputs=uncertain_inputs, **kwargs)
    
    def init_loss(self,cache_vars=True):
        super(SSGP, self).init_loss()
        utils.print_with_stamp('Initialising expression graph for sparse spectral training loss function',self.name)
        idims = self.D
        odims = self.E

        if self.iA is None:
            self.iA = S(np.zeros((self.E,2*self.n_inducing,2*self.n_inducing),dtype='float64'), name="%s>iA"%(self.name))
        if self.Lmm is None:
            self.Lmm = S(np.zeros((self.E,2*self.n_inducing,2*self.n_inducing),dtype='float64'), name="%s>Lmm"%(self.name))
        if self.beta_ss is None:
            self.beta_ss = S(np.zeros((self.E,2*self.n_inducing),dtype='float64'), name="%s>beta_ss"%(self.name))
        
        # sample initial unscaled spectral points
        self.set_spectral_samples()

        #init variables
        N = self.X.shape[0].astype('float64')
        M = self.sr.shape[1].astype('float64')
        Mi = 2*self.sr.shape[1]
        sf2 = tt.exp(2*self.loghyp[:,idims])
        sf2M = sf2/M
        sn2 = tt.exp(2*self.loghyp[:,idims+1])
        srdotX = self.sr.dot(self.X.T)
        phi_f = tt.concatenate( [tt.sin(srdotX), tt.cos(srdotX)], axis=1 ).astype('float64') # E x 2*n_inducing x N
        
        # TODO vectorize these ops
        def log_marginal_likelihood(sf2M, sn2, phi_f, Y, EyeM):
            phi_f.ndim
            A = sf2M*phi_f.dot(phi_f.T) + sn2*EyeM
            Lmm = cholesky(A)
            iA = solve_upper_triangular(Lmm.T, solve_lower_triangular(Lmm,EyeM))
            Yc = solve_lower_triangular(Lmm,(phi_f.dot(Y)))
            beta_ss = sf2M*solve_upper_triangular(Lmm.T,Yc)

            loss_ss = 0.5*( Y.dot(Y) - sf2M*Yc.dot(Yc) )/sn2 + tt.sum(tt.log(tt.diag(Lmm))) + (0.5*N - M)*tt.log(sn2) + 0.5*N*np.log(2*np.pi)
            
            return loss_ss,iA,Lmm,beta_ss
        
        (loss_ss,iA,Lmm,beta_ss),updts = theano.scan(fn=log_marginal_likelihood, sequences=[sf2M,sn2,phi_f,self.Y.T], non_sequences=[tt.eye(Mi)], allow_gc=False,name='%s>logL_ss'%(self.name))
        
        iA = tt.unbroadcast(iA,0) if iA.broadcastable[0] else iA
        Lmm = tt.unbroadcast(Lmm,0) if Lmm.broadcastable[0] else Lmm
        beta_ss = tt.unbroadcast(beta_ss,0) if beta_ss.broadcastable[0] else beta_ss

        if cache_vars:
            # we are going to save the intermediate results in the following shared variables, so we can use them during prediction without having to recompute them
            updts = [(self.iA,iA),(self.Lmm,Lmm),(self.beta_ss,beta_ss)]
        else:
            self.iA = iA 
            self.Lmm = Lmm 
            self.beta_ss = beta_ss
            updts=None

        if self.snr_penalty is not None:
            penalty_params = {'log_snr': np.log(1000), 'log_ls': np.log(100), 'log_std': tt.log(self.X.std(0)*(N/(N-1.0))), 'p': 30}
            loss_ss += self.snr_penalty(self.loghyp)

        # Compute the gradients for the sum of loss for all output dimensions
        dloss_ss = tt.grad(loss_ss.sum(),[self.loghyp,self.w])

        utils.print_with_stamp('Compiling sparse spectral training loss function',self.name)
        self.loss_ss_fn = F((),loss_ss,name='%s>loss_ss'%(self.name), profile=self.profile, mode=self.compile_mode, allow_input_downcast=True, updates=updts)
        utils.print_with_stamp('Compiling gradient of sparse spectral training loss function',self.name)
        self.dloss_ss_fn = F((),(loss_ss,dloss_ss[0],dloss_ss[1]),name='%s>dloss_ss'%(self.name), profile=self.profile, mode=self.compile_mode, allow_input_downcast=True, updates=updts)

    def set_spectral_samples(self,w=None):
        idims = self.D
        odims = self.E
        if w is None:
            w = np.random.randn(self.n_inducing,odims,idims).astype('float64')
        else:
            w = w.reshape((self.n_inducing,odims,idims)).astype('float64')
    
        self.set_params({'w': w})
        
        if self.sr is None:
            self.sr = (self.w*tt.exp(-self.loghyp[:,:idims])).transpose(1,0,2)

    def loss_ss(self, params, parameter_shapes):
        loghyp,w = utils.unwrap_params(params,parameter_shapes)
        self.set_params({'loghyp': loghyp, 'w': w})
        loss,dloss_lh,dloss_sr = self.dloss_ss_fn()
        loss = np.array(loss)
        dloss_lh = np.array(dloss_lh)
        dloss_sr = np.array(dloss_sr)
        loss = loss.sum()
        dloss = utils.wrap_params([dloss_lh,dloss_sr])
        # on a 64bit system, scipy optimize complains if we pass a 32 bit float
        res = (loss.astype(np.float64), dloss.astype(np.float64))
        utils.print_with_stamp('%s'%(str(res[0])),self.name,True)
        return res

    def train(self, pretrain_full=True):
        if pretrain_full:
            # train the full GP ( if dataset too large, take a random subsample)
            X_full = None
            Y_full = None
            n_subsample = 2048
            X = self.X.get_value()
            if X.shape[0] > n_subsample:
                utils.print_with_stamp('Training full gp with random subsample of size %d'%(n_subsample),self.name)
                idx = np.arange(X.shape[0]); np.random.shuffle(idx); idx= idx[:n_subsample]
                X_full = X
                Y_full = self.Y.get_value()
                self.set_dataset(X_full[idx],Y_full[idx])

            super(SSGP, self).train()

            if X_full is not None:
                # restore full dataset for SSGP training
                utils.print_with_stamp('Restoring full dataset',self.name)
                self.set_dataset(X_full,Y_full)

        idims = self.D
        odims = self.E

        if self.loss_ss_fn is None or self.should_recompile:
            self.init_loss()

        # initialize spectral samples
        loss = self.loss_ss_fn()
        best_w = self.w.get_value()

        # try a couple spectral samples and pick the one with the lowest loss
        for i in xrange(100):
            self.set_spectral_samples()
            loss_i = self.loss_ss_fn()
            for d in xrange(odims):
                if np.all(loss_i[d] < loss[d]):
                    loss[d] = loss_i[d]
                    best_w[:,d,:] = self.w.get_value()[:,d,:]

        self.set_spectral_samples( best_w )

        # train the pseudo input locations
        utils.print_with_stamp('loss SS: %s'%(np.array(self.loss_ss_fn())),self.name)
        # wrap loghyp plus sr (save shapes)
        p0 = [self.loghyp.get_value(),self.w.get_value()]
        parameter_shapes = [p.shape for p in p0]
        m_loss_ss = utils.MemoizeJac(self.loss_ss)
        opt_res = minimize(m_loss_ss, utils.wrap_params(p0), args=parameter_shapes, jac=m_loss_ss.derivative, method=self.min_method, tol=self.conv_thr, options={'maxiter': int(self.max_evals)})
        print ''
        loghyp,w = utils.unwrap_params(opt_res.x,parameter_shapes)
        self.set_params({'loghyp': loghyp, 'w': w})
        utils.print_with_stamp('loss SS: %s'%(np.array(self.loss_ss_fn())),self.name)
        self.trained = True

    def predict_symbolic(self,mx,Sx):
        odims = self.E
        idims = self.D

        # compute the mean and variance for each output dimension
        mean = [[]]*odims
        variance = [[]]*odims
        for i in xrange(odims):
            sr = self.sr[i]
            M = sr.shape[0].astype('float64')
            sf2 = tt.exp(2*self.loghyp[i,idims])
            sn2 = tt.exp(2*self.loghyp[i,idims+1])
            # sr.T.dot(x) for all sr and X. size n_inducing x N
            srdotX = sr.dot(mx)
            # convert to sin cos
            phi_x = tt.concatenate([ tt.sin(srdotX), tt.cos(srdotX) ])

            mean[i] = phi_x.T.dot(self.beta_ss[i])
            phi_x_L = solve_lower_triangular(self.Lmm[i],phi_x)
            variance[i] = sn2*(1 + (sf2/M)*phi_x_L.dot( phi_x_L ))

        # reshape output variables
        M = tt.stack(mean).tt.flatten()
        S = tt.diag(tt.stack(variance).tt.flatten())
        V = tt.zeros((self.D,self.E))

        return M,S,V

class SSGP_UI(SSGP, GP_UI):
    ''' Sparse Spectral Gaussian Process Regression with Uncertain Inputs'''
    def __init__(self, X_dataset=None, Y_dataset=None, name='SSGP_UI', idims=None, odims=None, profile=False, n_inducing=100, **kwargs):
        SSGP.__init__(self,X_dataset,Y_dataset,name=name,idims=idims,odims=odims,profile=profile,n_inducing=n_inducing,uncertain_inputs=True, **kwargs)

    def predict_symbolic(self,mx,Sx):
        #if self.N < self.n_inducing:
            # stick with the full GP
        #    return GP_UI.predict_symbolic(self,mx,Sx)

        idims = self.D
        odims = self.E
        
        # precompute some variables
        Ms = self.sr.shape[1]
        sf2M = tt.exp(2*self.loghyp[:,idims])/tt.cast(Ms,'float64')
        sn2 = tt.exp(2*self.loghyp[:,idims+1])
        srdotx = self.sr.dot(mx)
        srdotSx = self.sr.dot(Sx) 
        srdotSxdotsr = tt.sum(srdotSx*self.sr,2)
        e = tt.exp(-0.5*srdotSxdotsr)
        cos_srdotx = tt.cos(srdotx)
        sin_srdotx = tt.sin(srdotx)
        cos_srdotx_e = tt.cos(srdotx)*e
        sin_srdotx_e = tt.sin(srdotx)*e

        # compute the mean vector
        mphi = tt.horizontal_stack( sin_srdotx_e, cos_srdotx_e ) # E x 2*Ms
        M = tt.sum( mphi*self.beta_ss, 1)

        # input output covariance
        mx_c = mx.dimshuffle(0,'x'); mx_r = mx.dimshuffle('x',0)
        sin_srdotx_e_r = sin_srdotx_e.dimshuffle(0,'x',1); cos_srdotx_e_r = cos_srdotx_e.dimshuffle(0,'x',1)
        c = tt.concatenate([ mx_c*sin_srdotx_e_r + srdotSx.transpose(0,2,1)*cos_srdotx_e_r, mx_c*cos_srdotx_e_r - srdotSx.transpose(0,2,1)*sin_srdotx_e_r ], axis=2) # E x D x 2*Ms
        beta_ss_r = self.beta_ss.dimshuffle(0,'x',1)
        V = tt.sum( c*beta_ss_r, 2 ).T - tt.outer(mx,M) # input outout covariance (notice this is not premultiplied by the input covariance inverse)
        
        srdotSxdotsr_c = srdotSxdotsr.dimshuffle(0,1,'x')
        srdotSxdotsr_r = srdotSxdotsr.dimshuffle(0,'x',1)
        M2 = tt.zeros((self.E,self.E),dtype='float64')
        # initialize indices
        indices = [ tt.as_index_variable(idx) for idx in np.triu_indices(self.E) ]

        def second_moments(i,j,M2,beta,iA,sn2,sf2M,sr,srdotSx,srdotSxdotsr_c,srdotSxdotsr_r,sin_srdotx,cos_srdotx):
            # compute the second moments of the spectral feature vectors
            siSxsj = srdotSx[i].dot(sr[j].T) #Ms x Ms
            sijSxsij = -0.5*(srdotSxdotsr_c[i] + srdotSxdotsr_r[j]) 
            em =  tt.exp(sijSxsij+siSxsj)      # MsxMs
            ep =  tt.exp(sijSxsij-siSxsj)     # MsxMs
            si = sin_srdotx[i]       # Msx1
            ci = cos_srdotx[i]       # Msx1 
            sj = sin_srdotx[j]       # Msx1
            cj = cos_srdotx[j]       # Msx1
            sicj = tt.outer(si,cj)    # MsxMs
            cisj = tt.outer(ci,sj)    # MsxMs
            sisj = tt.outer(si,sj)    # MsxMs
            cicj = tt.outer(ci,cj)    # MsxMs
            sm = (sicj-cisj)*em
            sp = (sicj+cisj)*ep
            cm = (sisj+cicj)*em
            cp = (cicj-sisj)*ep
            
            # Populate the second moment matrix of the feature vector
            Q_up = tt.concatenate([cm-cp,sm+sp],axis=1)
            Q_lo = tt.concatenate([sp-sm,cm+cp],axis=1)
            Q = tt.concatenate([Q_up,Q_lo],axis=0)

            # Compute the second moment of the output
            m2 = 0.5*matrix_dot(beta[i], Q, beta[j].T)
            
            m2 = theano.ifelse.ifelse(tt.eq(i,j), m2 + sn2[i]*(1.0 + sf2M[i]*tt.sum(self.iA[i]*Q)), m2)
            M2 = tt.set_subtensor(M2[i,j], m2)
            M2 = theano.ifelse.ifelse(tt.eq(i,j), M2 , tt.set_subtensor(M2[j,i], m2))

            return M2

        M2_,updts = theano.scan(fn=second_moments, 
                               sequences=indices,
                               outputs_info=[M2],
                               non_sequences=[self.beta_ss,self.iA,sn2,sf2M,self.sr,srdotSx,srdotSxdotsr_c,srdotSxdotsr_r,sin_srdotx,cos_srdotx],
                               allow_gc=False,
                               name='%s>M2_scan'%(self.name))
        M2 = M2_[-1]
        S = M2 - tt.outer(M,M)

        return M,S,V


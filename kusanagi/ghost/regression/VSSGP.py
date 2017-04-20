from .GP import *

class VSSGP(GP):
    ''' Variational Sparse Spectral Gaussian Process Regression by Gal and Turner 2015'''
    def __init__(self, X_dataset=None, Y_dataset=None, name='VSSGP', idims=None, odims=None, profile=False, n_inducing=100, n_components=2, uncertain_inputs=False, **kwargs):
        self.n_inducing = n_inducing
        self.n_components = n_components
        self.opt_A = True
        self.randomised_phases = True
        super(VSSGP, self).__init__(X_dataset,Y_dataset,name=name,idims=idims,odims=odims,profile=profile,n_inducing=n_inducing,uncertain_inputs=uncertain_inputs, **kwargs)
    
    def init_params(self):
        '''
        initializes the parameter set for VSSGP. Some parameters are stored as
        their logarithms (or tangent ), to ensure they always fall within the valid
        bounds when optimizing
        '''
        X,Y = self.X.get_value(), self.Y.get_value()
        D,E = X.shape[-1], Y.shape[-1]
        nb,nc = self.n_inducing, self.n_components
        # mean and covariances of basis frequencies
        mw = np.random.randn(nc,nb,D); logsw = np.log(1e-2*(np.random.randn(nc,nb,D))**2)
        # inducing inputs
        z = np.random.multivariate_normal(X.mean(0),1.25*np.atleast_2d(np.cov(X.T)),(nc,nb))
        # signal std (one per component)
        logsf = np.log(1 + 1e-2*np.random.randn(nc,)) + np.log(Y.std(ddof=1))
        # noise std
        logsn = np.log(0.1*Y.std(ddof=1))
        # feature lengthscales ( nc x D )
        logL = np.log(1 + 1e-2*np.random.randn(nc,1)) + np.log(X.std(0,ddof=1))[None,:]
        # spectral mixture periods
        logp = np.log(np.random.uniform(0,1e32,size=(nc,D)))

        # create shared variables for every parameter
        params = OrderedDict(list(zip(('mw','logsw','z','logsf','logsn','logL','logp'),(mw,logsw,z,logsf,logsn,logL,logp))))
        # parameters of the uniform distributions for the basis phases
        b = np.random.uniform(0,2*np.pi,size=(2,nc,nb))
        if self.randomised_phases:
            tanb = np.tan(0.5*(b[0,:,:]-np.pi))
            params['tanb'] = tanb
            self.fixed_params.append('tanb')
        else:
            b_lo, b_hi = b.min(axis=0), b.max(axis=0) 
            tanb = np.tan(0.5*(b_lo-np.pi))
            tanb_delta = np.tan(0.5*((b_hi-b_lo)-np.pi))
            params['tanb'] = tanb
            params['tanb_delta'] = tanb_delta
        if not self.opt_A:
            # mean and covariances of the fourier coefficients
            md = np.random.randn(nc,nb,E); logsd = np.log(1e-2*np.ones((nc,nb,E)))
            params['md'] = md
            params['logsd'] = logsd
        self.set_params(params)

    def compute_feature_matrix(self,X):
        mw,logsw,tanb,z,logsf,logsn,logL,logp = self.mw,self.logsw,self.tanb,self.z,self.logsf,self.logsn,self.logL,self.logp
        nc,nb,D = mw.shape
        N,D = X.shape
        iL = tt.exp(-logL)/(2*np.pi)
        ip = tt.exp(-logp) 
        sf2 = tt.exp(2*logsf)
        sn2 = tt.exp(2*logsn)
        sw = tt.exp(logsw)
        b = 2*tt.arctan(tanb) + np.pi

        # scale and center the dataset around each inducing input 
        Xsc = 2*np.pi*(X[None,:,:] - z[:,:,None,:])
        # compute the expected value of the cos term wrt the phases b
        Ew = iL[:,None,:]*mw + ip[:,None,:]
        w_dot_x = (Ew[:,:,None,:]*Xsc).sum(-1)
        
        # compute the cos term
        if self.randomised_phases:
            a = w_dot_x + b[:,:,None]
            mcos = tt.cos(a)
            mcos2 = tt.cos(2*a)
        else:
            b_hi = b + (2*tt.arctan(self.tanb_delta) + np.pi)
            alpha = b[:,:,None] 
            beta = b_hi[:,:,None]
            a = w_dot_x + alpha
            b = w_dot_x + beta
            mcos = ( tt.sin(b) - tt.sin(a) )/(beta-alpha) 
            mcos2 = ( tt.sin(2*b) - tt.sin(2*a) )/(beta-alpha) 

        # compute the expected value wrt w
        e = tt.exp( -0.5*(((iL[:,None,None,:]*Xsc)**2)*sw[:,:,None,:]).sum(-1) )
        # get the expected value of the feature vector phi
        sf2K = 2*sf2/nb
        mphi = (sf2K**0.5)[:,None,None]*e*mcos
        # reshape into an N x (nc*nb) matrix
        mphi = mphi.transpose(2,0,1).reshape((N,nc*nb))
        # get the expected value of phi.T.dot(phi) (second moment of phi)
        mcos2 = sf2K[:,None,None]*(0.5 + 0.5*(e**4)*mcos2)
        mphiTphi = mphi.T.dot(mphi)
        mphiTphi = mphiTphi - tt.diag(tt.diag(mphiTphi)) + tt.diag(mcos2.sum(-1).flatten())
        return mphi, mphiTphi, mcos2

    def init_loss(self):
        utils.print_with_stamp('Initialising expression graph for sparse spectral training loss function',self.name)

        if not hasattr(self,'iA'):
            self.iA = S(np.zeros((self.n_inducing*self.n_components,self.n_inducing*self.n_components),dtype='float64'), name="%s>iA"%(self.name))
        if not hasattr(self,'Lmm'):
            self.Lmm = S(np.zeros((self.n_inducing*self.n_components,self.n_inducing*self.n_components),dtype='float64'), name="%s>Lmm"%(self.name))
        if not hasattr(self,'beta_ss'):
            self.beta_ss = S(np.zeros((self.n_inducing*self.n_components,self.E),dtype='float64'), name="%s>beta_ss"%(self.name))
        
        # intialize parameters of the model
        X,Y = self.X,self.Y
        self.init_params()
        mphi,mphiTphi,mcos2 = self.compute_feature_matrix(self.X)

        K = mphiTphi.shape[0]
        N,E = Y.shape
        EyeK = tt.eye(K)
        sn2 = tt.exp(2*self.logsn)
        tau = tt.exp(-2*self.logsn)
        
        # log likelihood term
        if self.opt_A:
            iSig = mphiTphi*tau + EyeK
            choliSig = cholesky(iSig)
            Sig = solve_upper_triangular(choliSig.T, solve_lower_triangular(choliSig,EyeK)) # sn2*Sig
            mphiTY = mphi.T.dot(Y)
            M = tau*solve_upper_triangular(choliSig.T, solve_lower_triangular(choliSig,mphiTY)) # Sig*EPhi.T*Y
            L_vb = - 0.5*N*E*tt.log(tau) + 0.5*N*E*np.log(2*np.pi) + 0.5*tau*(Y**2).sum() - 0.5*tau*(mphiTY*M).sum() + 0.5*E*tt.sum(2*tt.log(tt.diag(choliSig))) 
            updts = [(self.iA,Sig),(self.Lmm,choliSig),(self.beta_ss,M)]
        else:
            M,Sig = self.md.transpose(2,0,1).reshape((E,K)).T, tt.exp(self.logsd).transpose(2,0,1).reshape((E,K)).T
            mphiTY = mphi.T.dot(Y)
            L_vb = - 0.5*N*E*tt.log(tau) + 0.5*N*E*np.log(2*np.pi) + 0.5*tau*(Y**2).sum() - tau*(mphiTY*M).sum() + 0.5*tau*tt.sum(mphiTphi*(tt.diag(Sig.sum(-1)) + (M[None,:,:]*M[:,None,:]).sum(-1)))
            updts = [(self.iA,Sig),(self.beta_ss,M)]

        # KL divergence for spectral basis frequencies
        L_vb += 0.5 * (tt.exp(self.logsw) + self.mw**2 - self.logsw - 1).sum()
        if not self.randomised_phases:
            b = 2*np.arctan(self.tanb) + np.pi 
            bdelta =2*np.arctan(self.tanb_delta) + np.pi 
            # KL divergence for spectral basis phases
            L_vb +=  (tt.log(2*np.pi/(bdelta))).sum() 
            # Contrainte penalty barriers ( to keep b > 0 and b+b_delta < 2*pi
            L_vb +=  -1e-9*(tt.log( 2*np.pi + (b+bdelta) ) + tt.log(b)).sum()

        if not self.opt_A:
            # KL divergence for fourier coefficients
            L_vb += 0.5 * (tt.exp(self.logsd) + self.md**2 - self.logsd - 1).sum()

        # snr penalty ( This helps to prevent overfitting )
        L_vb += (((self.logsf - self.logsn)/np.log(1000))**30).sum()
        # lengthscale penalty ( we don't want them to grow too large as that would make the gradients go to zero )
        L_vb += (((self.logL - np.log(self.X.std(0)))/np.log(100))**30).sum()
        # penalty for large sn. helps escapipng local minima for small datasets.
        L_vb += 100*self.logsn

        # Compute the gradients for the sum of loss for all output dimensions
        dL_vb = tt.grad(L_vb.sum(),self.get_params(symbolic=True))
        dretvars = [L_vb]
        dretvars.extend(dL_vb)
        utils.print_with_stamp('Compiling training loss function',self.name)
        self.loss_fn = F((),L_vb,name='%s>loss_ss'%(self.name), profile=self.profile, mode=self.compile_mode, allow_input_downcast=True,updates=updts, on_unused_input='ignore')
        utils.print_with_stamp('Compiling gradient of training loss function',self.name)
        self.dloss_fn = F((),dretvars,name='%s>dloss_ss'%(self.name), profile=self.profile, mode=self.compile_mode, allow_input_downcast=True,updates=updts, on_unused_input='ignore')
    
    def loss(self,new_p,parameter_shapes):
        p=utils.unwrap_params(new_p,parameter_shapes)
        param_names = [pname for pname in self.param_names if pname not in self.fixed_params]
        pdict = dict(list(zip(param_names,p)))
        self.set_params(pdict)
        ret = self.dloss_fn()
        loss,dloss = ret[0], utils.wrap_params(ret[1:])
        # on a 64bit system, scipy optimize complains if we pass a 32 bit float
        res = (loss.astype(np.float64), dloss.astype(np.float64))
        self.n_evals+=1
        utils.print_with_stamp('loss: %s    \t n_evals: %d'%(str(res[0]), self.n_evals),self.name,True)
        return res

    def train(self):
        if self.loss_fn is None or self.should_recompile:
            self.init_loss()

        p0 = self.get_params()
        parameter_shapes = [p.shape for p in p0]
        #utils.print_with_stamp('Current hyperparameters:\n%s'%(p0),self.name)
        utils.print_with_stamp('loss: %s'%(np.array(self.loss_fn())),self.name)
        m_loss = utils.MemoizeJac(self.loss)
        p0 = utils.wrap_params(p0)
        self.n_evals=0
        opt_res = minimize(m_loss, p0, jac=m_loss.derivative, args=parameter_shapes, method=self.min_method, tol=self.conv_thr, options={'maxiter': self.max_evals})
        print('')
        new_p = opt_res.x 
        self.state_changed = not np.allclose(p0,new_p,1e-6,1e-9)
        #utils.print_with_stamp('New hyperparameters:\n%s'%(new_p),self.name)
        p=utils.unwrap_params(new_p,parameter_shapes)
        param_names = [pname for pname in self.param_names if pname not in self.fixed_params]
        pdict = dict(list(zip(param_names,p)))
        self.set_params(pdict)
        utils.print_with_stamp('loss: %s'%(np.array(self.loss_fn())),self.name)
        self.trained = True
    
    def predict_symbolic(self,mx,Sx):
        odims = self.E
        idims = self.D
        mphi,mphiTphi,mcos2 = self.compute_feature_matrix(mx[None,:])
        M = mphi.dot(self.beta_ss).flatten()
        sn2 = tt.exp(2*self.logsn)
        if self.opt_A:
            S = sn2*tt.eye(M.shape[0]) + tt.sum(mphiTphi*self.iA)*tt.eye(M.shape[0]) + self.beta_ss.T.dot(mphiTphi - mphi.T.dot(mphi)).dot(self.beta_ss)
        else:
            S = sn2*tt.eye(M.shape[0]) + tt.diag((tt.diag(mphiTphi)[:,None]*self.iA).sum(0)) + self.beta_ss.T.dot(mphiTphi - mphi.T.dot(mphi)).dot(self.beta_ss)
        V = tt.zeros((self.D,self.E))

        return M,S,V

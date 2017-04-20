from .GP import *
from scipy.cluster.vq import kmeans

class SPGP(GP):
    '''Sparse Pseudo Input FITC approximation Snelson and Gharammani 2005'''
    def __init__(self, X_dataset=None, Y_dataset=None, name = 'SPGP', idims=None, odims=None, profile=False, n_inducing = 100, uncertain_inputs=False, **kwargs):
        self.X_sp = None # inducing inputs (symbolic variable)
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
        GP.__init__(self,X_dataset,Y_dataset,name=name,idims=idims,odims=odims,profile=profile,uncertain_inputs=uncertain_inputs, **kwargs)

    def init_pseudo_inputs(self):
        assert self.N > self.n_inducing, "Dataset must have more than n_inducing [ %n ] to enable inference with sparse pseudo inputs"%(self.n_inducing)
        self.should_recompile = True
        # pick initial cluster centers from dataset
        X = self.X.get_value()
        X_sp_ = utils.kmeanspp(X,self.n_inducing)

        # perform kmeans to get initial cluster centers
        utils.print_with_stamp('Initialising pseudo inputs',self.name)
        X_sp_, dist = kmeans(X, X_sp_, iter=200,thresh=1e-9)
        # initialize symbolic tensor variable if necessary (this will create the self.X_sp atttribute)
        self.set_params({'X_sp': X_sp_})

    def set_dataset(self,X_dataset,Y_dataset):
        # set the dataset on the parent class
        super(SPGP, self).set_dataset(X_dataset,Y_dataset)
        if self.N <= self.n_inducing:
            utils.print_with_stamp('Dataset is not large enough for using pseudo inputs. Training full GP.',self.name)
            self.X_sp = None
            self.loss_sp_fn = None
            self.dloss_sp_fn = None
            self.beta_sp = None
            self.Lmm = None
            self.Amm = None
            self.should_recompile = False

        if self.N > self.n_inducing and self.X_sp is None:
            utils.print_with_stamp('Dataset is large enough for using pseudo inputs. You should reinitiialise the training loss function and predictions.',self.name)
            # init the shared variable for the pseudo inputs
            self.init_pseudo_inputs()
            self.should_recompile = True
        
    def init_loss(self, cache_vars=True):
        # initialize the training loss function of the GP class
        super(SPGP, self).init_loss(cache_vars)
        # here loss and dloss have already been innitialised, sow e can replace loss and dloss
        # only if we have enough data to train the pseudo inputs ( i.e. self.N > self.n_inducing)
        if self.N > self.n_inducing:
            utils.print_with_stamp('Initialising FITC training loss function',self.name)
            self.should_recompile = False
            odims = self.E
            idims = self.D
            
            # initialize shared variables
            if self.iKmm is None:
                self.iKmm = S(np.zeros((self.E,self.n_inducing,self.n_inducing)), name="%s>iKmm"%(self.name))
            if self.Lmm is None:
                self.Lmm = S(np.zeros((self.E,self.n_inducing,self.n_inducing)), name="%s>Lmm"%(self.name))
            if self.Amm is None:
                self.Amm = S(np.zeros((self.E,self.n_inducing,self.n_inducing)), name="%s>Amm"%(self.name))
            if self.iBmm is None:
                self.iBmm = S(np.zeros((self.E,self.n_inducing,self.n_inducing)), name="%s>iBmm"%(self.name))
            if self.beta_sp is None:
                self.beta_sp = S(np.zeros((self.E,self.n_inducing)), name="%s>beta_sp"%(self.name))

            # initialize the training loss function of the sparse FITC approximation
            def log_marginal_likelihood(Y,loghyp,X,X_sp,EyeM):
                # TODO allow for different pseudo inputs for each dimension
                # initialise the (before compilation) kernel function
                loghyps = (loghyp[:idims+1],loghyp[idims+1])
                kernel_func = partial(cov.Sum, loghyps, self.covs)

                ll = tt.exp(loghyp[:idims])
                sf2 = tt.exp(2*loghyp[idims])
                sn2 = tt.exp(2*loghyp[idims+1])
                N = X.shape[0].astype('float64')
                M = X_sp.shape[0].astype('float64')

                ridge = 1e-6
                Kmm = kernel_func(X_sp) + ridge*EyeM
                Kmn = kernel_func(X_sp, X)
                Lmm = cholesky(Kmm)
                iKmm = solve_upper_triangular(Lmm.T, solve_lower_triangular(Lmm,EyeM))
                Lmn  = solve_lower_triangular(Lmm,Kmn)
                diagQnn =  tt.diag(Lmn.T.dot(Lmn))

                # Gamma = diag(Knn - Qnn) + sn2*I
                Gamma = sf2 - diagQnn + sn2
                Gamma_inv = 1.0/Gamma

                # these operations are done to avoid inverting K_sp = (Qnn+Gamma)
                Kmn_ = Kmn*tt.sqrt(Gamma_inv)                    # Kmn_*Gamma^-.5
                Yi = Y*(tt.sqrt(Gamma_inv))                      # Gamma^-.5* Y
                Bmm = Kmm + (Kmn_).dot(Kmn_.T)                  # Kmm + Kmn * Gamma^-1 * Knm
                Amm = cholesky(Bmm)
                iBmm = solve_upper_triangular(Amm.T, solve_lower_triangular(Amm,EyeM))

                Yci = solve_lower_triangular(Amm,Kmn_.dot(Yi) )
                beta_sp = solve_upper_triangular(Amm.T,Yci)

                log_det_K_sp = tt.sum(tt.log(Gamma)) - 2*tt.sum(tt.log(tt.diag(Lmm))) + 2*tt.sum(tt.log(tt.diag(Amm)))

                loss_sp = 0.5*( Yi.dot(Yi) - Yci.dot(Yci) + log_det_K_sp + N*np.log(2*np.pi) )

                return loss_sp, iKmm, Lmm, Amm, iBmm, beta_sp
            
            (loss_sp, iKmm, Lmm, Amm, iBmm, beta_sp),updts = theano.scan(fn=log_marginal_likelihood, sequences=[self.Y.T,self.loghyp], non_sequences=[self.X,self.X_sp,tt.eye(self.X_sp.shape[0])],allow_gc=False)
            
            iKmm = tt.unbroadcast(iKmm,0) if iKmm.broadcastable[0] else iKmm
            Lmm = tt.unbroadcast(Lmm,0) if Lmm.broadcastable[0] else Lmm
            Amm = tt.unbroadcast(Amm,0) if Amm.broadcastable[0] else Amm
            iBmm = tt.unbroadcast(iBmm,0) if iBmm.broadcastable[0] else iBmm
            beta_sp = tt.unbroadcast(beta_sp,0) if beta_sp.broadcastable[0] else beta_sp
    
            if cache_vars:
                # we are going to save the intermediate results in the following shared variables, so we can use them during prediction without having to recompute them
                updts = [(self.iKmm,iKmm),(self.Lmm,Lmm),(self.Amm,Amm),(self.iBmm,iBmm),(self.beta_sp,beta_sp)]
            else:
                self.iKmm = iKmm 
                self.Lmm = Lmm 
                self.Amm = Amm 
                self.iBmm = iBmm 
                self.beta_sp = beta_sp
                updts=None


            # TODO include the log hyperparameters in the optimization
            # TODO give the option for separate inducing inputs for every output dimension
            dloss_sp = tt.grad(loss_sp.sum(),self.X_sp)

            # Compile the theano functions
            utils.print_with_stamp('Compiling FITC training loss function',self.name)
            self.loss_sp_fn = F((),loss_sp,name='%s>loss_sp'%(self.name), profile=self.profile, mode=self.compile_mode, allow_input_downcast=True, updates=updts)
            utils.print_with_stamp('Compiling gradient of FITC training loss function',self.name)
            self.dloss_sp_fn = F((),(loss_sp,dloss_sp),name='%s>dloss_sp'%(self.name), profile=self.profile, mode=self.compile_mode, allow_input_downcast=True,updates=updts)
            
    def predict_symbolic(self,mx,Sx):
        if self.N <= self.n_inducing:
            # stick with the full GP
            return super(SPGP, self).predict_symbolic(mx,Sx)

        idims = self.D
        odims = self.E

        # compute the mean and variance for each output dimension
        mean = [[]]*odims
        variance = [[]]*odims
        def predict_odim(Lmm,Amm,beta_sp,loghyp,X_sp,mx):
            loghyps = (loghyp[:idims+1],loghyp[idims+1])
            kernel_func = partial(cov.Sum, loghyps, self.covs)
            
            k = kernel_func(mx[None,:],X_sp).flatten()
            mean = k.dot(beta_sp)
            kL = solve_lower_triangular(Lmm,k)
            kA = solve_lower_triangular(Amm,k)
            variance = kernel_func(mx[None,:],all_pairs=False) - kL.dot(kL) -  kA.dot(kA)
            
            return mean, variance
        
        (M,S), updts = theano.scan(fn=predict_odim, sequences=[self.Lmm,self.Amm,self.beta_sp,self.loghyp], non_sequences=[self.X_sp,mx],allow_gc=False)

        # reshape output variables
        M = M.flatten()
        S = tt.diag(S.flatten())
        V = tt.zeros((self.D,self.E))

        return M,S,V
    
    def loss_sp(self,X_sp):
        self.set_params({'X_sp': X_sp})
        res = self.dloss_sp_fn()
        loss = np.array(res[0]).sum()
        dloss = np.array(res[1]).flatten()
        # on a 64bit system, scipy optimize complains if we pass a 32 bit float
        res = (loss.astype(np.float64),dloss.astype(np.float64))
        utils.print_with_stamp('%s'%(str(res[0])),self.name,True)
        return res

    def train(self):
        if self.loss_sp_fn is None or self.should_recompile:
            self.init_loss()
        # train the full GP
        super(SPGP, self).train()

        if self.N > self.n_inducing:
            # train the pseudo input locations
            if self.loss_sp_fn is None:
                self.init_loss()
            utils.print_with_stamp('loss SP: %s'%(np.array(self.loss_sp_fn())),self.name)
            m_loss_sp = utils.MemoizeJac(self.loss_sp)
            opt_res = minimize(m_loss_sp, self.X_sp.get_value(), jac=m_loss_sp.derivative, method=self.min_method, tol=self.conv_thr, options={'maxiter': int(self.max_evals)})
            print('')
            X_sp = opt_res.x.reshape(self.X_sp.get_value(borrow=True).shape)
            self.set_params({'X_sp': X_sp})
            utils.print_with_stamp('loss SP: %s'%(np.array(self.loss_sp_fn())),self.name)
        self.trained = True

class SPGP_UI(SPGP,GP_UI):
    def __init__(self, X_dataset=None, Y_dataset=None, name = 'SPGP_UI', idims=None, odims=None,profile=False, n_inducing = 100, **kwargs):
        SPGP.__init__(self,X_dataset,Y_dataset,name=name,idims=idims,odims=odims,profile=profile,n_inducing=n_inducing,uncertain_inputs=True, **kwargs)

    def predict_symbolic(self,mx,Sx):
        if self.N <= self.n_inducing:
            # stick with the full GP
            return GP_UI.predict_symbolic(self,mx,Sx)

        idims = self.D
        odims = self.E

        #centralize inputs 
        zeta = self.X_sp - mx
        
        # initialize some variables
        sn2 = tt.exp(2*self.loghyp[:,idims+1])
        sf2 = tt.exp(2*self.loghyp[:,idims])
        eyeE = tt.tile(tt.eye(idims),(odims,1,1))
        lscales = tt.exp(self.loghyp[:,:idims])
        iL = eyeE/lscales.dimshuffle(0,1,'x')

        # predictive mean
        inp = iL.dot(zeta.T).transpose(0,2,1) 
        iLdotSx = iL.dot(Sx)
        B = tt.stack([iLdotSx[i].dot(iL[i]) for i in range(odims)]) + tt.eye(idims)   #TODO vectorize this
        t = tt.stack([inp[i].dot(matrix_inverse(B[i])) for i in range(odims)])      # E x N x D
        c = sf2/tt.sqrt(tt.stack([det(B[i]) for i in range(odims)]))
        l = tt.exp(-0.5*tt.sum(inp*t,2))
        lb = l*self.beta_sp # beta_sp should have been precomputed in init_loss # E x N dot E x N
        M = tt.sum(lb,1)*c
        
        # input output covariance
        tiL = tt.stack([t[i].dot(iL[i]) for i in range(odims)])
        #V = Sx.dot(tt.stack([tiL[i].T.dot(lb[i]) for i in xrange(odims)]).T*c)
        V = tt.stack([tiL[i].T.dot(lb[i]) for i in range(odims)]).T*c

        # predictive covariance
        logk = 2*self.loghyp[:,None,idims] - 0.5*tt.sum(inp*inp,2)
        logk_r = logk.dimshuffle(0,'x',1)
        logk_c = logk.dimshuffle(0,1,'x')
        Lambda = tt.square(iL)
        R = tt.dot((Lambda.dimshuffle(0,'x',1,2) + Lambda).transpose(0,1,3,2),Sx.T).transpose(0,1,3,2) + tt.eye(idims)
        z_= Lambda.dot(zeta.T).transpose(0,2,1) 
        
        M2 = tt.zeros((self.E,self.E))
        # initialize indices
        indices = [ tt.as_index_variable(idx) for idx in np.triu_indices(self.E) ]

        def second_moments(i,j,M2,beta,iK,sf2,R,logk_c,logk_r,z_,Sx):
            # This comes from Deisenroth's thesis ( Eqs 2.51- 2.54 )
            Rij = R[i,j]
            n2 = logk_c[i] + logk_r[j] + utils.maha(z_[i],-z_[j],0.5*matrix_inverse(Rij).dot(Sx))
            Q = tt.exp( n2 )/tt.sqrt(det(Rij))
            # Eq 2.55
            m2 = matrix_dot(beta[i], Q, beta[j])
            
            m2 = theano.ifelse.ifelse(tt.eq(i,j), m2 - tt.sum(iK[i]*Q) + sf2[i], m2)
            M2 = tt.set_subtensor(M2[i,j], m2)
            M2 = theano.ifelse.ifelse(tt.eq(i,j), M2 , tt.set_subtensor(M2[j,i], m2))
            return M2

        M2_,updts = theano.scan(fn=second_moments, 
                               sequences=indices,
                               outputs_info=[M2],
                               non_sequences=[self.beta_sp,(self.iKmm - self.iBmm),sf2,R,logk_c,logk_r,z_,Sx],
                               allow_gc=False)
        M2 = M2_[-1]
        S = M2 - tt.outer(M,M)

        return M,S,V

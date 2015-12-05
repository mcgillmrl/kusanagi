import os,sys
from functools import partial

import numpy as np
import theano
import theano.tensor as T
from scipy.optimize import minimize
from theano import function as F, shared as S
from theano.misc.pkl_utils import dump as t_dump, load as t_load
from theano.tensor.nlinalg import matrix_dot
from theano.sandbox.linalg import psd,matrix_inverse,det

import cov
import utils

class GP(object):
    def __init__(self, X_dataset, Y_dataset, name='GP', profile=False):
        self.profile= profile
        self.compile_mode = theano.compile.get_default_mode()#.excluding('scanOp_pushout_seqs_ops')

        self.X_ = None; self.Y_ = None
        self.X = None; self.Y = None; self.loghyp = None
        self.name = name
        self.idims = X_dataset.shape[1]
        self.odims = Y_dataset.shape[1]
        self.filename = '%s_%d_%d_%s_%s'%(self.name,self.idims,self.odims,theano.config.device,theano.config.floatX)
        self.should_save = False

        try:
            # try loading from pickled file, to avoid recompiling
            self.load()
            if (X_dataset.shape[0] != self.X_.shape[0] or Y_dataset.shape[0] != self.Y_.shape[0]) or not( np.allclose(X_dataset,self.X_) and np.allclose(Y_dataset,self.Y_) ):
                self.set_dataset(X_dataset,Y_dataset)
        
        except IOError:
            # initialize the class if no pickled version is available
            assert X_dataset.shape[0] == Y_dataset.shape[0], "X_dataset and Y_dataset must have the same number of rows"
            self.set_dataset(X_dataset,Y_dataset)
            self.init_log_likelihood()
            self.init_predict()
            self.save()
        
        utils.print_with_stamp('Finished initialising GP',self.name)
    
    def set_dataset(self,X_dataset,Y_dataset):
        utils.print_with_stamp('Updating GP dataset',self.name)
        # ensure we don't change the number of input and output dimensions ( the number of samples can change)
        if self.X_ is not None:
            assert self.X_.shape[1] == X_dataset.shape[1]
        if self.Y_ is not None:
            assert self.Y_.shape[1] == Y_dataset.shape[1]

        # first, assign the numpy arrays to class members
        if theano.config.floatX == 'float32':
            self.X_ = X_dataset.astype(np.float32)
            self.Y_ = Y_dataset.astype(np.float32)
        else:
            self.X_ = X_dataset
            self.Y_ = Y_dataset
        idims = self.X_.shape[1]
        odims = self.Y_.shape[1]
        N = self.X_.shape[0]
        self.idims = idims
        self.odims = odims
        self.N = N

        # and initialize the loghyperparameters of the gp ( this code supports squared exponential only, at the moment)
        self.loghyp_ = np.zeros((odims,idims+2))
        if theano.config.floatX == 'float32':
            self.loghyp_ = self.loghyp_.astype(np.float32)
        self.loghyp_[:,:idims] = self.X_.std(0)
        self.loghyp_[:,idims] = self.Y_.std(0)
        self.loghyp_[:,idims+1] = 0.1*self.loghyp_[:,idims]
        self.loghyp_ = np.log(self.loghyp_)
        
        # now we create symbolic shared variables ( borrowing the memory from the numpy arrays)
        if self.X is None:
            self.X = S(self.X_,name='X', borrow=True)
        else:
            self.X.set_value(self.X_, borrow = True)
        if self.Y is None:
            self.Y = S(self.Y_,name='Y', borrow=True)
        else:
            self.Y.set_value(self.Y_, borrow = True)
        # this creates one theano shared variable for the loghyperparameters of each output dimension, separately
        if self.loghyp is None:
            self.loghyp = [S(self.loghyp_[i,:],name='loghyp', borrow=True) for i in xrange(odims)]
        else:
            for i in xrange(odims):
                self.loghyp[i].set_value(self.loghyp_[i,:], borrow = True)

        # we should be saving, since we updated the trianing dataset
        self.should_save = True

    def set_loghyp(self, loghyp):
        loghyp = loghyp.reshape(self.loghyp_.shape)
        if theano.config.floatX == 'float32':
            loghyp = loghyp.astype(np.float32)
        np.copyto(self.loghyp_,loghyp)

    def init_log_likelihood(self):
        utils.print_with_stamp('Initialising expression graph for log likelihood',self.name)
        idims = self.idims
        odims = self.odims

        self.kernel_func = [[]]*odims
        self.K = [[]]*odims
        self.iK = [[]]*odims
        self.beta = [[]]*odims
        nlml = [[]]*odims
        dnlml = [[]]*odims
        covs = (cov.SEard, cov.Noise)
        for i in xrange(odims):
            # initialise the (before compilation) kernel function
            loghyps = (self.loghyp[i][:idims+1],self.loghyp[i][idims+1])
            self.kernel_func[i] = partial(cov.Sum, loghyps, covs)

            # We initialise the kernel matrices (one for each output dimension)
            self.K[i] = self.kernel_func[i](self.X)
            self.K[i].name = 'K_%d'%(i)
            self.iK[i] = matrix_inverse(psd(self.K[i]))
            self.iK[i].name = 'iK_%d'%(i)
            self.beta[i] = self.iK[i].dot(self.Y[:,i])
            self.beta[i].name = 'beta_%d'%(i)

            # And finally, the negative log marginal likelihood ( again, one for each dimension; although we could share
            # the loghyperparameters across all output dimensions and train the GPs jointly)
            nlml[i] = 0.5*(self.Y[:,i].T.dot(self.beta[i]) + T.log(det(psd(self.K[i]))) + self.N*T.log(2*np.pi) )/self.N
            # Compute the gradients for each output dimension independently
            dnlml[i] = T.jacobian(nlml[i].flatten(),self.loghyp[i])
        nlml = T.stacklists(nlml)
        dnlml = T.stacklists(dnlml)
        # Compile the theano functions
        utils.print_with_stamp('Compiling log likelihood',self.name)
        self.nlml = F((),nlml,name='%s>nlml'%(self.name), profile=self.profile, mode=self.compile_mode)
        utils.print_with_stamp('Compiling jacobian of log likelihood',self.name)
        self.dnlml = F((),(nlml,dnlml),name='%s>dnlml'%(self.name), profile=self.profile, mode=self.compile_mode)
    
    def init_predict(self):
        utils.print_with_stamp('Initialising expression graph for prediction',self.name)
        odims = self.odims
        # and a expresion to evaluate the kernel vector at a new evaluation point
        if theano.config.floatX == 'float32':
            x_mean = T.fmatrix('x')
        else:
            x_mean = T.dmatrix('x')

        # compute the mean and variance for each output dimension
        mean = [[]]*odims
        variance = [[]]*odims
        for i in xrange(odims):
            k = self.kernel_func[i](x_mean,self.X)
            mean = k.dot(self.beta[i])
            variance = self.kernel_func[i](x_mean,all_pairs=False) - (k*(k.dot( self.iK[i] )) ).sum(axis=1)

        # compile the prediction function
        M = T.stacklists(mean).T
        S = T.stacklists(variance).T
        utils.print_with_stamp('Compiling mean and variance of prediction',self.name)
        self.predict_ = F([x_mean],(M,S),name='%s>predict_'%(self.name), profile=self.profile, mode=self.compile_mode)

        # compile the derivatives wrt the evaluation point
        dMdm = T.jacobian(M.flatten(),x_mean)
        dSdm = T.jacobian(S.flatten(),x_mean)

        utils.print_with_stamp('Compiling derivatives of mean and variance of prediction',self.name)
        self.predict_d_ = F([x_mean], (M,dMdm,S,dSdm),name='%s>predict_d_'%(self.name), profile=self.profile, mode=self.compile_mode)
    
    def predict(self,x_mean,x_cov = None):
        # cast to float 32 if necessary
        if theano.config.floatX == 'float32':
            x_mean = x_mean.astype(np.float32)

        odims = self.odims
        idims = self.idims
        n = x_mean.shape[0]
        if len(x_mean.shape) == 1:
            # convert to row vector
            x_mean = x_mean[None,:]

        res = None
        if self.name == 'GP':
            res = self.predict_(x_mean)
        else:
            if x_cov is None:
                x_cov = np.zeros((n,idims,idims))
            if theano.config.floatX == 'float32':
                x_cov = x_cov.astype(np.float32)
            res = self.predict_(x_mean, x_cov)
        return res

    def predict_d(self,x_mean,x_cov=None):
        # cast to float 32 if necessary
        if theano.config.floatX == 'float32':
            x_mean = x_mean.astype(np.float32)
            if x_cov is not None:
                x_cov = x_cov.astype(np.float32)
        odims = self.odims
        idims = self.idims
        n = x_mean.shape[0]
        if len(x_mean.shape) == 1:
            # convert to row vector
            x_mean = x_mean[None,:]
        res = None

        if x_cov is None or self.name == 'GP':
            if self.name == 'GP':
                res = self.predict_(x_mean)
            else:
                res = self.predict_(x_mean, np.zeros((n_test,idims,idims)) )
        else:
            res = self.predict_d_(x_mean, x_cov)
        return res
    
    def loss(self,loghyp):
        self.set_loghyp(loghyp)
        res = self.dnlml()
        nlml = np.array(res[0]).sum()
        dnlml = np.array(res[1]).flatten()
        # on a 64bit system, scipy optimize complains if we pass a 32 bit float
        res = (nlml.astype(np.float64),dnlml.astype(np.float64)) if theano.config.floatX == 'float32' else (nlml,dnlml)
        utils.print_with_stamp('%s, %s'%(str(res[0]),str(np.exp(loghyp))),self.name,True)
        return res

    def train(self):
        init_hyp = self.loghyp_.copy()
        utils.print_with_stamp('Current hyperparameters:',self.name)
        print np.exp(self.loghyp_)
        utils.print_with_stamp('nlml: %s'%(np.array(self.nlml())),self.name)
        opt_res = minimize(self.loss, self.loghyp_, jac=True, method="L-BFGS-B", tol=1e-6, options={'maxiter': 100})
        print ''
        loghyp = opt_res.x.reshape(self.loghyp_.shape)
        self.should_save = not np.allclose(init_hyp,loghyp,1e-6,1e-9)
        np.copyto(self.loghyp_,loghyp)
        utils.print_with_stamp('New hyperparameters:',self.name)
        print np.exp(self.loghyp_)
        utils.print_with_stamp('nlml: %s'%(np.array(self.nlml())),self.name)
        self.save()

    def load(self):
        with open(self.filename+'.zip','rb') as f:
            utils.print_with_stamp('Loading compiled GP with %d inputs and %d outputs'%(self.idims,self.odims),self.name)
            state = t_load(f)
            self.X = state[0]
            self.Y = state[1]
            self.loghyp = state[2]
            self.set_dataset(state[3],state[4])
            self.set_loghyp(state[5])
            self.nlml = state[6]
            self.dnlml = state[7]
            self.predict_ = state[8]
            self.predict_d_ = state[9]
        self.should_save = False

    def save(self):
        sys.setrecursionlimit(100000)
        if self.should_save:
            with open(self.filename+'.zip','wb') as f:
                utils.print_with_stamp('Saving compiled GP with %d inputs and %d outputs'%(self.idims,self.odims),self.name)
                state = (self.X,self.Y,self.loghyp,self.X_,self.Y_,self.loghyp_,self.nlml,self.dnlml,self.predict_,self.predict_d_)
                t_dump(state,f,2)

class GPUncertainInputs(GP):
    def __init__(self, X_dataset, Y_dataset, name = 'GPUncertainInputs',profile=False):
        super(GPUncertainInputs, self).__init__(X_dataset,Y_dataset,name=name,profile=profile)

    def init_predict(self):
        utils.print_with_stamp('Initialising expression graph for prediction',self.name)
        idims = self.idims
        odims = self.odims

        # Note that this handles n_samples inputs
        if theano.config.floatX == 'float32':
            x_mean = T.fmatrix('x')      # n_samples x idims
            x_cov = T.ftensor3('x_cov')  # n_samples x idims x idims
        else:
            x_mean = T.dmatrix('x')      # n_samples x idims
            x_cov = T.dtensor3('x_cov')  # n_samples x idims x idims

        #centralize inputs 
        zeta = self.X[None,:,:] - x_mean[:,None,:]
        
        # predictive mean and covariance (for each output dimension)
        M = [] # mean
        V = [] # inv(x_cov).dot(input_output_cov)
        M2 = [[]]*(odims**2) # second moment

        def M_helper(inp_k,B_k,sf2):
            t_k = inp_k.dot(matrix_inverse(psd(B_k)))
            c_k = sf2/T.sqrt(det(psd(B_k)))
            return (t_k,c_k)
            
        #predictive second moment ( only the lower triangular part, including the diagonal)
        def M2_helper(logk_i_k, logk_j_k, z_ij_k, R_k, x_cov_k):
            nk2 = logk_i_k[:,None] + logk_j_k[None,:] - utils.maha(z_ij_k,z_ij_k,matrix_inverse(psd(R_k)).dot(x_cov_k))
            tk = 1.0/T.sqrt(det(psd(R_k)))
            Qk = tk*T.exp( nk2 )
            
            return Qk

        logk=[[]]*odims
        Lambda=[[]]*odims
        for i in xrange(odims):
            # rescale input dimensions by inverse lengthscales
            iL = T.exp(-self.loghyp[i][:idims])
            inp = zeta*iL  # Note this is assuming a diagonal scaling matrix on the kernel

            # predictive mean ( which depends on input covariance )
            B = iL[:,None]*x_cov*iL + T.eye(idims)
            (t,c), updts = theano.scan(fn=M_helper,sequences=[inp,B], non_sequences=[T.exp(2*self.loghyp[i][idims])], strict=True)
            l = T.exp(-0.5*T.sum(inp*t,2))
            lb = l*self.beta[i] # beta should have been precomputed in init_log_likelihood
            mean = T.sum(lb,1)*c;
            mean.name = 'M_%d'%(i)
            M.append(mean)

            # inv(x_cov) times input output covariance
            tiL = t*iL
            v, updts = theano.scan(fn=lambda tiL_k,lb_k,c_k: tiL_k.T.dot(lb_k)*c_k,sequences=[tiL,lb,c])
            V.append(v)
            
            # predictive covariance
            logk[i] = 2*self.loghyp[i][idims] - 0.5*T.sum(inp*inp,2)
            Lambda[i] = iL*iL
            for j in xrange(i+1):
                # This comes from Deisenroth's thesis ( Eqs 2.51- 2.54 )
                z_ij = zeta*Lambda[i] + zeta*Lambda[j]
                R = x_cov*(Lambda[i] + Lambda[j]) + T.eye(idims)
    
                Q,updts = theano.scan(fn=M2_helper, sequences=(logk[i],logk[j],z_ij,R,x_cov))
                Q.name = 'Q_%d%d'%(i,j)

                # Eq 2.55
                m2 = matrix_dot(self.beta[i],Q,self.beta[j])
                if i == j:
                    iKi = self.iK[i].dot(T.eye(self.N))
                    m2 =  m2 - T.sum(iKi*Q,(1,2)) + T.exp(2*self.loghyp[i][idims])
                else:
                    M2[j*odims+i] = m2
                m2.name = 'M2_%d%d'%(i,j)
                M2[i*odims+j] = m2

        M = T.stacklists(M).T
        V = T.stacklists(V).transpose(1,2,0)
        M2 = T.stacklists(M2).T
        S = M2 - (M[:,:,None]*M[:,None,:]).flatten(2)

        utils.print_with_stamp('Compiling mean and variance of prediction',self.name)
        self.predict_ = F([x_mean,x_cov],(M,S,V), name='%s>predict_'%(self.name), profile=self.profile, mode=self.compile_mode)

        # compile the derivatives wrt the evaluation point
        dMdm = T.jacobian(M.flatten(),x_mean)
        dVdm = T.jacobian(V.flatten(),x_mean)
        dSdm = T.jacobian(S.flatten(),x_mean)
        dMds = T.jacobian(M.flatten(),x_cov)
        dVds = T.jacobian(V.flatten(),x_cov)
        dSds = T.jacobian(S.flatten(),x_cov)

        utils.print_with_stamp('Compiling derivatives of mean and variance of prediction',self.name)
        self.predict_d_ = F([x_mean,x_cov], (M,dMdm,dMds,S,dSdm,dSds,V,dVdm,dVds), name='%s>predict_d_'%(self.name), profile=self.profile, mode=self.compile_mode)

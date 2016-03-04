import os,sys
from functools import partial

import numpy as np
import theano
import theano.tensor as T
from scipy.optimize import minimize, basinhopping
from scipy.cluster.vq import kmeans
from theano import function as F, shared as S
from theano.misc.pkl_utils import dump as t_dump, load as t_load
from theano.tensor.nlinalg import matrix_dot
from theano.sandbox.linalg import psd,matrix_inverse,det,cholesky
from theano.tensor.slinalg import solve_lower_triangular, solve_upper_triangular

import cov
import SNRpenalty
import utils
from utils import gTrig, gTrig2,gTrig_np, wrap_params, unwrap_params

class GP(object):
    def __init__(self, X_dataset=None, Y_dataset=None, name='GP', idims=None, odims=None, profile=False, uncertain_inputs=False, hyperparameter_gradients=False, snr_penalty=SNRpenalty.SEard):
        # theano options
        self.profile= profile
        self.compile_mode = theano.compile.get_default_mode()#.excluding('scanOp_pushout_seqs_ops')

        # GP options
        self.min_method = "L-BFGS-B"
        self.state_changed = False
        self.should_recompile = False
        self.uncertain_inputs = uncertain_inputs
        self.hyperparameter_gradients = hyperparameter_gradients
        self.snr_penalty = snr_penalty
        
        # dimension related variables
        if X_dataset is None:
            if idims is None:
                raise ValueError('You need to either provide X_dataset (n x idims numpy array) or a value for idims') 
            self.D = idims
        else:
            self.D = X_dataset.shape[1]

        if Y_dataset is None:
            if odims is None:
                raise ValueError('You need to either provide Y_dataset (n x odims numpy array) or a value for odims') 
            self.E = odims
        else:
            self.E = Y_dataset.shape[1]

        #symbolic varianbles
        self.X_ = None; self.Y_ = None; self.loghyp_=None
        self.loghyp = None
        self.X = None; self.Y = None
        self.K = None; self.L = None; self.beta = None;
        self.nlml = None

        # compiled functions
        self.predict_ = None
        self.predict_d_ = None

        # name of this class for printing command line output and saving
        self.name = name
        # filename for saving
        self.filename = '%s_%d_%d_%s_%s'%(self.name,self.D,self.E,theano.config.device,theano.config.floatX)
        
        try:
            # try loading from pickled file, to avoid recompiling
            self.load()
            if (X_dataset is not None and Y_dataset is not None):
                self.set_dataset(X_dataset,Y_dataset)
        except IOError:
            utils.print_with_stamp('Initiliasing new GP regressor [ Could not open %s.zip ]'%(self.filename),self.name)
            # initialize the class if no pickled version is available
            if X_dataset is not None and Y_dataset is not None:
                self.set_dataset(X_dataset,Y_dataset)
                self.init_log_likelihood()
        
        utils.print_with_stamp('Finished initialising GP regressor',self.name)
    
    def set_dataset(self,X_dataset,Y_dataset):
        #utils.print_with_stamp('Updating GP dataset',self.name)
        # ensure we don't change the number of input and output dimensions ( the number of samples can change)
        assert X_dataset.shape[0] == Y_dataset.shape[0], "X_dataset and Y_dataset must have the same number of rows"
        if self.X_ is not None:
            assert self.X_.shape[1] == X_dataset.shape[1]
        if self.Y_ is not None:
            assert self.Y_.shape[1] == Y_dataset.shape[1]

        # first, assign the numpy arrays to class members
        self.X_ = X_dataset.astype(theano.config.floatX )
        self.Y_ = Y_dataset.astype(theano.config.floatX)
        # dims = non_angle_dims + 2*angle_dims
        self.N = self.X_.shape[0]
        self.D = X_dataset.shape[1]
        self.E = Y_dataset.shape[1]

        # now we create symbolic shared variables
        if self.X is None:
            self.X = S(self.X_,name='%s>X'%(self.name),borrow=True)
        else:
            self.X.set_value(self.X_,borrow=True)
        if self.Y is None:
            self.Y = S(self.Y_,name='%s>Y'%(self.name),borrow=True)
        else:
            self.Y.set_value(self.Y_,borrow=True)

        # init log hyperparameters
        self.init_loghyp()

        # we should be saving, since we updated the trianing dataset
        self.state_changed = True

    def init_loghyp(self,reinit=False):
        idims = self.D; odims = self.E; 
        # initialize the loghyperparameters of the gp ( this code supports squared exponential only, at the moment)
        if self.loghyp_ is None:
            reinit = True
            self.loghyp_ = np.zeros((odims,idims+2))

        if reinit:
            X_ = self.X_; Y_ = self.Y_
            self.loghyp_[:,:idims] = X_.std(0,ddof=1)
            self.loghyp_[:,idims] = Y_.std(0,ddof=1)
            self.loghyp_[:,idims+1] = 0.1*self.loghyp_[:,idims]
            self.loghyp_ = np.log(self.loghyp_)

        self.set_loghyp(self.loghyp_)

    def set_loghyp(self, loghyp):
        loghyp = loghyp.reshape(self.loghyp_.shape).astype(theano.config.floatX)
        self.loghyp_ = loghyp
        # this creates one theano shared variable for the log hyperparameters
        if self.loghyp is None:
            self.loghyp = S(self.loghyp_,name='%s>loghyp'%(self.name),borrow=True)
        else:
            self.loghyp.set_value(self.loghyp_,borrow=True)

    def get_params(self, symbolic=True):
        if symbolic:
            retvars = [self.loghyp,self.X,self.Y]
            return retvars
        else:
            retvars = [self.loghyp_,self.X_, self.Y_]
            return retvars

    def set_params(self, params):
        loghyp = params[0]
        inputs = params[1]
        targets = params[2]
        self.set_dataset(inputs,targets)
        self.set_loghyp(loghyp)

    def init_log_likelihood(self):
        utils.print_with_stamp('Initialising expression graph for full GP log likelihood',self.name)
        idims = self.D
        odims = self.E

        self.kernel_func = [[]]*odims
        self.K = [[]]*odims
        self.L = [[]]*odims
        self.beta = [[]]*odims
        nlml = [[]]*odims
        dnlml = [[]]*odims
        covs = (cov.SEard, cov.Noise)
        N = self.X.shape[0].astype(theano.config.floatX)
        for i in xrange(odims):
            # initialise the (before compilation) kernel function
            loghyps = (self.loghyp[i,:idims+1],self.loghyp[i,idims+1])
            self.kernel_func[i] = partial(cov.Sum, loghyps, covs)

            # We initialise the kernel matrices (one for each output dimension)
            self.K[i] = self.kernel_func[i](self.X)
            self.L[i] = cholesky(self.K[i])
            Yc = solve_lower_triangular(self.L[i],self.Y[:,i])
            self.beta[i] = solve_upper_triangular(self.L[i].T,Yc)

            # And finally, the negative log marginal likelihood ( again, one for each dimension; although we could share
            # the loghyperparameters across all output dimensions and train the GPs jointly)
            nlml[i] = 0.5*(Yc.T.dot(Yc) + 2*T.sum(T.log(T.diag(self.L[i]))) + N*T.log(2*np.pi) )

        nlml = T.stack(nlml)
        if self.snr_penalty is not None:
            penalty_params = {'log_snr': np.log(1000), 'log_ls': np.log(100), 'log_std': T.log(self.X.std(0)*(N/(N-1.0))), 'p': 30}
            nlml += self.snr_penalty(self.loghyp)

        # Compute the gradients for the sum of nlml for all output dimensions
        dnlml = T.jacobian(nlml.sum(),self.loghyp)

        # Compile the theano functions
        utils.print_with_stamp('Compiling full GP log likelihood',self.name)
        self.nlml = F((),nlml,name='%s>nlml'%(self.name), profile=self.profile, mode=self.compile_mode, allow_input_downcast=True)
        utils.print_with_stamp('Compiling jacobian of full GP log likelihood',self.name)
        self.dnlml = F((),(nlml,dnlml),name='%s>dnlml'%(self.name), profile=self.profile, mode=self.compile_mode, allow_input_downcast=True)
        self.state_changed = True # for saving
    
    def init_predict(self, derivs=False):
        utils.print_with_stamp('Initialising expression graph for prediction',self.name)
        # Note that this handles n_samples inputsa
        # initialize variable for input vector ( input mean in the case of uncertain inputs )
        mx = T.vector('mx')
        Sx = T.matrix('Sx')
        # initialize variable for input covariance 
        input_vars = [mx] if not self.uncertain_inputs else [mx,Sx]
        
        # get prediction
        output_vars = self.predict_symbolic(mx,Sx)
        prediction = []
        for o in output_vars:
            if o is not None:
                prediction.append(o)
        
        if not derivs:
            # compile prediction
            utils.print_with_stamp('Compiling mean and variance of prediction',self.name)
            self.predict_ = F(input_vars,prediction,name='%s>predict_'%(self.name), profile=self.profile, mode=self.compile_mode, allow_input_downcast=True)
            self.state_changed = True # for saving
        else:
            # compute the derivatives wrt the input vector ( or input mean in the case of uncertain inputs)
            prediction_derivatives = list(prediction)
            for p in prediction:
                prediction_derivatives.append( T.jacobian( p.flatten(), mx ).flatten(2) )
            if self.uncertain_inputs:
                # compute the derivatives wrt the input covariance
                for p in prediction:
                    prediction_derivatives.append( T.jacobian( p.flatten(), Sx ).flatten(2) )
            if self.hyperparameter_gradients:
                # compute the derivatives wrt the hyperparameters
                for p in prediction:
                    prediction_derivatives.append( T.jacobian( p.flatten(), self.loghyp ) )
                    prediction_derivatives.append( T.jacobian( p.flatten(), self.X ) )
                    prediction_derivatives.append( T.jacobian( p.flatten(), self.Y ) )
            #prediction_derivatives contains  [p1, p2, ..., pn, dp1/dmx, dp2/dmx, ..., dpn/dmx, dp1/dSx, dp2/dSx, ..., dpn/dSx, dp1/dloghyp, dp2/dloghyp, ..., dpn/loghyp]
            utils.print_with_stamp('Compiling mean and variance of prediction with jacobians',self.name)
            self.predict_d_ = F(input_vars, prediction_derivatives, name='%s>predict_d_'%(self.name), profile=self.profile, mode=self.compile_mode, allow_input_downcast=True)
            self.state_changed = True # for saving

    def predict_symbolic(self,mx,Sx):
        odims = self.E

        # compute the mean and variance for each output dimension
        mean = [[]]*odims
        variance = [[]]*odims
        for i in xrange(odims):
            k = self.kernel_func[i](mx[None,:],self.X)
            mean[i] = k.dot(self.beta[i])
            kc = solve_lower_triangular(self.L[i],k.flatten())
            variance[i] = self.kernel_func[i](mx[None,:],all_pairs=False) - kc.dot(kc)  #TODO verify this computation

        # reshape output variables
        M = T.stack(mean).T.flatten()
        S = T.diag(T.stack(variance).T.flatten())

        return M,S
    
    def predict(self,mx,Sx = None, derivs=False):
        predict = None
        if not derivs:
            if self.predict_ is None:
                self.init_predict(derivs=derivs)
            predict = self.predict_
        else:
            if self.predict_d_ is None:
                self.init_predict(derivs=derivs)
            predict = self.predict_d_

        odims = self.E
        idims = self.D
        res = None
        if self.uncertain_inputs:
            if Sx is None:
                Sx = np.zeros((idims,idims))
            res = predict(mx, Sx)
        else:
            res = predict(mx)
        return res
    
    def loss(self,loghyp):
        self.set_loghyp(loghyp)
        nlml,dnlml = self.dnlml()
        nlml = nlml.sum()
        dnlml = dnlml.flatten()
        # on a 64bit system, scipy optimize complains if we pass a 32 bit float
        res = (nlml.astype(np.float64), dnlml.astype(np.float64))
        #utils.print_with_stamp('%s, %s'%(str(res[0]),str(np.exp(loghyp))),self.name,True)
        utils.print_with_stamp('%s'%(str(res[0])),self.name,True)
        return res

    def train(self):
        utils.print_with_stamp('Current hyperparameters:',self.name)
        loghyp0 = self.loghyp_.copy()
        print (loghyp0)
        utils.print_with_stamp('nlml: %s'%(np.array(self.nlml())),self.name)
        try:
            opt_res = minimize(self.loss, loghyp0, jac=True, method=self.min_method, tol=1e-9, options={'maxiter': 500})
        except ValueError:
            opt_res = minimize(self.loss, loghyp0, jac=True, method='CG', tol=1e-9, options={'maxiter': 500})
        print ''
        loghyp = opt_res.x.reshape(self.loghyp_.shape)
        self.state_changed = not np.allclose(loghyp0,loghyp,1e-6,1e-9)
        self.set_loghyp(loghyp)
        utils.print_with_stamp('New hyperparameters:',self.name)
        print (self.loghyp_)
        utils.print_with_stamp('nlml: %s'%(np.array(self.nlml())),self.name)

    def load(self):
        with open(self.filename+'.zip','rb') as f:
            utils.print_with_stamp('Loading compiled GP with %d inputs and %d outputs'%(self.D,self.E),self.name)
            state = t_load(f)
            self.set_state(state)
        self.state_changed = False
    
    def save(self):
        sys.setrecursionlimit(100000)
        if self.state_changed:
            with open(self.filename+'.zip','wb') as f:
                utils.print_with_stamp('Saving compiled GP with %d inputs and %d outputs'%(self.D,self.E),self.name)
                t_dump(self.get_state(),f,2)
            self.state_changed = False

    def set_state(self,state):
        self.X = state[0]
        self.Y = state[1]
        self.loghyp = state[2]
        self.K = state[3]
        self.L = state[4]
        self.beta = state[5]
        self.set_dataset(state[6],state[7])
        self.set_loghyp(state[8])
        self.nlml = state[9]
        self.dnlml = state[10]
        self.predict_ = state[11]
        self.predict_d_ = state[12]
        self.kernel_func = state[13]

    def get_state(self):
        return [self.X,self.Y,self.loghyp,self.K,self.L,self.beta,self.X_,self.Y_,self.loghyp_,self.nlml,self.dnlml,self.predict_,self.predict_d_,self.kernel_func]

class GP_UI(GP):
    def __init__(self, X_dataset=None, Y_dataset=None, name = 'GP_UI', idims=None, odims=None, profile=False, uncertain_inputs=True, hyperparameter_gradients=False):
        super(GP_UI, self).__init__(X_dataset,Y_dataset,name=name,idims=idims,odims=odims,profile=profile,uncertain_inputs=True,hyperparameter_gradients=hyperparameter_gradients)

    def predict_symbolic(self,mx,Sx):
        idims = self.D
        odims = self.E

        #centralize inputs 
        zeta = self.X - mx
        
        # initialize some variables
        sf2 = T.exp(2*self.loghyp[:,idims])
        eyeE = T.tile(T.eye(idims),(odims,1,1))
        lscales = T.exp(self.loghyp[:,:idims])
        iL = eyeE/lscales[:,:,None]

        # predictive mean
        inp = iL.dot(zeta.T).transpose(0,2,1) 
        iLdotSx = iL.dot(Sx)
        B = T.stack([iLdotSx[i].dot(iL[i]) for i in xrange(odims)]) + T.eye(idims)   #TODO vectorize this
        t = T.stack([inp[i].dot(matrix_inverse(B[i])) for i in xrange(odims)])      # E x N x D
        c = sf2/T.sqrt(T.stack([det(B[i]) for i in xrange(odims)]))
        l = T.exp(-0.5*T.sum(inp*t,2))
        lb = l*self.beta # beta should have been precomputed in init_log_likelihood # E x N dot E x N
        M = T.sum(lb,1)*c
        
        # inv(Sx) times input output covariance
        tiL = T.stack([t[i].dot(iL[i]) for i in xrange(odims)])
        V = T.stack([tiL[i].T.dot(lb[i]) for i in xrange(odims)]).T*c

        # predictive covariance
        logk = 2*self.loghyp[:,None,idims] - 0.5*T.sum(inp*inp,2)
        Lambda = iL**2
        R = T.dot((Lambda[:,None,:,:] + Lambda).transpose(0,1,3,2),Sx.T).transpose(0,1,3,2) + T.eye(idims)
        z_= Lambda.dot(zeta.T).transpose(0,2,1) 
        M2 = [[]]*(odims**2) # second moment
        N = self.X.shape[0]
        for i in xrange(odims):
            for j in xrange(i+1):
                # This comes from Deisenroth's thesis ( Eqs 2.51- 2.54 )
                Rij = R[i,j]
                n2 = logk[i][:,None] + logk[j][None,:] + utils.maha(z_[i],-z_[j],0.5*matrix_inverse(Rij).dot(Sx))
                Q = T.exp( n2 )/T.sqrt(det(Rij))
                # Eq 2.55
                m2 = matrix_dot(self.beta[i], Q, self.beta[j])
                if i == j:
                    iKi = solve_upper_triangular(self.L[i].T, solve_lower_triangular(self.L[i],T.eye(N)))
                    m2 =  m2 - T.sum(iKi*Q) + sf2[i]
                else:
                    M2[j*odims+i] = m2
                M2[i*odims+j] = m2

        M2 = T.stack(M2).T
        S = M2.reshape((odims,odims))
        S = S - T.outer(M,M)

        return M,S,V
        
class SPGP(GP):
    def __init__(self, X_dataset, Y_dataset, name = 'SPGP',profile=False, n_basis = 100, uncertain_inputs=False, hyperparameter_gradients=False):
        self.X_sp_ = None # inducing inputs (data)
        self.X_sp = None # inducing inputs (symbolic variable)
        self.nlml_sp = None
        self.dnlml_sp = None
        self.beta_sp = None
        self.Lmm = None
        self.Amm = None
        self.should_recompile = False
        self.n_basis = n_basis
        # intialize parent class params
        super(SPGP, self).__init__(X_dataset,Y_dataset,name=name,profile=profile,uncertain_inputs=uncertain_inputs,hyperparameter_gradients=hyperparameter_gradients)

    def init_pseudo_inputs(self):
        assert self.N > self.n_basis, "Dataset must have more than n_basis [ %n ] to enable inference with sparse pseudo inputs"%(self.n_basis)
        self.should_recompile = True
        # pick initial cluster centers from dataset
        self.X_sp_ = utils.kmeanspp(self.X_,self.n_basis).astype(theano.config.floatX)

        # perform kmeans to get initial cluster centers
        utils.print_with_stamp('Initialising pseudo inputs',self.name)
        self.X_sp_, dist = kmeans(self.X_, self.X_sp_, iter=200,thresh=1e-12)
        # initialize symbolic tensor variable if necessary
        if self.X_sp is None:
            self.X_sp = S(self.X_sp_,name='%s>X_sp'%(self.name),borrow=True)
        else:
            self.X_sp.set_value(self.X_sp_,borrow=True)

    def set_dataset(self,X_dataset,Y_dataset):
        # set the dataset on the parent class
        super(SPGP, self).set_dataset(X_dataset,Y_dataset)
        if self.N <= self.n_basis:
            utils.print_with_stamp('Dataset is not large enough for using pseudo inputs. Training full GP.',self.name)
            self.X_sp = None
            self.X_sp_ = None
            self.nlml_sp = None
            self.dnlml_sp = None
            self.beta_sp = None
            self.Lmm = None
            self.Amm = None
            self.should_recompile = False

        if self.N > self.n_basis and self.X_sp is None:
            utils.print_with_stamp('Dataset is large enough for using pseudo inputs. You should reinitiialise the log likelihood and predictions.',self.name)
            # init the shared variable for the pseudo inputs
            self.init_pseudo_inputs()
            self.should_recompile = True
        
    def init_log_likelihood(self):
        # initialize the log likelihood of the GP class
        super(SPGP, self).init_log_likelihood()
        # here nlml and dnlml have already been innitialised, sow e can replace nlml and dnlml
        # only if we have enough data to train the pseudo inputs ( i.e. self.N > self.n_basis)
        if self.N > self.n_basis:
            utils.print_with_stamp('Initialising FITC log likelihood',self.name)
            self.should_recompile = False
            odims = self.E
            idims = self.D
            # initialize the log likelihood of the sparse FITC approximation
            self.Lmm = [[]]*odims
            self.Amm = [[]]*odims
            self.beta_sp = [[]]*odims
            nlml_sp = [[]]*odims
            dnlml_sp = [[]]*odims
            for i in xrange(odims):
                X_sp_i = self.X_sp  # TODO allow for different pseudo inputs for each dimension
                ll = T.exp(self.loghyp[i,:idims])
                sf2 = T.exp(2*self.loghyp[i,idims])
                sn2 = T.exp(2*self.loghyp[i,idims+1])
                N = self.X.shape[0]
                M = X_sp_i.shape[0]

                Kmm = self.kernel_func[i](X_sp_i) + 1e-6*T.eye(self.X_sp.shape[0])
                Kmn = self.kernel_func[i](X_sp_i, self.X)
                Lmm = cholesky(Kmm)
                Lmn = solve_lower_triangular(Lmm,Kmn)
                diagQnn =  T.diag(Lmn.T.dot(Lmn))

                # Gamma = diag(Knn - Qnn) + sn2*I
                Gamma = sf2 - diagQnn + sn2
                Gamma_inv = 1.0/Gamma

                # these operations are done to avoid inverting K_sp = (Qnn+Gamma)
                Kmn_ = Kmn*T.sqrt(Gamma_inv)                    # Kmn_*Gamma^-.5
                Yi = self.Y[:,i]*(T.sqrt(Gamma_inv))            # Gamma^-.5* Y
                Bmm = Kmm + (Kmn_).dot(Kmn_.T)                  # Kmm + Kmn * Gamma^-1 * Knm
                Amm = cholesky(Bmm)

                Yci = solve_lower_triangular(Amm,Kmn_.dot(Yi) )
                self.Lmm[i] = Lmm
                self.Amm[i] = Amm
                self.beta_sp[i] = solve_upper_triangular(Amm.T,Yci)

                log_det_K_sp = T.sum(T.log(Gamma)) - 2*T.sum(T.log(T.diag(Lmm))) + 2*T.sum(T.log(T.diag(Amm)))

                nlml_sp[i] = 0.5*( Yi.dot(Yi) - Yci.dot(Yci) + log_det_K_sp + N*np.log(2*np.pi) )

            nlml_sp = T.stack(nlml_sp)
            # TODO include the log hyperparameters in the optimization
            # TODO give the optiion for separate inducing inputs for every output dimension
            dnlml_sp = T.jacobian(nlml_sp.sum(),self.X_sp)

            # Compile the theano functions
            utils.print_with_stamp('Compiling FITC log likelihood',self.name)
            self.nlml_sp = F((),nlml_sp,name='%s>nlml_sp'%(self.name), profile=self.profile, mode=self.compile_mode, allow_input_downcast=True)
            utils.print_with_stamp('Compiling jacobian of FITC log likelihood',self.name)
            self.dnlml_sp = F((),(nlml_sp,dnlml_sp),name='%s>dnlml_sp'%(self.name), profile=self.profile, mode=self.compile_mode, allow_input_downcast=True)
            
    def predict_symbolic(self,mx,Sx):
        if self.N < self.n_basis:
            # stick with the full GP
            return super(SPGP, self).predict_symbolic()

        odims = self.E

        # compute the mean and variance for each output dimension
        mean = [[]]*odims
        variance = [[]]*odims
        for i in xrange(odims):
            k = self.kernel_func[i](mx[None,:],self.X_sp).flatten()
            mean[i] = k.dot(self.beta_sp[i])
            kL = solve_lower_triangular(self.Lmm[i],k)
            kA = solve_lower_triangular(self.Amm[i],k)
            variance[i] = self.kernel_func[i](mx[None,:],all_pairs=False) - kL.dot(kL) -  kA.dot(kA)

        # reshape the prediction output variables
        M = T.stack(mean).T.flatten()
        S = T.diag(T.stack(variance).T.flatten())

        return M,S

    def set_X_sp(self, X_sp):
        X_sp = X_sp.reshape(self.X_sp_.shape).astype(theano.config.floatX)
        np.copyto(self.X_sp_,X_sp)
        self.X_sp.set_value(self.X_sp_,borrow=True)
    
    def get_params(self, symbolic=True):
        retvars = super(SPGP,self).get_params(symbolic)
        if symbolic:
            retvars.append(self.X_sp)
        else:
            retvars.append(self.X_sp_)
        return retvars

    def loss_sp(self,X_sp):
        self.set_X_sp(X_sp)
        res = self.dnlml_sp()
        nlml = np.array(res[0]).sum()
        dnlml = np.array(res[1]).flatten()
        # on a 64bit system, scipy optimize complains if we pass a 32 bit float
        res = (nlml.astype(np.float64),dnlml.astype(np.float64))
        utils.print_with_stamp('%s'%(str(res[0])),self.name,True)
        return res

    def train(self):
        # train the full GP
        super(SPGP, self).train()

        if self.N > self.n_basis:
            # train the pseudo input locations
            utils.print_with_stamp('nlml SP: %s'%(np.array(self.nlml_sp())),self.name)
            opt_res = minimize(self.loss_sp, self.X_sp_, jac=True, method=self.min_method, tol=1e-9, options={'maxiter': 1000})
            print ''
            X_sp = opt_res.x.reshape(self.X_sp_.shape)
            self.set_X_sp(X_sp)
            utils.print_with_stamp('nlml SP: %s'%(np.array(self.nlml_sp())),self.name)

    def set_state(self,state):
        self.X_sp = state[-7]
        self.X_sp_ = state[-6]
        self.beta_sp = state[-5]
        self.Lmm = state[-4]
        self.Amm = state[-3]
        self.nlml_sp = state[-2]
        self.dnlml_sp = state[-1]
        super(SPGP,self).set_state(state[:-7])
        self.should_recompile = False

    def get_state(self):
        state = super(SPGP,self).get_state()
        state.append(self.X_sp)
        state.append(self.X_sp_)
        state.append(self.beta_sp)
        state.append(self.Lmm)
        state.append(self.Amm)
        state.append(self.nlml_sp)
        state.append(self.dnlml_sp)
        return state

class SPGP_UI(SPGP,GP_UI):
    def __init__(self, X_dataset, Y_dataset, name = 'SPGP_UI',profile=False, n_basis = 100, hyperparameter_gradients=False):
        super(SPGP_UI, self).__init__(X_dataset,Y_dataset,name=name,profile=profile,n_basis=n_basis,uncertain_inputs=True,hyperparameter_gradients=hyperparameter_gradients)

    def predict_symbolic(self,mx,Sx):
        if self.N < self.n_basis:
            # stick with the full GP
            return GP_UI.predict_symbolic(self,mx,Sx)

        idims = self.D
        odims = self.E

        #centralize inputs 
        zeta = self.X - mx
        
        # initialize some variables
        sf2 = T.exp(2*self.loghyp[:,idims])
        eyeE = T.tile(T.eye(idims),(odims,1,1))
        lscales = T.exp(self.loghyp[:,:idims])
        iL = eyeE/lscales[:,:,None]

        # predictive mean
        inp = iL.dot(zeta.T).transpose(0,2,1) 
        iLdotSx = iL.dot(Sx)
        B = T.stack([iLdotSx[i].dot(iL[i]) for i in xrange(odims)]) + T.eye(idims)   #TODO vectorize this
        t = T.stack([inp[i].dot(matrix_inverse(B[i])) for i in xrange(odims)])      # E x N x D
        c = sf2/T.sqrt(T.stack([det(B[i]) for i in xrange(odims)]))
        l = T.exp(-0.5*T.sum(inp*t,2))
        lb = l*self.beta_sp # beta_sp should have been precomputed in init_log_likelihood # E x N dot E x N
        M = T.sum(lb,1)*c
        
        # inv(Sx) times input output covariance
        tiL = T.stack([t[i].dot(iL[i]) for i in xrange(odims)])
        V = T.stack([tiL[i].T.dot(lb[i]) for i in xrange(odims)]).T*c

        # predictive covariance
        logk = 2*self.loghyp[:,None,idims] - 0.5*T.sum(inp*inp,2)
        Lambda = iL**2
        R = T.dot((Lambda[:,None,:,:] + Lambda).transpose(0,1,3,2),Sx.T).transpose(0,1,3,2) + T.eye(idims)
        z_= Lambda.dot(zeta.T).transpose(0,2,1) 
        M2 = [[]]*(odims**2) # second moment
        N = self.X.shape[0]
        for i in xrange(odims):
            for j in xrange(i+1):
                # This comes from Deisenroth's thesis ( Eqs 2.51- 2.54 )
                Rij = R[i,j]
                n2 = logk[i][:,None] + logk[j][None,:] + utils.maha(z_[i],-z_[j],0.5*matrix_inverse(Rij).dot(Sx))
                Q = T.exp( n2 )/T.sqrt(det(Rij))
                # Eq 2.55
                m2 = matrix_dot(self.beta_sp[i], Q, self.beta_sp[j])
                if i == j:
                    iKi = solve_upper_triangular(self.Lmm[i].T, solve_lower_triangular(self.Lmm[i],T.eye(self.n_basis)))
                    iBi = solve_upper_triangular(self.Amm[i].T, solve_lower_triangular(self.Amm[i],T.eye(self.n_basis)))
                    m2 =  m2 - T.sum((iKi - iBi)*Q) + sf2[i]
                else:
                    M2[j*odims+i] = m2.flatten()
                M2[i*odims+j] = m2.flatten()

        M2 = T.stack(M2).T
        S = M2.reshape((odims,odims))
        S = S - T.outer(M,M)

        return M,S,V

# RBF network (GP with uncertain inputs/deterministic outputs)
class RBFGP(GP_UI):
    def __init__(self, X_dataset=None, Y_dataset=None, idims=None, odims=None, sat_func=None, name = 'RBFGP',profile=False):
        self.sat_func = sat_func
        if self.sat_func is not None:
            name += '_sat'
        self.loghyp_full=None
        super(RBFGP, self).__init__(X_dataset,Y_dataset,idims=idims,odims=odims,name=name,profile=profile,hyperparameter_gradients=True)

    def set_state(self,state):
        self.loghyp_full = state[-1]
        super(RBFGP,self).set_state(state[:-1])

    def get_state(self):
        state = super(RBFGP,self).get_state()
        state.append(self.loghyp_full)
        return state
    
    def get_params(self, symbolic=True):
        if symbolic:
            retvars = [self.loghyp_full,self.X,self.Y]
            return retvars
        else:
            retvars = [self.loghyp_,self.X_, self.Y_]
            return retvars
    
    def set_loghyp(self, loghyp):
        loghyp = loghyp.reshape(self.loghyp_.shape).astype(theano.config.floatX)
        
        self.loghyp_ = loghyp
        # this creates one theano shared variable for the log hyperparameters
        if self.loghyp_full is None:
            self.loghyp_full = S(self.loghyp_,name='%s>loghyp'%(self.name),borrow=True)
            self.loghyp = T.zeros_like(self.loghyp_full)
            self.loghyp = T.set_subtensor(self.loghyp[:,:-2], self.loghyp_full[:,:-2])
            self.loghyp = T.set_subtensor(self.loghyp[:,-2:], theano.gradient.disconnected_grad(self.loghyp_full[:,-2:]))
        else:
            self.loghyp_full.set_value(self.loghyp_,borrow=True)

    def predict_symbolic(self,mx,Sx):
        idims = self.D
        odims = self.E

        #centralize inputs 
        zeta = self.X - mx
        
        # initialize some variables
        sf2 = T.exp(2*self.loghyp[:,idims])
        eyeE = T.tile(T.eye(idims),(odims,1,1))
        lscales = T.exp(self.loghyp[:,:idims])
        iL = eyeE/lscales[:,:,None]

        # predictive mean
        inp = iL.dot(zeta.T).transpose(0,2,1) 
        iLdotSx = iL.dot(Sx)
        B = T.stack([iLdotSx[i].dot(iL[i]) for i in xrange(odims)]) + T.eye(idims)   #TODO vectorize this
        t = T.stack([inp[i].dot(matrix_inverse(B[i])) for i in xrange(odims)])      # E x N x D
        c = sf2/T.sqrt(T.stack([det(B[i]) for i in xrange(odims)]))
        l = T.exp(-0.5*T.sum(inp*t,2))
        lb = l*self.beta # beta should have been precomputed in init_log_likelihood # E x N
        M = T.sum(lb,1)*c
        
        # inv(Sx) times input output covariance
        tiL = T.stack([t[i].dot(iL[i]) for i in xrange(odims)])
        V = T.stack([tiL[i].T.dot(lb[i]) for i in xrange(odims)]).T*c

        # predictive covariance
        logk = 2*self.loghyp[:,None,idims] - 0.5*T.sum(inp*inp,2)
        Lambda = iL**2
        R = T.dot((Lambda[:,None,:,:] + Lambda).transpose(0,1,3,2),Sx.T).transpose(0,1,3,2) + T.eye(idims)
        z_= Lambda.dot(zeta.T).transpose(0,2,1) 
        M2 = [[]]*(odims**2) # second moment
        N = self.X.shape[0]
        for i in xrange(odims):
            for j in xrange(i+1):
                # This comes from Deisenroth's thesis ( Eqs 2.51- 2.54 )
                Rij = R[i,j]
                n2 = logk[i][:,None] + logk[j][None,:] + utils.maha(z_[i],-z_[j],0.5*matrix_inverse(Rij).dot(Sx))
                Q = T.exp( n2 )/T.sqrt(det(Rij))
                # Eq 2.55
                m2 = matrix_dot(self.beta[i],Q,self.beta[j].T)
                if i == j:
                    m2 = m2 + 1e-6
                else:
                    M2[j*odims+i] = m2
                M2[i*odims+j] = m2

        M2 = T.stack(M2).T
        S = M2.reshape((odims,odims))
        S = S - T.outer(M,M)

        # apply saturating function to the output if available
        if self.sat_func is not None:
            # compute the joint input output covariance
            M,S,U = self.sat_func(M,S)
            V = V.dot(U)

        return M,S,V

class SSGP(GP):
    ''' Sparse Spectral Gaussian Process Regression'''
    def __init__(self, X_dataset=None, Y_dataset=None, name='SSGP', idims=None, odims=None, profile=False, n_basis=100,  uncertain_inputs=False, hyperparameter_gradients=False):
        self.w = None
        self.w_ = None
        self.sr = None
        self.A = None
        self.Lmm = None
        self.beta_ss = None
        self.nlml_ss = None
        self.dnlml_ss = None
        self.n_basis = n_basis
        super(SSGP, self).__init__(X_dataset,Y_dataset,name=name,idims=idims,odims=odims,profile=profile,uncertain_inputs=uncertain_inputs,hyperparameter_gradients=hyperparameter_gradients)

    def set_state(self,state):
        self.w = state[-8]
        self.w_ = state[-7]
        self.sr = state[-6]
        self.beta_ss = state[-5]
        self.A = state[-4]
        self.Lmm = state[-3]
        self.nlml_ss = state[-2]
        self.dnlml_ss = state[-1]
        super(SSGP,self).set_state(state[:-8])
        self.should_recompile = False

    def get_state(self):
        state = super(SSGP,self).get_state()
        state.append(self.w)
        state.append(self.w_)
        state.append(self.sr)
        state.append(self.beta_ss)
        state.append(self.A)
        state.append(self.Lmm)
        state.append(self.nlml_ss)
        state.append(self.dnlml_ss)
        return state

    def init_log_likelihood(self):
        super(SSGP, self).init_log_likelihood()
        utils.print_with_stamp('Initialising expression graph for sparse spectral log likelihood',self.name)
        idims = self.D
        odims = self.E

        self.A = [[]]*odims
        self.Lmm = [[]]*odims
        self.iA = [[]]*odims
        self.beta_ss = [[]]*odims
        nlml_ss = [[]]*odims
        
        # sample initial unscaled spectral points
        self.w_ = np.random.randn(self.n_basis,odims,idims).astype(theano.config.floatX)
        self.w = S(self.w_,name='%s>w'%(self.name),borrow=True)
        self.sr = (self.w*T.exp(-self.loghyp[:,:idims])).transpose(1,0,2)

        N = self.X.shape[0]
        for i in xrange(odims):
            sr = self.sr[i]
            M = sr.shape[0].astype(theano.config.floatX)
            sf2 = T.exp(2*self.loghyp[i,idims])
            sn2 = T.exp(2*self.loghyp[i,idims+1])
            # sr.T.dot(x) for all sr ( n_basis x *D) and X (N x D). size n_basis x N
            srdotX = sr.dot(self.X.T)
            # convert to sin cos
            phi_f = T.vertical_stack(T.sin(srdotX), T.cos(srdotX))
            
            sf2M = sf2/M
            self.A[i] = sf2M*phi_f.dot(phi_f.T) + sn2*T.eye(2*sr.shape[0])
            self.Lmm[i] = cholesky(self.A[i])
            Yi = self.Y[:,i]
            Yci = solve_lower_triangular(self.Lmm[i],phi_f.dot(Yi))
            self.beta_ss[i] = sf2M*solve_upper_triangular(self.Lmm[i].T,Yci)
            
            nlml_ss[i] = 0.5*( Yi.dot(Yi) - sf2M*Yci.dot(Yci) )/sn2 + T.sum(T.log(T.diag(self.Lmm[i]))) + (0.5*N - M)*T.log(sn2) + 0.5*N*np.log(2*np.pi)
        
        nlml_ss = T.stack(nlml_ss)
        self.beta_ss = T.stack(self.beta_ss)
        if self.snr_penalty is not None:
            penalty_params = {'log_snr': np.log(1000), 'log_ls': np.log(100), 'log_std': T.log(self.X.std(0)*(N/(N-1.0))), 'p': 30}
            nlml_ss += self.snr_penalty(self.loghyp)

        # Compute the gradients for the sum of nlml for all output dimensions
        dnlml_wrt_lh = T.jacobian(nlml_ss.sum(),self.loghyp)
        dnlml_wrt_w = T.jacobian(nlml_ss.sum(),self.w)

        utils.print_with_stamp('Compiling sparse spectral log likelihood',self.name)
        self.nlml_ss = F((),nlml_ss,name='%s>nlml'%(self.name), profile=self.profile, mode=self.compile_mode, allow_input_downcast=True)
        utils.print_with_stamp('Compiling jacobian of sparse spectral log likelihood',self.name)
        self.dnlml_ss = F((),(nlml_ss,dnlml_wrt_lh,dnlml_wrt_w),name='%s>dnlml'%(self.name), profile=self.profile, mode=self.compile_mode, allow_input_downcast=True)

    def set_spectral_samples(self,w):
        w = w.reshape(self.w_.shape).astype(theano.config.floatX)
        np.copyto(self.w_,w)
        self.w.set_value(self.w_,borrow=True)
    
    def get_params(self, symbolic=True):
        retvars = super(SSGP,self).get_params(symbolic)
        if symbolic:
            retvars.append(self.w)
        else:
            retvars.append(self.w_)
        return retvars

    def loss_ss(self, params, parameter_shapes):
        loghyp,w = unwrap_params(params,parameter_shapes)
        self.set_loghyp(loghyp)
        self.set_spectral_samples(w)
        nlml,dnlml_lh,dnlml_sr = self.dnlml_ss()
        nlml = nlml.sum()
        dnlml = wrap_params([dnlml_lh,dnlml_sr])
        # on a 64bit system, scipy optimize complains if we pass a 32 bit float
        res = (nlml.astype(np.float64), dnlml.astype(np.float64))
        #utils.print_with_stamp('%s, %s'%(str(res[0]),str(np.exp(loghyp))),self.name,True)
        utils.print_with_stamp('%s'%(str(res[0])),self.name,True)
        return res

    def train(self):
        # train the full GP ( if dataset too large, take a random subsample)
        X_full = None
        Y_full = None
        n_subsample = 512
        if self.X_.shape[0] > n_subsample:
            utils.print_with_stamp('Training full gp with random subsample of size %d'%(n_subsample),self.name)
            idx = np.arange(self.X_.shape[0]); np.random.shuffle(idx); idx= idx[:n_subsample]
            X_full = self.X_
            Y_full = self.Y_
            self.set_dataset(self.X_[idx],self.Y_[idx])

        super(SSGP, self).train()

        if X_full is not None:
            # restore full dataset for SSGP training
            utils.print_with_stamp('Restoring full dataset',self.name)
            self.set_dataset(X_full,Y_full)

        idims = self.D
        odims = self.E

        # initialize spectral samples
        nlml = self.nlml_ss()
        best_w = self.w_.copy()
        # try a couple spectral samples and pick the one with the lowest nlml
        for i in xrange(100):
            self.set_spectral_samples( np.random.randn(self.n_basis,odims,idims) )
            nlml_i = self.nlml_ss()
            if np.all(nlml_i < nlml):
                nlml = nlml_i
                best_w = self.w_.copy()
        self.set_spectral_samples( best_w )

        # train the pseudo input locations
        utils.print_with_stamp('nlml SS: %s'%(np.array(self.nlml_ss())),self.name)
        # wrap loghyp plus sr (save shapes)
        p0 = [self.loghyp_,self.w_]
        parameter_shapes = [p.shape for p in p0]
        opt_res = minimize(self.loss_ss, wrap_params(p0), args=parameter_shapes, jac=True, method=self.min_method, tol=1e-9, options={'maxiter': 500})
        print ''
        loghyp,w = unwrap_params(opt_res.x,parameter_shapes)
        self.set_loghyp(loghyp)
        self.set_spectral_samples(w)
        utils.print_with_stamp('nlml SS: %s'%(np.array(self.nlml_ss())),self.name)

    def predict_symbolic(self,mx,Sx):
        odims = self.E
        idims = self.D

        # compute the mean and variance for each output dimension
        mean = [[]]*odims
        variance = [[]]*odims
        for i in xrange(odims):
            sr = self.sr[i]
            M = sr.shape[0]
            sf2 = T.exp(2*self.loghyp[i,idims])
            sn2 = T.exp(2*self.loghyp[i,idims+1])
            # sr.T.dot(x) for all sr and X. size n_basis x N
            srdotX = sr.dot(mx)
            # convert to sin cos
            phi_x = T.concatenate([ T.sin(srdotX), T.cos(srdotX) ])

            mean[i] = phi_x.T.dot(self.beta_ss[i])
            phi_x_L = solve_lower_triangular(self.Lmm[i],phi_x)
            variance[i] = sn2*(1 + (sf2/M)*phi_x_L.dot( phi_x_L ))

        # reshape output variables
        M = T.stack(mean).T.flatten()
        S = T.diag(T.stack(variance).T.flatten())

        return M,S

class SSGP_UI(SSGP, GP_UI):
    ''' Sparse Spectral Gaussian Process Regression with Uncertain Inputs'''
    def __init__(self, X_dataset=None, Y_dataset=None, name='SSGP_UI', idims=None, odims=None, profile=False, n_basis=100,  uncertain_inputs=True, hyperparameter_gradients=False):
        SSGP.__init__(self,X_dataset,Y_dataset,name=name,idims=idims,odims=odims,profile=profile,n_basis=n_basis,uncertain_inputs=True,hyperparameter_gradients=hyperparameter_gradients)

    def predict_symbolic(self,mx,Sx):
        if self.N < self.n_basis:
            # stick with the full GP
            return GP_UI.predict_symbolic(self,mx,Sx)

        idims = self.D
        odims = self.E
        
        # precompute some variables
        Ms = self.sr.shape[1]
        srdotx = self.sr.dot(mx)
        srdotSx = self.sr.dot(Sx) 
        srdotSxdotsr = T.sum(srdotSx*self.sr,2)
        e = T.exp(-0.5*srdotSxdotsr)
        cos_srdotx = T.cos(srdotx)
        sin_srdotx = T.sin(srdotx)
        cos_srdotx_e = T.cos(srdotx)*e
        sin_srdotx_e = T.sin(srdotx)*e

        # compute the mean vector
        mphi = T.horizontal_stack( sin_srdotx_e, cos_srdotx_e ) # E x 2*Ms
        M = T.sum( mphi*self.beta_ss, 1)

        # inv(s) times input output covariance
        c = T.concatenate([ mx[:,None]*sin_srdotx_e[:,None,:] + srdotSx.transpose(0,2,1)*cos_srdotx_e[:,None,:], mx[:,None]*cos_srdotx_e[:,None,:] - srdotSx.transpose(0,2,1)*sin_srdotx_e[:,None,:] ], axis=2) # E x D x 2*Ms
        v = T.sum( c*(self.beta_ss[:,None,:]), 2 ).T - T.outer(mx,M)
        V = matrix_inverse(Sx).dot(v)
        
        # TODO vectorize this operation
        M2 = [[]]*(odims**2) # second moment
        for i in xrange(odims):
            # initalize some variables
            sf2 = T.exp(2*self.loghyp[i,idims])
            sn2 = T.exp(2*self.loghyp[i,idims+1])
            # predictive covariance
            for j in xrange(i+1):
                # compute the second moments of the spectral feature vectors
                '''
                siSxsj = srdotSx[i].dot(self.sr[j].T) #Ms x Ms
                sijSxsij = -0.5*(srdotSxdotsr[i,:,None] + srdotSxdotsr[j,None,:]) 
                em =  T.exp(sijSxsij+siSxsj)      # MsxMs
                ep =  T.exp(sijSxsij-siSxsj)     # MsxMs
                si = sin_srdotx[i]       # Msx1
                ci = cos_srdotx[i]       # Msx1   
                sj = sin_srdotx[j]       # Msx1
                cj = cos_srdotx[j]       # Msx1
                sicj = T.outer(si,cj)    # MsxMs
                cisj = T.outer(ci,sj)    # MsxMs
                sisj = T.outer(si,sj)    # MsxMs
                cicj = T.outer(ci,cj)    # MsxMs
                sm = (sicj-cisj)*em
                sp = (sicj+cisj)*ep
                cm = (sisj+cicj)*em
                cp = (cicj-sisj)*ep
                '''
                srdotx_m_ij = (srdotx[i][:,None] - srdotx[j][None,:])   # MsxMs
                srdotx_p_ij = (srdotx[i][:,None] + srdotx[j][None,:])   # MsxMs
                sr_m_ij = (self.sr[i][:,None,:] - self.sr[j][None,:,:])           # MsxMsxD
                sr_p_ij = (self.sr[i][:,None,:] + self.sr[j][None,:,:])           # MsxMsxD
                em =  T.exp(-0.5*T.sum(sr_m_ij.dot(Sx)*sr_m_ij,2))      # MsxMs
                ep =  T.exp(-0.5*T.sum(sr_p_ij.dot(Sx)*sr_p_ij,2))      # MsxMs
                sm = T.sin( srdotx_m_ij )*em
                sp = T.sin( srdotx_p_ij )*ep
                cm = T.cos( srdotx_m_ij )*em
                cp = T.cos( srdotx_p_ij )*ep
                
                
                # Populate the second moment matrix of the feature vector
                Qij = T.zeros((2*Ms,2*Ms))
                Qij = T.set_subtensor( Qij[:Ms,:Ms] , cm - cp )
                Qij = T.set_subtensor( Qij[:Ms,Ms:] , sm + sp )
                Qij = T.set_subtensor( Qij[Ms:,:Ms] , sp - sm )
                Qij = T.set_subtensor( Qij[Ms:,Ms:] , cm + cp )

                # Compute the second moment of the output
                m2 = matrix_dot(self.beta_ss[i], 0.5*Qij, self.beta_ss[j].T)

                if i == j:
                    # if i==j we need to add the trace term
                    iAi = solve_upper_triangular(self.Lmm[i].T, solve_lower_triangular(self.Lmm[i],T.eye(2*Ms)))
                    m2 =  m2 + sn2*(1 + (sf2/Ms)*T.sum(iAi*Qij))
                else:
                    M2[j*odims+i] = m2
                M2[i*odims+j] = m2

        M2 = T.stack(M2)
        S = M2.reshape((odims,odims))
        S = S - T.outer(M,M)

        return M,S,V

class VSSGP(GP):
    ''' Variational Sparse Spectral Gaussian Process Regression'''
    def __init__(self, X_dataset=None, Y_dataset=None, name='VSSGP', idims=None, odims=None, profile=False, n_basis=100,  uncertain_inputs=False, hyperparameter_gradients=False):
        self.n_basis = n_basis
        super(VSSGP, self).__init__(X_dataset,Y_dataset,name=name,idims=idims,odims=odims,profile=profile,uncertain_inputs=uncertain_inputs,hyperparameter_gradients=hyperparameter_gradients)

    def init_log_likelihood(self):
        super(VSSGP, self).init_log_likelihood()
        utils.print_with_stamp('Initialising expression graph for sparse spectral log likelihood',self.name)
        idims = self.D
        odims = self.E

        self.A = [[]]*odims
        self.iA = [[]]*odims
        self.beta_ss = [[]]*odims
        nlml_ss = [[]]*odims
        
        # sample initial unscaled spectral points
        self.w_ = np.random.randn(self.n_basis,odims,idims)
        self.w = S(self.w_,name='%s>w'%(self.name),borrow=True)
        self.sr = (self.w*T.exp(-self.loghyp[:,:idims])).transpose(1,0,2)

        N = self.X.shape[0]
        for i in xrange(odims):
            sr = self.sr[i]
            M = sr.shape[0]
            sf2 = T.exp(2*self.loghyp[i,idims])
            sn2 = T.exp(2*self.loghyp[i,idims+1])
            Yi = self.Y[:,i]
            nlml_ss[i] = -0.5*N*T.log(2*np.pi/sn2) - sn2*Yi.dot(Yi) - 0.5*T.log(det(psd((1/sn2)*self.iA[i]))) + Yi.T.dot(m_phi).dot(self.iA[i]) .dot(m_phi.T).dot(Yi)
        
        nlml_ss = T.stack(nlml_ss)
        if self.snr_penalty is not None:
            penalty_params = {'log_snr': np.log(1000), 'log_ls': np.log(100), 'log_std': T.log(self.X.std(0)*(N/(N-1.0))), 'p': 30}
            nlml_ss += self.snr_penalty(self.loghyp)

        # Compute the gradients for the sum of nlml for all output dimensions
        dnlml_wrt_lh = T.jacobian(nlml_ss.sum(),self.loghyp)
        dnlml_wrt_w = T.jacobian(nlml_ss.sum(),self.w)

        utils.print_with_stamp('Compiling sparse spectral log likelihood',self.name)
        self.nlml_ss = F((),nlml_ss,name='%s>nlml'%(self.name), profile=self.profile, mode=self.compile_mode, allow_input_downcast=True)
        utils.print_with_stamp('Compiling jacobian of sparse spectral log likelihood',self.name)
        self.dnlml_ss = F((),(nlml_ss,dnlml_wrt_lh,dnlml_wrt_w),name='%s>dnlml'%(self.name), profile=self.profile, mode=self.compile_mode, allow_input_downcast=True)



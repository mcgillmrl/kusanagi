import os
import numpy as np
import sys
import theano
import theano.tensor as T

from functools import partial
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

class GP(object):
    def __init__(self, X_dataset=None, Y_dataset=None, name='GP', idims=None, odims=None, profile=theano.config.profile, uncertain_inputs=False, hyperparameter_gradients=False, snr_penalty=SNRpenalty.SEard):
        # theano options
        self.profile= profile
        self.compile_mode = theano.compile.get_default_mode()#.excluding('scanOp_pushout_seqs_ops')

        # GP options
        self.min_method = "L-BFGS-B"
        self.state_changed = False
        self.should_recompile = False
        self.trained = False
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
        self.loghyp = None
        self.X = None; self.Y = None
        self.iK = None; self.L = None; self.beta = None;
        self.nlml = None

        # compiled functions
        self.predict_fn = None
        self.predict_d_fn = None

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
            utils.print_with_stamp('Initialising new GP regressor [ Could not open %s.zip ]'%(self.filename),self.name)
            # initialize the class if no pickled version is available
            if X_dataset is not None and Y_dataset is not None:
                self.set_dataset(X_dataset,Y_dataset)
                self.init_loss()
        
        self.ready = False
        utils.print_with_stamp('Finished initialising GP regressor',self.name)
    
    def set_dataset(self,X_dataset,Y_dataset):
        # ensure we don't change the number of input and output dimensions ( the number of samples can change)
        assert X_dataset.shape[0] == Y_dataset.shape[0], "X_dataset and Y_dataset must have the same number of rows"
        if self.X is not None:
            assert self.X.get_value(borrow=True).shape[1] == X_dataset.shape[1]
        if self.Y is not None:
            assert self.Y.get_value(borrow=True).shape[1] == Y_dataset.shape[1]
        
        # first, convert numpy arrays to appropriate type
        X_dataset = X_dataset.astype( theano.config.floatX )
        Y_dataset = Y_dataset.astype( theano.config.floatX )
        # dims = non_angle_dims + 2*angle_dims
        self.N = X_dataset.shape[0]
        self.D = X_dataset.shape[1]
        self.E = Y_dataset.shape[1]

        # now we create symbolic shared variables
        if self.X is None:
            self.X = S(X_dataset,name='%s>X'%(self.name),borrow=True)
        else:
            self.X.set_value(X_dataset,borrow=True)
        if self.Y is None:
            self.Y = S(Y_dataset,name='%s>Y'%(self.name),borrow=True)
        else:
            self.Y.set_value(Y_dataset,borrow=True)

        if not self.trained:
            # init log hyperparameters
            self.init_loghyp()

        # we should be saving, since we updated the trianing dataset
        self.state_changed = True
        if (self.N > 0):
            self.ready = True
        self.trained = False

    def append_dataset(self,X_dataset,Y_dataset):
        if self.X is None:
            self.set_dataset(X_dataset,Y_dataset)
        else:
            X_ = np.vstack((self.X.get_value(), X_dataset.astype(theano.config.floatX)))
            self.X.set_value(X_,borrow=True)
            Y_ = np.vstack((self.Y.get_value(), Y_dataset.astype(theano.config.floatX)))
            self.Y.set_value(Y_,borrow=True)

    def init_loghyp(self):
        idims = self.D; odims = self.E; 
        # initialize the loghyperparameters of the gp ( this code supports squared exponential only, at the moment)
        X = self.X.get_value(); Y = self.Y.get_value()
        loghyp = np.zeros((odims,idims+2))
        loghyp[:,:idims] = X.std(0,ddof=1)
        loghyp[:,idims] = Y.std(0,ddof=1)
        loghyp[:,idims+1] = 0.1*loghyp[:,idims]
        loghyp = np.log(loghyp)

        self.set_loghyp(loghyp)
        self.trained = False

    def set_loghyp(self, loghyp):
        # this creates one theano shared variable for the log hyperparameters
        if self.loghyp is None:
            self.loghyp = S(loghyp,name='%s>loghyp'%(self.name),borrow=True)
        else:
            loghyp = loghyp.reshape(self.loghyp.get_value(borrow=True).shape).astype(theano.config.floatX)
            self.loghyp.set_value(loghyp,borrow=True)

    def get_params(self, symbolic=True, all_shared=False):
        if symbolic:
            retvars = [self.loghyp]
        else:
            retvars = [ self.loghyp.get_value() ]
        return retvars

    def get_all_shared_vars(self):
        return [attr for attr in self.__dict__.values() if isinstance(attr,T.sharedvar.SharedVariable)]

    def set_params(self, params):
        loghyp = params[0]
        inputs = params[1]
        targets = params[2]
        self.set_dataset(inputs,targets)
        self.set_loghyp(loghyp)

    def init_loss(self, cache_vars=True):
        utils.print_with_stamp('Initialising expression graph for full GP training loss function',self.name)
        idims = self.D
        odims = self.E

        self.kernel_func = [[]]*odims
        K = [[]]*odims
        iK = [[]]*odims
        L = [[]]*odims
        beta = [[]]*odims
        nlml = [[]]*odims
        dnlml = [[]]*odims
        covs = (cov.SEard, cov.Noise)
        # these are shared variables for the kernel matrix, its cholesky decomposition and K^-1 dot Y
        if self. iK is None:
            self.iK = S(np.zeros((self.E,self.N,self.N),dtype=theano.config.floatX), name="%s>iK"%(self.name))
        if self.L is None:
            self.L = S(np.zeros((self.E,self.N,self.N),dtype=theano.config.floatX), name="%s>L"%(self.name))
        if self.beta is None:
            self.beta = S(np.zeros((self.E,self.N),dtype=theano.config.floatX), name="%s>beta"%(self.name))
        N = self.X.shape[0].astype(theano.config.floatX)
        for i in xrange(odims):
            # initialise the (before compilation) kernel function
            loghyps = (self.loghyp[i,:idims+1],self.loghyp[i,idims+1])
            self.kernel_func[i] = partial(cov.Sum, loghyps, covs)

            # We initialise the kernel matrices (one for each output dimension)
            K[i] = self.kernel_func[i](self.X)
            L[i] = cholesky(K[i])
            iK[i] = solve_upper_triangular(L[i].T, solve_lower_triangular(L[i],T.eye(self.X.shape[0])))
            Yc = solve_lower_triangular(L[i],self.Y[:,i])
            beta[i] = solve_upper_triangular(L[i].T,Yc)

            # And finally, the negative log marginal likelihood ( again, one for each dimension; although we could share
            # the loghyperparameters across all output dimensions and train the GPs jointly)
            nlml[i] = 0.5*(Yc.T.dot(Yc) + 2*T.sum(T.log(T.diag(L[i]))) + N*T.log(2*np.pi) )
        
        nlml = T.stack(nlml)
        if cache_vars:
            iK = T.stack(iK); iK = T.unbroadcast(iK,0) if iK.broadcastable[0] else iK
            L = T.stack(L); L = T.unbroadcast(L,0) if L.broadcastable[0] else L
            beta = T.stack(beta); beta = T.unbroadcast(beta,0) if beta.broadcastable[0] else beta
    
            # we are going to save the intermediate results in the following shared variables, so we can use them during prediction without having to recompute them
            updts =[(self.iK,iK),(self.L,L),(self.beta,beta)]
        else:
            self.iK = iK 
            self.L = L 
            self.beta = beta
            updts=None

        # we add some penalty to avoid having parameters that are too large
        if self.snr_penalty is not None:
            penalty_params = {'log_snr': np.log(1000), 'log_ls': np.log(100), 'log_std': T.log(self.X.std(0)*(N/(N-1.0))), 'p': 30}
            nlml += self.snr_penalty(self.loghyp)

        # Compute the gradients for the sum of nlml for all output dimensions
        dnlml = T.jacobian(nlml.sum(),self.loghyp)

        # Compile the theano functions
        utils.print_with_stamp('Compiling full GP training loss function',self.name)
        self.nlml = F((),nlml,name='%s>nlml'%(self.name), profile=self.profile, mode=self.compile_mode, allow_input_downcast=True, updates=updts)
        utils.print_with_stamp('Compiling jacobian of full GP training loss function',self.name)
        self.dnlml = F((),(nlml,dnlml),name='%s>dnlml'%(self.name), profile=self.profile, mode=self.compile_mode, allow_input_downcast=True, updates=updts)
        self.state_changed = True # for saving
    
    def init_predict(self, derivs=False):
        if self.nlml is None:
            self.init_loss()

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
            self.predict_fn = F(input_vars,prediction,name='%s>predict_'%(self.name), profile=self.profile, mode=self.compile_mode, allow_input_downcast=True)
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
            self.predict_d_fn = F(input_vars, prediction_derivatives, name='%s>predict_d_'%(self.name), profile=self.profile, mode=self.compile_mode, allow_input_downcast=True)
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
        V = T.zeros((self.D,self.E))

        return M,S,V
    
    def predict(self,mx,Sx = None, derivs=False):
        predict = None
        if not derivs:
            if self.predict_fn is None or self.should_recompile:
                self.init_predict(derivs=derivs)
            predict = self.predict_fn
        else:
            if self.predict_d_fn is None or self.should_recompile:
                self.init_predict(derivs=derivs)
            predict = self.predict_d_fn

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
        if self.nlml is None or self.should_recompile:
            self.init_loss()

        utils.print_with_stamp('Current hyperparameters:',self.name)
        loghyp0 = self.loghyp.get_value(borrow=True)
        print (loghyp0)
        utils.print_with_stamp('nlml: %s'%(np.array(self.nlml())),self.name)
        m_loss = utils.MemoizeJac(self.loss)
        try:
            opt_res = minimize(m_loss, loghyp0, jac=m_loss.derivative, method=self.min_method, tol=1e-12, options={'maxiter': 500})
        except ValueError:
            opt_res = minimize(m_loss, loghyp0, jac=m_loss.derivative, method='CG', tol=1e-12, options={'maxiter': 500})
        print ''
        loghyp = opt_res.x.reshape(loghyp0.shape)
        self.state_changed = not np.allclose(loghyp0,loghyp,1e-6,1e-9)
        self.set_loghyp(loghyp)
        utils.print_with_stamp('New hyperparameters:',self.name)
        print (self.loghyp.get_value(borrow=True))
        utils.print_with_stamp('nlml: %s'%(np.array(self.nlml())),self.name)
        self.trained = True

    def load(self):
        path = os.path.join(utils.get_run_output_dir(),self.filename+'.zip')
        with open(path,'rb') as f:
            utils.print_with_stamp('Loading compiled GP from %s'%(self.filename),self.name)
            state = t_load(f)
            self.set_state(state)
        self.state_changed = False
    
    def save(self):
        sys.setrecursionlimit(100000)
        if self.state_changed:
            path = os.path.join(utils.get_run_output_dir(),self.filename+'.zip')
            with open(path,'wb') as f:
                utils.print_with_stamp('Saving compiled GP with %d inputs and %d outputs'%(self.D,self.E),self.name)
                t_dump(self.get_state(),f,2)
            self.state_changed = False

    def set_state(self,state):
        i = utils.integer_generator()
        self.X = state[i.next()]
        self.N = self.X.get_value(borrow=True).shape[0]
        self.Y = state[i.next()]
        self.loghyp = state[i.next()]
        self.iK = state[i.next()]
        self.L = state[i.next()]
        self.beta = state[i.next()]
        self.nlml = state[i.next()]
        self.dnlml = state[i.next()]
        self.predict_fn = state[i.next()]
        self.predict_d_fn = state[i.next()]
        self.kernel_func = state[i.next()]
        self.trained = state[i.next()]

    def get_state(self):
        return [self.X,self.Y,self.loghyp,self.iK,self.L,self.beta,self.nlml,self.dnlml,self.predict_fn,self.predict_d_fn,self.kernel_func,self.trained]

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
        iL = eyeE/lscales.dimshuffle(0,1,'x')

        # predictive mean
        inp = iL.dot(zeta.T).transpose(0,2,1) 
        iLdotSx = iL.dot(Sx)
        B = T.stack([iLdotSx[i].dot(iL[i]) for i in xrange(odims)]) + T.eye(idims)   #TODO vectorize this
        t = T.stack([inp[i].dot(matrix_inverse(B[i])) for i in xrange(odims)])      # E x N x D
        c = sf2/T.sqrt(T.stack([det(B[i]) for i in xrange(odims)]))
        l = T.exp(-0.5*T.sum(inp*t,2))
        lb = l*self.beta # beta should have been precomputed in init_loss # E x N dot E x N
        M = T.sum(lb,1)*c
        
        # input output covariance
        tiL = T.stack([t[i].dot(iL[i]) for i in xrange(odims)])
        #V = Sx.dot(T.stack([tiL[i].T.dot(lb[i]) for i in xrange(odims)]).T*c)
        V = T.stack([tiL[i].T.dot(lb[i]) for i in xrange(odims)]).T*c

        # predictive covariance
        logk = 2*self.loghyp[:,None,idims] - 0.5*T.sum(inp*inp,2)
        logk_r = logk.dimshuffle(0,'x',1)
        logk_c = logk.dimshuffle(0,1,'x')
        Lambda = iL**2
        R = T.dot((Lambda.dimshuffle(0,'x',1,2) + Lambda).transpose(0,1,3,2),Sx.T).transpose(0,1,3,2) + T.eye(idims)
        z_= Lambda.dot(zeta.T).transpose(0,2,1) 
        M2 = [[]]*(odims**2) # second moment
        N = self.X.shape[0]
        for i in xrange(odims):
            for j in xrange(i+1):
                # This comes from Deisenroth's thesis ( Eqs 2.51- 2.54 )
                Rij = R[i,j]
                n2 = logk_c[i] + logk_r[j] + utils.maha(z_[i],-z_[j],0.5*matrix_inverse(Rij).dot(Sx))
                Q = T.exp( n2 )/T.sqrt(det(Rij))
                # Eq 2.55
                m2 = matrix_dot(self.beta[i], Q, self.beta[j])
                if i == j:
                    m2 =  m2 - T.sum(self.iK[i]*Q) + sf2[i]
                else:
                    M2[j*odims+i] = m2
                M2[i*odims+j] = m2

        M2 = T.stack(M2).T
        S = M2.reshape((odims,odims))
        S = S - T.outer(M,M)

        return M,S,V
        
class SPGP(GP):
    def __init__(self, X_dataset=None, Y_dataset=None, name = 'SPGP', idims=None, odims=None, profile=False, n_basis = 100, uncertain_inputs=False, hyperparameter_gradients=False):
        self.X_sp = None # inducing inputs (symbolic variable)
        self.nlml_sp = None
        self.dnlml_sp = None
        self.beta_sp = None
        self.iKmm = None
        self.iBmm = None
        self.Lmm = None
        self.Amm = None
        self.should_recompile = False
        self.n_basis = n_basis
        # intialize parent class params
        super(SPGP, self).__init__(X_dataset,Y_dataset,name=name,idims=idims,odims=odims,profile=profile,uncertain_inputs=uncertain_inputs,hyperparameter_gradients=hyperparameter_gradients)

    def set_state(self,state):
        i = utils.integer_generator(-8)
        self.X_sp = state[i.next()]
        self.iKmm = state[i.next()]
        self.iBmm = state[i.next()]
        self.Lmm = state[i.next()]
        self.Amm = state[i.next()]
        self.beta_sp = state[i.next()]
        self.nlml_sp = state[i.next()]
        self.dnlml_sp = state[i.next()]
        super(SPGP,self).set_state(state[:-8])
        self.should_recompile = False

    def get_state(self):
        state = super(SPGP,self).get_state()
        state.append(self.X_sp)
        state.append(self.iKmm)
        state.append(self.iBmm)
        state.append(self.Lmm)
        state.append(self.Amm)
        state.append(self.beta_sp)
        state.append(self.nlml_sp)
        state.append(self.dnlml_sp)
        return state

    def init_pseudo_inputs(self):
        assert self.N > self.n_basis, "Dataset must have more than n_basis [ %n ] to enable inference with sparse pseudo inputs"%(self.n_basis)
        self.should_recompile = True
        # pick initial cluster centers from dataset
        X = self.X.get_value()
        X_sp_ = utils.kmeanspp(X,self.n_basis).astype(theano.config.floatX)

        # perform kmeans to get initial cluster centers
        utils.print_with_stamp('Initialising pseudo inputs',self.name)
        X_sp_, dist = kmeans(X, X_sp_, iter=200,thresh=1e-12)
        # initialize symbolic tensor variable if necessary
        if self.X_sp is None:
            self.X_sp = S(X_sp_,name='%s>X_sp'%(self.name),borrow=True)
        else:
            self.X_sp.set_value(X_sp_,borrow=True)

    def set_dataset(self,X_dataset,Y_dataset):
        # set the dataset on the parent class
        super(SPGP, self).set_dataset(X_dataset,Y_dataset)
        if self.N <= self.n_basis:
            utils.print_with_stamp('Dataset is not large enough for using pseudo inputs. Training full GP.',self.name)
            self.X_sp = None
            self.nlml_sp = None
            self.dnlml_sp = None
            self.beta_sp = None
            self.Lmm = None
            self.Amm = None
            self.should_recompile = False

        if self.N > self.n_basis and self.X_sp is None:
            utils.print_with_stamp('Dataset is large enough for using pseudo inputs. You should reinitiialise the training loss function and predictions.',self.name)
            # init the shared variable for the pseudo inputs
            self.init_pseudo_inputs()
            self.should_recompile = True
        
    def init_loss(self, cache_vars=True):
        # initialize the training loss function of the GP class
        super(SPGP, self).init_loss(cache_vars)
        # here nlml and dnlml have already been innitialised, sow e can replace nlml and dnlml
        # only if we have enough data to train the pseudo inputs ( i.e. self.N > self.n_basis)
        if self.N > self.n_basis:
            utils.print_with_stamp('Initialising FITC training loss function',self.name)
            self.should_recompile = False
            odims = self.E
            idims = self.D
            # initialize the training loss function of the sparse FITC approximation
            Kmm = [[]]*odims
            iKmm = [[]]*odims
            Lmm = [[]]*odims
            Amm = [[]]*odims
            iBmm = [[]]*odims
            beta_sp = [[]]*odims
            nlml_sp = [[]]*odims
            dnlml_sp = [[]]*odims
            if self.iKmm is None:
                self.iKmm = S(np.zeros((self.E,self.n_basis,self.n_basis),dtype=theano.config.floatX), name="%s>iKmm"%(self.name))
            if self.Lmm is None:
                self.Lmm = S(np.zeros((self.E,self.n_basis,self.n_basis),dtype=theano.config.floatX), name="%s>Lmm"%(self.name))
            if self.Amm is None:
                self.Amm = S(np.zeros((self.E,self.n_basis,self.n_basis),dtype=theano.config.floatX), name="%s>Amm"%(self.name))
            if self.iBmm is None:
                self.iBmm = S(np.zeros((self.E,self.n_basis,self.n_basis),dtype=theano.config.floatX), name="%s>iBmm"%(self.name))
            if self.beta_sp is None:
                self.beta_sp = S(np.zeros((self.E,self.n_basis),dtype=theano.config.floatX), name="%s>beta_sp"%(self.name))
            ridge = 1e-6 if theano.config.floatX == 'float64' else 5e-4
            for i in xrange(odims):
                X_sp_i = self.X_sp  # TODO allow for different pseudo inputs for each dimension
                ll = T.exp(self.loghyp[i,:idims])
                sf2 = T.exp(2*self.loghyp[i,idims])
                sn2 = T.exp(2*self.loghyp[i,idims+1])
                N = self.X.shape[0].astype(theano.config.floatX)
                M = X_sp_i.shape[0].astype(theano.config.floatX)

                Kmm[i] = self.kernel_func[i](X_sp_i) + ridge*T.eye(self.X_sp.shape[0])
                Kmn = self.kernel_func[i](X_sp_i, self.X)
                Lmm[i] = cholesky(Kmm[i])
                iKmm[i] = solve_upper_triangular(Lmm[i].T, solve_lower_triangular(Lmm[i],T.eye(self.n_basis)))
                Lmn = solve_lower_triangular(Lmm[i],Kmn)
                diagQnn =  T.diag(Lmn.T.dot(Lmn))

                # Gamma = diag(Knn - Qnn) + sn2*I
                Gamma = sf2 - diagQnn + sn2
                Gamma_inv = 1.0/Gamma

                # these operations are done to avoid inverting K_sp = (Qnn+Gamma)
                Kmn_ = Kmn*T.sqrt(Gamma_inv)                    # Kmn_*Gamma^-.5
                Yi = self.Y[:,i]*(T.sqrt(Gamma_inv))            # Gamma^-.5* Y
                Bmm = Kmm[i] + (Kmn_).dot(Kmn_.T)                  # Kmm + Kmn * Gamma^-1 * Knm
                Amm[i] = cholesky(Bmm)
                iBmm[i] = solve_upper_triangular(Amm[i].T, solve_lower_triangular(Amm[i],T.eye(self.n_basis)))

                Yci = solve_lower_triangular(Amm[i],Kmn_.dot(Yi) )
                beta_sp[i] = solve_upper_triangular(Amm[i].T,Yci)

                log_det_K_sp = T.sum(T.log(Gamma)) - 2*T.sum(T.log(T.diag(Lmm[i]))) + 2*T.sum(T.log(T.diag(Amm[i])))

                nlml_sp[i] = 0.5*( Yi.dot(Yi) - Yci.dot(Yci) + log_det_K_sp + N*np.log(2*np.pi).astype(theano.config.floatX) )

            nlml_sp = T.stack(nlml_sp)
            if cache_vars:
                iKmm = T.stack(iKmm); iKmm = T.unbroadcast(iKmm,0) if iKmm.broadcastable[0] else iKmm
                Lmm = T.stack(Lmm); Lmm = T.unbroadcast(Lmm,0) if Lmm.broadcastable[0] else Lmm
                Amm = T.stack(Amm); Amm = T.unbroadcast(Amm,0) if Amm.broadcastable[0] else Amm
                iBmm = T.stack(iBmm); iBmm = T.unbroadcast(iBmm,0) if iBmm.broadcastable[0] else iBmm
                beta_sp = T.stack(beta_sp); beta_sp = T.unbroadcast(beta_sp,0) if beta_sp.broadcastable[0] else beta_sp
        
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
            dnlml_sp = T.jacobian(nlml_sp.sum(),self.X_sp)

            # Compile the theano functions
            utils.print_with_stamp('Compiling FITC training loss function',self.name)
            self.nlml_sp = F((),nlml_sp,name='%s>nlml_sp'%(self.name), profile=self.profile, mode=self.compile_mode, allow_input_downcast=True, updates=updts)
            utils.print_with_stamp('Compiling jacobian of FITC training loss function',self.name)
            self.dnlml_sp = F((),(nlml_sp,dnlml_sp),name='%s>dnlml_sp'%(self.name), profile=self.profile, mode=self.compile_mode, allow_input_downcast=True,updates=updts)
            
    def predict_symbolic(self,mx,Sx):
        if self.N <= self.n_basis:
            # stick with the full GP
            return super(SPGP, self).predict_symbolic(mx,Sx)

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
        V = T.zeros((self.D,self.E))

        return M,S,V

    def set_X_sp(self, X_sp):
        X_sp = X_sp.reshape(self.X_sp.get_value(borrow=True).shape).astype(theano.config.floatX)
        self.X_sp.set_value(X_sp,borrow=True)
    
    def get_params(self, symbolic=True):
        retvars = super(SPGP,self).get_params(symbolic=False)
        retvars.append(self.X_sp)
        if not symbolic:
            retvars = [ r.get_value(borrow=True) for r in retvars]
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
        if self.nlml_sp is None or self.should_recompile:
            self.init_loss()
        # train the full GP
        super(SPGP, self).train()

        if self.N > self.n_basis:
            # train the pseudo input locations
            if self.nlml_sp is None:
                self.init_loss()
            utils.print_with_stamp('nlml SP: %s'%(np.array(self.nlml_sp())),self.name)
            m_loss_sp = utils.MemoizeJac(self.loss_sp)
            opt_res = minimize(m_loss_sp, self.X_sp.get_value(), jac=m_loss_sp.derivative, method=self.min_method, tol=1e-12, options={'maxiter': 1000})
            print ''
            X_sp = opt_res.x.reshape(self.X_sp.get_value(borrow=True).shape)
            self.set_X_sp(X_sp)
            utils.print_with_stamp('nlml SP: %s'%(np.array(self.nlml_sp())),self.name)
        self.trained = True

class SPGP_UI(SPGP,GP_UI):
    def __init__(self, X_dataset=None, Y_dataset=None, name = 'SPGP_UI', idims=None, odims=None,profile=False, n_basis = 100, hyperparameter_gradients=False):
        super(SPGP_UI, self).__init__(X_dataset,Y_dataset,name=name,idims=idims,odims=odims,profile=profile,n_basis=n_basis,uncertain_inputs=True,hyperparameter_gradients=hyperparameter_gradients)

    def predict_symbolic(self,mx,Sx):
        if self.N <= self.n_basis:
            # stick with the full GP
            return GP_UI.predict_symbolic(self,mx,Sx)

        idims = self.D
        odims = self.E

        #centralize inputs 
        zeta = self.X_sp - mx
        
        # initialize some variables
        sf2 = T.exp(2*self.loghyp[:,idims])
        eyeE = T.tile(T.eye(idims),(odims,1,1))
        lscales = T.exp(self.loghyp[:,:idims])
        iL = eyeE/lscales.dimshuffle(0,1,'x')

        # predictive mean
        inp = iL.dot(zeta.T).transpose(0,2,1) 
        iLdotSx = iL.dot(Sx)
        B = T.stack([iLdotSx[i].dot(iL[i]) for i in xrange(odims)]) + T.eye(idims)   #TODO vectorize this
        t = T.stack([inp[i].dot(matrix_inverse(B[i])) for i in xrange(odims)])      # E x N x D
        c = sf2/T.sqrt(T.stack([det(B[i]) for i in xrange(odims)]))
        l = T.exp(-0.5*T.sum(inp*t,2))
        lb = l*self.beta_sp # beta_sp should have been precomputed in init_loss # E x N dot E x N
        M = T.sum(lb,1)*c
        
        # input output covariance
        tiL = T.stack([t[i].dot(iL[i]) for i in xrange(odims)])
        #V = Sx.dot(T.stack([tiL[i].T.dot(lb[i]) for i in xrange(odims)]).T*c)
        V = T.stack([tiL[i].T.dot(lb[i]) for i in xrange(odims)]).T*c

        # predictive covariance
        logk = 2*self.loghyp[:,None,idims] - 0.5*T.sum(inp*inp,2)
        logk_r = logk.dimshuffle(0,'x',1)
        logk_c = logk.dimshuffle(0,1,'x')
        Lambda = iL**2
        R = T.dot((Lambda.dimshuffle(0,'x',1,2) + Lambda).transpose(0,1,3,2),Sx.T).transpose(0,1,3,2) + T.eye(idims)
        z_= Lambda.dot(zeta.T).transpose(0,2,1) 
        M2 = [[]]*(odims**2) # second moment
        N = self.X.shape[0]
        for i in xrange(odims):
            for j in xrange(i+1):
                # This comes from Deisenroth's thesis ( Eqs 2.51- 2.54 )
                Rij = R[i,j]
                n2 = logk_c[i] + logk_r[j] + utils.maha(z_[i],-z_[j],0.5*matrix_inverse(Rij).dot(Sx))
                Q = T.exp( n2 )/T.sqrt(det(Rij))
                # Eq 2.55
                m2 = matrix_dot(self.beta_sp[i], Q, self.beta_sp[j])
                if i == j:
                    m2 =  m2 - T.sum((self.iKmm[i] - self.iBmm[i])*Q) + sf2[i]
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
        self.sat_func = state[-2]
        self.loghyp_full = state[-1]
        super(RBFGP,self).set_state(state[:-2])

    def get_state(self):
        state = super(RBFGP,self).get_state()
        state.append(self.sat_func)
        state.append(self.loghyp_full)
        return state
    
    def get_params(self, symbolic=True, all_shared=False):
        retvars = [self.loghyp_full,self.X,self.Y]
        if not symbolic:
            retvars = [ r.get_value(borrow=True) for r in retvars]
        return retvars
    
    def set_loghyp(self, loghyp):
        # this creates one theano shared variable for the log hyperparameters
        if self.loghyp_full is None:
            self.loghyp_full = S(loghyp,name='%s>loghyp'%(self.name),borrow=True)
            self.loghyp = T.concatenate([self.loghyp_full[:,:-2], theano.gradient.disconnected_grad(self.loghyp_full[:,-2:])], axis=1)
        else:
            loghyp = loghyp.reshape(self.loghyp_full.get_value(borrow=True).shape).astype(theano.config.floatX)
            self.loghyp_full.set_value(loghyp,borrow=True)

    def predict_symbolic(self,mx,Sx):
        idims = self.D
        odims = self.E

        #centralize inputs 
        zeta = self.X - mx
        
        # initialize some variables
        sf2 = T.exp(2*self.loghyp[:,idims])
        eyeE = T.tile(T.eye(idims),(odims,1,1))
        lscales = T.exp(self.loghyp[:,:idims])
        iL = eyeE/lscales.dimshuffle(0,1,'x')

        # predictive mean
        inp = iL.dot(zeta.T).transpose(0,2,1) 
        iLdotSx = iL.dot(Sx)
        B = T.stack([iLdotSx[i].dot(iL[i]) for i in xrange(odims)]) + T.eye(idims)   #TODO vectorize this
        t = T.stack([inp[i].dot(matrix_inverse(B[i])) for i in xrange(odims)])      # E x N x D
        c = sf2/T.sqrt(T.stack([det(B[i]) for i in xrange(odims)]))
        l = T.exp(-0.5*T.sum(inp*t,2))
        lb = l*self.beta # beta should have been precomputed in init_loss # E x N
        M = T.sum(lb,1)*c
        
        # input output covariance
        tiL = T.stack([t[i].dot(iL[i]) for i in xrange(odims)])
        #V = Sx.dot(T.stack([tiL[i].T.dot(lb[i]) for i in xrange(odims)]).T*c)
        V = T.stack([tiL[i].T.dot(lb[i]) for i in xrange(odims)]).T*c

        # predictive covariance
        logk = 2*self.loghyp[:,None,idims] - 0.5*T.sum(inp*inp,2)
        logk_r = logk.dimshuffle(0,'x',1)
        logk_c = logk.dimshuffle(0,1,'x')
        Lambda = iL**2
        R = T.dot((Lambda.dimshuffle(0,'x',1,2) + Lambda).transpose(0,1,3,2),Sx.T).transpose(0,1,3,2) + T.eye(idims)
        z_= Lambda.dot(zeta.T).transpose(0,2,1) 
        M2 = [[]]*(odims**2) # second moment
        N = self.X.shape[0]
        for i in xrange(odims):
            for j in xrange(i+1):
                # This comes from Deisenroth's thesis ( Eqs 2.51- 2.54 )
                Rij = R[i,j]
                n2 = logk_c[i] + logk_r[j] + utils.maha(z_[i],-z_[j],0.5*matrix_inverse(Rij).dot(Sx))
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
            # saturate the output
            M,S,U = self.sat_func(M,S)
            # compute the joint input output covariance
            V = V.dot(U)

        return M,S,V

class SSGP(GP):
    ''' Sparse Spectral Gaussian Process Regression'''
    def __init__(self, X_dataset=None, Y_dataset=None, name='SSGP', idims=None, odims=None, profile=False, n_basis=100,  uncertain_inputs=False, hyperparameter_gradients=False):
        self.w = None
        self.sr = None
        self.Lmm = None
        self.iA = None
        self.beta_ss = None
        self.nlml_ss = None
        self.dnlml_ss = None
        self.n_basis = n_basis
        super(SSGP, self).__init__(X_dataset,Y_dataset,name=name,idims=idims,odims=odims,profile=profile,uncertain_inputs=uncertain_inputs,hyperparameter_gradients=hyperparameter_gradients)

    def set_state(self,state):
        i = utils.integer_generator(-7)
        self.w = state[i.next()]
        self.sr = state[i.next()]
        self.beta_ss = state[i.next()]
        self.Lmm = state[i.next()]
        self.iA = state[i.next()]
        self.nlml_ss = state[i.next()]
        self.dnlml_ss = state[i.next()]
        super(SSGP,self).set_state(state[:-7])
        self.should_recompile = False

    def get_state(self):
        state = super(SSGP,self).get_state()
        state.append(self.w)
        state.append(self.sr)
        state.append(self.beta_ss)
        state.append(self.Lmm)
        state.append(self.iA)
        state.append(self.nlml_ss)
        state.append(self.dnlml_ss)
        return state

    def set_dataset(self,X_dataset,Y_dataset):
        # set the dataset on the parent class
        super(SSGP, self).set_dataset(X_dataset,Y_dataset)

    def init_loss(self,cache_vars=True):
        super(SSGP, self).init_loss()
        utils.print_with_stamp('Initialising expression graph for sparse spectral training loss function',self.name)
        idims = self.D
        odims = self.E

        A = [[]]*odims
        Lmm = [[]]*odims
        iA = [[]]*odims
        beta_ss = [[]]*odims
        nlml_ss = [[]]*odims
        if self.iA is None:
            self.iA = S(np.zeros((self.E,2*self.n_basis,2*self.n_basis),dtype=theano.config.floatX), name="%s>iA"%(self.name))
        if self.Lmm is None:
            self.Lmm = S(np.zeros((self.E,2*self.n_basis,2*self.n_basis),dtype=theano.config.floatX), name="%s>Lmm"%(self.name))
        if self.beta_ss is None:
            self.beta_ss = S(np.zeros((self.E,2*self.n_basis),dtype=theano.config.floatX), name="%s>beta_sp"%(self.name))
        
        # sample initial unscaled spectral points
        self.set_spectral_samples()

        #init variables
        N = self.X.shape[0].astype(theano.config.floatX)
        M = self.sr.shape[1].astype(theano.config.floatX)
        Mi = 2*self.sr.shape[1]
        sf2 = T.exp(2*self.loghyp[:,idims])
        sf2M = sf2/M
        sn2 = T.exp(2*self.loghyp[:,idims+1])
        srdotX = self.sr.dot(self.X.T)
        phi_f = T.concatenate( [T.sin(srdotX), T.cos(srdotX)], axis=1 ) # E x 2*n_basis x N
        
        # TODO vectorize these ops
        for i in xrange(odims):
            sf2M_i = sf2M[i]; phi_f_i = phi_f[i]; sn2_i = sn2[i]
            A[i] = sf2M_i*phi_f_i.dot(phi_f_i.T) + sn2_i*T.eye(Mi)
            Lmm[i] = cholesky(A[i])
            iA[i] = solve_upper_triangular(Lmm[i].T, solve_lower_triangular(Lmm[i],T.eye(Mi)))

            Yi = self.Y[:,i]
            Yci = solve_lower_triangular(Lmm[i],phi_f_i.dot(Yi))
            beta_ss[i] = sf2M_i*solve_upper_triangular(Lmm[i].T,Yci)
            
            nlml_ss[i] = 0.5*( Yi.dot(Yi) - sf2M_i*Yci.dot(Yci) )/sn2_i + T.sum(T.log(T.diag(Lmm[i]))) + (0.5*N - M)*T.log(sn2_i) + 0.5*N*np.log(2*np.pi).astype(theano.config.floatX)

        nlml_ss = T.stack(nlml_ss)
        if cache_vars:
            iA = T.stack(iA); iA = T.unbroadcast(iA,0) if iA.broadcastable[0] else iA
            Lmm = T.stack(Lmm); Lmm = T.unbroadcast(Lmm,0) if Lmm.broadcastable[0] else Lmm
            beta_ss = T.stack(beta_ss); beta_ss = T.unbroadcast(beta_ss,0) if beta_ss.broadcastable[0] else beta_ss
    
            # we are going to save the intermediate results in the following shared variables, so we can use them during prediction without having to recompute them
            updts = [(self.iA,iA),(self.Lmm,Lmm),(self.beta_ss,beta_ss)]
        else:
            self.iA = iA 
            self.Lmm = Lmm 
            beta_ss = T.stack(beta_ss); beta_ss = T.unbroadcast(beta_ss,0) if beta_ss.broadcastable[0] else beta_ss
            updts=None

        if self.snr_penalty is not None:
            penalty_params = {'log_snr': np.log(1000), 'log_ls': np.log(100), 'log_std': T.log(self.X.std(0)*(N/(N-1.0))), 'p': 30}
            nlml_ss += self.snr_penalty(self.loghyp)

        # Compute the gradients for the sum of nlml for all output dimensions
        dnlml_wrt_lh = T.jacobian(nlml_ss.sum(),self.loghyp)
        dnlml_wrt_w = T.jacobian(nlml_ss.sum(),self.w)

        utils.print_with_stamp('Compiling sparse spectral training loss function',self.name)
        self.nlml_ss = F((),nlml_ss,name='%s>nlml_ss'%(self.name), profile=self.profile, mode=self.compile_mode, allow_input_downcast=True, updates=updts)
        utils.print_with_stamp('Compiling jacobian of sparse spectral training loss function',self.name)
        self.dnlml_ss = F((),(nlml_ss,dnlml_wrt_lh,dnlml_wrt_w),name='%s>dnlml_ss'%(self.name), profile=self.profile, mode=self.compile_mode, allow_input_downcast=True, updates=updts)

    def set_spectral_samples(self,w=None):
        if w is None:
            idims = self.D
            odims = self.E
            w = np.random.randn(self.n_basis,odims,idims).astype(theano.config.floatX)
        else:
            if self.w is not None:
                w = w.reshape(self.w.get_value(borrow=True).shape).astype(theano.config.floatX)
            else:
                w = w.reshape((self.n_basis,odims,idims)).astype(theano.config.floatX)


        if self.w is None:
            idims = self.D
            self.w = S(w,name='%s>w'%(self.name),borrow=True)
            self.sr = (self.w*T.exp(-self.loghyp[:,:idims])).transpose(1,0,2)
        else:
            self.w.set_value(w,borrow=True)
    
    def get_params(self, symbolic=True, all_shared=False):
        retvars = super(SSGP,self).get_params(symbolic=False)
        retvars.append(self.w)
        if not symbolic:
            retvars = [ r.get_value(borrow=True) for r in retvars]
        return retvars

    def loss_ss(self, params, parameter_shapes):
        loghyp,w = utils.unwrap_params(params,parameter_shapes)
        self.set_loghyp(loghyp)
        self.set_spectral_samples(w)
        nlml,dnlml_lh,dnlml_sr = self.dnlml_ss()
        nlml = nlml.sum()
        dnlml = utils.wrap_params([dnlml_lh,dnlml_sr])
        # on a 64bit system, scipy optimize complains if we pass a 32 bit float
        res = (nlml.astype(np.float64), dnlml.astype(np.float64))
        utils.print_with_stamp('%s'%(str(res[0])),self.name,True)
        return res

    def train(self, pretrain_full=True):
        if pretrain_full:
            # train the full GP ( if dataset too large, take a random subsample)
            X_full = None
            Y_full = None
            n_subsample = 1024
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

        if self.nlml_ss is None or self.should_recompile:
            self.init_loss()

        # initialize spectral samples
        nlml = self.nlml_ss()
        best_w = self.w.get_value()

        # try a couple spectral samples and pick the one with the lowest nlml
        for i in xrange(100):
            self.set_spectral_samples()
            nlml_i = self.nlml_ss()
            for d in xrange(odims):
                if np.all(nlml_i[d] < nlml[d]):
                    nlml[d] = nlml_i[d]
                    best_w[:,d,:] = self.w.get_value()[:,d,:]

        self.set_spectral_samples( best_w )

        # train the pseudo input locations
        utils.print_with_stamp('nlml SS: %s'%(np.array(self.nlml_ss())),self.name)
        # wrap loghyp plus sr (save shapes)
        p0 = [self.loghyp.get_value(),self.w.get_value()]
        parameter_shapes = [p.shape for p in p0]
        m_loss_ss = utils.MemoizeJac(self.loss_ss)
        opt_res = minimize(m_loss_ss, utils.wrap_params(p0), args=parameter_shapes, jac=m_loss_ss.derivative, method=self.min_method, tol=1e-12, options={'maxiter': 1000})
        print ''
        loghyp,w = utils.unwrap_params(opt_res.x,parameter_shapes)
        self.set_loghyp(loghyp)
        self.set_spectral_samples(w)
        utils.print_with_stamp('nlml SS: %s'%(np.array(self.nlml_ss())),self.name)
        self.trained = True

    def predict_symbolic(self,mx,Sx):
        odims = self.E
        idims = self.D

        # compute the mean and variance for each output dimension
        mean = [[]]*odims
        variance = [[]]*odims
        for i in xrange(odims):
            sr = self.sr[i]
            M = sr.shape[0].astype(theano.config.floatX)
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
        V = T.zeros((self.D,self.E))

        return M,S,V

class SSGP_UI(SSGP, GP_UI):
    ''' Sparse Spectral Gaussian Process Regression with Uncertain Inputs'''
    def __init__(self, X_dataset=None, Y_dataset=None, name='SSGP_UI', idims=None, odims=None, profile=False, n_basis=100,  uncertain_inputs=True, hyperparameter_gradients=False):
        SSGP.__init__(self,X_dataset,Y_dataset,name=name,idims=idims,odims=odims,profile=profile,n_basis=n_basis,uncertain_inputs=True,hyperparameter_gradients=hyperparameter_gradients)

    def predict_symbolic(self,mx,Sx, method=1):
        if method == 1:
            # fast compilation
            return self.predict_symbolic_1(mx,Sx)
        else:
            # fast running
            return self.predict_symbolic_2(mx,Sx)

    def predict_symbolic_1(self,mx,Sx):
        if self.N < self.n_basis:
            # stick with the full GP
            return GP_UI.predict_symbolic(self,mx,Sx)

        idims = self.D
        odims = self.E
        
        # precompute some variables
        sf2 = T.exp(2*self.loghyp[:,idims])
        sn2 = T.exp(2*self.loghyp[:,idims+1])
        E = self.sr.shape[0]
        Ms = self.sr.shape[1]
        oidx = T.arange(E); sidx = T.arange(Ms)
        srdotx = self.sr.dot(mx)
        srdotSx = self.sr.dot(Sx) 
        siSxsj = srdotSx.dot(self.sr.transpose(1,2,0)).transpose(0,3,1,2) #Ms x Ms
        srdotSxdotsr = siSxsj[oidx,oidx][:,sidx,sidx]
        e = T.exp(-0.5*srdotSxdotsr)
        cos_srdotx = T.cos(srdotx)
        sin_srdotx = T.sin(srdotx)
        cos_srdotx_e = cos_srdotx*e
        sin_srdotx_e = sin_srdotx*e

        # compute the mean vector
        mphi = T.horizontal_stack( sin_srdotx_e, cos_srdotx_e ) # E x 2*Ms
        M = T.sum( mphi*self.beta_ss, 1)

        # input output covariance
        mx_c = mx.dimshuffle(0,'x'); mx_r = mx.dimshuffle('x',0)
        sin_srdotx_e_r = sin_srdotx_e.dimshuffle(0,'x',1); cos_srdotx_e_r = cos_srdotx_e.dimshuffle(0,'x',1)
        c = T.concatenate([ mx_c*sin_srdotx_e_r + srdotSx.transpose(0,2,1)*cos_srdotx_e_r, mx_c*cos_srdotx_e_r - srdotSx.transpose(0,2,1)*sin_srdotx_e_r ], axis=2) # E x D x 2*Ms
        beta_ss_r = self.beta_ss.dimshuffle(0,'x',1)
        V = matrix_inverse(Sx).dot(T.sum( c*beta_ss_r, 2 ).T - T.outer(mx,M))
        
        # compute the second moment matrix
        sijSxsij = -0.5*(srdotSxdotsr.dimshuffle(0,'x',1,'x') + srdotSxdotsr.dimshuffle('x',0,'x',1)) 
        em =  T.exp(sijSxsij+siSxsj)      # MsxMs
        ep =  T.exp(sijSxsij-siSxsj)     # MsxMs
        sin_srdotx_c = sin_srdotx.dimshuffle(0,'x',1,'x')
        sin_srdotx_r = sin_srdotx.dimshuffle('x',0,'x',1)
        cos_srdotx_c = cos_srdotx.dimshuffle(0,'x',1,'x')
        cos_srdotx_r = cos_srdotx.dimshuffle('x',0,'x',1)
        ss = sin_srdotx_c*sin_srdotx_r
        sc = sin_srdotx_c*cos_srdotx_r
        cs = cos_srdotx_c*sin_srdotx_r
        cc = cos_srdotx_c*cos_srdotx_r
        sm = (sc-cs)*em
        sp = (sc+cs)*ep
        cm = (ss+cc)*em
        cp = (cc-ss)*ep
        Qu = T.concatenate([cm-cp,sm+sp],axis=3)
        Ql = T.concatenate([sp-sm,cm+cp],axis=3)
        Q = T.concatenate([Qu,Ql],axis=2)
        M2 = 0.5*T.sum(beta_ss_r*T.sum(Q*beta_ss_r,-1),-1)

        # compute the additional diagonal term for the second moment matrix
        sf2Ms = (sf2/Ms.astype(theano.config.floatX))
        diagm2 = T.diag( sn2*(1 + sf2Ms*T.sum(self.iA*Q[oidx,oidx],[1,2])) )
        M2 = M2 + diagm2

        # compute the predictive covariance
        S = M2 - T.outer(M,M)
        
        return M,S,V

    def predict_symbolic_2(self,mx,Sx):
        if self.N < self.n_basis:
            # stick with the full GP
            return GP_UI.predict_symbolic(self,mx,Sx)

        idims = self.D
        odims = self.E
        
        # precompute some variables
        Ms = self.sr.shape[1]
        sf2M = T.exp(2*self.loghyp[:,idims])/T.cast(Ms,theano.config.floatX)
        sn2 = T.exp(2*self.loghyp[:,idims+1])
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

        # input output covariance
        mx_c = mx.dimshuffle(0,'x'); mx_r = mx.dimshuffle('x',0)
        sin_srdotx_e_r = sin_srdotx_e.dimshuffle(0,'x',1); cos_srdotx_e_r = cos_srdotx_e.dimshuffle(0,'x',1)
        c = T.concatenate([ mx_c*sin_srdotx_e_r + srdotSx.transpose(0,2,1)*cos_srdotx_e_r, mx_c*cos_srdotx_e_r - srdotSx.transpose(0,2,1)*sin_srdotx_e_r ], axis=2) # E x D x 2*Ms
        beta_ss_r = self.beta_ss.dimshuffle(0,'x',1)
        V = matrix_inverse(Sx).dot(T.sum( c*beta_ss_r, 2 ).T - T.outer(mx,M))
        
        # TODO vectorize this operation
        srdotSxdotsr_c = srdotSxdotsr.dimshuffle(0,1,'x')
        srdotSxdotsr_r = srdotSxdotsr.dimshuffle(0,'x',1)
        M2 = [[]]*(odims**2) # second moment
        for i in xrange(odims):
            # predictive covariance
            for j in xrange(i+1):
                # compute the second moments of the spectral feature vectors
                siSxsj = srdotSx[i].dot(self.sr[j].T) #Ms x Ms
                sijSxsij = -0.5*(srdotSxdotsr_c[i] + srdotSxdotsr_r[j]) 
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
                
                # Populate the second moment matrix of the feature vector
                Qij_up = T.concatenate([cm-cp,sm+sp],axis=1)
                Qij_lo = T.concatenate([sp-sm,cm+cp],axis=1)
                Qij = T.concatenate([Qij_up,Qij_lo],axis=0)

                # Compute the second moment of the output
                m2 = matrix_dot(self.beta_ss[i], 0.5*Qij, self.beta_ss[j].T)

                if i == j:
                    # if i==j we need to add the trace term
                    m2 =  m2 + sn2[i]*(1 + sf2M[i]*T.sum(self.iA[i]*Qij ))
                else:
                    M2[j*odims+i] = m2
                M2[i*odims+j] = m2

        M2 = T.stack(M2)
        S = M2.reshape((odims,odims))
        S = S - T.outer(M,M)

        return M,S,V

class VSSGP(SSGP):
    ''' Variational Sparse Spectral Gaussian Process Regression'''
    def __init__(self, X_dataset=None, Y_dataset=None, name='VSSGP', idims=None, odims=None, profile=False, n_basis=100,  uncertain_inputs=False, hyperparameter_gradients=False):
        self.n_basis = n_basis
        super(VSSGP, self).__init__(X_dataset,Y_dataset,name=name,idims=idims,odims=odims,profile=profile,n_basis=n_basis,uncertain_inputs=uncertain_inputs,hyperparameter_gradients=hyperparameter_gradients)

    def init_loss(self):
        super(VSSGP, self).init_loss()
        utils.print_with_stamp('Initialising expression graph for sparse spectral training loss function',self.name)
        idims = self.D
        odims = self.E

        self.A = [[]]*odims
        self.iA = [[]]*odims
        self.beta_ss = [[]]*odims
        nlml_ss = [[]]*odims
        
        # hyperparameters for the prior parameters
        # for the spectral points distribution
        self.w_mean_ = np.zeros(self.n_basis,self.D)
        self.w_cov_ = np.ones(self.n_basis,self.D)
        # for the inducing spectral points distribution
        self.z_ = np.zeros(self.n_basis,self.D)
        # for the phases
        self.b_bounds_ = np.tile(np.array([0,2*np.pi]),(self.n_basis,1))
        # for the fourier coefficients
        self.fc_mean_ = np.zeros(self.E,self.D)
        self.fc_cov_ = np.ones(self.E,self.D)

        # initialize shared variables for all parameters
        self.w_mean = S(self.w_mean_,name='%s>w_mean'%(self.name),borrow=True)
        self.w_cov = S(self.w_cov_,name='%s>w_cov'%(self.name),borrow=True)
        self.z = S(self.z_,name='%s>z'%(self.name),borrow=True)
        self.b_bounds = S(self.b_bounds_,name='%s>b_bounds'%(self.name),borrow=True)
        self.fc_mean = S(self.fc_mean_,name='%s>fc_mean'%(self.name),borrow=True)
        self.fc_cov = S(self.fc_cov_,name='%s>fc_cov'%(self.name),borrow=True)

        N = self.X.shape[0].astype(theano.config.floatX)
        for i in xrange(odims):
            Ms = sr.shape[0]
            sf2 = T.exp(2*self.loghyp[i,idims])
            sn2 = T.exp(2*self.loghyp[i,idims+1])
            ill = T.exp(-self.loghyp[i,:idims])
            
            x_nm = 2*np.pi*(iLL*(self.X[:,None,:]-self.z[None,:,:])) #  N x 1 x D - 1 x Ms x D = N x Ms x D
            x_Sw_x = T.sum(x_nm*x_nm*self.w_cov,-1)  #  N x Ms, we only store the diagonal of w_cov, so we do not need to perform dot products
            uw_x = T.sum(self.w_mean*x_nm,-1) # N x Ms
            exp_x_Sw_x = T.exp(-0.5*x_Sw_x)
            m_cos_w = ( sin(uw_k + self.b_bounds[:,1]) - sin(uw_k + self.b_bounds[:,0]) )/(self.b_bounds[:,1] - self.b_bounds[:,0])
            m_cos_2w = ( sin(2*(uw_k + self.b_bounds[:,1])) - sin(2*(uw_k + self.b_bounds[:,0])) )/(self.b_bounds[:,1] - self.b_bounds[:,0])
            m_phi = T.sqrt(2*sf2/M)*exp_x_Sw_x*m_cos_w # N x Ms
            m_K =  m_phi.T.dot(m_phi) # Ms x Ms
            m_K = m_K - T.diag(m_K) + 0.5*(1+(exp_x_Sx_x**4)*m_cos_2w)
            A = m_K + sn2[i]*T.eye(self.w.shape[0])
            Lmm = cholesky(A)
            Yi = self.Y[:,i]
            Yci = solve_lower_triangular(Lmm,m_phi.dot(Yi))
            Yi = self.Y[:,i]
            
            nlml_ss[i] = -0.5*N*T.log(2*np.pi/sn2) - 0.5*sn2*Yi.dot(Yi) + 0.5*sn2*Yci.dot(Yci) + T.sum(T.log(T.diag(Lmm/sn2)))
        
        nlml_ss = T.stack(nlml_ss)
        if self.snr_penalty is not None:
            penalty_params = {'log_snr': np.log(1000), 'log_ls': np.log(100), 'log_std': T.log(self.X.std(0)*(N/(N-1.0))), 'p': 30}
            nlml_ss += self.snr_penalty(self.loghyp)

        # Compute the gradients for the sum of nlml for all output dimensions
        dnlml_wrt_lh = T.jacobian(nlml_ss.sum(),self.loghyp)
        dnlml_wrt_w = T.jacobian(nlml_ss.sum(),self.w)

        utils.print_with_stamp('Compiling sparse spectral training loss function',self.name)
        self.nlml_ss = F((),nlml_ss,name='%s>nlml'%(self.name), profile=self.profile, mode=self.compile_mode, allow_input_downcast=True)
        utils.print_with_stamp('Compiling jacobian of sparse spectral training loss function',self.name)
        self.dnlml_ss = F((),(nlml_ss,dnlml_wrt_lh,dnlml_wrt_w),name='%s>dnlml'%(self.name), profile=self.profile, mode=self.compile_mode, allow_input_downcast=True)



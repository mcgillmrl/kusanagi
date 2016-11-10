import os
import numpy as np
import theano
import theano.tensor as T

from collections import OrderedDict
from functools import partial
from scipy.optimize import minimize, basinhopping
from scipy.cluster.vq import kmeans
from theano import function as F, shared as S
from theano.tensor.nlinalg import matrix_dot
from theano.sandbox.linalg import psd,matrix_inverse,det,cholesky
from theano.tensor.slinalg import solve_lower_triangular, solve_upper_triangular

import cov
import SNRpenalty
from kusanagi import utils
from kusanagi.base.Loadable import Loadable

class GP(Loadable):
    def __init__(self, X_dataset=None, Y_dataset=None, name='GP', idims=None, odims=None, profile=theano.config.profile, uncertain_inputs=False, snr_penalty=SNRpenalty.SEard, filename=None, **kwargs):
        # theano options
        self.profile= profile
        self.compile_mode = theano.compile.get_default_mode()#.excluding('scanOp_pushout_seqs_ops')

        # GP options
        self.max_evals = kwargs['max_evals'] if 'max_evals' in kwargs else 500
        self.conv_thr = kwargs['conv_thr'] if 'conv_thr' in kwargs else 1e-12
        self.min_method = kwargs['min_method'] if 'min_method' in kwargs else "L-BFGS-B"
        self.state_changed = True
        self.should_recompile = False
        self.trained = False
        self.uncertain_inputs = uncertain_inputs
        self.snr_penalty = snr_penalty
        self.covs = (cov.SEard, cov.Noise)
        self.fixed_params = []
        
        # dimension related variables
        self.N = 0
        if X_dataset is None:
            if idims is None:
                raise ValueError('You need to either provide X_dataset (n x idims numpy array) or a value for idims') 
            self.D = idims

        if Y_dataset is None:
            if odims is None:
                raise ValueError('You need to either provide Y_dataset (n x odims numpy array) or a value for odims') 
            self.E = odims

        #symbolic varianbles
        self.param_names = []
        self.loghyp = None; self.logsn = None
        self.X = None; self.Y = None;
        self.iK = None; self.L = None; self.beta = None; self.nigp=None; self.Y_var=None; self.X_cov=None
        self.kernel_func = None
        self.loss_fn = None; self.dloss_fn=None

        # compiled functions
        self.predict_fn = None
        self.predict_d_fn = None
        
        # name of this class for printing command line output and saving
        self.name = name
        # filename for saving
        self.filename = '%s_%d_%d_%s_%s'%(self.name,self.D,self.E,theano.config.device,theano.config.floatX)
        if filename is not None:
            self.filename = filename
            Loadable.__init__(self,name=name,filename=self.filename)
            self.load()
        else:
            Loadable.__init__(self,name=name,filename=self.filename)
        

        # register theanno functions and shared variables for saving
        self.register_types([T.sharedvar.SharedVariable, theano.compile.function_module.Function])
        # register additional variables for saving
        self.register(['trained', 'param_names', 'fixed_params'])
        
        # initialize the class if no pickled version is available
        if X_dataset is not None and Y_dataset is not None:
            utils.print_with_stamp('Initialising new GP regressor',self.name)
            self.set_dataset(X_dataset,Y_dataset)
            utils.print_with_stamp('Finished initialising GP regressor',self.name)
        
        self.ready = False

    def load(self, output_folder=None,output_filename=None):
        ''' loads the state from file, and initializes additional variables'''
        # load state
        super(GP,self).load(output_folder,output_filename)
        
        # initialize missing variables
        if hasattr(self,'X') and self.X:
            self.N = self.X.get_value(borrow=True).shape[0]
            self.D = self.X.get_value(borrow=True).shape[1]
        if hasattr(self,'Y') and self.Y:
            self.E = self.Y.get_value(borrow=True).shape[1]
        if hasattr(self,'loghyp') and self.loghyp:
            self.logsn = self.loghyp[:,-1]

    def get_dataset(self):
        return self.X.get_value(), self.Y.get_value()

    def set_dataset(self,X_dataset,Y_dataset,X_cov=None,Y_var=None):
        # ensure we don't change the number of input and output dimensions ( the number of samples can change)
        assert X_dataset.shape[0] == Y_dataset.shape[0], "X_dataset and Y_dataset must have the same number of rows"
        if hasattr(self,'X') and self.X:
            assert self.X.get_value(borrow=True).shape[1] == X_dataset.shape[1]
        if hasattr(self,'Y') and self.Y:
            assert self.Y.get_value(borrow=True).shape[1] == Y_dataset.shape[1]
        
        # first, convert numpy arrays to appropriate type
        X_dataset = X_dataset.astype( 'float64' )
        Y_dataset = Y_dataset.astype( 'float64' )
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
        
        self.X_cov = X_cov
        if Y_var is not None:
            if self.Y_var is None:
                self.Y_var = S(Y_var,name='%s>Y_var'%(self.name),borrow=True)
            else:
                self.Y_var.set_value(Y_var,borrow=True)

        if not self.trained:
            # init log hyperparameters
            self.init_params()

        # we should be saving, since we updated the trianing dataset
        self.state_changed = True
        if (self.N > 0):
            self.ready = True

    def append_dataset(self,X_dataset,Y_dataset,X_cov=None,Y_var=None):
        if self.X is None:
            self.set_dataset(X_dataset,Y_dataset,X_cov,Y_var)
        else:
            X_ = np.vstack((self.X.get_value(), X_dataset.astype(self.X.dtype)))
            Y_ = np.vstack((self.Y.get_value(), Y_dataset.astype(self.Y.dtype)))
            X_cov_ = None
            if X_cov is not None and hasattr(self,'X_cov') and self.X_cov:
                X_cov_ = np.vstack((self.X_cov, X_cov.astype(self.X_cov.dtype)))
            Y_var_ = None
            if Y_var is not None and hasattr(self,'Y_var'):
                Y_var_ = np.vstack((self.Y_var.get_value(), Y_var.astype(self.Y_var.dtype)))
            
            self.set_dataset(X_,Y_,X_cov_,Y_var_)

    def init_params(self):
        utils.print_with_stamp('Initialising parameters' ,self.name)
        idims = self.D; odims = self.E; 
        # initialize the loghyperparameters of the gp ( this code supports squared exponential only, at the moment)
        X = self.X.get_value(); Y = self.Y.get_value()
        loghyp = np.zeros((odims,idims+2))
        loghyp[:,:idims] = 0.5*X.std(0,ddof=1)
        loghyp[:,idims] = 0.5*Y.std(0,ddof=1)
        loghyp[:,idims+1] = 0.1*loghyp[:,idims]
        loghyp = np.log(loghyp)
        
        # set params will either create the loghyp attribute, or update its value
        self.set_params({'loghyp': loghyp})
        # create logsn (used in PILCO)
        if self.logsn is None:
            self.logsn = self.loghyp[:,-1]

    def set_params(self, params):
        if type(params) is list:
            params = dict(zip(self.param_names,params))
        for pname in params.keys():
            # create shared variable if it doesn't exist
            if pname not in self.__dict__ or self.__dict__[pname] is None:
                p = S(params[pname],name='%s>%s'%(self.name,pname),borrow=True)
                self.__dict__[pname] = p
                if pname not in self.param_names:
                    self.param_names.append(pname)
            # otherwise, update the value of the shared variable
            else:
                p = self.__dict__[pname]
                pv = params[pname].reshape(p.get_value(borrow=True).shape)
                p.set_value(pv,borrow=True)

    def get_params(self, symbolic=False, as_dict=False, ignore_fixed=True):
        if ignore_fixed:
            params = [ self.__dict__[pname] for pname in self.param_names if (pname in self.__dict__ and self.__dict__[pname] and not pname in self.fixed_params) ]
        else:
            params = [ self.__dict__[pname] for pname in self.param_names if (pname in self.__dict__ and self.__dict__[pname]) ]

        if not symbolic:
            params = [ p.get_value() for p in params]
        if as_dict:
            params = dict(zip(self.param_names,params))
        return params

    def get_all_shared_vars(self, as_dict=False):
        if as_dict:
            return [(attr_name,self.__dict__[attr_name]) for attr_name in self.__dict__.keys() if isinstance(self.__dict__[attr_name],T.sharedvar.SharedVariable)]
        else:
            return [attr for attr in self.__dict__.values() if isinstance(attr,T.sharedvar.SharedVariable)]

    def init_loss(self, cache_vars=True, compile_funcs=True):
        utils.print_with_stamp('Initialising expression graph for full GP training loss function',self.name)
        idims = self.D
        odims = self.E

        # these are shared variables for the kernel matrix, its cholesky decomposition and K^-1 dot Y
        if self. iK is None:
            self.iK = S(np.zeros((self.E,self.N,self.N),dtype='float64'), name="%s>iK"%(self.name))
        if self.L is None:
            self.L = S(np.zeros((self.E,self.N,self.N),dtype='float64'), name="%s>L"%(self.name))
        if self.beta is None:
            self.beta = S(np.zeros((self.E,self.N),dtype='float64'), name="%s>beta"%(self.name))
        if self.X_cov is not None and self.nigp is None:
            self.nigp = S(np.zeros((self.E,self.N),dtype='float64'), name="%s>nigp"%(self.name))

        N = self.X.shape[0].astype('float64')
        
        def log_marginal_likelihood(Y,loghyp,i,X,EyeN,nigp=None,y_var=None):
            # initialise the (before compilation) kernel function
            loghyps = (loghyp[:idims+1],loghyp[idims+1])
            kernel_func = partial(cov.Sum, loghyps, self.covs)

            # We initialise the kernel matrices (one for each output dimension)
            K = kernel_func(X)
            # add the contribution from the input noise
            if nigp:
                K += T.diag(nigp[i])
            # add the contribution from the output uncertainty (acts as weight)
            if y_var:
                K += T.diag(y_var[i])
            L = cholesky(K)
            iK = solve_upper_triangular(L.T, solve_lower_triangular(L,EyeN))
            Yc = solve_lower_triangular(L,Y)
            beta = solve_upper_triangular(L.T,Yc)

            # And finally, the negative log marginal likelihood ( again, one for each dimension; although we could share
            # the loghyperparameters across all output dimensions and train the GPs jointly)
            loss = 0.5*(Yc.T.dot(Yc) + 2*T.sum(T.log(T.diag(L))) + N*T.log(2*np.pi) )

            return loss,iK,L,beta
        
        nseq = [self.X,T.eye(self.X.shape[0])]
        if self.nigp:
            nseq.append(self.nigp)
        if self.Y_var:
            nseq.append(self.Y_var.T)
        (loss,iK,L,beta),updts = theano.scan(fn=log_marginal_likelihood, sequences=[self.Y.T,self.loghyp,T.arange(self.X.shape[0])], non_sequences=nseq, allow_gc=False)

        iK = T.unbroadcast(iK,0) if iK.broadcastable[0] else iK
        L = T.unbroadcast(L,0) if L.broadcastable[0] else L
        beta = T.unbroadcast(beta,0) if beta.broadcastable[0] else beta
    
        if cache_vars:
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
            loss += self.snr_penalty(self.loghyp)

        # Compute the gradients for the sum of loss for all output dimensions
        dloss = T.grad(loss.sum(),self.loghyp)

        # Compile the theano functions
        if compile_funcs:
            utils.print_with_stamp('Compiling full GP training loss function',self.name)
            self.loss_fn = F((),loss,name='%s>loss'%(self.name), profile=self.profile, mode=self.compile_mode, allow_input_downcast=True, updates=updts)
            utils.print_with_stamp('Compiling gradient of full GP training loss function',self.name)
            self.dloss_fn = F((),(loss,dloss),name='%s>dloss'%(self.name), profile=self.profile, mode=self.compile_mode, allow_input_downcast=True, updates=updts)
        self.state_changed = True # for saving
    
    def init_predict(self, init_loss=True, compile_funcs=True):
        if init_loss and self.loss_fn is None:
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
        
        # compile prediction
        utils.print_with_stamp('Compiling mean and variance of prediction',self.name)
        self.predict_fn = F(input_vars,prediction,name='%s>predict_'%(self.name), profile=self.profile, mode=self.compile_mode, allow_input_downcast=True)
        self.state_changed = True # for saving

    def predict_symbolic(self,mx,Sx):
        idims = self.D
        odims = self.E

        # compute the mean and variance for each output dimension
        def predict_odim(L,beta,loghyp,X,mx):
            loghyps = (loghyp[:idims+1],loghyp[idims+1])
            kernel_func = partial(cov.Sum, loghyps, self.covs)

            k = kernel_func(mx[None,:],X)
            mean = k.dot(beta)
            kc = solve_lower_triangular(L,k.flatten())
            variance = kernel_func(mx[None,:],all_pairs=False) - kc.dot(kc)

            return mean, variance
        
        (M,S), updts = theano.scan(fn=predict_odim, sequences=[self.L,self.beta,self.loghyp], non_sequences=[self.X,mx], allow_gc = False)

        # reshape output variables
        M = M.flatten()
        S = T.diag(S.flatten())
        V = T.zeros((self.D,self.E))

        return M,S,V
    
    def predict(self,mx,Sx = None):
        predict = None
        if self.predict_fn is None or self.should_recompile:
            self.init_predict()
        predict = self.predict_fn

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
        self.set_params({'loghyp': loghyp})
        if self.nigp:
            # update the nigp parameter using the derivative of the mean function
            dM2 = self.dM2_fn()
            nigp = ((dM2[:,:,:,None]*self.X_cov[None]).sum(2)*dM2).sum(-1)
            self.nigp.set_value(nigp)

        loss,dloss = self.dloss_fn()
        loss = loss.sum()
        dloss = dloss.flatten()
        # on a 64bit system, scipy optimize complains if we pass a 32 bit float
        res = (loss.astype(np.float64), dloss.astype(np.float64))
        utils.print_with_stamp('%s'%(str(res[0])),self.name,True)

        return res

    def train(self):
        if self.loss_fn is None or self.should_recompile:
            self.init_loss()

        if self.nigp and not hasattr(self, 'dM2_fn'):
            idims = self.D
            utils.print_with_stamp('Compiling derivative of mean function at training inputs',self.name)
            # we need to evaluate the derivative of the mean function at the training inputs
            def dM2_f_i(mx,beta,loghyp,X):
                loghyps = (loghyp[:idims+1],loghyp[idims+1])
                kernel_func = partial(cov.Sum, loghyps, self.covs)
                k = kernel_func(mx[None,:],X).flatten()
                mean = k.dot(beta)
                dmean = theano.tensor.jacobian(mean.flatten(),mx)
                return dmean.flatten()**2
            
            def dM2_f(beta,loghyp,X):
                # iterate over training inputs
                dM2_o, updts = theano.scan(fn=dM2_f_i, sequences=[X], non_sequences=[beta,loghyp,X], allow_gc = False)
                return dM2_o

            # iterate over output dimensions
            dM2, updts = theano.scan(fn=dM2_f, sequences=[self.beta,self.loghyp], non_sequences=[self.X], allow_gc = False)

            self.dM2_fn = F((),dM2,name='%s>dM2'%(self.name), profile=self.profile, mode=self.compile_mode, allow_input_downcast=True, updates=updts)


        loghyp0 = self.loghyp.eval()
        utils.print_with_stamp('Current hyperparameters:\n%s'%(loghyp0),self.name)
        utils.print_with_stamp('loss: %s'%(np.array(self.loss_fn())),self.name)
        m_loss = utils.MemoizeJac(self.loss)
        self.n_evals=0
        try:
            opt_res = minimize(m_loss, loghyp0, jac=m_loss.derivative, method=self.min_method, tol=self.conv_thr, options={'maxiter': self.max_evals})
        except ValueError:
            opt_res = minimize(m_loss, loghyp0, jac=m_loss.derivative, method='CG', tol=self.conv_thr, options={'maxiter': self.max_evals})
        print ''
        loghyp = opt_res.x.reshape(loghyp0.shape)
        self.state_changed = not np.allclose(loghyp0,loghyp,1e-6,1e-9)
        self.set_params({'loghyp': loghyp})
        utils.print_with_stamp('New hyperparameters:\n%s'%(self.loghyp.eval()),self.name)
        utils.print_with_stamp('loss: %s'%(np.array(self.loss_fn())),self.name)
        self.trained = True

class GP_UI(GP):
    def __init__(self, X_dataset=None, Y_dataset=None, name = 'GP_UI', idims=None, odims=None, profile=False, **kwargs):
        super(GP_UI, self).__init__(X_dataset,Y_dataset,name=name,idims=idims,odims=odims,profile=profile,uncertain_inputs=True, **kwargs)

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
        iLdotSx = iL.dot(Sx.astype('float64')) # force the matrix inverse to be done with double precision
        B = T.stack([iLdotSx[i].dot(iL[i]) for i in xrange(odims)]) + T.eye(idims)                              #TODO vectorize this
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
        R = T.dot((Lambda.dimshuffle(0,'x',1,2) + Lambda).transpose(0,1,3,2),Sx.astype('float64').T).transpose(0,1,3,2) + T.eye(idims) # again forcing the matrix inverse to be done with double precision
        z_= Lambda.dot(zeta.T).transpose(0,2,1) 
        
        M2 = T.zeros((self.E,self.E),dtype='float64')
        # initialize indices
        indices = [ T.as_index_variable(idx) for idx in np.triu_indices(self.E) ]

        def second_moments(i,j,M2,beta,iK,sf2,R,logk_c,logk_r,z_,Sx):
            # This comes from Deisenroth's thesis ( Eqs 2.51- 2.54 )
            Rij = R[i,j]
            n2 = logk_c[i] + logk_r[j] + utils.maha(z_[i],-z_[j],0.5*matrix_inverse(Rij).dot(Sx))
            Q = T.exp( n2 )/T.sqrt(det(Rij))
            # Eq 2.55
            m2 = matrix_dot(beta[i], Q, beta[j])
            
            m2 = theano.ifelse.ifelse(T.eq(i,j), m2 - T.sum(iK[i]*Q) + sf2[i], m2)
            M2 = T.set_subtensor(M2[i,j], m2)
            M2 = theano.ifelse.ifelse(T.eq(i,j), M2 , T.set_subtensor(M2[j,i], m2))
            return M2

        M2_,updts = theano.scan(fn=second_moments, 
                               sequences=indices,
                               outputs_info=[M2],
                               non_sequences=[self.beta,self.iK,sf2,R,logk_c,logk_r,z_,Sx],
                               allow_gc=False)
        M2 = M2_[-1]
        S = M2 - T.outer(M,M)

        return M,S,V
        
class SPGP(GP):
    def __init__(self, X_dataset=None, Y_dataset=None, name = 'SPGP', idims=None, odims=None, profile=False, n_basis = 100, uncertain_inputs=False, **kwargs):
        self.X_sp = None # inducing inputs (symbolic variable)
        self.loss_sp_fn = None
        self.dloss_sp_fn = None
        self.beta_sp = None
        self.iKmm = None
        self.iBmm = None
        self.Lmm = None
        self.Amm = None
        self.should_recompile = False
        self.n_basis = n_basis
        # intialize parent class params
        GP.__init__(self,X_dataset,Y_dataset,name=name,idims=idims,odims=odims,profile=profile,uncertain_inputs=uncertain_inputs, **kwargs)

    def init_pseudo_inputs(self):
        assert self.N > self.n_basis, "Dataset must have more than n_basis [ %n ] to enable inference with sparse pseudo inputs"%(self.n_basis)
        self.should_recompile = True
        # pick initial cluster centers from dataset
        X = self.X.get_value()
        X_sp_ = utils.kmeanspp(X,self.n_basis)

        # perform kmeans to get initial cluster centers
        utils.print_with_stamp('Initialising pseudo inputs',self.name)
        X_sp_, dist = kmeans(X, X_sp_, iter=200,thresh=1e-9)
        # initialize symbolic tensor variable if necessary (this will create the self.X_sp atttribute)
        self.set_params({'X_sp': X_sp_})

    def set_dataset(self,X_dataset,Y_dataset):
        # set the dataset on the parent class
        super(SPGP, self).set_dataset(X_dataset,Y_dataset)
        if self.N <= self.n_basis:
            utils.print_with_stamp('Dataset is not large enough for using pseudo inputs. Training full GP.',self.name)
            self.X_sp = None
            self.loss_sp_fn = None
            self.dloss_sp_fn = None
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
        # here loss and dloss have already been innitialised, sow e can replace loss and dloss
        # only if we have enough data to train the pseudo inputs ( i.e. self.N > self.n_basis)
        if self.N > self.n_basis:
            utils.print_with_stamp('Initialising FITC training loss function',self.name)
            self.should_recompile = False
            odims = self.E
            idims = self.D
            
            # initialize shared variables
            if self.iKmm is None:
                self.iKmm = S(np.zeros((self.E,self.n_basis,self.n_basis),dtype='float64'), name="%s>iKmm"%(self.name))
            if self.Lmm is None:
                self.Lmm = S(np.zeros((self.E,self.n_basis,self.n_basis),dtype='float64'), name="%s>Lmm"%(self.name))
            if self.Amm is None:
                self.Amm = S(np.zeros((self.E,self.n_basis,self.n_basis),dtype='float64'), name="%s>Amm"%(self.name))
            if self.iBmm is None:
                self.iBmm = S(np.zeros((self.E,self.n_basis,self.n_basis),dtype='float64'), name="%s>iBmm"%(self.name))
            if self.beta_sp is None:
                self.beta_sp = S(np.zeros((self.E,self.n_basis),dtype='float64'), name="%s>beta_sp"%(self.name))

            # initialize the training loss function of the sparse FITC approximation
            def log_marginal_likelihood(Y,loghyp,X,X_sp,EyeM):
                # TODO allow for different pseudo inputs for each dimension
                # initialise the (before compilation) kernel function
                loghyps = (loghyp[:idims+1],loghyp[idims+1])
                kernel_func = partial(cov.Sum, loghyps, self.covs)

                ll = T.exp(loghyp[:idims])
                sf2 = T.exp(2*loghyp[idims])
                sn2 = T.exp(2*loghyp[idims+1])
                N = X.shape[0].astype('float64')
                M = X_sp.shape[0].astype('float64')

                ridge = 1e-6
                Kmm = kernel_func(X_sp) + ridge*EyeM
                Kmn = kernel_func(X_sp, X)
                Lmm = cholesky(Kmm)
                iKmm = solve_upper_triangular(Lmm.T, solve_lower_triangular(Lmm,EyeM))
                Lmn  = solve_lower_triangular(Lmm,Kmn)
                diagQnn =  T.diag(Lmn.T.dot(Lmn))

                # Gamma = diag(Knn - Qnn) + sn2*I
                Gamma = sf2 - diagQnn + sn2
                Gamma_inv = 1.0/Gamma

                # these operations are done to avoid inverting K_sp = (Qnn+Gamma)
                Kmn_ = Kmn*T.sqrt(Gamma_inv)                    # Kmn_*Gamma^-.5
                Yi = Y*(T.sqrt(Gamma_inv))                      # Gamma^-.5* Y
                Bmm = Kmm + (Kmn_).dot(Kmn_.T)                  # Kmm + Kmn * Gamma^-1 * Knm
                Amm = cholesky(Bmm)
                iBmm = solve_upper_triangular(Amm.T, solve_lower_triangular(Amm,EyeM))

                Yci = solve_lower_triangular(Amm,Kmn_.dot(Yi) )
                beta_sp = solve_upper_triangular(Amm.T,Yci)

                log_det_K_sp = T.sum(T.log(Gamma)) - 2*T.sum(T.log(T.diag(Lmm))) + 2*T.sum(T.log(T.diag(Amm)))

                loss_sp = 0.5*( Yi.dot(Yi) - Yci.dot(Yci) + log_det_K_sp + N*np.log(2*np.pi) )

                return loss_sp, iKmm, Lmm, Amm, iBmm, beta_sp
            
            (loss_sp, iKmm, Lmm, Amm, iBmm, beta_sp),updts = theano.scan(fn=log_marginal_likelihood, sequences=[self.Y.T,self.loghyp], non_sequences=[self.X,self.X_sp,T.eye(self.X_sp.shape[0])],allow_gc=False)
            
            iKmm = T.unbroadcast(iKmm,0) if iKmm.broadcastable[0] else iKmm
            Lmm = T.unbroadcast(Lmm,0) if Lmm.broadcastable[0] else Lmm
            Amm = T.unbroadcast(Amm,0) if Amm.broadcastable[0] else Amm
            iBmm = T.unbroadcast(iBmm,0) if iBmm.broadcastable[0] else iBmm
            beta_sp = T.unbroadcast(beta_sp,0) if beta_sp.broadcastable[0] else beta_sp
    
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
            dloss_sp = T.grad(loss_sp.sum(),self.X_sp)

            # Compile the theano functions
            utils.print_with_stamp('Compiling FITC training loss function',self.name)
            self.loss_sp_fn = F((),loss_sp,name='%s>loss_sp'%(self.name), profile=self.profile, mode=self.compile_mode, allow_input_downcast=True, updates=updts)
            utils.print_with_stamp('Compiling gradient of FITC training loss function',self.name)
            self.dloss_sp_fn = F((),(loss_sp,dloss_sp),name='%s>dloss_sp'%(self.name), profile=self.profile, mode=self.compile_mode, allow_input_downcast=True,updates=updts)
            
    def predict_symbolic(self,mx,Sx):
        if self.N <= self.n_basis:
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
        S = T.diag(S.flatten())
        V = T.zeros((self.D,self.E))

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

        if self.N > self.n_basis:
            # train the pseudo input locations
            if self.loss_sp_fn is None:
                self.init_loss()
            utils.print_with_stamp('loss SP: %s'%(np.array(self.loss_sp_fn())),self.name)
            m_loss_sp = utils.MemoizeJac(self.loss_sp)
            opt_res = minimize(m_loss_sp, self.X_sp.get_value(), jac=m_loss_sp.derivative, method=self.min_method, tol=self.conv_thr, options={'maxiter': int(self.max_evals)})
            print ''
            X_sp = opt_res.x.reshape(self.X_sp.get_value(borrow=True).shape)
            self.set_params({'X_sp': X_sp})
            utils.print_with_stamp('loss SP: %s'%(np.array(self.loss_sp_fn())),self.name)
        self.trained = True

class SPGP_UI(SPGP,GP_UI):
    def __init__(self, X_dataset=None, Y_dataset=None, name = 'SPGP_UI', idims=None, odims=None,profile=False, n_basis = 100, **kwargs):
        SPGP.__init__(self,X_dataset,Y_dataset,name=name,idims=idims,odims=odims,profile=profile,n_basis=n_basis,uncertain_inputs=True, **kwargs)

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
        
        M2 = T.zeros((self.E,self.E),dtype='float64')
        # initialize indices
        indices = [ T.as_index_variable(idx) for idx in np.triu_indices(self.E) ]

        def second_moments(i,j,M2,beta,iK,sf2,R,logk_c,logk_r,z_,Sx):
            # This comes from Deisenroth's thesis ( Eqs 2.51- 2.54 )
            Rij = R[i,j]
            n2 = logk_c[i] + logk_r[j] + utils.maha(z_[i],-z_[j],0.5*matrix_inverse(Rij).dot(Sx))
            Q = T.exp( n2 )/T.sqrt(det(Rij))
            # Eq 2.55
            m2 = matrix_dot(beta[i], Q, beta[j])
            
            m2 = theano.ifelse.ifelse(T.eq(i,j), m2 - T.sum(iK[i]*Q) + sf2[i], m2)
            M2 = T.set_subtensor(M2[i,j], m2)
            M2 = theano.ifelse.ifelse(T.eq(i,j), M2 , T.set_subtensor(M2[j,i], m2))
            return M2

        M2_,updts = theano.scan(fn=second_moments, 
                               sequences=indices,
                               outputs_info=[M2],
                               non_sequences=[self.beta_sp,(self.iKmm - self.iBmm),sf2,R,logk_c,logk_r,z_,Sx],
                               allow_gc=False)
        M2 = M2_[-1]
        S = M2 - T.outer(M,M)

        return M,S,V

# RBF network (GP with uncertain inputs/deterministic outputs)
class RBFGP(GP_UI):
    def __init__(self, X_dataset=None, Y_dataset=None, idims=None, odims=None, sat_func=None, name = 'RBFGP',profile=False, **kwargs):
        self.sat_func = sat_func
        if self.sat_func is not None:
            name += '_sat'
        self.loghyp_full=None
        super(RBFGP, self).__init__(X_dataset,Y_dataset,idims=idims,odims=odims,name=name,profile=profile, **kwargs)
        
        # register additional variables for saving
        self.register(['sat_func'])
        self.register(['iK','beta','L'])

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
        iLdotSx = iL.dot(Sx.astype('float64')) # force the matrix inverse to be done with double precision
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
        R = T.dot((Lambda.dimshuffle(0,'x',1,2) + Lambda).transpose(0,1,3,2),Sx.astype('float64').T).transpose(0,1,3,2) + T.eye(idims) # again forcing the matrix inverse to be done with double precision
        z_= Lambda.dot(zeta.T).transpose(0,2,1) 
        
        if self.E == 1:
            # for some reason, compiling the policy gradients breaks when the output dimension of this class is one
            #  TODO: do the same in the other classes that compute second_moments
            # with a scan loop,
            # This comes from Deisenroth's thesis ( Eqs 2.51- 2.54 )
            Rij = R[0,0]
            n2 = logk_c[0] + logk_r[0] + utils.maha(z_[0],-z_[0],0.5*matrix_inverse(Rij).dot(Sx))
            Q = T.exp( n2 )/T.sqrt(det(Rij))
            # Eq 2.55
            m2 = matrix_dot(self.beta[0],Q,self.beta[0].T)
            m2 = m2 + 1e-6
            M2 = T.stack([m2])
        else:
            M2 = T.zeros((self.E,self.E),dtype='float64')
            # initialize indices
            indices = [ T.as_index_variable(idx) for idx in np.triu_indices(self.E) ]

            def second_moments(i,j,M2,beta,R,logk_c,logk_r,z_,Sx):
                # This comes from Deisenroth's thesis ( Eqs 2.51- 2.54 )
                Rij = R[i,j]
                n2 = logk_c[i] + logk_r[j] + utils.maha(z_[i],-z_[j],0.5*matrix_inverse(Rij).dot(Sx))
                Q = T.exp( n2 )/T.sqrt(det(Rij))
                # Eq 2.55
                m2 = matrix_dot(beta[i], Q, beta[j])
                
                m2 = theano.ifelse.ifelse(T.eq(i,j), m2 + 1e-6 , m2)
                M2 = T.set_subtensor(M2[i,j], m2)
                M2 = theano.ifelse.ifelse(T.eq(i,j), M2 , T.set_subtensor(M2[j,i], m2))
                return M2

            M2_,updts = theano.scan(fn=second_moments, 
                                sequences=indices,
                                outputs_info=[M2],
                                non_sequences=[self.beta,R,logk_c,logk_r,z_,Sx],
                                allow_gc=False)
            M2 = M2_[-1]

        S = M2 - T.outer(M,M)

        # apply saturating function to the output if available
        if self.sat_func is not None:
            # saturate the output
            M,S,U = self.sat_func(M,S)
            # compute the joint input output covariance
            V = V.dot(U)

        return M,S,V

class SSGP(GP):
    ''' Sparse Spectral Gaussian Process Regression'''
    def __init__(self, X_dataset=None, Y_dataset=None, name='SSGP', idims=None, odims=None, profile=False, n_basis=100,  uncertain_inputs=False, **kwargs):
        self.w = None
        self.sr = None
        self.Lmm = None
        self.iA = None
        self.beta_ss = None
        self.loss_ss_fn = None
        self.dloss_ss_fn = None
        self.n_basis = n_basis
        GP.__init__(self,X_dataset,Y_dataset,name=name,idims=idims,odims=odims,profile=profile,uncertain_inputs=uncertain_inputs, **kwargs)
    
    def init_loss(self,cache_vars=True):
        super(SSGP, self).init_loss()
        utils.print_with_stamp('Initialising expression graph for sparse spectral training loss function',self.name)
        idims = self.D
        odims = self.E

        if self.iA is None:
            self.iA = S(np.zeros((self.E,2*self.n_basis,2*self.n_basis),dtype='float64'), name="%s>iA"%(self.name))
        if self.Lmm is None:
            self.Lmm = S(np.zeros((self.E,2*self.n_basis,2*self.n_basis),dtype='float64'), name="%s>Lmm"%(self.name))
        if self.beta_ss is None:
            self.beta_ss = S(np.zeros((self.E,2*self.n_basis),dtype='float64'), name="%s>beta_ss"%(self.name))
        
        # sample initial unscaled spectral points
        self.set_spectral_samples()

        #init variables
        N = self.X.shape[0].astype('float64')
        M = self.sr.shape[1].astype('float64')
        Mi = 2*self.sr.shape[1]
        sf2 = T.exp(2*self.loghyp[:,idims])
        sf2M = sf2/M
        sn2 = T.exp(2*self.loghyp[:,idims+1])
        srdotX = self.sr.dot(self.X.T)
        phi_f = T.concatenate( [T.sin(srdotX), T.cos(srdotX)], axis=1 ).astype('float64') # E x 2*n_basis x N
        
        # TODO vectorize these ops
        def log_marginal_likelihood(sf2M, sn2, phi_f, Y, EyeM):
            phi_f.ndim
            A = sf2M*phi_f.dot(phi_f.T) + sn2*EyeM
            Lmm = cholesky(A)
            iA = solve_upper_triangular(Lmm.T, solve_lower_triangular(Lmm,EyeM))
            Yc = solve_lower_triangular(Lmm,(phi_f.dot(Y)))
            beta_ss = sf2M*solve_upper_triangular(Lmm.T,Yc)

            loss_ss = 0.5*( Y.dot(Y) - sf2M*Yc.dot(Yc) )/sn2 + T.sum(T.log(T.diag(Lmm))) + (0.5*N - M)*T.log(sn2) + 0.5*N*np.log(2*np.pi)
            
            return loss_ss,iA,Lmm,beta_ss
        
        (loss_ss,iA,Lmm,beta_ss),updts = theano.scan(fn=log_marginal_likelihood, sequences=[sf2M,sn2,phi_f,self.Y.T], non_sequences=[T.eye(Mi)], allow_gc=False)
        
        iA = T.unbroadcast(iA,0) if iA.broadcastable[0] else iA
        Lmm = T.unbroadcast(Lmm,0) if Lmm.broadcastable[0] else Lmm
        beta_ss = T.unbroadcast(beta_ss,0) if beta_ss.broadcastable[0] else beta_ss

        if cache_vars:
            # we are going to save the intermediate results in the following shared variables, so we can use them during prediction without having to recompute them
            updts = [(self.iA,iA),(self.Lmm,Lmm),(self.beta_ss,beta_ss)]
        else:
            self.iA = iA 
            self.Lmm = Lmm 
            self.beta_ss = beta_ss
            updts=None

        if self.snr_penalty is not None:
            penalty_params = {'log_snr': np.log(1000), 'log_ls': np.log(100), 'log_std': T.log(self.X.std(0)*(N/(N-1.0))), 'p': 30}
            loss_ss += self.snr_penalty(self.loghyp)

        # Compute the gradients for the sum of loss for all output dimensions
        dloss_ss = T.grad(loss_ss.sum(),[self.loghyp,self.w])

        utils.print_with_stamp('Compiling sparse spectral training loss function',self.name)
        self.loss_ss_fn = F((),loss_ss,name='%s>loss_ss'%(self.name), profile=self.profile, mode=self.compile_mode, allow_input_downcast=True, updates=updts)
        utils.print_with_stamp('Compiling gradient of sparse spectral training loss function',self.name)
        self.dloss_ss_fn = F((),(loss_ss,dloss_ss[0],dloss_ss[1]),name='%s>dloss_ss'%(self.name), profile=self.profile, mode=self.compile_mode, allow_input_downcast=True, updates=updts)

    def set_spectral_samples(self,w=None):
        idims = self.D
        odims = self.E
        if w is None:
            w = np.random.randn(self.n_basis,odims,idims).astype('float64')
        else:
            w = w.reshape((self.n_basis,odims,idims)).astype('float64')
    
        self.set_params({'w': w})
        
        if self.sr is None:
            self.sr = (self.w*T.exp(-self.loghyp[:,:idims])).transpose(1,0,2)

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
    def __init__(self, X_dataset=None, Y_dataset=None, name='SSGP_UI', idims=None, odims=None, profile=False, n_basis=100, **kwargs):
        SSGP.__init__(self,X_dataset,Y_dataset,name=name,idims=idims,odims=odims,profile=profile,n_basis=n_basis,uncertain_inputs=True, **kwargs)

    def predict_symbolic(self,mx,Sx):
        #if self.N < self.n_basis:
            # stick with the full GP
        #    return GP_UI.predict_symbolic(self,mx,Sx)

        idims = self.D
        odims = self.E
        
        # precompute some variables
        Ms = self.sr.shape[1]
        sf2M = T.exp(2*self.loghyp[:,idims])/T.cast(Ms,'float64')
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
        V = T.sum( c*beta_ss_r, 2 ).T - T.outer(mx,M) # input outout covariance (notice this is not premultiplied by the input covariance inverse)
        
        srdotSxdotsr_c = srdotSxdotsr.dimshuffle(0,1,'x')
        srdotSxdotsr_r = srdotSxdotsr.dimshuffle(0,'x',1)
        M2 = T.zeros((self.E,self.E),dtype='float64')
        # initialize indices
        indices = [ T.as_index_variable(idx) for idx in np.triu_indices(self.E) ]

        def second_moments(i,j,M2,beta,iA,sn2,sf2M,sr,srdotSx,srdotSxdotsr_c,srdotSxdotsr_r,sin_srdotx,cos_srdotx):
            # compute the second moments of the spectral feature vectors
            siSxsj = srdotSx[i].dot(sr[j].T) #Ms x Ms
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
            Q_up = T.concatenate([cm-cp,sm+sp],axis=1)
            Q_lo = T.concatenate([sp-sm,cm+cp],axis=1)
            Q = T.concatenate([Q_up,Q_lo],axis=0)

            # Compute the second moment of the output
            m2 = 0.5*matrix_dot(beta[i], Q, beta[j].T)
            
            m2 = theano.ifelse.ifelse(T.eq(i,j), m2 + sn2[i]*(1.0 + sf2M[i]*T.sum(self.iA[i]*Q)), m2)
            M2 = T.set_subtensor(M2[i,j], m2)
            M2 = theano.ifelse.ifelse(T.eq(i,j), M2 , T.set_subtensor(M2[j,i], m2))

            return M2

        M2_,updts = theano.scan(fn=second_moments, 
                               sequences=indices,
                               outputs_info=[M2],
                               non_sequences=[self.beta_ss,self.iA,sn2,sf2M,self.sr,srdotSx,srdotSxdotsr_c,srdotSxdotsr_r,sin_srdotx,cos_srdotx],
                               allow_gc=False)
        M2 = M2_[-1]
        S = M2 - T.outer(M,M)

        return M,S,V

class VSSGP(GP):
    ''' Variational Sparse Spectral Gaussian Process Regression'''
    def __init__(self, X_dataset=None, Y_dataset=None, name='VSSGP', idims=None, odims=None, profile=False, n_basis=100, n_components=2, uncertain_inputs=False, **kwargs):
        self.n_basis = n_basis
        self.n_components = n_components
        self.opt_A = True
        self.randomised_phases = True
        super(VSSGP, self).__init__(X_dataset,Y_dataset,name=name,idims=idims,odims=odims,profile=profile,n_basis=n_basis,uncertain_inputs=uncertain_inputs, **kwargs)
    
    def init_params(self):
	''' initializes the parameter set for VSSGP. Some parameters are stored as 
	    their logarithms (or tangent ), to ensure they always fall within the valid 
	    bounds when optimizing '''
	X,Y = self.X.get_value(), self.Y.get_value()
	D,E = X.shape[-1], Y.shape[-1]
	nb,nc = self.n_basis, self.n_components
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
        params = OrderedDict(zip(('mw','logsw','z','logsf','logsn','logL','logp'),(mw,logsw,z,logsf,logsn,logL,logp)))
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
	iL = T.exp(-logL)/(2*np.pi)
	ip = T.exp(-logp) 
	sf2 = T.exp(2*logsf)
	sn2 = T.exp(2*logsn)
	sw = T.exp(logsw)
        b = 2*T.arctan(tanb) + np.pi

	# scale and center the dataset around each inducing input 
	Xsc = 2*np.pi*(X[None,:,:] - z[:,:,None,:])
	# compute the expected value of the cos term wrt the phases b
        Ew = iL[:,None,:]*mw + ip[:,None,:]
        w_dot_x = (Ew[:,:,None,:]*Xsc).sum(-1)
        
        # compute the cos term
        if self.randomised_phases:
            a = w_dot_x + b[:,:,None]
	    mcos = T.cos(a)
            mcos2 = T.cos(2*a)
        else:
            b_hi = b + (2*T.arctan(self.tanb_delta) + np.pi)
            alpha = b[:,:,None] 
            beta = b_hi[:,:,None]
            a = w_dot_x + alpha
            b = w_dot_x + beta
	    mcos = ( T.sin(b) - T.sin(a) )/(beta-alpha) 
            mcos2 = ( T.sin(2*b) - T.sin(2*a) )/(beta-alpha) 

	# compute the expected value wrt w
	e = T.exp( -0.5*(((iL[:,None,None,:]*Xsc)**2)*sw[:,:,None,:]).sum(-1) )
	# get the expected value of the feature vector phi
	sf2K = 2*sf2/nb
	mphi = (sf2K**0.5)[:,None,None]*e*mcos
	# reshape into an N x (nc*nb) matrix
	mphi = mphi.transpose(2,0,1).reshape((N,nc*nb))
	# get the expected value of phi.T.dot(phi) (second moment of phi)
	mcos2 = sf2K[:,None,None]*(0.5 + 0.5*(e**4)*mcos2)
	mphiTphi = mphi.T.dot(mphi)
	mphiTphi = mphiTphi - T.diag(T.diag(mphiTphi)) + T.diag(mcos2.sum(-1).flatten())
	return mphi, mphiTphi, mcos2

    def init_loss(self):
        utils.print_with_stamp('Initialising expression graph for sparse spectral training loss function',self.name)

        if not hasattr(self,'iA'):
            self.iA = S(np.zeros((self.n_basis*self.n_components,self.n_basis*self.n_components),dtype='float64'), name="%s>iA"%(self.name))
        if not hasattr(self,'Lmm'):
            self.Lmm = S(np.zeros((self.n_basis*self.n_components,self.n_basis*self.n_components),dtype='float64'), name="%s>Lmm"%(self.name))
        if not hasattr(self,'beta_ss'):
            self.beta_ss = S(np.zeros((self.n_basis*self.n_components,self.E),dtype='float64'), name="%s>beta_ss"%(self.name))
        
        # intialize parameters of the model
        X,Y = self.X,self.Y
        self.init_params()
	mphi,mphiTphi,mcos2 = self.compute_feature_matrix(self.X)

        K = mphiTphi.shape[0]
        N,E = Y.shape
        EyeK = T.eye(K)
        sn2 = T.exp(2*self.logsn)
        tau = T.exp(-2*self.logsn)
        
        # log likelihood term
        if self.opt_A:
            iSig = mphiTphi*tau + EyeK
            choliSig = cholesky(iSig)
            Sig = solve_upper_triangular(choliSig.T, solve_lower_triangular(choliSig,EyeK)) # sn2*Sig
            mphiTY = mphi.T.dot(Y)
            M = tau*solve_upper_triangular(choliSig.T, solve_lower_triangular(choliSig,mphiTY)) # Sig*EPhi.T*Y
            L_vb = - 0.5*N*E*T.log(tau) + 0.5*N*E*np.log(2*np.pi) + 0.5*tau*(Y**2).sum() - 0.5*tau*(mphiTY*M).sum() + 0.5*E*T.sum(2*T.log(T.diag(choliSig))) 
            updts = [(self.iA,Sig),(self.Lmm,choliSig),(self.beta_ss,M)]
        else:
            M,Sig = self.md.transpose(2,0,1).reshape((E,K)).T, T.exp(self.logsd).transpose(2,0,1).reshape((E,K)).T
            mphiTY = mphi.T.dot(Y)
            L_vb = - 0.5*N*E*T.log(tau) + 0.5*N*E*np.log(2*np.pi) + 0.5*tau*(Y**2).sum() - tau*(mphiTY*M).sum() + 0.5*tau*T.sum(mphiTphi*(T.diag(Sig.sum(-1)) + (M[None,:,:]*M[:,None,:]).sum(-1)))
            updts = [(self.iA,Sig),(self.beta_ss,M)]

        # KL divergence for spectral basis frequencies
        L_vb += 0.5 * (T.exp(self.logsw) + self.mw**2 - self.logsw - 1).sum()
        if not self.randomised_phases:
            b = 2*np.arctan(self.tanb) + np.pi 
            bdelta =2*np.arctan(self.tanb_delta) + np.pi 
            # KL divergence for spectral basis phases
            L_vb +=  (T.log(2*np.pi/(bdelta))).sum() 
            # Contrainte penalty barriers ( to keep b > 0 and b+b_delta < 2*pi
            L_vb +=  -1e-9*(T.log( 2*np.pi + (b+bdelta) ) + T.log(b)).sum()

        if not self.opt_A:
            # KL divergence for fourier coefficients
            L_vb += 0.5 * (T.exp(self.logsd) + self.md**2 - self.logsd - 1).sum()

        # snr penalty ( This helps to prevent overfitting )
        L_vb += (((self.logsf - self.logsn)/np.log(1000))**30).sum()
        # lengthscale penalty ( we don't want them to grow too large as that would make the gradients go to zero )
        L_vb += (((self.logL - np.log(self.X.std(0)))/np.log(100))**30).sum()
        # penalty for large sn. helps escapipng local minima for small datasets.
        L_vb += 100*self.logsn

        # Compute the gradients for the sum of loss for all output dimensions
        dL_vb = T.grad(L_vb.sum(),self.get_params(symbolic=True))
        dretvars = [L_vb]
        dretvars.extend(dL_vb)
        utils.print_with_stamp('Compiling training loss function',self.name)
        self.loss_fn = F((),L_vb,name='%s>loss_ss'%(self.name), profile=self.profile, mode=self.compile_mode, allow_input_downcast=True,updates=updts, on_unused_input='ignore')
        utils.print_with_stamp('Compiling gradient of training loss function',self.name)
        self.dloss_fn = F((),dretvars,name='%s>dloss_ss'%(self.name), profile=self.profile, mode=self.compile_mode, allow_input_downcast=True,updates=updts, on_unused_input='ignore')
    
    def loss(self,new_p,parameter_shapes):
        p=utils.unwrap_params(new_p,parameter_shapes)
        param_names = [pname for pname in self.param_names if pname not in self.fixed_params]
        pdict = dict(zip(param_names,p))
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
        print ''
        new_p = opt_res.x 
        self.state_changed = not np.allclose(p0,new_p,1e-6,1e-9)
        #utils.print_with_stamp('New hyperparameters:\n%s'%(new_p),self.name)
        p=utils.unwrap_params(new_p,parameter_shapes)
        param_names = [pname for pname in self.param_names if pname not in self.fixed_params]
        pdict = dict(zip(param_names,p))
        self.set_params(pdict)
        utils.print_with_stamp('loss: %s'%(np.array(self.loss_fn())),self.name)
        self.trained = True
    
    def predict_symbolic(self,mx,Sx):
        odims = self.E
        idims = self.D
	mphi,mphiTphi,mcos2 = self.compute_feature_matrix(mx[None,:])
	M = mphi.dot(self.beta_ss).flatten()
	sn2 = T.exp(2*self.logsn)
        if self.opt_A:
            S = sn2*T.eye(M.shape[0]) + T.sum(mphiTphi*self.iA)*T.eye(M.shape[0]) + self.beta_ss.T.dot(mphiTphi - mphi.T.dot(mphi)).dot(self.beta_ss)
        else:
            S = sn2*T.eye(M.shape[0]) + T.diag((T.diag(mphiTphi)[:,None]*self.iA).sum(0)) + self.beta_ss.T.dot(mphiTphi - mphi.T.dot(mphi)).dot(self.beta_ss)
        V = T.zeros((self.D,self.E))

        return M,S,V

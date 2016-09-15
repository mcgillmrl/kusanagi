import numpy as np
import theano
import utils

from ghost.regression.GP import RBFGP, SSGP_UI,GP
from ghost.regression.NN import NN
from ghost.regression import cov
from ghost.control.saturation import gSat
from functools import partial
from scipy.optimize import minimize
from scipy.cluster.vq import kmeans2,vq
from utils import gTrig2, gTrig2_np
from base.Loadable import Loadable

# GP based controller
class RBFPolicy(RBFGP):
    def __init__(self, idims=None, odims=None, sat_func=gSat,  m0=None, S0=None, maxU=[10], n_basis=10, angle_dims=[], name='RBFPolicy', filename=None, max_evals=750 ,*kwargs):
        self.maxU = np.array(maxU)
        self.n_basis = n_basis
        self.angle_dims = angle_dims
        self.name = name
        if sat_func:
            # set the model to be a RBF with saturated outputs
            sat_func = partial(sat_func, e=maxU)

        if filename is not None:
            # try loading from file
            super(RBFPolicy, self).__init__(idims=0, odims=0, sat_func=sat_func, max_evals=max_evals, name=self.name, filename=filename)
            #self.load()
        else:
            self.m0 = np.array(m0)
            self.S0 = np.array(S0)
            
            if not idims:
                idims = len(self.m0) + len(self.angle_dims)
            if not odims:
                odims = len(self.maxU)
            super(RBFPolicy, self).__init__(idims=idims, odims=odims, sat_func=sat_func, max_evals=max_evals, name=self.name)
        
        # make sure we always get the parameters in the same order
        self.param_names = ['X','Y','loghyp_full']
        self.X_train = None
        self.Y_train = None
        self.Y_train_var = None
        self.X_cov = None

    def load(self, output_folder=None,output_filename=None):
        ''' loads the state from file, and initializes additional variables'''
        # load state
        super(RBFGP,self).load(output_folder,output_filename)
        
        # initialize mising variables
        self.loghyp = theano.tensor.concatenate([self.loghyp_full[:,:-2], theano.gradient.disconnected_grad(self.loghyp_full[:,-2:])], axis=np.array(1,dtype='int64'))

        # loghyp is no longer the trainable paramter
        if 'loghyp' in self.param_names: self.param_names.remove('loghyp')

    def init_params(self,compile_funcs=True):
        if hasattr(self,'X_train') and self.X_train is not None:
            # initialize tthe mean and covariance of the inputs
            X = self.X_train.get_value(); Y = self.Y_train.get_value()
            idims = X.shape[1]; odims = Y.shape[1]; 
            m0,S0 = X.mean(0), np.cov(X.T,ddof=1);

            # init inputs and targets as subset of dataset
            #idx = np.arange(X.shape[0]); np.random.shuffle(idx); idx= idx[:self.n_basis]
            inputs = utils.kmeanspp(X,self.n_basis)
            inputs,idx = kmeans2(X,inputs)
            targets = np.vstack([ Y[idx==i].mean() for i in xrange(self.n_basis) ])
            
            # initialize log hyper parameters
            l0 = np.zeros((odims,idims+2))
            l0[:,:idims] = 0.5*X.std(0,ddof=1)
            l0[:,idims] = 1.0#Y.std(0,ddof=1)
            l0[:,idims+1] = 0.1#*Y.std(0,ddof=1) + 1e-2
            l0 = np.log(l0)

            # init policy targets according to output distribution
            #targets = np.random.multivariate_normal(Y.mean(0),np.atleast_2d(np.cov(Y.T)),self.n_basis)
            targets = 0.1*np.random.randn(self.n_basis,self.maxU.size)
        else:
            # initialize tthe mean and covariance of the inputs
            m0,S0 = self.m0,self.S0
            if len(self.angle_dims)>0:
                m0, S0 = utils.gTrig2_np(np.array(m0)[None,:], np.array(S0)[None,:,:], self.angle_dims, len(m0))
                m0 = m0.squeeze(); S0 = S0.squeeze();
            # init inputs
            L_noise = np.linalg.cholesky(S0)
            inputs = np.array([m0 + np.random.randn(S0.shape[1]).dot(L_noise) for i in xrange(self.n_basis)]);

            # set the initial log hyperparameters (1 for linear dimensions, 0.7 for angular)
            l0 = np.hstack([np.ones(self.m0.size-len(self.angle_dims)),0.7*np.ones(2*len(self.angle_dims)),1,0.01])
            l0 = np.log(np.tile(l0,(self.maxU.size,1)))

            # init policy targets close to zero
            targets = 0.1*np.random.randn(self.n_basis,self.maxU.size)
        
        
        self.trained = False

        # set the parameters
        self.N = inputs.shape[0]
        self.D = inputs.shape[1]
        self.E = targets.shape[1]
        
        self.set_params( {'X': inputs, 'Y': targets} )
        self.set_params( {'loghyp_full': l0} )
        
        # don't optimize the signal and noise variances
        self.loghyp = theano.tensor.concatenate([self.loghyp_full[:,:-2], theano.gradient.disconnected_grad(self.loghyp_full[:,-2:])], axis=np.array(1,dtype='int64'))

        # loghyp is no longer the trainable paramter
        if 'loghyp' in self.param_names: self.param_names.remove('loghyp')
        
        # initialize loss and predictions
        super(RBFGP,self).init_loss(cache_vars=False,compile_funcs=compile_funcs)
        self.init_predict(init_loss=False)

    def set_dataset(self,X_dataset,Y_dataset,Y_var):
        # first, convert numpy arrays to appropriate type
        X_dataset = X_dataset.astype( 'float64' )
        Y_dataset = Y_dataset.astype( 'float64' )
        # now we create symbolic shared variables
        if self.X_train is None:
            self.X_train = theano.shared(X_dataset,name='%s>X_train'%(self.name),borrow=True)
        else:
            self.X_train.set_value(X_dataset,borrow=True)
        if self.Y_train is None:
            self.Y_train = theano.shared(Y_dataset,name='%s>Y_train'%(self.name),borrow=True)
        else:
            self.Y_train.set_value(Y_dataset,borrow=True)
        
        if Y_var is not None:
            if self.Y_train_var is None:
                self.Y_train_var = theano.shared(Y_var,name='%s>Y_train_var'%(self.name),borrow=True)
            else:
                self.Y_train_var.set_value(Y_var,borrow=True)

        # we should be saving, since we updated the training dataset
        self.state_changed = True
        self.trained = False

    def append_dataset(self,X_dataset,Y_dataset,Y_var=None):
        if self.X is None:
            self.set_dataset(X_dataset,Y_dataset,X_cov,Y_var)
        else:
            X_ = np.vstack((self.X_train.get_value(), X_dataset.astype(self.X_train.dtype)))
            Y_ = np.vstack((self.Y_train.get_value(), Y_dataset.astype(self.Y_train.dtype)))
            Y_var_ = None
            if Y_var is not None and hasattr(self,'Y__train_var'):
                Y_var_ = np.vstack((self.Y_train_var.get_value(), Y_var.astype(self.Y_var.dtype)))
            
            self.set_dataset(X_,Y_,X_cov_,Y_var_)

    def init_loss(self,compile_funcs=True):
        self.init_params(compile_funcs=False)
        if not self.X_train or not self.Y_train:
            return

        X_train = self.X_train
        Y_train = self.Y_train
        Y_train_var = self.Y_train_var

        # compute predictions for the whole dataset
        SX = theano.tensor.zeros((X_train.shape[0],X_train.shape[1],X_train.shape[1]))
        
        def predict_odim(L,beta,loghyp,X,mx,*args):
            idims = self.X.shape[1]
            loghyps = (loghyp[:idims+1],loghyp[idims+1])
            kernel_func = partial(cov.Sum, loghyps, self.covs)

            k = kernel_func(mx,X)
            mean = k.dot(beta)
            return mean
        
        Y_pred, updts = theano.scan(fn=predict_odim, 
                                    sequences=[self.L,self.beta,self.loghyp], 
                                    non_sequences=[self.X,self.X_train]+self.get_all_shared_vars(), 
                                    strict=True, 
                                    allow_gc=False)

        N = self.X_train.shape[0].astype('float64')
        # compute euclidean loss
        delta = Y_pred.T-Y_train
        loss = (0.5*((delta**2)/(Y_train_var+1e-6)).sum() + 1e-3*(self.beta**2).sum())/N
        
        #compute gradients
        dloss = theano.tensor.grad(loss,self.get_params(symbolic=True))
        
        if compile_funcs:
            utils.print_with_stamp('Compiling supervised training loss function',self.name)
            self.loss_fn = theano.function([],loss, updates=updts)
            self.dloss_fn = theano.function([],[loss]+dloss,updates=updts)
    
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
        utils.print_with_stamp('Current hyperparameters:\n',self.name)
        for p in p0:
            print p
        utils.print_with_stamp('loss: %s'%(np.array(self.loss_fn())),self.name)
        m_loss = utils.MemoizeJac(self.loss)
        p0 = utils.wrap_params(p0)
        self.n_evals=0
        opt_res = minimize(m_loss, p0, jac=m_loss.derivative, args=parameter_shapes, method=self.min_method, tol=self.conv_thr, options={'maxiter': self.max_evals})
        print ''
        new_p = opt_res.x 
        self.state_changed = not np.allclose(p0,new_p,1e-6,1e-9)
        new_p=utils.unwrap_params(new_p,parameter_shapes)
        utils.print_with_stamp('New hyperparameters:\n',self.name)
        for p in new_p:
            print p
        param_names = [pname for pname in self.param_names if pname not in self.fixed_params]
        pdict = dict(zip(param_names,new_p))
        self.set_params(pdict)
        utils.print_with_stamp('loss: %s'%(np.array(self.loss_fn())),self.name)
        self.trained = True

    def evaluate(self, m, s=None, t=None, symbolic=False):
        D = m.shape[0]
        if symbolic:
            if s is None:
                s = theano.tensor.zeros((D,D))
            ret = self.predict_symbolic(m,s)
        else:
            if s is None:
                s = np.zeros((D,D))
            ret = self.predict(m,s)
        return ret 

# random controller
class RandPolicy:
    def __init__(self, maxU=[10], random_walk=False):
        self.maxU = np.array(maxU)
        #self.last_u = np.zeros_like(np.array(maxU))
        self.last_u = 0.5*((2*np.random.random(self.maxU.size)-1.0)).reshape(self.maxU.shape)*self.maxU
        self.random_walk=random_walk
        

    def evaluate(self, m, s=None, t=None, symbolic=False):
        if self.random_walk:
            ret = self.last_u + 0.3*((2*np.random.random(self.maxU.size)-1.0)).reshape(self.maxU.shape)*self.maxU
            ret = np.min ( (ret.flatten(), self.maxU.flatten()), axis=0  ) 
            ret = np.max ( (ret.flatten(), -self.maxU.flatten()), axis=0  ) 
            ret = ret.reshape(self.maxU.shape)
        else:
            ret = ((2*np.random.random(self.maxU.size)-1.0)).reshape(self.maxU.shape)*self.maxU

        self.last_u = ret
        U = len(self.maxU)
        D = m.shape[0]
        return ret, np.zeros((U,U)), np.zeros((D,U))

# linear time varying policy
class LocalLinearPolicy(Loadable):
    def __init__(self, H, dt, m0, S0=None, maxU=[10], angle_dims=[], name='LocalLinearPolicy', **kwargs):
        self.maxU = np.array(maxU)
        self.angle_dims = angle_dims
        self.H = H
        self.dt = dt
        self.m0 = m0
        D = len(self.m0)
        self.S0 = S0 if S0 is not None else np.zeros((D,D))
        self.t = 0
        self.noise = 0
        self.name = name
        self.init_params()

        Loadable.__init__(self,name=name,filename=self.filename)
        # register theano functions and shared variables for saving
        self.register_types([theano.tensor.sharedvar.SharedVariable, theano.compile.function_module.Function])

    def init_params(self):
        H_steps = int(np.ceil(self.H/self.dt))
        self.state_changed = False

        # set random (uniform distribution) controls
        u = self.maxU*(2*np.random.random((H_steps,len(self.maxU))) - 1)
        self.u_nominal = theano.shared(u)

        # intialize the nominal states to the appropriate size
        m0, S0 = utils.gTrig2_np(np.array(self.m0)[None,:], np.array(self.S0)[None,:,:], self.angle_dims, len(self.m0))
        self.triu_indices = np.triu_indices(m0.size)
        z0 = np.concatenate([m0.flatten(),S0[0][self.triu_indices]])
        z = np.tile(z0,(H_steps,1))
        self.z_nominal = theano.shared(z)

        # initialize the open loop and feedback matrices 
        I = np.zeros( (H_steps, len(self.maxU)) )
        L = np.zeros( (H_steps, len(self.maxU), z0.size) )
        self.I = theano.shared(I)
        self.L = theano.shared(L)
        
        # set a meaningful filename
        self.filename = self.name+'_'+str(len(self.m0))+'_'+str(len(self.maxU))

    def evaluate(self, m, s=None, t=None, symbolic=False):
        D = m.shape[0]
        if t is not None:
            self.t = t
        t = self.t

        u,z,I,L = self.u_nominal,self.z_nominal,self.I,self.L

        if symbolic:
            T = theano.tensor
        else:
            T = np
            u,z,I,L=u.get_value(),z.get_value(),I.get_value(),L.get_value()
            
        if s is None:
            s = T.zeros((D,D))
        
        # construct flattened state covariance vector
        z_t = T.concatenate([m.flatten(),s[self.triu_indices]])
        # compute control
        u_t = u[t] + I[t] + L[t].dot(z_t - z[t])
        # add random noise if requested (only for non symbolic)
        if not symbolic and self.noise and self.noise > 0:
            u_t += self.noise*T.random.randn(*u_t.shape)

        # limit the controller output
        #u_t = T.maximum(u_t, -self.maxU)
        #u_t = T.minimum(u_t, self.maxU)

        U = u_t.shape[0]
        self.t+=1
        return u_t, T.zeros((U,U)), T.zeros((D,U))

    def get_params(self, symbolic=False, t=None):
        params = [self.u_nominal,self.z_nominal,self.I,self.L]

        if not symbolic:
            params = [ p.get_value() for p in params]
        return params

    def get_all_shared_vars(self):
        return [attr for attr in self.__dict__.values() if isinstance(attr,theano.tensor.sharedvar.SharedVariable)]

class AdjustedPolicy:
    def __init__(self, source_policy, maxU=[10], angle_dims=[], name='AdjustedPolicy', adjustment_model_class=SSGP_UI, use_control_input=True, **kwargs):
        self.use_control_input = use_control_input
        self.angle_dims = angle_dims
        self.name = name
        self.maxU=maxU
        
        self.source_policy = source_policy
        #self.source_policy.init_loss(cache_vars=False)
        #self.source_policy.init_predict()
        self.adjustment_model = adjustment_model_class(idims=self.source_policy.D, odims=self.source_policy.E, name='AdjustmentModel',**kwargs) #TODO we may add a saturating function here
    
    def init_params(self):
        #self.source_policy.init_params() TODO
        pass

    def evaluate(self, m, S=None, t=None, symbolic=False):
        T = theano.tensor if symbolic else np
        # get the output of the source policy
        mu,Su,Cu = self.source_policy.evaluate(m,S,t,symbolic)

        if self.adjustment_model.trained == True:
            # initialize the inputs to the policy adjustment function
            adj_input_m = m
            adj_input_S = S if S is not None else T.zeros((m.size,m.size))

            if self.use_control_input:
                adj_input_m = T.concatenate([adj_input_m,mu])
                # fill input convariance matrix
                q = adj_input_S.dot(Cu)
                Sxu_up = T.concatenate([adj_input_S,q],axis=1)
                Sxu_lo = T.concatenate([q.T,Su],axis=1)
                adj_input_S = T.concatenate([Sxu_up,Sxu_lo],axis=0) # [D+U]x[D+U]

            if symbolic:
                madj,Sadj,Cadj = self.adjustment_model.predict_symbolic(adj_input_m,adj_input_S)
            else:
                madj,Sadj,Cadj = self.adjustment_model.predict(adj_input_m,adj_input_S)

            # compute the adjusted control distribution
            mu = mu + madj
            Sxu_adj = adj_input_S.dot(Cadj)
            Su_adj = Sxu_adj[m.size:]
            Su = Su + Sadj + Su_adj + Su_adj.T
            if S is not None:
                if symbolic:
                    Cu = Cu + theano.tensor.nlinalg.matrix_inverse(S).dot(Sxu_adj[:m.size])
                else:
                    Cu = Cu + np.linalg.pinv(S).dot(Sxu_adj[:m.size])

        return mu,Su,Cu

    def get_params(self, symbolic=False):
        return self.adjustment_model.get_params(symbolic)

    def set_params(self,params):
        return self.adjustment_model.set_params(params)

    def get_all_shared_vars(self):
        return self.source_policy.get_all_shared_vars()+self.adjustment_model.get_all_shared_vars()

    def load(self, output_folder=None,output_filename=None):
        self.adjustment_model.load(output_folder,output_filename)

    def save(self, output_folder=None,output_filename=None):
        self.adjustment_model.save(output_folder,output_filename)

# GP based controller
class NNPolicy(NN):
    def __init__(self, m0=None, S0=None, maxU=[10], hidden_dims=[20,20,20], angle_dims=[], name='NNPolicy', filename=None):
        self.maxU = np.array(maxU)
        self.angle_dims = angle_dims
        self.name = name
        # set the model to be a RBF with saturated outputs
        sat_func = partial(gSat, e=maxU)

        if filename is not None:
            # try loading from file
            self.uncertain_inputs = True
            self.filename = filename
            self.sat_func = sat_func
            self.load()
        else:
            self.m0 = np.array(m0)
            self.S0 = np.array(S0)

            policy_idims = len(self.m0) + len(self.angle_dims)
            policy_odims = len(self.maxU)
            super(NNPolicy, self).__init__(idims=policy_idims, hidden_dims=hidden_dims, odims=policy_odims, sat_func=sat_func, name=self.name)
            
            # check if we need to initialize
            params = self.get_params()
            for p in params:
                if p is None or p.size == 0:
                    self.init_params()
                    break

    def init_params(self):
        self.init_loss()
        self.init_predict()

    def evaluate(self, m, s=None, t=None, derivs=False, symbolic=False):
        D = m.shape[0]
        if symbolic:
            if s is None:
                s = theano.tensor.zeros((D,D))
            ret = self.predict_symbolic(m,s)
        else:
            if s is None:
                s = np.zeros((D,D))
            ret = self.predict(m,s) if not derivs else self.predict_d(m,s)
        return ret 

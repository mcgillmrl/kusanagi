import os,sys
from functools import partial

import numpy as np
import theano
import theano.tensor as T
from scipy.optimize import minimize
from theano import function as F, shared as S
from theano.misc.pkl_utils import dump as t_dump, load as t_load
from theano.sandbox.linalg import psd
from theano.tensor.nlinalg import matrix_inverse,det

sys.setrecursionlimit(10000)
theano.config.reoptimize_unpickled_function = False

def maha(X1,X2=None,M=None):
    ''' Returns the squared Mahalanobis distance'''
    X2 = X1 if X2 is None else X2
    D = []
    if M is None:
        D = T.sum(X1**2,1)[:,None] + T.sum(X2**2,1) - 2*X1.dot(X2.T);
    else:
        X1M = X1.dot(M)
        X2M = X2.dot(M)
        D = T.sum(X1M*X1,1)[:,None] + T.sum(X2M*X2,1) - 2*X1M.dot(X2.T)
    return D

def covSEard(loghyp,X1,X2=None):
    ''' Squared exponential kernel with diagonal scaling matrix (one lengthscale per dimension)'''
    n = 1; idims = 1
    if(X1.ndim == 2):
        n,idims = X1.shape
    elif(X2.ndim == 2):
        n,idims = X2.shape
    else:
        idims = X1.shape[0]
    X2 = X1 if X2 is None else X2
    D = maha(X1,X2,T.diag(T.exp(-2*loghyp[:idims])))
    return T.exp(2*loghyp[idims] - 0.5*D)

def covNoise(loghyp,X1,X2=None,D=None):
    ''' Noise kernel. Takes as an input a distance matrix D and creates a new matrix 
    as Kij = sn2 if Dij == 0 else 0'''
    if D is None:
        X2 = X1 if X2 is None else X2
        D = maha(X1,X2)
    K = T.isclose(D,0)*T.exp(2*loghyp)
    return K

def covSum(loghyp_l, cov_l, X1, X2=None):
    ''' Returns the sum of multiple covariance functions'''
    K = sum([cov_l[i](loghyp_l[i],X1,X2) for i in xrange(len(cov_l)) ] )
    return K

class GP(object):
    def __init__(self, X_dataset, Y_dataset, name='GP'):
        self.X_ = None; self.Y_ = None
        self.X = None; self.Y = None; self.loghyp = None
        self.name = name
        idims = X_dataset.shape[1]
        odims = Y_dataset.shape[1]
        self.filename = '%s_%d_%d.zip'%(self.name,idims,odims)
        try:
            # try loading from pickled file, to avoid recompiling
            self.load()
            self.set_dataset(X_dataset,Y_dataset)
        
        except IOError:
            # initialize the class if no pickled version is available
            assert X_dataset.shape[0] == Y_dataset.shape[0], "X_dataset and Y_dataset must have the same number of rows"
            print '[%f] %s > Initializing GP expression graphs'%(time(),self.name)
            self.set_dataset(X_dataset,Y_dataset)
            self.init_log_likelihood()
            self.init_predict()
            print '[%f] %s > Finished initialising GP'%(time(),self.name)
            self.save()
    
    def set_dataset(self,X_dataset,Y_dataset):
        # ensure we don't change the number of input and output dimensions ( the number of samples can change)
        if self.X_ is not None:
            assert self.X_.shape[1] == X_dataset.shape[1]
        if self.Y_ is not None:
            assert self.Y_.shape[1] == Y_dataset.shape[1]
        # first, assign the numpy arrays to class members
        self.X_ = X_dataset
        self.Y_ = Y_dataset
        idims = X_.shape[1]
        odims = Y_.shape[1]
        N = X_.shape[0]
        self.idims = idims
        self.odims = odims
        self.N = N

        # and initialize the loghyperparameters of the gp ( this code supports squared exponential only, at the moment)
        self.loghyp_ = np.zeros((odims,idims+2))
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

    def set_loghyp(self, loghyp):
        np.copyto(self.loghyp_,loghyp)

    def init_log_likelihood(self):
        idims = self.idims
        odims = self.odims
        # initialize the (before compilation) kernel function
        covs = (covSEard, covNoise)
        loghyps = [ (self.loghyp[i][:idims+1],self.loghyp[i][idims+1]) for i in xrange(odims) ]
        self.kernel_func = [ partial(covSum, loghyps[i], covs) for i in xrange(odims) ]

        # We initialize the kernel matrices (one for each output dimension)
        self.K = [ self.kernel_func[i](self.X) for i in xrange(odims) ]
        self.iK = [ matrix_inverse(psd(self.K[i])) for i in xrange(odims) ]
        self.beta = [ self.iK[i].dot(self.Y[:,i]) for i in xrange(odims) ]

        # And finally, the negative log marginal likelihood ( again, one for each dimension; although we could share
        # the loghyperparameters across all output dimensions and train the GPs jointly)
        nlml = [ 0.5*(self.Y[:,i].T.dot(self.beta[i]) + T.log(det(psd(self.K[i]))) + self.N*T.log(2*np.pi) )/self.N  for i in xrange(odims) ]

        # Compute the gradients for each output dimension independently
        dnlml = [ T.jacobian(nlml[i].flatten(),self.loghyp[i]) for i in xrange(odims)]

        # Compile the theano functions
        print '[%f] %s > Compiling log likelihood'%(time(),self.name)
        self.nlml = F((),nlml)
        #self.nlml_i = [ F((),nlml[i]) for i in xrange(odims)]
        print '[%f] %s > Compiling jacobian of log likelihood'%(time(),self.name)
        self.dnlml = F((),dnlml)
        #self.dnlml_i = [ F((),dnlml[i]) for i in xrange(odims)]
    
    def init_predict(self):
        odims = self.odims
        # and a expresion to evaluate the kernel vector at a new evaluation point
        x_mean = T.dmatrix('x')
        self.k = [ self.kernel_func[i](x_mean,self.X) for i in xrange(odims) ]

        # compute the mean and variance for each output dimension
        mean = [ self.k[i].dot(self.beta[i]) for i in xrange(odims)]
        #variance = [ - (self.k[i] * (self.k[i].dot( self.iK[i] )) ).sum(axis=1)  for i in xrange(odims) ]
        variance = [ T.diag(self.kernel_func[i](x_mean,x_mean)) - (self.k[i] * (self.k[i].dot( self.iK[i] )) ).sum(axis=1)  for i in xrange(odims) ]

        # compile the prediction function
        M = T.stacklists(mean).T
        S = T.stacklists(variance).T
        print '[%f] %s > Compiling mean and variance of prediction'%(time(),self.name)
        self.predict_ = F([x_mean],(M,S))

        # compile the derivatives wrt the evaluation point
        dMdm = T.jacobian(M.flatten(),x_mean)
        dSdm = T.jacobian(S.flatten(),x_mean)

        print '[%f] %s > Compiling derivatives of mean and variance of prediction'%(time(),self.name)
        self.predict_d = F([x_mean], (dMdm,dSdm))
    
    def predict(self,x_mean,x_cov = None):
        odims = self.odims
        if len(x_mean.shape) == 1:
            # convert to row vector
            x_mean = x_mean[None,:]
        res = self.predict_(x_mean)

        return res

    def predict_d(self,x_mean,x_cov=None):
        odims = self.odims
        if len(x_mean.shape) == 1:
            # convert to row vector
            x_mean = x_mean[None,:]
        res = self.predict_d(x_mean)
        return res
    
    def loss(self,loghyp):
        loghyp = loghyp.reshape(self.loghyp_.shape)
        np.copyto(self.loghyp_,loghyp)
        nlml = sum(self.nlml())
        dnlml = np.array(self.dnlml()).flatten()
        #print nlml
        return nlml,dnlml

    def train(self):
        print '[%f] %s > Current hyperparameters:'%(time(),self.name)
        print np.exp(self.loghyp_)
        print '[%f] %s > nlml:'%(time(),self.name)
        print self.nlml()
        opt_res = minimize(self.loss, self.loghyp_, jac=True, method="L-BFGS-B", tol=1e-12, options={'maxiter': 500})
        loghyp = opt_res.x.reshape(self.loghyp_.shape)
        np.copyto(self.loghyp_,loghyp)
        self.save()
        print '[%f] %s > New hyperparameters:'%(time(),self.name)
        print np.exp(self.loghyp_)
        print '[%f] %s > nlml:'%(time(),self.name)
        print self.nlml()

    def load(self):
        with open(self.filename,'rb') as f:
            state = t_load(f)
            self.X = state[0]
            self.Y = state[1]
            self.loghyp = state[2]
            self.set_dataset(state[3],state[4])
            self.set_loghyp(state[5])
            self.nlml = state[6]
            self.dnlml = state[7]
            self.predict_ = state[8]
            self.predict_ = state[9]

    def save(self):
        with open(self.filename,'wb') as f:
            state = (self.X,self.Y,self.loghyp,self.X_,self.Y_,self.loghyp_,self.nlml,self.dnlml,self.predict_,self.predict_d)
            t_dump(state,f,2)

    def __setstate__(self,state):
        self.X_ = state[0]
        self.Y_ = state[1]
        self.loghyp_ = state[2]
        self.idims = state[3]
        self.odims = state[4]
        self.X = state[5]
        self.Y = state[6]
        self.loghyp = state[7]
        self.kernel_func = state[8]
        self.K = state[9]
        self.iK = state[10]
        self.beta = state[11]
        self.nlml = state[12]
        self.dnlml = state[13]
        self.predict_ = state[14]
        self.predict_d = state[15]

class GPUncertainInputs(GP):
    def __init__(self, X_dataset, Y_dataset, name = 'GPUncertainInputs'):
        super(GPUncertainInputs, self).__init__(X_dataset,Y_dataset,name=name)

    def init_predict(self):
        idims = self.idims
        odims = self.odims

        # Note that this handles n_samples inputs
        x_mean = T.dmatrix('x')      # n_samples x idims
        x_cov = T.dtensor3('x_cov')  # n_samples x idims x idims

        #centralize inputs 
        zeta, updts = theano.scan(fn=lambda x_mean_i: self.X - x_mean_i, sequences=x_mean)
        
        # predictive mean and covariance (for each output dimension)
        M = [] # mean
        V = [] # inv(x_cov).dot(input_output_cov)
        M2 = [] # second moment
        for i in xrange(odims):
            # rescale input dimensions by inverse lengthscales
            iL = T.diag(T.exp(-self.loghyp[i][:idims]))
            inp = zeta.dot(iL)
            # predictive mean ( which depends on input covariance )
            B = iL.dot(x_cov.T).transpose(2,0,1).dot(iL) + T.eye(idims)
            t, updts = theano.scan(fn=lambda inp_k,B_k: inp_k.dot(matrix_inverse(B_k)),sequences=(inp,B))
            l = T.exp(-0.5*T.sum(inp*t,2))
            lb = l*self.beta[i] # beta should have been precomputed in init_log_likelihood
            c, updts = theano.scan(fn=lambda B_k: T.exp(2*self.loghyp[i][idims])/T.sqrt(det(B_k)), sequences=B);
            mean = T.sum(lb,1)*c;
            M.append(mean)

            # inv(x_cov) times input output covariance
            tiL = t.dot(iL)
            v, updts = theano.scan(fn=lambda tiL_k,lb_k,c_k: tiL_k.T.dot(lb_k)*c_k,sequences= (tiL,lb,c))
            V.append(v)
            
            #predictive second moment ( only the lower triangular part, including the diagonal)
            def M2_helper(zeta_k,zeta_i_k, zeta_j_k, z_ij_k, R_k, x_cov_k):
                n_k = 0.5*(T.diag(zeta_k.dot(zeta_i_k.T))[:,None] + T.diag(zeta_k.dot(zeta_j_k.T))[None,:] - maha(z_ij_k,z_ij_k,matrix_inverse(psd(R_k)).dot(x_cov_k)))
                t_k = 1.0/T.sqrt(det(psd(R_k)))
                Q_k = t_k*T.exp( 2*(self.loghyp[i][idims] + self.loghyp[j][idims]) - n_k )

                return Q_k
        
            Lambda_i = T.diag(T.exp(-2*self.loghyp[i][:idims]))
            for j in xrange(odims):
                # This comes from Deisenroth's thesis ( Eqs 2.51- 2.54 )
                Lambda_j = T.diag(T.exp(-2*self.loghyp[j][:idims]))

                zeta_i = zeta.dot(Lambda_i)
                zeta_j = zeta.dot(Lambda_j)
                
                Lambda = Lambda_i + Lambda_j

                z_ij = zeta_i + zeta_j

                R = x_cov.dot(Lambda) + T.eye(idims)
    
                Q,updts = theano.scan(fn=M2_helper, sequences=(zeta,zeta_i,zeta_j,z_ij,R,x_cov) )
                
                # Eq 2.55
                m2 = self.beta[i].dot(Q.T).T.dot(self.beta[j])
                if i == j:
                    m2 =  m2 - T.sum(self.iK[i]*Q,(1,2)) + T.exp(2*self.loghyp[i][idims])

                M2.append(m2)
            
        M = T.stacklists(M).T
        V = T.stacklists(V).transpose(1,2,0)
        M2 = T.stacklists(M2).T
        S = M2 - (M[:,:,None]*M[:,None,:]).flatten(2)

        print '[%f] %s > Compiling mean and variance of prediction'%(time(),self.name)
        self.predict_M = F([x_mean,x_cov],M)
        self.predict_V = F([x_mean, x_cov],V)
        self.predict_S = F([x_mean,x_cov],S)

        # compile the derivatives wrt the evaluation point
        dMdm = T.jacobian(M.flatten(),x_mean)
        dVdm = T.jacobian(V.flatten(),x_mean)
        dSdm = T.jacobian(S.flatten(),x_mean)
        dMds = T.jacobian(M.flatten(),x_cov)
        dVds = T.jacobian(V.flatten(),x_cov)
        dSds = T.jacobian(S.flatten(),x_cov)

        print '[%f] %s > Compiling derivatives of mean and variance of prediction'%(time(),self.name)
        self.predict_dMdm = F([x_mean,x_cov], dMdm)
        self.predict_dVdm = F([x_mean,x_cov], dVdm)
        self.predict_dSdm = F([x_mean,x_cov], dSdm)
        self.predict_dMds = F([x_mean,x_cov], dMds)
        self.predict_dVds = F([x_mean,x_cov], dVds)
        self.predict_dSds = F([x_mean,x_cov], dSds)

    def predict(self,x_mean,x_cov):
        odims = self.odims
        if len(x_mean.shape) == 1:
            # convert to row vector
            x_mean = x_mean[None,:]
        n = x_mean.shape[0]

        #M = np.array(self.predict_M(x_mean,x_cov))
        M = self.predict_M(x_mean,x_cov)
        V = self.predict_V(x_mean,x_cov)
        S = self.predict_S(x_mean,x_cov).reshape(n,odims,odims)
        
        return (M,V,S)

    def predict_d(self,x_mean,x_cov):
        odims = self.odims
        if len(x_mean.shape) == 1:
            # convert to row vector
            x_mean = x_mean[None,:]
        n = x_mean.shape[0]

        dMdm = self.predict_dMdm(x_mean,x_cov)
        dVdm = self.predict_dVdm(x_mean,x_cov)
        dSdm = self.predict_dSdm(x_mean,x_cov)
        dMds = self.predict_dMds(x_mean,x_cov)
        dVds = self.predict_dVds(x_mean,x_cov)
        dSds = self.predict_dSds(x_mean,x_cov)

        return (dMdm,dVdm,dSdm,dMds,dVds,dSds)

if __name__=='__main__':
    from time import time
    def f(X):
        #return X[:,0] + X[:,1]**2 + np.exp(-0.5*(np.sum(X**2,1)))
        return np.exp(-500*(np.sum(0.001*(X**2),1)))

    n_samples = 100
    n_test = 10
    idims = 2
    odims = 3

    np.set_printoptions(linewidth=500)
    np.random.seed(31337)
    
    X_ = 10*(np.random.rand(n_samples,idims) - 0.5)
    Y_ = np.empty((n_samples,odims))
    for i in xrange(odims):
        Y_[:,i] =  (i+1)*f(X_) + 0.01*(np.random.rand(n_samples)-0.5)
    
    gp = GP(X_,Y_)
    gp.train()

    gpu = GPUncertainInputs(X_,Y_)
    gpu.train()

    X_ = 10*(np.random.rand(n_test,idims) - 0.5)
    Y_ = np.empty((n_test,odims))
    for i in xrange(odims):
        Y_[:,i] =  (i+1)*f(X_) + 0.01*(np.random.rand(n_test)-0.5)

    r1 = gp.predict(X_,np.zeros((n_test,idims,idims)))
    r2 = gpu.predict(X_,np.zeros((n_test,idims,idims)))

    for i in xrange(n_test):
        print Y_[i,:],','
        print r1[0][i],','
        print r2[0][i],','

        print r1[1][i],','
        print r2[2][i],','
        print '---'

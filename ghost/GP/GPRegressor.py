import os,sys
from functools import partial
from datetime import datetime

import numpy as np
import theano
import theano.tensor as T
from scipy.optimize import minimize
from theano import function as F, shared as S
from theano.misc.pkl_utils import dump as t_dump, load as t_load
from theano.sandbox.linalg import psd
from theano.tensor.nlinalg import matrix_inverse,det

sys.setrecursionlimit(100000)
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
    #print theano.printing.debugprint(D)
    #raw_input()

    K = T.exp(2*loghyp[idims] - 0.5*D)
    return K + 1e-4*T.log(n)*T.eye(n) if X1 is X2 else K

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
        self.idims = X_dataset.shape[1]
        self.odims = Y_dataset.shape[1]
        self.filename = '%s_%d_%d_%s.zip'%(self.name,self.idims,self.odims,theano.config.device)
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
        
        print '[%s] %s > Finished initialising GP'%(str(datetime.now()),self.name)
    
    def set_dataset(self,X_dataset,Y_dataset):
        print '[%s] %s > Updating GP dataset'%(str(datetime.now()),self.name)
        # ensure we don't change the number of input and output dimensions ( the number of samples can change)
        if self.X_ is not None:
            assert self.X_.shape[1] == X_dataset.shape[1]
        if self.Y_ is not None:
            assert self.Y_.shape[1] == Y_dataset.shape[1]
        # first, assign the numpy arrays to class members
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
        np.copyto(self.loghyp_,loghyp)

    def init_log_likelihood(self):
        print '[%s] %s > Initialising expression graph for log likelihood'%(str(datetime.now()),self.name)
        idims = self.idims
        odims = self.odims
        # initialize the (before compilation) kernel function
        covs = (covSEard, covNoise)
        loghyps = [ (self.loghyp[i][:idims+1],self.loghyp[i][idims+1]) for i in xrange(odims) ]
        self.kernel_func = [ partial(covSum, loghyps[i], covs) for i in xrange(odims) ]
        #for i in xrange(odims):
            #print theano.printing.debugprint(loghyps[i][0])
            #print theano.printing.debugprint(loghyps[i][1])
            #raw_input()

        # We initialize the kernel matrices (one for each output dimension)
        self.K = [ self.kernel_func[i](self.X) for i in xrange(odims) ]
        self.iK = [ matrix_inverse(psd(self.K[i])) for i in xrange(odims) ]
        self.iK = [ matrix_inverse(self.K[i]) for i in xrange(odims) ]
        self.beta = [ self.iK[i].dot(self.Y[:,i]) for i in xrange(odims) ]

        #for i in xrange(odims):
            #print theano.printing.debugprint(self.K[i])
            #raw_input()
        self.K_ = F((),self.K)
        # And finally, the negative log marginal likelihood ( again, one for each dimension; although we could share
        # the loghyperparameters across all output dimensions and train the GPs jointly)
        nlml = [ 0.5*(self.Y[:,i].T.dot(self.beta[i]) + T.log(det(psd(self.K[i]))) + self.N*T.log(2*np.pi) )/self.N  for i in xrange(odims) ]

        #for i in xrange(odims):
            #print theano.printing.debugprint(nlml[i])
            #raw_input()

        # Compute the gradients for each output dimension independently
        dnlml = [ T.jacobian(nlml[i].flatten(),self.loghyp[i]) for i in xrange(odims)]

        # Compile the theano functions
        print '[%s] %s > Compiling log likelihood'%(str(datetime.now()),self.name)
        self.nlml = F((),nlml)
        #print theano.printing.debugprint(self.nlml)
        #raw_input()
        #self.nlml_i = [ F((),nlml[i]) for i in xrange(odims)]
        print '[%s] %s > Compiling jacobian of log likelihood'%(str(datetime.now()),self.name)
        self.dnlml = F((),dnlml)
        #self.dnlml_i = [ F((),dnlml[i]) for i in xrange(odims)]
    
    def init_predict(self):
        print '[%s] %s > Initialising expression graph for prediction'%(str(datetime.now()),self.name)
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
        print '[%s] %s > Compiling mean and variance of prediction'%(str(datetime.now()),self.name)
        self.predict_ = F([x_mean],(M,S))

        # compile the derivatives wrt the evaluation point
        dMdm = T.jacobian(M.flatten(),x_mean)
        dSdm = T.jacobian(S.flatten(),x_mean)

        print '[%s] %s > Compiling derivatives of mean and variance of prediction'%(str(datetime.now()),self.name)
        self.predict_d_ = F([x_mean], (dMdm,dSdm))
    
    def predict(self,x_mean,x_cov = None):
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
            res = self.predict_(x_mean, x_cov)
        return res

    def predict_d(self,x_mean,x_cov=None):
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
        loghyp = loghyp.reshape(self.loghyp_.shape)
        np.copyto(self.loghyp_,loghyp)
        nlml = sum(self.nlml())
        dnlml = np.array(self.dnlml()).flatten()
        #print nlml
        #print np.exp(loghyp)
        return nlml,dnlml

    def train(self):
        init_hyp = self.loghyp_.copy()
        print '[%s] %s > Current hyperparameters:'%(str(datetime.now()),self.name)
        print np.exp(self.loghyp_)
        print '[%s] %s > nlml: %s'%(str(datetime.now()),self.name, np.array(self.nlml()))
        opt_res = minimize(self.loss, self.loghyp_, jac=True, method="L-BFGS-B", tol=1e-12, options={'maxiter': 500})
        loghyp = opt_res.x.reshape(self.loghyp_.shape)
        self.should_save = not np.allclose(init_hyp,loghyp,1e-6,1e-9)
        np.copyto(self.loghyp_,loghyp)
        self.save()
        print '[%s] %s > New hyperparameters:'%(str(datetime.now()),self.name)
        print np.exp(self.loghyp_)
        print '[%s] %s > nlml: %s'%(str(datetime.now()),self.name, np.array(self.nlml()))

    def load(self):
        with open(self.filename,'rb') as f:
            print '[%s] %s > Loading compiled GP with %d inputs and %d outputs'%(str(datetime.now()),self.name,self.idims,self.odims)
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
        if self.should_save:
            with open(self.filename,'wb') as f:
                print '[%s] %s > Saving compiled GP with %d inputs and %d outputs'%(str(datetime.now()),self.name,self.idims,self.odims)
                state = (self.X,self.Y,self.loghyp,self.X_,self.Y_,self.loghyp_,self.nlml,self.dnlml,self.predict_,self.predict_d_)
                t_dump(state,f,2)

class GPUncertainInputs(GP):
    def __init__(self, X_dataset, Y_dataset, name = 'GPUncertainInputs'):
        super(GPUncertainInputs, self).__init__(X_dataset,Y_dataset,name=name)

    def init_predict(self):
        print '[%s] %s > Initialising expression graph for prediction'%(str(datetime.now()),self.name)
        idims = self.idims
        odims = self.odims

        # Note that this handles n_samples inputs
        x_mean = T.dmatrix('x')      # n_samples x idims
        x_cov = T.dtensor3('x_cov')  # n_samples x idims x idims

        #centralize inputs 
        zeta, updts = theano.scan(fn=lambda x_mean_i,X: X - x_mean_i, sequences=[x_mean], non_sequences=[self.X], strict=True)
        
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
            t, updts = theano.scan(fn=lambda inp_k,B_k: inp_k.dot(matrix_inverse(B_k)),sequences=[inp,B])
            l = T.exp(-0.5*T.sum(inp*t,2))
            lb = l*self.beta[i] # beta should have been precomputed in init_log_likelihood
            c, updts = theano.scan(fn=lambda B_k,log_sf: T.exp(2*log_sf)/T.sqrt(det(B_k)), sequences=[B], non_sequences=[self.loghyp[i][idims]], strict=True);
            mean = T.sum(lb,1)*c;
            M.append(mean)

            # inv(x_cov) times input output covariance
            tiL = t.dot(iL)
            v, updts = theano.scan(fn=lambda tiL_k,lb_k,c_k: tiL_k.T.dot(lb_k)*c_k,sequences=[tiL,lb,c])
            V.append(v)
            
            #predictive second moment ( only the lower triangular part, including the diagonal)
            def M2_helper(zeta_k,zeta_i_k, zeta_j_k, z_ij_k, R_k, x_cov_k, log_sf2_ij):
                n_k = 0.5*(T.diag(zeta_k.dot(zeta_i_k.T))[:,None] + T.diag(zeta_k.dot(zeta_j_k.T))[None,:] - maha(z_ij_k,z_ij_k,matrix_inverse(psd(R_k)).dot(x_cov_k)))
                t_k = 1.0/T.sqrt(det(psd(R_k)))
                Q_k = t_k*T.exp( log_sf2_ij - n_k )

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
    
                Q,updts = theano.scan(fn=M2_helper, sequences=(zeta,zeta_i,zeta_j,z_ij,R,x_cov), non_sequences=[2*(self.loghyp[i][idims] + self.loghyp[j][idims])], strict=True)
                
                # Eq 2.55
                m2 = self.beta[i].dot(Q.T).T.dot(self.beta[j])
                if i == j:
                    m2 =  m2 - T.sum(self.iK[i]*Q,(1,2)) + T.exp(2*self.loghyp[i][idims])
                M2.append(m2)

        M = T.stacklists(M).T
        V = T.stacklists(V).transpose(1,2,0)
        M2 = T.stacklists(M2).T
        S = M2 - (M[:,:,None]*M[:,None,:]).flatten(2)

        print '[%s] %s > Compiling mean and variance of prediction'%(str(datetime.now()),self.name)
        self.predict_ = F([x_mean,x_cov],(M,S,V))

        # compile the derivatives wrt the evaluation point
        dMdm = T.jacobian(M.flatten(),x_mean)
        dVdm = T.jacobian(V.flatten(),x_mean)
        dSdm = T.jacobian(S.flatten(),x_mean)
        dMds = T.jacobian(M.flatten(),x_cov)
        dVds = T.jacobian(V.flatten(),x_cov)
        dSds = T.jacobian(S.flatten(),x_cov)

        print '[%s] %s > Compiling derivatives of mean and variance of prediction'%(str(datetime.now()),self.name)
        self.predict_d_ = F([x_mean,x_cov], (dMdm,dMds,dSdm,dSds,dVdm,dVds))

def test_random():
    # test function
    def f(X):
        #return X[:,0] + X[:,1]**2 + np.exp(-0.5*(np.sum(X**2,1)))
        return np.exp(-500*(np.sum(0.001*(X**2),1)))

    n_samples = 500
    n_test = 10
    idims = 2
    odims = 1

    np.set_printoptions(linewidth=500)
    np.random.seed(31337)
    
    X_ = 10*(np.random.rand(n_samples,idims) - 0.5)
    Y_ = np.empty((n_samples,odims))
    for i in xrange(odims):
        Y_[:,i] =  (i+1)*f(X_) + 0.01*(np.random.rand(n_samples)-0.5)
    
    #gp = GP(X_,Y_)
    #gp.train()

    gpu = GPUncertainInputs(X_,Y_)
    gpu.train()

    X_ = 10*(np.random.rand(n_test,idims) - 0.5)
    Y_ = np.empty((n_test,odims))
    for i in xrange(odims):
        Y_[:,i] =  (i+1)*f(X_) + 0.01*(np.random.rand(n_test)-0.5)

    #r1 = gp.predict(X_,np.zeros((n_test,idims,idims)))
    r2 = gpu.predict(X_)

    for i in xrange(n_test):
        print Y_[i,:],','
     #   print r1[0][i],','
        print r2[0][i],','

      #  print r1[1][i],','
        print r2[1][i],','
        print '---'

def test_sonar():
    from scipy.io import loadmat
    dataset = loadmat('/media/diskstation/Kingfisher/matlab.mat')
    
    Xd = np.array(dataset['mat'][:,0:2])
    Xd += 1e-2*np.random.rand(*(Xd.shape))
    Yd = np.array(dataset['mat'][:,2])[:,None]

    gp = GP(Xd,Yd)
    #gp = GPUncertainInputs(Xd,Yd)
    #print '[%s] %s > training'%(str(datetime.now()),'main')
    #gp.train()
    #print '[%s] %s > done training'%(str(datetime.now()),'main')
    
    n_test=500
    xg,yg = np.meshgrid ( np.linspace(Xd[:,0].min(),Xd[:,0].max(),n_test) , np.linspace(Xd[:,1].min(),Xd[:,1].max(),n_test) )
    X_test= np.vstack((xg.flatten(),yg.flatten())).T
    n = X_test.shape[0]
    print '[%s] %s > predicting'%(str(datetime.now()),'main')

    M = []; S = []
    batch_size=5000
    for i in xrange(0,n,batch_size):
        next_i = min(i+batch_size,n)
        print 'batch %d , %d'%(i,next_i)
        r = gp.predict(X_test[i:next_i])
        M.append(r[0])
        S.append(r[1])

    M = np.vstack(M)
    S = np.vstack(S)

    print '[%s] %s > done predicting'%(str(datetime.now()),'main')
    from matplotlib import pyplot as plt
    plt.figure()
    plt.imshow(M.reshape(n_test,n_test), origin='lower')

    plt.figure()
    plt.imshow(S.reshape(n_test,n_test), origin='lower')
    plt.show()


if __name__=='__main__':
    #test_random()
    test_sonar()

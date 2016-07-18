from ghost.regression.NN import NN
from matplotlib import pyplot as plt
from utils import gTrig2_np, print_with_stamp
from scipy.signal import convolve2d
from time import time
import numpy as np
from matplotlib import pyplot as plt

def test_random(gp_type='GP',angi=[]):
    # test function
    def f(X):
        #return X[:,0] + X[:,1]**2 + np.exp(-0.5*(np.sum(X**2,1)))
        #return np.exp(-0.25*(np.sum((X**2),1)))*np.sin(2.5*X.sum(1))
        #return 2*np.exp(-0.5*(np.sum(X**2,1))) - 1
        ret=np.zeros((X.shape[0]))
        ret[X.max(1)>0]=1
        ret -= 0.5
        return ret

    n_samples = 500
    n_test = 2000
    idims = 1
    odims = 1
    np.random.seed(31337)
    
    #  ================== train dataset ==================
    Xd = 10*(np.random.rand(n_samples,idims) - 0.5)
    idx = np.argsort(Xd[:,0])
    Xd = Xd[idx]
    Yd = np.empty((n_samples,odims))
    for i in xrange(odims):
        Yd[:,i] =  (i+1)*f(Xd) + 0.01*(np.random.rand(n_samples)-0.5)
    kk = 0.1*convolve2d(np.array([[1,2,3,2,1]]),np.array([[1,2,3,2,1]]).T)/9.0;
    if len(angi)>0:
        ss = convolve2d(np.eye(idims),kk,'same')
        Xd,bbb = gTrig2_np(Xd,np.tile(ss,(n_samples,1)).reshape(n_samples,idims,idims), angi, idims)
    #  ================== test  dataset ==================
    Xtest = 50*(np.random.rand(n_test,idims) - 0.5)
    idx = np.argsort(Xtest[:,0])
    Xtest = Xtest[idx]
    Ytest = np.empty((n_test,odims))
    for i in xrange(odims):
        Ytest[:,i] =  (i+1)*f(Xtest) + 0.01*(np.random.rand(n_test)-0.5)

    if len(angi)>0:
        ss = convolve2d(np.eye(idims),kk,'same')
        Xtest,bbb = gTrig2_np(Xtest,np.tile(ss,(n_test,1)).reshape(n_test,idims,idims), angi, idims)
    
    nn = NN(Xd.shape[1],[100,100,100,100],Yd.shape[1])
    nn.set_dataset(Xd,Yd)
    nn.train()
    import lasagne
    print [p.shape for p in lasagne.layers.get_all_param_values(nn.network)]
    
    plt.figure()
    plt.plot(Xd,Yd)
    plt.figure()
    plt.plot(Xtest,Ytest)
    
    ss = convolve2d(np.eye(Xtest.shape[1]),kk,'same')
    avg_time_per_call = 0
    Ypred = []
    Yvar = []
    for i in xrange(n_test):
        st = time()
        res = nn.predict(Xtest[i,:],ss,derivs=False)
        avg_time_per_call += time()-st

        print Xtest[i,:],','
        print Ytest[i,:],','
        for j in xrange(len(res)):
           print res[j],','
        print avg_time_per_call/(i+1.0)
        print '---'
        Ypred.append(res[0])
        Yvar.append(res[1][0])
    
    Ypred = np.concatenate(Ypred,axis=0)
    Yvar = np.concatenate(Yvar,axis=0)
    lower_bound = Ypred - 2*np.sqrt(Yvar).squeeze()
    upper_bound = Ypred + 2*np.sqrt(Yvar).squeeze()
    plt.fill_between(Xtest.squeeze(), lower_bound, upper_bound, alpha=0.2)
    plt.plot(Xtest,Ypred)
    plt.show()
    print_with_stamp('avg_time_per_call %f'%(avg_time_per_call/n_test),'main')

if __name__=='__main__':
    np.set_printoptions(linewidth=500, precision=17, suppress=True)
    test_random()

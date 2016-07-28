import numpy as np
import argparse
import utils

from ghost.regression.NN import NN
from matplotlib import pyplot as plt
from utils import gTrig_np, gTrig2_np, print_with_stamp
from scipy.signal import convolve2d
from scipy.stats import multivariate_normal
from time import time
from theano import d3viz
from theano.printing import pydotprint

np.set_printoptions(linewidth=500, precision=17, suppress=True)

def test_func1(X,ftype=2):
    if ftype==1:
        ret = np.exp(-0.05*(np.sum((X**2),1)))*np.sin(2.5*X.sum(1))
    else:
        ret=np.zeros((X.shape[0]))
        ret[X.max(1)>0]=1
        ret -= 0.5
        ret = (ret + 0.1*np.sin(2.5*X.sum(1)))
    return ret

def build_dataset(idims=9,odims=6,angi=[],f=test_func1,n_train=500,n_test=50, input_noise=0.01, output_noise=0.01,rand_seed=None):
    if rand_seed is not None:
        np.random.seed(rand_seed)
    #  ================== train dataset ==================
    # sample training points
    x_train = 15*(np.random.rand(n_train,idims) - 0.5)
    # generate the output at the training points
    y_train = np.empty((n_train,odims))
    for i in xrange(odims):
        y_train[:,i] =  (i+1)*f(x_train) + output_noise*(np.random.randn(n_train))
    x_train = gTrig_np(x_train, angi)
    
    #  ================== test  dataset ==================
    # generate testing points
    kk = input_noise*convolve2d(np.array([[1,2,3,2,1]]),np.array([[1,2,3,2,1]]).T)/9.0;
    s_test = convolve2d(np.eye(idims),kk,'same')
    s_test = np.tile(s_test,(n_test,1)).reshape(n_test,idims,idims)
    x_test = 60*(np.random.rand(n_test,idims) - 0.5)
    # generate the output at the test points
    y_test = np.empty((n_test,odims))
    for i in xrange(odims):
        y_test[:,i] =  (i+1)*f(x_test)
    if len(angi)>0:
        x_test,s_test = gTrig2_np(x_test,s_test, angi, idims)

    return (x_train,y_train),(x_test,y_test,s_test)

def write_profile_files(gp):
    d3viz.d3viz(gp.dnlml, 'dnlml.html')
    d3viz.d3viz(gp.predict_fn, 'predict.html')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_train', nargs='?', type=int, help='Number of training samples. Default: 500.', default=500)
    parser.add_argument('--n_test', nargs='?', type=int, help='Number of testing samples. Default: 500', default=500)
    parser.add_argument('--idims', nargs='?', type=int, help='Input dimensions. Default: 4', default=4)
    parser.add_argument('--odims', nargs='?', type=int, help='Output dimensions. Default: 2', default=2)
    parser.add_argument('--noise1', nargs='?', type=float, help='Measurement noise of training targets. Default: 0.01', default=0.01)
    parser.add_argument('--noise2', nargs='?', type=float, help='Noise on test inputs. Default: 0.01', default=0.01)
    args = parser.parse_args()

    idims = args.idims
    odims = args.odims
    n_train = args.n_train
    n_test = args.n_test
    utils.print_with_stamp("Building test dataset",'main')
    train_dataset,test_dataset = build_dataset(idims=idims,odims=odims,n_train=n_train,n_test=n_test, output_noise=args.noise1, input_noise=args.noise2, rand_seed=31337)
    utils.print_with_stamp("Building regressor",'main')
    nn = NN(idims,[100,100,100,100],odims)
    nn.set_dataset(train_dataset[0],train_dataset[1])

    nn.train()
    nn.save()

    utils.print_with_stamp("Testing regressor",'main')
    test_mX = test_dataset[0]
    test_sX = test_dataset[2]
    test_Y = test_dataset[1]
    errors = []
    probs = []
    preds = []
    for i in xrange(n_test):
        ret = nn.predict(test_mX[i], test_sX[i])
        preds.append(ret)
        print '============%04d============'%(i)
        print 'Test Point:\n%s'%(test_mX[i])
        print 'Ground Truth:\n%s'%(test_Y[i])
        print 'Mean Prediction:\n%s'%(ret[0])
        print 'Prediction Covariance:\n%s'%(ret[1])
        print 'Input/Output Covariance:\n%s'%(ret[2])
        errors.append(np.sqrt(((ret[0]-test_Y[i])**2).sum()))
        print 'Error:\t%f'%(errors[-1])
        probs.append(np.log(multivariate_normal.pdf(test_Y[i],mean=ret[0],cov=ret[1])))
        print 'Log Probability of Ground Truth:\t%f'%(probs[-1])

    errors = np.array(errors)
    probs = np.array(probs)
    print '============================='
    print 'Min/Max/Mean Prediction Error:\t %f / %f / %f'%(errors.min(),errors.max(),errors.mean())
    print 'Min/Max/Mean Log Probablity:\t %f / %f / %f'%(probs.min(),probs.max(),probs.mean())

    if idims==1 and odims==1:
        # plot regression result
        idx = np.argsort(train_dataset[0][:,0])
        Xtr = train_dataset[0][idx]
        Ytr = train_dataset[1][idx]
        
        idx = np.argsort(test_dataset[0][:,0])
        Xts = test_dataset[0][idx]
        SXts = test_dataset[2][idx]
        Yts = test_dataset[1][idx]

        plt.figure()
        plt.scatter(Xtr,Ytr)
        plt.plot(Xts,Yts)
        
        Ypred,Yvar,Yio = zip(*preds)
        Ypred = np.concatenate(Ypred,axis=0)[idx]
        Yvar = np.concatenate(Yvar,axis=0)[idx]
        alpha = 0.5
        for i in xrange(4):
            alpha = alpha/2.0
            lower_bound = Ypred - i*np.sqrt(Yvar).squeeze()
            upper_bound = Ypred + i*np.sqrt(Yvar).squeeze()
            plt.fill_between(Xts.squeeze(), lower_bound, upper_bound, alpha=alpha)

        plt.plot(Xts,Ypred)
        plt.show()
        print_with_stamp('avg_time_per_call %f'%(avg_time_per_call/n_test),'main')


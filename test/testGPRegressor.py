from ghost.regression.GP import *
from matplotlib import pyplot as plt
from utils import gTrig_np, gTrig2_np, print_with_stamp
from scipy.signal import convolve2d
from time import time
from theano import d3viz
from theano.printing import pydotprint

np.set_printoptions(linewidth=500, precision=17, suppress=True)

def test_func1(X):
    return np.exp(-0.5*(np.sum((X**2),1)))*np.sin(X.sum(1))

def build_dataset(idims=9,odims=6,angi=[],f=test_func1,n_train=500,n_test=50,rand_seed=None):
    if rand_seed is not None:
        np.random.seed(rand_seed)
    #  ================== train dataset ==================
    # sample training points
    x_train = 10*(np.random.rand(n_train,idims) - 0.5)
    # generate the output at the training points
    y_train = np.empty((n_train,odims))
    for i in xrange(odims):
        y_train[:,i] =  (i+1)*f(x_train) + 0.01*(np.random.rand(n_train)-0.5)
    x_train = gTrig_np(x_train, angi)
    
    #  ================== test  dataset ==================
    # generate testing points
    kk = 0.01*convolve2d(np.array([[1,2,3,2,1]]),np.array([[1,2,3,2,1]]).T)/9.0;
    s_test = convolve2d(np.eye(idims),kk,'same')
    s_test = np.tile(s_test,(n_test,1)).reshape(n_test,idims,idims)
    x_test = 10*(np.random.rand(n_test,idims) - 0.5)
    # generate the output at the test points
    y_test = np.empty((n_test,odims))
    for i in xrange(odims):
        y_test[:,i] =  (i+1)*f(x_test) + 0.01*(np.random.rand(n_test)-0.5)
    if len(angi)>0:
        x_test,s_test = gTrig2_np(x_test,s_test, angi, idims)

    return (x_train,y_train),(x_test,y_test,s_test)

def build_GP(idims=9, odims=6, gp_type='GP', profile=theano.config.profile):
    if gp_type == 'GP_UI':
        gp = GP_UI(idims=idims,odims=odims,profile=profile)
    elif gp_type == 'RBFGP':
        gp = RBFGP(idims=idims,odims=odims,profile=profile)
    elif gp_type == 'SPGP':
        gp = SPGP(idims=idims,odims=odims,profile=profile,n_basis=100)
    elif gp_type == 'SPGP_UI':
        gp = SPGP_UI(idims=idims,odims=odims,profile=profile,n_basis=100)
    elif gp_type == 'SSGP':
        gp = SSGP_UI(idims=idims,odims=odims,profile=profile,n_basis=100)
    elif gp_type == 'SSGP_UI':
        gp = SSGP_UI(idims=idims,odims=odims,profile=profile,n_basis=100)
    else:
        gp = GP(idims=idims,odims=odims,profile=profile)
    return gp

def write_profile_files(gp):
    d3viz.d3viz(gp.dnlml, 'dnlml.html')
    d3viz.d3viz(gp.predict_fn, 'predict.html')

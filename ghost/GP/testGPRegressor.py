from GPRegressor import *

def test_random():
    # test function
    def f(X):
        #return X[:,0] + X[:,1]**2 + np.exp(-0.5*(np.sum(X**2,1)))
        return np.exp(-500*(np.sum(0.001*(X**2),1)))

    n_samples = 100
    n_test = 10
    idims = 7
    odims = 3
    np.random.seed(31337)
    
    X_ = 10*(np.random.rand(n_samples,idims) - 0.5)
    Y_ = np.empty((n_samples,odims))
    for i in xrange(odims):
        Y_[:,i] =  (i+1)*f(X_) + 0.01*(np.random.rand(n_samples)-0.5)
    
    #gp = GP(X_,Y_)
    #gp.train()

    gpu = GP_UI(X_,Y_,profile=True)
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

    if gpu.profile:
        write_profile_files(gpu)

def write_profile_files(gp):
    from theano import d3viz
    root_path = '/localdata/juan/theano/'
    formatter = d3viz.formatting.PyDotFormatter()
    #d3viz.d3viz(gp.K[0],root_path+'html/K'+theano.config.device+'.html')
    #d3viz.d3viz(gp.iK[0],root_path+'html/iK'+theano.config.device+'.html')
    #d3viz.d3viz(gp.beta[0],root_path+'html/beta'+theano.config.device+'.html')

    d3viz.d3viz(gp.nlml,root_path+'html/nlml'+theano.config.device+'.html')
    nlml_graph = formatter(gp.nlml)
    nlml_graph.write_png(root_path+'png/nlml'+theano.config.device+'.png')

    d3viz.d3viz(gp.dnlml,root_path+'html/dnlml'+theano.config.device+'.html')
    dnlml_graph = formatter(gp.nlml)
    dnlml_graph.write_png(root_path+'png/dnlml'+theano.config.device+'.png')

    d3viz.d3viz(gp.predict_,root_path+'html/predict_'+theano.config.device+'.html')
    predict_graph =  formatter(gp.predict_)
    predict_graph.write_png(root_path+'png/predict_'+theano.config.device+'.png')

    #d3viz.d3viz(gp.predict_d_,root_path+'html/predict_d_'+theano.config.device+'.html')
    #predict_d_graph = formatter(gp.predict_d_)
    #predict_d_graph.write_png(root_path+'png/predict_d_'+theano.config.device+'.png')

def test_sonar():
    from scipy.io import loadmat
    dataset = loadmat('/media/diskstation/Kingfisher/matlab.mat')
    
    #idx = np.random.choice(np.arange(dataset['mat'].shape[0]),1500)
    #Xd = np.array(dataset['mat'][idx,0:2])
    #Yd = np.array(dataset['mat'][idx,2])[:,None]
    Xd = np.array(dataset['mat'][:,0:2])
    Yd = np.array(dataset['mat'][:,2])[:,None]

    #gp = GP(Xd,Yd, profile=False)
    gp = GP_UI(Xd,Yd, profile=False)
    #gp = SPGP(Xd,Yd, profile=False, n_inducing = 2)
    utils.print_with_stamp('training','main')
    gp.train()
    utils.print_with_stamp('done training','main')
    
    n_test=25
    xg,yg = np.meshgrid ( np.linspace(Xd[:,0].min(),Xd[:,0].max(),n_test) , np.linspace(Xd[:,1].min(),Xd[:,1].max(),n_test) )
    X_test= np.vstack((xg.flatten(),yg.flatten())).T
    n = X_test.shape[0]
    utils.print_with_stamp('predicting','main')

    M = []; S = []
    batch_size=25
    for i in xrange(0,n,batch_size):
        next_i = min(i+batch_size,n)
        utils.print_with_stamp('batch %d , %d'%(i,next_i))
        r = gp.predict(X_test[i:next_i])
        M.append(r[0])
        S.append(r[1])

    M = np.vstack(M)
    S = np.vstack(S)

    utils.print_with_stamp('done predicting','main')
    from matplotlib import pyplot as plt
    plt.figure()
    plt.imshow(M.reshape(n_test,n_test), origin='lower')

    plt.figure()
    plt.imshow(S.reshape(n_test,n_test), origin='lower')
    plt.show()

    if gp.profile:
        write_profile_files(gp)

def test_K():
    import cov
    from scipy.io import loadmat
    dataset = loadmat('/media/diskstation/Kingfisher/matlab.mat')
    
    X_ = np.array(dataset['mat'][:,0:2])
    Y_ = np.array(dataset['mat'][:,2])[:,None]
    if theano.config.floatX == 'float32':
        X_ = X_.astype(np.float32)
        Y_ = Y_.astype(np.float32)

    idims = X_.shape[1]
    odims = Y_.shape[1]
    N = X_.shape[0]

    # and initialize the loghyperparameters of the gp ( this code supports squared exponential only, at the moment)
    loghyp_ = np.zeros((odims,idims+2))
    if theano.config.floatX == 'float32':
        loghyp_ = loghyp_.astype(np.float32)
    loghyp_[:,:idims] = X_.std(0)
    loghyp_[:,idims] = Y_.std(0)
    loghyp_[:,idims+1] = 0.1*loghyp_[:,idims]
    loghyp_ = np.log(loghyp_)

    X = S(X_,name='X', borrow=False)
    Y = S(Y_,name='Y', borrow=False)
    loghyp = [S(loghyp_[i,:],name='loghyp', borrow=False) for i in xrange(odims)]

    # We initialise the kernel matrices (one for each output dimension)
    K = [ kernel_func[i](X) for i in xrange(odims) ]
    iK = [ matrix_inverse(psd(K[i])) for i in xrange(odims) ]
    beta = [ iK[i].dot(Y[:,i]) for i in xrange(odims) ]

    # And finally, the negative log marginal likelihood ( again, one for each dimension; although we could share
    # the loghyperparameters across all output dimensions and train the GPs jointly)
    nlml = [ 0.5*(Y[:,i].T.dot(beta[i]) + T.log(det(psd(K[i]))) + N*T.log(2*np.pi) )/N for i in xrange(odims) ]

    fK = F((),beta)
    from time import time
    utils.print_with_stamp('evaluating','main')
    for i in xrange(10):
        fK()
    print fK()
    print time()-start
    utils.print_with_stamp('done predicting','main')

def test_K_means():
    from scipy.io import loadmat
    import theano
    dataset = loadmat('/media/diskstation/Kingfisher/matlab.mat')
    
    idx = np.random.choice(np.arange(dataset['mat'].shape[0]),1000)
    Xd = np.array(dataset['mat'][idx,0:2])
    Yd = np.array(dataset['mat'][idx,2])[:,None]
    #Xd = np.array(dataset['mat'][:,0:2])
    #Yd = np.array(dataset['mat'][:,2])[:,None]
    Xd = Xd - Xd.mean(0)
    k = 300
    Wd = np.random.multivariate_normal(Xd.mean(0),np.diag(Xd.var(0)),k)
    W, km = utils.get_kmeans_func(Wd)

    batch_size = 125
    n_trials=20
    learning_rate = 0.0025
    should_break = False
    prev_err =  km(Xd[0:batch_size],learning_rate)
    utils.print_with_stamp('start','main')
    from time import time
    start = time()
    for i in xrange(1000):
        for j in xrange(1,Xd.shape[0],batch_size):
            err =  km(Xd[j:j+batch_size],learning_rate*(0.95**i))
            if (abs(prev_err - err) <= 1e-6):
                print i
                should_break = True
                break;
        if should_break:
            break;
        prev_err = err
    utils.print_with_stamp('done','main')
    print (time()-start)

if __name__=='__main__':
    np.set_printoptions(linewidth=500)
    #test_random()
    test_sonar()
    #test_K()
    #test_K_means()




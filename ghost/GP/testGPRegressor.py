from GPRegressor import *

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

def write_profile_files(gp):
    from theano import d3viz
    root_path = '/localdata/juan/theano/'
    formatter = d3viz.formatting.PyDotFormatter()
    d3viz.d3viz(gp.K[0],root_path+'html/K'+theano.config.device+'.html')
    d3viz.d3viz(gp.iK[0],root_path+'html/iK'+theano.config.device+'.html')
    d3viz.d3viz(gp.beta[0],root_path+'html/beta'+theano.config.device+'.html')

    d3viz.d3viz(gp.nlml,root_path+'html/nlml'+theano.config.device+'.html')
    nlml_graph = formatter(gp.nlml)
    nlml_graph.write_png(root_path+'png/nlml'+theano.config.device+'.png')

    d3viz.d3viz(gp.dnlml,root_path+'html/dnlml'+theano.config.device+'.html')
    dnlml_graph = formatter(gp.nlml)
    dnlml_graph.write_png(root_path+'png/dnlml'+theano.config.device+'.png')

    d3viz.d3viz(gp.predict_,root_path+'html/predict_'+theano.config.device+'.html')
    predict_graph =  formatter(gp.predict_)
    predict_graph.write_png(root_path+'png/predict_'+theano.config.device+'.png')

    d3viz.d3viz(gp.predict_d_,root_path+'html/predict_d_'+theano.config.device+'.html')
    predict_d_graph = formatter(gp.predict_d_)
    predict_d_graph.write_png(root_path+'png/predict_d_'+theano.config.device+'.png')

def test_sonar():
    from scipy.io import loadmat
    dataset = loadmat('/media/diskstation/Kingfisher/matlab.mat')
    
    Xd = np.array(dataset['mat'][:,0:2])
    Xd += 1e-2*np.random.rand(*(Xd.shape))
    Yd = np.array(dataset['mat'][:,2])[:,None]

    #gp = GP(Xd,Yd, profile=True)
    gp = GPUncertainInputs(Xd,Yd, profile=True)
    utils.print_with_stamp('training','main')
    gp.train()
    utils.print_with_stamp('done training','main')

    
    n_test=50
    xg,yg = np.meshgrid ( np.linspace(Xd[:,0].min(),Xd[:,0].max(),n_test) , np.linspace(Xd[:,1].min(),Xd[:,1].max(),n_test) )
    X_test= np.vstack((xg.flatten(),yg.flatten())).T
    n = X_test.shape[0]
    utils.print_with_stamp('predicting','main')

    M = []; S = []
    batch_size=100
    for i in xrange(0,n,batch_size):
        next_i = min(i+batch_size,n)
        print 'batch %d , %d'%(i,next_i)
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

if __name__=='__main__':
    #test_random()
    test_sonar()

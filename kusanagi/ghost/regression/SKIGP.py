from .GP import *

class SKIGP(GP):
    '''Structured Kernel Interpolation GP (otherwise known as KISS-GP (with further extensions from MSGP) by Wilson, Nkish et al 2015'''
    def __init__(self, X_dataset=None, Y_dataset=None, name='SKIGP', idims=None, odims=None, profile=False, n_inducing=100,  uncertain_inputs=False, **kwargs):
        self.n_inducing=n_inducing
        GP.__init__(self,X_dataset,Y_dataset,name=name,idims=idims,odims=odims,profile=profile,uncertain_inputs=uncertain_inputs, **kwargs)

    def init_grid(self,pad=5,full=False):
        ''' returns grid for inducing inputs as theano tensors '''
        # get the bounds of the grid
        bounds = tt.concatenate([self.X.min(0)[:,None], self.X.max(0)[:,None]],axis=1) # D x 2
        
        # get the step size of the grid for each dimension
        step = (bounds[:,1] - bounds[:,0])/self.n_inducing
        slices = [slice(bounds[i,0]-pad*step[i],bounds[i,1]+pad*step[i],step[i]) for i in range(self.E)]

        return tt.mgrid[slices] if full else tt.ogrid[slices]

    def get_interpolation_weights(self,U,X):
        pass

    def get_kronmvm(self,cov,U):
        pass

    def init_loss(self):
        utils.print_with_stamp('Initialising expression graph for sparse spectral training loss function',self.name)
        self.Kuu = S(np.zeros((self.E,self.n_inducing,self.n_inducing),dtype='float64'), name="%s>Kuu"%(self.name))
        self.W = S(np.zeros((self.E,self.N,self.n_inducing),dtype='float64'), name="%s>W"%(self.name))

        N = self.X.shape[0].astype('float64')
        
        # get grid
        U = self.init_grid()
        
        
        return U




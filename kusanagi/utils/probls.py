import numpy as np
import theano
from kusanagi import utils
from scipy.stats import mvn
from scipy.special import erf
from matplotlib import pyplot as plt

class cubic_spline_gp(object):
    '''
    Implements the cubic spline GP from the Mahsereci and Hennig NIPS 2015 paper
    Based on their MATLAB implementation
    '''
    def __init__(self,offset=10):
        self.offset = offset
        self.t = []
        self.y = []
        self.dy = []
        self.sigma_y = []
        self.sigma_dy = []

    def k(self,a,b):
        min_ab = np.minimum(a,b)+self.offset
        return (min_ab**3)/3.0 + 0.5*np.abs(a-b)*(min_ab**2)

    def kd(self,a,b):
        a_ = a+self.offset
        b_ = b+self.offset
        return np.where(a<b,(a_**2)/2, a_*b_ - 0.5*(b_**2))

    def dk(self,a,b):
        a_ = a+self.offset
        b_ = b+self.offset
        return np.where(a>b,(b_**2)/2, a_*b_ - 0.5*(a_**2))

    def dkd(self,a,b):
        return np.minimum(a+self.offset,b+self.offset)

    def ddk(self,a,b):
        return np.where(a<b, b-a, 0.0)

    def ddkd(self,a,b):
        return np.where(a<b, 1.0, 0.0)

    def dddk(self,a,b):
        return np.where(a<b, -1.0, 0.0)

    def update(self, t, y, dy, sigma_y, sigma_dy):
        # append datapoint
        n = len(self.t)
        self.t.append(t)
        self.y.append(y)
        self.dy.append(dy)
        self.sigma_y.append(sigma_y)
        self.sigma_dy.append(sigma_dy)

        # update kernel matrix TODO: try to only compute the new entries in the matrix
        N = len(self.t)
        k_new = np.array([[ (self.k(ti,tj), self.kd(ti,tj), self.dkd(ti,tj)) for tj in self.t] for ti in self.t])
        k_top = np.concatenate([k_new[:,:,0],k_new[:,:,1]], axis=1)
        k_bottom = np.concatenate([k_new[:,:,1].T,k_new[:,:,2]], axis=1)
        K = np.concatenate([k_top,k_bottom])

        S = np.empty((2*N,))
        S[:N] = np.array(self.sigma_y)**2
        S[N:] = np.array(self.sigma_dy)**2
        self.K_n = K + np.diag(S)
        self.beta = np.linalg.solve(self.K_n,np.array(self.y+self.dy))

    def m(self,t):
        T = np.array(self.t)
        return np.concatenate([self.k(t,T),self.kd(t,T)]).dot(self.beta)
    
    def d1m(self,t):
        T = np.array(self.t)
        return np.concatenate([self.dk(t,T),self.dkd(t,T)]).dot(self.beta)

    def d2m(self,t):
        T = np.array(self.t)
        return np.concatenate([self.ddk(t,T),self.ddkd(t,T)]).dot(self.beta)

    def d3m(self,t):
        T = np.array(self.t)
        return np.concatenate([self.dddk(t,T),np.zeros(len(self.t))]).dot(self.beta)

    def V(self,t1,t2):
        T = np.array(self.t)
        ktt = self.k(t1,t2)
        k1 = np.concatenate([self.k(t1,T),self.kd(t1,T)])
        k2 = np.concatenate([self.k(t2,T),self.kd(t2,T)])
        return ktt - k1.dot(np.linalg.solve(self.K_n,k2))

    def Vd(self,t1,t2):
        T = np.array(self.t)
        kdtt = self.kd(t1,t2)
        k1 = np.concatenate([self.k(t1,T),self.kd(t1,T)])
        k2 = np.concatenate([self.dk(t2,T),self.dkd(t2,T)])
        return kdtt - k1.dot(np.linalg.solve(self.K_n,k2))

    def dV(self,t1,t2):
        T = np.array(self.t)
        dktt = self.dk(t1,t2)
        k1 = np.concatenate([self.dk(t1,T),self.dkd(t1,T)])
        k2 = np.concatenate([self.k(t2,T),self.kd(t2,T)])
        return dktt - k1.dot(np.linalg.solve(self.K_n,k2))
    
    def dVd(self,t1,t2):
        T = np.array(self.t)
        dkdtt = self.dkd(t1,t2)
        k1 = np.concatenate([self.dk(t1,T),self.dkd(t1,T)])
        k2 = np.concatenate([self.dk(t2,T),self.dkd(t2,T)])
        return dkdtt - k1.dot(np.linalg.solve(self.K_n,k2))

    def cubic_minimum(self,t):
        # since the mean belief is a cubic function, we can find the
        # parameters of the cubic via its derivatives, and solve for the roots
        d1mt =  self.d1m(t)
        d2mt =  self.d2m(t)
        d3mt =  self.d3m(t)
        a = 0.5*d3mt
        b = d2mt - d3mt*t
        c = d1mt + 0.5*d3mt*t*t - d2mt*t


        # if this spline segment is close to a quadratic and solution is minimum (via 2nd derivative test)
        if abs(d3mt) < 1e-9:
            tmin = (d1mt - d2mt*t)/d2mt if b > 0 else None
            return tmin

        # determinant
        D = b*b - 4*a*c
        sqrD= np.sqrt(D)
        
        # cubic poly extrema
        t1 =  (-b - sqrD)/(2*a)
        t2 =  (-b + sqrD)/(2*a)

        # pick the minimum ( second derivative of cubic poly > 0 ) d3mt*x + (d2mt-d3mt*t) > 0 
        if d3mt*t1 + b > 0:
            return t1
        elif d3mt*t2 + b > 0:
            return t2
        else:
            return None

def gauss_cdf(z):
    return 0.5*(1.0 + erf(z/np.sqrt(2.0)))
def gauss_pdf(z):
    return np.exp(-0.5*z**2)/np.sqrt(2.0*np.pi)
def EI(m,v,eta):
    # evaluates the expected improvement when seaching for a minimum
    d = (eta-m)
    s = np.sqrt(v)
    z = d/s
    return d*gauss_cdf(z) + s*gauss_pdf(z)
    
def probWolfe(t,gp,c1=0.05,c2=0.8,strong_wolfe=True):
    # evaluates the joint pdf for the probablistic wolfe conditions
    m0 = gp.m(0.0)
    d1m0 = gp.d1m(.00)
    V00 = gp.V(0.0,0.0)
    Vd00 = gp.Vd(0.0,0.0)
    dVd00 = gp.dVd(0.0,0.0)
    
    # mean
    mt = gp.m(t)
    d1mt = gp.d1m(t)
    ma = m0 - mt + c1*t*d1m0 # armijo rule
    mb = d1mt - c2*d1m0      # curvature condition
    
    # cov
    dV0t = gp.dV(0.0,t)
    dVd0t = gp.dVd(0.0,t)

    Caa = V00 + ((c1*t)**2)*dVd00 + gp.V(t,t) + 2*(c1*t*(Vd00 - dV0t) - gp.V(0.0,t))
    Cbb = (c2**2)*dVd00 - 2*c2*dVd0t + gp.dVd(t,t)
    
    if Caa < 0 or Cbb < 0: # undefined
        return 0.0

    if Caa < 1e-9 and Cbb < 1e-9:   # near deterministic case
        return 1.0 if ma>=0 and mb>=0 else 0.0

    Cab = -c2*(Vd00+c1*t*dVd00) + c2*dV0t + gp.dV(t,0.0) + c1*dVd0t - gp.Vd(t,t)
    
    #evaluate the integral
    lower = [-ma/np.sqrt(Caa),-mb/np.sqrt(Cbb)]
    if strong_wolfe:
        upper = [np.inf,np.inf]
        infin = np.array([1,1])
    else:
        b_ = (2*c2*(np.abs(d1m0)+ 2*np.sqrt(dVd00)) - mb)/np.sqrt(Cbb)
        upper = [np.inf,b_]
        infin = np.array([1,2])
    rho = Cab/np.sqrt(Caa*Cbb)
    # using mvndst from scipy which is undocumented, but from the fortran doc:
    # first argument are lower bounds (n dimensional vector)
    # second are upper bounds
    # third is an indicator vector infin, where
    #       if infin[d]  < 0, integration is done from -infinity to infinity
    #       if infin[d] == 0, integration is done from -infinity to upper[d]
    #       if infin[d] == 1, integration is done from  lower[d] to infinity
    #       if infin[d] == 2, integration is done from  lower[d] to upper[d]
    # fourth is an array with correlation coefficients (off diagonal covariances)
    #the function returns the value of the integral of a multivariate normal density function,
    # with mean zero and covariance with diagonal elements normalized to 1
    ret = mvn.mvndst(lower,upper,infin,np.array([rho]))
    return ret[1]
import pdb

def prob_line_search(func, x0, f0, df0, var_f0, var_df0, alpha0, search_direction, max_ls=10, extrapolation_step=1.0, wolfe_threshold=0.3):
    plt.ion()
    # scaling factor for line search
    beta = np.abs(search_direction.dot(df0)) 
    
    # scaled noise
    sigma_y = np.sqrt(var_f0)/(alpha0*beta);
    sigma_dy = np.sqrt((search_direction**2).dot(var_df0))/beta;
    
    # gp model for line search
    gp = cubic_spline_gp()

    # initialize gp with measurement at line search step = 0
    y = 0.0
    t_curr = 0.0
    dy = df0.dot(search_direction)/beta
    gp.update(t_curr, y, dy, sigma_y, sigma_dy)

    print('||||||start|||||||')
    print(f0,df0.dot(search_direction),0.0,t_curr)
    # start line search with step = 1.0
    t_curr = 1.0
    # extrapolation step
    ext = extrapolation_step
    # number of line searches so far
    ls = 0
    while ls < max_ls:
        # step in line search direction and evaluate (new batch)
        step_size = t_curr*alpha0
        
        xt = x0+step_size*search_direction
        f,df,var_f,var_df = func(xt)
        print('||||||'+str(ls)+'|||||||')
        print(f,df.dot(search_direction), step_size, t_curr)
        #print df
        
        # scaled noise
        sigma_y = np.sqrt(var_f)/(alpha0*beta);
        sigma_dy = np.sqrt((search_direction**2).dot(var_df))/beta;

        # project and scale
        y = (f - f0)/(alpha0*beta)
        dy = df.dot(search_direction)/beta

        # update gp
        gp.update(t_curr, y, dy, sigma_y, sigma_dy)

        plt.figure('gp1')
        plt.clf()
        ax11 = plt.subplot(211)
        t_plot = np.arange(0,gp.t[-1]+2*ext,0.01)
        y_plot = np.array([gp.m(ti) for ti in t_plot])
        var_plot = np.array([gp.V(ti,ti) for ti in t_plot])
        plt.plot(t_plot,y_plot)
        alpha = 0.5
        for i in range(6):
            alpha = alpha/1.5
            lower_bound = y_plot - i*np.sqrt(var_plot).squeeze()
            upper_bound = y_plot + i*np.sqrt(var_plot).squeeze()
            plt.fill_between(t_plot, lower_bound, upper_bound, alpha=alpha)
        
      
        plt.show()
        plt.waitforbuttonpress(0.01)

        # evaluate probabilistic wolfe conditions at evaluated points (excluding t=0)
        for ti in gp.t[1:]:
            if probWolfe(ti,gp) > wolfe_threshold:
                plt.figure('gp1')
                ax11 = plt.subplot(211)
                plt.scatter(np.array([ti]),np.array([gp.m(ti)]),c='g',marker='o')
                plt.show()
                plt.waitforbuttonpress(0.01)
                #pdb.set_trace()
                step_size = ti*alpha0
                xt = x0+step_size*search_direction
                f,df,var_f,var_df = func(xt)
                return step_size, xt, f, df, var_f, var_df

        # find the minimum of currently evaluated step sizes
        minm,tmin = min([(gp.m(ti),ti) for ti in gp.t], key=lambda x: x[0])
        
        # find the minima of the cubic spline interpolation
        t_cand = []
        t_sorted = sorted(gp.t)
        for t1,t2 in zip(t_sorted,t_sorted[1:]):
            # find minimum in each cell
            t_cell = t1 + 1e-3*(t2 - t1)
            t_min = gp.cubic_minimum(t_cell)
            # if the cubic polynomial has a minimum in the cell
            if t_min is not None and t_min > t1 and t_min <= t2:
                t_cand.append(t_min)

        # append an extrapolation candidate
        t_cand.append(t_sorted[-1] + ext)
    
        def score1(ti,gp):
            return EI(gp.m(ti),gp.V(ti,ti),minm)
        def score2(ti,gp):
            return EI(gp.m(ti),gp.V(ti,ti),minm)
        def score3(ti,gp):
            return score1(ti,gp)*score2(ti,gp)
        # evaluate expected improvement wrt minm, weighted by the strong wolfe condition, to select next evaluation step
        max_EIp,t_curr = max([( score1(ti,gp) ,ti) for ti in t_cand], key=lambda x: x[0])
        
        plt.figure('gp1')
        ax12 = plt.subplot(212, sharex=ax11)
        ei_plot   = np.array([EI(gp.m(ti),gp.V(ti,ti),minm) for ti in t_plot])
        pw_plot   = np.array([probWolfe(ti,gp) for ti in t_plot])
        eipw_plot = np.array([EI(gp.m(ti),gp.V(ti,ti),minm)*probWolfe(ti,gp) for ti in t_plot])
        l1, = plt.plot( t_plot,   ei_plot, label='EI'   )
        l2, = plt.plot( t_plot,   pw_plot, label='pW'   )
        l3, = plt.plot( t_plot, eipw_plot, label='EIpW' )
        plt.plot([t_plot[0],t_plot[-1]],[0.3,0.3],c='k')
        plt.legend(handles=[l1,l2,l3])
        plt.show()
        plt.waitforbuttonpress(0.01)
     
        # if the extrapolation point was selected, increase extrapolation step
        if t_curr == t_cand[-1]:
            ext *= 1.3
        ls +=1
        plt.figure('gp1')
        ax11 = plt.subplot(211)
        plt.scatter(t_cand,np.array([gp.m(ti) for ti in t_cand]), c='k', marker='o')
        plt.scatter(gp.t,np.array([gp.m(ti) for ti in gp.t]), c='k', marker='x')
        plt.scatter(np.array([t_curr]),np.array([gp.m(t_curr)]),c='r',marker='o')
       
        plt.show()
        plt.waitforbuttonpress(0.01)
        #pdb.set_trace()
        #raw_input()

    ## didn't find a good expected improvement satisfying the wolfe constraint. use the smallest function value
    step_size = tmin*alpha0
    xt = x0+step_size*search_direction
    f,df,var_f,var_df = func(xt)
    return step_size, xt, f, df, var_f, var_df

def minimize(func,x0, alpha0=0.01, max_nfevals=1000):
    #loss returns loss_mean, grad_of_loss_mean, loss_variance, grad_of_loss_variance
    f,df,var_f,var_df = func(x0)

    search_dir = -df
    xt = x0
    step_size=alpha0
    utils.print_with_stamp('new step_size: %f'%(step_size),'ProbLS')
    nfevals = 1

    while nfevals < max_nfevals:
        step_size, xt, f, df, var_f, var_df = prob_line_search(func,xt,f,df,var_f,var_df,step_size,search_dir)
        utils.print_with_stamp('new step_size: %f'%(step_size),'ProbLS')
        # set new initial line search step size to be 1.3 longer than previous
        step_size *= 1.3
        if step_size == 0.0:
            step_size=alpha0
        search_dir = -df

def compile_loss_fn(losses, params, updates=None, callback=None):
    ''' 
    compiles two loss function compatible with the minimize_probls method.
    TODO allow for various SGD methods (e.g. adam, nesterov, rmsprop)

    '''
    # mean and variance of loss (assuming first axis is the batch index)
    utils.print_with_stamp("Computing loss mean and variance",'ProbLS')
    m_loss, S_loss = losses.mean(0), losses.var(0)
    # mean and variance of gradients
    # TODO compute the variance of gradients efficiently
    utils.print_with_stamp("Computing gradient mean and variance",'ProbLS')
    grads = theano.tensor.jacobian(losses, params)
    m_grad, S_grad = list(zip(*[(g.mean(0).flatten(),g.var(0).flatten()) for g in grads]))
    m_grad, S_grad = theano.tensor.concatenate(m_grad), theano.tensor.concatenate(S_grad)
    loss_fn = theano.function([],[m_loss,m_grad,S_loss,S_grad], updates=updates)

    utils.print_with_stamp("Done compiling.",'ProbLS')
    return loss_fn


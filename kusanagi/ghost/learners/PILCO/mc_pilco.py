from kusanagi.ghost.learners.PILCO import *
from functools import partial

class MC_PILCO(PILCO):
    def __init__(self, params, plant_class, policy_class, cost_func=None, viz_class=None,
                 dynmodel_class=kreg.GP_UI, n_samples=10, experience=None, async_plant=False,
                 name='MC_PILCO', filename_prefix=None):
        super(MC_PILCO, self).__init__(params, plant_class, policy_class, cost_func,
                                       viz_class, dynmodel_class, experience, async_plant,
                                       name, filename_prefix)
        self.resample = params['resample'] if 'resample' in params else False

        randint = lasagne.random.get_rng().randint(1, 2147462579)
        self.m_rng = theano.sandbox.rng_mrg.MRG_RandomStreams(randint)
        self.trajectory_samples = theano.shared(np.array(n_samples).astype('int32'),
                                                name="%s>trajectory_samples"%(self.name))

        # draw initial set of particles
        z0 = np.random.randn(n_samples, self.mx0.get_value().size)
        Lx0 = np.linalg.cholesky(self.Sx0.get_value())
        x0 = self.mx0.get_value() + z0.dot(Lx0.T)
        # create shared variables for initial state samples
        self.x0 = theano.shared(x0.astype(tt.config.floatX), name=self.name+'_x0')

    def propagate_state(self, x, u=None,*args, **kwargs):
        ''' Given a set of input states, this function returns predictions for the next states.
            This is done by 1) evaluating the current policy 2) using the dynamics model to estimate 
            the next state. If x has shape [n,D] where n is tthe number of samples and D is the state dimension,
            this function will return x_next with shape [n,D] representing the next states and costs with shape [n,1]
        '''
        dynmodel = self.dynamics_model if kwargs.get('dynmodel',None) is None else kwargs.get('dynmodel',None)
        policy = self.policy if kwargs.get('policy',None) is None else kwargs.get('policy',None)
        resample = kwargs.get('resample',self.resample)
        iid_per_eval = kwargs.get('iid_per_eval',False)

        D = self.mx0.get_value().size

        # resample if requested
        if resample:
            n,D = x.shape
            n = n.astype(theano.config.floatX)
            mx = x.mean(0)
            Sx = x.T.dot(x)/n - tt.outer(mx,mx)
            z = self.m_rng.normal(x.shape)
            x = mx + z.dot(tt.slinalg.cholesky(Sx).T)

        # convert angles from input states to their complex representation
        xa = utils.gTrig(x,self.angle_idims,D)

        if u is None:
            # compute control signal (with noisy state measurement)
            sn = tt.exp(dynmodel.logsn)
            x_noisy = x + self.m_rng.normal(x.shape).dot(tt.diag(sn))
            xa_noisy = utils.gTrig(x_noisy,self.angle_idims,D)
            u = policy.evaluate(xa_noisy, symbolic=True, iid_per_eval=iid_per_eval, return_samples=True)
        
        # build state-control vectors
        xu = tt.concatenate([xa,u],axis=1)

        # predict the change in state given current state-control for each particle 
        delta_x = dynmodel.predict_symbolic(xu, iid_per_eval=iid_per_eval, return_samples=True)
        
        # compute the successor states
        x_next = x + delta_x

        return x_next

    def rollout_single_step(self, x, gamma, gamma0, *args, **kwargs):
        '''
        Propagates the state distribution and computes the associated cost
        '''
        dynmodel = self.dynamics_model\
        if kwargs.get('dynmodel', None) is None\
        else kwargs.get('dynmodel', None)
        
        policy = self.policy\
        if kwargs.get('policy', None) is None\
        else kwargs.get('policy', None)
        
        cost = self.cost\
        if kwargs.get('cost', None) is None\
        else kwargs.get('cost', None)

        # compute next state distribution
        x_next = self.propagate_state(x, None, dynmodel,policy)

        # get cost
        sn = tt.exp(dynmodel.logsn)
        x_next_noisy = x_next + self.m_rng.normal(x.shape).dot(tt.diag(sn))
        c_next = cost(x_next_noisy,None)

        # jacobian for debugging
        #jac = theano.tensor.jacobian(x_next.flatten(),x).reshape((x.shape[0],x.shape[1],x.shape[0],x.shape[1])).diagonal(axis1=0,axis2=2)
        
        return [gamma*c_next, x_next, gamma*gamma0]

    def get_rollout(self, x0, H, gamma0, n_samples, dynmodel=None, policy=None, cost=None):
        ''' Given some initial state particles x0, and a prediction horizon H 
        (number of timesteps), returns a set of trajectories sampled from the dynamics model and 
        the discounted costs for each step in the trajectory.'''
        utils.print_with_stamp('Computing symbolic expression graph for belief state propagation',self.name)
        dynmodel = self.dynamics_model if dynmodel is None else dynmodel
        policy = self.policy if policy is None else policy
        cost = self.cost if cost is None else cost
        
        # initial state cost ( fromt he input distribution )
        c0 = self.cost(x0,None)

        # define the rollout step function for the provided dynmodel, policy and cost
        rollout_single_step = partial(self.rollout_single_step, dynmodel=dynmodel,policy=policy,cost=cost)

        # these are the shared variables that will be used in the scan graph.
        # we need to pass them as non_sequences here (see: http://deeplearning.net/software/theano/library/scan.html)
        shared_vars = []
        shared_vars.extend(dynmodel.get_all_shared_vars())
        shared_vars.extend(policy.get_all_shared_vars())
        # loop over the planning horizon
        rollout_output, rollout_updts = theano.scan(fn=rollout_single_step, 
                                            outputs_info=[None,x0,gamma0], 
                                            non_sequences=[gamma0]+shared_vars,
                                            n_steps=H, 
                                            strict=True,
                                            allow_gc=False,
                                            name="%s>rollout_scan"%(self.name))

        costs, trajectories, gamma_ = rollout_output[:3]

        # prepend the initial cost
        costs = tt.concatenate([c0.dimshuffle('x',0), costs]).T  # first axis: batch, second axis: time step

        # prepend the initial states
        trajectories = tt.concatenate([x0.dimshuffle('x',0,1), trajectories]).transpose(1,0,2) # first axis; batch, second axis: time step

        costs.name = 'c_list'
        trajectories.name = 'x_list'

        return [costs, trajectories], rollout_updts
    
    def get_policy_value(self, x0 = None, H = None, gamma0 = None, n_samples=None,
                         dynmodel=None, policy=None, cost_per_sample=False):
        ''' Returns a symbolic expression (theano tensor variable) for the value of the current policy '''
        x0 = self.x0 if x0 is None else x0
        H = self.H_steps if H is None else H
        gamma0 = self.gamma0 if gamma0 is None else gamma0
        n_samples = self.trajectory_samples if n_samples is None else gamma0

        [costs, trajectories], updts = self.get_rollout(x0, H, gamma0, n_samples, dynmodel, policy)
        if cost_per_sample:
            return costs, updts
        else:
            expected_accumulated_cost = costs[:,1:].sum(1).mean() # first axis: batch, second axis: timestep
            #expected_accumulated_cost += 0.001*trajectories[:,1:].var(0).sum(-1).mean() # variance of states at every timestep

            expected_accumulated_cost.name = 'J'
            return expected_accumulated_cost, updts
    
    def compile_rollout(self, x0 = None, H = None, gamma0 = None, n_samples=None,
                        dynmodel=None, policy=None, cost_per_sample=False):
        ''' Compiles a theano function graph that compute the predicted states and discounted costs for a given
        inital state and prediction horizon. Unlike GP based PILCO, This function returns trajectory samples instead
        of means and covariances for everty time step'''
        x0 = self.x0 if x0 is None else x0
        H = self.H_steps if H is None else H
        gamma0 = self.gamma0 if gamma0 is None else gamma0
        n_samples = self.trajectory_samples if n_samples is None else gamma0

        [costs, trajectories], updts = self.get_rollout(x0, H, gamma0, n_samples, dynmodel, policy)

        if cost_per_sample:
            ret = [costs, trajectories]
        else:
            ret = [tt.mean(costs,axis=0), theano.tensor.var(costs,axis=0), trajectories]

        utils.print_with_stamp('Compiling trajectory rollout function',self.name)
        rollout_fn = theano.function([], 
                                     ret, 
                                     allow_input_downcast=True, 
                                     updates=updts,
                                     name='%s>rollout_fn'%(self.name))
        utils.print_with_stamp("Done compiling.",self.name)
        return rollout_fn

    def compile_policy_gradients(self, x0 = None, H = None, gamma0 = None, n_samples=None,
                                 dynmodel=None, policy=None):
        ''' Compiles a theano function graph that computes the gradients of the expected accumulated a given
        initial state and prediction horizon. The function will return the value of the policy, followed by 
        the policy gradients.'''
        if policy is None:
            policy= self.policy

        expected_accumulated_cost, updts = self.get_policy_value(x0,H,gamma0,n_samples,dynmodel,policy)
        # get the policy parameters (as theano tensor variables)
        p = policy.get_params(symbolic=True)
        # the policy gradients will have the same shape as the policy parameters (i.e. this returns a list
        # of theano tensor variables with the same dtype and ndim as the parameters in p )
        if gamma0.get_value() == 1.0:
            H = self.H_steps if H is None else H
            expected_accumulated_cost = expected_accumulated_cost/H
        dJdp = self.get_policy_gradients(expected_accumulated_cost,p)
        retvars = [expected_accumulated_cost]
        retvars.extend(dJdp)

        utils.print_with_stamp('Compiling policy gradients',self.name)
        policy_gradient_fn = theano.function([],
                                                retvars, 
                                                allow_input_downcast=True, 
                                                updates=updts,
                                                name="%s>policy_gradient_fn"%(self.name))
        utils.print_with_stamp("Done compiling.",self.name)
        return policy_gradient_fn

    def update(self):
        if not hasattr(self,'update_fn') or self.update_fn is None:
            # draw initial set of particles
            z0 = self.m_rng.normal((self.trajectory_samples,self.mx0.size))
            Lx0 = tt.slinalg.cholesky(self.Sx0)
            x0 = self.mx0 + z0.dot(Lx0.T)
            updates = [(self.x0,x0)]
            self.update_fn = theano.function([],[],updates=updates)

        self.update_fn()

        # update dynamics model and policy if needed
        if hasattr(self.dynamics_model,'update'):
            self.dynamics_model.update()
        if hasattr(self.policy,'update'):
            self.policy.update()

from kusanagi.ghost.learners.PILCO import *
from lasagne.utils import unroll_scan

from theano.gof import Variable


def fast_jacobian(expr, wrt, chunk_size=16, func=None):
    assert isinstance(expr, Variable), \
        "tensor.jacobian expects a Variable as `expr`"
    assert expr.ndim < 2, \
        ("tensor.jacobian expects a 1 dimensional variable as "
         "`expr`. If not use flatten to make it a vector")

    num_chunks = tt.ceil(1.0 * expr.shape[0] / chunk_size)
    num_chunks = tt.cast(num_chunks, 'int32')
    steps = tt.arange(num_chunks)
    remainder = expr.shape[0] % chunk_size

    def chunk_grad(i):
        wrt_rep = tt.tile(wrt, (chunk_size, 1))
        if func is not None:
            expr_rep = func(wrt_rep)
        else:
            expr_rep, _ = theano.scan(
                fn=lambda wrt_: theano.clone(expr, {wrt: wrt_}),
                sequences=wrt_rep)
        chunk_expr_grad = tt.roll(
            tt.identity_like(expr_rep),
            i * chunk_size,
            axis=1)
        return tt.grad(cost=None,
                      wrt=wrt_rep,
                      known_grads={
                          expr_rep: chunk_expr_grad
                      })

    grads, _ = theano.scan(chunk_grad, sequences=steps)
    grads = grads.reshape((chunk_size * grads.shape[0], wrt.shape[0]))
    jac = ifelse.ifelse(tt.eq(remainder, 0), grads, grads[:expr.shape[0], :])
    return jac

class MC_PILCO(PILCO):
    def __init__(self, params, plant_class, policy_class, cost_func=None, viz_class=None, dynmodel_class=kreg.GP_UI, n_samples=20, experience = None, async_plant=False, name='MC_PILCO', filename_prefix=None):
        super(MC_PILCO, self).__init__(params, plant_class, policy_class, cost_func,viz_class, dynmodel_class, experience, async_plant, name, filename_prefix)
        self.m_rng = theano.sandbox.rng_mrg.MRG_RandomStreams(lasagne.random.get_rng().randint(1,2147462579))
        self.trajectory_samples = theano.shared(np.array(n_samples).astype('int32'), name="%s>trajectory_samples"%(self.name) ) 
        D = self.mx0.get_value().size
        self.xp = theano.shared(np.zeros((n_samples, D),dtype=theano.config.floatX))
        self.dxdxp = theano.shared(np.zeros((n_samples*D, n_samples, D),dtype=theano.config.floatX))
        self.resample = params['resample'] if 'resample' in params else False
    
    def propagate_state(self,x,dynmodel=None,policy=None,cost=None,resample=None):
        ''' Given a set of input states, this function returns predictions for the next states.
            This is done by 1) evaluating the current policy 2) using the dynamics model to estimate 
            the next state. If x has shape [n,D] where n is tthe number of samples and D is the state dimension,
            this function will return x_next with shape [n,D] representing the next states and costs with shape [n,1]
        '''
        if dynmodel is None:
            dynmodel = self.dynamics_model
        if policy is None:
            policy= self.policy
        if cost is None:
            cost= self.cost
        if resample is None:
            resample=self.resample

        n,D = x.shape
        n= n.astype(theano.config.floatX)
        D_ = self.mx0.get_value().size

        # convert angles from input states to their complex representation
        xa = utils.gTrig(x,self.angle_idims,D_)

        # compute control signal (with noisy state measurement)
        sn2 = tt.exp(2*dynmodel.logsn)
        x_ = x + self.m_rng.normal(x.shape).dot(tt.diag(tt.sqrt(sn2)))
        xa_ = utils.gTrig(x_,self.angle_idims,D_)
        u = policy.evaluate(xa, symbolic=True)[0]

        # build state-control vectors
        xu = tt.concatenate([xa,u],axis=1)

        # predict the change in state given current state-control for each particle 
        delta_x = dynmodel.predict_symbolic(xu, iid_per_eval=False, return_samples=True)
        
        # compute the successor states
        x_next = x + delta_x

        # get cost ( via moment matching, which will penalize policies that result in multimodal trajectory distributions )
        mx_next = x_next.mean(0)
        Sx_next = x_next.T.dot(x_next)/n - tt.outer(mx_next,mx_next) #+ tt.diag(sn2)
        mc_next, Sc_next = cost(mx_next,Sx_next)
        #c_next = cost(x_next,None)[0]
        #mc_next = c_next.mean(0)
        #Sc_next = c_next.var(0)

        if resample:
            z = self.m_rng.normal(x.shape)
            x_next = mx_next + z.dot(tt.slinalg.cholesky(Sx_next).T)

        return [mc_next, Sc_next, x_next]


    def get_rollout(self, mx0, Sx0, H, gamma0, n_samples, dynmodel=None, policy=None):
        ''' Given some initial state distribution Normal(mx0,Sx0), and a prediction horizon H 
        (number of timesteps), returns a set of trajectories sampled from the dynamics model and 
        the discounted costs for each step in the trajectory.'''
        utils.print_with_stamp('Computing symbolic expression graph for belief state propagation',self.name)
        if dynmodel is None:
            dynmodel = self.dynamics_model
        if policy is None:
            policy= self.policy
        # these are the shared variables that will be used in the graph.
        # we need to pass them as non_sequences here (see: http://deeplearning.net/software/theano/library/scan.html)
        shared_vars = []
        shared_vars.extend(dynmodel.get_all_shared_vars())
        shared_vars.extend(policy.get_all_shared_vars())
        
        # draw initial set of particles
        z0 = self.m_rng.normal((n_samples,mx0.shape[0]))
        Lx0 = tt.slinalg.cholesky(Sx0)
        x0 = mx0 + z0.dot(Lx0.T)

        # initial state cost ( fromt he input distribution )
        mv0, Sv0 = self.cost(mx0,Sx0)

        # do first iteration of scan here (so if we have any learning iteration updates that depend on computations inside the scan loop, all the inputs are defined outisde the scan)
        [mv1, Sv1, x1] = self.propagate_state(x0,dynmodel,policy)
        mv1 *= gamma0; Sv1 *= gamma0*gamma0
        
        updates = theano.updates.OrderedUpdates()
        updates += dynmodel.get_dropout_masks()

        # this defines the loop where the state is propaagated
        def rollout_single_step(mv,Sv,x,gamma,gamma0,*args):
            [mv_next, Sv_next, x_next] = self.propagate_state(x,dynmodel,policy)
            return [gamma*mv_next, gamma*gamma*Sv_next, x_next, gamma*gamma0]
        # loop over the planning horizon
        rollout_output, rollout_updts = theano.scan(fn=rollout_single_step, 
                                            outputs_info=[mv1,Sv1,x1,gamma0*gamma0], 
                                            non_sequences=[gamma0]+shared_vars,
                                            n_steps=H-1, 
                                            strict=True,
                                            allow_gc=False,
                                            name="%s>rollout_scan"%(self.name))
        updates += rollout_updts
        mean_costs, var_costs, trajectories, gamma_ = rollout_output
        
        if hasattr(dynmodel,'learning_iteration_updates') and dynmodel.learning_iteration_updates is not None:
            updates += dynmodel.learning_iteration_updates

        if hasattr(policy,'learning_iteration_updates') and policy.learning_iteration_updates is not None:
            updates += policy.learning_iteration_updates

        # prepend the initial cost
        mean_costs = tt.concatenate([mv0.dimshuffle('x'),mv1.dimshuffle('x'), mean_costs])
        var_costs = tt.concatenate([Sv0.dimshuffle('x'),Sv1.dimshuffle('x'), var_costs])

        # prepend the initial states
        trajectories = tt.concatenate([x0.dimshuffle('x',0,1),x1.dimshuffle('x',0,1), trajectories]).transpose(1,0,2)

        mean_costs.name = 'mc_list'
        var_costs.name = 'Sc_list'
        trajectories.name = 'x_list'
        return [mean_costs,var_costs, trajectories], updates
    
    def get_policy_value(self, mx0 = None, Sx0 = None, H = None, gamma0 = None, n_samples=None, dynmodel=None, policy=None):
        ''' Returns a symbolic expression (theano tensor variable) for the value of the current policy '''
        mx0 = self.mx0 if mx0 is None else mx0
        Sx0 = self.Sx0 if Sx0 is None else Sx0
        H = self.H_steps if H is None else H
        gamma0 = self.gamma0 if gamma0 is None else gamma0
        n_samples = self.trajectory_samples if n_samples is None else gamma0

        [mcosts, Scosts, trajectories], updts = self.get_rollout(mx0,Sx0,H,gamma0, n_samples, dynmodel, policy)
        expected_accumulated_cost = mcosts[1:].sum()
        expected_accumulated_cost.name = 'J'
        return expected_accumulated_cost, updts
    
    def compile_rollout(self, mx0 = None, Sx0 = None, H = None, gamma0 = None, n_samples=None, dynmodel=None, policy=None):
        ''' Compiles a theano function graph that compute the predicted states and discounted costs for a given
        inital state and prediction horizon. Unlike GP based PILCO, This function returns trajectory samples instead
        of means and covariances for everty time step'''
        mx0 = self.mx0 if mx0 is None else mx0
        Sx0 = self.Sx0 if Sx0 is None else Sx0
        H = self.H_steps if H is None else H
        gamma0 = self.gamma0 if gamma0 is None else gamma0
        n_samples = self.trajectory_samples if n_samples is None else gamma0

        [mcosts, Scosts, trajectories], updts = self.get_rollout(mx0,Sx0,H,gamma0, n_samples, dynmodel, policy)

        utils.print_with_stamp('Compiling trajectory rollout function',self.name)
        rollout_fn = theano.function([], 
                                     [mcosts,Scosts,trajectories], 
                                     allow_input_downcast=True, 
                                     updates=updts,
                                     name='%s>rollout_fn'%(self.name))
        utils.print_with_stamp("Done compiling.",self.name)
        return rollout_fn

    def compile_policy_gradients(self, mx0 = None, Sx0 = None, H = None, gamma0 = None, n_samples=None, dynmodel=None, policy=None):
        ''' Compiles a theano function graph that computes the gradients of the expected accumulated a given
        initial state and prediction horizon. The function will return the value of the policy, followed by 
        the policy gradients.'''
        if policy is None:
            policy= self.policy

        expected_accumulated_cost, updts = self.get_policy_value(mx0,Sx0,H,gamma0,n_samples,dynmodel,policy)
        # get the policy parameters (as theano tensor variables)
        p = policy.get_params(symbolic=True)
        # the policy gradients will have the same shape as the policy parameters (i.e. this returns a list
        # of theano tensor variables with the same dtype and ndim as the parameters in p )
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

    def rollout(self, mx0, Sx0, H_steps, discount, use_scan=True, dynmodel=None, policy=None):
        ''' Function that computes trajectory rollouts. Ensures the compiled rollout function is initialised before we can call it'''
        # draw initial set of particles
        if use_scan:
            return super(MC_PILCO,self).rollout(mx0, Sx0, H_steps, discount)

        utils.print_with_stamp('r')
        D = mx0.size
        z0 = np.random.multivariate_normal(np.zeros((D,)),np.eye(D),self.trajectory_samples.get_value().tolist())
        Lx0 = np.linalg.cholesky(Sx0)
        x0 = mx0 + z0.dot(Lx0.T)
        self.xp.set_value(x0.astype(theano.config.floatX))

        # call propagate_fn 
        if not hasattr(self,'propagate_fn') or self.propagate_fn is None:
            if dynmodel is None:
                dynmodel = self.dynamics_model
            if policy is None:
                policy= self.policy
            shared_vars = []
            shared_vars.extend(dynmodel.get_all_shared_vars())
            shared_vars.extend(policy.get_all_shared_vars())
            [c_t, x_t], updates = self.propagate_state(self.xp,dynmodel,policy)
            updates += [(self.xp,x_t)] # update the state variable
            self.propagate_fn = theano.function([],[c_t,x_t], updates=updates, name='%s>propagate_fn'%(self.name))
        
        costs, trajectories = [],[]
        for i in range(H_steps):
            cost, x = self.propagate_fn()
            costs.append(cost)
            trajectories.append(x)
            print cost.mean()

        return [np.array(costs),np.array(trajectories)]

    def policy_gradient(self, mx0, Sx0, H_steps, discount, use_scan=True, dynmodel=None, policy=None):
        ''' Function that computes the gradients of the parameters of the current policy, from trajectory samples. 
            Ensures the compiled policy gradients function is initialised before we can call it'''
        if use_scan:
            return super(MC_PILCO,self).rollout(mx0, Sx0, H_steps, discount)
        utils.print_with_stamp('pg')
        # draw initial set of particles
        D = mx0.size
        z0 = np.random.multivariate_normal(np.zeros((D,)),np.eye(D),self.trajectory_samples.get_value().tolist())
        Lx0 = np.linalg.cholesky(Sx0)
        x0 = mx0 + z0.dot(Lx0.T)
        self.xp.set_value(x0.astype(theano.config.floatX))

        # call propagate_d_fn 
        if not hasattr(self,'propagate_d_fn') or self.propagate_d_fn is None:
            if dynmodel is None:
                dynmodel = self.dynamics_model
            if policy is None:
                policy= self.policy

            # get the one step predictions
            shared_vars = []
            shared_vars.extend(dynmodel.get_all_shared_vars())
            shared_vars.extend(policy.get_all_shared_vars())
            [c_t, x_t], updates = self.propagate_state(self.xp,dynmodel,policy)
            updates += [(self.xp,x_t)] # update the state variable
            print updates
            # get the policy parameters (as theano tensor variables)
            p = policy.get_params(symbolic=True)
            # the policy gradients will have the same shape as the policy parameters (i.e. this returns a list
            # of theano tensor variables with the same dtype and ndim as the parameters in p )
            dcdp = fast_jacobian(c_t, p) # [NxU0, NxU1, .. , NxUP ]
            dxdp = fast_jacobian(x_t.flatten(), p) # [N*DxU0, N*DxU1, .. , N*DxUP ]
            dxdxp = fast_jacobian(x_t.flatten(), self.xp) # [N*DxNxD ]
            updates += [(self.dxdp,dxdp)] # update the state derivatives
            retvars = [c_t]
            retvars.extend([dcdp,dxdp,dxdxp])

            self.propagate_d_fn = theano.function([],retvars, updates=updates, name='%s>propagate_d_fn'%(self.name))
        
        costs, trajectories = [],[]
        for i in range(H_steps):
            ret = self.propagate_d_fn()
            print ret[0].mean()
            print len(ret),len(ret[1]),len(ret[2]),len(ret[3])

        return [np.array(costs),np.array(trajectories)]

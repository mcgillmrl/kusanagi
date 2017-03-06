from kusanagi.ghost.learners.PILCO import *
from functools import partial
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
    def __init__(self, params, plant_class, policy_class, cost_func=None, viz_class=None, dynmodel_class=kreg.GP_UI, n_samples=15, experience = None, async_plant=False, name='MC_PILCO', filename_prefix=None):
        super(MC_PILCO, self).__init__(params, plant_class, policy_class, cost_func,viz_class, dynmodel_class, experience, async_plant, name, filename_prefix)
        self.m_rng = theano.sandbox.rng_mrg.MRG_RandomStreams(lasagne.random.get_rng().randint(1,2147462579))
        self.trajectory_samples = theano.shared(np.array(n_samples).astype('int32'), name="%s>trajectory_samples"%(self.name) ) 
        D = self.mx0.get_value().size
        self.xp = theano.shared(np.zeros((n_samples, D),dtype=theano.config.floatX))
        self.dxdxp = theano.shared(np.zeros((n_samples*D, n_samples, D),dtype=theano.config.floatX))
        self.resample = params['resample'] if 'resample' in params else False
    
    def propagate_state(self,x,u=None,*args,**kwargs):
        ''' Given a set of input states, this function returns predictions for the next states.
            This is done by 1) evaluating the current policy 2) using the dynamics model to estimate 
            the next state. If x has shape [n,D] where n is tthe number of samples and D is the state dimension,
            this function will return x_next with shape [n,D] representing the next states and costs with shape [n,1]
        '''
        dynmodel = self.dynamics_model if kwargs.get('dynmodel',None) is None else kwargs.get('dynmodel',None)
        policy = self.policy if kwargs.get('policy',None) is None else kwargs.get('policy',None)
        resample = kwargs.get('resample',self.resample)
        iid_per_eval = kwargs.get('iid_per_eval',False)

        # resample if requested
        if resample:
            n,D = x.shape
            n = n.astype(theano.config.floatX)
            mx = x.mean(0)
            Sx = x.T.dot(x)/n - tt.outer(mx,mx)
            z = self.m_rng.normal(x.shape)
            x = mx + z.dot(tt.slinalg.cholesky(Sx).T)

        # convert angles from input states to their complex representation
        D_ = self.mx0.get_value().size
        xa = utils.gTrig(x,self.angle_idims,D_)

        if u is None:
            # compute control signal (with noisy state measurement)
            sn = tt.exp(dynmodel.logsn)
            x_noisy = x + self.m_rng.normal(x.shape).dot(tt.diag(sn))
            xa_noisy = utils.gTrig(x_noisy,self.angle_idims,D_)
            u = policy.evaluate(xa_noisy, symbolic=True, iid_per_eval=iid_per_eval, return_samples=True)
        
        # build state-control vectors
        xu = tt.concatenate([xa,u],axis=1)

        # predict the change in state given current state-control for each particle 
        delta_x = dynmodel.predict_symbolic(xu, iid_per_eval=iid_per_eval, return_samples=True)
        
        # compute the successor states
        x_next = x + delta_x

        return x_next

    def rollout_single_step(self,mc,Sc,x,gamma,gamma0,*args,**kwargs):
        ''' 
        Propagates the state distribution and computes the associated cost
        '''
        dynmodel = self.dynamics_model if kwargs.get('dynmodel',None) is None else kwargs.get('dynmodel',None)
        policy = self.policy if kwargs.get('policy',None) is None else kwargs.get('policy',None)
        cost = self.cost if kwargs.get('cost',None) is None else kwargs.get('cost',None)
        # compute next state distribution
        x_next = self.propagate_state(x,None,dynmodel,policy)

        # get cost ( via moment matching, which will penalize policies that result in multimodal trajectory distributions )
        n,D = x.shape
        n = n.astype(theano.config.floatX)
        sn2 = tt.exp(2*dynmodel.logsn)
        mx_next = x_next.mean(0)
        Sx_next = x_next.T.dot(x_next)/n - tt.outer(mx_next,mx_next)
        mc_next, Sc_next = cost(mx_next,Sx_next + tt.diag(sn2))

        return [gamma*mc_next, gamma*gamma*Sc_next, x_next, gamma*gamma0]

    def get_rollout(self, mx0, Sx0, H, gamma0, n_samples, dynmodel=None, policy=None, cost=None):
        ''' Given some initial state distribution Normal(mx0,Sx0), and a prediction horizon H 
        (number of timesteps), returns a set of trajectories sampled from the dynamics model and 
        the discounted costs for each step in the trajectory.'''
        utils.print_with_stamp('Computing symbolic expression graph for belief state propagation',self.name)
        dynmodel = self.dynamics_model if dynmodel is None else dynmodel
        policy = self.policy if policy is None else policy
        cost = self.cost if cost is None else cost
        
        # draw initial set of particles
        z0 = self.m_rng.normal((n_samples,mx0.shape[0]))
        Lx0 = tt.slinalg.cholesky(Sx0)
        x0 = mx0 + z0.dot(Lx0.T)

        # initial state cost ( fromt he input distribution )
        mv0, Sv0 = self.cost(mx0,Sx0)

        # define the rollout step function for the provided dynmodel, policy and cost
        rollout_single_step = partial(self.rollout_single_step, dynmodel=dynmodel,policy=policy,cost=cost)

        # do first iteration of scan here (so if we have any learning iteration updates that depend on computations inside the scan loop, all the inputs are defined outisde the scan)
        mv1, Sv1, x1, gamma1 = rollout_single_step(mv0,Sv0,x0,gamma0,gamma0)

        # make sure we return the dictionary with the updates that should occur during rollouts and at the beginning of each rollout
        updates = theano.updates.OrderedUpdates()
        updates += dynmodel.get_updates()  # this call need to happen **before** the call to scan! This is to ensure that the mask updates are initialized
        if hasattr(policy,'get_updates'):
            updates += policy.get_updates()  # this call need to happen **before** the call to scan! This is to ensure that the mask updates are initialized

        # these are the shared variables that will be used in the scan graph.
        # we need to pass them as non_sequences here (see: http://deeplearning.net/software/theano/library/scan.html)
        shared_vars = [mx0,Sx0]
        shared_vars.extend(dynmodel.get_all_shared_vars())
        shared_vars.extend(policy.get_all_shared_vars())
        # loop over the planning horizon
        rollout_output, rollout_updts = theano.scan(fn=rollout_single_step, 
                                            outputs_info=[mv1,Sv1,x1,gamma1], 
                                            non_sequences=[gamma0]+shared_vars,
                                            n_steps=H-1, 
                                            strict=True,
                                            allow_gc=False,
                                            name="%s>rollout_scan"%(self.name))

        updates += rollout_updts
        
        mean_costs, var_costs, trajectories, gamma_ = rollout_output

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


    def train_dynamics_from_rollouts(self,batchsize=100, maxiters=5000, dynmodel=None, dynmodel_class=None, dynmodel_params=None):
        ''' Treats the dynamics fitting as training a recurrent neural net. For training LSTMs, we need to
            make sure we can feed the data one step at a time, and reset the internal state of the LSTM
            at the start of every episode/sequence'''
        # TODO this assumes we're using a neural net built with lasagne
        if dynmodel is None:
            if hasattr(self,'dynamics_model'):
                dynmodel = self.dynamics_model

        if dynmodel is None:
            if dynmodel_class is None: dynmodel_class = self.dynmodel_class
            if dynmodel_params is None: dynmodel_params = self.dynmodel_params
            # initialize dynamics model
            dynamics_filename = self.filename+'_dynamics'
            dynmodel = dynmodel_class(filename=dynamics_filename,**dynmodel_params)

            if self.dynamics_model is None:
                self.dynamics_model = dynmodel

        # get dataset as < (x_t,u_t), delta_x_{t+1} > for updating the dataset statstics (mean and stdev of inputs and outputs)
        X,Y = self.experience.get_dynmodel_dataset(angle_dims=self.angle_idims, deltas=True)
        dynmodel.update_dataset_statistics(X,Y)

        if not hasattr(self,'train_dynamics_fn'):
            dynmodel.init_loss()
            # we will generate trajectories by applying actions from the experience data
            # and compute errors from the differences between predicted and observed states
            X = tt.tensor3('X')  #  H x n x D
            U = tt.tensor3('U')  #  H x n x U

            # do first iteration of scan here (so if we have any learning iteration updates that depend on computations inside the scan loop, all the inputs are defined outisde the scan)
            def propagate_fn(u,x,*args):
                return self.propagate_state(x,u,dynmodel=dynmodel)

            x1 = propagate_fn(U[0],X[0])     # n x D
            updates = theano.updates.OrderedUpdates()
            updates += dynmodel.get_dropout_masks()
            
            nll = tt.square((x1 - x[1])*tt.exp(-pred_noise)).sum(-1) + pred_noise.sum(-1)
            #nll = tt.square(x1[None,:,:] - X[1]).sum(-1)
            loss = nll.mean()
            if hasattr(dynmodel,'get_regularization_term'):
                loss += dynmodel.get_regularization_term()
            utils.print_with_stamp('Compiling rollout based dynamics model loss',self.name)
            params = lasagne.layers.get_all_params(dynmodel.network,trainable=True)
            updates += lasagne.updates.adam(loss,params,learning_rate=1e-3)
            #if dynmodel.learn_noise and not dynmodel.heteroscedastic:
            #    updates.update(lasagne.updates.adam(loss,[pred_noise],learning_rate=1e-4))
            self.train_dynamics_fn = theano.function([X,U],loss,updates=updates)
            utils.print_with_stamp("Done compiling.",self.name)


        # get trajectory data
        X = np.array(self.experience.states) # ep x H x D
        U = np.array(self.experience.actions) # ep x H x U

        # build all subsequences
        seqs = {}
        X_strides = (X.itemsize*X.shape[1]*X.shape[2],X.itemsize*X.shape[2],X.itemsize*X.shape[2],X.itemsize)
        U_strides = (U.itemsize*U.shape[1]*U.shape[2],U.itemsize*U.shape[2],U.itemsize*U.shape[2],U.itemsize)
        for l in xrange(2,4):
            X_l = np.lib.stride_tricks.as_strided(X, shape=(X.shape[0],X.shape[1]-l+1,l,X.shape[2]), strides=X_strides)
            U_l = np.lib.stride_tricks.as_strided(U, shape=(U.shape[0],U.shape[1]-l+1,l,U.shape[2]), strides=X_strides)
            # The reshapes generate a data copy, so there might be better ways to handle this
            X_l = X_l.reshape(X_l.shape[0]*X_l.shape[1],X_l.shape[2],X_l.shape[3])
            U_l = U_l.reshape(U_l.shape[0]*U_l.shape[1],U_l.shape[2],U_l.shape[3])
            seqs[l] = (X_l,U_l)

        # make sure we have dropout samples of the appropriate size
        dynmodel.draw_model_samples(batchsize)

        # training loop
        for i in xrange(maxiters):
            # sample trajectories (or trajectory segments) from dataset 
            seq_len = 2#np.random.randint(3,10)
            X_l, U_l = seqs[seq_len]
            indices = np.random.choice(X_l.shape[1],batchsize)
            Xi, Ui =X_l[indices].transpose(1,0,2),U_l[indices].transpose(1,0,2)
            loss = self.train_dynamics_fn(Xi,Ui)
            #loss = self.train_dynamics_fn(X.tranpose(1,0,2),Y.transpose(1,0,2))
            utils.print_with_stamp('loss: %f iter: %d'%(loss,i),same_line=True,name=self.name)
        print ''
        dynmodel.draw_model_samples(self.trajectory_samples.get_value())
        return dynmodel

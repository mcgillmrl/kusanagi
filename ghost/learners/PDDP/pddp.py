import numpy as np
from utils import print_with_stamp, gTrig_np, gTrig2, gTrig2_np
from ghost.learners.EpisodicLearner import EpisodicLearner
from ghost.learners.ExperienceDataset import ExperienceDataset
from ghost.regression.GP import GP_UI
from ghost.control import LocalLinearPolicy
from theano.tensor.nlinalg import matrix_inverse, pinv
import theano
from theano.misc.pkl_utils import dump as t_dump, load as t_load

class PDDP(EpisodicLearner):
    def __init__(self, params, plant_class, policy_class=LocalLinearPolicy, cost_func=None, viz_class=None, dynmodel_class=GP_UI, experience = None, async_plant=False, name='PDDP', wrap_angles=False, filename_prefix=None):
        self.dynamics_model = None
        self.wrap_angles = wrap_angles
        self.rollout=None
        self.policy_gradient=None
        self.mx0 = np.array(params['x0']).squeeze()
        self.Sx0 = np.array(params['S0']).squeeze()
        self.angle_idims = params['angle_dims']
        self.maxU = params['policy']['maxU']        

        # input dimensions to the dynamics model are (state dims - angle dims) + 2*(angle dims) + control dims
        dyn_idims = len(self.mx0) + len(self.angle_idims) + len(self.maxU)
        # output dimensions are state dims
        dyn_odims = len(self.mx0)
        # initialize dynamics model (TODO pass this as argument to constructor)
        if 'dynmodel' not in params:
            params['dynmodel'] = {}
        params['dynmodel']['idims'] = dyn_idims
        params['dynmodel']['odims'] = dyn_odims
        self.params = params
        self.dynamics_model = dynmodel_class(**params['dynmodel'])
        self.next_episode = 0
        self.min_cost = None
        self.backward_propagation_function = None
        self.forward_pass = None
        self.policy_update = None


        # initialise parent class
        filename_prefix = name+'_'+self.dynamics_model.name if filename_prefix is None else filename_prefix
        super(PDDP, self).__init__(params, plant_class, policy_class, cost_func,viz_class, experience, async_plant, name, filename_prefix)


    def save(self):
        ''' Saves the state of the learner, including the parameters of the policy and the dynamics model'''
        super(PDDP,self).save()
        self.dynamics_model.save()

    def load(self):
        ''' Loads the state of the learner, including the parameters of the policy and the dynamics model'''
        super(PDDP,self).load()
        self.dynamics_model.load()

    def init_rollout(self, derivs=False):
        ''' This compiles the rollout function, which applies the policy and predicts the next state 
            of the system using the learned GP dynamics model '''
                # define the function for a single propagation step
        def rollout_single_step(mx,Sx, eval_t = None, u=None, use_gTrig = False):
            D = Sx.shape[1]
            # compute distribution of control signal
            logsn = self.dynamics_model.loghyp[:,-1]
            Sx_ = Sx + theano.tensor.diag(0.5*theano.tensor.exp(2*logsn))# noisy state measurement
            u_prev, Su, Cu = self.policy.evaluate(mx, Sx_,symbolic=True, t = eval_t, u=u, use_gTrig = use_gTrig)
            if u is not None:
                u_prev = u
            # compute state control joint distribution
            #mx, Sx, _ = gTrig2(mx, Sx, self.angle_idims, len(self.mx0))
            mxu = theano.tensor.concatenate([mx,u_prev])
            q = Sx.dot(Cu)
            Sxu_up = theano.tensor.concatenate([Sx,q],axis=1)
            Sxu_lo = theano.tensor.concatenate([q.T,Su],axis=1)
            Sxu = theano.tensor.concatenate([Sxu_up,Sxu_lo],axis=0)

            #  predict the change in state given current state-action
            # C_deltax = inv (Sxu) dot Sxu_deltax
            m_deltax, S_deltax, C_deltax = self.dynamics_model.predict_symbolic(mxu,Sxu)

            # compute the successor state distribution
            mx_next = mx + m_deltax
            Sx_deltax = Sxu[:D,:].dot(C_deltax)
            Sx_next = Sx + S_deltax + Sx_deltax + Sx_deltax.T

            #  get cost:
            #mcost, Scost = self.cost_symbolic(mx_next,Sx_next)

            return [mx_next,Sx_next,u_prev]

        # define input variables
        print_with_stamp('Compiling forward dynamics/backprop/local_optimal_policy functions',self.name)
        mx = theano.tensor.vector('mx')
        Sx = theano.tensor.matrix('Sx')
        H = theano.tensor.iscalar('H')
        gamma = theano.tensor.scalar('gamma')
        
        # For this, you'll need to read how theano.scan and theano.tensor.grad work ( look at pilco.py for an example )
        def forward_dynamics(z, u):
            ''' This computes the Jacobian matrices Fu and Fx which satisfy the first order expansion of the dynamics.
            We use a Gaussian belief augmented state vector to incorporate uncertainty.'''
            # TODO use the rollout single step file to do the forward dynamics
            # this should return the list of all matrices (F_k)^u and (F_k)^x
            # ( Eq. (11) )
            n_shape = self.params['x0'].shape[0]
            mx_next, Sx_next, _ = self.rollout_single_step(z[:n_shape],z[n_shape:].reshape((n_shape,n_shape)),u = u)
            z_next = theano.tensor.concatenate((mx_next, Sx_next.flatten()))
            z_next_shape = z_next.shape[0]
            u_shape = u.shape[0]
            Fx = theano.tensor.jacobian(z_next.flatten(), z, disconnected_inputs='ignore')
            Fu = theano.tensor.jacobian(z_next.flatten(), u, disconnected_inputs='ignore')
            Fx = Fx.reshape((z_next_shape, z_next_shape))
            Fu = Fu.reshape((z_next_shape, u_shape))

            return [Fx, Fu]

        def backward_propagation(z,u,V,Vx,Vxx,Fx,Fu): 
            #TODO compute Q_t and the associated L_t and I_t
            # ( Eq. (17) ). Use self.cost_symbolic(mx,Sx) to get the cost of the current state. We
            # might need to create a new cost function that takes u as an optional input, so we can call self.cost_symbolic(mx,Sx,u)
            # you only need to use the mean of the cost to compute the derivatives ( See the paragraph after Eq 13 )
            # Here C is used in place of script L
            n_shape = self.params['x0'].shape[0]
            n = n_shape*n_shape + n_shape
            m = u.shape[0]
            C, _ = self.cost_symbolic(z[:n_shape],z[n_shape:].reshape((n_shape,n_shape)),u = u)
            Cx = theano.tensor.jacobian(C.flatten(), z)
            Cxx = theano.tensor.jacobian(Cx.flatten(), z)
            Cu = theano.tensor.jacobian(C.flatten(),u,disconnected_inputs='ignore')
            Cux = theano.tensor.jacobian(Cu.flatten(),z,disconnected_inputs='ignore')
            Cuu = theano.tensor.jacobian(Cu.flatten(),u,disconnected_inputs='ignore')

            Cx = theano.tensor.reshape(Cx, (n,))
            Cxx = theano.tensor.reshape(Cxx, (n,n))
            Cu = theano.tensor.reshape(Cu, (m,))
            Cux = theano.tensor.reshape(Cux, (m,n))
            Cuu = theano.tensor.reshape(Cuu, (m,m))

            Qx = Cx + Vx.dot(Fx)
            Qu = Cu + Vx.dot(Fu)
            Qxx = Cxx + theano.tensor.transpose(Fx).dot(Vxx.dot(Fx))
            Qux = Cux + theano.tensor.transpose(Fu).dot(Vxx.dot(Fx))
            Quu = Cuu + theano.tensor.transpose(Fu).dot(Vxx.dot(Fu))

            I = -pinv(Quu).dot(Qu)
            L = -pinv(Quu).dot(Qux)

            V_prev = V + Qu.dot(I)
            Vx_prev = Qx + Qu.dot(L)
            Vxx_prev = Qxx + Qux.T.dot(L)

            return [V_prev, Vx_prev, Vxx_prev, I, L]

        #compile the previous functions in theano
        z = theano.tensor.vector('z')
        z_next = theano.tensor.vector('z_next')
        mx = theano.tensor.vector('mx')
        Sx = theano.tensor.matrix('Sx')
        u = theano.tensor.vector('u')
        V = theano.tensor.scalar('V')
        Vx = theano.tensor.vector('Vx')
        Vxx = theano.tensor.matrix('Vxx')
        Fx = theano.tensor.matrix('Fx')
        Fu = theano.tensor.matrix('Fu')
        I = theano.tensor.vector('I')
        L = theano.tensor.matrix('L')
        self.rollout_single_step = rollout_single_step
        self.forward_dynamics = forward_dynamics
        self.backward_propagation_function = theano.function([z,u,V,Vx,Vxx,Fx,Fu], backward_propagation(z,u,V,Vx,Vxx,Fx,Fu))
        print_with_stamp('Compiled backward_propagation_function',self.name)

    def compile_forward_pass(self):

        Fx = theano.tensor.matrix('Fx')
        Fu = theano.tensor.matrix('Fu')
        z = theano.tensor.vector('z')
        u = theano.tensor.vector('u')
        z_list = theano.tensor.matrix('z_list')
        u_list = theano.tensor.matrix('u_list')

        def forward_pass_func(z,u, *args):
            return self.forward_dynamics(z, u)

        shared_vars = []
        shared_vars.extend(self.dynamics_model.get_all_shared_vars())
        shared_vars.extend(self.policy.get_all_shared_vars())
        (Fx_list, Fu_list), updts = theano.scan(fn=forward_pass_func, 
                                                      outputs_info=None, 
                                                      non_sequences=shared_vars,
                                                      sequences=[z_list,u_list], 
                                                      strict=True,
                                                      allow_gc=False)
        self.forward_pass = theano.function([z_list, u_list], [Fx_list, Fu_list], updates = updts)
        print_with_stamp('Compiled forward_pass',self.name)


    def compile_policy_update(self):
        H_steps = int(np.ceil(self.H/self.plant.dt))
        mx0 = theano.tensor.vector('mx0')
        Sx0 = theano.tensor.matrix('Sx0')
        temp_u, _ , _ ,_ = self.policy.get_params() 
        u0 = theano.tensor.zeros(temp_u[0].shape)
        u0 = theano.tensor.unbroadcast(u0,0)
        sum0 = theano.tensor.as_tensor_variable(0.0).astype(theano.config.floatX)

        def policy_update_func(t, mx, Sx, u, sum0, *args):
            mx_next, Sx_next, u_new = self.rollout_single_step(mx, Sx, eval_t = t)
            step_cost, _ = self.cost_symbolic(mx_next, Sx_next)
            return mx_next, Sx_next, u_new, step_cost


        shared_vars = []
        shared_vars.extend(self.dynamics_model.get_all_shared_vars())
        shared_vars.extend(self.policy.get_all_shared_vars())
        (mx_list,Sx_list,u_list, cost_list), updts = theano.scan(fn=policy_update_func, 
                                                      outputs_info=[mx0, Sx0, u0, sum0],
                                                      sequences=[theano.tensor.arange(H_steps)],
                                                      non_sequences=shared_vars,
                                                      strict=True,
                                                      allow_gc=False)
        mx_list = theano.tensor.concatenate([mx0.dimshuffle('x',0), mx_list])
        Sx_list = theano.tensor.concatenate([Sx0.dimshuffle('x',0,1), Sx_list])
        self.policy_update = theano.function(inputs=[mx0, Sx0], outputs=[mx_list, Sx_list, u_list, cost_list.sum()], updates = updts)
        print_with_stamp('Compiled policy_update',self.name)

    def train_policy(self):
        ''' Runs the trajectory optimization loop: 1. forward pass 2. backward propagation 3. update policy 4. repeat '''
        	#STEP 1: Use forward dynamics to generate Fx and Fu for each timestep. Requires the state from 0 to T

        	#STEP 2: Use backward propagation to create I and L for each timestep which is used to improve the control. For this we require state and action
        	# 		 at each timestep, Fx and Fu at each timestep, and V, Vx, Vxx from the timestep following the current one.

        	#STEP 3: Forward propagate to get new optimal trajectory using new I and L. Our first action's delta_u is initialized as the matrix I and we calculate the
        	#		 t+1th step using the change between the nominal state z_bar and the new state z, and I and L to get delta_u for this state's action. For this we will
        	#		 need to use rollout_single_step without the policy.evaluate step, and instead use the new optimal action. This should return the sequence of states and
        	#		 actions that correspond to the new optimal trajectory. 

        #SETUP FOR STEP 1
        H_steps = int(np.ceil(self.H/self.plant.dt))
        Fx_list = [None for a0 in xrange(H_steps)]
        Fu_list = [None for a0 in xrange(H_steps)]
        z_nominal = [None for a0 in xrange(H_steps)]

        #SETUP FOR STEP 2
        V_list = [None for a0 in xrange(H_steps)]
        Vx_list = [None for a0 in xrange(H_steps)]
        Vxx_list = [None for a0 in xrange(H_steps)]
        I_list = [None for a0 in xrange(H_steps)]
        L_list = [None for a0 in xrange(H_steps)]
        
        if self.forward_pass is None or self. policy_update is None or self.min_cost is None:
            self.init_rollout()
            self.compile_policy_update()
            self.compile_forward_pass()
        
        # Get an initial min cost with the current dynamics model ( we need to reinintialize the minimum cost every time, since
        # the predictions from past runs might be completely wrong
        self.policy.t = 0
        mx_list, Sx_list, u_list, self.min_cost = self.policy_update(self.params['x0'], self.params['S0'])
        self.policy.set_params(uin = np.array(u_list))
        for i in xrange(0,H_steps):
            z_nominal[i] = np.concatenate([mx_list[i].flatten(),Sx_list[i].flatten()])
        self.policy.set_params(zin = np.array(z_nominal))
        self.policy.t = 0  

        print_with_stamp("Initial predicted cost of trajectory: [ %f ]"%(self.min_cost), self.name)

        #MAIN LOOP
        print_with_stamp('Training policy parameters [Iteration %d]'%(self.learning_iteration), self.name)
        self.learning_iteration += 1
        while self.n_evals < self.max_evals:
            #STEP 1
            print_with_stamp('Current policy iteration number: [%d] ... Running Forward Dynamics'%(self.n_evals), self.name, same_line=False)
            self.policy.t = 0
            u_list, z_list, _, _ = self.policy.get_params()
            Fx_list, Fu_list = self.forward_pass(z_list,u_list)

            #STEP 2
            print_with_stamp('Current policy iteration number: [%d] ... Running Backpropagation'%(self.n_evals), self.name, same_line=False)
            n = len(self.params['x0']) + len(self.angle_idims)
            n = n*n + n
            V_list[-1] = 1
            Vx_list[-1] = np.zeros((n,))
            Vxx_list[-1] = np.zeros((n,n))

            for i in reversed(xrange(1,H_steps)):
                V_list[i-1], Vx_list[i-1], Vxx_list[i-1], I_list[i], L_list[i] = self.backward_propagation_function(z_list[i],u_list[i],V_list[i],Vx_list[i],Vxx_list[i],Fx_list[i],Fu_list[i])
            _, _, _, I_list[0], L_list[0] = self.backward_propagation_function(z_list[0],u_list[0],V_list[0],Vx_list[0],Vxx_list[0],Fx_list[0],Fu_list[0])
            self.policy.t = 0

            self.policy.set_params(Ain = np.array(L_list))
            self.policy.set_params(Bin = np.array(I_list))

            #STEP 3
            print_with_stamp('Current policy iteration number: [%d] ... Running Trajectory Update'%(self.n_evals), self.name, same_line=False)
            abort = False
            self.policy.alpha.set_value(1.0)
            mx_list, Sx_list, u_list, trajectory_cost = self.policy_update(self.params['x0'], self.params['S0'])
            self.policy.t = 0
            line_search_iters = 0
            while trajectory_cost > self.min_cost:
                self.policy.alpha.set_value( self.policy.alpha.get_value()*0.5 )
                mx_list, Sx_list, u_list, trajectory_cost = self.policy_update(self.params['x0'], self.params['S0'])
                self.policy.t = 0
                line_search_iters += 1
                print_with_stamp('Current cost: %f, Current alpha: %f,  Linesearch iteration: %d    '%(trajectory_cost,self.policy.alpha.get_value(),line_search_iters),self.name,True)
                if line_search_iters == 100: # TODO put this as user parameter
                    abort = True
                    break
            if line_search_iters>0:
                print''
            mx_list = mx_list[:-1]
            Sx_list = Sx_list[:-1]

            #EXIT AND SAVE
            if not abort:
                cost_improvement = trajectory_cost - self.min_cost
                self.min_cost = trajectory_cost
                print_with_stamp("Finished with %d line search iterations and cost [ %f ] ( improvement [ %f ] )"%(line_search_iters,self.min_cost, cost_improvement), self.name)
                self.policy.set_params(uin = np.array(u_list))
                for i in xrange(0,H_steps):
                    z_nominal[i] = np.concatenate([mx_list[i].flatten(),Sx_list[i].flatten()])
                self.policy.set_params(zin = np.array(z_nominal))
                self.policy.t = 0        
                self.n_evals += 1
                self.policy.state_changed = True
                if abs(cost_improvement) < 1e-9:
                    self.n_evals = self.max_evals
            else:
                print_with_stamp("Could not find a better policy in this iteration. ( Current best cost: [ %f ] )"%(self.min_cost), self.name)
                self.n_evals = self.max_evals
        print "\n"
        #self.experience.reset()
        self.n_evals=0


    def train_dynamics(self):
        print_with_stamp('Training dynamics model',self.name)
        
        X = []
        Y = []
        x0 = []
        n_episodes = len(self.experience.states)
        # construct training dataset
        for i in xrange(n_episodes):
            x = np.array(self.experience.states[i])
            u = np.array(self.experience.actions[i])
            x0.append(x[0])

            # inputs are states, concatenated with actions (except for the last entry)
            x_ = gTrig_np(x, self.angle_idims)
            X.append( np.hstack((x_[:-1],u[:-1])) )
            # outputs are changes in state
            Y.append( x[1:] - x[:-1] )

        X = np.vstack(X)
        Y = np.vstack(Y)

        # get distribution of initial states
        x0 = np.array(x0)
        if n_episodes > 1:
            self.mx0 = x0.mean(0)[None,:]
            self.Sx0 = np.cov(x0.T)[None,:,:]
        else:
            self.mx0 = x0[None,:]
            self.Sx0 = 1e-2*np.eye(self.mx0.size)[None,:,:]

        if self.wrap_angles:
            # wrap angle differences to [-pi,pi]
            Y[:,self.angle_idims] = (Y[:,self.angle_idims] + np.pi) % (2 * np.pi ) - np.pi

        if self.dynamics_model is None:
            self.dynamics_model = GP_UI(X,Y)
        else:
            self.dynamics_model.set_dataset(X,Y)
 
        print_with_stamp('Dataset size:: Inputs: [ %s ], Targets: [ %s ]  '%(self.dynamics_model.X.shape.eval(),self.dynamics_model.Y.shape.eval()),self.name)
        self.dynamics_model.train()
        self.dynamics_model.save()
        print_with_stamp('Done training dynamics model',self.name)

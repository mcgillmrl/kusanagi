import numpy as np
from utils import print_with_stamp, gTrig_np, gTrig2, gTrig2_np
from ghost.learners.EpisodicLearner import *
from ghost.regression.GPRegressor import GP_UI
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
        def rollout_single_step(mx,Sx, alpha = 1, eval_t = None, u=None, with_angles = True):
            D=mx.shape[0]

            # compute distribution of control signal
            logsn = self.dynamics_model.loghyp[:,-1]
            Sx_ = Sx + theano.tensor.diag(0.5*theano.tensor.exp(2*logsn))# noisy state measurement
            u_prev, Su, Cu = self.policy.evaluate(mx, Sx_,symbolic=True, alpha = alpha, t = eval_t, u=u)
            
            # compute state control joint distribution
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
            mcost, Scost = self.cost_symbolic(mx_next,Sx_next, with_angles = with_angles)

            return [mx_next,Sx_next,u_prev]

        # define input variables
        print_with_stamp('Compiling forward dynamics/backprop/local_optimal_policy functions',self.name)
        mx = theano.tensor.vector('mx')
        Sx = theano.tensor.matrix('Sx')
        H = theano.tensor.iscalar('H')
        gamma = theano.tensor.scalar('gamma')
        
        # For this, you'll need to read how theano.scan and theano.tensor.grad work ( look at pilco.py for an example )
        def forward_dynamics(mx, Sx):
            ''' This computes the Jacobian matrices Fu and Fx which satisfy the first order expansion of the dynamics.
            We use a Gaussian belief augmented state vector to incorporate uncertainty.'''
            # TODO use the rollout single step file to do the forward dynamics
            # this should return the list of all matrices (F_k)^u and (F_k)^x
            # ( Eq. (11) )
            mx, Sx, _ = gTrig2(mx, Sx, self.angle_idims, mx.shape[0])
            [mx_next,Sx_next,u_prev] = rollout_single_step(mx, Sx, with_angles = False)
            mx_next, Sx_next, _ = gTrig2(mx_next, Sx_next, self.angle_idims, mx.shape[0])
            z = theano.tensor.concatenate((mx.flatten(),Sx.flatten()))
            z_next = theano.tensor.concatenate((mx_next.flatten(),Sx_next.flatten()))
            z_next_shape = z_next.shape[0]
            u_shape = u_prev.shape[0]
            Fx = theano.tensor.jacobian(z_next, z, disconnected_inputs='ignore')
            Fu = theano.tensor.jacobian(z_next, u_prev, disconnected_inputs='ignore')
            Fx = Fx.reshape((z_next_shape, z_next_shape))
            Fu = Fu.reshape((z_next_shape, u_shape))

            return [Fx, Fu, mx_next, Sx_next, u_prev]

        def backward_propagation(mx,Sx,u,V,Vx,Vxx,Fx,Fu): 
            #TODO compute Q_t and the associated L_t and I_t
            # ( Eq. (17) ). Use self.cost_symbolic(mx,Sx) to get the cost of the current state. We
            # might need to create a new cost function that takes u as an optional input, so we can call self.cost_symbolic(mx,Sx,u)
            # you only need to use the mean of the cost to compute the derivatives ( See the paragraph after Eq 13 )
            # Here C is used in place of script L
            n = mx.shape[0]
            n = n*n + n
            m = u.shape[0]
            z = theano.tensor.concatenate((mx, Sx.flatten()))
            C, _ = self.cost_symbolic(z[:mx.shape[0]],z[mx.shape[0]:].reshape(Sx.shape),u = u, with_angles = False)
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
        mx = theano.tensor.vector('mx')
        Sx = theano.tensor.matrix('Sx')
        self.forward_dynamics = forward_dynamics
        u = theano.tensor.vector('u')
        V = theano.tensor.scalar('V')
        Vx = theano.tensor.vector('Vx')
        Vxx = theano.tensor.matrix('Vxx')
        Fx = theano.tensor.matrix('Fx')
        Fu = theano.tensor.matrix('Fu')
        self.backward_propagation_function = theano.function([mx,Sx,u,V,Vx,Vxx,Fx,Fu], backward_propagation(mx,Sx,u,V,Vx,Vxx,Fx,Fu))
        print_with_stamp("Backprop compiled",self.name)
        I = theano.tensor.vector('I')
        L = theano.tensor.matrix('L')
        alpha = theano.tensor.scalar('alpha')
        self.rollout_single_step = rollout_single_step

    def compile_policy_update(self):
        H_steps = int(np.ceil(self.H/self.plant.dt))
        mx0 = theano.tensor.vector('mx0')
        Sx0 = theano.tensor.matrix('Sx0')
        temp_u, _ , _ ,_ = self.policy.get_params() 
        u0 = theano.tensor.zeros(temp_u[0].shape)
        u0 = theano.tensor.unbroadcast(u0,0)
        alpha0 = theano.tensor.scalar('alpha0')
        sum0 = theano.tensor.scalar('sum0')

        def policy_update_func(mx, Sx, alpha, u, sum0, *args):
            mx_next, Sx_next, u_new = self.rollout_single_step(mx, Sx, alpha = alpha, u = u)
            step_cost, _ = self.cost_symbolic(mx_next, Sx_next)
            return mx_next, Sx_next, alpha, u_new, step_cost


        shared_vars = []
        shared_vars.extend(self.dynamics_model.get_params(symbolic=True))
        shared_vars.extend(self.policy.get_params(symbolic=True))
        (mx_list,Sx_list,alpha_list,u_list, cost_list), updts = theano.scan(fn=policy_update_func, 
                                                      outputs_info=[mx0, Sx0, alpha0, u0, sum0], 
                                                      non_sequences=shared_vars,
                                                      n_steps=H_steps, 
                                                      strict=True,
                                                      allow_gc=False)
        mx_list = theano.tensor.concatenate([mx0.dimshuffle('x',0), mx_list])
        Sx_list = theano.tensor.concatenate([Sx0.dimshuffle('x',0,1), Sx_list])
        self.policy_update = theano.function([mx0, Sx0, alpha0, sum0], [mx_list, Sx_list, alpha_list, u_list, cost_list.sum()], updates = updts)
        print_with_stamp("Policy/Trajectory update compiled",self.name)

    def compile_forward_pass(self):
        H_steps = int(np.ceil(self.H/self.plant.dt))
        x0 = theano.tensor.vector('x0')
        S0 = theano.tensor.matrix('S0')
        temp_u, _ , _ ,_ = self.policy.get_params() 
        u0 = theano.tensor.zeros(temp_u[0].shape)
        u0 = theano.tensor.unbroadcast(u0,0)
        Fx0 = theano.tensor.zeros((x0.shape[0]*x0.shape[0] + x0.shape[0], x0.shape[0]*x0.shape[0] + x0.shape[0]))
        Fu0 = theano.tensor.zeros((x0.shape[0]*x0.shape[0] + x0.shape[0],u0.shape[0]))
        Fu0 = theano.tensor.unbroadcast(Fu0, 1)

        def forward_func(Fx,Fu,mx,Sx,u,*args):
             Fx, Fu, mx_next, Sx_next, u_prev = self.forward_dynamics(mx,Sx)
             return Fx, Fu, mx_next, Sx_next, u_prev
        
        shared_vars = []
        shared_vars.extend(self.dynamics_model.get_params(symbolic=True))
        shared_vars.extend(self.policy.get_params(symbolic=True))
        (Fx_list,Fu_list,mx_list,Sx_list,u_list), updts = theano.scan(fn=forward_func, 
                                                      outputs_info=[Fx0,Fu0,x0, S0, u0], 
                                                      non_sequences=shared_vars,
                                                      n_steps=H_steps, 
                                                      strict=True,
                                                      allow_gc=False)
        mx_list = theano.tensor.concatenate([x0.dimshuffle('x',0), mx_list])
        Sx_list = theano.tensor.concatenate([S0.dimshuffle('x',0,1), Sx_list])
        self.forward_pass = theano.function([x0, S0], [Fx_list,Fu_list,mx_list,Sx_list,u_list], updates = updts)
        print_with_stamp("Forward dynamics compiled",self.name)

    def train_policy(self):
        ''' Runs the trajectory optimization loop: 1. forward pass 2. backward propagation 3. update policy 4. repeat '''
        	#STEP 1: Use forward dynamics to generate Fx and Fu for each timestep. Requires the state from 0 to T

        	#STEP 2: Use backward propagation to create I and L for each timestep which is used to improve the control. For this we require state and action
        	# 		 at each timestep, Fx and Fu at each timestep, and V, Vx, Vxx from the timestep following the current one.

        	#STEP 3: Forward propagate to get new optimal trajectory using new I and L. Our first action's delta_u is initialized as the matrix I and we calculate the
        	#		 t+1th step using the change between the nominal state z_bar and the new state z, and I and L to get delta_u for this state's action. For this we will
        	#		 need to use rollout_single_step without the policy.evaluate step, and instead use the new optimal action. This should return the sequence of states and
        	#		 actions that correspond to the new optimal trajectory. 

        
        self.policy.t = 0
        if self.forward_pass is None or self. policy_update is None or self.min_cost is None:
            self.init_rollout()
            self.compile_forward_pass()
            self.compile_policy_update()
            _, _, _, _, self.min_cost = self.policy_update(self.plant.x0, self.plant.S0, 0, 0)
            print "MIN COST INIT:" + str(self.min_cost)

        #SETUP FOR STEP 1
        H_steps = int(np.ceil(self.H/self.plant.dt))
        z_nominal = [None for a0 in xrange(H_steps)]


        #SETUP FOR STEP 2
        V_list = [None for a0 in xrange(H_steps)]
        Vx_list = [None for a0 in xrange(H_steps)]
        Vxx_list = [None for a0 in xrange(H_steps)]
        I_list = [None for a0 in xrange(H_steps)]
        L_list = [None for a0 in xrange(H_steps)]


        #MAIN LOOP
        print_with_stamp('Training policy parameters [Iteration %d]'%(self.learning_iteration), self.name)
        self.learning_iteration += 1
        while self.n_evals < self.max_evals:



            print_with_stamp('Current policy iteration number: [%d]'%(self.n_evals), self.name, same_line=False)
            self.policy.t = 0

            #STEP 1
            print_with_stamp('Current policy iteration number: [%d] ... Running Forward Dynamics'%(self.n_evals), self.name, same_line=False)

            Fx_list, Fu_list, mx_list, Sx_list, u_list = self.forward_pass(self.plant.x0, self.plant.S0)
            mx_list = mx_list[:-1]
            Sx_list = Sx_list[:-1]
            self.policy.t = 0

            for i in xrange(0,H_steps):
                z_nominal[i] = np.concatenate([mx_list[i].flatten(),Sx_list[i].flatten()])
            zin = np.array(z_nominal)
            self.policy.set_params(zin = zin)
            uin = np.array(u_list)
            self.policy.set_params(uin = uin)

            #STEP 2
            print_with_stamp('Current policy iteration number: [%d] ... Running Backpropagation'%(self.n_evals), self.name, same_line=False)
            n = len(self.plant.x0)
            n = n*n + n
            V_list[-1] = 0
            Vx_list[-1] = np.zeros((n,))
            Vxx_list[-1] = np.zeros((n,n))

            for i in reversed(xrange(1,H_steps)):
                V_list[i-1], Vx_list[i-1], Vxx_list[i-1], I_list[i], L_list[i] = self.backward_propagation_function(mx_list[i],Sx_list[i],u_list[i],V_list[i],Vx_list[i],Vxx_list[i],Fx_list[i],Fu_list[i])
            _, _, _, I_list[0], L_list[0] = self.backward_propagation_function(mx_list[0],Sx_list[0],u_list[0],V_list[0],Vx_list[0],Vxx_list[0],Fx_list[0],Fu_list[0])
            self.policy.t = 0

            self.policy.set_params(Bin = np.array(I_list))
            self.policy.set_params(Ain = np.array(L_list))


            #STEP 3
            print_with_stamp('Current policy iteration number: [%d] ... Running Trajectory Update'%(self.n_evals), self.name, same_line=False)
            abort = False
            alpha = 1.0
            mx_list, Sx_list, _, u_list, trajectory_cost = self.policy_update(self.plant.x0, self.plant.S0, alpha, 0)
            self.policy.t = 0
            line_search_iters = 0
            while trajectory_cost > self.min_cost:
                alpha = alpha*0.8
                mx_list, Sx_list, _, u_list, trajectory_cost = self.policy_update(self.plant.x0, self.plant.S0, alpha, 0)
                self.policy.t = 0
                line_search_iters += 1

                if line_search_iters == 100:
                    abort = True
                    break
            
            mx_list = mx_list[:-1]
            Sx_list = Sx_list[:-1]
            

            #EXIT AND SAVE
            if not abort:
                self.min_cost = trajectory_cost
                print "Finished with " + str(line_search_iters) + " line search iterations and cost " + str(self.min_cost)
                uin = np.array(u_list)
                self.policy.set_params(uin = uin)
                for i in xrange(0,H_steps):
                    z_nominal[i] = np.concatenate([mx_list[i].flatten(),Sx_list[i].flatten()])
                zin = np.array(z_nominal)
                self.policy.set_params(zin = zin)
                self.policy.t = 0        
                self.n_evals += 1
                self.policy.state_changed = True
            else:
                print "Could not find a better policy in this iteration. (Current best cost: " + str(self.min_cost) + ")"
                self.n_evals = self.max_evals
        print "\n"
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
 
        self.dynamics_model.train()
        self.dynamics_model.save()
        print_with_stamp('Done training dynamics model',self.name)

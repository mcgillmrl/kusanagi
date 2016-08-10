import lasagne
import numpy as np
import os
import theano
import time
import utils

from functools import partial
from matplotlib import pyplot as plt
from scipy.optimize import minimize, basinhopping

from ghost.learners.ExperienceDataset import ExperienceDataset
from ghost.control import RandPolicy
from base.Loadable import Loadable

DETERMINISTIC_MIN_METHODS = ['L-BFGS-B', 'TNC', 'BFGS', 'SLSQP', 'CG']
STOCHASTIC_MIN_METHODS = {'SGD': lasagne.updates.sgd,
                          'MOMENTUM': lasagne.updates.momentum,
                          'NESTEROV': lasagne.updates.nesterov_momentum,
                          'NESTEROV_MOMENTUM': lasagne.updates.nesterov_momentum,
                          'ADAGRAD': lasagne.updates.adagrad,
                          'RMSPROP': lasagne.updates.rmsprop,
                          'ADADELTA': lasagne.updates.adadelta,
                          'ADAM': lasagne.updates.adam}

class EpisodicLearner(Loadable):
    def __init__(self, params, plant_class, policy_class, cost_func=None, viz_class=None, experience = None, async_plant=False, name='EpisodicLearner', filename_prefix=None, learn_from_iteration=-1, task_name = None):
        self.name = name
        if task_name is not None:
            os.environ['KUSANAGI_RUN_OUTPUT'] = os.path.join(utils.get_output_dir(),task_name)
            utils.print_with_stamp("Changed KUSANAGI_RUN_OUTPUT to %s"%(os.environ['KUSANAGI_RUN_OUTPUT']))
        # initialize plant
        if 'x0' not in params['plant']:
            params['plant']['x0'] = params['x0']
            params['plant']['S0'] = params['S0']
        self.plant = plant_class(**params['plant'])
        # initialize policy
        params['policy']['angle_dims'] = params['angle_dims']
        self.policy = policy_class(**params['policy'])
        # set filename    
        self.filename = self.name+'_'+self.plant.name+'_'+self.policy.name if filename_prefix is None else filename_prefix+'_'+self.plant.name+'_'+self.policy.name
        # initialize cost
        params['cost']['angle_dims'] = params['angle_dims']
        self.cost = partial(cost_func, params=params['cost']) if cost_func is not None else None
        self.evaluate_cost = None
        # initialize vizualization
        self.viz = viz_class(self.plant) if viz_class is not None else None
        # initialize experience dataset
        self.experience = ExperienceDataset(filename_prefix=self.filename) if experience is None else experience
        self.learn_from_iteration = learn_from_iteration #for usage of this parameter, see the load() method below
        # initialize learner state variables
        self.n_episodes = 0
        self.angle_idims = params['angle_dims'] if 'angle_dims' in params else []
        self.H = params['H'] if 'H' in params else 10.0
        self.discount = params['discount'] if 'discount' in params else 1.0
        self.max_evals = params['max_evals'] if 'max_evals' in params else 150
        self.conv_thr = params['conv_thr'] if 'conv_thr' in params else 1e-12
        self.learning_rate = params['learning_rate'] if 'learning_rate' in params else 1.0
        self.min_method = params['min_method'] if 'min_method' in params else "L-BFGS-B"
        self.async_plant = async_plant
        self.learning_iteration = 0;
        self.n_evals = 0
        self.realtime = params['realtime'] if 'realtime' in params else True
        
        Loadable.__init__(self,name=name,filename=self.filename)
        self.register(['n_episodes','angle_idims','async_plant','evaluate_cost','H','discount','learning_iteration','n_evals'])

        # try loading from file, initialize from scratch otherwise
        utils.print_with_stamp('Initialising new %s learner'%(self.name),self.name)
        self.state_changed = True

    def load(self, output_folder=None,output_filename=None):
        # load learner state
        super(EpisodicLearner,self).load(output_folder,output_filename)
        
        # load policy and experience separately
        policy_filename = None
        if output_filename is not None:
            policy_filename = output_filename + '_policy'
        self.policy.load(output_folder,policy_filename)
        
        experience_filename = None
        if output_filename is not None:
            experience_filename = output_filename + '_experience'
        self.experience.load(output_folder,experience_filename)

        #initialize cost if neeeded
        self.init_cost()

        # initialize policy if needed
        p = self.policy.get_params()
        for pi in p:
            if pi is None or pi.size == 0:
                self.policy.set_default_parameters()
                break
        
        '''USAGE OF LEARN_FROM_ITERATION
        -1: Resume learning from most recent state
        0: Restart learning completely (re-do random trials)
        RANDOM : if you pass this string, you will keep the random trial data
        n>0 : resume from policy n (currently does not keep experience from policy n having been applied, we need to apply_controller to get data from this policy)'''
        '''
        if self.learn_from_iteration != -1: #if we want to load from a specific iteration, revert policy and experience to what it was at that iter
                if self.learn_from_iteration == 0:
                    utils.print_with_stamp('Resetting data completely')
                    self.experience.time_stamps = []
                    self.experience.states = []
                    self.experience.actions = []
                    self.experience.immediate_cost = []
                    self.experience.episode_labels = []
                    self.experience.curr_episode = -1
                    self.experience.policy_history = []
                    self.experience.episode_labels = []
                    self.policy.set_default_parameters()
                elif not hasattr(self.experience, 'policy_history'):
                    pass
                else:
                    utils.print_with_stamp('Loading from iteration %s and reverting datasets to that iteration'%(str(self.learn_from_iteration)))
                    entry_num = 0
                    try:
                        while self.experience.episode_labels[entry_num] != self.learn_from_iteration:
                            entry_num += 1
                        # while self.experience.episode_labels[entry_num] == self.learn_from_iteration:
                        #     entry_num += 1
                    except IndexError:
                        utils.print_with_stamp('ERROR: You are trying to load from an iteration that has not been performed. Press enter to continue by loading the most recent data, or CTRL-C to close.')
                        raw_input()
                        entry_num -= 1
                        self.learn_from_iteration = self.experience.episode_labels[entry_num]
                    self.experience.time_stamps = self.experience.time_stamps[:entry_num]
                    self.experience.states = self.experience.states[:entry_num]
                    self.experience.actions = self.experience.actions[:entry_num]
                    self.experience.immediate_cost = self.experience.immediate_cost[:entry_num]
                    self.experience.episode_labels = self.experience.episode_labels[:entry_num]
                    self.experience.curr_episode = entry_num-1
                    if self.learn_from_iteration is not "RANDOM":
                        self.policy.set_params(self.experience.policy_history[self.learn_from_iteration-1])
                        self.experience.policy_history = self.experience.policy_history[:self.learn_from_iteration]
                        self.learning_iteration = self.learn_from_iteration + 1
                    else: 
                        self.policy.set_default_parameters()
                        self.experience.policy_history = []
        '''

    def save(self, output_folder=None,output_filename=None):
        # save learner state
        super(EpisodicLearner,self).save(output_folder,output_filename)
        
        policy_filename = None
        if output_filename is not None:
            policy_filename = output_filename + '_policy'
        self.policy.save(output_folder,policy_filename)
        if hasattr(self.experience, 'policy_history'):
            self.experience.policy_history.append(self.policy.get_params(symbolic=False))
            
        experience_filename = None
        if output_filename is not None:
            experience_filename = output_filename + '_experience'
            
        self.experience.save(output_folder,experience_filename)

    def get_snapshot_content_paths(self, output_folder=None):
		output_folder = utils.get_output_dir() if output_folder is None else output_folder
		content_paths = [os.path.join(output_folder,self.policy.filename+'.zip'),
						 os.path.join(output_folder,self.experience.filename+'.zip'),
						 os.path.join(output_folder,self.filename+'.zip')]
		return content_paths

    def save_snapshot(self, output_folder=None, output_prefix='snapshot'):
		output_folder = utils.get_output_dir() if output_folder is None else output_folder
		snapshot_header = os.path.join(output_folder,output_prefix)
		content_paths = self.get_snapshot_content_paths(output_folder)
		utils.save_snapshot_zip(snapshot_header,content_paths)

    def stop(self):
        ''' Stops the plant, the visualization (if available) and saves the state of the learner'''
        self.plant.stop()
        if self.viz is not None:
            self.viz.stop()

    def init_cost(self):
        if not self.evaluate_cost:
            if self.cost:
                utils.print_with_stamp('Compiling cost function',self.name)
                mx = theano.tensor.vector('mx')
                Sx = theano.tensor.matrix('Sx')
                self.evaluate_cost = theano.function((mx,Sx),self.cost(mx,Sx), allow_input_downcast=True)
            else:
                utils.print_with_stamp('No cost function provided',self.name)
    
    def set_cost(self, new_cost_func, new_cost_params):
        ''' Replaces the old cost function with a new one (and recompiles it)'''
        self.cost = partial(new_cost_func, params=new_cost_params)
        self.evaluate_cost = None
        self.init_cost()

    def set_experience(self, new_experience):
        #get name form the experience 
        experience_filename = self.experience.filename
        self.experience = new_experience
        self.experience.filename = experience_filename

    def apply_controller(self,H=None,random_controls=False):
        '''
        Starts the plant and applies the current policy to the plant for a duration specified by H (in seconds). If  H is not set, it will run for self.H seconds. If the random_controls paramter is set to True, the current policy is ignored and random controls between [-self.policy.maxU, self.policy.maxU ] will be sent 
        to the plant

        @param H Horizon for applying controller (in seconds)
        @param random_controls Boolean flag that specifies whether to use the current policy or apply random controls
        '''
        if random_controls:
            policy = RandPolicy(self.policy.maxU)
        else:
            policy = self.policy

        #initialize cost if neeeded
        self.init_cost()

        # initialize policy if needed
        p = self.policy.get_params()
        for pi in p:
            if pi is None or pi.size == 0:
                self.policy.set_default_parameters()
                break

        # mark the start of the episode
        self.experience.new_episode(random = random_controls, learning_iteration = self.learning_iteration)

        # start robot
        utils.print_with_stamp('Starting data collection run',self.name)
        if H is  None:
            H = self.H
        utils.print_with_stamp('Running for %f seconds'%(H),self.name)

        if self.async_plant:
            self.plant.start()
        if self.viz is not None:
            self.viz.start()

        exec_time = time.time()
        x_t, t0 = self.plant.get_state()
        Sx_t = np.zeros((x_t.shape[0],x_t.shape[0]))
        L_noise = np.linalg.cholesky(self.plant.noise)
        if self.plant.noise is not None:
            # randomize state
            x_t = x_t + np.random.randn(x_t.shape[0]).dot(L_noise);
        t = t0
        
        H_steps = int(np.ceil(H/self.plant.dt))
        # do rollout
        #while t <= t0 + H:
        for i in xrange(H_steps):
            # convert input angle dimensions to complex representation
            x_t_ = utils.gTrig_np(x_t[None,:], self.angle_idims).flatten()
            #  get command from policy (this should be fast, or at least account for delays in processing):
            u_t = policy.evaluate(x_t_)[0].flatten()
            #  send command to robot:
            self.plant.apply_control(u_t)
            if self.evaluate_cost is not None:
                #  get cost:
                c_t = self.evaluate_cost(x_t, Sx_t)
                # append to experience dataset
                self.experience.add_sample(t,x_t,u_t,c_t)
                #print t,x_t,u_t,c_t[0]
            else:
                # append to experience dataset
                self.experience.append(t,x_t,u_t,0)

            # step the plant if necessary
            if not self.async_plant:
                self.plant.step()

            # sleep to match the desired sample rate
            if self.realtime:
                exec_time = time.time() - exec_time
                if exec_time < self.plant.dt:
                    time.sleep(self.plant.dt-exec_time)

            #  get robot state (this should ensure synchronicity by blocking until dt seconds have passed):
            exec_time = time.time()
            x_t, t = self.plant.get_state()
            if self.plant.noise is not None:
                # randomize state
                x_t = x_t + np.random.randn(x_t.shape[0]).dot(L_noise);
            
            if self.plant.done:
                break
        # add last state to experience
        x_t_ = utils.gTrig_np(x_t[None,:], self.angle_idims).flatten()
        u_t = np.zeros_like(u_t)
        if self.evaluate_cost is not None:
            c_t = self.evaluate_cost(x_t, Sx_t)
            self.experience.add_sample(t,x_t,u_t,c_t)
            #print t,x_t,u_t,c_t[0]
        else:
            self.experience.append(t,x_t,u_t,0)

        # stop robot
        run_value = np.array(self.experience.immediate_cost[-1][:-1])
        utils.print_with_stamp('Done. Stopping robot. Value of run [%f]'%(run_value.sum()),self.name)

        self.plant.stop()
        if self.viz is not None:
            self.viz.stop()
        self.n_episodes += 1
        return self.experience

    def train_policy(self, H=None):
        if H is not None:
            self.H = H
        # optimize value wrt to the policy parameters
        self.learning_iteration+=1
        self.n_evals=0
        utils.print_with_stamp('Training policy parameters [Iteration %d]'%(self.learning_iteration), self.name)

        v0 = self.value()
        utils.print_with_stamp('Initial value estimate [%f]'%(v0),self.name) 
        p0 = self.policy.get_params(symbolic=False)
        self.best_p = [v0,p0]
        min_method = self.min_method.upper()
        # deterministic gradients
        if min_method in DETERMINISTIC_MIN_METHODS:
            parameter_shapes = [p.shape for p in p0]
            m_loss = utils.MemoizeJac(self.loss)
        
            # setup alternative minimization methods (in case the one selected fails)
            successful = None
            min_methods = [min_method]
            for i in xrange(len(DETERMINISTIC_MIN_METHODS)): 
                if DETERMINISTIC_MIN_METHODS[i] not in min_methods: 
                    min_methods.append(DETERMINISTIC_MIN_METHODS[i])

            # keep on trying to optimize with all the methods, until one succeds, or we go through all of them
            for i in range(len(min_methods)):
                try:
                    utils.print_with_stamp("Using %s optimizer"%(min_methods[i]),self.name)
                    opt_res = minimize(m_loss, utils.wrap_params(p0),
                                       jac=m_loss.derivative, 
                                       args=parameter_shapes, 
                                       method=min_methods[i], 
                                       tol=self.conv_thr, 
                                       options={'maxiter': self.max_evals})
                    # break the loop since we succeeded
                    self.policy.set_params(utils.unwrap_params(opt_res.x,parameter_shapes))
                    break
                except ValueError:
                    utils.print_with_stamp("Optimization using %s failed"%(min_methods[i]),self.name)
                    v0,p0 = self.best_p
                    self.policy.set_params(p0)

        # stochastic gradients
        elif min_method in STOCHASTIC_MIN_METHODS.keys():
            utils.print_with_stamp("Using %s optimizer"%(min_method),self.name)
            # compile optimizer f not available
            if not hasattr(self,'train_fn'):
                # get the value as a symbolic expression
                v,updts = self.get_policy_value()
                # get the updates using the desired minimization method
                utils.print_with_stamp("Compiling optimizer",self.name)
                min_method_updt = STOCHASTIC_MIN_METHODS[min_method]
                p = self.policy.get_params(symbolic=True)
                updates = min_method_updt(v,p,learning_rate=self.learning_rate)
                updates += updts
                self.train_fn = theano.function([],v,updates=updates)
                utils.print_with_stamp("Done compiling.",self.name)

            # training loop   
            for i in xrange(self.max_evals):
                # evaluate current policy and update parameters
                v = self.train_fn()
                p = self.policy.get_params(symbolic=False)
                if v<self.best_p[0]:
                    self.best_p = [v,p]
                self.n_evals+=1
                utils.print_with_stamp('Current value: %s, Total evaluations: %d    '%(str(v),self.n_evals),
                                        self.name,True)
        else:
            error_str = 'Unknown minimization method %s' % (self.min_method)
            utils.print_with_stamp(error_str,self.name)
            raise ValueError(error_str)
        
        print '' 

        v,p = self.best_p
        self.policy.set_params(p)
        utils.print_with_stamp('Done training. New value [%f]'%(v),self.name)
        self.state_changed = True

    def loss(self, policy_parameters, parameter_shapes):
        p = utils.unwrap_params(policy_parameters, parameter_shapes)
        # set policy parameters
        self.policy.set_params( p )

        # compute value + derivatives
        v,dv = self.value(True)
        dv = utils.wrap_params(dv)
        v,dv = (np.array(v).astype(np.float64),np.array(dv).astype(np.float64))
        if v<self.best_p[0]:
            self.best_p = [v,p]

        self.n_evals+=1
        utils.print_with_stamp('Current value: %s, Total evaluations: %d    '%(str(v),self.n_evals),self.name,True)
        return v,dv


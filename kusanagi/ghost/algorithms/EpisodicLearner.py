import lasagne
import numpy as np
import os
import theano
import theano.tensor as tt
import time

from functools import partial

from kusanagi import utils
from kusanagi.ghost.algorithms.ExperienceDataset import ExperienceDataset
from kusanagi.ghost.control import RandPolicy
from kusanagi.base.Loadable import Loadable
from kusanagi.utils import probls

from scipy.optimize import minimize, basinhopping
from kusanagi import utils
DETERMINISTIC_MIN_METHODS = ['L-BFGS-B', 'TNC', 'BFGS', 'SLSQP', 'CG']
STOCHASTIC_MIN_METHODS = {'SGD': lasagne.updates.sgd,
                          'MOMENTUM': lasagne.updates.momentum,
                          'NESTEROV': lasagne.updates.nesterov_momentum,
                          'NESTEROV_MOMENTUM': lasagne.updates.nesterov_momentum,
                          'ADAGRAD': lasagne.updates.adagrad,
                          'RMSPROP': lasagne.updates.rmsprop,
                          'ADADELTA': lasagne.updates.adadelta,
                          'ADAM': lasagne.updates.adam,
                          'NADAM': utils.updates.nadam,
                          'ADAMAX': lasagne.updates.adamax,
                         }

# TODO remove any theano dependency from here
class EpisodicLearner(Loadable):
    def __init__(self, params, plant_class, policy_class, cost_func=None,
                 experience=None, async_plant=False, name='EpisodicLearner', filename_prefix=None):
        self.name = name
        # initialize plant
        if 'x0' not in params['plant']:
            params['plant']['x0'] = params['x0']
            params['plant']['S0'] = params['S0']
        self.plant = plant_class(**params['plant'])
        # initialize policy
        params['policy']['angle_dims'] = params['angle_dims']
        self.policy = policy_class(**params['policy'])
        # set filename
        self.filename = self.name if filename_prefix is None else filename_prefix
        self.filename += '_'+self.plant.name+'_'+self.policy.name
        # initialize cost
        params['cost']['angle_dims'] = params['angle_dims']
        self.cost = partial(cost_func, params=params['cost']) if cost_func is not None else None
        self.evaluate_cost = None
        # initialize experience dataset
        if experience:
            self.experience = experience
        else:
            self.experience = ExperienceDataset(filename_prefix=self.filename)
        # initialize learner state variables
        self.n_episodes = 0
        self.angle_idims = params['angle_dims'] if 'angle_dims' in params else []
        self.H = params['H'] if 'H' in params else 10.0
        self.dt = params['plant']['dt']
        self.discount = params['discount'] if 'discount' in params else 1.0
        self.max_evals = params['max_evals'] if 'max_evals' in params else 150
        self.grad_clip = params['clip'] if 'clip' in params else 10.0
        self.conv_thr = params['conv_thr'] if 'conv_thr' in params else 1e-12
        self.learning_rate = params['learning_rate'] if 'learning_rate' in params else 1e-3
        self.min_method = params['min_method'] if 'min_method' in params else "L-BFGS-B"
        self.random_walk = params['random_walk'] if 'random_walk' in params else False

        self.async_plant = async_plant
        self.learning_iteration = 0
        self.n_evals = 0
        self.realtime = params['realtime'] if 'realtime' in params else True

        # register attributes to save
        Loadable.__init__(self, name=name, filename=self.filename)
        self.register(['n_episodes',
                       'angle_idims',
                       'async_plant',
                       'cost',
                       'evaluate_cost',
                       'H',
                       'dt',
                       'discount',
                       'learning_iteration',
                       'n_evals'])

        # try loading from file, initialize from scratch otherwise
        utils.print_with_stamp('Initialising new %s learner'%(self.name), self.name)
        self.state_changed = True

    def load(self, output_folder=None, output_filename=None):
        # load learner state
        super(EpisodicLearner, self).load(output_folder, output_filename)
        self.plant.dt = self.dt
        utils.print_with_stamp('Cost parameters: %s'%(self.cost.keywords['params']), self.name)

        # load policy and experience separately
        policy_filename = None
        if output_filename is not None:
            policy_filename = output_filename + '_policy'
        self.policy.load(output_folder, policy_filename)

        experience_filename = None
        if output_filename is not None:
            experience_filename = output_filename + '_experience'
        self.experience.load(output_folder, experience_filename)

        #initialize cost if neeeded
        self.init_cost()

        # initialize policy if needed
        p = self.policy.get_params()
        if len(p) == 0:
            self.policy.init_params()

    def save(self, output_folder=None, output_filename=None):
        #initialize cost if neeeded
        self.init_cost()

        # initialize policy if needed
        p = self.policy.get_params()
        if len(p) == 0:
            self.policy.init_params()

        policy_filename = None
        if output_filename is not None:
            policy_filename = output_filename + '_policy'
        self.policy.save(output_folder, policy_filename)

        experience_filename = None
        if output_filename is not None:
            experience_filename = output_filename + '_experience'

        self.experience.save(output_folder, experience_filename)

        # save learner state
        self.dt = self.plant.dt
        super(EpisodicLearner, self).save(output_folder, output_filename)

    def load_snapshot(self, zip_filepath, output_folder=None, extract_folder=None):
        output_folder = utils.get_output_dir() if output_folder is None else output_folder
        if extract_folder is None:
            extract_folder = output_folder
        utils.unzip_snapshot(os.path.join(output_folder, zip_filepath), extract_folder)

    def get_snapshot_content_paths(self, output_folder=None):
        output_folder = utils.get_output_dir() if output_folder is None else output_folder
        content_paths = [os.path.join(output_folder, self.policy.filename+'.zip'),
                         os.path.join(output_folder, self.experience.filename+'.zip'),
                         os.path.join(output_folder, self.filename+'.zip')]
        return content_paths

    def save_snapshot(self, output_folder=None, output_prefix='snapshot', with_timestamp=False):
        output_folder = utils.get_output_dir() if output_folder is None else output_folder
        snapshot_header = os.path.join(output_folder, output_prefix)
        content_paths = self.get_snapshot_content_paths(output_folder)
        utils.save_snapshot_zip(snapshot_header, content_paths, with_timestamp)

    def init_cost(self):
        if not self.evaluate_cost:
            if self.cost:
                utils.print_with_stamp('Compiling cost function', self.name)
                utils.print_with_stamp('Cost parameters: %s'%(self.cost.keywords['params']),
                                       self.name)
                mx = tt.vector('mx')
                self.evaluate_cost = theano.function([mx], self.cost(mx, None)[0],
                                                     allow_input_downcast=True,
                                                     on_unused_input='ignore')
            else:
                utils.print_with_stamp('No cost function provided', self.name)

    def set_cost(self, new_cost_func, new_cost_params):
        ''' Replaces the old cost function with a new one (and recompiles it)'''
        if 'angle_dims' not in new_cost_params:
            new_cost_params['angle_dims'] = self.angle_idims
        if self.cost is not None:
            if self.cost.func == new_cost_func and self.cost.keywords['params'] == new_cost_params:
                # do nothing, as the current cost has the same parameters
                return
        self.cost = partial(new_cost_func, params=new_cost_params)
        self.evaluate_cost = None
        self.init_cost()

    def set_experience(self, new_experience):
        # get state from the new experience
        self.experience.set_state(new_experience.get_state())

    def set_episode(self, episode=-1):
        if episode > 0:
            utils.print_with_stamp('Setting current episode to %d'%(episode), self.name)
            if episode == 0:
                self.experience.reset()
                self.policy.init_params()
            elif episode <= self.experience.current_episode:
                self.experience.truncate(episode)
                self.policy.set_params(self.experience_dataset.policy_parameters[episode])

    def apply_controller(self, H=None, random_controls=False, callback=None):
        '''
        Starts the plant and applies the current policy to the plant for a duration specified by H
        (in seconds). If  H is not set, it will run for self.H seconds. If the random_controls
        parameter is set to True, the current policy is ignored and random controls between
        [-self.policy.maxU, self.policy.maxU ] will be sent to the plant

        @param H Horizon for applying controller (in seconds)
        @param random_controls Boolean flag that specifies whether to use the current policy or
               to apply random controls
        '''

        #initialize cost if neeeded
        self.init_cost()

        if random_controls:
            utils.print_with_stamp('Applying controls uniformly at random', self.name)
            policy = RandPolicy(self.policy.maxU, self.random_walk)
            p = []
        else:
            policy = self.policy
            # initialize policy if needed
            p = self.policy.get_params()
            if len(p) == 0:
                self.policy.init_params()

        # mark the start of the episode
        self.experience.new_episode(policy_params=p)

        # start robot
        utils.print_with_stamp('Starting data collection run', self.name)
        if H is  None:
            H = self.H
        utils.print_with_stamp('Running for %f seconds'%(H), self.name)

        exec_time = time.time()
        x_t = self.plant.reset()

        H_steps = int(np.ceil(H/self.plant.dt))

        # do rollout
        for t in range(H_steps):
            # convert input angle dimensions to complex representation
            x_t_ = utils.gTrig_np(x_t[None, :], self.angle_idims).flatten()
            #  get command from policy
            # (this should be fast, or at least account for delays in processing):
            u_t = policy.evaluate(x_t_, t)[0].flatten()

            # apply control and step the plant
            x_t, c_t, done, info = self.plant.step(u_t)
            info['done'] = done

            # append to experience dataset
            self.experience.add_sample(t, x_t, u_t, c_t, info)

            if callable(callback):
                callback(x_t, c_t, done, info, self)

            if self.plant.done:
                break

        # stop robot
        run_value = np.array(self.experience.costs[-1][:-1])
        utils.print_with_stamp('Done. Stopping robot. Value of run [%f]'%(run_value.sum()),
                               self.name)
        if hasattr(self.plant, 'stop'):
            self.plant.stop()

        self.n_episodes = len(self.experience.states)
        return self.experience

    def train_policy(self, H=None):
        if H is not None:
            self.H = H

        # optimize value wrt to the policy parameters
        self.learning_iteration+=1
        self.iter_time = 0
        self.start_time = time.time()
        self.n_evals = 0
        utils.print_with_stamp('Training policy parameters [Iteration %d]'%\
        (self.learning_iteration), self.name)

        v0 = self.value()
        utils.print_with_stamp('Initial value estimate [%f]'%(v0), self.name) 
        p0 = self.policy.get_params(symbolic=False)
        self.best_p = [v0,p0,0]
        if not hasattr(self, 'average_p'):
            self.p_avg = [np.zeros_like(p) for p in p0]

        parameter_shapes = [p.shape for p in p0]
        m_loss = utils.MemoizeJac(self.loss,args=(parameter_shapes,))
        min_method = self.min_method.upper()
        # deterministic gradients
        if min_method in DETERMINISTIC_MIN_METHODS:
            parameter_shapes = [p.shape for p in p0]
            m_loss = utils.MemoizeJac(self.loss)

            # setup alternative minimization methods (in case the one selected fails)
            min_methods = [min_method]
            for i in range(len(DETERMINISTIC_MIN_METHODS)):
                if DETERMINISTIC_MIN_METHODS[i] not in min_methods:
                    min_methods.append(DETERMINISTIC_MIN_METHODS[i])

            # keep on trying to optimize with all the methods, until one succeds, or we go through
            # all of them
            for i in range(len(min_methods)):
                try:
                    utils.print_with_stamp("Using %s optimizer"%(min_methods[i]), self.name)
                    opt_res = minimize(m_loss, utils.wrap_params(p0),
                                       jac=m_loss.derivative,
                                       args=parameter_shapes,
                                       method=min_methods[i],
                                       tol=self.conv_thr,
                                       options={'maxiter': self.max_evals,
                                                'maxfun': self.max_evals,
                                                'maxcor': 100,
                                                'maxls': 30,
                                                'ftol': 1e6*np.finfo(float).eps,
                                                'gtol': 1.0e-6}
                                      )
                    # break the loop since we succeeded
                    self.policy.set_params(utils.unwrap_params(opt_res.x, parameter_shapes))
                    break
                except ValueError:
                    print('')
                    utils.print_with_stamp("Optimization with %s failed"%(min_methods[i]), self.name)
                    v0,p0 = self.best_p
                    self.policy.set_params(p0)

        # stochastic gradients
        elif min_method in list(STOCHASTIC_MIN_METHODS.keys()):
            utils.print_with_stamp("Using %s optimizer"%(min_method), self.name)
            # compile optimizer f not available
            if not (hasattr(self,'train_fn') and self.train_fn):
                # get the value as a symbolic expression
                v,updts = self.get_policy_value()
                # get the updates using the desired minimization method
                utils.print_with_stamp("Compiling optimizer", self.name)
                min_method_updt = STOCHASTIC_MIN_METHODS[min_method]
                p = self.policy.get_params(symbolic=True)
                reg=0
                if hasattr(self.policy, 'get_regularization_term'):
                    # get regularization term
                    reg += self.policy.get_regularization_term(1e-2,1e-2)
                dJdp = self.get_policy_gradients(v+reg,p,clip=self.grad_clip)
                lr = theano.tensor.scalar('lr')
                updates = min_method_updt(dJdp,p,learning_rate=lr)
                updates += updts
                # add temporal smoothing
                self.train_fn = theano.function([lr],[v]+dJdp,updates=updates)
                utils.print_with_stamp("Done compiling.", self.name)

            # training loop
            for i in range(self.max_evals):
                # evaluate current policy and update parameters
                if hasattr(self,'update') and self.update:
                    # if there are any variable we want to update before evaluating the gradient
                    self.update()
                p = self.policy.get_params(symbolic=False)
                lr = self.learning_rate#*(1.0/(self.learning_iteration)**0.1)
                ret = self.train_fn(lr)     # v corresponds to the parameters before the training update
                v = ret[0]; dJdp=ret[1:4];# jac = ret[4:]
                if v < self.best_p[0]:
                    self.best_p = [v,p,i]
                self.n_evals+=1
                gmag = [np.sqrt((p**2).sum()) for p in dJdp]
                gmax = [p.max() for p in dJdp]
                out_str = 'Current value: %E, Total evaluations: %d, gm: %s, lr: %f'
                utils.print_with_stamp(out_str%(v, self.n_evals, gmag, lr), self.name,True)
        # probabilistic line searches
        elif min_method == 'PROBLS':
            if not hasattr(self,'loss_fn'):
                from kusanagi.utils import probls
                costs, updts = self.get_policy_value(cost_per_sample=True)
                losses = costs[:,1:].sum(1)
                params = self.policy.get_params(symbolic=True)
                
                compiled_loss_fn = probls.compile_loss_fn(losses, params, updates=updts)
                # TODO we should need to pas the parameters around
                def loss_fn(p):
                    p = utils.unwrap_params(p, parameter_shapes)
                    self.policy.set_params(p)
                    return compiled_loss_fn()
                self.loss_fn = loss_fn
            
            # minimize via SGD with probabilistic line search
            probls.minimize(self.loss_fn,utils.wrap_params(p0))

        else:
            error_str = 'Unknown minimization method %s' % (self.min_method)
            utils.print_with_stamp(error_str, self.name)
            raise ValueError(error_str)
     
        print('') 
        v,p,i = self.best_p
        #beta=0.5
        #p = [ (1-beta)*p[i] + beta*p0[i] for i in range(len(p)) ]
        self.policy.set_params(p)
        v = self.value()
        utils.print_with_stamp('Done training. New value [%f] iter: [%d]'%(v, i), self.name)
        self.state_changed = True

    def loss(self, policy_parameters, parameter_shapes):
        p = utils.unwrap_params(policy_parameters, parameter_shapes)
        # set policy parameters
        self.policy.set_params(p)

        # compute value + derivatives
        v,dv = self.value(True)
        dv = utils.wrap_params(dv)
        v,dv = (np.array(v).astype(np.float64),np.array(dv).astype(np.float64))
        self.n_evals+=1
        
        if v<self.best_p[0]:
            self.best_p = [v, p, self.n_evals]

        end_time = time.time()
        self.iter_time = ((end_time - self.start_time) - self.iter_time)/self.n_evals + self.iter_time
        
        utils.print_with_stamp('Current value: %s, Total evaluations: %d, \
        Avg. time per call: %f   '%(str(v),self.n_evals,self.iter_time),self.name,True)
        self.start_time = time.time()
        return v,dv


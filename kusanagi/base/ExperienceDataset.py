import theano
import numpy as np
from kusanagi import utils
from kusanagi.base.Loadable import Loadable


class ExperienceDataset(Loadable):
    ''' Class used to store data from runs with a learning agent'''
    def __init__(self, name='Experience', filename_prefix=None, filename=None):
        self.name = name
        self.time_stamps = []
        self.states = []
        self.actions = []
        self.costs = []
        self.info = []
        self.policy_parameters = []
        self.curr_episode = -1
        self.state_changed = True
        if filename is not None:
            self.filename = filename
        else:
            self.filename = (self.name+'_dataset'
                             if filename_prefix is None
                             else filename_prefix+'_dataset')
            utils.print_with_stamp(
                'Initialising new experience dataset', self.name)

        Loadable.__init__(self, name=name, filename=self.filename)

        # if a filename was passed, try loading it
        if filename is not None:
            self.load()

        self.register_types([list])
        self.register(['curr_episode'])

    def load(self, output_folder=None, output_filename=None):
        ''' Loads the state from file, and initializes additional variables'''
        # load state
        ret = super(ExperienceDataset, self).load(output_folder, output_filename)

        # if the policy parameters were saved as shared variables
        for i in range(len(self.policy_parameters)):
            pi = self.policy_parameters[i]
            for j in range(len(pi)):
                pij = self.policy_parameters[i][j]
                if isinstance(pij, theano.tensor.sharedvar.SharedVariable):
                    self.policy_parameters[i][j] = pij.get_value()
        return ret

    def add_sample(self, x_t=None, u_t=None, c_t=None, info=None, t=None):
        '''
            Adds new set of observations to the current episode
        '''
        curr_episode = self.curr_episode
        if curr_episode < 0:
            self.new_episode()
        self.states[curr_episode].append(x_t)
        self.actions[curr_episode].append(u_t)
        self.costs[curr_episode].append(c_t)
        self.info[curr_episode].append(info)
        self.time_stamps[curr_episode].append(t)
        self.state_changed = True

    def new_episode(self, policy_params=None):
        '''
            Adds new episode to the experience dataset
        '''
        self.time_stamps.append([])
        self.states.append([])
        self.actions.append([])
        self.costs.append([])
        self.info.append([])
        if policy_params:
            self.policy_parameters.append(policy_params)
        else:
            self.policy_parameters.append([])

        self.curr_episode += 1
        self.state_changed = True

    def append_episode(self, states, actions, costs,
                        infos=None, policy_params=None, ts=None):
        if policy_params is not None:
            self.policy_parameters.append(policy_params)
        if infos is not None:
            self.info.append(infos)
        if ts is not None:
            self.time_stamps.append(ts)
        self.states.append(states)
        self.actions.append(actions)
        self.costs.append(costs)

    def n_samples(self):
        ''' Returns the total number of samples in this dataset '''
        return sum([len(s) for s in self.states])

    def n_episodes(self):
        ''' Returns the total number of episodes in this dataset '''
        return len(self.states)

    def reset(self):
        ''' Empties the internal data structures'''
        fmt = 'Resetting experience dataset'
        fmt += '(WARNING: data from %s will be overwritten)'
        utils.print_with_stamp(fmt % (self.filename), self.name)
        self.time_stamps = []
        self.states = []
        self.actions = []
        self.costs = []
        self.info = []
        self.policy_parameters = []
        self.curr_episode = -1
        # Let's give people a last chance of recovering their data. Also, we
        # don't want to save an empty experience dataset
        self.state_changed = False

    def truncate(self, episode):
        ''' Resets the experience to start from the given episode number'''
        if episode <= self.curr_episode and episode > 0:
            # Let's give people a last chance of recovering their data. Also,
            # we don't want to save an empty experience dataset
            fmt = 'Resetting experience dataset to episode %d'
            fmt += ' (WARNING: data from %s will be overwritten)'
            utils.print_with_stamp(fmt % (episode, self.filename), self.name)
            self.curr_episode = episode
            self.time_stamps = self.time_stamps[:episode]
            self.states = self.states[:episode]
            self.actions = self.actions[:episode]
            self.costs = self.costs[:episode]
            self.info = self.info[:episode]
            self.policy_parameters = self.policy_parameters[:episode]
            self.state_changed = True

    def get_dynmodel_dataset(self, deltas=True, filter_episodes=None,
                             angle_dims=None, x_steps=1,
                             u_steps=1, output_steps=1, return_costs=False,
                             stack=False):
        '''
        Returns a dataset where the inputs are state_actions and the outputs
        are next steps.
        Parameters:
        -----------
        deltas: wheter to return changes in state
                (x_t - x_{t-1}, x_{t-1} - x_{t-2}, ...)
                or future states
                (x_t, x_{t-1}, x_{t-2}, ...), in the output
        filter_episodes: list containing  episode indices to extract from
                         which to extract data.
                         if list empty or undefined ( equal to None ),
                         extracts data from all episodes
        angle_dims: indices of input state dimensions to linearize, by
                    converting to complex
                    representation \theta => (sin(\theta), cos(\theta))
        x_steps: how many steps in the past to concatenate as input
        u_steps: how many steps in the past to concatenate as input
        output_steps: how many steps in the future to concatenate as output
        return_costs: whether to append the cost feedback to the output
        stack: whether to stack or concatenate the multi step data
        Returns:
        --------
        X: if stack is False X is a numpy array of shape
           [n, x_steps*D + u_steps*U], where n is the number of data samples,
           D the input state dimensions.abs if stack is True, the shape of X
           is [n, x_steps, D + U]
        '''
        filter_episodes = filter_episodes or []
        angle_dims = angle_dims or []
        inputs, targets = [], []
        join = np.stack if stack else np.concatenate
        if stack:
            # ignore the u_steps parameter
            u_steps = x_steps
            # output steps
            output_steps = x_steps + output_steps - 1

        if not isinstance(filter_episodes, list):
            filter_episodes = [filter_episodes]
        if len(filter_episodes) < 1:
            # use all data
            filter_episodes = list(range(self.n_episodes()))
        for epi in filter_episodes:
            if len(self.states[epi]) == 0:
                continue
            # get state action pairs for current episode
            states, actions = np.array(
                self.states[epi]), np.array(self.actions[epi])
            # convert input angle dimensions to complex representation
            states_ = utils.gTrig_np(np.array(states), angle_dims)
            # pad with initial state for the first x_steps timesteps
            states_ = np.concatenate([states_[[0]*(x_steps-1)], states_])
            # get input states up to x_steps in the past.

            states_ = join(
                [states_[i:i-x_steps-(output_steps-1), :]
                 for i in range(x_steps)],
                axis=1)
            # same for actions (u_steps in the past, pad with zeros for the
            # first u_steps)
            actions_ = np.concatenate(
                [np.zeros((u_steps-1, actions.shape[1])), actions])
            actions_ = join(
                [actions_[i:i-u_steps-(output_steps-1), :]
                 for i in range(u_steps)],
                axis=1)

            # create input vector
            inp = np.concatenate([states_, actions_], axis=-1)

            # get output states up to output_steps in the future
            H = states.shape[0]
            ostates = join(
                [states[i:H-(output_steps-i-1), :]
                 for i in range(output_steps)],
                axis=1)

            #  create output vector
            tgt = (ostates[1:, :] - ostates[:-1, :]
                   if deltas else ostates[1:, :])

            # append costs if requested
            if return_costs:
                costs = np.array(self.costs[epi])
                ocosts = join(
                    [costs[i:H-(output_steps-i-1), :]
                     for i in range(output_steps)],
                    axis=1)

                tgt = np.concatenate([tgt, ocosts[1:, :]], axis=-1)

            inputs.append(inp)
            targets.append(tgt)

        ret = np.concatenate(inputs), np.concatenate(targets)
        return ret

        def sample_initial_state(self, n_samples=1):
            # collect initial states
            x0 = [ep[0] for ep in self.states]
            # sample indices
            idx = np.random.choice(range(len(x0)), n_samples)
            return np.array(x0)[idx]

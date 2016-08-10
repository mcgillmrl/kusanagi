import os,sys
import utils
from base.Loadable import Loadable

class ExperienceDataset(Loadable):
    ''' Class used to store data from runs with a learning agent'''
    def __init__(self, name='Experience', filename_prefix=None, filename=None):
        self.name = name
        if filename is not None:
            self.filename=filename
        else:
            self.filename = self.name+'_dataset' if filename_prefix is None else filename_prefix+'_dataset'

            utils.print_with_stamp('Initialising new experience dataset',self.name)
            self.time_stamps = []
            self.states = []
            self.actions = []
            self.immediate_cost = []
            self.policy_history = []
            self.episode_labels = []
            self.curr_episode = -1
            self.state_changed = True
        
        Loadable.__init__(self,name=name,filename=self.filename)

        # if a filename was passed, try loading it
        if filename is not None:
            self.load()

        self.register_types([list])
        self.register(['curr_episode'])

    def add_sample(self,t,x_t=None,u_t=None,c_t=None):
        curr_episode = self.curr_episode
        self.time_stamps[curr_episode].append(t)
        self.states[curr_episode].append(x_t)
        self.actions[curr_episode].append(u_t)
        self.immediate_cost[curr_episode].append(c_t)
        self.state_changed = True

    def add_episode(self, state):
        i = utils.integer_generator()
        self.time_stamps.append(state[i.next()])
        self.states.append(state[i.next()])
        self.actions.append(state[i.next()])
        self.immediate_cost.append(state[i.next()])
        self.curr_episode += 1
        try:
            self.policy_history.append(state[i.next()])
            self.episode_labels.append(state[i.next()])
        except IndexError:
            pass

    def new_episode(self, random = False, learning_iteration = -1):
        self.time_stamps.append([])
        self.states.append([])
        self.actions.append([])
        self.immediate_cost.append([])
        self.curr_episode += 1
        self.state_changed = True
        try:
            if random:
                self.episode_labels.append("RANDOM")
            else:
                self.episode_labels.append(learning_iteration)
        except AttributeError:
            pass

    def n_samples(self):
        ''' Returns the total number of samples in this dataset '''
        return sum([len(s) for s in self.states])

    def reset(self):
        utils.print_with_stamp('Resetting experience dataset',self.name)
        self.time_stamps = []
        self.states = []
        self.actions = []
        self.immediate_cost = []
        self.curr_episode = -1
        self.state_changed = False
        self.policy_history = []
        self.episode_labels = []

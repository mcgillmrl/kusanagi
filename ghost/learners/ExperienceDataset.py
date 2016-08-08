import os,sys
import utils
from theano.misc.pkl_utils import dump as t_dump, load as t_load
class ExperienceDataset(object):
    ''' Class used to store data from runs with a learning agent'''
    def __init__(self, name='Experience', filename_prefix=None, filename=None):
        if filename is not None:
            self.name = name
            self.filename=filename
            self.load()
        else:
            self.name = name
            self.filename = self.name+'_dataset' if filename_prefix is None else filename_prefix+'_dataset'

            utils.print_with_stamp('Initialising new experience dataset',self.name)
            self.time_stamps = []
            self.states = []
            self.actions = []
            self.immediate_cost = []
            self.curr_episode = -1
            self.state_changed = False
            self.policy_history = []
            self.episode_labels = []

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

    def load(self, output_folder=None,output_filename=None):
        output_folder = utils.get_output_dir() if output_folder is None else output_folder
        [output_filename, self.filename] = utils.sync_output_filename(output_filename, self.filename, '.zip')
        path = os.path.join(output_folder,output_filename)
        with open(path,'rb') as f:
            utils.print_with_stamp('Loading experience dataset from %s.zip'%(self.filename),self.name)
            state = t_load(f)
            self.set_state(state)
        self.state_changed = False
    
    def save(self, output_folder=None,output_filename=None):
        sys.setrecursionlimit(100000)
        if self.state_changed or output_folder is not None or output_filename is not None:
            output_folder = utils.get_output_dir() if output_folder is None else output_folder
            [output_filename, self.filename] = utils.sync_output_filename(output_filename, self.filename, '.zip')
            path = os.path.join(output_folder,output_filename)
            with open(path,'wb') as f:
                utils.print_with_stamp('Saving experience dataset to %s.zip'%(self.filename),self.name)
                t_dump(self.get_state(),f,2)
            self.state_changed = False

    def set_state(self,state):
        i = utils.integer_generator()
        self.time_stamps = state[i.next()]
        self.states = state[i.next()]
        self.actions = state[i.next()]
        self.immediate_cost = state[i.next()]
        self.curr_episode = state[i.next()]
        try:
            self.policy_history = state[i.next()]
            self.episode_labels = state[i.next()]
        except IndexError:
            pass

    def get_state(self):
        try:
            ret = [self.time_stamps,self.states,self.actions,self.immediate_cost,self.curr_episode, self.policy_history, self.episode_labels]
        except AttributeError:
            ret = [self.time_stamps,self.states,self.actions,self.immediate_cost,self.curr_episode]
        return ret

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

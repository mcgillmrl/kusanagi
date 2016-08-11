import gym, time
from gym.envs.mujoco import mujoco_env
import numpy as np

class Fixer(object):
    action_list = []
    step_list = []

    def __init__(self, env):
        self.env = env
        
    def get_action(self):
        for i_episode in xrange(20):
            observation = self.env.reset()
            for t in xrange(100):
                self.env.render()
                action = self.env.action_space.sample()
                #print action
                self.action_list.append(np.take(action, 0))    
                observation, reward, done, info = self.env.step(action)
                #print observation, reward, done, info
                #time.sleep(0.1)
                if done:
                    #print "Episode finished after {} timesteps".format(t+1)
                    break
        return self.action_list
                   
    def get_step(self):
        for i_episode in xrange(20):
            observation = self.env.reset()
            for t in xrange(100):
                #self.env.render()
                action = self.env.action_space.sample()
                #print action
                observation, reward, done, info = self.env.step(action)
                #print observation, reward, done, info
                self.step_list.append([observation, reward, done, info])
                if done:
                    break
        return self.step_list
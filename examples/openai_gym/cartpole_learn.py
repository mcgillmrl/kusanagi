import signal,sys
import gym
import numpy as np
from functools import partial
from ghost.learners.PILCO import PILCO
from ghost.control import RBFPolicy

if __name__ == '__main__':
    env = gym.make('InvertedPendulum-v0')
    env.reset()
    for _ in xrange(1000):
        env.render()
        a = env.action_space.sample()
        x,r,done,info = env.step(a) # take a random action
        print x,a,r,done,info

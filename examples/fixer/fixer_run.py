import gym, time
from gym.envs.mujoco import mujoco_env
from fixer import Fixer

env = gym.make('InvertedDoublePendulum-v1')

my_list = []
fx = Fixer(env)

#my_list = fx.get_action()
#print my_list

print fx.get_step()
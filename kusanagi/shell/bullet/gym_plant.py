import pybullet as p
import gym
from kusanagi.shell import Plant

class GymPlant(Plant):
    def __init__(self, gym_env, render=True):
        super(GymPlant, self).__init__(self)
        self.gym_env = gym_env
        self.render = render
        self.done = False

    def step(self):
        self.t = self.t + self.dt
        self.x, reward, self.done, info = self.gym_env.step(self.u)
        if self.render:
            self.gym_env.render()
        return self.x

    def reset_state(self):
        self.done = False
        self.gym_env.reset()


from abc import ABC

import gym
from gym.spaces import Discrete, Box
import numpy as np
from WindAI.floris.optimize_AI import plotfarm


class FarmEnv(gym.Env):

    def __init__(self, config):
        self.numwt = config["num_wind_turbines"]
        self.farm = config["farm"]
        self.best_power = config["initial_power"]
        self.count_steps = 0
        self.cur_yaws = np.full((self.numwt,), 0, dtype=np.int32)
        # action space is the yaw angles for the wt between -10° and 10°
        self.action_space = Box(low=np.full((self.numwt,), -10, dtype=np.int32), high=np.full((self.numwt,), 10, dtype=np.int32), shape=(self.numwt,), dtype=np.int32)

        # observation space TODO
        self.observation_space = Box(low=np.full((self.numwt,), -10, dtype=np.int32), high=np.full((self.numwt,), 10, dtype=np.int32), shape=(self.numwt,), dtype=np.int32)

    def reset(self):
        self.cur_yaws = np.full((self.numwt,), -10, dtype=np.int32)
        self.count_steps = 0
        return self.cur_yaws

    def step(self, action, plot=False):
        # Changes the yaw angles
        self.cur_yaws = action
        self.farm.calculate_wake(yaw_angles=action)
        # power output
        power = self.farm.get_farm_power()

        # reward calc
        delta = (power-self.best_power)
        reward = (delta/self.best_power)*100

        #if power > self.best_power:
        #    self.best_power = power

        self.count_steps += 1

        if self.count_steps > 50:
            done = True
        else:
            done = False

        if plot:
            plotfarm(self.farm)
            print(f'{reward} % par rapport au nominal')

        return self.cur_yaws, reward, done, {}

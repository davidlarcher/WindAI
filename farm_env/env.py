from abc import ABC

import gym
from math import isnan
from gym.spaces import Discrete, Box
import numpy as np
from WindAI.floris.optimize_AI import plotfarm


class FarmEnv(gym.Env):
    """
    Description:
        A wind farm is controlled by yaw angles
    Observation:
        Type: Box(n+2)
        Num     Observation                        Min                Max
        0       current yaw angle (°)             -max_yaw            max_yaw
        ...     ...                               ...                 ...
        n       current yaw angle (°)             -max_yaw            max_yaw
        n+1     wind angle (°)                     0                  359
        n+2     wind speed (kts)                   min_wind_speed     max_wind_speed

    Actions:
        Type: Discrete(2)
        Num   Action                     Min                Max
        0     yaw angle (°)             -max_yaw            max_yaw
        ...
        n     yaw angle (°)             -max_yaw            max_yaw

    Reward:
        Reward is power increase for every step taken, including the termination step
    Starting State:
        TBD
    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """
    def __init__(self, config):
        self.numwt = config["num_wind_turbines"]

        # initialize yaw boundaries
        self.allowed_yaw = config["max_yaw"]
        self.max_yaw = 0
        self.min_yaw = 0

        self.min_wind_speed = config["min_wind_speed"]
        self.max_wind_speed = config["max_wind_speed"]
        self.min_wind_angle = config["min_wind_angle"]
        self.max_wind_angle = config["max_wind_angle"]
        self.farm = config["farm"]

        self.best_power = 0
        self.count_steps = 0
        self.initialized_yaw_angle = 0
        self.initialized_yaws = np.full((self.numwt,), 0, dtype=np.int32)
        self.cur_yaws = np.full((self.numwt,), 0, dtype=np.int32)
        self.cur_wind_speed = [8]  # in kts
        self.cur_wind_angle = [270]  # in degrees
        self.current_nominal_power = 0

        # action space is the yaw angles for the wt , to be multiplied by -max_yaw° and max_yaw°
        action_low = np.full((self.numwt,), -1., dtype=np.float32)
        action_high = np.full((self.numwt,), 1., dtype=np.float32)
        self.action_space = Box(low=action_low, high=action_high, shape=(self.numwt,))
        # observation space TODO
        observation_high = np.append(
            np.array([359]*self.numwt),  # yaw positions for all wind turbines in °, 360° is 0°
            np.array([self.max_wind_angle, self.max_wind_speed])
        )
        observation_low = np.append(
            np.array([0] * self.numwt),  # yaw positions for all wind turbines
            np.array([self.min_wind_angle, self.min_wind_speed])
        )
        print(f'observation low : {observation_low}')
        print(f'observation high : {observation_high}')
        self.observation_space = Box(low=observation_low, high=observation_high, shape=(self.numwt+2,), dtype=np.int32)
        print(f'observation space : {self.observation_space}')

    def reset(self):
        self.count_steps = 0

        # Define wind conditions for this episode
        self.cur_wind_speed = np.random.randint(self.min_wind_speed, self.max_wind_speed)
        self.cur_wind_angle = np.random.randint(self.min_wind_angle, self.max_wind_angle)

        self.initialized_yaw_angle = int((self.cur_wind_angle + 90) % 360)
        self.max_yaw = int((self.initialized_yaw_angle + self.allowed_yaw) % 360)
        self.min_yaw = int((self.initialized_yaw_angle - self.allowed_yaw) % 360)

        self.initialized_yaws = np.full((self.numwt,), self.initialized_yaw_angle, dtype=np.int32)
        self.cur_yaws = self.initialized_yaws

        # Update the flow in the model
        self.farm.reinitialize_flow_field(wind_direction=[self.cur_wind_angle], wind_speed=[self.cur_wind_speed])
        self.farm.calculate_wake(yaw_angles=self.cur_yaws)
        self.current_nominal_power = self.farm.get_farm_power()

        state = np.append(
            self.cur_yaws,
            np.array([self.cur_wind_angle, self.cur_wind_speed]))

        # print(f'initial state is {state}')
        # print(f'observation space is {self.observation_space}')

        return state  # return current state of the environment

    def step(self, action, plot=False):
        # Changes the yaw angles
        commanded_yaws = action * self.allowed_yaw
        # print(f'action {commanded_yaws}')
        self.cur_yaws = (self.initialized_yaws + commanded_yaws.astype(int)) % 360
        observation = np.append(
            self.cur_yaws,
            np.array([self.cur_wind_angle, self.cur_wind_speed]))
        self.farm.calculate_wake(yaw_angles=self.cur_yaws)
        # power output
        power = self.farm.get_farm_power()
        # print(f'power : {power}')

        # reward calc
        delta = (power-self.current_nominal_power)
        reward = (delta/self.current_nominal_power)

        # if power > self.best_power:
        #    self.best_power = power

        self.count_steps += 1

        if self.count_steps > 50 :  # or np.any(self.cur_yaws > self.max_yaw) or np.any(self.cur_yaws < self.min_yaw) or isnan(power):
            done = True
            reward = 0
        else:
            done = False

        if plot:
            improvement = reward * 100
            plotfarm(self.farm, self.cur_wind_angle, self.cur_wind_speed,  improvement)
            print(f'{improvement} % par rapport au nominal')

        # print(f'returning observation {observation}')

        return observation, reward, done, {}

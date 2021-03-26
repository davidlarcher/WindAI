import gym
from math import isnan
from gym.spaces import Discrete, MultiDiscrete, Box
import numpy as np
from WindAI.floris.tools import flow_data
from WindAI.floris.utilities import cosd, sind, sin_cos_to_angle
from WindAI.floris.optimize_AI import farminit
import pandas as pd


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
        self.farm = config['farm']
        self.numwt = config['num_wind_turbines']

        # initialize yaw boundaries
        self.allowed_yaw = config["max_yaw"]

        self.min_wind_speed = config["min_wind_speed"]
        self.max_wind_speed = config["max_wind_speed"]
        self.min_wind_angle = config["min_wind_angle"]
        self.max_wind_angle = config["max_wind_angle"]

        self.continuous_action_space = config["continuous_action_space"]

        self.best_explored_power = {}
        self.count_steps = 0
        self.initialized_yaw_angle = 0
        self.cur_yaws = np.full((self.numwt,), 0, dtype=np.int32)

        self.turbine_powers = np.full((self.numwt,), 0, dtype=np.float64)
        self.turbulent_intensities = np.full((self.numwt,), 0, dtype=np.float64)
        self.thrust_coefs = np.full((self.numwt,), 0, dtype=np.float64)
        self.wt_speed_u = np.full((self.numwt,), 0, dtype=np.float64)
        self.wt_speed_v = np.full((self.numwt,), 0, dtype=np.float64)

        self.cur_wind_speed = [8.]  # in kts
        self.cur_wind_angle = [270.]  # in degrees

        self.initial_wind_angle = 0  # in kts
        self.max_wind_direction_variation = 10,  # max wind angle variation during episode

        self.cur_nominal_power = 0
        self.cur_power = 0
        self.cur_power_ratio = 0
        self.cur_nominal_ti_sum = 0

        if self.continuous_action_space:
            # action space is the yaw angles for the wt , to be multiplied by allowed_yaw°
            action_low = np.full((self.numwt,), -1., dtype=np.float32)
            action_high = np.full((self.numwt,), 1., dtype=np.float32)
            self.action_space = Box(low=action_low, high=action_high, shape=(self.numwt,))
        else:
            # discrete action space
            self.action_space = MultiDiscrete(np.full((self.numwt,), 2 * self.allowed_yaw, dtype=np.float32))

        print(f'action space : {self.action_space}')
        print(f'action space : {self.action_space.sample()}')

        # observation space TODO
        observation_high = np.concatenate((
            #  np.array([self.max_wind_angle, self.max_wind_speed])),
            np.array([1.] * self.numwt),  # yaw max positions for all wind turbines
            #  np.array([1] * self.numwt),  # x axis wind speed for all wind turbines
            #  np.array([1] * self.numwt),  # y axis wind speed for all wind turbines
            np.array([1] * self.numwt),  # max turbulence intensity for all wind turbines
            np.array([1]),  # max power ratio
            #  np.array([1] * self.numwt),  # max thrust coef for all wind turbines
            #  np.array([1] * self.numwt),  # max power coef for all wind turbines
            np.array([1]),  # max sinus wind angle
            np.array([1]),  # max cosinus wind angle
            np.array([1])),  # max normalized wind speed (range 2 to 25.5 m.s-1)
            axis=0
        )
        observation_low = np.concatenate((
            #  np.array([self.min_wind_angle, self.min_wind_speed])),
            np.array([-1] * self.numwt),  # yaw min positions for all wind turbines
            #  np.array([-1] * self.numwt),  # x axis wind speed for all wind turbines
            #  np.array([-1] * self.numwt),  # y axis wind speed for all wind turbines
            np.array([0.] * self.numwt),  # min turbulence intensity for all wind turbines
            np.array([-1]),  # min power ratio
            #  np.array([0] * self.numwt),  # min thrust coef for all wind turbines
            #  np.array([0] * self.numwt),  # min power coef for all wind turbines
            np.array([-1]),  # min sinus wind angle
            np.array([-1]),  # min cosinus wind angle
            np.array([0])),  # min normalized wind speed (range 2 to 25.5 m.s-1)
            axis=0
        )
        print(f'observation low : {observation_low}')
        print(f'observation high : {observation_high}')
        self.observation_space = Box(low=observation_low, high=observation_high, shape=(self.numwt*2+4,), dtype=np.float64)
        print(f'observation space : {self.observation_space}')

    def reset(self, wd=None, ws=None):
        self.count_steps = 0
        self.cur_yaws = np.full((self.numwt,), 0, dtype=np.int32)

        # Define wind conditions for this episode

        # check wind speed in range (2 to 25,5)
        assert self.max_wind_speed < 25.5, "max wind speed too high"
        assert self.min_wind_speed > 2., "min wind speed too low"

        if wd:
            self.cur_wind_angle = wd
        else:
            self.cur_wind_angle = np.random.randint(self.min_wind_angle, self.max_wind_angle)

        if ws:
            self.cur_wind_speed = ws
        else:
            self.cur_wind_speed = np.random.uniform(self.min_wind_speed, self.max_wind_speed)

        self.initial_wind_angle = self.cur_wind_angle

        # Update the flow in the model
        print(f'wind angle {self.cur_wind_angle}')
        print(f'wind speed {self.cur_wind_speed}')
        self.farm.reinitialize_flow_field(wind_direction=[self.cur_wind_angle], wind_speed=[self.cur_wind_speed])
        self.farm.calculate_wake()
        self.cur_nominal_power = self.farm.get_farm_power()
        self.best_explored_power[self.cur_wind_angle] = self.cur_nominal_power
        self.cur_nominal_ti_sum = np.sum(self.farm.get_turbine_ti())

        state = self.get_observation()
        # print(f'initial state is {state}')
        # print(f'observation space is {self.observation_space}')

        return state  # return current state of the environment

    def get_observation(self):
        self.turbulent_intensities = (np.array(self.farm.get_turbine_ti())-0.055)/0.07 #rescaling
        self.cur_power = self.farm.get_farm_power()

        # self.thrust_coefs = self.farm.get_turbine_ct()

        #
        # turbine_powers = self.farm.get_turbine_power()
        # self.turbine_powers = turbine_powers / np.max(turbine_powers)
        #
        # wind_speed_points_at_wt = pd.DataFrame(self.farm.get_set_of_points(self.farm_layout[0], self.farm_layout[1], [80.] * self.numwt).head(self.numwt))
        # u_wind_speed_points_at_wt = np.array(wind_speed_points_at_wt.u)
        # v_wind_speed_points_at_wt = np.array(wind_speed_points_at_wt.v)
        # self.wt_speed_u = u_wind_speed_points_at_wt / self.cur_wind_speed
        # self.wt_speed_v = v_wind_speed_points_at_wt / self.cur_wind_speed

        current_yaws = self.cur_yaws / self.allowed_yaw
        self.cur_power_ratio = (self.cur_power - self.cur_nominal_power) / self.cur_nominal_power
        observation = np.concatenate((
            #  self.wt_speed_u,
            # self.wt_speed_v,
            current_yaws,
            self.turbulent_intensities,
            # self.thrust_coefs,
            # self.turbine_powers,
            np.array([self.cur_power_ratio]),
            np.array([sind(self.cur_wind_angle)]),
            np.array([cosd(self.cur_wind_angle)]),
            np.array([self.cur_wind_speed / 25.5]),
        ),
            axis=0
        )

        return observation

    def step(self, action, no_variation=False):

        # check actions validity
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        #  Execute the actions
        if self.continuous_action_space:
            self.cur_yaws = action * self.allowed_yaw
        else:
            self.cur_yaws = action - self.allowed_yaw

        print(f'current yaws {self.cur_yaws}')

        if not no_variation:
            # Apply wind variation
            if self.cur_wind_angle <= self.initial_wind_angle + self.max_wind_direction_variation[0] or self.cur_wind_angle >= self.initial_wind_angle - self.max_wind_direction_variation[0]:
                self.cur_wind_angle = self.cur_wind_angle + np.random.randint(-1, 2)
            self.farm.reinitialize_flow_field(wind_direction=[self.cur_wind_angle], wind_speed=[self.cur_wind_speed])
            print(f'new {self.cur_wind_angle}')
            self.farm.calculate_wake()
            self.cur_nominal_power = self.farm.get_farm_power()
        # Get the Observations from the simulation
        self.farm.calculate_wake(yaw_angles=self.cur_yaws)

        observation = self.get_observation()

        # check observation
        err_msg = "%r (%s) invalid" % (observation, type(observation))
        assert self.observation_space.contains(observation), err_msg



        # reward calc
        # power_ratio = (self.cur_power - self.best_explored_power[self.cur_wind_angle]) / self.best_explored_power[self.cur_wind_angle]

        reward = self.cur_power_ratio * 100
        print(f'power ratio       {self.cur_power_ratio}')

        # if self.cur_power > self.best_explored_power[self.cur_wind_angle]:
        #     self.best_explored_power[self.cur_wind_angle] = self.cur_power

        self.count_steps += 1

        # Done Evaluation
        if self.count_steps == 30:
            done = True
        else:
            done = False

        return observation, reward, done, {}

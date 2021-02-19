
"""Example of a custom gym environment and model. Run this for a demo.
This example shows:
  - using a custom environment
  - using a custom model
  - using Tune for grid search
You can visualize experiment results in ~/ray_results using TensorBoard.
"""
import argparse
import os
import numpy as np
import pygame
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg

import ray
from ray import tune
from ray.tune import grid_search
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.sac import SACTrainer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.sac import SACTrainer
from ray.rllib.agents.ddpg import DDPGTrainer
from WindAI.floris.optimize_AI import farminit, plotfarm
from WindAI.floris import tools as wfct
from WindAI.farm_env.env import FarmEnv
from WindAI.agent_configs import config_PPO, config_SAC, config_DDPG
tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="SAC")
parser.add_argument("--torch", action="store_false")
parser.add_argument("--num-wt-rows", type=int, default=1)
parser.add_argument("--num-wt-cols", type=int, default=2)


class CustomModel(TFModelV2):
    """Example of a keras custom model that just delegates to an fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(CustomModel, self).__init__(obs_space, action_space, num_outputs,
                                          model_config, name)
        self.model = FullyConnectedNetwork(obs_space, action_space,
                                           num_outputs, model_config, name)
        self.register_variables(self.model.variables())

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()


class TorchCustomModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        print(f'CUDA {torch.cuda.is_available()}')
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.torch_sub_model = TorchFC(obs_space, action_space, num_outputs,
                                       model_config, name)

    def forward(self, input_dict, state, seq_lens):
        input_dict["obs"] = input_dict["obs"].float()
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])

if __name__ == "__main__":

    args = parser.parse_args()
    # register_env("corridor", lambda config: SimpleCorridor(config))
    ModelCatalog.register_custom_model(
        "my_model", TorchCustomModel if args.torch else CustomModel)

    pygame.init()

    ray.init()
    agent = {}
    config = {}

    # initialize Farm
    # env variables
    ws = 10.  # wind speed in knots
    wd = 0.  # wind direction in degrees
    wd_change = 1
    ws_change = 1
    farm = farminit(args.num_wt_rows, args.num_wt_cols)
    farm.reinitialize_flow_field(wind_direction=[wd], wind_speed=[ws])
    farm.calculate_wake()
    power = farm.get_farm_power()
    print(f'initial power {power}')

    env_config = {
        "env_config": {
            "num_wind_turbines": args.num_wt_rows * args.num_wt_cols,
            "farm": farm,
            "max_yaw": 20,
            "continuous_action_space": True,
            "min_wind_speed": 10.,  # m.s-1 (range from 2 to 25.5)
            "max_wind_speed": 10.,  # m.s-1 (range from 2 to 25.5)
            "min_wind_angle": 250.,
            "max_wind_angle": 290.
        }}

    # instantiate env class
    env = FarmEnv(env_config['env_config'])

    general_config = {
        "env": FarmEnv,
        "model": {
            "custom_model": "my_model",
        },
        "framework": "torch" if args.torch else "tf",
    }
    if args.run == "PPO":
        agent_config = config_PPO
        config = {
            **env_config,
            **agent_config,
            **general_config
        }
        agent = PPOTrainer(config=config)
    elif args.run == "SAC":
        agent_config = config_SAC
        config = {
            **env_config,
            **agent_config,
            **general_config
        }
        agent = SACTrainer(config=config)
    elif args.run == "DDPG":
        agent_config = config_DDPG
        config = {
            **env_config,
            **agent_config,
            **general_config
        }
        agent = DDPGTrainer(config=config)

    checkpoint_path = '/home/david/ray_results/SAC/SAC_FarmEnv_ff600_00000_0_2021-02-06_14-34-11/checkpoint_50/checkpoint-50'

    agent.restore(checkpoint_path=checkpoint_path)

    font = pygame.font.Font('freesansbold.ttf', 20)
    textX = 10
    textY = 10
    # arrow indicating wind direction
    arrow_Img = pygame.image.load('wind-compass.png')
    arrow_x = 250
    arrow_y = 5


    screen = pygame.display.set_mode((800, 600))

    pygame.display.set_caption('WindAI')


    def update_env():

        farm.reinitialize_flow_field(wind_direction=[wd], wind_speed=[ws])
        farm.calculate_wake()
        nominal_power = farm.get_farm_power()
        obs = env.reset(wd=wd, ws=ws)
        action = agent.compute_action(obs)
        obs, reward, done, info = env.step(action=action)
        yaw_angles = action * 20
        farm.calculate_wake(yaw_angles=yaw_angles)
        steering_power = farm.get_farm_power()

        hor_plane = farm.get_hor_plane(
            height=farm.floris.farm.turbines[0].hub_height)  # x_resolution=400, y_resolution=100)
        # Plot and show
        fig, ax = plt.subplots()
        wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        size = canvas.get_width_height()
        raw_data = renderer.tostring_rgb()
        surf = pygame.image.fromstring(raw_data, size, "RGB")
        return surf, nominal_power, steering_power

    farm_img, power, new_power = update_env()

    def show_env_params(x, y, pwr, new_pwr):
        wind_direction = font.render("Wind direction " + str(wd) + "°", True, (0, 0, 0))
        wind_speed = font.render("Wind speed " + str(ws) + "°", True, (0, 0, 0))
        current_power = font.render("current power " + str(int(pwr)) + "W", True, (0, 0, 0))
        nw_power = font.render("new power " + str(int(new_pwr)) + "W", True, (0, 0, 0))
        var_pwr = (new_power - pwr) / pwr * 100
        var_power = font.render("power variation" + str(int(var_pwr)) + "%", True, (0, 0, 0))
        screen.blit(wind_direction, (x, y))
        screen.blit(wind_speed, (x, y + 30))
        screen.blit(current_power, (x, y + 60))
        screen.blit(nw_power, (x, y + 90))
        screen.blit(var_power, (x, y + 120))


    def arrow():
        rotated_image = pygame.transform.rotate(arrow_Img, -wd)
        screen.blit(rotated_image, (arrow_x, arrow_y))

    def farm_viz():
        screen.blit(farm_img, (0, 0))


    # Game loop
    running = True

    while running:

        screen.fill((255, 255, 255))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    if wd < wd_change:
                        wd = 360 - wd_change
                    else:
                        wd -= wd_change
                    farm_img, power, new_power = update_env()
                if event.key == pygame.K_RIGHT:
                    if wd + wd_change >= 360:
                        wd = wd + wd_change - 360
                    else:
                        wd += wd_change
                    farm_img, power, new_power = update_env()
        farm_viz()
        show_env_params(textX, textY, power, new_power)
        arrow()
        pygame.display.update()

    # for cur_yaws_angle in range(-20,20, 20):
        # cur_yaws = np.full((num_wt,), cur_yaws_angle, dtype=np.int32)
        # print(f'current yaws {cur_yaws}')
        # farm.calculate_wake(yaw_angles=cur_yaws)
        # turbine_powers = np.array(farm.get_turbine_power())
        # turbine_powers = turbine_powers / np.max(turbine_powers)
        # print(f'turbulent powers : {turbine_powers}')
        #
        # turbine_ti = farm.get_turbine_ti()
        # print(f'turbulent intensity : {turbine_ti}')
        # farm.get_turbine_ct()
        #
        # turbine_ct = farm.get_turbine_ct()
        # print(f'turbine thrust coef : {turbine_ct}')
        #
        # layout = farm.get_turbine_layout()
        # print(f'layout {layout}')
        #
        # points = farm.get_set_of_points(layout[0], layout[1],
        #                                                       [80.] * num_wt)
        # u_speed = np.array(points.u) / wind_speed
        # v_speed = np.array(points.v) / wind_speed
        # print(u_speed)
        # print(v_speed)
        # # print(np.array(points.v))
        # print(np.degrees(np.arctan2(v_speed, u_speed)))
        # print(points.w)





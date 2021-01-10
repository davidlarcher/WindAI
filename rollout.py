
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
import tensorflow as tf
tf.config.list_physical_devices('GPU')


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
from WindAI.farm_env.env import FarmEnv
from WindAI.floris.optimize_AI import farminit, plotfarm

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--torch", action="store_false")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-iters", type=int, default=1)
parser.add_argument("--stop-timesteps", type=int, default=100000)
parser.add_argument("--stop-reward", type=float, default=100.)
parser.add_argument("--num-wt-rows", type=int, default=2)
parser.add_argument("--num-wt-cols", type=int, default=3)


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

    # initialize Farm
    farm = farminit(args.num_wt_rows, args.num_wt_cols)
    # Initial power output
    # initial_power = farm.get_farm_power()
    # print(f'initial power {initial_power}')

    num_wt = args.num_wt_rows * args.num_wt_cols
    wind_angle = 270.
    wind_speed = 10.
    farm.reinitialize_flow_field(wind_direction=wind_angle, wind_speed=wind_speed)

    for cur_yaws_angle in range(-20,20, 2):
        cur_yaws = np.full((num_wt,), cur_yaws_angle, dtype=np.int32)
        print(f'current yaws {cur_yaws}')
        farm.calculate_wake(yaw_angles=cur_yaws)
        turbine_powers = np.array(farm.get_turbine_power())
        turbine_powers = turbine_powers / np.max(turbine_powers)
        print(f'turbulent powers : {turbine_powers}')

        turbine_ti = farm.get_turbine_ti()
        print(f'turbulent intensity : {turbine_ti}')
        farm.get_turbine_ct()

        turbine_ct = farm.get_turbine_ct()
        print(f'turbine thrust coef : {turbine_ct}')

        layout = farm.get_turbine_layout()
        print(f'layout {layout}')

        points = farm.get_set_of_points(layout[0], layout[1],
                                                              [80.] * num_wt)
        u_speed = np.array(points.u) / wind_speed
        v_speed = np.array(points.v) / wind_speed
        print(u_speed)
        print(v_speed)
        print(np.array(points.v))
        print(np.degrees(np.arctan2(v_speed, u_speed)))
        print(points.w)

        plotfarm(farm, wind_angle, 8.)



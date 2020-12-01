
"""Example of a custom gym environment and model. Run this for a demo.
This example shows:
  - using a custom environment
  - using a custom model
  - using Tune for grid search
You can visualize experiment results in ~/ray_results using TensorBoard.
"""
import argparse
import os

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
parser.add_argument("--stop-iters", type=int, default=10)
parser.add_argument("--stop-timesteps", type=int, default=100000)
parser.add_argument("--stop-reward", type=float, default=100.)
parser.add_argument("--num-wt-rows", type=int, default=3)
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
    plotfarm(farm, 270., 8.)
    # Initial power output
    # initial_power = farm.get_farm_power()
    # print(f'initial power {initial_power}')

    ray.init(num_gpus=4)

    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))
    ModelCatalog.register_custom_model(
        "my_model", TorchCustomModel if args.torch else CustomModel)

    env_config = {
            "num_wind_turbines": args.num_wt_rows * args.num_wt_cols,
            "farm": farm,
            "max_yaw": 20,
            "min_wind_speed": 5,
            "max_wind_speed": 6,
            "min_wind_angle": 250,
            "max_wind_angle": 290
        }

    config = {
        "env": FarmEnv,
        "env_config": env_config,
        # Use GPUs if `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": 4,  # int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "model": {
            "custom_model": "my_model",
        },
        "vf_share_layers": True,
        # "vf_loss_coeff": 0.5,
        "lr": 1e-2, # grid_search([1e-2, 1e-4, 1e-6]),  # try different lrs
        "num_workers": 2,  # parallelism
        "framework": "torch" if args.torch else "tf",
    }

    config_SAC = {
        "env": FarmEnv,
        "env_config": env_config,
        # Use GPUs if `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": 4,  # int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "model": {
            "custom_model": "my_model",
        },
        # "vf_share_layers": True,
        # "vf_loss_coeff": 0.5,
        #  "lr": grid_search([1e-2, 1e-4, 1e-6]),  # try different lrs
        "num_workers": 4,  # parallelism
        "framework": "torch" if args.torch else "tf",
        "policy_model": {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [512, 512],
        },
    }

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    results = tune.run(args.run, config=config, stop=stop, checkpoint_at_end=True)

    # list of lists: one list per checkpoint; each checkpoint list contains
    # 1st the path, 2nd the metric value
    checkpoints = results.get_trial_checkpoints_paths(
        trial = results.get_best_trial("episode_reward_mean", mode='max'),
        metric = "episode_reward_mean")
    checkpoint_path, _ = checkpoints[0]
    agent = PPOTrainer(config=config)
    agent.restore(checkpoint_path=checkpoint_path)

    # instantiate env class
    env = FarmEnv(env_config)

    # run until episode ends
    episode_reward = 0
    done = False
    for i in range(0, 10):
        obs = env.reset()
        action = agent.compute_action(obs)
        obs, reward, done, info = env.step(action=action, plot=True)
        episode_reward += reward

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)
    ray.shutdown()

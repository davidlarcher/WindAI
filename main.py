
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
from ray.rllib.agents.ddpg import DDPGTrainer
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from WindAI.farm_env.env import FarmEnv
from WindAI.agent_configs import config_PPO, config_SAC, config_DDPG
from WindAI.floris.optimize_AI import farminit, plotfarm
from pprint import pprint

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()
parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="SAC")
parser.add_argument("--torch", action="store_false")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-iters", type=int, default=1000)
parser.add_argument("--stop-timesteps", type=int, default=1000000)
parser.add_argument("--stop-reward", type=float, default=100.)
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

    # initialize Farm
    farm = farminit(args.num_wt_rows, args.num_wt_cols)
    # plotfarm(farm, 270., 8.)
    # Initial power output
    # initial_power = farm.get_farm_power()
    # print(f'initial power {initial_power}')

    ray.init(num_gpus=4)

    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))
    ModelCatalog.register_custom_model(
        "my_model", TorchCustomModel if args.torch else CustomModel)

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

    general_config = {
        "env": FarmEnv,
        "model": {
            "custom_model": "my_model",
        },
        "framework": "torch" if args.torch else "tf",
        "callbacks": DefaultCallbacks,
    }

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    agent = {}
    agent_config = {}
    if args.run == "PPO":
        agent_config = config_PPO
    elif args.run == "SAC":
        agent_config = config_SAC
    elif args.run == "DDPG":
        agent_config = config_DDPG

    config = {
        **env_config,
        **agent_config,
        **general_config
    }
    print(config)
    results = tune.run(
        args.run,
        config=config,
        stop=stop,
        checkpoint_freq=10,
        checkpoint_at_end=True,
        # restore="/home/david/ray_results/SAC/SAC_FarmEnv_5aa8e_00000_0_2021-01-21_18-23-19/checkpoint_199/checkpoint-199",
    )
    if args.run == "PPO":
        agent = PPOTrainer(config=config)
    elif args.run == "SAC":
        agent = SACTrainer(config=config)
    elif args.run == "DDPG":
        agent = DDPGTrainer(config=config)

    # list of lists: one list per checkpoint; each checkpoint list contains
    # 1st the path, 2nd the metric value
    checkpoints = results.get_trial_checkpoints_paths(
        trial = results.get_best_trial("episode_reward_mean", mode='max'),
        metric = "episode_reward_mean")
    checkpoint_path, _ = checkpoints[0]
    print(f'checkpoint_path {checkpoint_path}')
    #  agent = PPOTrainer(config=config_PPO)

    agent.restore(checkpoint_path=checkpoint_path)
    policy = agent.get_policy()
    if args.torch:
        pprint(repr(policy))
    else:
        policy.base_model.summary()
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


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
from WindAI.floris.optimize_AI import farminit, plotfarm
from pprint import pprint

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()
parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="SAC")
parser.add_argument("--torch", action="store_false")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-iters", type=int, default=100)
parser.add_argument("--stop-timesteps", type=int, default=10000000)
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
            "num_wind_turbines": args.num_wt_rows * args.num_wt_cols,
            "farm": farm,
            "max_yaw": 20,
            "min_wind_speed": 10.,  # m.s-1 (range from 2 to 25.5)
            "max_wind_speed": 10.,  # m.s-1 (range from 2 to 25.5)
            "min_wind_angle": 250.,
            "max_wind_angle": 290.
        }

    config_PPO = {
        "env": FarmEnv,
        "env_config": env_config,
        # Use GPUs if `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": 0,  # int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        # "num_gpus_per_worker": 1,
        "num_workers": 2,  # parallelism
        "model": {
            "custom_model": "my_model",
        },
        "callbacks": DefaultCallbacks,
        #  "vf_share_layers": True,
        #  "vf_loss_coeff": 0.5,
        "lr": 1e-3,  # grid_search([1e-2, 1e-4, 1e-6]),  # try different lrs
        "framework": "torch" if args.torch else "tf",
        # Should use a critic as a baseline (otherwise don't use value baseline;
        # required for using GAE).
        "use_critic": True,
        # If true, use the Generalized Advantage Estimator (GAE)
        # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
        "use_gae": True,
        # The GAE(lambda) parameter.
        "lambda": 1.0,
        # Initial coefficient for KL divergence.
        "kl_coeff": 0.2,
        # Size of batches collected from each worker.
        "rollout_fragment_length": 200,
        # Number of timesteps collected for each SGD round. This defines the size
        # of each SGD epoch.
        "train_batch_size": 4000,
        # Total SGD batch size across all devices for SGD. This defines the
        # minibatch size within each epoch.
        "sgd_minibatch_size": 128,
        # Whether to shuffle sequences in the batch when training (recommended).
        "shuffle_sequences": True,
        # Number of SGD iterations in each outer loop (i.e., number of epochs to
        # execute per train batch).
        "num_sgd_iter": 30,
        # Stepsize of SGD.
        # Learning rate schedule.
        "lr_schedule": None,
        # Share layers for value function. If you set this to True, it's important
        # to tune vf_loss_coeff.
        "vf_share_layers": False,
        # Coefficient of the value function loss. IMPORTANT: you must tune this if
        # you set vf_share_layers: True.
        "vf_loss_coeff": 1.0,
        # Coefficient of the entropy regularizer.
        "entropy_coeff": 0.0,
        # Decay schedule for the entropy regularizer.
        "entropy_coeff_schedule": None,
        # PPO clip parameter.
        "clip_param": 0.3,
        # Clip param for the value function. Note that this is sensitive to the
        # scale of the rewards. If your expected V is large, increase this.
        "vf_clip_param": 10.0,
        # If specified, clip the global norm of gradients by this amount.
        "grad_clip": None,
        # Target value for KL divergence.
        "kl_target": 0.01,
        # Whether to rollout "complete_episodes" or "truncate_episodes".
        "batch_mode": "truncate_episodes",
        # Which observation filter to apply to the observation.
        "observation_filter": "NoFilter",
        # Uses the sync samples optimizer instead of the multi-gpu one. This is
        # usually slower, but you might want to try it if you run into issues with
        # the default optimizer.
        "simple_optimizer": False,
        # Whether to fake GPUs (using CPUs).
        # Set this to True for debugging on non-GPU machines (set `num_gpus` > 0).
        "_fake_gpus": False,
        # Switch on Trajectory View API for PPO by default.
        # NOTE: Only supported for PyTorch so far.
        "_use_trajectory_view_api": True,
    }

    config_SAC = {
        "env": FarmEnv,
        "env_config": env_config,
        "model": {
            "custom_model": "my_model",
        },
        # "vf_share_layers": True,
        # "vf_loss_coeff": 0.5,
        "lr": 1e-3,  # grid_search([1e-2, 1e-4, 1e-6]),  # try different lrs
        "framework": "torch" if args.torch else "tf",
        # === Model ===
        # Use two Q-networks (instead of one) for action-value estimation.
        # Note: Each Q-network will have its own target network.
        "twin_q": True,
        # Use a e.g. conv2D state preprocessing network before concatenating the
        # resulting (feature) vector with the action input for the input to
        # the Q-networks.
        "use_state_preprocessor": False,
        # Model options for the Q network(s).
        "Q_model": {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [512, 512],
        },
        # Model options for the policy function.
        "policy_model": {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [512, 512],
        },
        # Unsquash actions to the upper and lower bounds of env's action space.
        # Ignored for discrete action spaces.
        "normalize_actions": True,

        # === Learning ===
        # Disable setting done=True at end of episode. This should be set to True
        # for infinite-horizon MDPs (e.g., many continuous control problems).
        "no_done_at_end": False,
        # Update the target by \tau * policy + (1-\tau) * target_policy.
        "tau": 5e-2,  # found with grid_search([5e-2, 5e-3, 5e-4]),  # 5e-3
        # Initial value to use for the entropy weight alpha.
        "initial_alpha": 1.0,
        # Target entropy lower bound. If "auto", will be set to -|A| (e.g. -2.0 for
        # Discrete(2), -3.0 for Box(shape=(3,))).
        # This is the inverse of reward scale, and will be optimized automatically.
        "target_entropy": None,
        # N-step target updates. If >1, sars' tuples in trajectories will be
        # postprocessed to become sa[discounted sum of R][s t+n] tuples.
        "n_step": 1,
        # Number of env steps to optimize for before returning.
        "timesteps_per_iteration": 1000,

        # === Replay buffer ===
        # Size of the replay buffer. Note that if async_updates is set, then
        # each worker will have a replay buffer of this size.
        "buffer_size": int(1e6),
        # If True prioritized replay buffer will be used.
        "prioritized_replay": False,
        "prioritized_replay_alpha": 0.6,
        "prioritized_replay_beta": 0.4,
        "prioritized_replay_eps": 1e-6,
        "prioritized_replay_beta_annealing_timesteps": 20000,
        "final_prioritized_replay_beta": 0.4,
        # Whether to LZ4 compress observations
        "compress_observations": False,
        # If set, this will fix the ratio of replayed from a buffer and learned on
        # timesteps to sampled from an environment and stored in the replay buffer
        # timesteps. Otherwise, the replay will proceed at the native ratio
        # determined by (train_batch_size / rollout_fragment_length).
        "training_intensity": None,

        # === Optimization ===
        "optimization": {
            "actor_learning_rate": 3e-4,  # 3e-4,
            "critic_learning_rate": 3e-4,  # 3e-4,
            "entropy_learning_rate": 3e-4,
        },
        # If not None, clip gradients during optimization at this value.
        "grad_clip": None,
        # How many steps of the model to sample before learning starts.
        "learning_starts": 1500,
        # Update the replay buffer with this many samples at once. Note that this
        # setting applies per-worker if num_workers > 1.
        "rollout_fragment_length": 1,
        # Size of a batched sampled from replay buffer for training. Note that
        # if async_updates is set, then each worker returns gradients for a
        # batch of this size.
        "train_batch_size": 256,
        # Update the target network every `target_network_update_freq` steps.
        "target_network_update_freq": 0,

        # === Parallelism ===
        # Whether to use a GPU for local optimization.
        "num_gpus": 0,
        # Number of workers for collecting samples with. This only makes sense
        # to increase if your environment is particularly slow to sample, or if
        # you"re using the Async or Ape-X optimizers.
        "num_workers": 0,
        # Whether to allocate GPUs for workers (if > 0).
        "num_gpus_per_worker": 0,
        # Whether to allocate CPUs for workers (if > 0).
        "num_cpus_per_worker": 1,
        # Whether to compute priorities on workers.
        "worker_side_prioritization": False,
        # Prevent iterations from going lower than this time span.
        "min_iter_time_s": 1,

        # Whether the loss should be calculated deterministically (w/o the
        # stochastic action sampling step). True only useful for cont. actions and
        # for debugging!
        "_deterministic_loss": False,
        # Use a Beta-distribution instead of a SquashedGaussian for bounded,
        # continuous action spaces (not recommended, for debugging only).
        "_use_beta_distribution": False,
    }

    config_ddpg = {
        "env": FarmEnv,
        "env_config": env_config,
            # === Twin Delayed DDPG (TD3) and Soft Actor-Critic (SAC) tricks ===
            # TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html
            # In addition to settings below, you can use "exploration_noise_type" and
            # "exploration_gauss_act_noise" to get IID Gaussian exploration noise
            # instead of OU exploration noise.
            # twin Q-net
            "twin_q": False,
            # delayed policy update
            "policy_delay": 1,
            # target policy smoothing
            # (this also replaces OU exploration noise with IID Gaussian exploration
            # noise, for now)
            "smooth_target_policy": False,
            # gaussian stddev of target action noise for smoothing
            "target_noise": 0.2,
            # target noise limit (bound)
            "target_noise_clip": 0.5,

            # === Evaluation ===
            # Evaluate with epsilon=0 every `evaluation_interval` training iterations.
            # The evaluation stats will be reported under the "evaluation" metric key.
            # Note that evaluation is currently not parallelized, and that for Ape-X
            # metrics are already only reported for the lowest epsilon workers.
            "evaluation_interval": None,
            # Number of episodes to run per evaluation period.
            "evaluation_num_episodes": 10,

            # === Model ===
            # Apply a state preprocessor with spec given by the "model" config option
            # (like other RL algorithms). This is mostly useful if you have a weird
            # observation shape, like an image. Disabled by default.
            "use_state_preprocessor": False,
            # Postprocess the policy network model output with these hidden layers. If
            # use_state_preprocessor is False, then these will be the *only* hidden
            # layers in the network.
            "actor_hiddens": [400, 300],
            # Hidden layers activation of the postprocessing stage of the policy
            # network
            "actor_hidden_activation": "relu",
            # Postprocess the critic network model output with these hidden layers;
            # again, if use_state_preprocessor is True, then the state will be
            # preprocessed by the model specified with the "model" config option first.
            "critic_hiddens": [400, 300],
            # Hidden layers activation of the postprocessing state of the critic.
            "critic_hidden_activation": "relu",
            # N-step Q learning
            "n_step": 1,

            # === Exploration ===
            "exploration_config": {
                # DDPG uses OrnsteinUhlenbeck (stateful) noise to be added to NN-output
                # actions (after a possible pure random phase of n timesteps).
                "type": "OrnsteinUhlenbeckNoise",
                # For how many timesteps should we return completely random actions,
                # before we start adding (scaled) noise?
                "random_timesteps": 1000,
                # The OU-base scaling factor to always apply to action-added noise.
                "ou_base_scale": 0.1,
                # The OU theta param.
                "ou_theta": 0.15,
                # The OU sigma param.
                "ou_sigma": 0.2,
                # The initial noise scaling factor.
                "initial_scale": 1.0,
                # The final noise scaling factor.
                "final_scale": 1.0,
                # Timesteps over which to anneal scale (from initial to final values).
                "scale_timesteps": 10000,
            },
            # Number of env steps to optimize for before returning
            "timesteps_per_iteration": 1000,
            # Extra configuration that disables exploration.
            "evaluation_config": {
                "explore": False
            },
            # === Replay buffer ===
            # Size of the replay buffer. Note that if async_updates is set, then
            # each worker will have a replay buffer of this size.
            "buffer_size": 50000,
            # If True prioritized replay buffer will be used.
            "prioritized_replay": True,
            # Alpha parameter for prioritized replay buffer.
            "prioritized_replay_alpha": 0.6,
            # Beta parameter for sampling from prioritized replay buffer.
            "prioritized_replay_beta": 0.4,
            # Time steps over which the beta parameter is annealed.
            "prioritized_replay_beta_annealing_timesteps": 20000,
            # Final value of beta
            "final_prioritized_replay_beta": 0.4,
            # Epsilon to add to the TD errors when updating priorities.
            "prioritized_replay_eps": 1e-6,
            # Whether to LZ4 compress observations
            "compress_observations": False,
            # If set, this will fix the ratio of replayed from a buffer and learned on
            # timesteps to sampled from an environment and stored in the replay buffer
            # timesteps. Otherwise, the replay will proceed at the native ratio
            # determined by (train_batch_size / rollout_fragment_length).
            "training_intensity": None,

            # === Optimization ===
            # Learning rate for the critic (Q-function) optimizer.
            "critic_lr": 1e-3,
            # Learning rate for the actor (policy) optimizer.
            "actor_lr": 1e-3,
            # Update the target network every `target_network_update_freq` steps.
            "target_network_update_freq": 0,
            # Update the target by \tau * policy + (1-\tau) * target_policy
            "tau": 0.002,
            # If True, use huber loss instead of squared loss for critic network
            # Conventionally, no need to clip gradients if using a huber loss
            "use_huber": False,
            # Threshold of a huber loss
            "huber_threshold": 1.0,
            # Weights for L2 regularization
            "l2_reg": 1e-6,
            # If not None, clip gradients during optimization at this value
            "grad_clip": None,
            # How many steps of the model to sample before learning starts.
            "learning_starts": 1500,
            # Update the replay buffer with this many samples at once. Note that this
            # setting applies per-worker if num_workers > 1.
            "rollout_fragment_length": 1,
            # Size of a batched sampled from replay buffer for training. Note that
            # if async_updates is set, then each worker returns gradients for a
            # batch of this size.
            "train_batch_size": 256,

            # === Parallelism ===
            # Number of workers for collecting samples with. This only makes sense
            # to increase if your environment is particularly slow to sample, or if
            # you're using the Async or Ape-X optimizers.
            "num_workers": 0,
            # Whether to compute priorities on workers.
            "worker_side_prioritization": False,
            # Prevent iterations from going lower than this time span
            "min_iter_time_s": 1,
    }

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    agent = {}
    config = {}
    if args.run == "PPO":
        config = config_PPO
    elif args.run == "SAC":
        config = config_SAC
    elif args.run == "DDPG":
        config = config_ddpg

    results = tune.run(
        args.run,
        config=config,
        stop=stop,
        checkpoint_at_end=True,
        # restore="/home/david/ray_results/PPO/PPO_FarmEnv_8a934_00000_0_2020-12-16_21-23-21/checkpoint_800/checkpoint-800",
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

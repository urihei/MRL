import os
from pathlib import Path

import torch
import yaml
from matplotlib import pyplot as plt

from agent import WalkingAgent
from constants import RunArgs
from env import Env
from scene_generator import SceneGenerator
from state import BaseState

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback


def train(run_args: RunArgs, n_agents, base_path, model_path, env, policy_kwargs, val_env=None):
    check_point_path = base_path / "check_points"
    check_point_path.mkdir(exist_ok=True)
    # env = ss.black_death_v3(env)
    env_train = ss.vector.MarkovVectorEnv(env, black_death=True)
    n_envs = max(run_args.n_env, n_agents) // n_agents
    env_train = ss.concat_vec_envs_v1(env_train, n_envs, num_cpus=run_args.n_cpu, base_class="stable_baselines3")
    if model_path.exists():
        rl_model = PPO.load(str(model_path), env=env_train, device=run_args.device)
    else:
        rl_model = PPO("CnnPolicy",
                       env_train,
                       gamma=run_args.gamma,
                       learning_rate=run_args.learning_rate,
                       n_steps=run_args.n_steps,
                       batch_size=run_args.batch_size,
                       n_epochs=run_args.n_epochs,
                       ent_coef=run_args.ent_coef,
                       vf_coef=run_args.vf_coef,
                       clip_range=run_args.clip_range,
                       clip_range_vf=None,
                       gae_lambda=run_args.gae_lambda,
                       verbose=1,
                       device=run_args.device,
                       policy_kwargs=policy_kwargs)
    save_step = int(run_args.check_point_freq // n_envs)
    call_backs_list = [
        CheckpointCallback(save_freq=save_step, save_path=str(check_point_path),
                           name_prefix='check_point')]
    if val_env is not None:
        val_env = ss.pettingzoo_env_to_vec_env_v1(val_env)
        val_env = ss.concat_vec_envs_v1(val_env, 16, num_cpus=1, base_class="stable_baselines3")
        call_backs_list.append(
            EvalCallback(eval_env=val_env, eval_freq=int(save_step // 10),  #
                         best_model_save_path=str(base_path)))

    print(f"start training. model path and name: {model_path}")
    rl_model.learn(total_timesteps=run_args.train_steps + 1000, callback=call_backs_list, log_interval=20)
    rl_model.save(model_path)
    return rl_model


def hand_play():
    script_dir = Path(__file__).parent
    with open(script_dir / "scene_config.yaml", "r") as f:
        sample_dict = yaml.safe_load(f)
    n_agents = 2
    scene_generator_arg = {'sample_dict': sample_dict, 'n_agents': n_agents}
    scene = SceneGenerator(**scene_generator_arg).sample()
    state_kwargs = {'height_factor': 1, 'width_factor': 1,
                    'state_height': 30, 'state_width': 20, 'n_levels': scene.get_max_state_level()}
    agent_args = {'state_class': BaseState, 'state_kwargs': state_kwargs}
    env = Env(300,
              number_of_agents=n_agents, agent_class=WalkingAgent, agent_args=agent_args,
              scene_generator=SceneGenerator, scene_generator_arg=scene_generator_arg)
    env.reset()
    finished = False
    fig, axes = plt.subplots(nrows=1, ncols=1)
    while not finished:
        env.render(axes=axes)
        d = {}
        for agent in env.agents:
            a = int(float(input(f"insert action for {agent}, {WalkingAgent.get_number_of_action()}:")))
            d[agent] = a
        observations, rewards, termination, truncated, info = env.step(d)
        if sum([t for t in truncated.values()]) + sum([t for t in termination.values()]) == len(rewards):
            finished = True


def main(base_model_path, scene_config_file, running_file_path=None):
    base_model_path = Path(base_model_path)
    base_model_path.mkdir(exist_ok=True, parents=True)
    method_model_path = base_model_path / "rl_model.zip"
    run_args = RunArgs.load(running_file_path)

    with open(scene_config_file, "r") as f:
        sample_dict = yaml.safe_load(f)
    n_agents = 2
    scene_generator_arg = {'sample_dict': sample_dict, 'n_agents': n_agents}
    scene = SceneGenerator(**scene_generator_arg).sample()
    state_class = BaseState
    state_kwargs = {'height_factor': 1, 'width_factor': 1,
                    'state_height': 2 * scene.s_height - 1, 'state_width': 2 * scene.s_width - 1,
                    'n_levels': scene.get_max_state_level()}
    agent_args = {'state_class': state_class, 'state_kwargs': state_kwargs}

    scene_generator_arg = {'sample_dict': sample_dict, 'n_agents': n_agents}
    env = Env(300,
              number_of_agents=n_agents, agent_class=WalkingAgent, agent_args=agent_args,
              scene_generator=SceneGenerator, scene_generator_arg=scene_generator_arg)
    val_env = None
    train(run_args, n_agents, base_model_path, method_model_path, env, state_class.get_policy_kwargs(), val_env)


if __name__ == "__main__":
    # hand_play()
    script_dir = Path(__file__).parent
    main(base_model_path=r"/home/urihein/data/ATC/res/new_version/tmp",
         scene_config_file=script_dir / "scene_config.yaml")

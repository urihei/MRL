import datetime
import logging
import os
import sys
from logging import handlers
from pathlib import Path

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
import ray
from ray import tune
from ray.rllib.algorithms import PPOConfig
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.utils.from_config import NotProvided
from ray.rllib.utils.metrics import EPISODE_RETURN_MEAN, ENV_RUNNER_RESULTS, NUM_ENV_STEPS_SAMPLED_LIFETIME
from ray.tune.logger import LegacyLoggerCallback, CSVLogger, JsonLogger, NoopLogger

from Logging import Logging
from agent import WalkingAgent
from constants import RunArgs
from env import Env, EnvR
from scene_generator import SceneGenerator
from state import BaseState


def inference_model(rl_model, env: Env, video_dir: Path, number_of_tests: int = -1, save_video: bool = False,
                    seed: int = 456891376):
    env.info.current_episode = 0
    observation, _ = env.reset()
    res = []
    info = {}
    fig, ax = plt.subplots()
    while env.info.dataset_times == 0 and number_of_tests >= env.info.current_episode:
        end_episode = False
        np.random.seed(seed + env.info.current_episode)
        if save_video:
            env.render(axes=ax)
            images = [img] * 25
        total_seconds = 0.0
        while not end_episode:
            actions = {}
            t_start = datetime.datetime.now()
            for agent_name, agent_o in observation.items():
                # TODO: remove the [0].item()
                actions[agent_name] = rl_model.predict(agent_o, deterministic=True)[0].item()
            # TODO: fix
            observation, rewards, termination, truncated, info = env.step(actions)
            total_seconds += (datetime.datetime.now() - t_start).total_seconds()
            end_episode = sum([v or truncated[k] for k, v in termination.items()]) == len(termination)
            if save_video:
                img = env.render(axes=ax)
                images.append(img)
        # if save_video:
        # im = Image.fromarray(img)
        # im.save(str(video_dir / f"final_{env.info.name}.jpeg"))
        # images = images + [img] * 25
        # video_name = f"video_{env.info.name}.mp4"
        # video = cv2.VideoWriter(str(video_dir / video_name), cv2.VideoWriter_fourcc(*'mp4v'), 25,
        #                         (images[0].shape[1], images[0].shape[0]))
        # for image in images:
        #     video.write(image[:, :, [2, 1, 0]])
        # cv2.destroyAllWindows()
        # video.release()
        # paths = {}
        # for agent_name, agent_o in env.agents_objects.items():
        #     the compare expect input [width, height]
        # paths[agent_name] = agent_o.get_path(0.1)[:, [1, 0]]
        # path_file = video_dir / f"path_{env.info.name}.pickle"
        # with path_file.open("wb") as f:
        #     pickle.dump(paths, f)
        # imageio.mimsave(video_dir / video_name, images)
        # np.random.seed(seed + env.info.current_episode)
        observation, _ = env.reset()
        # agent_dict = {k: v for k, v in info.items() if isinstance(k, str) and k.startswith('agent_')}
        # res.append({
        #     'agents': len(agent_dict),
        #     'landing_points': info.get(SceneObjectsType.LandingPoint, info.get(SceneObjectsType.LandingPointIls, 0)),
        #     'Clouds': info.get(SceneObjectsType.Cloud, 0.0),
        #     'Buildings': info.get(SceneObjectsType.Building, 0.0),
        #     'distance_start': np.mean([d['initial dist to landing'] for d in agent_dict.values()]),
        #     'n_steps_alive': np.mean([d['n_steps_alive'] for d in agent_dict.values()]),
        #     'total_rotation': np.mean([d['total_rotation'] for d in agent_dict.values()]),
        #     'trajectory length': np.mean([d['distance_traveled_in_traj'] for d in agent_dict.values()]),
        #     'n_straight_actions': np.mean([d['n_straight_actions'] for d in agent_dict.values()]),
        #     'n_steps in cloud risk area': np.mean([d['n_steps in cloud risk area'] for d in agent_dict.values()]),
        #     'n_steps in historic-path': np.mean([d['n_steps in historic path'] for d in agent_dict.values()]),
        #     "cause_of_death_not_landing": sum(
        #         [(not d['landed'] and d['cause of death'] == 0) for d in agent_dict.values()]),
        #     "cause_of_death_building": sum(
        #         [d['cause of death'] == CauseOfDeath.CRASH_PLANE_TO_BUILDING for d in agent_dict.values()]),
        #     "cause_of_death_cloud": sum(
        #         [d['cause of death'] == CauseOfDeath.DEATH_IN_CLOUD for d in agent_dict.values()]),
        #     "cause_of_death_out_of_bound": sum(
        #         [d['cause of death'] == CauseOfDeath.OUT_OF_BOARD for d in agent_dict.values()]),
        #     "cause_of_death_historic_path": sum(
        #         [d['cause of death'] == CauseOfDeath.DEATH_IN_HISTORIC_PATH for d in agent_dict.values()]),
        #     "cause_of_death_crash": sum(
        #         [d['cause of death'] == CauseOfDeath.CRASH_PLANE_TO_PLANE for d in agent_dict.values()]),
        #     "cause_of_death_fuel": sum(
        #         [d['cause of death'] == CauseOfDeath.OUT_OF_FUEL for d in agent_dict.values()]),
        #     'landed': sum([d['landed'] for d in agent_dict.values()]),
        #     'running_time in seconds': total_seconds,
        # })
    # res_df = pd.DataFrame(res)
    # death_cause_columns = ['cause_of_death_not_landing', 'cause_of_death_building', 'cause_of_death_cloud',
    #                        'cause_of_death_out_of_bound', 'cause_of_death_historic_path', 'cause_of_death_crash',
    #                        'cause_of_death_fuel']
    # res_df[[f"{c}_prob" for c in death_cause_columns]] = res_df[death_cause_columns].div(
    #     res_df[['agents', 'landing_points']].min(axis=1) - res_df['landed'], axis=0)
    # mean_row = res_df.mean()
    # res_df.loc['mean'] = mean_row
    # res_df.loc['std'] = res_df.std()
    # print(mean_row)
    # return res_df


def train(run_args: RunArgs, n_agents, base_path, model_path, env_args, policy_kwargs):
    # ray.init(log_to_driver=False)
    check_point_path = base_path / "check_points"
    load_model_path = base_path
    check_point_path.mkdir(exist_ok=True)
    run_args = RunArgs()
    env = Env(**env_args)
    config = PPOConfig()
    config = config.training(gamma=run_args.gamma,
                             lambda_=run_args.gae_lambda, vf_loss_coeff=run_args.vf_coef,
                             entropy_coeff=run_args.ent_coef, clip_param=run_args.clip_range,
                             lr=run_args.learning_rate, kl_coeff=0.3,
                             train_batch_size=run_args.batch_size)
    config = config.environment(env=EnvR, env_config=env_args).callbacks(Logging)
    config = config.multi_agent(policies={"p0"}, policy_mapping_fn=lambda aid, *args, **kwargs: "p0")
    config = config.framework("torch").debugging(logger_config={'type': NoopLogger})
    config = config.rl_module(
        rl_module_spec=MultiAgentRLModuleSpec(
            module_specs={
                "p0": SingleAgentRLModuleSpec(
                    module_class=PPOTorchRLModule,
                    observation_space=env.observation_space['agent_0'],
                    action_space=env.action_space['agent_0'],
                    model_config_dict={"conv_filters": [[32, 8, 4],
                                                        [64, 4, 3]],
                                       "post_fcnet_hiddens": [512]
                                       },
                    catalog_class=PPOCatalog,
                )
            }
        ),
        _enable_rl_module_api=NotProvided
    )
    config = config.env_runners(num_env_runners=10)
    config = config.api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)
    # algo_high_lr = config.build()
    # for _ in range(20):
    #     train_results = algo_high_lr.train()
    #     Add the phase to the result dict.
    # train_results["phase"] = 1
    # train.report(train_results)
    # phase_high_lr_time = train_results[NUM_ENV_STEPS_SAMPLED_LIFETIME]
    # logging.info(f"lr time: {phase_high_lr_time}")
    # checkpoint_training_high_lr = algo_high_lr.save()
    # algo_high_lr.stop()
    ray.init(_temp_dir=str((base_path).resolve()))
    stop = {f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": 200.0}
    analysis = tune.run(run_or_experiment="PPO",
                        stop={"timesteps_total": env_args.get('time_steps', 512)},
                        checkpoint_config=ray.train.CheckpointConfig(
                            checkpoint_frequency=(env_args.get('check_point_freq', 512) //
                                                  env_args.get('batch_size', 512)),
                            checkpoint_at_end=True),
                        config=config.to_dict(),
                        storage_path=base_path.resolve(),
                        restore=None,
                        log_to_file=True)


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


def main(base_model_path, scene_config_file, running_file_path=None, only_test=False):
    base_model_path = Path(base_model_path)
    base_model_path.mkdir(exist_ok=True, parents=True)
    method_model_path = base_model_path / "rl_model.zip"
    run_args = RunArgs.load(running_file_path)

    with open(scene_config_file, "r") as f:
        sample_dict = yaml.safe_load(f)

    log_path = base_model_path / "logging"
    log_path.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        handlers=[
                            handlers.RotatingFileHandler(log_path / 'info.log', maxBytes=100000000, backupCount=5),
                            logging.StreamHandler(stream=sys.stdout)]
                        )
    logging.info("path: {}".format(log_path))

    n_agents = 2
    # Scene
    scene_generator_arg = {'sample_dict': sample_dict, 'n_agents': n_agents}
    scene = SceneGenerator(**scene_generator_arg).sample()

    # Agents
    state_class = BaseState
    state_kwargs = {'height_factor': 1, 'width_factor': 1,
                    'state_height': 2 * scene.s_height - 1, 'state_width': 2 * scene.s_width - 1,
                    'n_levels': scene.get_max_state_level()}
    agent_args = {'state_class': state_class, 'state_kwargs': state_kwargs}

    env_args = dict(max_steps=300, number_of_agents=n_agents, agent_class=WalkingAgent, agent_args=agent_args,
                    scene_generator=SceneGenerator, scene_generator_arg=scene_generator_arg)
    if not only_test:
        train(run_args, n_agents, base_model_path, method_model_path, env_args,
              state_class.get_policy_kwargs())
    else:
        rl_model = None
    video_dir = base_model_path / "videos"
    video_dir.mkdir(exist_ok=True)
    # inference_model(rl_model, env, video_dir, number_of_tests=100, save_video=True)


if __name__ == "__main__":
    # hand_play()
    script_dir = Path(__file__).parent
    main(base_model_path=r"/home/urihein/data/ATC/res/new_version/tmp",
         scene_config_file=script_dir / "scene_config.yaml", only_test=False)

from typing import Optional, Union, Dict

import gymnasium as gym
from ray.rllib import BaseEnv, Policy, RolloutWorker
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.rl_module import RLModule
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import EpisodeType, PolicyID


class Logging(DefaultCallbacks):
    def on_episode_step(
            self,
            *,
            episode: Union[EpisodeType, Episode, EpisodeV2],
            env_runner: Optional[EnvRunner] = None,
            metrics_logger: Optional[MetricsLogger] = None,
            env: Optional[gym.Env] = None,
            env_index: int,
            rl_module: Optional[RLModule] = None,
            # TODO (sven): Deprecate these args.
            worker: Optional[EnvRunner] = None,
            base_env: Optional[BaseEnv] = None,
            policies: Optional[Dict[PolicyID, Policy]] = None,
            **kwargs,

    ):
        # Make sure this episode is ongoing.
        # assert episode.length > 0, (
        #     "ERROR: `on_episode_step()` callback should not be called right "
        #     "after env reset!"
        # )
        # train_results['env_runners']
        for agent, agent_episode in episode.agent_episodes.items():
            info = agent_episode.get_infos(-1)
            if not isinstance(info['actions'], list):
                metrics_logger.log_value(key=(f"{agent}", "action", info['actions']), value=1, reduce='sum')
            if not isinstance(info['events'], list):
                metrics_logger.log_value(key=(f"{agent}", "events", info['events']), value=1, reduce='sum')
            for agent_property, value in info.items():
                if agent_property in ["actions", "events"]:
                    continue
                if value:
                    metrics_logger.log_value(key=(f"{agent}", agent_property), value=value, reduce='sum')

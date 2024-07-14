import logging
from typing import Any, Type, Set

import matplotlib.pyplot as plt
import numpy as np
from gymnasium.core import ObsType, ActType
from gymnasium.spaces import Dict
from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import AgentID

import tools
from agent import Agent
from info import Info
from scene import Scene


class Env(MultiAgentEnv):
    metadata = {"render_modes": ["human"], "name": "simple_env"}
    env_count = 0

    def __init__(self,
                 max_steps: int,
                 number_of_agents, agent_class: Type[Agent], agent_args: dict[str, Any],
                 scene_generator, scene_generator_arg: dict[str, Any]):
        super().__init__()
        self.env_count += 1
        # Env base parameters
        self.max_steps = max_steps
        self.n_steps = 0
        self.n_epoch = 0
        self.t: float = 0.0

        # Agents
        self._agent_ids = set([tools.create_agent_name(i) for i in range(number_of_agents)])
        self.max_num_agents = number_of_agents
        self.agents_objects = {}
        for name in self._agent_ids:
            self.agents_objects[name] = agent_class(name=name, **agent_args)
            shared_data = self.agents_objects[name].get_shared_data()
            agent_args.update(shared_data)
        self.agent_name_mapping = dict(zip(self._agent_ids, list(range(self.max_num_agents))))
        self.agent_class = agent_class
        self.agents = []

        # Spaces
        self.observation_space: Dict = Dict({agent: self.agents_objects[agent].get_observation_space() for agent in
                                             self._agent_ids})
        self.action_space: Dict = Dict({agent: self.agents_objects[agent].get_action_space() for agent in
                                        self._agent_ids})

        # Scene
        self.scene_generator_object = scene_generator(**scene_generator_arg)
        self.scene_generator = None
        self.scene: Scene = self.scene_generator_object.sample()

        # Info
        self.info = Info(self._agent_ids)

    def step(
            self, actions: dict[AgentID, ActType]
    ) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        active_agents = []
        dt = 0.0
        for agent, action in actions.items():
            if agent not in self.agents:
                continue
            ao: Agent = self.agents_objects[agent]
            active_agents.append(agent)
            dt = ao.execute_action(action)
            is_legal = self.scene.visit(ao, self.t + dt)
            if not is_legal:
                ao.restore_location()
            ao.save_path()
            self.info.update_agent(agent, action, ao.get_events(), ao.get_properties())
        self.t += dt

        new_agents, is_end = self.scene.visit_all([self.agents_objects[agent] for agent in active_agents], self.t)
        for agent in new_agents:
            self.agents_objects[agent].start_agent()
        self.agents += new_agents

        for agent in self._agent_ids:
            if self.agents_objects[agent].is_terminated() and agent in self.agents:
                self.agents.remove(agent)
        terminated = sum([self.agents_objects[agent].is_terminated() for agent in self._agent_ids]
                         ) == len(self._agent_ids) or is_end
        termination = {agent: terminated for agent in self._agent_ids}
        truncate = self.max_steps <= self.n_steps
        truncated = {agent: truncate and not termination[agent] for agent in self._agent_ids}

        all_rewards = self.agent_class.get_all_rewards([self.agents_objects[agent] for agent in active_agents])
        rewards = {}
        for agent in self._agent_ids:
            rewards[agent] = all_rewards / self.max_num_agents
            if agent in active_agents:
                rewards[agent] += self.agents_objects[agent].get_reward()

        info = self.info.info_current()
        # end_episode = sum([termination[agent] or truncated[agent] for agent in actions.keys()]) == len(actions)
        observations = {agent: self.observe(agent) for agent in self.agents}
        # observations.update(
        #     {agent: np.zeros(self.observation_space[agent].shape).astype(np.float32) for agent in self._agent_ids
        #      if agent not in self.agents})
        # print(f"P:{info}")
        # print(f"observations agents: {list(observations)}, reward: {rewards}, termination: {termination},"
        #              f"truncated: {truncated}")
        return observations, rewards, termination, truncated, info

    def reset(
            self,
            seed: int | None = None,
            options: dict | None = None,
    ) -> tuple[dict[AgentID, Any], dict[AgentID, Any]]:
        # self.logger.info(self.info.info())
        self.n_steps = 0
        self.t = 0.0
        self.n_epoch += 1
        # Get next scene
        if self.scene_generator is None:
            self.scene_generator = self.scene_generator_object.generator()
        try:
            self.scene = next(self.scene_generator)
        except StopIteration:
            # self.logger.info("Finish going over all the maps")
            self.info.dataset_times += 1
            self.scene_generator = self.scene_generator_object.generator()
            self.scene = next(self.scene_generator)

        # Reset agents.
        for agent in self._agent_ids:
            self.agents_objects[agent].reset()
        self.scene.set_start_positions(self.agents_objects)
        self.info.reset()
        observations = {agent: ao.scene_to_observation(self.scene, list(self.agents_objects.values()))
                        for agent, ao in self.agents_objects.items()}
        return observations, self.info.get_dict()

    def render(self, axes=None) -> None | np.ndarray | str | list:
        if axes is None:
            fig, axes = plt.subplots(nrows=1, ncols=1)
        else:
            fig = None
        self.scene.render(axes)
        for agent in self.agents:
            ao = self.agents_objects[agent]
            ao.render(axes)
        return fig

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most "up to date" possible)
        at any time after reset() is called.
        """
        return self.agents_objects[agent].scene_to_observation(
            self.scene, [ao for name, ao in self.agents_objects.items() if name in self.agents])

    # def get_agent_ids(self) -> Set[AgentID]:
class EnvR(Env):
    def __init__(self, kwargs):
        super().__init__(**kwargs)

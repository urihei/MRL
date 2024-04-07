import logging
from copy import copy
from typing import Any, Type

import matplotlib.pyplot as plt
import numpy as np
from pettingzoo import ParallelEnv
from pettingzoo.utils.env import AgentID, ObsType, ActionType

import tools
from agent import Agent
from info import Info
from scene import Scene


class Env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "simple_env"}

    def __init__(self,
                 max_steps: int,
                 number_of_agents, agent_class: Type[Agent], agent_args: dict[str, Any],
                 scene_generator, scene_generator_arg: dict[str, Any]):
        # Env base parameters
        self.max_steps = max_steps
        self.n_steps = 0
        self.n_epoch = 0
        self.t: float = 0.0

        # Agents
        self.possible_agents = [tools.create_agent_name(i) for i in range(number_of_agents)]
        self.agents_objects = {}
        for name in self.possible_agents:
            self.agents_objects[name] = agent_class(name=name, **agent_args)
            shared_data = self.agents_objects[name].get_shared_data()
            agent_args.update(shared_data)
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(self.max_num_agents))))
        self.agent_class = agent_class

        # Spaces
        self.observation_spaces = {agent: self.agents_objects[agent].get_observation_space() for agent in
                                   self.possible_agents}
        self.action_spaces = {agent: self.agents_objects[agent].get_action_space() for agent in
                              self.possible_agents}

        # Scene
        self.scene_generator_object = scene_generator(**scene_generator_arg)
        self.scene_generator = None
        self.scene: Scene = self.scene_generator_object.sample()

        # Info
        self.info = Info(self.possible_agents)

    def step(
            self, actions: dict[AgentID, ActionType]
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

        new_agents = self.scene.visit_all([self.agents_objects[agent] for agent in active_agents], self.t)
        self.agents += new_agents

        all_rewards = self.agent_class.get_all_rewards()
        rewards = {}
        termination = {}
        truncated = {}
        for agent in self.possible_agents:
            rewards[agent] = all_rewards / self.max_num_agents
            if agent in active_agents:
                ao = self.agents_objects[agent]
                rewards[agent] += ao.get_reward()
                termination[agent] = ao.is_terminated()
                if termination[agent]:
                    self.agents.remove(agent)
                else:
                    truncated[agent] = self.max_steps >= self.n_steps

        info = self.info.info_current()
        # end_episode = sum([termination[agent] or truncated[agent] for agent in actions.keys()]) == len(actions)
        observations = {agent: self.observe(agent) for agent in active_agents}
        return observations, rewards, termination, truncated, info

    def reset(
            self,
            seed: int | None = None,
            options: dict | None = None,
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        self.n_steps = 0
        self.t = 0.0
        # Get next scene
        if self.scene_generator is None:
            self.scene_generator = self.scene_generator_object.generator()
        try:
            self.scene = next(self.scene_generator)
        except StopIteration:
            logging.info("Finish going over all the maps")
            self.info.dataset_times += 1
            self.scene_generator = self.scene_generator_object.generator()
            self.scene = next(self.scene_generator)

        # Reset agents.
        for agent in self.possible_agents:
            self.agents_objects[agent].reset()

        self.info.reset()
        observations = {agent: ao.scene_to_observation(self.scene) for agent, ao in self.agents_objects.items()}
        return observations, self.info.get_dict()

    def render(self) -> None | np.ndarray | str | list:
        fig, axes = plt.subplot()
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
        return self.agents_objects[agent].scene_to_observation(self.scene)

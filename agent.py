import abc
from abc import ABC
from copy import copy
from typing import Any, Type

import matplotlib.patches as m_patches
from gymnasium.core import ActType
from gymnasium.spaces.space import Space
from matplotlib.axes import Axes

import tools
from constants import Events
from state import State
from tools import Location


class Agent(ABC):
    def __init__(self, name, state_class: Type[State], width=-1, height=-1, state_kwargs: dict[str, Any] = None):
        if state_kwargs is None:
            state_kwargs = {}
        self.name = name
        if height > 0 and width > 0:
            self.location: Location = Location(height, width)
        else:
            self.location = None
        self.events_history = []
        self.action_history: list[ActType] = []
        self.path: list[Location] = []
        self.is_alive: bool = False
        self.is_start: bool = False
        self.complete_task_position: Location | None = None
        self.t = 0.0
        self.state_class = state_class
        self.state = state_class(**state_kwargs)

    @abc.abstractmethod
    def execute_action(self, action: ActType) -> float:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_reward(self) -> float:
        raise NotImplementedError()

    def kill_agent(self):
        self.is_alive = False

    def start_agent(self):
        self.is_start = True
        self.is_alive = True

    def is_terminated(self) -> bool:
        return not self.is_alive and self.is_start

    def reset(self):
        self.location = None
        self.action_history = []
        self.path = []
        self.events_history = []
        self.is_alive: bool = False
        self.is_start: bool = False

    @abc.abstractmethod
    def get_observation_space(self) -> Space:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_action_space(self) -> Space:
        raise NotImplementedError()

    def restore_location(self):
        if self.path:
            self.location = self.path[-1]

    def save_path(self):
        self.path.append(copy(self.location))

    def get_events(self):
        if not self.events_history:
            return []
        return self.events_history[-1]

    def get_properties(self):
        if len(self.path) > 1:
            return {'distance': self.location.get_distance(self.path[-2])}
        return {}

    def get_shared_data(self) -> dict[str, Any]:
        return {}

    @classmethod
    @abc.abstractmethod
    def get_all_rewards(cls, agent_list: list['Agent']):
        return 0.0

    @abc.abstractmethod
    def scene_to_observation(self, scene, agent_list: list['Agent']):
        raise NotImplementedError()

    def render(self, axes: Axes):
        color = tools.get_color(self.name)
        art = m_patches.Circle((self.location.width, self.location.height), 0.5, color=color)
        axes.add_artist(art)

    def update_event(self, event: Events, t):
        if len(self.events_history) < t:
            self.events_history += [[]] * (int(t) - len(self.events_history))
        self.events_history[-1].append(event)

    def complete_task(self, position: Location):
        self.complete_task_position = position
        self.update_event(Events.complete_task, self.t)
        self.is_alive = False

    def get_location(self):
        return copy(self.location)

    def set_location(self, height, width):
        self.location = Location(height=height, width=width)

    def set_t(self, t):
        self.t = t

    def get_policy_kwargs(self):
        return self.state_class.get_policy_kwargs()


class WalkingAgent(Agent):
    def __init__(self, name, state_class: Type[State], width=-1, height=-1, state_kwargs: dict[str, Any] = None,
                 step_size=1, dt=1):
        if state_kwargs is None:
            state_kwargs = {}
        super().__init__(name, state_class=state_class, width=width, height=height, state_kwargs=state_kwargs)
        self.step_size = step_size
        self.dt = dt
        self.action_reward = 0.0

    def execute_action(self, action: ActType) -> float:
        self.action_reward = 0.0
        if action % 2 == 0:
            self.location.height += (action - 1) * self.step_size
        else:
            self.location.width += (action - 2) * self.step_size
        return self.dt

    def get_observation_space(self) -> Space:
        from gymnasium.spaces import Box
        return Box(low=0, high=1, shape=self.state.get_state_size(), )

    def get_action_space(self) -> Space:
        from gymnasium.spaces import Discrete
        return Discrete(4)

    def get_reward(self) -> float:
        if not self.is_alive:
            return 0.0
        if self.complete_task_position:
            return 1.0
        return -0.01

    @classmethod
    def get_all_rewards(cls, agent_list: list['WalkingAgent']):
        all_landing = bool(agent_list)
        for ao in agent_list:
            all_landing = all_landing and ao.complete_task_position
            if not all_landing:
                all_landing = False
                break
        return 10.0 if all_landing else -0.1

    def scene_to_observation(self, scene, agent_list: list['Agent']):
        self.state.update(scene=scene)
        self.state.set_agents([ao.location.to_tuple() for ao in agent_list],
                              colors=[tools.agent_name_to_number(ao.name) for ao in agent_list])
        return self.state.get_state(scene, center=self.location.to_tuple())

    def render(self, axes: Axes):
        axes.plot(self.location.width, self.location.height, '.', color=tools.get_color(self.name))

    @classmethod
    def get_number_of_action(cls):
        return 4

    def kill_agent(self):
        super().kill_agent()
        self.action_reward += -10

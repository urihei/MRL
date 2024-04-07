import abc
from abc import ABC
from copy import copy
from typing import Any

import numpy as np
from gymnasium.spaces.space import Space
from matplotlib.axes import Axes
import matplotlib.patches as m_patches
from pettingzoo.utils.env import ActionType

import tools
from tools import Location
from constants import Events


class Agent(ABC):
    def __init__(self, name, width, height):
        self.name = name
        self.location: Location = Location(height, width)
        self.events_history = []
        self.action_history: list[ActionType] = []
        self.path: list[Location] = []
        self.is_alive: bool = False
        self.landing: Location | None = None
        self.t = 0.0

    @abc.abstractmethod
    def execute_action(self, action: ActionType) -> float:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_reward(self) -> float:
        raise NotImplementedError()

    def kill_agent(self):
        self.is_alive = False

    def is_terminated(self) -> bool:
        return self.is_alive

    @abc.abstractmethod
    def reset(self):
        self.location = None
        self.action_history = []
        self.path = []
        self.events_history = []
        self.is_alive: bool = False

    @abc.abstractmethod
    def get_agent_status(self) -> dict[str, str]:
        raise NotImplementedError()

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
        return self.events_history[-1]

    def get_properties(self):
        if len(self.path) > 1:
            return {'distance': self.location.get_distance(self.path[-2])}
        return {}

    @abc.abstractmethod
    def get_shared_data(self) -> dict[str, Any]:
        return {}

    @classmethod
    @abc.abstractmethod
    def get_all_rewards(cls):
        return 0.0

    @abc.abstractmethod
    def scene_to_observation(self, scene):
        raise NotImplementedError()

    @abc.abstractmethod
    def render(self, axes: Axes):
        color = tools.get_color(self.name)
        art = m_patches.Circle((self.location.width, self.location.height), 0.5, color=color)
        axes.add_artist(art)

    def update_event(self, event: Events, t):
        if t > self.t:
            self.t = t
            self.events_history.append([])
        self.events_history[-1].append(event)

    def land(self, landing: Location):
        self.landing = landing
        self.is_alive = False

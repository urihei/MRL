import abc
from abc import ABC
from copy import copy

import numpy as np
from matplotlib.axes import Axes
import matplotlib.patches as m_patches
from pettingzoo.utils.env import AgentID

from agent import Agent
from tools import Location, Time


class SceneObject(ABC):
    @abc.abstractmethod
    def __init__(self, name, **kwargs):
        self.name = name

    @abc.abstractmethod
    def visit(self, ao_list: list[Agent], t: float):
        raise NotImplementedError()

    @abc.abstractmethod
    def render(self, axes: Axes):
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def sample(cls, n: int, ind: int, max_height, max_width, **kwargs) -> 'SceneObject':
        raise NotImplementedError()

    @classmethod
    def load(cls, param):
        return cls(**param)

    def to_dict(self):
        return {'name': self.name}

    def get_shared_data(self, shared_data):
        return shared_data


class BaseStart(SceneObject):
    def __init__(self, location_param: dict[str, float], agent_list: list[AgentID], time_between_agents):
        super().__init__('base_start')
        self.location = Location(**location_param)
        self.agent_list = copy(agent_list)
        self.original_agent_list = agent_list
        self.time_between_agents = time_between_agents
        self.t = -np.inf

    def visit(self, ao_list: list[Agent], t: float):
        if self.agent_list and (self.t + self.time_between_agents) < t:
            return self.agent_list.pop()

    def render(self, axes: Axes):
        art = m_patches.Rectangle((self.location.width - 0.5, self.location.height - 0.5), 1, 1,
                                  color='m')
        axes.add_artist(art)

    @classmethod
    def sample(cls, n, ind: int, max_height, max_width,
               l_height=0.0, u_height: float = None, l_width=0.0, u_width: float = None,
               time_between_agents: float = 1.0, all_agents: list[AgentID] = None, p_height=None, p_width=None,
               assigned_agent_list=None) -> 'BaseStart':
        if u_height is None:
            u_height = max_height if p_height is None else int(max_height * p_height)
        if u_width is None:
            u_width = max_width if p_width is None else int(max_width * p_width)

        if all_agents is None or len(all_agents) < 1:
            raise ValueError(f"Must specify the possible agents: {all_agents}")
        np.random.shuffle(all_agents)

        if assigned_agent_list is None:
            assigned_agent_list = []

        location = Location.sample(l_height=l_height, u_height=u_height, l_width=l_width, u_width=u_width)
        number_of_agents = int(len(all_agents) // n)
        if ind < len(all_agents) % n:
            number_of_agents += 1
        agent_list = []
        for agent in all_agents:
            if agent not in assigned_agent_list:
                agent_list.append(agent)
        return cls(location_param=location.to_dict(), agent_list=agent_list, time_between_agents=time_between_agents)

    def to_dict(self):
        d = super().to_dict()
        d.update({
            'location': self.location.to_dict(),
            'agent_list': self.original_agent_list,
            'time_between_agents': self.time_between_agents
        })

    def get_shared_data(self, shared_data):
        return {'assigned_agent_list': self.agent_list + shared_data.get('assigned_agent_list', [])}


class BaseEnd(SceneObject):

    def __init__(self, location_param: dict[str, float], time_param: dict[str, float], landing_radius: float):
        super().__init__('base_end')
        self.location = Location(**location_param)
        self.time = Time(**time_param)
        self.landing_radius = landing_radius

    def visit(self, ao_list: list[Agent], t: float):
        if not self.time.is_active(t):
            return
        for ao in ao_list:
            if self.location.get_distance(ao.location) < self.landing_radius:
                ao.landed()

    def render(self, axes: Axes):
        art = m_patches.Rectangle((self.location.width - 0.5, self.location.height - 0.5), 1, 1,
                                  color='r')
        axes.add_artist(art)

    @classmethod
    def sample(cls, n, ind: int, max_height, max_width,
               l_height=0.0, u_height: float = None, l_width=0.0, u_width: float = None,
               ) -> 'BaseEnd':
        if u_height is None:
            u_height = max_height if p_height is None else int(max_height * p_height)
        if u_width is None:
            u_width = max_width if p_width is None else int(max_width * p_width)

        if all_agents is None or len(all_agents) < 1:
            raise ValueError(f"Must specify the possible agents: {all_agents}")
        np.random.shuffle(all_agents)

        if assigned_agent_list is None:
            assigned_agent_list = []

        location = Location.sample(l_height=l_height, u_height=u_height, l_width=l_width, u_width=u_width)
        number_of_agents = int(len(all_agents) // n)
        if ind < len(all_agents) % n:
            number_of_agents += 1
        agent_list = []
        for agent in all_agents:
            if agent not in assigned_agent_list:
                agent_list.append(agent)
        return cls(location_param=location.to_dict(), agent_list=agent_list, time_between_agents=time_between_agents)

    def to_dict(self):
        d = super().to_dict()
        d.update({
            'location': self.location.to_dict(),
            'agent_list': self.original_agent_list,
            'time_between_agents': self.time_between_agents
        })

    def get_shared_data(self, shared_data):
        return {'assigned_agent_list': self.agent_list + shared_data.get('assigned_agent_list', [])}


all_objects = {
    'base_start': BaseStart,
    'base_end': BaseEnd,
}

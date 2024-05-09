import abc
from abc import ABC
from copy import copy, deepcopy
from typing import Callable

import numpy as np
from matplotlib.axes import Axes
import matplotlib.patches as m_patches
from pettingzoo.utils.env import AgentID

import tools
from agent import Agent
from tools import Location, Time, SampleDef


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

    @classmethod
    def dict_to_SampleDef(cls, cfg: dict, **kwargs):
        return SampleDef(n_l=cfg['n_l'], n_u=cfg['n_u'], min_distance=cfg['min_distance'],
                         object_class=cls, object_parameters=deepcopy(cfg['param']))

    def is_active(self, t):
        return True


class StartingPosition(ABC):
    @abc.abstractmethod
    def get_all_starting_positions(self):
        # Return agent location dictionary
        raise NotImplementedError()


class LocationObject(SceneObject, ABC):
    def __init__(self, name, location_param: dict[str, float], state_level: int = -1, state_value: int = 1, **kwargs):
        super().__init__(name)
        self.location = Location(**location_param)
        self.state_level: int = state_level
        self.state_value: int = state_value

    def get_location(self, rescale_function: Callable[[float, float], tuple[float, float]] = None):
        if rescale_function is None:
            rescale_function = lambda x, y: (x, y)
        return [rescale_function(*self.location.to_tuple())]

    def to_dict(self):
        d = super().to_dict()
        d['location_param'] = self.location.to_dict()
        return d


class BaseStart(LocationObject, StartingPosition):
    def get_all_starting_positions(self):
        return {agent: self.location.to_tuple() for agent in self.agent_list}

    def __init__(self, location_param: dict[str, float], agent_list: list[AgentID], time_between_agents):
        super().__init__('base_start', location_param)
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
               sample_location=None,
               time_between_agents: float = 1.0, all_agents: list[AgentID] = None, assigned_agent_list=None
               ) -> 'BaseStart':
        if all_agents is None or len(all_agents) < 1:
            raise ValueError(f"Must specify the possible agents: {all_agents}")
        np.random.shuffle(all_agents)

        if assigned_agent_list is None:
            assigned_agent_list = []

        location = Location.sample(height_s=max_height, width_s=max_width, param=sample_location)
        number_of_agents = int(len(all_agents) // n)
        if ind < len(all_agents) % n:
            number_of_agents += 1
        agent_list = []
        for agent in all_agents:
            if agent not in assigned_agent_list:
                agent_list.append(agent)
            if len(agent_list) >= number_of_agents:
                break
        return cls(location_param=location.to_dict(), agent_list=agent_list, time_between_agents=time_between_agents)

    def to_dict(self):
        d = super().to_dict()
        d.update({
            'agent_list': self.original_agent_list,
            'time_between_agents': self.time_between_agents
        })

    def get_shared_data(self, shared_data):
        return {'assigned_agent_list': self.agent_list + shared_data.get('assigned_agent_list', [])}

    @classmethod
    def dict_to_SampleDef(cls, cfg: dict, number_of_agents=1):
        obj = super().dict_to_SampleDef(cfg=cfg)
        obj.object_class = cls
        obj.object_parameters['all_agents'] = [tools.create_agent_name(x) for x in range(number_of_agents)]
        obj.object_parameters['sample_location'] = tools.SampleLocationParam(**cfg['param']['location'])
        del obj.object_parameters['location']
        return obj


class BaseEnd(LocationObject):

    def __init__(self, location_param: dict[str, float], time_param: dict[str, float], landing_radius: float):
        super().__init__('base_end', location_param=location_param, state_level=0)
        self.time = Time(**time_param)
        self.landing_radius = landing_radius

    def visit(self, ao_list: list[Agent], t: float):
        if not self.time.is_active(t):
            return
        for ao in ao_list:
            if self.location.get_distance(ao.location) < self.landing_radius:
                ao.complete_task(self.location)

    def render(self, axes: Axes):
        art = m_patches.Rectangle((self.location.width - 0.5, self.location.height - 0.5), 1, 1,
                                  color='r')
        axes.add_artist(art)

    @classmethod
    def sample(cls, n, ind: int, max_height=None, max_width=None, sample_location=None, landing_radius=1.0,
               total_time=-1, sample_time=None) -> 'BaseEnd':
        location_param = Location.sample(
            height_s=max_height, width_s=max_width, param=sample_location).to_dict()
        time_param = Time.sample(total_time=total_time, param=sample_time).to_dict()
        return cls(location_param=location_param, time_param=time_param, landing_radius=landing_radius)

    def to_dict(self):
        d = super().to_dict()
        d.update({
            'time_param': self.time.to_dict(),
            'landing_radius': self.landing_radius
        })

    def get_shared_data(self, shared_data):
        return {}

    @classmethod
    def dict_to_SampleDef(cls, cfg: dict, number_of_agents=1):
        obj = super().dict_to_SampleDef(cfg=cfg)
        obj.object_class = cls
        obj.object_parameters['sample_location'] = tools.SampleLocationParam(**cfg['param']['location'])
        del obj.object_parameters['location']
        if 'time' in cfg['param']:
            obj.object_parameters['sample_time'] = tools.SampleTimeParam(**cfg['param']['time'])
            del obj.object_parameters['time']
        return obj

    def get_location(self, rescale_function: Callable[[float, float], tuple[float, float]] = None
                     ) -> list[tuple[float, float]]:
        if rescale_function is None:
            rescale_function = lambda x, y: (x, y)
        return tools.fill_circle(rescale_function(*self.location.to_tuple()),
                                 *rescale_function(self.landing_radius, self.landing_radius))

    def is_active(self, t):
        return self.time.is_active(t)


all_objects = {
    'BaseStart': BaseStart,
    'BaseEnd': BaseEnd,
}

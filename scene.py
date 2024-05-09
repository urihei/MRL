from copy import copy
from typing import Any, Type, Union

import numpy as np
from matplotlib.axes import Axes
from pettingzoo.utils.env import AgentID

from agent import Agent
from constants import Events
from scene_object import SceneObject, StartingPosition
from tools import SampleDef


class Scene(object):
    def __init__(self, s_height, s_width, scene_objects: list[SceneObject]):
        self.s_height = s_height
        self.s_width = s_width
        self.scene_objects: list[SceneObject] = scene_objects
        self.active_hash = 0

    def visit(self, ao: Agent, t: float) -> bool:
        if not self.is_in_scene(ao.location.height, ao.location.width):
            ao.update_event(Events.out_of_bound, t)
            ao.kill_agent()
            return False
        return True

    def visit_all(self, ao_list: list[Agent], t: float) -> list[AgentID]:
        new_agents_set = set()
        self.active_hash = 0
        for so in self.scene_objects:
            r = so.visit(ao_list, t)
            self.active_hash = 2 * self.active_hash + int(so.is_active(t))
            if r is not None:
                new_agents_set.add(r)
        for ao in ao_list:
            ao.set_t(t)
        return list(new_agents_set)

    def render(self, axes: Axes):
        axes.set_xlim(0, self.s_width)
        axes.set_ylim(0, self.s_height)
        for so in self.scene_objects:
            so.render(axes)

    @classmethod
    def sample(cls, height_s, width_s, sample_list: list[SampleDef]) -> 'Scene':
        o_list = []
        for sample_o in sample_list:
            assert sample_o.min_distance >= 0
            n = np.random.randint(sample_o.n_l, sample_o.n_u + 1)
            object_parameters = copy(sample_o.object_parameters)
            shared_data = {}
            o_location = []
            for ind in range(n):
                min_distance = -1
                while sample_o.min_distance > min_distance:
                    obj = sample_o.object_class.sample(n=n, ind=ind, max_height=height_s, max_width=width_s,
                                                       **object_parameters)
                    min_distance = np.inf
                    if hasattr(obj, 'location'):
                        if o_location:
                            min_distance = min([obj.location.get_distance(loc) for loc in o_location])
                o_location.append(obj.location)

                shared_data = obj.get_shared_data(shared_data)
                object_parameters.update(shared_data)
                o_list.append(obj)
        return cls(height_s, width_s, o_list)

    @classmethod
    def load(cls, cfg: dict[str, Union[float, list[tuple[Type[SceneObject], dict[str, Any]]]]]) -> 'Scene':
        o_list = []
        for c, p in cfg['object_list']:
            o_list.append(c.load(**p))
        return cls(cfg['height'], cfg['width'], o_list)

    def to_dict(self) -> dict[str, Union[float, list[tuple[Type[SceneObject], dict[str, Any]]]]]:
        d = {'height': self.s_height, 'width': self.s_width}
        o_list = []
        for o in self.scene_objects:
            o_list.append((o.name, o.to_dict()))
        d['object_list'] = o_list
        return d

    def is_in_scene(self, height, width):
        return (0 <= height < self.s_height) and 0 <= width < self.s_width

    def set_start_positions(self, agent_dict: dict[str, 'Agent']):
        for so in self.scene_objects:
            if isinstance(so, StartingPosition):
                so: StartingPosition
                starting_positions = so.get_all_starting_positions()
                for agent, location in starting_positions.items():
                    agent_dict[agent].set_location(location[0], location[1])

    def get_max_state_level(self):
        max_level = 0
        for so in self.scene_objects:
            level = getattr(so, 'state_level', None)
            if level:
                max_level = max(level, max_level)
        return max_level + 1

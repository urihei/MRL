import math
from typing import Type, Any

import matplotlib.colors as m_colors
import numpy as np
from dataclasses import dataclass


def create_agent_name(x: int):
    return f'agent_{x}'


def agent_name_to_number(name: str):
    return int(name.replace('agent_', ''))


colors = list(m_colors.TABLEAU_COLORS.values())


def get_color(name: str):
    n = agent_name_to_number(name)
    return colors[n]


def to_angle(location):
    return math.atan2(location[0], location[1])


def fill_shape(locations: np.ndarray) -> np.ndarray:
    if locations.shape[0] <= 2:
        return locations
    # sort by angle
    tmp = locations - locations.mean(axis=0)
    sort_index = np.argsort(np.arctan2(tmp[:, 0], tmp[:, 1]))
    locations = locations[sort_index, :]
    max_y = int(np.max(locations[:, 0]))
    min_y = int(np.min(locations[:, 0]))
    all_locations = set([(t[0], t[1]) for t in locations])
    for y in range(min_y, max_y + 1):
        x_intersections = set()
        ind1 = 0
        for i in range(1, locations.shape[0] + 1):
            ind2 = i % locations.shape[0]
            y1, x1 = locations[ind1, :]
            y2, x2 = locations[ind2, :]
            # Check if the point is above the minimum y coordinate of the edge
            # Check if the point is below the maximum y coordinate of the edge
            if y1 != y2 and min(y1, y2) <= y <= max(y1, y2):
                # Calculate the x-intersection of the line connecting the point to the edge
                x_intersection = (y - y1) * (x2 - x1) / (y2 - y1) + x1
                x_intersections.add(x_intersection)
            ind1 = ind2
        x_intersections = list(x_intersections)
        x_intersections.sort()
        for i in range(0, len(x_intersections), 2):
            if i + 1 < len(x_intersections):
                all_locations.update([(y, x) for x in range(int(x_intersections[i]), int(x_intersections[i + 1]))])
    all_locations = np.array(list(all_locations))
    return all_locations


def fill_circle(location: tuple[float, float], landing_radius_h: float, landing_radius_w: float
                ) -> list[tuple[float, float]]:
    locations = [location]
    rh2 = landing_radius_h ** 2
    rw2 = landing_radius_w ** 2
    for h in range(int(location[0] - landing_radius_h), int(location[0] + landing_radius_h)):
        h2 = h ** 2 / rh2
        for w in range(int(location[1] - landing_radius_w), int(location[1] + landing_radius_w)):
            if 1 >= h2 + ((w ** 2) / rw2):
                locations.append((h, w))
    return locations


@dataclass
class SampleLocationParam(object):
    l_height: float = None
    u_height: float = None
    s_p_height: float = None
    e_p_height: float = None
    l_width: float = None
    u_width: float = None
    s_p_width: float = None
    e_p_width: float = None


class Location(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def get_distance(self, other: 'Location'):
        return np.sqrt((self.height - other.height) ** 2 + (self.width - other.width) ** 2)

    def to_dict(self):
        return {'height': self.height, 'width': self.width}

    @classmethod
    def sample(cls, height_s, width_s, param: SampleLocationParam):
        if param.l_height is None:
            param.l_height = 0.0 if param.s_p_height is None else int(height_s * param.s_p_height)
        if param.u_height is None:
            param.u_height = height_s if param.e_p_height is None else int(height_s * param.e_p_height)
        if param.l_width is None:
            param.l_width = 0.0 if param.s_p_width is None else int(width_s * param.s_p_width)
        if param.u_width is None:
            param.u_width = width_s if param.e_p_width is None else int(width_s * param.e_p_width)
        height = param.l_height + (param.u_height - param.l_height) * np.random.rand()
        width = param.l_width + (param.u_width - param.l_width) * np.random.rand()
        return cls(height=height, width=width)

    def to_tuple(self):
        return self.height, self.width


@dataclass
class SampleTimeParam(object):
    l_start: float = None
    u_start: float = None
    s_p_start: float = None
    e_p_start: float = None
    l_end: float = None
    u_end: float = None
    s_p_end: float = None
    e_p_end: float = None


class Time(object):
    def __init__(self, start=0.0, end=np.inf):
        self.start = start
        self.end = end

    def is_active(self, t):
        return self.start <= t < self.end

    def to_dict(self):
        return {'start': self.start, 'end': self.end}

    @classmethod
    def sample(cls, total_time, param: SampleTimeParam):
        if total_time < 0 and (param is None or (param.l_start is None or param.u_start is None)):
            start = 0.0
        else:
            if param.l_start is None:
                param.l_start = 0.0 if param.s_p_start is None else int(total_time * param.s_p_start)
            if param.u_start is None:
                param.u_start = param.l_start if param.e_p_start is None else int(total_time * param.e_p_start)
            start = param.l_start + (param.u_start - param.l_start) * np.random.rand()

        if total_time < 0 and (param is None or (param.l_end is None or param.u_end is None)):
            end = np.inf
        else:
            if param.l_end is None:
                param.l_end = 0.0 if param.s_p_end is None else int(total_time * param.s_p_end)
            if param.u_end is None:
                param.u_end = total_time if param.e_p_end is None else int(total_time * param.e_p_end)
            end = param.l_end + (param.u_end - param.l_end) * np.random.rand()
        return cls(start=start, end=end)


class SampleDef(object):
    def __init__(self, n_l: int, n_u: int, object_class, object_parameters: dict[str, Any],
                 min_distance: float = 0.0):
        self.n_l = n_l
        self.n_u = n_u
        self.min_distance = min_distance
        self.object_class = object_class
        self.object_parameters = object_parameters

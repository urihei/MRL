import matplotlib.colors as m_colors
import numpy as np


def create_agent_name(x: int):
    return f'agent_{x}'


def agent_name_to_number(name: str):
    return int(name.replace('agent_', ''))


colors = list(m_colors.TABLEAU_COLORS.values())


def get_color(name: str):
    n = agent_name_to_number(name)
    return colors[n]


class Location(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def get_distance(self, other: 'Location'):
        return np.sqrt((self.height - other.height) ** 2 + (self.width - other.width) ** 2)

    def to_dict(self):
        return {'height': self.height, 'width': self.width}

    @classmethod
    def sample(cls, l_height, u_height, l_width, u_width):
        height = l_height + (u_height - l_height) * np.random.rand()
        width = l_width + (u_width - l_width) * np.random.rand()
        return cls(height=height, width=width)


class Time(object):
    def __init__(self, start=0.0, end=np.inf):
        self.start = start
        self.end = end

    def is_active(self, t):
        return self.start <= t < self.end

    def to_dict(self):
        return {'start': self.start, 'end': self.end}

    @classmethod
    def sample(cls, l_start, u_start, l_end, u_end):
        start = l_start + (u_start - l_start) * np.random.rand()
        end = l_end + (u_end - l_end) * np.random.rand()
        return cls(start=start, end=end)

import abc

import numpy as np
from gymnasium import spaces
from matplotlib.axes import Axes
import torch.nn as nn
import torch as th
import tools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scene import Scene
    from scene_object import LocationObject


class FeatureCNN(nn.Module):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        # noinspection PyTypeChecker
        super().__init__()
        assert features_dim > 0
        self._observation_space = observation_space
        self._features_dim = features_dim
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class State(abc.ABC):
    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def get_state(self, scene: 'Scene', center: tuple[float, float]):
        pass

    @abc.abstractmethod
    def get_state_size(self):
        pass

    @abc.abstractmethod
    def update(self, scene: 'Scene'):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def render(self, ax: Axes):
        pass

    @staticmethod
    def get_policy_kwargs():
        return {}

    @abc.abstractmethod
    def set_agents(self, locations: list[tuple[float, float]]):
        pass


class BaseState(State):

    def __init__(self, height_factor, width_factor, state_height, state_width, n_levels):
        super().__init__()
        # Level 0 agents locations
        self.size = [0, 0, n_levels + 1]
        self.ar = None
        self.height_factor: float = height_factor
        self.width_factor: float = width_factor
        self.state_size = [n_levels + 1, state_height, state_width]
        self.state = None
        self.scene_hash = -1
        self.agent_location = None

    def update(self, scene: 'Scene'):
        if scene.active_hash == self.scene_hash:
            return
        self.size[0] = int(scene.s_height * self.height_factor)
        self.size[1] = int(scene.s_width * self.width_factor)
        if self.ar is None:
            self.ar = np.zeros(self.size, dtype=float)
        for so in scene.scene_objects:
            state_level = getattr(so, 'state_level', -1)
            if state_level >= 0:
                so: LocationObject
                #     if so.state_level + 2 > self.size[2]:
                #         ar = np.zeros((self.size[0], self.size[1], so.state_level + 2), dtype=float)
                #         if self.ar is not None:
                #             ar[:, :, :self.size[2]] = self.ar
                #         self.size[2] = so.state_level + 2
                #         self.ar = ar
                locations = self.get_location(so)
                self.ar[locations[:, 0].astype(int), locations[:, 1].astype(int), state_level + 1] = so.state_value
        self.scene_hash = scene.active_hash

    def get_state(self, scene: 'Scene', center: tuple[float, float]):
        if self.state is None or self.state.shape != (self.size[2], self.state_size[1], self.state_size[2]):
            self.state = np.zeros((self.size[2], self.state_size[1], self.state_size[2]), dtype=float)
            self.state_size[0] = self.size[2]
        else:
            self.state.fill(0)
        self.update(scene)
        center = self.localize(*center)
        center = int(center[0]), int(center[1])
        size_h, size_w = self.state_size[1] // 2, self.state_size[2] // 2
        if center[0] - size_h >= 0:
            s_h = center[0] - size_h
            out_h = 0
        else:
            s_h = 0
            out_h = size_h - center[0]
        e_h = min(self.ar.shape[0], center[0] + size_h)
        if center[1] - size_w >= 0:
            s_w = center[1] - size_w
            out_w = 0
        else:
            s_w = 0
            out_w = size_w - center[1]
        e_w = min(self.ar.shape[1], center[1] + size_w)
        out_h_e, out_w_e = out_h + e_h - s_h, out_w + e_w - s_w
        self.state[:, out_h: out_h_e, out_w: out_w_e] = np.moveaxis(self.ar[s_h: e_h, s_w: e_w, :], 2, 0)
        return self.state

    def reset(self):
        self.ar.fill(0)
        self.scene_hash = -1

    def render(self, ax: Axes):
        a = np.zeros((self.size[0], self.size[1]))
        for i in range(self.size[2]):
            a += 2 ** i * (self.ar[:, :, i] > 0)
        ax.imshow(a)

    def get_location(self, so: 'LocationObject'):
        locations = np.array(so.get_location(self.localize))
        return locations

    def localize(self, h: float, w: float):
        return (self.height_factor * h), (self.width_factor * w)

    def localize_array(self, locations: np.ndarray):
        locations[:, 0] = (self.height_factor * locations[:, 0])
        locations[:, 1] = (self.width_factor * locations[:, 1])
        return locations

    def set_agents(self, locations: list[tuple[float, float]], colors: list[int] = None):
        if self.agent_location is not None:
            self.ar[self.agent_location[:, 0].astype(int), self.agent_location[:, 1].astype(int), 0] = 0

        if colors is None:
            colors = np.ones(len(locations))
        else:
            colors = np.array(colors)
        a = np.array(locations)
        a = self.localize_array(a)
        self.agent_location = a
        self.ar[a[:, 0].astype(int), a[:, 1].astype(int), 0] = colors

    @staticmethod
    def get_policy_kwargs():
        return dict(features_extractor_class=FeatureCNN, features_extractor_kwargs=dict(features_dim=512))

    def get_state_size(self):
        return self.state_size

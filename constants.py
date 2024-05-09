import os
from pathlib import Path

import yaml


class Events(object):
    out_of_bound = 0
    complete_task = 1


class RunArgs(object):
    def __init__(self, device=-1, n_cpu=0, n_env=-1, gamma=0.98, learning_rate=0.0003, check_point_freq=int(1e6),
                 train_steps=int(1e7), n_steps=128, batch_size=4096, n_epochs=4, vf_coef=0.5, ent_coef=0.05,
                 clip_range=0.2, gae_lambda=0.9):
        self.device = self.get_device(device=device)
        self.n_cpu = n_cpu if os.name != 'nt' else 0
        self.n_env = n_env
        self.gamma = gamma if gamma > 0 else 0.98
        self.learning_rate = learning_rate if learning_rate > 0 else 0.0003
        self.check_point_freq = check_point_freq if check_point_freq > 0 else 1000000
        self.train_steps = train_steps
        self.n_steps = n_steps if n_steps > 0 else 128
        self.batch_size = batch_size if batch_size > 0 else 1536
        self.n_epochs = n_epochs if n_epochs > 0 else 4
        self.ent_coef = ent_coef if ent_coef > 0 else 0.05
        self.vf_coef = vf_coef if vf_coef > 0 else 0.5
        self.clip_range = clip_range if clip_range > 0 else 0.2
        self.gae_lambda = gae_lambda if gae_lambda > 0 else 0.9

    @staticmethod
    def get_device(device):
        import torch
        device_code = device
        if device_code < 0:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda', index=device_code)
        return device

    @classmethod
    def load(cls, file_path: Path = None):
        if not (file_path and file_path.exists()):
            return cls()
        with file_path.open('r') as f:
            d = yaml.safe_load(f)
        return cls(**d)

from pathlib import Path
from typing import Dict, Any
import yaml

import scene_object
from scene import Scene
import matplotlib.pyplot as plt


class SceneGenerator(object):
    def __init__(self, sample_dict: Dict[str, Any], n_agents,
                 number_of_scene: int = -1, save_path: str = None, max_time=float('inf')):
        self.number_of_scene = number_of_scene
        self.save_path = Path(save_path) if save_path is not None else None
        self.sample_dict = sample_dict
        self.n_agents = n_agents
        self.max_time = max_time
        size_dict = self.sample_dict.get('size', {})
        self.height_s = size_dict.get('height', 100)
        self.width_s = size_dict.get('width', 100)
        self.sample_list = self.get_sample_list()

    def generator(self):
        count = self.number_of_scene + 1
        scene_counter = 0
        while count >= 0:
            scene = Scene.sample(self.height_s, self.width_s, self.sample_list)
            if self.save_path is not None:
                scene_dict = scene.to_dict()
                out_file = self.save_path / f"{scene_counter:04}.yaml"
                with out_file.open('w') as f:
                    yaml.dump(scene_dict, f)
                scene_counter += 1
            yield scene
            count -= self.number_of_scene > 0

    def sample(self):
        return Scene.sample(self.height_s, self.width_s, self.sample_list)

    def get_sample_list(self):
        objects = self.sample_dict.get('objects', {})
        sample_list = []
        for object_name, object_dict in objects.items():
            object_class = scene_object.all_objects[object_name]
            sample_list.append(object_class.dict_to_SampleDef(cfg=object_dict, number_of_agents=self.n_agents))
        return sample_list


class SceneLoader(object):

    def __init__(self, scene_path):
        self.scene_path = Path(scene_path)
        if self.scene_path.is_file():
            self.scene_files = [self.scene_path]
        elif self.scene_path.is_dir():
            self.scene_files = list(self.scene_path.rglob("*.yaml"))
        else:
            raise ValueError(f"Invalid scene_path: {self.scene_path}")

    def generator(self):
        for scene_file in self.scene_files:
            yield self.get_object(scene_file)

    def sample(self):
        return self.get_object(self.scene_files[0])

    @staticmethod
    def get_object(scene_file):
        with scene_file.open("r") as f:
            scene_dict = yaml.safe_load(f)
            atc_scene = Scene.load(scene_dict)
            return atc_scene


def main():
    script_dir = Path(__file__).parent
    scene_yaml = script_dir / "scene_config.yaml"
    n_agents = 1
    number_of_scene = 10
    with scene_yaml.open() as f:
        scene_dict = yaml.safe_load(f)
        obj = SceneGenerator(scene_dict, n_agents=n_agents, number_of_scene=number_of_scene)
        for i, scene in enumerate(obj.generator()):
            if i % 1000 == 0:
                print(f'{i}')


if __name__ == "__main__":
    main()

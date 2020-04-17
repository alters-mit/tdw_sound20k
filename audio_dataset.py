from tdw.py_impact import PyImpact, CollisionInfo, ObjectInfo
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.librarian import ModelLibrarian, ModelRecord, MaterialLibrarian, MaterialRecord
from tdw.output_data import OutputData, Environments, Rigidbodies
import itertools
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List, Dict, Tuple
from subprocess import Popen, call
from platform import system
from time import sleep
from json import dumps
from abc import ABC, abstractmethod
from scenes import CornerSound20k, FloorSound20k
from weighted_collection import WeightedCollection

RNG = np.random.RandomState(0)


class AudioDataset(Controller):
    SCENES = [CornerSound20k, FloorSound20k]

    def __init__(self, total_num: int = 20378, output_dir: Path = Path("D:/audio_dataset"), port: int = 1071):
        assert system() == "Windows", "This controller only works in Windows."

        self.output_dir = output_dir
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        self.total_num = total_num

        object_info_dict = PyImpact.get_object_info()
        self.object_info: List[ObjectInfo] = []
        for name in object_info_dict:
            if object_info_dict[name].mass < 10:
                self.object_info.append(object_info_dict[name])

        self.num_objects_per_trial = WeightedCollection(int)
        self.num_objects_per_trial.add_many({1: 1,
                                             2: 14,
                                             3: 41,
                                             4: 32,
                                             5: 11,
                                             6: 1})
        super().__init__(port=port)

    def trial(self) -> None:
        scene = AudioDataset.SCENES[RNG.randint(0, len(AudioDataset.SCENES))]()
        self.communicate(scene.get_commands(self))
        center = scene.get_center(self)

        obj = self.object_info[RNG.randint(0, len(self.object_info))]
        max_y = scene.get_max_y()
        y = RNG.uniform(max_y - 0.5, max_y)
        o_id = 0
        resp = self.communicate([self.get_add_object(obj.name, object_id=o_id, library=obj.library,
                                                     position={"x": center["x"], "y": y, "z": center["z"]}),
                                 {"$type": "set_mass",
                                  "id": o_id,
                                  "mass": obj.mass},
                                 {"$type": "set_physic_material",
                                  "id": o_id,
                                  "bounciness": obj.bounciness,
                                  "static_friction": 0.1,
                                  "dynamic_friction": 0.8},
                                 {"$type": "rotate_object_by",
                                  "angle": RNG.uniform(0, 20),
                                  "id": o_id,
                                  "axis": "pitch",
                                  "is_world": True},
                                 {"$type": "apply_force_magnitude_to_object",
                                  "magnitude": RNG.uniform(0, 2),
                                  "id": o_id},
                                 {"$type": "send_rigidbodies",
                                  "frequency": "always"}])
        done = False
        while not done:
            rigidbodies: Optional[Rigidbodies] = None
            for r in resp[:-1]:
                r_id = OutputData.get_data_type_id(r)
                if r_id == "rigi":
                    rigidbodies = Rigidbodies(r)
            for i in range(rigidbodies.get_num()):
                if rigidbodies.get_id(i) == o_id:
                    done = rigidbodies.get_sleeping(i)
            if not done:
                resp = self.communicate([])
        self.communicate({"$type": "destroy_all_objects"})

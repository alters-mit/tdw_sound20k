from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.librarian import MaterialLibrarian, MaterialRecord
import numpy as np
from typing import List, Dict
from abc import ABC, abstractmethod
from pathlib import Path

RNG = np.random.RandomState(0)


class _Scene(ABC):
    """
    A recipe to initialize a scene.
    """

    _MODEL_LIBRARY_PATH = str(Path("models/models.json").resolve())

    def __init__(self):
        # A list of object IDs for the scene objects and the model names.
        self.object_ids: Dict[int, str] = {}

    @abstractmethod
    def get_commands(self, c: Controller) -> List[dict]:
        """
        :param c: The controller.

        :return: A list of commands to initialize a scene.
        """

        raise Exception()

    @abstractmethod
    def get_center(self, c: Controller) -> Dict[str, float]:
        """
        :param c: The controller.

        :return: The "center" of the scene, as a Vector3 dictionary.
        """

        raise Exception()

    @abstractmethod
    def get_max_y(self) -> float:
        """
        :return: The maximum y value for the camera (avatar) and starting height for an object.
        """

        raise Exception()


class _ProcGenRoom(_Scene):
    """
    Initialize the ProcGen room.
    """

    def get_commands(self, c: Controller) -> List[dict]:
        # Initialize the material library.
        if c.material_librarian is None:
            c.material_librarian = MaterialLibrarian()
        material: MaterialRecord = self._get_floor_material(c.material_librarian)
        # Load the scene and an empty room.
        # Add the material to the floor.
        return [{"$type": "load_scene"},
                TDWUtils.create_empty_room(12, 12),
                {"$type": "add_material",
                 "name": material.name,
                 "url": material.get_url()},
                {"$type": "set_proc_gen_floor_material",
                 "name": material.name},
                {"$type": "set_proc_gen_floor_texture_scale",
                 "scale": {"x": 8, "y": 8}}]

    @abstractmethod
    def _get_floor_material(self, lib: MaterialLibrarian) -> MaterialRecord:
        """
        :param lib: The material library.

        :return: A visual material to apply to the floor.
        """

        raise Exception()

    def get_max_y(self) -> float:
        return 2


class FloorSound20k(_ProcGenRoom):
    """
    Initialize a scene with a floor that mimics a Sound20k scene (the floor is always wood).
    """

    def _get_floor_material(self, lib: MaterialLibrarian) -> MaterialRecord:
        wood_materials = lib.get_all_materials_of_type("Wood")
        return wood_materials[RNG.randint(0, len(wood_materials))]

    def get_center(self, c: Controller) -> Dict[str, float]:
        return {"x": 0, "y": 0, "z": 0}


class CornerSound20k(_ProcGenRoom):
    """
    Initialize a scene with a floor that mimics a Sound20k scene (the floor is always wood).
    The "center" is offset to a corner.
    """

    def get_commands(self, c: Controller) -> List[dict]:
        commands = super().get_commands(c)
        # Set the wall material too.
        mat_name = ""
        for cmd in commands:
            if cmd["$type"] == "set_proc_gen_floor_material":
                mat_name = cmd["name"]
                break
        assert mat_name != ""
        commands.extend([{"$type": "set_proc_gen_walls_material",
                          "name": mat_name},
                         {"$type": "set_proc_gen_walls_texture_scale",
                          "scale": {"x": 8, "y": 8}}])
        return commands

    def _get_floor_material(self, lib: MaterialLibrarian) -> MaterialRecord:
        wood_materials = lib.get_all_materials_of_type("Wood")
        return wood_materials[RNG.randint(0, len(wood_materials))]

    def get_center(self, c: Controller) -> Dict[str, float]:
        return {"x": 4, "y": 0, "z": 4}


class _FloorWithObject(FloorSound20k):
    """
    Simple Sound20K floor scene with a single object in the center of the room.
    """

    @abstractmethod
    def _get_model_name(self) -> str:
        """
        :return: The name of the model.
        """

        raise Exception()

    @abstractmethod
    def _get_model_scale(self) -> Dict[str, float]:
        """
        :return: The scale of the object as a Vector3.
        """

        raise Exception()

    @abstractmethod
    def _get_library(self) -> str:
        """
        :return: The library .json file path.
        """

        raise Exception()

    def get_commands(self, c: Controller) -> List[dict]:
        model_name = self._get_model_name()
        o_id = c.get_unique_id()
        self.object_ids.update({o_id: model_name})
        commands = super().get_commands(c)
        commands.extend([c.get_add_object(model_name, object_id=o_id, library=self._get_library()),
                         {"$type": "scale_object",
                          "id": o_id,
                          "scale_factor": self._get_model_scale()},
                         {"$type": "set_mass",
                          "id": o_id,
                          "mass": 1000}])
        return commands


class LargeBowl(_FloorWithObject):
    """
    A large ceramic bowl.
    """

    def _get_model_name(self) -> str:
        return "int_kitchen_accessories_le_creuset_bowl_30cm"

    def _get_library(self) -> str:
        return "models_full.json"

    def _get_model_scale(self) -> Dict[str, float]:
        return {"x": 6, "y": 6, "z": 6}


class Ramp(_FloorWithObject):
    """
    A simple ramp.
    """

    def _get_model_name(self) -> str:
        return "ramp_with_platform"

    def _get_model_scale(self) -> Dict[str, float]:
        return {"x": 1, "y": 1, "z": 1}

    def _get_library(self) -> str:
        return "models_special.json"


class RoundTable(_FloorWithObject):
    """
    A large round wooden table.
    """

    def _get_model_name(self) -> str:
        return "enzo_industrial_loft_pine_metal_round_dining_table"

    def _get_library(self) -> str:
        return "models_full.json"

    def _get_model_scale(self) -> Dict[str, float]:
        return {"x": 1, "y": 1, "z": 1}


class StairRamp(_FloorWithObject):
    """
    A simple staircase.
    """

    def _get_model_name(self) -> str:
        return "stair_ramp"

    def _get_model_scale(self) -> Dict[str, float]:
        return {"x": 1, "y": 1, "z": 1}

    def _get_library(self) -> str:
        return _Scene._MODEL_LIBRARY_PATH

    def get_commands(self, c: Controller) -> List[dict]:
        commands = super().get_commands(c)
        commands.append({"$type": "teleport_object",
                         "id": list(self.object_ids.keys())[0],
                         "position": {"x": 0, "y": 0, "z": -0.25}})
        return commands


class UnevenTerrain(_Scene):
    """
    Load an outdoor scene with uneven terrain.
    """

    def get_center(self, c: Controller) -> Dict[str, float]:
        return TDWUtils.VECTOR3_ZERO

    def get_max_y(self) -> float:
        return 4

    def get_commands(self, c: Controller) -> List[dict]:
        return [c.get_add_scene(scene_name="building_site")]

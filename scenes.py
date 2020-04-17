from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.librarian import MaterialLibrarian, MaterialRecord
import numpy as np
from typing import List, Dict
from abc import ABC, abstractmethod

RNG = np.random.RandomState(0)


class _Scene(ABC):
    """
    A recipe to initialize a scene.
    """

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

from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.librarian import ModelLibrarian
from tdw.py_impact import PyImpact, AudioMaterial
from typing import List, Dict, Tuple
from abc import ABC, abstractmethod
from pathlib import Path
from tdw_scene_audio import TDWSceneAudio, Realistic, Unrealistic, Chaos
from tdw_scene_size import TDWSceneSize, StandardSize, SmallSize, RandomSize
from weighted_collection import WeightedCollection
from numpy.random import RandomState
import numpy as np


class Scene(ABC):
    """
    A recipe to initialize a scene.
    """

    _MODEL_LIBRARY_PATH = str(Path("models/models.json").resolve())

    # A list of object IDs for the scene objects and the model names.
    OBJECT_IDS: Dict[int, str] = {}
    _OBJECT_INFO = PyImpact.get_object_info()
    # Append custom data.
    _custom_object_info = PyImpact.get_object_info(Path("models/object_info.csv"))
    for obj in _custom_object_info:
        _OBJECT_INFO.update({obj: _custom_object_info[obj]})

    def set_parameters(self) -> None:
        """
        Set the scene parameters f or the next trial.
        """

        return

    def initialize_scene(self, c: Controller) -> List[dict]:
        """
        Add these commands to the beginning of the list of initialization commands.

        :param c: The controller.

        :return: A list of commands to initialize a scene.
        """

        # Clean up all objects.
        Scene.OBJECT_IDS.clear()

        # Custom commands to initialize the scene.
        commands = self._initialize_scene(c)[:]

        # Send bounds data (for the new objects).
        commands.append({"$type": "send_bounds",
                         "frequency": "once"})

        return commands

    @abstractmethod
    def _initialize_scene(self, c: Controller) -> List[dict]:
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

    @abstractmethod
    def get_scene_id(self) -> int:
        """
        :return: The unique identifier of this scene type.
        """

        raise Exception()

    @staticmethod
    def _init_object(c: Controller, name: str, pos: Dict[str, float], rot: Dict[str, float]) -> List[dict]:
        """
        :param c: The controller.
        :param name: The name of the model.
        :param pos: The initial position of the model.
        :param rot: The initial rotation of the model.

        :return: A list of commands to instantiate an object from ObjectInfo values.
        """

        o_id = c.get_unique_id()
        Scene.OBJECT_IDS.update({o_id: name})
        info = Scene._OBJECT_INFO[name]
        return [c.get_add_object(name,
                                 object_id=o_id,
                                 position=pos,
                                 rotation=rot,
                                 library=info.library),
                {"$type": "set_mass",
                 "id": o_id,
                 "mass": info.mass},
                {"$type": "set_physic_material",
                 "id": o_id,
                 "bounciness": info.bounciness,
                 "static_friction": 0.1,
                 "dynamic_friction": 0.8}]

    @staticmethod
    def get_camera_angles() -> Tuple[float, float]:
        """
        :return: Range of valid camera angles.
        """

        return 0, 360

    @abstractmethod
    def get_surface_material(self) -> AudioMaterial:
        """
        :return: The audio material of the surface.
        """

        raise Exception()


class _ProcGenRoom(Scene, ABC):
    """
    Initialize the ProcGen room.
    """

    def get_max_y(self) -> float:
        return 3.5

    def get_surface_material(self) -> AudioMaterial:
        return AudioMaterial.hardwood

    def _initialize_scene(self, c: Controller) -> List[dict]:
        return []


class FloorSound20k(_ProcGenRoom):
    """
    Initialize a scene with a floor that mimics a Sound20k scene (the floor is always wood).
    """

    def get_center(self, c: Controller) -> Dict[str, float]:
        return {"x": 0, "y": 0, "z": 0}

    def get_scene_id(self) -> int:
        return 0


class CornerSound20k(_ProcGenRoom):
    """
    Initialize a scene with a floor that mimics a Sound20k scene (the floor is always wood).
    The "center" is offset to a corner.
    """

    def get_center(self, c: Controller) -> Dict[str, float]:
        return {"x": 4, "y": 0, "z": 4}

    @staticmethod
    def get_camera_angles() -> Tuple[float, float]:
        return 120, 250

    def get_scene_id(self) -> int:
        return 1


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
    def _get_library(self) -> str:
        """
        :return: The library .json file path.
        """

        raise Exception()

    def _initialize_scene(self, c: Controller) -> List[dict]:
        commands = super()._initialize_scene(c)
        model_name = self._get_model_name()
        o_id = c.get_unique_id()
        Scene.OBJECT_IDS.update({o_id: model_name})
        commands.extend([c.get_add_object(model_name, object_id=o_id, library=self._get_library()),
                         {"$type": "set_mass",
                          "id": o_id,
                          "mass": 1000},
                         {"$type": "set_physic_material",
                          "id": o_id,
                          "bounciness": Scene._OBJECT_INFO[model_name].bounciness,
                          "static_friction": 0.1,
                          "dynamic_friction": 0.8}])
        return commands


class StairRamp(_FloorWithObject):
    """
    A simple staircase.
    """

    def _get_model_name(self) -> str:
        return "stair_ramp"

    def _get_library(self) -> str:
        return Scene._MODEL_LIBRARY_PATH

    def _initialize_scene(self, c: Controller) -> List[dict]:
        commands = super()._initialize_scene(c)
        commands.append({"$type": "teleport_object",
                         "id": list(Scene.OBJECT_IDS.keys())[0],
                         "position": {"x": 0, "y": 0, "z": -0.3}})
        return commands

    def get_scene_id(self) -> int:
        return 2


class RoundTable(_FloorWithObject):
    """
    A large round wooden table.
    """

    def _get_model_name(self) -> str:
        return "enzo_industrial_loft_pine_metal_round_dining_table"

    def _get_library(self) -> str:
        return "models_full.json"

    def get_scene_id(self) -> int:
        return 3


class UnevenTerrain(_FloorWithObject):
    """
    Load an outdoor scene with uneven terrain.
    """

    def _get_model_name(self) -> str:
        return "uneven_terrain"

    def _get_library(self) -> str:
        return Scene._MODEL_LIBRARY_PATH

    def _initialize_scene(self, c: Controller) -> List[dict]:
        commands = super()._initialize_scene(c)
        # Let the object settle.
        commands.append({"$type": "step_physics",
                         "frames": 3})
        return commands

    def get_scene_id(self) -> int:
        return 4


class LargeBowl(_FloorWithObject):
    """
    A large ceramic bowl.
    """

    def _get_model_name(self) -> str:
        return "int_kitchen_accessories_le_creuset_bowl_30cm"

    def _get_library(self) -> str:
        return Scene._MODEL_LIBRARY_PATH

    def get_scene_id(self) -> int:
        return 5


class Ramp(_FloorWithObject):
    """
    A simple ramp.
    """

    def _get_model_name(self) -> str:
        return "ramp_with_platform"

    def _get_library(self) -> str:
        return "models_special.json"

    def get_scene_id(self) -> int:
        return 6


class DeskAndChair(FloorSound20k):
    """
    A desk, a chair, and a shelf with some boxes, facing a wall.
    """

    def get_center(self, c: Controller) -> Dict[str, float]:
        return {"x": 0, "y": 0, "z": 3.8}

    @staticmethod
    def get_camera_angles() -> Tuple[float, float]:
        return 270, 300

    def _initialize_scene(self, c: Controller) -> List[dict]:
        c.model_librarian = ModelLibrarian("models_full.json")
        commands = super()._initialize_scene(c)

        # Add a table, chair, and boxes.
        commands.extend(self._init_object(c, "b05_table_new",
                                          pos={"x": 0, "y": 0, "z": 4.33},
                                          rot=TDWUtils.VECTOR3_ZERO))
        commands.extend(self._init_object(c, "chair_willisau_riale",
                                          pos={"x": 0, "y": 0, "z": 3.7},
                                          rot=TDWUtils.VECTOR3_ZERO))
        commands.extend(self._init_object(c, "iron_box",
                                          pos={"x": 0.13, "y": 0.65, "z": 4.83},
                                          rot=TDWUtils.VECTOR3_ZERO))
        commands.extend(self._init_object(c, "iron_box",
                                          pos={"x": -0.285, "y": 1.342, "z": 4.79},
                                          rot={"x": 90, "y": 0, "z": 0}))
        # Add a shelf with a custom scale.
        shelf_id = c.get_unique_id()
        shelf_name = "metal_lab_shelf"
        Scene.OBJECT_IDS.update({shelf_id: shelf_name})
        commands.extend([c.get_add_object(shelf_name,
                         object_id=shelf_id,
                         rotation={"x": 0, "y": -90, "z": 0},
                         position={"x": 0, "y": 0, "z": 4.93}),
                         {"$type": "set_mass",
                          "id": shelf_id,
                          "mass": 400},
                         {"$type": "set_physic_material",
                          "id": shelf_id,
                          "bounciness": Scene._OBJECT_INFO[shelf_name].bounciness,
                          "static_friction": 0.1,
                          "dynamic_friction": 0.8},
                         {"$type": "scale_object",
                          "id": shelf_id,
                          "scale_factor": {"x": 1, "y": 1.5, "z": 1.8}}])
        return commands

    def get_scene_id(self) -> int:
        return 7


class DiningTableAndChairs(FloorSound20k):
    """
    A dining table with 8 chairs around it.
    """

    def _initialize_scene(self, c: Controller) -> List[dict]:
        c.model_librarian = ModelLibrarian("models_full.json")
        # Initialize the scene.
        commands = super()._initialize_scene(c)
        chair_name = "brown_leather_dining_chair"
        # Create the the table.
        commands.extend(self._init_object(c=c,
                                          name="quatre_dining_table",
                                          pos=TDWUtils.VECTOR3_ZERO,
                                          rot=TDWUtils.VECTOR3_ZERO))
        # Create 8 chairs around the table.
        commands.extend(self._init_object(c=c,
                                          name=chair_name,
                                          pos={"x": 0, "y": 0, "z": -1.55},
                                          rot={"x": 0, "y": 0, "z": 0}))
        commands.extend(self._init_object(c=c,
                                          name=chair_name,
                                          pos={"x": 0, "y": 0, "z": 1.55},
                                          rot={"x": 0, "y": 180, "z": 0}))
        commands.extend(self._init_object(c=c,
                                          name=chair_name,
                                          pos={"x": -1, "y": 0, "z": -0.85},
                                          rot={"x": 0, "y": 90, "z": 0}))
        commands.extend(self._init_object(c=c,
                                          name=chair_name,
                                          pos={"x": -1, "y": 0, "z": 0},
                                          rot={"x": 0, "y": 90, "z": 0}))
        commands.extend(self._init_object(c=c,
                                          name=chair_name,
                                          pos={"x": -1, "y": 0, "z": 0.85},
                                          rot={"x": 0, "y": 90, "z": 0}))
        commands.extend(self._init_object(c=c,
                                          name=chair_name,
                                          pos={"x": 1, "y": 0, "z": -0.85},
                                          rot={"x": 0, "y": -90, "z": 0}))
        commands.extend(self._init_object(c=c,
                                          name=chair_name,
                                          pos={"x": 1, "y": 0, "z": 0},
                                          rot={"x": 0, "y": -90, "z": 0}))
        commands.extend(self._init_object(c=c,
                                          name=chair_name,
                                          pos={"x": 1, "y": 0, "z": 0.85},
                                          rot={"x": 0, "y": -90, "z": 0}))

        return commands

    def get_scene_id(self) -> int:
        return 8


class TDWScene(Scene):
    """
    A "TDW" scene has randomized room sizes and ResonanceAudio surface materials.
    These parameters are all chosen "weighted-randomly"; most of the time they will be "plausible",
    but sometimes they will be "chaotic".
    """

    # Tend towards "plausible" reverb spaces; note that none of them will ever be Sound20K
    _REVERB_PARAMETERS = WeightedCollection(TDWSceneAudio)
    _REVERB_PARAMETERS.add_many({Chaos: 1,
                                 Unrealistic: 3,
                                 Realistic: 6})

    _ROOM_SIZES = WeightedCollection(TDWSceneSize)
    _ROOM_SIZES.add_many({StandardSize: 5,
                          SmallSize: 3,
                          RandomSize: 2})
    _RNG = RandomState(0)

    _SCENE_INDEX_SHIFT = 8

    def __init__(self):
        self._reverb = TDWScene._REVERB_PARAMETERS.get()
        self._scene_id = 0

    def set_parameters(self) -> None:
        self._reverb = TDWScene._REVERB_PARAMETERS.get()
        self._scene_id = self._reverb.audio_id << TDWScene._SCENE_INDEX_SHIFT

    def get_scene_id(self) -> int:
        return self._scene_id

    def _initialize_scene(self, c: Controller) -> List[dict]:
        # Get a weighted-random room size.
        width, length = TDWScene._ROOM_SIZES.get().get_size()
        return [{"$type": "load_scene"},
                TDWUtils.create_empty_room(width, length),
                {"$type": "destroy_all_objects"},
                self._reverb.get_command(),
                {"$type": "create_avatar",
                 "type": "A_Img_Caps_Kinematic",
                 "id": "a"},
                {"$type": "add_environ_audio_sensor"},
                {"$type": "toggle_image_sensor"}]

    def get_surface_material(self) -> AudioMaterial:
        return self._reverb.get_audio_material()

    def get_center(self, c: Controller) -> Dict[str, float]:
        # Slightly randomize the center.
        return {"x": TDWScene._RNG.uniform(-0.075, 0.075),
                "y": 0,
                "z": TDWScene._RNG.uniform(-0.075, 0.075)}

    def get_max_y(self) -> float:
        return 3.5


class Marbles(TDWScene):
    """
    A scene with some marbles on the floor.
    """

    def set_parameters(self) -> None:
        super().set_parameters()
        self._scene_id += 1024 << TDWScene._SCENE_INDEX_SHIFT

    def _initialize_scene(self, c: Controller) -> List[dict]:
        commands = super()._initialize_scene(c)
        r = 0.6
        # Get all points within the circle defined by the radius.
        p0 = np.array((0, 0))
        for x in np.arange(-r, r, 0.2):
            for z in np.arange(-r, r, 0.2):
                p1 = np.array((x, z))
                dist = np.linalg.norm(p0 - p1)
                if dist < r:
                    commands.extend(self._init_object(c, name="marble",
                                                      pos={"x": x, "y": 0, "z": z},
                                                      rot=TDWUtils.VECTOR3_ZERO))
        return commands


def get_sound20k_scenes() -> List[Scene]:
    """
    :return: A list of scenes, based on their frequency in the original Sound20K dataset.
    """

    return [FloorSound20k(), CornerSound20k(), StairRamp(), RoundTable(), UnevenTerrain(), LargeBowl(), Ramp(),
            DeskAndChair(), DiningTableAndChairs()]


def get_tdw_scenes() -> List[Scene]:
    """
    :return: A list of TDW scene types.
    """

    tdw_scenes = [TDWScene(), Marbles()]
    scenes = get_sound20k_scenes()[:]
    for i in range(3):
        scenes.extend(tdw_scenes[:])
    return scenes

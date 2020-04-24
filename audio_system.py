from typing import List, Optional
from abc import ABC, abstractmethod


class AudioSystem(ABC):
    @abstractmethod
    def init_audio(self) -> Optional[dict]:
        """
        :return: The command to initialize audio in the scene.
        """

        raise Exception()

    @abstractmethod
    def add_audio_sensor(self, avatar_id: str = "a") -> dict:
        """
        :param avatar_id: The ID of the avatar.

        :return: The command to add an audio sensor to the avatar.
        """

        raise Exception()

    @abstractmethod
    def play_audio_data(self) -> bool:
        """
        :return: True if the system uses `play_audio_data`, False if it uses `play_point_source_data`.
        """

        raise Exception()


class StandardAudio(AudioSystem):
    """
    Standard Unity Engine audio system.
    """

    def add_audio_sensor(self, avatar_id: str = "a") -> dict:
        return {"$type": "add_audio_sensor",
                "avatar_id": avatar_id}

    def play_audio_data(self) -> bool:
        return True

    def init_audio(self) -> Optional[dict]:
        return None


class ResonanceAudio(AudioSystem):
    """
    Resonance Audio system.
    """

    def __init__(self, env_id: int = 0, floor: str = "parquet", ceiling: str = "acousticTile", walls: str = "smoothPlaster"):
        """
        :param env_id: Add a reverb space to this environment (room).
        :param floor: The floor reverb material.
        :param ceiling: The ceiling reverb material.
        :param walls: The walls reverb material.
        """

        self.env_id = env_id
        self.floor = floor
        self.walls = walls
        self.ceiling = ceiling

    def add_audio_sensor(self, avatar_id: str = "a") -> dict:
        return {"$type": "add_environ_audio_sensor",
                "avatar_id": avatar_id}

    def init_audio(self) -> Optional[dict]:
        return {"$type": "set_reverb_space_simple",
                "env_id": self.env_id,
                "reverb_floor_material": self.floor,
                "reverb_ceiling_material": self.ceiling,
                "reverb_front_wall_material": self.walls,
                "reverb_back_wall_material": self.walls,
                "reverb_left_wall_material": self.walls,
                "reverb_right_wall_material": self.walls}

    def play_audio_data(self) -> bool:
        return False

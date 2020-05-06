from abc import ABC, abstractmethod
from enum import Enum
from numpy.random import RandomState
from tdw.py_impact import AudioMaterial


class SurfaceMaterial(Enum):
    """
    Resonance Audio surface materials.
    """

    smoothPlaster = 0
    roughPlaster = 1
    glass = 2
    parquet = 3
    marble = 4
    grass = 5
    concrete = 6
    brick = 7
    tile = 8
    acousticTile = 9


class _ReverbSpaceParameters:
    """
    Parameters for a Resonance Audio reverb space
    """

    def __init__(self, floor: SurfaceMaterial,
                 ceiling: SurfaceMaterial,
                 front_wall: SurfaceMaterial,
                 back_wall: SurfaceMaterial,
                 left_wall: SurfaceMaterial,
                 right_wall: SurfaceMaterial):
        """
        :param floor: The floor material.
        :param ceiling: The ceiling material.
        :param front_wall: The front wall material.
        :param back_wall: The back wall material.
        :param left_wall: The left wall material.
        :param right_wall: The right wall material.
        """

        self.floor = floor
        self.ceiling = ceiling
        self.front_wall = front_wall
        self.back_wall = back_wall
        self.left_wall = left_wall
        self.right_wall = right_wall


class TDWSceneAudio(ABC):
    """
    Audio system for a "TDW" scene.
    """

    RNG = RandomState(0)

    @abstractmethod
    def _get_reverb_space_parameters(self) -> _ReverbSpaceParameters:
        """
        :return: The parameters used to create the reverb space.
        """

        raise Exception()

    @abstractmethod
    def get_audio_material(self) -> AudioMaterial:
        """
        :return: The surface audio material (for PyImpact).
        """

        raise Exception()

    def get_command(self) -> dict:
        """
        :return: The command to create the reverb space.
        """

        parameters = self._get_reverb_space_parameters()
        return {"$type": "set_reverb_space_simple",
                "env_id": 0,
                "reverb_floor_material": parameters.floor.name,
                "reverb_ceiling_material": parameters.ceiling.name,
                "reverb_front_wall_material": parameters.front_wall.name,
                "reverb_back_wall_material": parameters.back_wall.name,
                "reverb_left_wall_material": parameters.left_wall.name,
                "reverb_right_wall_material": parameters.right_wall.name}


class Realistic(TDWSceneAudio):
    """
    A "realistic" reverb space with random surface materials.
    All of the surface materials are "plausible" (e.g. walls are never made out of grass).
    The walls all have the same surface material.
    """

    _FLOOR = [SurfaceMaterial.concrete,
              SurfaceMaterial.grass,
              SurfaceMaterial.marble,
              SurfaceMaterial.tile,
              SurfaceMaterial.brick,
              SurfaceMaterial.parquet]
    _WALL = [SurfaceMaterial.brick,
             SurfaceMaterial.concrete,
             SurfaceMaterial.acousticTile,
             SurfaceMaterial.roughPlaster,
             SurfaceMaterial.smoothPlaster]
    _CEILING = [SurfaceMaterial.brick,
                SurfaceMaterial.concrete,
                SurfaceMaterial.roughPlaster,
                SurfaceMaterial.smoothPlaster,
                SurfaceMaterial.acousticTile]

    _AUDIO_MATERIALS = {SurfaceMaterial.concrete: AudioMaterial.ceramic,
                        SurfaceMaterial.grass: AudioMaterial.cardboard,
                        SurfaceMaterial.marble: AudioMaterial.metal,
                        SurfaceMaterial.tile: AudioMaterial.ceramic,
                        SurfaceMaterial.brick: AudioMaterial.wood,
                        SurfaceMaterial.parquet: AudioMaterial.hardwood}

    def __init__(self):
        self._floor = Realistic._FLOOR[TDWSceneAudio.RNG.randint(0, len(Realistic._FLOOR))]

    def _get_reverb_space_parameters(self) -> _ReverbSpaceParameters:
        wall = Realistic._WALL[TDWSceneAudio.RNG.randint(0, len(Realistic._WALL))]
        ceiling = Realistic._CEILING[TDWSceneAudio.RNG.randint(0, len(Realistic._CEILING))]
        return _ReverbSpaceParameters(floor=self._floor, ceiling=ceiling,
                                      front_wall=wall, back_wall=wall, left_wall=wall, right_wall=wall)

    def get_audio_material(self) -> AudioMaterial:
        # A PyImpact audio material that corresponds to the ResonanceAudio surface material.
        return Realistic._AUDIO_MATERIALS[self._floor]


class Unrealistic(Realistic):
    """
    An "unrealistic" reverb space.
    All of the surface materials are plausible, but each of the walls can have a different material.
    """

    def _get_reverb_space_parameters(self) -> _ReverbSpaceParameters:
        return _ReverbSpaceParameters(floor=Realistic._FLOOR[TDWSceneAudio.RNG.randint(0, len(Realistic._FLOOR))],
                                      ceiling=Realistic._CEILING[TDWSceneAudio.RNG.randint(0, len(Realistic._CEILING))],
                                      front_wall=Realistic._WALL[TDWSceneAudio.RNG.randint(0, len(Realistic._WALL))],
                                      back_wall=Realistic._WALL[TDWSceneAudio.RNG.randint(0, len(Realistic._WALL))],
                                      left_wall=Realistic._WALL[TDWSceneAudio.RNG.randint(0, len(Realistic._WALL))],
                                      right_wall=Realistic._WALL[TDWSceneAudio.RNG.randint(0, len(Realistic._WALL))])


class Chaos(TDWSceneAudio):
    """
    A "chaotic" reverb space. All materials are random.
     The PyImpact material might not match the ResonanceAudio floor material.
    """

    _MATERIALS = [m for m in SurfaceMaterial]
    _PY_IMPACT_MATERIALS = [m for m in AudioMaterial]

    def _get_reverb_space_parameters(self) -> _ReverbSpaceParameters:
        return _ReverbSpaceParameters(floor=Chaos._MATERIALS[TDWSceneAudio.RNG.randint(0, len(Chaos._MATERIALS))],
                                      ceiling=Chaos._MATERIALS[TDWSceneAudio.RNG.randint(0, len(Chaos._MATERIALS))],
                                      front_wall=Chaos._MATERIALS[TDWSceneAudio.RNG.randint(0, len(Chaos._MATERIALS))],
                                      back_wall=Chaos._MATERIALS[TDWSceneAudio.RNG.randint(0, len(Chaos._MATERIALS))],
                                      left_wall=Chaos._MATERIALS[TDWSceneAudio.RNG.randint(0, len(Chaos._MATERIALS))],
                                      right_wall=Chaos._MATERIALS[TDWSceneAudio.RNG.randint(0, len(Chaos._MATERIALS))])

    def get_audio_material(self) -> AudioMaterial:
        return Chaos._PY_IMPACT_MATERIALS[TDWSceneAudio.RNG.randint(0, len(Chaos._PY_IMPACT_MATERIALS))]


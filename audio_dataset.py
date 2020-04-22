from tdw.py_impact import CollisionInfo
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.py_impact import PyImpact, AudioMaterial
from tdw.output_data import Bounds
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from platform import system
from scenes import SOUND20K

RNG = np.random.RandomState(0)


class AudioDataset(Controller):
    def __init__(self, total_num: int = 20378, output_dir: Path = Path("D:/audio_dataset"), port: int = 1071):
        assert system() == "Windows", "This controller only works in Windows."

        self.output_dir = output_dir
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        self.total_num = total_num
        self.py_impact = PyImpact()

        self.object_info = PyImpact.get_object_info()

        # Resonance audio
        # TODO is it allowed? Get an init and a command.

        super().__init__(port=port)

        # Global settings.
        self.communicate([{"$type": "set_screen_size",
                           "width": 128,
                           "height": 128},
                          {"$type": "set_post_process",
                           "value": False},
                          {"$type": "set_render_quality",
                           "render_quality": 1},
                          {"$type": "set_shadows",
                           "value": False}])

    def trial(self) -> None:
        scene = SOUND20K[RNG.randint(0, len(SOUND20K))]()
        init_commands = scene.get_commands(self)
        init_commands.append({"$type": "send_bounds",
                              "frequency": "once"})
        resp = self.communicate(init_commands)
        center = scene.get_center(self)

        obj_name = list(self.object_info.keys())[RNG.randint(0, len(self.object_info))]
        obj_info = self.object_info[obj_name]
        max_y = scene.get_max_y()
        o_x = RNG.uniform(center["x"] - 0.05, center["x"] + 0.05)
        o_y = RNG.uniform(max_y - 0.5, max_y)
        o_z = RNG.uniform(center["z"] - 0.05, center["z"] + 0.05)
        o_id = 0
        # Create the object and apply a force.
        commands = [self.get_add_object(obj_name, object_id=o_id, library=obj_info.library,
                                        position={"x": o_x, "y": o_y, "z": o_z}),
                    {"$type": "set_mass",
                     "id": o_id,
                     "mass": obj_info.mass},
                    {"$type": "set_physic_material",
                     "id": o_id,
                     "bounciness": obj_info.bounciness,
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
                     "frequency": "always"},
                    {"$type": "send_collisions",
                     "enter": True,
                     "exit": False,
                     "stay": False,
                     "collision_types": ["obj", "env"]}]
        # Parse bounds data to get the centroid of all objects currently in the scene.
        bounds = Bounds(resp[0])
        if bounds.get_num() == 0:
            look_at = {"x": center["x"], "y": 0.1, "z": center["z"]}
        else:
            centers = []
            for i in range(bounds.get_num()):
                centers.append(bounds.get_center(i))
            centers_x, centers_y, centers_z = zip(*centers)
            centers_len = len(centers_x)
            look_at = {"x": sum(centers_x) / centers_len,
                       "y": sum(centers_y) / centers_len,
                       "z": sum(centers_z) / centers_len}
        # Add the avatar.
        r = RNG.uniform(2.3, 2.6)
        a_x = center["x"] + r
        a_y = RNG.uniform(1.8, 2.2)
        a_z = center["y"] + r
        cam_angle_min, cam_angle_max = scene.get_camera_angles()
        theta = RNG.uniform(cam_angle_min, cam_angle_max)
        rad = np.radians(theta)
        a_x = np.cos(rad) * (a_x - center["x"]) - np.sin(rad) * (a_z - center["z"]) + center["x"]
        a_z = np.sin(rad) * (a_x - center["x"]) + np.cos(rad) * (a_z - center["z"]) + center["z"]
        commands.extend(TDWUtils.create_avatar(position={"x": a_x, "y": a_y, "z": a_z},
                                               look_at=look_at))
        resp = self.communicate(commands)
        done = False
        collision_infos: Dict[int, CollisionInfo] = {}
        while not done:
            commands = []
            collisions, environment_collisions, rigidbodies = PyImpact.get_collisions(resp)
            for collision in collisions:
                if PyImpact.is_valid_collision(collision):
                    # Get the audio material and amp.
                    collider_id = collision.get_collider_id()
                    collider_material, collider_amp = self._get_object_info(collider_id, scene.object_ids, obj_name)
                    collidee_id = collision.get_collider_id()
                    collidee_material, collidee_amp = self._get_object_info(collidee_id, scene.object_ids, obj_name)
                    if collidee_id not in collision_infos:
                        collision_infos.update({collidee_id: CollisionInfo(amp=collidee_amp)})
                    impact_sound_command, collision_infos[collidee_id] = self.py_impact.get_impact_sound_command(
                        collision=collision,
                        rigidbodies=rigidbodies,
                        id1=collider_id,
                        mat1=collider_material.name,
                        id2=collidee_id,
                        mat2=collidee_material.name,
                        amp2re1=collider_amp / collidee_amp,
                        coll_info=collision_infos[collidee_id],
                        target_id=collidee_id)
                    commands.append(impact_sound_command)
            # Handle environment collision.
            for collision in environment_collisions:
                collider_id = collision.get_object_id()
                collider_material, collider_amp = self._get_object_info(collider_id, scene.object_ids, obj_name)
                surface_material = scene.get_surface_material()
                if collider_id not in collision_infos:
                    collision_infos.update({collider_id: CollisionInfo()})
                impact_sound_command, collision_infos[collider_id] = self.py_impact.get_impact_sound_command(
                    collision=collision,
                    rigidbodies=rigidbodies,
                    id1=collider_id,
                    mat1=collider_material.name,
                    id2=-1,
                    mat2=surface_material.name,
                    amp2re1=collider_amp / 0.5,
                    coll_info=collision_infos[collider_id],
                    target_id=collider_id)
                commands.append(impact_sound_command)
            # If there were no collisions, check for movement.
            if len(commands) == 0:
                done = True
                for i in range(rigidbodies.get_num()):
                    if not rigidbodies.get_sleeping(i):
                        done = False
                        break
            # Continue the trial.
            if not done:
                self.communicate(commands)

        self.communicate({"$type": "destroy_all_objects"})

    def _get_object_info(self, o_id: int, object_ids: Dict[int, str], drop_name: str) -> Tuple[AudioMaterial, float]:
        """
        :param o_id: The object ID.
        :param object_ids: The scene object IDs.
        :param drop_name: The name of the dropped object.

        :return: The audio material and amp associated with the object.
        """

        if o_id in object_ids:
            return self.object_info[object_ids[o_id]].material, self.object_info[object_ids[o_id]].amp
        else:
            return self.object_info[drop_name].material, self.object_info[drop_name].amp


AudioDataset().trial()
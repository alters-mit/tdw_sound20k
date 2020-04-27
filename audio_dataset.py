from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.py_impact import PyImpact, AudioMaterial
from tdw.output_data import Bounds
from tdw.librarian import ModelLibrarian
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from platform import system
from scenes import get_sound20k_scenes, Scene
from subprocess import Popen, call, check_output
from json import loads
from distutils import dir_util
from os import devnull
from tqdm import tqdm
import re

RNG = np.random.RandomState(0)


class AudioDataset(Controller):
    def __init__(self, total_num: int = 20378, output_dir: Path = Path("D:/audio_dataset"), port: int = 1071):
        """
        :param total_num: The total number of files to generate.
        :param output_dir: The output directory for the files.
        :param port: The socket port.
        """
        
        assert system() == "Windows", "This controller only works in Windows."

        self.output_dir = output_dir
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        self.total_num = total_num
        self.py_impact = PyImpact()

        self.recorder_pid: Optional[int] = None

        self.object_info = PyImpact.get_object_info()

        lib_full = ModelLibrarian("models_full.json")
        lib_sound20k = ModelLibrarian(str(Path("models/models.json").resolve()))
        lib_special = ModelLibrarian("models_special.json")
        self.libs = {"models_full.json": lib_full,
                     "models_special.json": lib_special,
                     "models/models.json": lib_sound20k}

        # Get the correct device to record system audio.
        devices = check_output(["fmedia", "--list-dev"]).decode("utf-8").split("Capture:")[1]
        dev_search = re.search("device #(.*): Stereo Mix", devices, flags=re.MULTILINE)
        assert dev_search is not None, "No suitable audio capture device found:\n" + devices
        self.capture_device = dev_search.group(1)

        super().__init__(port=port)

        # Global settings.
        self.communicate([{"$type": "set_screen_size",
                           "width": 256,
                           "height": 256},
                          {"$type": "set_time_step",
                           "time_step": 0.02},
                          {"$type": "set_target_framerate",
                           "framerate": 60},
                          {"$type": "set_physics_solver_iterations",
                           "iterations": 20}])

    def remove_output_directory(self) -> None:
        """
        Delete the old directory.
        """

        dir_util.remove_tree(str(self.output_dir.resolve()))

    def stop_recording(self) -> None:
        """
        Kill the recording process.
        """

        if self.recorder_pid is not None:
            with open(devnull, "w+") as f:
                call(['taskkill', '/F', '/T', '/PID', str(self.recorder_pid)], stderr=f, stdout=f)

    def sound20k(self, total: int = 20378) -> None:
        """
        Generate a dataset analogous to Sound20K.

        :param total: The total number of audio files to create.
        """

        # Load models by wnid.
        wnids = loads(Path("models/wnids_sound20k.json").read_text(encoding="utf-8"))
        num_per_wnid = int(total / len(wnids))

        scenes = get_sound20k_scenes()
        pbar = tqdm(total=total)

        for wnid in wnids:
            pbar.set_description(wnid)
            self.process_wnid(scenes, wnids[wnid], num_per_wnid, pbar)
        pbar.close()

    def process_wnid(self, scenes: List[Scene], models: List[Dict[str, str]], num_total: int, pbar: Optional[tqdm]) -> None:
        """
        Generate .wav files from all models in the category.

        :param scenes: The scenes that a trial can use.
        :param models: The names of the models in the category and their libraries.
        :param num_total: The total number of files to generate for this category.
        :param pbar: The progress bar.
        """

        num_images_per_model = int(num_total / len(models))
        num_scenes_per_model = int(num_images_per_model / len(scenes))

        # The number of files generated for the wnid.
        count = 0
        # The number of files generated for the current model.
        model_count = 0
        # The model being used to generate files.
        model_index = 0
        # The number of files for the current model that have used the current scene.
        scene_count = 0
        # The scene being used to generate files.
        scene_index = 0

        while count < num_total:
            self.trial(scene=scenes[scene_index], obj_name=models[model_index]["name"],
                       obj_library=models[model_index]["library"], file_count=model_count)
            count += 1
            # Iterate through scenes.
            scene_count += 1
            if scene_count > num_scenes_per_model:
                scene_index += 1
                if scene_index > len(scenes):
                    scene_index = 0
            # Iterate through models.
            model_count += 1
            if model_count > num_images_per_model:
                model_index += 1
                # If this is a new model, reset the scene count.
                scene_index = 0
                scene_count = 0
                if model_index > len(models):
                    model_index = 0
            if pbar is not None:
                pbar.update(1)

    def trial(self, scene: Scene, obj_name: str, obj_library: str, file_count: int) -> None:
        """
        Run a trial in a scene that has been initialized.

        :param scene: Data for the current scene.
        :param obj_name: The name of the object that will be dropped.
        :param obj_library: The object's library.
        :param file_count: The number of files with this object so far.
        """

        self.py_impact.reset()
        obj_info = self.object_info[obj_name]
        record = self.libs[obj_library].get_record(obj_name)

        output_dir = self.output_dir.joinpath(record.wnid)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        filename = output_dir.joinpath(obj_name + "_" + TDWUtils.zero_padding(file_count, 4) + ".wav")
        output_path = output_dir.joinpath(filename)
        # Skip files that already exist.
        while output_path.exists():
            return

        # Initialize the scene, positioning objects, furniture, etc.
        resp = self.communicate(scene.initialize_scene(self))
        center = scene.get_center(self)

        max_y = scene.get_max_y()
        o_x = RNG.uniform(center["x"] - 0.05, center["x"] + 0.05)
        o_y = RNG.uniform(max_y - 0.5, max_y)
        o_z = RNG.uniform(center["z"] - 0.05, center["z"] + 0.05)
        o_id = 0
        # Create the object and apply a force.
        commands = [{"$type": "add_object",
                     "name": record.name,
                     "url": record.get_url(),
                     "scale_factor": record.scale_factor,
                     "position": {"x": o_x, "y": o_y, "z": o_z},
                     "category": record.wcategory,
                     "id": o_id},
                    {"$type": "set_mass",
                     "id": o_id,
                     "mass": obj_info.mass},
                    {"$type": "set_physic_material",
                     "id": o_id,
                     "bounciness": obj_info.bounciness,
                     "static_friction": 0.1,
                     "dynamic_friction": 0.8},
                    {"$type": "rotate_object_by",
                     "angle": RNG.uniform(-30, 30),
                     "id": o_id,
                     "axis": "yaw",
                     "is_world": True},
                    {"$type": "rotate_object_by",
                     "angle": RNG.uniform(0, 20),
                     "id": o_id,
                     "axis": "pitch",
                     "is_world": True},
                    {"$type": "rotate_object_by",
                     "angle": RNG.uniform(-45, 45),
                     "id": o_id,
                     "axis": "roll",
                     "is_world": True},
                    {"$type": "apply_force_magnitude_to_object",
                     "magnitude": RNG.uniform(0, 4),
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
        r = RNG.uniform(2.3, 4.6)
        a_x = center["x"] + r
        a_y = RNG.uniform(1.5, 3)
        a_z = center["y"] + r
        cam_angle_min, cam_angle_max = scene.get_camera_angles()
        theta = RNG.uniform(cam_angle_min, cam_angle_max)
        rad = np.radians(theta)
        a_x = np.cos(rad) * (a_x - center["x"]) - np.sin(rad) * (a_z - center["z"]) + center["x"]
        a_z = np.sin(rad) * (a_x - center["x"]) + np.cos(rad) * (a_z - center["z"]) + center["z"]
        commands.extend(TDWUtils.create_avatar(position={"x": a_x, "y": a_y, "z": a_z},
                                               look_at=look_at))
        # Add the audio sensor.
        # Disable the image sensor (this is audio-only).
        commands.extend([scene.audio_system.add_audio_sensor(),
                         {"$type": "toggle_image_sensor"}])

        # Send the commands.
        resp = self.communicate(commands)
        with open(devnull, "w+") as f:
            self.recorder_pid = Popen(["fmedia",
                                       "--record",
                                       f"--dev-capture={self.capture_device}",
                                       f"--out={str(output_path.resolve())}"],
                                      stderr=f).pid

        # Loop until all objects are sleeping.
        done = False
        while not done:
            commands = []
            collisions, environment_collisions, rigidbodies = PyImpact.get_collisions(resp)
            for collision in collisions:
                if PyImpact.is_valid_collision(collision):
                    # Get the audio material and amp.
                    collider_id = collision.get_collider_id()
                    collider_material, collider_amp = self._get_object_info(collider_id, Scene.OBJECT_IDS, obj_name)
                    collidee_id = collision.get_collider_id()
                    collidee_material, collidee_amp = self._get_object_info(collidee_id, Scene.OBJECT_IDS, obj_name)
                    impact_sound_command = self.py_impact.get_impact_sound_command(
                        collision=collision,
                        rigidbodies=rigidbodies,
                        target_id=collidee_id,
                        target_amp=collidee_amp,
                        target_mat=collidee_material.name,
                        other_id=collider_id,
                        other_mat=collider_material.name,
                        other_amp=collider_amp,
                        play_audio_data=scene.audio_system.play_audio_data())
                    commands.append(impact_sound_command)
            # Handle environment collision.
            for collision in environment_collisions:
                collider_id = collision.get_object_id()
                collider_material, collider_amp = self._get_object_info(collider_id, Scene.OBJECT_IDS, obj_name)
                surface_material = scene.get_surface_material()
                impact_sound_command = self.py_impact.get_impact_sound_command(
                    collision=collision,
                    rigidbodies=rigidbodies,
                    target_id=collider_id,
                    target_amp=collider_amp,
                    target_mat=collider_material.name,
                    other_id=-1,
                    other_amp=0.1,
                    other_mat=surface_material.name,
                    play_audio_data=scene.audio_system.play_audio_data())
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
                resp = self.communicate(commands)
        # Stop video capture.
        self.stop_recording()

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

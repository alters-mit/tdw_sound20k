from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils, AudioUtils
from tdw.py_impact import PyImpact, AudioMaterial
from tdw.output_data import OutputData, Bounds, Transforms, Rigidbodies, AudioSources
from tdw.librarian import ModelLibrarian, ModelRecord
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from scenes import get_sound20k_scenes, get_tdw_scenes, Scene
from json import loads
from distutils import dir_util
from tqdm import tqdm
import sqlite3
import json

RNG = np.random.RandomState(0)


class AudioDataset(Controller):
    def __init__(self, output_dir: Path = Path("D:/audio_dataset"), total: int = 28602, port: int = 1071):
        """
        :param output_dir: The output directory for the files.
        :param port: The socket port.
        :param total: The total number of files per sub-set.
        """

        self.total = total

        self.output_dir = output_dir
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        db_path = self.output_dir.joinpath('results.db')
        self.conn = sqlite3.connect(str(db_path.resolve()))
        self.db_c = self.conn.cursor()
        # Sound20K table.
        if self.db_c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sound20k'").\
                fetchone() is None:
            self.db_c.execute("CREATE TABLE sound20k (path text, scene integer, cam_x real, cam_y real, cam_z real,"
                              "obj_x real, obj_y real, obj_z real, mass real, static_friction real, dynamic_friction "
                              "real, yaw real, pitch real, roll real, force real)")
        # Scenes table.
        if self.db_c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='scenes'").\
                fetchone() is None:
            self.db_c.execute("CREATE TABLE scenes (id integer, commands text)")

        self.py_impact = PyImpact()

        self.object_info = PyImpact.get_object_info()
        sound20k_object_info = PyImpact.get_object_info(Path("models/object_info.csv"))
        for obj_info in sound20k_object_info:
            if obj_info in self.object_info:
                continue
            else:
                self.object_info.update({obj_info: sound20k_object_info[obj_info]})

        self.libs: Dict[str, ModelLibrarian] = {}
        # Load all model libraries into memory.
        for lib_name in ModelLibrarian.get_library_filenames():
            self.libs.update({lib_name: ModelLibrarian(lib_name)})
        # Add the custom model library.
        self.libs.update({"models/models.json": ModelLibrarian(str(Path("models/models.json").resolve()))})

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

    def process_sub_set(self, name: str, models_mat_file: str, init_commands: List[dict], scenes: List[Scene]) -> None:
        """
        Process a sub-set of the complete dataset (e.g. all of Sound20K).

        :param name: The name of the sub-set.
        :param models_mat_file: The models per material data filename.
        :param init_commands: The commands used to initialize the entire process (this is sent only once).
        :param scenes: The scenes that can be loaded.
        """

        print(name)
        # Load models by wnid.
        materials: Dict[str, List[str]] = loads(Path(f"models/{models_mat_file}.json").read_text(encoding="utf-8"))
        num_per_material = int(self.total / len(materials))

        # Load the scene.
        self.communicate(init_commands)

        pbar = tqdm(total=self.total)

        for material in materials:
            pbar.set_description(material)
            self.process_material(root_dir=self.output_dir.joinpath(name),
                                  scenes=scenes,
                                  material=material,
                                  models=materials[material],
                                  num_total=num_per_material,
                                  pbar=pbar)
        pbar.close()

    def sound20k_set(self) -> None:
        """
        Generate a dataset analogous to Sound20K.
        """

        sound20k_init_commands = [{"$type": "load_scene"},
                                  TDWUtils.create_empty_room(12, 12),
                                  {"$type": "set_proc_gen_walls_scale",
                                   "walls": TDWUtils.get_box(12, 12),
                                   "scale": {"x": 1, "y": 4, "z": 1}},
                                  {"$type": "set_reverb_space_simple",
                                   "env_id": 0,
                                   "reverb_floor_material": "parquet",
                                   "reverb_ceiling_material": "acousticTile",
                                   "reverb_front_wall_material": "smoothPlaster",
                                   "reverb_back_wall_material": "smoothPlaster",
                                   "reverb_left_wall_material": "smoothPlaster",
                                   "reverb_right_wall_material": "smoothPlaster"},
                                  {"$type": "create_avatar",
                                   "type": "A_Img_Caps_Kinematic",
                                   "id": "a"},
                                  {"$type": "add_environ_audio_sensor"},
                                  {"$type": "toggle_image_sensor"}]

        self.process_sub_set("Sound20K", "models_per_material_sound20k", sound20k_init_commands, get_sound20k_scenes())

    def tdw_set(self) -> None:
        self.process_sub_set("TDW", "models_per_material_tdw", [], get_tdw_scenes())

    def process_material(self, root_dir: Path, scenes: List[Scene], models: List[str], material: str, num_total: int, pbar: Optional[tqdm]) -> None:
        """
        Generate .wav files from all models with the material.

        :param root_dir: The root output directory.
        :param scenes: The scenes that a trial can use.
        :param models: The names of the models in the category and their libraries.
        :param num_total: The total number of files to generate for this category.
        :param material: The name of the material.
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

        output_dir = root_dir.joinpath(material)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        while count < num_total:
            obj_name = models[model_index]
            filename = output_dir.joinpath(material + "_" + TDWUtils.zero_padding(count, 4) + ".wav")
            # Get the expected output path.
            output_path = output_dir.joinpath(filename)

            # Do a trial if the file doesn't exist yet.
            if not output_path.exists():
                try:
                    self.trial(scene=scenes[scene_index],
                               record=self.libs[self.object_info[models[model_index]].library].get_record(obj_name),
                               output_path=output_path,
                               scene_index=scene_index)
                finally:
                    # Stop recording audio.
                    AudioUtils.stop()

            count += 1
            # Iterate through scenes.
            scene_count += 1
            if scene_count > num_scenes_per_model:
                scene_id = scenes[scene_index].get_id()
                # Add the scene to the database.
                scene_db = self.db_c.execute("SELECT * FROM scenes WHERE id=?", (scene_id,)).fetchone()
                if scene_db is None:
                    self.db_c.execute("INSERT INTO scenes VALUES(?,?)",
                                      (scene_id, json.dumps(scenes[scene_index].initialize_scene(self))))
                    self.conn.commit()
                scene_index += 1
                scene_count = 0
                if scene_index >= len(scenes):
                    scene_index = 0
            # Iterate through models.
            model_count += 1
            if model_count > num_images_per_model:
                model_index += 1
                if model_index >= len(models):
                    model_index = 0
                model_count = 0
                # If this is a new model, reset the scene count.
                scene_index = 0
                scene_count = 0
                # Unload the asset bundles because we are done with this model.
                self.communicate({"$type": "unload_asset_bundles"})
            if pbar is not None:
                pbar.update(1)

    def trial(self, scene: Scene, record: ModelRecord, output_path: Path, scene_index: int) -> None:
        """
        Run a trial in a scene that has been initialized.

        :param scene: Data for the current scene.
        :param record: The model's metadata record.
        :param output_path: Write the .wav file to this path.
        :param scene_index: The scene identifier.
        """

        self.py_impact.reset(initial_amp=0.05)

        # Initialize the scene, positioning objects, furniture, etc.
        resp = self.communicate(scene.initialize_scene(self))
        center = scene.get_center(self)

        max_y = scene.get_max_y()

        # The object's initial position.
        o_x = RNG.uniform(center["x"] - 0.15, center["x"] + 0.15)
        o_y = RNG.uniform(max_y - 0.5, max_y)
        o_z = RNG.uniform(center["z"] - 0.15, center["z"] + 0.15)
        # Physics values.
        mass = self.object_info[record.name].mass + RNG.uniform(self.object_info[record.name].mass * -0.15,
                                                                self.object_info[record.name].mass * 0.15)
        static_friction = RNG.uniform(0.1, 0.3)
        dynamic_friction = RNG.uniform(0.7, 0.9)
        # Angles of rotation.
        yaw = RNG.uniform(-30, 30)
        pitch = RNG.uniform(0, 45)
        roll = RNG.uniform(-45, 45)
        # The force applied to the object.
        force = RNG.uniform(0, 5)
        # The avatar's position.
        a_r = RNG.uniform(1.5, 2.2)
        a_x = center["x"] + a_r
        a_y = RNG.uniform(1.5, 3)
        a_z = center["z"] + a_r
        cam_angle_min, cam_angle_max = scene.get_camera_angles()
        theta = np.radians(RNG.uniform(cam_angle_min, cam_angle_max))
        a_x = np.cos(theta) * (a_x - center["x"]) - np.sin(theta) * (a_z - center["z"]) + center["x"]
        a_z = np.sin(theta) * (a_x - center["x"]) + np.cos(theta) * (a_z - center["z"]) + center["z"]

        o_id = 0
        # Create the object and apply a force.
        commands = [{"$type": "add_object",
                     "name": record.name,
                     "url": record.get_url(),
                     "scale_factor": record.scale_factor,
                     "position": {"x": o_x,
                                  "y": o_y,
                                  "z": o_z},
                     "category": record.wcategory,
                     "id": o_id},
                    {"$type": "set_mass",
                     "id": o_id,
                     "mass": mass},
                    {"$type": "set_physic_material",
                     "id": o_id,
                     "bounciness": self.object_info[record.name].bounciness,
                     "static_friction": static_friction,
                     "dynamic_friction": dynamic_friction},
                    {"$type": "rotate_object_by",
                     "angle": yaw,
                     "id": o_id,
                     "axis": "yaw",
                     "is_world": True},
                    {"$type": "rotate_object_by",
                     "angle": pitch,
                     "id": o_id,
                     "axis": "pitch",
                     "is_world": True},
                    {"$type": "rotate_object_by",
                     "angle": roll,
                     "id": o_id,
                     "axis": "roll",
                     "is_world": True},
                    {"$type": "apply_force_magnitude_to_object",
                     "magnitude": force,
                     "id": o_id},
                    {"$type": "send_rigidbodies",
                     "frequency": "always"},
                    {"$type": "send_collisions",
                     "enter": True,
                     "exit": False,
                     "stay": False,
                     "collision_types": ["obj", "env"]},
                    {"$type": "send_transforms",
                     "frequency": "always"}]
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
        # Set the position at a given distance (r) from the center of the scene.
        # Rotate around that position to a random angle constrained by the scene's min and max angles.
        commands.extend([{"$type": "teleport_avatar_to",
                          "position": {"x": a_x, "y": a_y, "z": a_z}},
                         {"$type": "look_at_position",
                          "position": look_at}])

        # Send the commands.
        resp = self.communicate(commands)

        AudioUtils.start(output_path=output_path, until=(0, 10))

        # Loop until all objects are sleeping.
        done = False
        while not done and AudioUtils.is_recording():
            commands = []
            collisions, environment_collisions, rigidbodies = PyImpact.get_collisions(resp)
            # Create impact sounds from object-object collisions.
            for collision in collisions:
                if PyImpact.is_valid_collision(collision):
                    # Get the audio material and amp.
                    collider_id = collision.get_collider_id()
                    collider_material, collider_amp = self._get_object_info(collider_id, Scene.OBJECT_IDS, record.name)
                    collidee_id = collision.get_collider_id()
                    collidee_material, collidee_amp = self._get_object_info(collidee_id, Scene.OBJECT_IDS, record.name)
                    impact_sound_command = self.py_impact.get_impact_sound_command(
                        collision=collision,
                        rigidbodies=rigidbodies,
                        target_id=collidee_id,
                        target_amp=collidee_amp,
                        target_mat=collidee_material.name,
                        other_id=collider_id,
                        other_mat=collider_material.name,
                        other_amp=collider_amp,
                        play_audio_data=False)
                    commands.append(impact_sound_command)
            # Create impact sounds from object-environment collisions.
            for collision in environment_collisions:
                collider_id = collision.get_object_id()
                if self._get_velocity(rigidbodies, collider_id) > 0:
                    collider_material, collider_amp = self._get_object_info(collider_id, Scene.OBJECT_IDS, record.name)
                    surface_material = scene.get_surface_material()
                    impact_sound_command = self.py_impact.get_impact_sound_command(
                        collision=collision,
                        rigidbodies=rigidbodies,
                        target_id=collider_id,
                        target_amp=collider_amp,
                        target_mat=collider_material.name,
                        other_id=-1,
                        other_amp=0.01,
                        other_mat=surface_material.name,
                        play_audio_data=False)
                    commands.append(impact_sound_command)
            # If there were no collisions, check for movement. If nothing is moving, the trial is done.
            if len(commands) == 0:
                transforms = AudioDataset._get_transforms(resp)
                done = True
                for i in range(rigidbodies.get_num()):
                    if self._is_moving(rigidbodies.get_id(i), transforms, rigidbodies):
                        done = False
                        break
            # Continue the trial.
            if not done:
                resp = self.communicate(commands)

        # Stop listening for anything except audio data..
        resp = self.communicate([{"$type": "send_rigidbodies",
                                  "frequency": "never"},
                                 {"$type": "send_transforms",
                                  "frequency": "never"},
                                 {"$type": "send_collisions",
                                  "enter": False,
                                  "exit": False,
                                  "stay": False,
                                  "collision_types": []},
                                 {"$type": "send_audio_sources",
                                  "frequency": "always"}])
        # Wait for the audio to finish.
        done = False
        while not done and AudioUtils.is_recording():
            done = True
            for r in resp[:-1]:
                if OutputData.get_data_type_id(r) == "audi":
                    audio_sources = AudioSources(r)
                    for i in range(audio_sources.get_num()):
                        if audio_sources.get_is_playing(i):
                            done = False
            if not done:
                resp = self.communicate([])
        # Cleanup.
        commands = [{"$type": "send_audio_sources",
                     "frequency": "never"},
                    {"$type": "destroy_object",
                     "id": o_id}]
        for scene_object_id in Scene.OBJECT_IDS:
            commands.append({"$type": "destroy_object",
                             "id": scene_object_id})
        self.communicate(commands)

        # Insert the trial's values into the database.
        self.db_c.execute("INSERT INTO sound20k VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                          (output_path.name, scene_index, a_x, a_y, a_z, o_x, o_y, o_z, mass,
                           static_friction, dynamic_friction, yaw, pitch, roll, force))
        self.conn.commit()

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

    @staticmethod
    def _get_transforms(resp: List[bytes]) -> Transforms:
        """
        :param resp: The output data response.

        :return: Transforms data.
        """

        for r in resp[:-1]:
            if OutputData.get_data_type_id(r) == "tran":
                return Transforms(r)
        raise Exception("Transforms output data not found!")

    @staticmethod
    def _get_velocity(rigidbodies: Rigidbodies, o_id: int) -> float:
        """
        :param rigidbodies: The rigidbody data.
        :param o_id: The ID of the object.

        :return: The velocity magnitude of the object.
        """

        for i in range(rigidbodies.get_num()):
            if rigidbodies.get_id(i) == o_id:
                return np.linalg.norm(rigidbodies.get_velocity(i))

    @staticmethod
    def _is_moving(o_id: int, transforms: Transforms, rigidbodies: Rigidbodies) -> bool:
        """
        :param o_id: The ID of the object.
        :param transforms: The Transforms output data.
        :param rigidbodies: The Rigidbodies output data.

        :return: True if the object is still moving.
        """

        y: Optional[float] = None
        sleeping: bool = False

        for i in range(transforms.get_num()):
            if transforms.get_id(i) == o_id:
                y = transforms.get_position(i)[1]
                break
        assert y is not None, f"y value is none for {o_id}"

        for i in range(rigidbodies.get_num()):
            if rigidbodies.get_id(i) == o_id:
                sleeping = rigidbodies.get_sleeping(i)
                break
        # If the object isn't sleeping, it is still moving.
        # If the object fell into the abyss, we don't count it as moving (to prevent an infinitely long simulation).
        return not sleeping and y > -10


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--dir", type=str, default="D:/audio_dataset", help="Output directory")
    args = parser.parse_args()

    a = AudioDataset(output_dir=Path(args.dir))
    a.sound20k_set()
    a.tdw_set()
    a.communicate({"$type": "terminate"})

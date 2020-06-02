# tdw_sound20k

**Seth Alter**

**May 2020**

## Overview

`audio_dataset.py` generates a dataset of audio files using TDW. This dataset is meant to be compared to the [Sound20K dataset](https://github.com/ztzhang/SoundSynth), and is split into two "sub-sets":

1. The first half of the dataset tries to mimic Sound20K scene setups, objects, audio parameters, etc. as closely as possible.
2. The second half of the dataset introduces new object categories not found in Sound20K and much more varied scene setups.

Each sub-set contains the same number of files as Sound20K (meaning that the total number of files in this dataset is twice that of Sound20K).

## Requirements

1. [fmedia](https://stsaz.github.io/fmedia/) (Follow the installation instructions closely)
2. Audio drivers
3. TDW
4. Access to TDW's `models_full.json` library

Additionally, `audio_dataset.py` has only been tested on Windows 10, and might not work on other platforms.

## Usage

```python
python3 audio_dataset.py --dir <output_directory>
```

NOTE: This will a _**long**_ time to generate a full dataset, possibly up to 96 hours.

### Output

```
output_directory
....results.db
....Sound20K/
....TDW/
```

- **results.db** is a database containing metadata of every trial in the dataset. Use sqlite to read it.
- **Sound20K/** is meant to replicate the original Sound20K dataset; **TDW/** expands upon it with additional models and scenarios. See below for more information.

## How to Use the Sound20K Dataset

If you want to compare the dataset created by `audio_dataset.py` to the original Sound20K dataset, you will need to convert the original Sound20K dataset, because it isn't labeled with useful object classification information and contains many files that aren't needed.

1. Download and extract [this file](http://sound.csail.mit.edu/data/sound-20k.tar.gz).
2. Run this script:

```python
python3 convert_sound20k_files.py --src <sound20k_source_directory> --dest <output_directory>
```

This will copy and rename each _relevant_ file into the `--dest` directory.

## Labels

Files are labeled by directory; each directory is an _audio material_.

There are six materials:

1. ceramic
2. glass
3. metal
4. hardwood
5. wood
6. cardboard

The labeling system is consistent between the TDW-generated dataset (`Sound20K/` and `TDW/`) and the converted Sound20K dataset.

**Example:**

```
sound20k_audio_dataset/
....ceramic/
........0000.wav
........0001.wav
........(etc.)
....glass/
........(etc.)

tdw_audio_dataset/
....TDW/
........ceramic/
............0000.wav
............0001.wav
............(etc.)
........glass/
............(etc.)
....Sound20K/
........ceramic/
............0000.wav
............0001.wav
............(etc.)
........glass/
............(etc.)
```

## What It Does

In every trial, `audio_dataset.py` does the following:

1. Load a scene
2. Load an object to drop
3. Set a random position for the audio sensor
4. Apply some randomness to the object (varying its mass, initial position, rotation, initial velocity, etc.)
5. Let the object fall.
6. Listen for collisions and play audio using PyImpact.
7. Wait for objects to stop moving, or end after 10 seconds (whichever happens first).
8. Stop recording.
9. Clean up the scene.

### Scenes

**Scenes** are defined as class in the script `scenes.py`. Each scene has initialization instructions and a name (used to identify it in the `results.db` database).

### Models

Most models are in TDW's default model libraries. A few are custom models, some pulled directly from the Sound20K models. These can be found in the `models/` directory, along with the model library, `models/models.json`.

## Sound20K

In the Sound20K sub-set, the dropped object is selected from a list derived from `models/wnids_sound20k.json`. The scene is selected from one of 9 scenes, each of which are variants of an empty room with a wood floor:

| Scene                  | Description                                                  |
| ---------------------- | ------------------------------------------------------------ |
| `FloorSound20K`        | An empty room.                                               |
| `CornerSound20K`       | Objects will spawn near a corner.                            |
| `StairRamp`            | There is a stair ramp in the center of the room.             |
| `RoundTable`           | There is a round wood table in the center of the room.       |
| `UnevenTerrain`        | Bumpy terrain mesh.                                          |
| `LargeBowl`            | There is a large bowl in the center of the room.             |
| `Ramp`                 | There is a ramp in the center of the room.                   |
| `DeskAndChair`         | There is a desk, a chair, and a shelf with a few objects adjacent to one of the walls; objects spawn above the desk and chair. |
| `DiningTableAndChairs` | There is a dining table surrounded by 8 chairs.              |

Scenes are always selected evenly per model; thus if there will be 90 audio files generated by a model, each scene will be selected for this model 10 times.

## TDW

In the TDW sub-set, the dropped object is selected from a list derived from `models/wnids_tdw.json`. There are only 2 types of scenes:

| Scene        | Description                                                  |
| ------------ | ------------------------------------------------------------ |
| `TDWScene`   | An empty room with random ResonanceAudio materials and a random size. |
| `TDWObjects` | Same as above, but with a few objects.One object, the "base object", has a large mass. The other "props" have low masses. |

Each of these scenes is _much_ more varied than any of the Sound20K scenes; they randomly select different initialization parameters.

#### Audio Options

See `tdw_scene_audio.py`

| Option        | Probability | Description                                                  |
| ------------- | ----------- | ------------------------------------------------------------ |
| `Realistic`   | 60%         | Surface materials are chosen randomly, but are "plausible" (e.g. walls are never made out of grass). The walls all have the same surface material. The PyImpact surface material matches the ResonanceAudio material (e.g. `wood` and `parquet`; see the Command API and PyImpact API for more information). |
| `Unrealistic` | 30%         | Surface materials are chosen randomly, but are "plausible" (e.g. walls are never made out of grass). Each of the walls can have a different material. The PyImpact surface material matches the ResonanceAudio material (see above). |
| `Chaos`       | 10%         | Surface materials are totally random. The PyImpact surface material is chosen randomly as well. |

#### Scene Scene Options

See `tdw_scene_size.py`

| Option         | Probability | Description                                                |
| -------------- | ----------- | ---------------------------------------------------------- |
| `StandardSize` | 50%         | 12x12 (the same as the scene used in the Sound20K sub-set) |
| `SmallSize`    | 30%         | 4x4                                                        |
| `RandomSize`   | 10%         | Random dimensions between 4x4 and 12x12                    |




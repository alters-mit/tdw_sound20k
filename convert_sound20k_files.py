from pathlib import Path
from tdw.tdw_utils import TDWUtils
import json
from typing import Dict
from distutils import file_util
from argparse import ArgumentParser

"""
Copy Sound20K audio files into a new directory.
Rename the files by wnid/object_num.wav
Some files will be excluded.
"""

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--src", default="D:/result", help="The original Sound20K audio file directory.")
    parser.add_argument("--dest", default="D:/sound20k", help="The directory to copy the renamed files to.")
    args = parser.parse_args()
    src_root = Path(args.src)
    dest_root = Path(args.dest)

    obj_counts: Dict[str, int] = {}
    wnid_data = json.loads(Path("sound20k_data/objects_by_wnid.json").read_text(encoding="utf-8"))

    for src in src_root.rglob("*.wav"):
        # Ignore merged files.
        if src.stem == "merged":
            continue
        # Parse the name of the file by object and material.
        objs = src.parts[3].split("-")[2:]
        mats = src.parts[4].split("-")[2:]
        for obj, mat, i in zip(objs, mats, range(len(objs))):
            if i + 1 == int(src.stem):
                # Ignore objects that aren't listed in the wnid data (they are deliberately excluded from the dataset).
                if obj not in wnid_data:
                    continue
                # Ignore files that use "non-canonical" materials.
                if mat not in wnid_data[obj]["materials"]:
                    continue
                if obj not in obj_counts:
                    obj_counts.update({obj: 0})
                dest_dir = dest_root.joinpath(wnid_data[obj]["wnid"])
                if not dest_dir.exists():
                    dest_dir.mkdir(parents=True)

                dest_filename = f"{obj}-{mat}_{TDWUtils.zero_padding(obj_counts[obj])}.wav"
                dest = dest_dir.joinpath(dest_filename)
                file_util.copy_file(str(src.resolve()), str(dest.resolve()))
                obj_counts[obj] += 1
    print("Done!")
    print(f"There are {sum(obj_counts.values())} files.")

from tdw.py_impact import PyImpact, CollisionInfo
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.librarian import ModelLibrarian, ModelRecord, MaterialLibrarian, MaterialRecord
from tdw.output_data import OutputData, Environments
import itertools
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List, Dict, Tuple
from subprocess import Popen, call
from platform import system
from time import sleep
from json import dumps
from abc import ABC, abstractmethod
from scenes import CornerSound20k, FloorSound20k

RNG = np.random.RandomState(0)


class AudioDataset(Controller):
    SCENES = [CornerSound20k, FloorSound20k]

    def __init__(self, total_num: int = 20378, output_dir: Path = Path("D:/audio_dataset"), port: int = 1071):
        assert system() == "Windows", "This controller only works in Windows."

        self.output_dir = output_dir
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        self.total_num = total_num
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.librarian import ModelLibrarian
from pathlib import Path

c = Controller()
c.model_librarian = ModelLibrarian(str(Path("models/models.json").resolve()))
c.start()
c.communicate(TDWUtils.create_empty_room(12, 12))
commands = [c.get_add_object("uneven_terrain", 0)]
for i in range(1):
    commands.append({"$type": "step_physics",
                     "frames": 1})
c.communicate(commands)
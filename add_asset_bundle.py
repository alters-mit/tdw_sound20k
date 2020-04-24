from tdw.asset_bundle_creator import AssetBundleCreator
from tdw.librarian import ModelLibrarian, ModelRecord
from tdw.backend.platforms import UNITY_TO_SYSTEM
from json import loads
from pathlib import Path

lib = ModelLibrarian(str(Path("models/models.json").resolve()))

src_root_dir = Path.home().joinpath("Downloads/ready")
dest_root_dir = Path("models")

models = {"1006": {"name": "uneven_terrain",
                   "path": "1006/1006.orig.obj"}}

a = AssetBundleCreator()

for model in models:
    m = models[model]
    src_dir = src_root_dir.joinpath(m["path"])
    assert src_dir.exists()
    # Create the asset bundles.
    asset_bundle_srcs, record_src = a.create_asset_bundle(src_dir, True, add_textures=False)
    record = ModelRecord(loads(record_src.read_text(encoding="utf-8")))
    record.name = m["name"]
    # Move and rename the asset bundles.
    for asset_bundle_src in asset_bundle_srcs:
        platform = UNITY_TO_SYSTEM[asset_bundle_src.parts[-2]]
        asset_bundle_dest_dir = dest_root_dir.joinpath(m["name"] + "/" + platform)
        if not asset_bundle_dest_dir.exists():
            asset_bundle_dest_dir.mkdir(parents=True)
        asset_bundle_src.replace(asset_bundle_dest_dir.joinpath(m["name"]))
        record.urls[platform] = f"models/{m['name']}/{platform}/{m['name']}"

    # Add the record.
    lib.add_or_update_record(record, overwrite=False, write=True)

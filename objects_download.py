import math
import shutil
import objaverse
from tqdm import tqdm
import trimesh
import json
import os
import numpy as np
from PIL import Image

print(f"{objaverse.__version__=}")

objects_path = "data/datasets/pin/hm3d/v1/objects/"
os.makedirs(objects_path, exist_ok=True)

# load data from json
objs = None
with open("data/object_ids.json", "rb") as file:
    uids = json.load(file)

# loading objs form objaverse
print("Downloading Objects:")

objects = objaverse.load_objects(uids=uids)

# view mesh with trimesh
DEBUG = False

for i, (obj_id, obj_path) in tqdm(enumerate(objects.items()), total=len(objects)):
    if DEBUG:
        mesh = trimesh.load(obj_path, process=True)
        print(f"Showing mesh: {obj_id}, {obj_path}")
        mesh.show(background=[127, 127, 127, 0])
        print("Done")

    
    destination_path = os.path.join(objects_path, obj_id) + ".glb"
    if not os.path.exists(destination_path):
        try:
            shutil.copy(obj_path, destination_path)
        except PermissionError:
            pass
        print(f"Copied {obj_path} to {destination_path}")
    else:
        print(f"Object at {destination_path} already exists")
    
print("Done")
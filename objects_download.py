import shutil
import objaverse
from tqdm import tqdm
import json
import os
import numpy as np

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
anno = objaverse.load_annotations(uids=uids)

unique_sources = np.unique([an['uri'].replace("https://", "").split("/")[0] for an in anno.values()])
unique_license = np.unique([an['license'] for an in anno.values()], return_counts=True)

for i, (obj_id, obj_path) in tqdm(enumerate(objects.items()), total=len(objects)):
    
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
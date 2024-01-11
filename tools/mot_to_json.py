import argparse
import os, sys
import pathlib
import warnings

import numpy as np
import json
from tqdm import tqdm
sys.path.append(os.getcwd())
from tools.valid_scenes import VALID_TEST

parser = argparse.ArgumentParser(
    description="MOTChallenge to Synthehicle JSON converter."
)


parser.add_argument(
    "-d",
    "--data_dir",
    type=str,
    required=False,
    help="Root directory of dataset, e.g., data/synthehicle/",
    default="carla/generate_carla_data/scenes_non_overlap/Town10HD_Opt/day_2024-01-07_13-36-53/"
)
parser.add_argument(
    "-c",
    "--cameras",
    type=str,
    required=False,
    help="Path to camera config file., e.g., splits/test.txt",
    default="carla/generate_carla_data/scenes_non_overlap/Town10HD_Opt/camera_info/"
)
parser.add_argument(
    "-p",
    "--pattern",
    type=str,
    required=False,
    help="Pattern to prediction files., e.g., gt/gt.txt",
    default="gt/gt.txt",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    required=False,
    help="Path to output json file, e.g., predictions.json",
    default="ground_truth.json",
)
# python -m tools.mot_to_json -d {SYNTHEHICLE_DATA} -c splits/train.txt -o prediction.json -p prediction.txt

args = parser.parse_args()

if args.cameras is not None:
    with open(os.path.join(os.getcwd(), args.cameras), "r") as file:
        cameras = file.read().splitlines()
else:
    cameras = VALID_TEST

output = {}

for cam in tqdm(cameras, desc="Iterating camera paths"):
    if cam not in VALID_TEST:
        warnings.warn(
            f"Camera {cam} is not a valid synthehicle test split camera and will be ignored in evaluation."
        )
    scene_name, cam_name = pathlib.Path(cam).parts
    file = os.path.join(args.data_dir, cam, args.pattern)
    if os.path.isfile(file):
        data = np.loadtxt(file, delimiter=",").astype(np.int64).tolist()
    else:
        raise FileNotFoundError(f"File does not exist: {file}")
    if scene_name in output.keys():
        if cam_name not in output[scene_name].keys():
            output[scene_name][cam_name] = data
        else:
            raise ValueError(f"Duplicate scene and camera: {scene_name}, {cam_name}")
    else:
        output[scene_name] = {}
        output[scene_name][cam_name] = data

with open(args.output, "w") as f:
    json.dump(output, f)

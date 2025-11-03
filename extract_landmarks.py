import os
import json
import shutil

source_dir = "CelebV-HQ/video_data_landmark/train"
dest_dir = "refined_dataset_landmark/train"
os.makedirs(dest_dir, exist_ok=True)

with open("json_files/frames_data.json", "r") as f:
    data_dict = json.load(f)

count = 0
for key, val in data_dict.items():
    source_sample_dir = os.path.join(source_dir, key)
    dest_sample_dir = os.path.join(dest_dir, key)
    os.makedirs(dest_sample_dir, exist_ok=True)
    for frame, ypr in val.items():
        if not os.path.exists(os.path.join(source_sample_dir, frame.split('_')[-1])):
            print(f"{key}/{frame} not found ---------------------------------------")
            continue
        shutil.copy(os.path.join(source_sample_dir, frame.split('_')[-1]), os.path.join(dest_sample_dir, frame.split('_')[-1]))
        count += 1

print(count)
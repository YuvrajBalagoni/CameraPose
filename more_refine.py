import os
import json
import shutil

# with open('refined_no_landmarks.json', 'r') as f:
#     data_dict = json.load(f)

data_path = "refined_dataset"
landmark_path = "refined_dataset_landmark"

# remove = {}
# remove['keys'] = []
# for key, val in data_dict.items():
#     if len(val) == 0:
#         continue
#     data_len = len(os.listdir(os.path.join(data_path, key)))
#     landmark_len = len(os.listdir(os.path.join(landmark_path, key)))
#     if data_len >= 2 and landmark_len >= 2 and data_len == landmark_len:
#         continue
#     remove['keys'].append(key)

# with open('remove.json', 'w') as f:
#     json.dump(remove, f, indent=4)

# ================= removing samples ====================== #

# with open('remove.json', 'r') as f:
#     data_dict = json.load(f)

# for key in data_dict['keys']:
#     shutil.rmtree(os.path.join(data_path, key))
#     shutil.rmtree(os.path.join(landmark_path, key))

# print(len(os.listdir(data_path)))

count = 0
for sample_dir in os.listdir(data_path):
    n = len(os.listdir(os.path.join(data_path, sample_dir)))
    count += n * (n - 1)

print(count)
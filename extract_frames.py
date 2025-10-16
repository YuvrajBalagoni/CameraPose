import os
import json
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt 
import shutil

with open("data.json", "r") as f:
    data_dict = json.load(f)

data_dir = "YPR_Dataset/example"
output_data_dir = "refined_dataset"
os.makedirs(output_data_dir, exist_ok=True)

yaw, pitch, roll = 0, 0, 0
total_frames = 0
num_arr = []
frames = {}
issues = 0
total_pairs = 0
for key, val in tqdm(data_dict.items()):
    frames[key] = {}
    output_sample_dir = os.path.join(output_data_dir, key)
    os.makedirs(output_sample_dir, exist_ok=True)

    ypr_path = os.path.join(data_dir, key, "ypr.npy")
    ypr = np.load(ypr_path)
    del_yaw, del_pitch, del_roll = val['del_yaw'], val['del_pitch'], val['del_roll']
    max_range = max(del_yaw, del_pitch, del_roll)
    if del_yaw == max_range:
        yaw += 1
        array = ypr[:, 0]
    elif del_pitch == max_range:
        pitch += 1
        array = ypr[:, 1]
    elif del_roll == max_range:
       roll += 1
       array = ypr[:, 2]

    frames[key][f"frame_{np.argmin(array):05d}.jpg"] = ypr[np.argmin(array)].tolist()
    frames[key][f"frame_{np.argmax(array):05d}.jpg"] = ypr[np.argmax(array)].tolist()
    if not os.path.exists(os.path.join(data_dir, key, f"frame_{np.argmin(array):05d}.jpg")):
        print(f"argmin issue in {key} ----------------------------------------------------")
        issues += 1
        continue
    if not os.path.exists(os.path.join(data_dir, key, f"frame_{np.argmax(array):05d}.jpg")):
        print(f"argmax issue in {key} ----------------------------------------------------")
        issues += 1
        continue
    shutil.copy(os.path.join(data_dir, key, f"frame_{np.argmin(array):05d}.jpg"), os.path.join(output_sample_dir, f"frame_{np.argmin(array):05d}.jpg"))
    shutil.copy(os.path.join(data_dir, key, f"frame_{np.argmax(array):05d}.jpg"), os.path.join(output_sample_dir, f"frame_{np.argmax(array):05d}.jpg"))

    n = np.abs(np.max(array) - np.min(array)) // 10

    num_arr.append(n+1)
    total_frames += n + 1

    if (n) < 2:
        continue
    
    change = 0
    for i in range(2, int(n)):
        new_val = np.min(array) + 10 * (i - 1)
        new_idx = np.argmin(np.abs(array - new_val))
        frames[key][f"frame_{new_idx:05d}.jpg"] = ypr[new_idx].tolist()
        if not os.path.exists(os.path.join(data_dir, key, f"frame_{new_idx:05d}.jpg")):
            print(f"index issue in {key} ----------------------------------------------------")
            change += 1
            issues += 1
            continue
        shutil.copy(os.path.join(data_dir, key, f"frame_{new_idx:05d}.jpg"), os.path.join(output_sample_dir, f"frame_{new_idx:05d}.jpg"))

    total_pairs += (n - change) * (n - change - 1)
print(total_frames)
print(f"{yaw} - {pitch} - {roll}")
print(f"total issue frames = {issues}")
print(f"total pairs = {total_pairs}")
x = np.linspace(0, len(num_arr), len(num_arr))
fig, ax = plt.subplots()
ax.plot(x, num_arr)
plt.savefig("num_plot.png", dpi=300)

with open("frames_data.json", "w") as f:
    json.dump(frames, f, indent=4)
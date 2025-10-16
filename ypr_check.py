import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
import json

data_dir = "YPR_Dataset/example"

with open("data.json", "r") as f:
    data_dict = json.load(f)

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax1 = ax[0]
ax2 = ax[1]
ax3 = ax[2]
data_dict = {}
list_samples = os.listdir(data_dir)
list_samples.sort()
total_count = 0
total_yaw, total_pitch, total_roll = 0, 0, 0
del_array = np.array([[]])
for sample in tqdm(list_samples):
    ypr_path = os.path.join(data_dir, sample, "ypr.npy")
    if not os.path.exists(ypr_path):
        print(f"------------ {sample} not found --------------")
        continue

    ypr_np = np.load(ypr_path)

    if ypr_np.ndim == 1:
        print(f"----------- {sample} dim only 1 --------------")
        continue

    del_yaw = np.abs(np.max(ypr_np[:, 0]) - np.min(ypr_np[:, 0]))
    del_pitch = np.abs(np.max(ypr_np[:, 1]) - np.min(ypr_np[:, 1]))
    del_roll = np.abs(np.max(ypr_np[:, 2]) - np.min(ypr_np[:, 2]))

    if del_yaw >= 15.0:
        total_yaw += 1
    if del_pitch >= 15.0:
        total_pitch += 1
    if del_roll >= 15.0:
        total_roll += 1
    
    if del_yaw >= 15.0 or del_pitch >= 15.0 or del_roll >= 15.0:
        total_count += 1
        data_dict[sample] = {
            "del_yaw" : del_yaw,
            "del_pitch": del_pitch,
            "del_roll": del_roll
        }

    # del_array = np.concatenate((del_array, np.array([del_yaw, del_pitch, del_roll])), axis=1)
with open("data.json", "w") as f:
    json.dump(data_dict, f, indent=4)
    
print(f"{total_yaw}, {total_pitch}, {total_roll} = {total_count}")
# ax1.set_title("Yaw", fontsize=14)

# ax1.set_xlabel("X-axis (frames)")
# ax1.set_ylabel("Y-axis (angle)")
# ax1.grid(True, linestyle='--', alpha=0.6)

# plt.tight_layout()
# fig.savefig("ypr_plot.png", dpi=300)

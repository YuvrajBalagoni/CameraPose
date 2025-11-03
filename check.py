# git clone https://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@github.com/shadabkhan87/beardstylegan.git
# git clone https://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@github.com/YuvrajBalagoni/multihairgan.git


# import os
# import shutil

# source_dir = "CPDataset/train"
# target_dir = "CPEDataset"
# target_train_dir = os.path.join(target_dir, "train")
# target_test_dir = os.path.join(target_dir, "test")
# os.makedirs(target_train_dir, exist_ok=True)
# os.makedirs(target_test_dir, exist_ok=True)

# for class_name in (os.listdir(source_dir)):
#     source_class_dir = os.path.join(source_dir, class_name)
#     target_train_class_dir = os.path.join(target_train_dir, class_name)
#     target_test_class_dir = os.path.join(target_test_dir, class_name)

#     os.makedirs(target_train_class_dir, exist_ok=True)
#     os.makedirs(target_test_class_dir, exist_ok=True)

#     img_list = os.listdir(source_class_dir)
#     img_list.sort()

#     split = int(0.8 * len(img_list))
#     for img in img_list[:split]:
#         shutil.copy(os.path.join(source_class_dir, img), os.path.join(target_train_class_dir, img))
#     for img in img_list[split:]:
#         shutil.copy(os.path.join(source_class_dir, img), os.path.join(target_test_class_dir, img))

#     print(f"class:{class_name}, train:{len(img_list[:split])}, test:{len(img_list[split:])}")

# import os
# import shutil

# data_dir = "refined_dataset/train"
# landmark_dir = "refined_dataset_landmark/train"
# dest_dir = "temp_dataset/train"
# dest_landmark_dir = "temp_dataset_landmark/train"
# os.makedirs(dest_dir, exist_ok=True)
# os.makedirs(dest_landmark_dir, exist_ok=True)

# vid_list = os.listdir(data_dir)
# vid_list.sort()
# i = 0
# for vid in vid_list:
#     if len(os.listdir(os.path.join(data_dir, vid))) >= 6:
#         shutil.copytree(os.path.join(data_dir, vid), os.path.join(dest_dir, vid))
#         shutil.copytree(os.path.join(landmark_dir, vid), os.path.join(dest_landmark_dir, vid))
#         i += 1
#     if i == 20:
#         print("--------- created dataset ------------")
#         break

########################################################################################################################################

# import os
# import torch 

# ckpt_path = "checkpoints/LandmarkGAN_deeper_run_5.2_921_0.pt"
# state = torch.load(ckpt_path)

# disc_state = state['dis_state_dict']

# first_layer = disc_state['layers.0.0.weight']
# print(first_layer.shape)

# additional_layers = torch.full([64, 3, 4, 4], torch.mean(first_layer)).to('cuda')
# print(additional_layers.shape)

# new_layer_weight = torch.cat([first_layer, additional_layers], dim=1)
# print(new_layer_weight.shape)

# state['dis_state_dict']['layer.0.0.weight'] = new_layer_weight

# =================================================== #

# import os

# data_dir = "refined_dataset/train"
# landmark_dir = "refined_dataset_landmark/train"

# img_list = os.listdir(data_dir)
# img_list.sort()

# count = 0
# f_count = 0
# for img in img_list:
#     if os.path.exists(os.path.join(landmark_dir, img)):
#         for frames in os.listdir(os.path.join(data_dir, img)):
#             if os.path.exists(os.path.join(landmark_dir, img, frames)):
#                 continue
#             else:
#                 print(f"---------- {img} - {frames} does not have landmarks ----------")
#                 f_count += 1
#     else:
#         print(f"---------- {img} landmark no no ----------")
#         count += 1

# print(f"missing landmarks : identities {count} \n frames {f_count}")

# ======================================================================= #

# import os
# import shutil

# landmark_dir = "refined_dataset_landmark"

# for img in os.listdir(landmark_dir):
#     if img == 'train':
#         print("found train :)")
#     else:
#         shutil.rmtree(os.path.join(landmark_dir, img))

# ======================================================================= #

import os
import json
import shutil

data_path = "refined_dataset/train"
landmark_path = "refined_dataset_landmark/train"
dest_path = "overfit_data/train"
dest_landmark_path = "overfit_data_landmark/train"
os.makedirs(dest_path, exist_ok=True)
os.makedirs(dest_landmark_path, exist_ok=True)

img_list = os.listdir(data_path)
img_list.sort()
count = 0
for img in img_list:
    if len(os.listdir(os.path.join(data_path, img))) >= 6:
        shutil.copytree(os.path.join(data_path, img), os.path.join(dest_path, img))
        shutil.copytree(os.path.join(landmark_path, img), os.path.join(dest_landmark_path, img))
        count += 1
    
    if count == 20:
        break
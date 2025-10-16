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

import os
import shutil

data_dir = "refined_dataset/train"
landmark_dir = "refined_dataset_landmark/train"
dest_dir = "temp_dataset/train"
dest_landmark_dir = "temp_dataset_landmark/train"
os.makedirs(dest_dir, exist_ok=True)
os.makedirs(dest_landmark_dir, exist_ok=True)

vid_list = os.listdir(data_dir)
vid_list.sort()
i = 0
for vid in vid_list:
    if len(os.listdir(os.path.join(data_dir, vid))) >= 6:
        shutil.copytree(os.path.join(data_dir, vid), os.path.join(dest_dir, vid))
        shutil.copytree(os.path.join(landmark_dir, vid), os.path.join(dest_landmark_dir, vid))
        i += 1
    if i == 20:
        print("--------- created dataset ------------")
        break

import mediapipe as mp 
from mediapipe.tasks.python.components.containers import NormalizedLandmark
import cv2 
import numpy as np
from tqdm.auto import tqdm
import os
import json

landmark_68 = False

def get_landmark_coords(
        landmarks: list[NormalizedLandmark], width: int, height: int
    ) -> np.ndarray:
    """Extract normalized landmark coordinates to array of pixel coordinates."""
    xyz = [(lm.x, lm.y, lm.z) for lm in landmarks]
    return np.multiply(xyz, [width, height, width]).astype(int)

def get_68_landmarks(landmark_result, image_width, image_height):
    landmark_points_68 = [
                    162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,
                    296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
                    380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87
                    ]
    landmark_coords_68 = get_landmark_coords([landmark_result.face_landmarks[0][i] for i in landmark_points_68], image_width, image_height)
    return landmark_coords_68

# def get_pivot(landmarks_np):
#     """
#     a - top head, b - low chin
#     c - left ear, d - right ear
#     """
#     pt_a, pt_b = landmarks_np[10], landmarks_np[152]
#     pt_c, pt_d = landmarks_np[234], landmarks_np[454]

#     dc = pt_d - pt_c
#     ab = pt_a - pt_b

#     in_dir = np.cross(dc, ab)

#     pivot = pt_b + lambda_in * in_dir
#     pivot += lambda_down * ab
#     return pivot

def draw_landmarks(landmark_coordinates, image_width, image_height):
    image_np = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    for (x, y, z) in landmark_coordinates:
        cv2.circle(image_np, (x, y), 1, (255, 255, 255), -1)
    return image_np

model_path = "CelebV-HQ/face_landmarker.task"
base_options = mp.tasks.BaseOptions(
    model_asset_path=model_path
)
options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    output_face_blendshapes=True,
)

image_dataset_path = "refined_dataset"
landmark_dataset_path = "refined_dataset_landmark"
os.makedirs(landmark_dataset_path, exist_ok=True)

parent_dirs = os.listdir(image_dataset_path)

no_lm = {}

with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:

    for parent_dir in tqdm(parent_dirs):
        # print(f"----------------- Processing directory {parent_dir} -----------------")
        os.makedirs(os.path.join(landmark_dataset_path, parent_dir), exist_ok=True)
        image_files = os.listdir(os.path.join(image_dataset_path, parent_dir))
        no_lm[parent_dir] = []

        for image_file in image_files:
            mp_image = mp.Image.create_from_file(os.path.join(image_dataset_path, parent_dir, image_file))

            landmark_result = landmarker.detect(mp_image)

            if len(landmark_result.face_landmarks) == 0:
                no_lm[parent_dir].append(image_file)
                continue

            if landmark_68:
                landmark_np = get_68_landmarks(landmark_result, mp_image.width, mp_image.height)
            else:
                landmark_np = get_landmark_coords(landmark_result.face_landmarks[0], mp_image.width, mp_image.height)

            landmark_image = draw_landmarks(landmark_np, mp_image.width, mp_image.height)
            cv2.imwrite(os.path.join(landmark_dataset_path, parent_dir, image_file), landmark_image)

with open('refined_no_landmarks.json', 'w') as f:
    json.dump(no_lm, f, indent=4)

    # print(landmark_np.shape)
    # landmark_np = np.array(landmark_result.face_landmarks[0])
    # print(landmark_np.shape)
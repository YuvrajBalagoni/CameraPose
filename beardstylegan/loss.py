import lpips as lpips_loss
import torch
import torch.nn as nn

def get_discriminator_incorrect_age_loss(discriminator, criterion, input, target, incorrect_landmark):
    """
    Calculates the loss for the discriminator when presented with a real image 
    but an incorrect age label. This forces the discriminator to learn the age condition.
    """
    # The discriminator should classify this combination as FAKE.
    # We use the real `target_image` but pair it with the wrong age (`input_age`).
    # incorrect_age_pred = discriminator(input.float(), target.float(), input_age)
    incorrect_age_pred = discriminator(target.float(), incorrect_landmark)
    incorrect_age_loss = criterion(incorrect_age_pred, torch.zeros_like(incorrect_age_pred))
    return incorrect_age_loss


def get_discriminator_loss(discriminator, criterion, output, input, target, target_landmark, incorrect_landmark):
    # discriminator_fake_pred = discriminator(input.float(), output.float(), target_age)
    discriminator_fake_pred = discriminator(output.float(), target_landmark)
    discriminator_fake_loss = criterion(discriminator_fake_pred, torch.zeros_like(discriminator_fake_pred))

    # discriminator_real_pred = discriminator(input.float(), target.float(), target_age)
    discriminator_real_pred = discriminator(target.float(), target_landmark)
    discriminator_real_loss = criterion(discriminator_real_pred, torch.ones_like(discriminator_real_pred))

    discriminator_incorrect_age_loss = get_discriminator_incorrect_age_loss(
        discriminator, criterion, input, target, incorrect_landmark
    )
    
    discriminator_loss = (discriminator_fake_loss + discriminator_real_loss + 0.5*discriminator_incorrect_age_loss) / 2.5
    return discriminator_loss

def get_generator_loss(discriminator, criterion, output, input, target_landmark):
    # discriminator_fake_pred = discriminator(input.float(), output.float(), target_age)
    discriminator_fake_pred = discriminator(output.float(), target_landmark)
    generator_loss = criterion(discriminator_fake_pred, torch.ones_like(discriminator_fake_pred))
    return generator_loss

# ============================================================================================================ #

# def get_landmark_coords(
#         landmarks: list[NormalizedLandmark], width: int, height: int
#     ) -> np.ndarray:
#     """Extract normalized landmark coordinates to array of pixel coordinates."""
#     xyz = [(lm.x, lm.y, lm.z) for lm in landmarks]
#     return np.multiply(xyz, [width, height, width]).astype(int)

# def draw_landmarks(landmark_coordinates, image_width, image_height):
#     image_np = np.zeros((image_height, image_width, 3), dtype=np.uint8)
#     for (x, y, z) in landmark_coordinates:
#         cv2.circle(image_np, (x, y), 1, (255, 255, 255), -1)
#     return image_np

# def get_landmark(options, output_image):
#     output_numpy_batch = output_image.clone().cpu().permute(0, 2, 3, 1).numpy()
#     batch_size = output_numpy_batch.shape[0]
#     landmark_image_list = []

#     with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
#         for i in range(batch_size):
#             image_array = output_numpy_batch[i].astype(np.uint8)
#             image_height, image_width, _ = image_array.shape
#             mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)
#             landmark_result = landmarker.detect(mp_image)

#             if len(landmark_result.face_landmarks) == 0:
#                 # Append a blank image if no face is detected
#                 blank_landmark_image = np.zeros((3, image_height, image_width), dtype=np.uint8)
#                 landmark_image_list.append(blank_landmark_image)
#                 continue

#             landmark_np = get_landmark_coords(
#                 landmark_result.face_landmarks[0], image_width, image_height
#             )
#             landmark_image = draw_landmarks(landmark_np, image_width, image_height)
#             landmark_image = landmark_image.transpose((2, 0, 1))
#             landmark_image_list.append(landmark_image)
    
#     landmark_array = np.stack(landmark_image_list)
#     landmark_tensor = torch.from_numpy(landmark_array).to(output_image.device)
#     return landmark_tensor.float()

# def get_landmark_loss(options, criterion, output_image, target_landmark):
#     predicted_landmarks = get_landmark(options, output_image)
#     landmark_loss = criterion(predicted_landmarks, target_landmark)
#     return landmark_loss

# ============================================================================================================ #

# Todo: Move to class init
lpips_loss_function = lpips_loss.LPIPS(net = 'vgg').to('cuda')
l1 = nn.L1Loss().to('cuda')

def loss_function(output, discriminator, input, target, target_landmark):
    lambda_l1 = 10
    lambda_lpips = 10
    lambda_adversarial = 5

    l1_loss = l1(output, target)
    lpips = lpips_loss_function(output, target).mean()
    adversarial_loss = get_generator_loss(discriminator, nn.BCELoss(), output, input, target_landmark)

    total_loss = lambda_l1 * l1_loss + lambda_lpips * lpips + lambda_adversarial * adversarial_loss
    return total_loss, l1_loss, lpips, adversarial_loss

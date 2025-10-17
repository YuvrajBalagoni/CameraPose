# import os
# import torch
# from torchvision import transforms as T
# import torchvision.transforms.functional as TF
# from PIL import Image
# import utils 

# # This helper function remains the same
# def preprocess_and_get_image(path):
#     """Opens an image, converts it to RGB, and returns it as a PyTorch tensor."""
#     if not os.path.isfile(path):
#         print(f"Input image {path} not found. Skipping.")
#         return None

#     try:
#         image = Image.open(path).convert('RGB')
#         image = TF.to_tensor(image)
#         return image
#     except Exception as e:
#         print(f"Error processing image {path}: {e}. Skipping.")
#         return None

# def run_inference_on_directory(input_dir, target_scores, output_dir, checkpoint_path):
#     """
#     Performs inference for all images in a directory with all specified target scores.

#     Args:
#         input_dir (str): Path to the directory containing input images.
#         target_scores (list): A list of scores (e.g., [1, 8, 9]) to apply.
#         output_dir (str): Path to the directory where results will be saved.
#         checkpoint_path (str): Path to the model checkpoint file.
#     """
#     # --- 1. Setup and Model Loading ---
#     if not os.path.isdir(input_dir):
#         print(f"Error: Input directory '{input_dir}' not found.")
#         return

#     # Create the output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
    
#     print("Loading model...")
#     generator, _, _, _, _ = utils.create_models_and_optimizers(checkpoint_path)
#     generator.eval()
#     print("Model loaded successfully.")

#     # Get a list of valid image files from the input directory
#     valid_extensions = ('.jpg', '.jpeg', '.png')
#     image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    
#     if not image_files:
#         print(f"No images found in '{input_dir}'.")
#         return

#     print(f"\nFound {len(image_files)} images to process with {len(target_scores)} scores each.\n")

#     # --- 2. Main Loop for Processing ---
#     for filename in image_files:
#         input_image_path = os.path.join(input_dir, filename)
        
#         # Preprocess the input image once
#         input_image_tensor = preprocess_and_get_image(input_image_path)
#         if input_image_tensor is None:
#             continue # Skip if the image failed to load
            
#         input_image_tensor = input_image_tensor.to(utils.device).unsqueeze(0)
        
#         # Get the base name of the file to create new output names
#         base_name, _ = os.path.splitext(filename)

#         # Loop through each target score for the current image
#         for score in target_scores:
#             print(f"Processing '{filename}' with score {score}...")
            
#             with torch.no_grad(): # Disable gradient calculation for efficiency
#                 output_image_tensor = generator(input_image_tensor, float(score))

#             # Convert tensor back to a PIL image
#             output_image = TF.to_pil_image(output_image_tensor.squeeze())
            
#             # --- 3. Saving the Result ---
#             output_filename = f"{base_name}_{score}.jpg"
#             output_path = os.path.join(output_dir, output_filename)
            
#             output_image.save(output_path)
#             print(f"✅ Result saved to: {output_path}\n")

#     print("--- All images processed. ---")


# # --- HOW TO USE ---

# # 1. Set the path to your model checkpoint
# checkpoint_path = '/home/shadab/multihairgan_run_3.12_3_6.pt'

# # 2. Set the path to your folder of input images
# input_directory = '/home/shadab/multihairtest'

# # 3. Define the list of scores you want to apply
# scores_to_try = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# # 4. Set the path where you want to save the results
# output_directory = '/home/shadab/multihairoutput/'

# # 5. Call the function
# run_inference_on_directory(input_directory, scores_to_try, output_directory, checkpoint_path)


import os
import torch
from torchvision import transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import utils 

# --- UPDATED HELPER FUNCTION ---
def preprocess_and_get_image(path, max_size=1024):
    """
    Opens an image, resizes it to a maximum size while maintaining aspect ratio,
    converts it to RGB, and returns it as a PyTorch tensor.
    """
    if not os.path.isfile(path):
        print(f"Input image {path} not found. Skipping.")
        return None

    try:
        image = Image.open(path).convert('RGB')

        # --- RESIZING LOGIC ---
        # Get original image dimensions
        original_width, original_height = image.size

        # Check if the longest side is greater than max_size
        if max(original_width, original_height) > max_size:
            # Calculate the scaling ratio
            ratio = max_size / max(original_width, original_height)
            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)
            
            # Resize the image using a high-quality filter
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"Resized '{os.path.basename(path)}' from {original_width}x{original_height} to {new_width}x{new_height}")

        image = TF.to_tensor(image)
        return image
        
    except Exception as e:
        print(f"Error processing image {path}: {e}. Skipping.")
        return None

# --- YOUR MAIN FUNCTION (NO CHANGES NEEDED) ---
def run_inference_on_directory(input_dir, target_scores, output_dir, checkpoint_path):
    """
    Performs inference for all images in a directory with all specified target scores.

    Args:
        input_dir (str): Path to the directory containing input images.
        target_scores (list): A list of scores (e.g., [1, 8, 9]) to apply.
        output_dir (str): Path to the directory where results will be saved.
        checkpoint_path (str): Path to the model checkpoint file.
    """
    # --- 1. Setup and Model Loading ---
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading model...")
    generator, _, _, _, _ = utils.create_models_and_optimizers(checkpoint_path)
    generator.eval()
    print("Model loaded successfully.")

    # Get a list of valid image files from the input directory
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"No images found in '{input_dir}'.")
        return

    print(f"\nFound {len(image_files)} images to process with {len(target_scores)} scores each.\n")

    # --- 2. Main Loop for Processing ---
    for filename in image_files:
        input_image_path = os.path.join(input_dir, filename)
        
        # Preprocess the input image once (this now includes resizing)
        input_image_tensor = preprocess_and_get_image(input_image_path)
        if input_image_tensor is None:
            continue # Skip if the image failed to load
            
        input_image_tensor = input_image_tensor.to(utils.device).unsqueeze(0)
        
        # Get the base name of the file to create new output names
        base_name, _ = os.path.splitext(filename)

        # Loop through each target score for the current image
        for score in target_scores:
            print(f"Processing '{filename}' with score {score}...")
            
            with torch.no_grad(): # Disable gradient calculation for efficiency
                output_image_tensor = generator(input_image_tensor, float(score))

            # Convert tensor back to a PIL image
            output_image = TF.to_pil_image(output_image_tensor.squeeze())
            
            # --- 3. Saving the Result ---
            output_filename = f"{base_name}_{score}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            
            output_image.save(output_path)
            print(f"✅ Result saved to: {output_path}\n")

    print("--- All images processed. ---")


# --- HOW TO USE ---

# 1. Set the path to your model checkpoint
checkpoint_path = '/home/shadab/multihairgan_run_3.12_3_6.pt'

# 2. Set the path to your folder of input images
input_directory = '/home/shadab/multihairtest'

# 3. Define the list of scores you want to apply
scores_to_try = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# 4. Set the path where you want to save the results
output_directory = '/home/shadab/multihairoutput/'

# 5. Call the function
run_inference_on_directory(input_directory, scores_to_try, output_directory, checkpoint_path)
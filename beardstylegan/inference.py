import os
import torch
from torchvision import transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import utils 

# Todo: factor out code that is common for training and inference. 

def preprocess_and_get_image(path):
    if not os.path.isfile(path):
        print(f"Input image {path} not found")
        exit()

    image = Image.open(path).convert('RGB')
    image = TF.to_tensor(image)
    return image
    

def run(input_image_path, landmark_path, checkpoint_path=None, generator=None, output_path=None):
    if generator is None and checkpoint_path is None: 
        print("Specify a model or model path for inference")
        exit()

    if generator is None:
        generator, _, _, _, _ = utils.create_models_and_optimizers(checkpoint_path)
        generator.eval()
        
    input_image = preprocess_and_get_image(input_image_path).to(utils.device).unsqueeze(0)
    landmark = preprocess_and_get_image(landmark_path).to(utils.device).unsqueeze(0)
    output_image = generator(input_image, landmark)
 
    print("Output ranges before norm are ", output_image.min().item(), output_image.max().item())
    # output_clamped = output.clamp(0, 1)
    output_image = output_image * 0.5 + 0.5
    print("Output ranges after norm are ", output_image.min().item(), output_image.max().item())

    # os.makedirs('output', exist_ok=True)
    output_image = TF.to_pil_image(output_image.squeeze())
    
    # output_prefix = f"{output_prefix}_{target_age}" 
    # output_path = "output/" + output_prefix
    # output_name = f"{output_path}_{os.path.basename(input_image_path)}"
    output_image.save(output_path)
    print("Output saved at ", output_path)
    
    return output_image

# checkpoint_path = '/home/shadab/beardstylegan/checkpoints/beardstylegan_run_1.14_5_1.pt'
# run('/home/shadab/jk_test_roi.png', 9, checkpoint_path)
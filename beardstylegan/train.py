import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm.auto import tqdm
import mediapipe as mp

import loss
import video_dataset
import utils
import inference
import wandb

#d = Discriminator(in_channels=4).to('cuda')
#criterion = nn.BCEWithLogitsLoss(reduction='mean') 
#x = torch.rand(2, 3, 512, 512).to('cuda')

# Todo: factor out code that is common for training and inference.

""" 
1 - (-0.3 to -0.3)
2 - (-0.3 to 0.0)
3 - (-0.3 to 0.3)
4 - (-0.15 to -0.15)
5 - (-0.15 to 0.0)
6 - (-0.15 to 0.15)
7 - (0.0 to -0.3)
8 - (0.0 to -0.15)
9 - (0.0 to 0.3)
10 - (0.0 to 0.15)
11 - (0.3 to -0.3)
12 - (0.3 to 0.0)
13 - (0.3 to 0.3)
14 - (0.15 to -0.15)
15 - (0.15 to 0.0)
16 - (0.15 to 0.15)
""" 
wandb.init(project="CPModel", name="run_4.3", config={
    "epochs": 1000,
    "batch_size": 5,
    "description": "Refined dataset",
})

data_loaders=video_dataset.get_dataloaders()
train_loader=data_loaders['train']
criterion = nn.BCELoss()

def log_metrics(iteration, saved_discriminator_loss, saved_generator_loss, l1_loss, lpips_loss, adversarial_loss):
    print(f"Losses: d = {saved_discriminator_loss:.5f}, g = {saved_generator_loss:.5f}, l1 = {l1_loss:.5f}, lpips = {lpips_loss:.5f}, adv = {adversarial_loss:.5f}")
    wandb.log({
        "iteration": iteration,
        "discriminator_loss": saved_discriminator_loss,
        "generator_loss": saved_generator_loss,
        "l1_loss": l1_loss,
        "lpips_loss": lpips_loss,
        "adversarial_loss": adversarial_loss
    })


def save_output(epoch, iteration, run_name, samples, output, generator):
    generator.eval()
    output_prefix = f"{run_name}_{epoch}_{iteration // 5000}"
    output_dir = "infer_output_test_img"
    testim_1_score_1 = inference.run('infer_imgs/LiamNeesonJ118_0.jpg', 'infer_landmark/-20_yaw_latest/landmark_-20_1.5_-2_LiamNeesonJ118_0.jpg', generator=generator, output_path=f"{output_dir}/inference_LiamNeeson_{output_prefix}.jpg")
    testim_1_score_2 = inference.run('temp_dataset/train/7n619EfuSPw_16/frame_00135.jpg', 'temp_dataset_landmark/train/7n619EfuSPw_16/frame_00192.jpg', generator=generator, output_path=f"{output_dir}/inference_Asmilekid_0_{output_prefix}.jpg")
    testim_1_score_2 = inference.run('temp_dataset/train/7n619EfuSPw_16/frame_00192.jpg', 'temp_dataset_landmark/train/7n619EfuSPw_16/frame_00135.jpg', generator=generator, output_path=f"{output_dir}/inference_Asmilekid_1_{output_prefix}.jpg")
    testim_1_score_3 = inference.run('temp_dataset/train/7tW66KxBOaQ_9/frame_00066.jpg', 'temp_dataset_landmark/train/7tW66KxBOaQ_9/frame_00088.jpg', generator=generator, output_path=f"{output_dir}/inference_lady_0_{output_prefix}.jpg")
    testim_1_score_3 = inference.run('temp_dataset/train/7tW66KxBOaQ_9/frame_00088.jpg', 'temp_dataset_landmark/train/7tW66KxBOaQ_9/frame_00066.jpg', generator=generator, output_path=f"{output_dir}/inference_lady_1_{output_prefix}.jpg")
    # testim_1_score_4 = inference.run('CelebV-HQ/video_data/test/_0tf2n3rlJU_0/00060.jpg', 'CelebV-HQ/video_data_landmark/test/_0tf2n3rlJU_0/00000.jpg', generator=generator, output_path=f"{output_dir}/inference_{output_prefix}_60_00.jpg")
    # testim_1_score_5 = inference.run('CelebV-HQ/video_data/test/_0tf2n3rlJU_0/00045.jpg', 'CelebV-HQ/video_data_landmark/test/_0tf2n3rlJU_0/00075.jpg', generator=generator, output_path=f"{output_dir}/inference_{output_prefix}_45_75.jpg")
    # testim_1_score_6 = inference.run('CelebV-HQ/video_data/test/_0tf2n3rlJU_0/00000.jpg', 'CelebV-HQ/video_data_landmark/test/_0tf2n3rlJU_0/00045.jpg', generator=generator, output_path=f"{output_dir}/inference_{output_prefix}_00_45.jpg")
    # testim_1_score_7 = inference.run('CelebV-HQ/video_data/test/_0tf2n3rlJU_0/00075.jpg', 'CelebV-HQ/video_data_landmark/test/_0tf2n3rlJU_0/00045.jpg', generator=generator, output_path=f"{output_dir}/inference_{output_prefix}_75_45.jpg")
    # testim_1_score_8 = inference.run('refined_dataset/train/_W4Em_fHubY_15/frame_00235.jpg', 'refined_dataset_landmark/train/_W4Em_fHubY_15/frame_00364.jpg', generator=generator, output_path=f"{output_dir}/inference_train_sample_{output_prefix}.jpg")
    # testim_1_score_9 = inference.run('refined_dataset/train/_r-SkfDZphk_2/frame_00006.jpg', 'refined_dataset/train/_r-SkfDZphk_2/frame_00041.jpg', generator=generator, output_path=f"{output_dir}/inference_train_sample_2_{output_prefix}.jpg")
    # testim_1_score_10 = inference.run('CelebV-HQ/video_data/train/_bkBt4Z6NQ8_3/00075.jpg', 'CelebV-HQ/video_data/train/_bkBt4Z6NQ8_3/00135.jpg', generator=generator, output_path=f"{output_dir}/inference_Angela_{output_prefix}.jpg")
    # testim_1_score_11 = inference.run('CPDataset/train/1/000024.png', 'CPDataset_landmark/train/11/000024.png', generator=generator, output_path=f"{output_dir}/inference_{output_prefix}_1_11.jpg")
    generator.train()

    os.makedirs('output', exist_ok=True)
    output_image = torch.concat((samples['input_image'][0, 0:3], output[0], samples['target_image'][0]), 2)
    output_image = output_image * 0.5 + 0.5
    output_image = TF.to_pil_image(output_image)
    output_image.save(f"output/output_{epoch+1}_{iteration//5000}.jpg")
    
    # wandb.log({
    #     "A) Training Example": wandb.Image(
    #         output_image,
    #         caption=f"Input | Generated | Target -- Epoch: {epoch+1}, Iter: {iteration}"
    #     ),
    #     "B) Test Image 1/Score 1": wandb.Image(testim_1_score_1),
    #     "B) Test Image 1/Score 6": wandb.Image(testim_1_score_6),
    #     "B) Test Image 1/Score 7": wandb.Image(testim_1_score_7),
    #     "B) Test Image 1/Score 8": wandb.Image(testim_1_score_8),
    #     "B) Test Image 1/Score 9": wandb.Image(testim_1_score_9),
    #     "C) Test Image 2/Score 1": wandb.Image(testim_2_score_1),
    #     "C) Test Image 2/Score 6": wandb.Image(testim_2_score_6),
    #     "C) Test Image 2/Score 7": wandb.Image(testim_2_score_7),
    #     "C) Test Image 2/Score 8": wandb.Image(testim_2_score_8),
    #     "C) Test Image 2/Score 9": wandb.Image(testim_2_score_9),
    #     "D) Test Image 3/Score 1": wandb.Image(testim_3_score_1),
    #     "D) Test Image 3/Score 6": wandb.Image(testim_3_score_6),
    #     "D) Test Image 3/Score 7": wandb.Image(testim_3_score_7),
    #     "D) Test Image 3/Score 8": wandb.Image(testim_3_score_8),
    #     "D) Test Image 3/Score 9": wandb.Image(testim_3_score_9),
    #     "E) Test Image 4/Score 1": wandb.Image(testim_4_score_1),
    #     "E) Test Image 4/Score 6": wandb.Image(testim_4_score_6),
    #     "E) Test Image 4/Score 7": wandb.Image(testim_4_score_7),
    #     "E) Test Image 4/Score 8": wandb.Image(testim_4_score_8),
    #     "E) Test Image 4/Score 9": wandb.Image(testim_4_score_9),
    #     "F) Test Image 5/Score 1": wandb.Image(testim_5_score_1),
    #     "G) Test Image 6/Score 1": wandb.Image(testim_6_score_1),
    # })

def save_checkpoint(epoch, iteration, run_name, generator, discriminator, generator_optim, discriminator_optim, mean_loss, description):
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = f"checkpoints/LandmarkGAN_deeper_{run_name}_{epoch+1}_{iteration // 5000}.pt"
    torch.save({
        'epoch': epoch,
        'description': description,
        'gen_state_dict': generator.state_dict(),
        'dis_state_dict': discriminator.state_dict(),
        'gen_optim_state_dict': generator_optim.state_dict(),
        'dis_optim_state_dict': discriminator_optim.state_dict(),
        'loss': mean_loss,
    }, checkpoint_path)

def train():
    run_name = "run_4.3"
    run_description = "Refined dataset"
    n_epochs = 1000
    checkpoint_path = "checkpoints/LandmarkGAN_deeper_run_4.2_151_0.pt" # or specify path to checkpoint to resume training

    generator, discriminator, generator_optim, discriminator_optim, _ = utils.create_models_and_optimizers(checkpoint_path=checkpoint_path)
    generator.train()
    discriminator.train()

    for epoch in range(n_epochs):
        iteration = 1
        print(f"Epoch {epoch+1}")

        mean_generator_loss = 0
        mean_discriminator_loss = 0

        for samples in tqdm(train_loader):
            output = generator(samples['input_image'], samples['landmark_target_image'])

            # Train discriminator
            discriminator_optim.zero_grad()
            discriminator_loss = loss.get_discriminator_loss(discriminator, criterion, output.detach(), samples['input_image'], samples['target_image'])
            discriminator_loss.backward()
            discriminator_optim.step()
            mean_discriminator_loss += discriminator_loss.item() / len(train_loader)

            # Train generator
            generator_optim.zero_grad()
            generator_loss, l1_loss, lpips_loss, adversarial_loss = loss.loss_function(
                output, discriminator, samples['input_image'], samples['target_image']
            )
            generator_loss.backward()
            generator_optim.step()
            mean_generator_loss += generator_loss.item() / len(train_loader)

            # Log metrics
            if iteration % 10 == 0:
                log_metrics(iteration, discriminator_loss.item(), generator_loss.item(), l1_loss, lpips_loss, adversarial_loss)

            # Save outputs and checkpoints
            if epoch % 10 == 0 and iteration == 1:
                save_output(epoch, iteration, run_name, samples, output, generator)

            if epoch % 10 == 0 and iteration == 1:
                save_checkpoint(epoch, iteration, run_name, generator, discriminator, generator_optim, discriminator_optim,
                                mean_generator_loss + mean_discriminator_loss, run_description)

            iteration += 1

train()

"""
python beardstylegan/train.py
"""
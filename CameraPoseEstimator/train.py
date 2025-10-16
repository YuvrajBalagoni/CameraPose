import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import random
from tqdm.auto import tqdm
import wandb

import dataset
import model
 
wandb.init(project="CameraPoseEstimator", name="run_1.0", config={
    "epochs": 100,
    "batch_size": 64,
    "description": "first run",
})

data_loaders=dataset.get_dataloaders()
train_loader=data_loaders['train']
test_loader=data_loaders['test']
criterion = nn.L1Loss()

def log_metrics(epoch, iteration, running_loss, test_loss):
    print(f"Epoch [{epoch}], Iteration [{iteration}], train_Loss: {running_loss:.4f}, Test Loss: {test_loss:.4f}")
    wandb.log({
        "epoch": epoch,
        "iteration": iteration,
        "train_loss": running_loss,
        "test_loss": test_loss,
    })

def  save_checkpoint(model, optimizer, epoch, iteration, path):
    torch.save({
        'epoch': epoch,
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved at {path}")
    wandb.save(path)

def train():
    run_name = "run_1.0"
    run_description = "first run"
    n_epochs = 100
    checkpoint_path = None  # or specify path to checkpoint to resume training

    model_cp = model.CameraPoseEstimator()
    model_cp = model_cp.to('cuda')

    optimizer = torch.optim.Adam(model_cp.parameters(), lr=0.0001)

    for epoch in range(n_epochs):
        print(f"epoch : {epoch + 1}")
        train_loss = 0.0
        model_cp.train()
        
        print(f"========== Train Loop ==========")
        for i, (inputs, labels) in enumerate(tqdm(train_loader)):
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            
            optimizer.zero_grad()
            outputs = model_cp(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if i % 100 == 0:
                log_metrics(epoch+1, i+1, train_loss/(i+1), 0.0)  # test_loss will be updated later
            
            if i % 1000 == 0 and i > 0:
                checkpoint_path = f"checkpoints/camera_pose_estimator_{run_name}_epoch{epoch+1}_iter{i+1}.pt"
                save_checkpoint(model_cp, optimizer, epoch+1, i+1, checkpoint_path)

        train_loss /= len(train_loader)

        print(f"========== Validation Loop ==========")
        model_cp.eval()
        with torch.no_grad():
            test_loss = 0.0
            for i, (val_inputs, val_labels) in enumerate(tqdm(test_loader)):
                val_inputs, val_labels = val_inputs.to('cuda'), val_labels.to('cuda')
                val_outputs = model_cp(val_inputs)
                v_loss = criterion(val_outputs, val_labels)
                test_loss += v_loss.item()

                if i % 10 == 0:
                    log_metrics(epoch+1, i+1, train_loss, test_loss/(i+1))  # train_loss is from training loop
                
                if i == 10:
                    break # Limiting to 10 batches for quick validation
            test_loss /= len(test_loader)

train()

"""
python CameraPoseEstimator/train.py
"""
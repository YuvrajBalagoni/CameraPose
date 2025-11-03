import torch
import os

from ageGAN import AgeGAN
from discriminator import Discriminator


device= 'cuda' if torch.cuda.is_available() else 'cpu'


def load_checkpoint(checkpoint_path, generator, generator_optim, discriminator, discriminator_optim): 
    assert os.path.isfile(checkpoint_path), f"Checkpoint file {checkpoint_path} not found" 
    state=torch.load(checkpoint_path) 
 
    generator.load_state_dict(state['gen_state_dict']) 
    generator_optim.load_state_dict(state['gen_optim_state_dict']) 

    # discriminator_state = state['dis_state_dict']
    # initial_layer = discriminator_state['layers.0.0.weight']
    # additional_layers = torch.full([64, 3, 4, 4], torch.mean(initial_layer)).to('cuda')
    # new_layer_weight = torch.cat([initial_layer, additional_layers], dim=1)
    # discriminator_state['layers.0.0.weight'] = new_layer_weight
    
    # old = False
    # if old:
    #     discriminator.load_state_dict(discriminator_state)
    # else:
    discriminator.load_state_dict(state['dis_state_dict']) 
    discriminator_optim.load_state_dict(state['dis_optim_state_dict']) 
 
    return generator, discriminator, generator_optim, discriminator_optim, state['epoch'] 
 
 
def create_models_and_optimizers(checkpoint_path=None): 
    # Generator 
    generator = AgeGAN().to(device) 
    generator_optim = torch.optim.Adam(generator.parameters(), lr=0.00003) 
     
    # Discriminator 
    discriminator = Discriminator(in_channels=6).to(device) 
    discriminator_optim = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-6) 
 
    if checkpoint_path is not None: 
        return load_checkpoint(checkpoint_path, generator, generator_optim, discriminator, discriminator_optim) 
    else: 
        print("Training model from scratch")
        # Todo: Initialize weights 
        return generator, discriminator, generator_optim, discriminator_optim, 0 
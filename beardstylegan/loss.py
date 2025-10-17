import lpips as lpips_loss
import torch
import torch.nn as nn

def get_discriminator_incorrect_age_loss(discriminator, criterion, input, target):
    """
    Calculates the loss for the discriminator when presented with a real image 
    but an incorrect age label. This forces the discriminator to learn the age condition.
    """
    # The discriminator should classify this combination as FAKE.
    # We use the real `target_image` but pair it with the wrong age (`input_age`).
    # incorrect_age_pred = discriminator(input.float(), target.float(), input_age)
    incorrect_age_pred = discriminator(target.float())
    incorrect_age_loss = criterion(incorrect_age_pred, torch.zeros_like(incorrect_age_pred))
    return incorrect_age_loss


def get_discriminator_loss(discriminator, criterion, output, input, target):
    # discriminator_fake_pred = discriminator(input.float(), output.float(), target_age)
    discriminator_fake_pred = discriminator(output.float())
    discriminator_fake_loss = criterion(discriminator_fake_pred, torch.zeros_like(discriminator_fake_pred))

    # discriminator_real_pred = discriminator(input.float(), target.float(), target_age)
    discriminator_real_pred = discriminator(target.float())
    discriminator_real_loss = criterion(discriminator_real_pred, torch.ones_like(discriminator_real_pred))

    discriminator_incorrect_age_loss = get_discriminator_incorrect_age_loss(
        discriminator, criterion, input, target
    )
    
    discriminator_loss = (discriminator_fake_loss + discriminator_real_loss + 0.5*discriminator_incorrect_age_loss) / 2.5
    return discriminator_loss

def get_generator_loss(discriminator, criterion, output, input):
    # discriminator_fake_pred = discriminator(input.float(), output.float(), target_age)
    discriminator_fake_pred = discriminator(output.float())
    generator_loss = criterion(discriminator_fake_pred, torch.ones_like(discriminator_fake_pred))
    return generator_loss

# Todo: Move to class init
lpips_loss_function = lpips_loss.LPIPS(net = 'vgg').to('cuda')
l1 = nn.L1Loss().to('cuda')

def loss_function(output, discriminator, input, target):
    lambda_l1 = 10
    lambda_lpips = 1
    lambda_adversarial = 1

    l1_loss = l1(output, target)
    lpips = lpips_loss_function(output, target).mean()
    adversarial_loss = get_generator_loss(discriminator, nn.BCELoss(), output, input)

    total_loss = lambda_l1 * l1_loss + lambda_lpips * lpips + lambda_adversarial * adversarial_loss
    return total_loss, l1_loss, lpips, adversarial_loss

"""
Training continuation script for CycleGAN H&E conversion.
Continues training from checkpoint if latest.pth exists.
"""
import os
import yaml
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from PIL import Image

from models.cyclegan import CycleGAN
from utils.data_loader import create_data_loaders
from utils.image_processing import tensor_to_numpy, save_image


class GANLoss(nn.Module):
    """GAN loss (adversarial loss)."""
    def __init__(self, target_real_label: float = 1.0, target_fake_label: float = 0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()
    
    def __call__(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        
        target_tensor = target_tensor.expand_as(prediction)
        return self.loss(prediction, target_tensor)


class CycleGANLoss:
    """CycleGAN loss functions."""
    def __init__(self, lambda_cycle: float = 10.0, lambda_identity: float = 0.5):
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.criterionGAN = GANLoss()
        self.criterionCycle = nn.L1Loss()
        self.criterionIdt = nn.L1Loss()
    
    def compute_generator_loss(self, model_output: dict, real_A: torch.Tensor, 
                               real_B: torch.Tensor) -> dict:
        """
        Compute generator losses.
        
        Args:
            model_output: Output from CycleGAN forward pass
            real_A: Real images from domain A
            real_B: Real images from domain B
        
        Returns:
            Dictionary of losses
        """
        fake_B = model_output['fake_B']
        fake_A = model_output['fake_A']
        rec_A = model_output['rec_A']
        rec_B = model_output['rec_B']
        idt_A = model_output['idt_A']
        idt_B = model_output['idt_B']
        
        # Adversarial losses
        pred_fake_B = model_output.get('pred_fake_B', None)
        pred_fake_A = model_output.get('pred_fake_A', None)
        
        if pred_fake_B is not None:
            loss_G_A2B = self.criterionGAN(pred_fake_B, True)
        else:
            loss_G_A2B = torch.tensor(0.0, device=real_A.device)
        
        if pred_fake_A is not None:
            loss_G_B2A = self.criterionGAN(pred_fake_A, True)
        else:
            loss_G_B2A = torch.tensor(0.0, device=real_A.device)
        
        # Cycle consistency losses
        loss_cycle_A = self.criterionCycle(rec_A, real_A) * self.lambda_cycle
        loss_cycle_B = self.criterionCycle(rec_B, real_B) * self.lambda_cycle
        
        # Identity losses
        loss_idt_A = self.criterionIdt(idt_A, real_A) * self.lambda_cycle * self.lambda_identity
        loss_idt_B = self.criterionIdt(idt_B, real_B) * self.lambda_cycle * self.lambda_identity
        
        # Total generator loss
        loss_G = loss_G_A2B + loss_G_B2A + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
        
        return {
            'loss_G': loss_G,
            'loss_G_A2B': loss_G_A2B,
            'loss_G_B2A': loss_G_B2A,
            'loss_cycle_A': loss_cycle_A,
            'loss_cycle_B': loss_cycle_B,
            'loss_idt_A': loss_idt_A,
            'loss_idt_B': loss_idt_B
        }
    
    def compute_discriminator_loss(self, pred_real: torch.Tensor, 
                                   pred_fake: torch.Tensor) -> torch.Tensor:
        """
        Compute discriminator loss.
        
        Args:
            pred_real: Predictions on real images
            pred_fake: Predictions on fake images
        
        Returns:
            Discriminator loss
        """
        loss_D_real = self.criterionGAN(pred_real, True)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D


def get_scheduler(optimizer, n_epochs: int, n_epochs_decay: int):
    """Learning rate scheduler with linear decay."""
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch - n_epochs_decay) / float(n_epochs - n_epochs_decay + 1)
        return lr_l
    
    return LambdaLR(optimizer, lr_lambda=lambda_rule)


def save_checkpoint(model: CycleGAN, optimizers: dict, epoch: int, 
                   checkpoint_dir: str, is_best: bool = False):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'G_A2B_state_dict': model.G_A2B.state_dict(),
        'G_B2A_state_dict': model.G_B2A.state_dict(),
        'D_A_state_dict': model.D_A.state_dict(),
        'D_B_state_dict': model.D_B.state_dict(),
        'optimizer_G_state_dict': optimizers['G'].state_dict(),
        'optimizer_D_A_state_dict': optimizers['D_A'].state_dict(),
        'optimizer_D_B_state_dict': optimizers['D_B'].state_dict(),
    }
    
    # Save latest
    torch.save(checkpoint, os.path.join(checkpoint_dir, 'latest.pth'))
    
    # Save epoch checkpoint
    if is_best or epoch % 10 == 0:
        torch.save(checkpoint, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'))


def load_checkpoint(checkpoint_path: str, model: CycleGAN, optimizers: dict, device: torch.device):
    """Load model checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model states
    model.G_A2B.load_state_dict(checkpoint['G_A2B_state_dict'])
    model.G_B2A.load_state_dict(checkpoint['G_B2A_state_dict'])
    model.D_A.load_state_dict(checkpoint['D_A_state_dict'])
    model.D_B.load_state_dict(checkpoint['D_B_state_dict'])
    
    # Load optimizer states
    optimizers['G'].load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizers['D_A'].load_state_dict(checkpoint['optimizer_D_A_state_dict'])
    optimizers['D_B'].load_state_dict(checkpoint['optimizer_D_B_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0)
    print(f"Checkpoint loaded. Resuming from epoch {start_epoch + 1}")
    
    return start_epoch


def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def train_continue():
    """Main training continuation function."""
    # Checkpoint path
    checkpoint_path = r'D:\A_WORK\task\checkpoints\latest.pth'
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Cannot continue training. Please run train.py first or ensure checkpoint exists.")
        return
    
    # Load configuration
    config = load_config()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create data loaders
    print('Loading data...')
    fluorescence_loader, he_loader = create_data_loaders(
        c01_path=config['data']['c01_path'],
        c02_path=config['data']['c02_path'],
        he_path=config['data']['he_stained_path'],
        batch_size=config['training']['batch_size'],
        image_size=config['image']['image_size'],
        normalize_range=tuple(config['image']['normalize_range']),
        num_workers=config['image']['num_workers'],
        is_training=True
    )
    
    print(f'Fluorescence images: {len(fluorescence_loader.dataset)}')
    if he_loader:
        print(f'H&E images: {len(he_loader.dataset)}')
    else:
        print('Warning: No H&E images found. Training will use only fluorescence images.')
        print('Consider using external H&E datasets or synthetic H&E images.')
    
    # Create model
    model = CycleGAN(
        input_nc=config['model']['input_nc'],
        output_nc=config['model']['output_nc'],
        ngf=config['model']['ngf'],
        ndf=config['model']['ndf'],
        n_residual_blocks=config['model']['n_residual_blocks'],
        dropout=config['model']['dropout']
    ).to(device)
    
    # Loss function
    loss_fn = CycleGANLoss(
        lambda_cycle=config['training']['lambda_cycle'],
        lambda_identity=config['training']['lambda_identity']
    )
    
    # Optimizers
    optimizer_G = Adam(
        list(model.G_A2B.parameters()) + list(model.G_B2A.parameters()),
        lr=config['training']['lr'],
        betas=(config['training']['beta1'], config['training']['beta2'])
    )
    optimizer_D_A = Adam(
        model.D_A.parameters(),
        lr=config['training']['lr'],
        betas=(config['training']['beta1'], config['training']['beta2'])
    )
    optimizer_D_B = Adam(
        model.D_B.parameters(),
        lr=config['training']['lr'],
        betas=(config['training']['beta1'], config['training']['beta2'])
    )
    
    optimizers = {
        'G': optimizer_G,
        'D_A': optimizer_D_A,
        'D_B': optimizer_D_B
    }
    
    # Load checkpoint
    checkpoint_epoch = load_checkpoint(checkpoint_path, model, optimizers, device)
    
    # Determine starting epoch (must be at least 32)
    if checkpoint_epoch < 32:
        print(f"Checkpoint epoch ({checkpoint_epoch}) is less than 32. Starting from epoch 32.")
        start_epoch = 32
    else:
        print(f"Resuming from checkpoint epoch {checkpoint_epoch + 1}")
        start_epoch = checkpoint_epoch + 1
    
    # Learning rate schedulers (adjusted for continuation)
    num_epochs_total = 200
    schedulers = {
        'G': get_scheduler(optimizer_G, num_epochs_total, 
                          config['training']['lr_decay_epochs']),
        'D_A': get_scheduler(optimizer_D_A, num_epochs_total,
                            config['training']['lr_decay_epochs']),
        'D_B': get_scheduler(optimizer_D_B, num_epochs_total,
                            config['training']['lr_decay_epochs'])
    }
    
    # Step schedulers to the correct epoch (step to match the epoch we're starting from)
    # Scheduler steps after each epoch, so we need to step (start_epoch - 1) times
    for scheduler in schedulers.values():
        for _ in range(start_epoch - 1):
            scheduler.step()
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir='runs')
    
    # Training loop - from epoch 32 to 200
    num_epochs = 200
    print_freq = config['training']['print_freq']
    
    print(f'Continuing training from epoch {start_epoch} to {num_epochs}...')
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        
        # Create iterators
        fluorescence_iter = iter(fluorescence_loader)
        he_iter = iter(he_loader) if he_loader else None
        
        epoch_losses = {
            'loss_G': 0.0,
            'loss_D_A': 0.0,
            'loss_D_B': 0.0
        }
        
        num_batches = len(fluorescence_loader)
        
        for batch_idx in tqdm(range(num_batches), desc=f'Epoch {epoch}/{num_epochs}'):
            # Get batches
            try:
                real_A, _ = next(fluorescence_iter)
            except StopIteration:
                fluorescence_iter = iter(fluorescence_loader)
                real_A, _ = next(fluorescence_iter)
            
            real_A = real_A.to(device)
            
            # Get H&E batch (if available)
            if he_iter:
                try:
                    real_B = next(he_iter)
                    if isinstance(real_B, tuple):
                        real_B = real_B[0]
                    real_B = real_B.to(device)
                except StopIteration:
                    he_iter = iter(he_loader)
                    real_B = next(he_iter)
                    if isinstance(real_B, tuple):
                        real_B = real_B[0]
                    real_B = real_B.to(device)
            else:
                # Use fluorescence images as both domains if no H&E data
                real_B = real_A.clone()
            
            # Forward pass
            model_output = model(real_A, real_B)
            fake_B = model_output['fake_B']
            fake_A = model_output['fake_A']
            
            # Train discriminators
            # Discriminator A
            optimizer_D_A.zero_grad()
            pred_real_A = model.D_A(real_A)
            pred_fake_A = model.D_A(fake_A.detach())
            loss_D_A = loss_fn.compute_discriminator_loss(pred_real_A, pred_fake_A)
            loss_D_A.backward()
            optimizer_D_A.step()
            
            # Discriminator B
            optimizer_D_B.zero_grad()
            pred_real_B = model.D_B(real_B)
            pred_fake_B = model.D_B(fake_B.detach())
            loss_D_B = loss_fn.compute_discriminator_loss(pred_real_B, pred_fake_B)
            loss_D_B.backward()
            optimizer_D_B.step()
            
            # Train generators
            optimizer_G.zero_grad()
            
            # Add discriminator predictions for loss computation
            model_output['pred_fake_B'] = model.D_B(fake_B)
            model_output['pred_fake_A'] = model.D_A(fake_A)
            
            losses = loss_fn.compute_generator_loss(model_output, real_A, real_B)
            losses['loss_G'].backward()
            optimizer_G.step()
            
            # Accumulate losses
            epoch_losses['loss_G'] += losses['loss_G'].item()
            epoch_losses['loss_D_A'] += loss_D_A.item()
            epoch_losses['loss_D_B'] += loss_D_B.item()
            
            # Log to TensorBoard
            global_step = (epoch - 1) * num_batches + batch_idx
            if batch_idx % print_freq == 0:
                writer.add_scalar('Loss/Generator', losses['loss_G'].item(), global_step)
                writer.add_scalar('Loss/Discriminator_A', loss_D_A.item(), global_step)
                writer.add_scalar('Loss/Discriminator_B', loss_D_B.item(), global_step)
        
        # Update learning rates
        for scheduler in schedulers.values():
            scheduler.step()
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        print(f'Epoch {epoch}/{num_epochs}:')
        print(f'  Generator Loss: {epoch_losses["loss_G"]:.4f}')
        print(f'  Discriminator A Loss: {epoch_losses["loss_D_A"]:.4f}')
        print(f'  Discriminator B Loss: {epoch_losses["loss_D_B"]:.4f}')
        
        # Save checkpoint
        save_checkpoint(model, optimizers, epoch, config['data']['checkpoint_path'])
        
        # Save sample images
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                sample_A = real_A[0:1]
                sample_output = model.G_A2B(sample_A)
                # Save sample (implementation in inference script)
    
    print('Training continuation completed!')
    writer.close()


if __name__ == '__main__':
    train_continue()


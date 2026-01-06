"""
CycleGAN model implementation.
Based on the paper: "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with instance normalization."""
    def __init__(self, in_channels: int):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3),
            nn.InstanceNorm2d(in_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Generator(nn.Module):
    """
    Generator network (ResNet-based).
    Transforms images from one domain to another.
    """
    def __init__(self, input_nc: int = 3, output_nc: int = 3, ngf: int = 64,
                 n_residual_blocks: int = 9, dropout: bool = False):
        """
        Args:
            input_nc: Number of input channels
            output_nc: Number of output channels
            ngf: Number of generator filters in first conv layer
            n_residual_blocks: Number of residual blocks
            dropout: Whether to use dropout
        """
        super(Generator, self).__init__()
        
        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(inplace=True)
            ]
        
        # Residual blocks
        mult = 2 ** n_downsampling
        for i in range(n_residual_blocks):
            model += [ResidualBlock(ngf * mult)]
            if dropout and i < 2:
                model += [nn.Dropout(0.5)]
        
        # Upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.ReLU(inplace=True)
            ]
        
        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, 7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Discriminator(nn.Module):
    """
    Discriminator network (PatchGAN).
    Classifies whether image patches are real or fake.
    """
    def __init__(self, input_nc: int = 3, ndf: int = 64, n_layers: int = 3):
        """
        Args:
            input_nc: Number of input channels
            ndf: Number of discriminator filters in first conv layer
            n_layers: Number of layers in discriminator
        """
        super(Discriminator, self).__init__()
        
        model = [
            nn.Conv2d(input_nc, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            model += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, stride=2, padding=1),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        model += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Output layer
        model += [nn.Conv2d(ndf * nf_mult, 1, 4, stride=1, padding=1)]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class CycleGAN(nn.Module):
    """
    Complete CycleGAN model with two generators and two discriminators.
    """
    def __init__(self, input_nc: int = 3, output_nc: int = 3, ngf: int = 64,
                 ndf: int = 64, n_residual_blocks: int = 9, dropout: bool = False):
        """
        Args:
            input_nc: Number of input channels
            output_nc: Number of output channels
            ngf: Number of generator filters
            ndf: Number of discriminator filters
            n_residual_blocks: Number of residual blocks in generator
            dropout: Whether to use dropout
        """
        super(CycleGAN, self).__init__()
        
        # Generators: G_A2B (fluorescence -> H&E), G_B2A (H&E -> fluorescence)
        self.G_A2B = Generator(input_nc, output_nc, ngf, n_residual_blocks, dropout)
        self.G_B2A = Generator(input_nc, output_nc, ngf, n_residual_blocks, dropout)
        
        # Discriminators: D_A (fluorescence domain), D_B (H&E domain)
        self.D_A = Discriminator(input_nc, ndf)
        self.D_B = Discriminator(input_nc, ndf)
    
    def forward(self, real_A: torch.Tensor, real_B: torch.Tensor) -> dict:
        """
        Forward pass through the network.
        
        Args:
            real_A: Real images from domain A (fluorescence)
            real_B: Real images from domain B (H&E)
        
        Returns:
            Dictionary containing:
                - fake_B: Generated H&E images
                - fake_A: Generated fluorescence images
                - rec_A: Reconstructed fluorescence images
                - rec_B: Reconstructed H&E images
                - idt_A: Identity mapping for domain A
                - idt_B: Identity mapping for domain B
        """
        # Forward cycle: A -> B -> A
        fake_B = self.G_A2B(real_A)
        rec_A = self.G_B2A(fake_B)
        
        # Backward cycle: B -> A -> B
        fake_A = self.G_B2A(real_B)
        rec_B = self.G_A2B(fake_A)
        
        # Identity mapping (for identity loss)
        idt_A = self.G_B2A(real_A)
        idt_B = self.G_A2B(real_B)
        
        return {
            'fake_B': fake_B,
            'fake_A': fake_A,
            'rec_A': rec_A,
            'rec_B': rec_B,
            'idt_A': idt_A,
            'idt_B': idt_B
        }


from torch import nn as nn
from torch.nn import functional as F
import torch


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, pad=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, pad),
            nn.InstanceNorm2d(out_channel),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(out_channel, out_channel, kernel_size, stride, pad),
            nn.InstanceNorm2d(out_channel),
            nn.LeakyReLU(0.2),
        )
        self.residual = nn.Identity()
        if in_channel != out_channel:
            self.residual = nn.Conv2d(in_channel, out_channel, kernel_size=1)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.residual(x) + self.conv(x)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Down sampling
        channels = [3, 64, 128, 128, 256, 512]
        blocks = [1, 2, 3, 3, 2]
        
        down_sample = []
        for i in range(len(channels)-1):
            in_channel = channels[i]
            out_channel = channels[i+1]
        
            for _ in range(blocks[i]):
                down_sample.append(ResidualBlock(in_channel, out_channel))
                in_channel = out_channel
            
            down_sample.append(nn.Upsample(scale_factor=0.5))
        
        self.down_sample_module = nn.Sequential(*down_sample)
        
        # up sample
        
        channels = channels[::-1]
        blocks = blocks[::-1]
        
        up_sample = []
        
        
        for i in range(len(channels)-1):
            in_channel = channels[i]
            out_channel = channels[i+1]
        
            for _ in range(blocks[i]):
                up_sample.append(ResidualBlock(in_channel, out_channel))
                in_channel = out_channel
            
            up_sample.append(nn.Upsample(scale_factor=2))

        self.up_sample_module = nn.Sequential(*up_sample)
        
    def forward(self,x:torch.Tensor)->torch.Tensor:
        
        x = self.down_sample_module(x)
        
        x = self.up_sample_module(x)
        
        x = F.tanh(x)
        
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Down sampling
        channels = [3, 64, 128, 128, 256, 512]
        blocks = [1, 2, 3, 3, 2]
        
        down_sample = []
        for i in range(len(channels)-1):
            in_channel = channels[i]
            out_channel = channels[i+1]
        
            for _ in range(blocks[i]):
                down_sample.append(ResidualBlock(in_channel, out_channel))
                in_channel = out_channel
            
            down_sample.append(nn.MaxPool2d(2, 2))
        
        self.down_sample_module = nn.Sequential(*down_sample)
        
        
        self.classifier = nn.Linear(out_channel, 1)
        
    def forward(self,x:torch.Tensor)->torch.Tensor:
        
        x = self.down_sample_module(x)
        
        x = F.adaptive_avg_pool2d(x, (1,1))
        
        x = x.flatten(1)
        
        x = self.classifier(x)
        
        
        return x
    
    
if __name__ == "__main__":
    generator = Generator()
    x = torch.randn(2, 3, 512, 512)
    
    out = generator(x)
    
    print(out.shape)
    
    discriminator = Discriminator()
    
    out = discriminator(out)
    
    print(out.shape)
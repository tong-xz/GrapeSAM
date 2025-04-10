import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet101_Weights

class ResNetBackbone(nn.Module):
    def __init__(self, down=8, o_cn=1, final='abs', pretrained=True):
        super(ResNetBackbone, self).__init__()
        self.down = down
        self.final = final
        
        # Load ResNet101 with weights parameter instead of pretrained
        weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet101(weights=weights)
        
        # Remove the average pooling and FC layers
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        
        # Adapter layer to convert 2048 channels to 512 channels
        self.adapter = nn.Conv2d(2048, 512, kernel_size=1)
        
        # Regression head
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, o_cn, 1)
        )
        self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = self.adapter(x)
        
        # ResNet outputs feature maps at 1/32 scale
        # We need to upsample based on the down parameter
        if self.down < 32:
            scale_factor = 32 // self.down
            x = F.interpolate(x, scale_factor=scale_factor)
        
        x = self.reg_layer(x)
        
        if self.final == 'abs':
            x = torch.abs(x)
        elif self.final == 'relu':
            x = torch.relu(x)
            
        return x
    
    def _initialize_weights(self):
        # Initialize the adapter and regression layer
        for m in [self.adapter] + list(self.reg_layer.modules()):
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def resnet101(down=8, bn=False, o_cn=1, final='abs'):
    """ResNet 101-layer model with pretrained weights from ImageNet
    
    Args:
        down: Downsampling factor (8 default)
        bn: Not used, kept for API compatibility
        o_cn: Output channel number (1 default)
        final: Final activation ('abs' or 'relu')
    
    Returns:
        ResNet101 model with same API as VGG19
    """
    # Create ResNet101 backbone with pretrained weights directly
    model = ResNetBackbone(down=down, o_cn=o_cn, final=final, pretrained=True)
    
    return model
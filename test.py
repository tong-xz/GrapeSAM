import torch
import torch.nn.functional as F

# Create random binary masks with shape (1, 1024, 1024)
masks = torch.randint(0, 2, (1, 1024, 1024), dtype=torch.float32)

masks = F.interpolate(masks.unsqueeze(0), size=(1024, 1024), mode='bilinear').squeeze(0)
areas = (masks > 0.5).float().squeeze().sum(dim=(-1, -2))
indices = torch.argsort(areas)
print(masks.shape)
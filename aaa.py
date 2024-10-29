import torch

# Load model weights
state_dict = torch.load("/home/xz/Dev/Dream/pretrain/sam-vit-base/pytorch_model.bin")
import pdb; pdb.set_trace()
# View the keys (layer names and weight shapes)
for key, value in state_dict.items():
    
    print(f"{key}: {value.shape}")

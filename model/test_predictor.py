from predictor import Trainer, setup_cfg
import torch

cfg = setup_cfg(
    "/data/Hypothesis/theorem/grape/Dream/config/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml",
    ["MODEL.WEIGHTS", "output/model_0214999.pth"],
)
print(cfg)
# model = Mask2FormerRunner(cfg)
model = Trainer.build_model(cfg)


# # Simulate a batched input with two images
# batched_inputs = [
#     {
#         "image": torch.randn(3, 256, 256),  # Random tensor simulating a (C, H, W) image
#         "instances": {
#             "gt_masks": torch.randint(
#                 0, 2, (5, 128, 128), dtype=torch.uint8
#             ),  # Simulating 5 binary masks
#             "gt_classes": torch.tensor(
#                 [1, 0, 2, 3, 4], dtype=torch.int64
#             ),  # Simulating 5 class labels
#         },
#         "height": 256,
#         "width": 256,
#     },
#     {
#         "image": torch.randn(3, 300, 300),  # Another image
#         "instances": {
#             "gt_masks": torch.randint(
#                 0, 2, (3, 150, 150), dtype=torch.uint8
#             ),  # Simulating 3 binary masks
#             "gt_classes": torch.tensor(
#                 [0, 1, 2], dtype=torch.int64
#             ),  # Simulating 3 class labels
#         },
#         "height": 300,
#         "width": 300,
#     },
# ]
import torch
import numpy as np
from detectron2.structures import Instances, Boxes


# Simulated data
def generate_random_input(device="cuda"):
    # Define image size
    height, width = 256, 256
    num_instances = 5

    # Generate a random image tensor (C, H, W)
    image = torch.rand(3, height, width, device=device)

    # Create random boxes (x1, y1, x2, y2)
    boxes = torch.tensor(
        np.random.uniform(0, min(height, width), (num_instances, 4)),
        dtype=torch.float32,
        device=device,
    )
    boxes[:, 2:] += boxes[:, :2]  # Ensure x2 > x1, y2 > y1
    boxes = Boxes(boxes)

    # Random classes for instances
    classes = torch.randint(0, 1, (num_instances,), dtype=torch.int64, device=device)

    # Generate random masks
    masks = torch.randint(
        0, 2, (num_instances, height, width), dtype=torch.uint8, device=device
    )

    # Create Instances object
    instances = Instances((height, width))
    instances.gt_classes = classes
    instances.gt_masks = masks

    # Metadata for the dataset

    # Assemble the batch input
    batched_inputs = [
        {
            "image": image,
            "instances": instances,
            "height": height,
            "width": width,
        }
    ]

    return batched_inputs


batched_inputs = generate_random_input()
pred = model(batched_inputs)
print(pred)
breakpoint()

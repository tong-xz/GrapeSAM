import numpy as np
import matplotlib.pyplot as plt
import gc
import torch
import time
from transformers import SamProcessor, SamModel


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    del mask
    gc.collect()


def show_masks_on_image(raw_image, masks, output_path):
    plt.figure(figsize=(10, 10))
    plt.imshow(np.array(raw_image))
    ax = plt.gca()
    ax.set_autoscale_on(False)

    # Process masks in smaller batches
    batch_size = 10
    for i in range(0, len(masks), batch_size):
        batch_masks = masks[i : i + batch_size]
        for mask in batch_masks:
            show_mask(mask, ax=ax, random_color=True)

        del batch_masks
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

    plt.axis("off")
    plt.savefig(output_path)
    plt.close()
    gc.collect()


from transformers import pipeline


from PIL import Image

img_path = "/home/xz/Downloads/feb-test/converted/IMG_0699.png"
raw_image = Image.open(img_path).convert("RGB")

sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

inputs = sam_processor(images=raw_image, return_tensors="pt").to("cuda")
tgt_shape = inputs["reshaped_input_sizes"][0]
target_height, target_width = tgt_shape[0], tgt_shape[1]

# Resize image to match target dimensions
if raw_image.size != (target_width, target_height):  # PIL uses (width, height) order
    raw_image = raw_image.resize(
        (target_width, target_height), Image.Resampling.LANCZOS
    )

generator = pipeline(
    "mask-generation",
    model="facebook/sam-vit-huge",
    device=0,
)


# Start prediction timing
pred_start_time = time.time()
outputs = generator(raw_image, points_per_batch=256)  # Reduced points_per_batch

pred_time = time.time() - pred_start_time
print(f"Prediction time: {pred_time:.2f} seconds")

# Move masks to CPU and convert to float32
masks = torch.as_tensor(np.array(outputs["masks"], dtype=np.float32), device="cpu")
masks = masks.unsqueeze(1)
print(masks.shape)

# Clear CUDA cache before showing masks
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Start visualization timing
vis_start_time = time.time()
output_path = img_path.rsplit(".", 1)[0] + "_masked2561.png"
show_masks_on_image(raw_image, masks, output_path)
vis_time = time.time() - vis_start_time
print(f"Visualization time: {vis_time:.2f} seconds")
print(f"Total processing time: {pred_time + vis_time:.2f} seconds")

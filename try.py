from PIL import Image
import requests
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained("facebook/sam-vit-base")
processor = AutoProcessor.from_pretrained("facebook/sam-vit-base")

img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/sam-car.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
input_points = [[[400, 650]]]  # 2D location of a window on the car
inputs = processor(images=raw_image, input_points=input_points, return_tensors="pt")

# Get segmentation mask
outputs = model(**inputs)

# Postprocess masks
masks = processor.post_process_masks(
    outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
)

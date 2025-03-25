# GrapeSAM  

**GrapeSAM** is a computer vision model designed for berry counting and grape segmentation tasks, leveraging advanced deep learning techniques.

## TODO
1. Point model dataset class
2. Clean dataset and publish on Huggingface and Zenodo
3. Clean code and ready to release

## 🚀 Installation  

To download the pre-trained model from Hugging Face, run:  

```bash
bash pretrain/download_huggingface.sh facebook/<model_name> <model_name>
```

Where `<model_name>` can be one of the following:  
- `sam-vit-base`  
- `sam-vit-large`  
- `sam-vit-huge`  

---

## 📖 Getting Started  

This repository provides scripts for **berry counting** and **grape segmentation**, with training and inference pipelines for each task.

---

## 🍇 Berry Counting  

### 🔥 Training  

To train the model using point-based annotations, run:

```bash
mkdir -p weights/point 
python3 model/point/train.py \
    --data-dir /home/xz/Dev/baseline-exp-playground/DATASET/vivid_processed \
    --save-dir ../../weights/point \
    --batch-size 32 \
    --max-epoch 2000 \
    --val-start 0 \
    --val-epoch 1 
```

### 🔍 Inference  

(To be added: Include details on inference process and commands.)  

---

## 🍇 Grape Segmentation  

🎭 Mask2Former  


### 🔥 Training 

To train the **Mask2Former** model using the **VIVID dataset**, modify the dataset registration script:  

- Uncomment line **49** in `model/mask2former/data/datasets/register_vivid_instance.py`:  
  
  ```python
  register_vivid_datasets()
  ```

### 🔍 Inference  

Use `model/test_predictor.py` to load the model and predict segmentation outputs on simulation images.

#### 🖼️ Demo: From Image(s) to Mask  

Run the following command to generate segmentation masks from input images:  

```bash
python model/mask2former_demo.py \
    --input {image_path(s), split by space} \
    --output {output_path or dir} \
    --config-file config/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml \
    --opts MODEL.WEIGHTS {model_path}
```

---

## 📝 License  

(Include licensing details here.)

---

## 📜 Citation  

If you use **GrapeSAM** in your research or application, please cite our work:  

(To be added: Include the appropriate citation format.)  

---

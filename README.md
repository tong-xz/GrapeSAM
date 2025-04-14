# GrapeSAM: A Unified Computer Vision Pipeline for Cluster Closure Computation

[[`arXiv`](https://arxiv.org/)] [[`Project`](https://bowenc0221.github.io/mask2former)] [[`BibTeX`](#CitingMask2Former)]

dddd

<div align="center">
  <img src="assets/teaser.png" width="90%" height="90%"/>
</div><br/>

## 📦 ViViD-5k Dataset

The ViViD-5k dataset can be downloaded from [hugging-face](link) and [zenodo](link).

## 🚀 Installation

### Environment Setup

```bash
conda create -n grapesam python=3.10
conda activate grapesam
pip install -r requirements.txt

```

If you want to train the **Mask2Former** model, please install the CUDA kernel for MSDeformAttn:

```bash
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```

The detailed guide from this [document](https://github.com/facebookresearch/Mask2Former/blob/main/INSTALL.md) 


### Checkpoint Download

To download the pre-trained **Segment Anything Model** weights from Hugging Face, run:

```bash
bash pretrain/download_huggingface.sh facebook/<model_name> <model_name>
```

Where `<model_name>` can be one of the following:

- `sam-vit-base`
- `sam-vit-large`
- `sam-vit-huge`

The mask2former and point localization model weights can be downloaded from [here](link).

---

## 📖 Getting Started

We provide the pipeline for computing the cluster closure, pure cluster segmentation, and berry counting separately.

### 1. Cluster Closure

Use `model/test_predictor.py` to load the model and predict segmentation outputs on simulation images.

### 2. Pure Cluster Mask

### 3. Pure Berry Localization

Run the following command to generate segmentation masks from input images:

```bash
python model/mask2former_demo.py \
    --input {image_path(s), split by space} \
    --output {output_path or dir} \
    --config-file config/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml \
    --opts MODEL.WEIGHTS {model_path}
```

---

## 🔥 Training

#### 1. Cluster Segmentation Model

To train the **Mask2Former** model using the **VIVID dataset**, modify the dataset registration script:

- Uncomment line **82** in `model/mask/mask2former/data/datasets/register_vivid_instance.py`:


  ```python
  register_vivid_datasets()
  ```

You may change the dataset path in the line **55**, if your dataset is not in the `./Vivid`. 

- Train the model:

```bash
cd model/mask
python train_net.py --num-gpus 1 --config-file ../../config/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml SOLVER.IMS_PER_BATCH 4
```


#### 2. Berry Counting Model

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

---

## 📜 Citation

```bibtex
@article{,
  title={},
  author={},
  journal={},
  year={2025}
}
```

## 🤝 Acknowledgements

We would like to thank the following projects:

- https://github.com/facebookresearch/segment-anything
- https://github.com/facebookresearch/Mask2Former
- https://github.com/jia-wan/GeneralizedLoss-Counting-Pytorch

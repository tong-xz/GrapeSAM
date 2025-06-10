# GrapeSAM: A Unified Computer Vision Pipeline for Cluster Closure Computation

[[`arXiv`](https://arxiv.org/)] [[`Project`](https://bowenc0221.github.io/mask2former)] [[`BibTeX`](#CitingMask2Former)]

> %CC Temporal Variation
<div align="center">
  <img src="assets/teaser2.png" width="100%" height="100%"/>
</div><br/>

## 📦 ViViD-5k Dataset

The dataset is available for download through:
- [Hugging Face](https://huggingface.co/datasets/ViViD-5k)
- [Google Drive](https://drive.google.com/drive/folders/1DedeM3kxSTjxnBtvx3nbHH3CIVplmhwy?usp=sharing)
- [Zenodo](https://zenodo.org/record/TBD)

## 🚀 Installation

Our work container two base parts: the **Mask2Former** model for cluster segmentation and the **Point Localization** model for berry counting. The following instructions will guide you through the installation process.

### Environment Setup
Install [docker](https://docs.docker.com/engine/install/) before running the following commands.

```bash
# Build the docker image
docker build -t grapesam-env .

# Run and mount the docker container
docker run --gpus all -it --rm \       
  -v $(pwd):/workspace \
  -w /workspace \
  grapesam-env bash

# After entering the container, install the CUDA kernel for MSDeformAttn:
sh /workspace/model/mask/mask2former/modeling/pixel_decoder/ops/make.sh
```

If you encounter any issues, please refer to the [Mask2Former installation guide](https://github.com/facebookresearch/Mask2Former/blob/main/INSTALL.md). 


### Checkpoint Download

To download the pre-trained **Segment Anything Model** weights from Hugging Face, run:

```bash
bash weights/download_hf.sh facebook/<model_name> <model_name>
```

Where `<model_name>` can be one of the following:

- `sam-vit-base`
- `sam-vit-large`
- `sam-vit-huge`

The mask2former and point localization model weights can be downloaded from [here](link).

---

## 📖 Getting Started

We provide the pipeline for computing the cluster closure, pure cluster segmentation, berry counting separately and the complete pipeline.

### Complete Pipeline

Run the following command to compute the cluster closure and berry counting:

```bash
python pipeline.py --point-ckpt <checkpoints/point_model.pth> --mask-ckpt <checkpoints/mask_model.pth> --input </data/Hypothesis/theorem/grape/Dream/test/ood> --output <test/ood_out> --sam-pth=</data/models/sam/huggingface/sam-vit-huge/> 
```

- `--point-ckpt`. Path to the pre-trained Point Localization model checkpoint.
- `--mask-ckpt`. Path to the pre-trained Mask2Former model checkpoint.
- `--input`. Path to the directory containing images.
- `--output`. Path to the output directory where results will be saved.
- `--sam-pth`. (Optitional) Path to the downloaded Segment Anything Model weights if you download it manually.

If you want to use the specific part of this pipeline, you could follow this instruction.

### Part 1. Cluster Mask

Run the following command to generate segmentation masks from input images:

```bash
python model/mask2former_demo.py \
    --input {image_path(s), split by space} \
    --output {output_path or dir} \
    --config-file config/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml \
    --opts MODEL.WEIGHTS {model_path}
```

### Part 2. Pure Berry Localization

```bash
python point_pipeline.py --img-dir <img_dir> --point-ckpt <checkpoints/point_model.pth> --save-dir <output_dir> --sam-pth=/data/models/sam/huggingface/sam-vit-huge/ --save-vis
```

- `--img-dir`. Path to the directory containing images.
- `--point-ckpt`. Path to the pre-trained Point Localization model checkpoint.
- `--save-dir`. Path to the output directory where results will be saved.
- `--sam-pth`. (Optional) Path to the downloaded Segment Anything Model weights if you download it manually.
- `--save-vis`. (Optional) Save the visualization results.

Then you could get the berry localization results containing the point coordinates in file `<img>_points.png` and the point location image `<img>.png`.


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

- `--num-gpus`. The number of GPUs to use for training.
- `--config-file`. The path to the configuration file for the model.
- `SOLVER.IMS_PER_BATCH`. The number of images per batch during training.



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

- `--data-dir`. Path to the directory containing the processed VIVID dataset.
- `--save-dir`. Path to the directory where the model checkpoints will be saved.
- `--batch-size`. The batch size for training.
- `--max-epoch`. The maximum number of epochs for training.
- `--val-start`. The epoch to start validation.
- `--val-epoch`. The frequency of validation epochs.


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

If you use the `ViViD-5k` dataset, please also cite the following works.
```bibtex
@article{blekos2023grape,
  title={A grape dataset for instance segmentation and maturity estimation},
  author={Blekos, Achilleas and Chatzis, Konstantinos and Kotaidou, Martha and Chatzis, Theocharis and Solachidis, Vassilios and Konstantinidis, Dimitrios and Dimitropoulos, Kosmas},
  journal={Agronomy},
  volume={13},
  number={8},
  pages={1995},
  year={2023},
  publisher={MDPI}
}
```


```bibtex
@article{pinheiro2023deep,
  title={Deep learning YOLO-based solution for grape bunch detection and assessment of biophysical lesions},
  author={Pinheiro, Isabel and Moreira, Germano and Queir{\'o}s da Silva, Daniel and Magalh{\~a}es, Sandro and Valente, Ant{\'o}nio and Moura Oliveira, Paulo and Cunha, M{\'a}rio and Santos, Filipe},
  journal={Agronomy},
  volume={13},
  number={4},
  pages={1120},
  year={2023},
  publisher={MDPI}
}
```

```bibtex
@article{sozzi2022wgrapeunipd,
  title={wGrapeUNIPD-DL: An open dataset for white grape bunch detection},
  author={Sozzi, Marco and Cantalamessa, Silvia and Cogato, Alessia and Kayad, Ahmed and Marinello, Francesco},
  journal={Data in Brief},
  volume={43},
  pages={108466},
  year={2022},
  publisher={Elsevier}
}
```

```bibtex
@dataset{morros2021ai4agriculture,
  author       = {Josep Ramon Morros and Tomas Pariente Lobo and Sergio Salmeron-Majadas and Javier Villazan and Diego Merino and Ana Antunes and Mihai Datcu and Chandrabali Karmakar and Edmundo Guerra and Despina-Athanasia Pantazi and George Stamoulis},
  title        = {{AI4Agriculture Grape Dataset} (1.0.0)},
  year         = {2021},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.5660081},
  url          = {https://doi.org/10.5281/zenodo.5660081},
  note         = {Data set}
}
```

## 🤝 Acknowledgements

We would like to thank the following projects:

- https://github.com/facebookresearch/segment-anything
- https://github.com/facebookresearch/Mask2Former
- https://github.com/jia-wan/GeneralizedLoss-Counting-Pytorch

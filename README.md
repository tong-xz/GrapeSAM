
# GrapeSAM
[Notion](https://www.notion.so/GrapeSAM-707c39118d9442ad9c33740ec8ce4456)
[Sheet](https://docs.google.com/spreadsheets/d/1DM6vgAi5Fy2fUahmxdyjnrkGGhRGVAz7F3GxPt5ehOg/edit?gid=0#gid=0)


## Checkpoint Download

```bash
bash pretrain/download_huggingface.sh facebook/<model_name> <model_name>
```
`model_name`: sam-vit-base, sam-vit-large, and sam-vit-huge




# Berry Counting 

# Grape Cluster Segmentation
## Mask2former

### Inference

`model/test_predictor.py` is the test of the predictor model. It will load the model and predict the output of the simulation image.

### Train

if you want to train the mark2former model by vivid dataset, please uncomment the line 49 ` register_vivid_datasets()` in `model/mask2former/data/datasets/register_vivid_instance.py`

### Demo 

From image(s) to mask

```bash
python model/mask2former_demo.py --input {image_path(s), split by space} --output {output_path or dir} --config-file config/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml --opts MODEL.WEIGHTS {model_path}"
```
## Train
```python
python train.py --batch_size 4 --epoch_num 500 --sam_ckpt ./weights/sam_vit_h_4b8939.pth --wandb
```

## Eval

```bash
python3 eval_prompter.py --root_dir ./data/exp/ --ckp_path ./weights/vivid6/point_decoder_11-13-07\:37\:21.pth --vis
```
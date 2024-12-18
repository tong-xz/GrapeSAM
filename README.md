
# GrapeSAM
[Notion](https://www.notion.so/GrapeSAM-707c39118d9442ad9c33740ec8ce4456)
[Sheet](https://docs.google.com/spreadsheets/d/1DM6vgAi5Fy2fUahmxdyjnrkGGhRGVAz7F3GxPt5ehOg/edit?gid=0#gid=0)


https://blog.csdn.net/qq_61676281/article/details/131845706

![alt text](assets/image.png)

## Checkpoint Download

How to download huggingface model weights?
```bash
cd pretrain
bash download_huggingface.sh facebook/sam-vit-huge sam-vit-huge
cd ..
```

## Train
```python
python train.py --batch_size 4 --epoch_num 500 --sam_ckpt ./weights/sam_vit_h_4b8939.pth --wandb
```

## Eval

```bash
python3 eval_prompter.py --root_dir ./data/exp/ --ckp_path ./weights/vivid6/point_decoder_11-13-07\:37\:21.pth --vis
```
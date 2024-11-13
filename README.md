
# GrapeSAM
[Notion](https://www.notion.so/GrapeSAM-707c39118d9442ad9c33740ec8ce4456)
[Sheet](https://docs.google.com/spreadsheets/d/1DM6vgAi5Fy2fUahmxdyjnrkGGhRGVAz7F3GxPt5ehOg/edit?gid=0#gid=0)


https://blog.csdn.net/qq_61676281/article/details/131845706

![alt text](assets/image.png)

# Config

How to download huggingface model weights?
```bash
cd pretrain
bash download_huggingface.sh facebook/sam-vit-huge sam-vit-huge
cd ..
```

## Train
```python
python3 train.py --batch_size 4 --epoch_num 5 --wandb
```

## Test

## Eval

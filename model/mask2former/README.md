env: counting

need: install the detectron2 lib, install the pixel decoder lib `ops` in the readme file.

# train

```py
python train_net.py --num-gpus 1 --config-file configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k.yaml SOLVER.IMS_PER_BATCH 6
```

6 image per batch in 1 gpu.

# inference

```py
python demo/demo.py --input /data/Hypothesis/proposition/Mask2Former/datasets/vivid/images/14.png.jpg --output ./out.png --config-file configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml --opts MODEL.WEIGHTS /data/Hypothesis/proposition/Mask2Former/output/model_0214999.pth
```

Add NMS. Need to modify the code to show the result, from [here](https://blog.csdn.net/qq_44324181/article/details/126242948).


# test

```py
python train_net.py --config-file configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml --eval-only MODEL.WEIGHTS output/model_0214999.pth SOLVER.IMS_PER_BATCH 16
```

This `SOLVER.IMS_PER_BATCH 16` may not work.

---

# dataset

Convert the Vivid dataset to train.

From the [doc](https://docs.cvat.ai/docs/manual/advanced/formats/format-coco/), CVAT could export the dataset in coco format.

There exist one issue when CVAT output the annotation in the coco format. The `iscrowd` key will be the `1`. From [here](https://github.com/facebookresearch/detectron2/issues/2415), I know this is not correct. The `iscrowd` key should be `0` for the instance segmentation task. Many people asked this issue in [Detectron2 issues](https://github.com/facebookresearch/detectron2/issues/5027).

I found the method in this [issue](https://github.com/cvat-ai/cvat/issues/7030).

```py
# install datum
pip install 'git+https://github.com/cvat-ai/datumaro@develop#egg=datumaro[default]'

# convert the json file to the `is_crowd: 0`
datum convert -o './output' -f 'coco_instances' -i 'your.json' -if 'coco_instances'  -- --segmentation-mode polygons

```

more info for the [datum](https://openvinotoolkit.github.io/datumaro/latest/docs/command-reference/context_free/convert.html)

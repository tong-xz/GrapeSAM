from hypo import Run, run


@run(cuda_visible_devices={0})
def pipeline():
    return [
        Run(
            name="run pipeline",
            command="python3 pipeline.py --point-ckpt checkpoints/point_model.pth --mask-ckpt checkpoints/mask_ckpt.pth --input test/feb-test/immature --output test/immature_out --sam-pth=/data/models/sam/huggingface/sam-vit-huge/",
        )
    ]

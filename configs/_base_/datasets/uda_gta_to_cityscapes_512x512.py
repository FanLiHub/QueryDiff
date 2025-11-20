_base_ = [
    "./uda_gta_512x512.py",
    "./uda_cityscapes_512x512.py",
]
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type="UDADataset",
        datasets=[
            {{_base_.train_gta}},
            {{_base_.train_cityscapes}},
        ],
        cfg=dict(
            # Rare Class Sampling
            rare_class_sampling=dict(
                min_pixels=3000, class_temp=0.01, min_crop_ratio=0.5)
        )
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="ConcatDataset",
        datasets=[
            {{_base_.val_cityscapes}},
        ],
    ),
)
test_dataloader = val_dataloader
val_evaluator = dict(
    type="DGIoUMetric", iou_metrics=["mIoU"], dataset_keys=["citys", "map", "bdd"]
)
test_evaluator=val_evaluator


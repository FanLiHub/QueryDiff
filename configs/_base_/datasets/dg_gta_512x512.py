_base_ = [
    "./gta_512x512.py",
    "./bdd100k_512x512.py",
    "./cityscapes_512x512.py",
    "./mapillary_512x512.py",

    # "./fog-acdc_512x512.py",
    # "./night-acdc_512x512.py",
    # "./rain-acdc_512x512.py",
    # "./snow-acdc_512x512.py",
]
train_dataloader = dict(
    batch_size=2,
    num_workers=6,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset={{_base_.train_gta}},
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
            {{_base_.val_bdd}},
            {{_base_.val_mapillary}},

            # {{_base_.val_night_acdc}},
            # {{_base_.val_fog_acdc}},
            # {{_base_.val_snow_acdc}},
            # {{_base_.val_rain_acdc}},
        ],
    ),
)
test_dataloader = val_dataloader
val_evaluator = dict(
    type="DGIoUMetric", iou_metrics=["mIoU"], dataset_keys=["citys", "map", "bdd"]
)
# val_evaluator = dict(
#     type="DGIoUMetric", iou_metrics=["mIoU"],
#     dataset_keys=["citys", "map", "bdd", "night/", "fog/", "snow/", "rain/"],
#     mean_used_keys=["night/", "fog/", "snow/", "rain/"],
# )
test_evaluator=val_evaluator

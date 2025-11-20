gta_type = "CityscapesDataset"
gta_type = "GTADataset"
gta_root = "/data/lf/data/gta/"
gta_crop_size = (512, 512)
gta_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="Resize", scale=(1280, 720)),
    dict(type="RandomCrop", crop_size=gta_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
gta_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1280, 720), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
train_gta = dict(
    type=gta_type,
    data_root=gta_root,
    data_prefix=dict(
        img_path="images",
        seg_map_path="labels",
    ),
    img_suffix=".png",
    seg_map_suffix="_labelTrainIds.png",
    pipeline=gta_train_pipeline,

    cfg=dict(
        # Rare Class Sampling
        rare_class_sampling=dict(
            min_pixels=3000, class_temp=0.01, min_crop_ratio=0.5)
    )
)
val_gta = dict(
    type=gta_type,
    data_root=gta_root,
    data_prefix=dict(
        img_path="images",
        seg_map_path="labels",
    ),
    img_suffix=".png",
    seg_map_suffix="_labelTrainIds.png",
    pipeline=gta_test_pipeline,
)
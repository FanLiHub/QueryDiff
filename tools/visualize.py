# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import sys
from mmengine.config import Config
from mmseg.utils import get_classes, get_palette
from mmengine.runner.checkpoint import _load_checkpoint

from mmseg.apis import inference_model
import tqdm
import mmengine
import torch
import numpy as np
from PIL import Image

# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence, Union
from mmengine import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmseg.registry import MODELS
from mmseg.utils import SampleList, dataset_aliases, get_classes, get_palette

def init_model(config: Union[str, Path, Config],
               checkpoint: Optional[str] = None,
               device: str = 'cuda:0',
               cfg_options: Optional[dict] = None):
    """Initialize a segmentor from config file.

    Args:
        config (str, :obj:`Path`, or :obj:`mmengine.Config`): Config file path,
            :obj:`Path`, or the config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str, optional) CPU/CUDA device option. Default 'cuda:0'.
            Use 'cpu' for loading model on CPU.
        cfg_options (dict, optional): Options to override some settings in
            the used config.
    Returns:
        nn.Module: The constructed segmentor.
    """
    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    elif 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None
    config.model.pretrained = None
    config.model.train_cfg = None
    init_default_scope(config.get('default_scope', 'mmseg'))

    model = MODELS.build(config.model)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        if 'meta' not in checkpoint:
            checkpoint['meta']={}
        dataset_meta = checkpoint['meta'].get('dataset_meta', None)
        # save the dataset_meta in the model for convenience
        if 'dataset_meta' in checkpoint.get('meta', {}):
            # mmseg 1.x
            model.dataset_meta = dataset_meta
        elif 'CLASSES' in checkpoint.get('meta', {}):
            # < mmseg 1.x
            classes = checkpoint['meta']['CLASSES']
            palette = checkpoint['meta']['PALETTE']
            model.dataset_meta = {'classes': classes, 'palette': palette}
        else:
            warnings.simplefilter('once')
            warnings.warn(
                'dataset_meta or class names are not saved in the '
                'checkpoint\'s meta data, classes and palette will be'
                'set according to num_classes ')
            num_classes = model.decode_head.num_classes
            dataset_name = None
            for name in dataset_aliases.keys():
                if len(get_classes(name)) == num_classes:
                    dataset_name = name
                    break
            if dataset_name is None:
                warnings.warn(
                    'No suitable dataset found, use Cityscapes by default')
                dataset_name = 'cityscapes'
            model.dataset_meta = {
                'classes': get_classes(dataset_name),
                'palette': get_palette(dataset_name)
            }
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="MMSeg test (and eval) a model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training configuration file."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint file for both the REIN and head models."
    )

    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Directory or file path of images to be processed."
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save the output images."
    )
    parser.add_argument("--suffix", default=".jpg", help="File suffix to filter images in the directory. Default is '.png'.")
    parser.add_argument("--not-recursive", action='store_false', help="Whether to search images recursively in subfolders. Default is recursive.")
    parser.add_argument("--search-key", default="", help="Keyword to filter images within the directory. Default is no filtering.")
    parser.add_argument(
        "--backbone",
        default="checkpoints/dinov2_converted.pth",
        help="Path to the backbone model checkpoint. Default is 'checkpoints/dinov2_vitl14_converted_1024x1024.pth'."
    )
    parser.add_argument("--tta", action="store_true", help="Enable test time augmentation. Default is disabled.")
    parser.add_argument("--device", default="cuda:0", help="Device to use for computation. Default is 'cuda:0'.")
    args = parser.parse_args()
    return args

def load_backbone(checkpoint: dict, backbone_path: str) -> None:
    converted_backbone_weight = _load_checkpoint(backbone_path, map_location="cpu")
    if "state_dict" in checkpoint:
        checkpoint["state_dict"].update(
            {f"backbone.{k}": v for k, v in converted_backbone_weight.items()}
        )
    else:
        checkpoint.update(
            {f"backbone.{k}": v for k, v in converted_backbone_weight.items()}
        )


classes = get_classes("cityscapes")
palette = get_palette("cityscapes")


def draw_sem_seg(sem_seg: torch.Tensor):
    num_classes = len(classes)
    sem_seg = sem_seg.data.squeeze(0)
    H, W = sem_seg.shape
    ids = torch.unique(sem_seg).cpu().numpy()
    legal_indices = ids < num_classes
    ids = ids[legal_indices]
    labels = np.array(ids, dtype=np.int64)
    colors = [palette[label] for label in labels]
    colors = [torch.tensor(color, dtype=torch.uint8).view(1, 1, 3) for color in colors]
    result = torch.zeros([H, W, 3], dtype=torch.uint8)
    for label, color in zip(labels, colors):
        result[sem_seg == label, :] = color
    return result.cpu().numpy()


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if "test_pipeline" not in cfg:
        cfg.test_pipeline = [
            dict(type="LoadImageFromFile"),
            dict(
                keep_ratio=True,
                scale=(
                    1920,
                    1080,
                ),
                type="Resize",
            ),
            dict(type="PackSegInputs"),
        ]
    model = init_model(cfg, args.checkpoint, device=args.device)
    model=model.cuda(args.device)
    state_dict = model.state_dict()
    load_backbone(state_dict, args.backbone)
    model.load_state_dict(state_dict)
    mmengine.mkdir_or_exist(args.save_dir)
    images = []
    if osp.isfile(args.images):
        images.append(args.images)
    elif osp.isdir(args.images):
        for im in mmengine.scandir(args.images, suffix=args.suffix, recursive=args.not_recursive):
            if args.search_key in im:
                images.append(osp.join(args.images, im))
    else:
        raise NotImplementedError()
    print(f"Collect {len(images)} images")
    for im_path in tqdm.tqdm(images):
        result = inference_model(model, im_path)
        pred = draw_sem_seg(result.pred_sem_seg)
        img = Image.open(im_path).convert("RGB")
        pred = Image.fromarray(pred).resize(
            [img.width, img.height], resample=Image.NEAREST
        )
        vis = Image.new("RGB", [img.width * 2, img.height])
        vis.paste(img, (0, 0))
        vis.paste(pred, (img.width, 0))
        vis.save(osp.join(args.save_dir, 'wholeshow56'+osp.basename(im_path)))

    print(f"Results are saved in {args.save_dir}")


if __name__ == "__main__":
    main()

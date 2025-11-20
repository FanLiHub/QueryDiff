# Better to Teach than to Give:Domain Generalized Semantic Segmentation via Agent Queries with Diffusion Model Guidance

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
[![Static Badge](https://img.shields.io/badge/View-Poster-purple)](https://icml.cc/virtual/2025/poster/44281)
[![Static Badge](https://img.shields.io/badge/Pub-ICML'25-red)](https://fanlihub.github.io/QueryDiff/static/pdf/Domain_Generalized_Semantic_Segmentation_via_Agent_Queries_with_Diffusion_Model_Guidance.pdf)
[![Static Badge](https://img.shields.io/badge/View-Project-green)](https://fanlihub.github.io/QueryDiff/)

This repository is the official PyTorch implementation of the **ICML 2025** (**Spotlight**) paper:
Better to Teach than to Give:Domain Generalized Semantic Segmentation via Agent Queries with Diffusion Model Guidance,
authored by Fan Li, Xuan Wang, Min Qi, Zhaoxiang Zhang, and Yuelei Xu.

**Abstract:**
Domain Generalized Semantic Segmentation (DGSS) trains a model on a labeled source domain to generalize to unseen target domains with consistent contextual distribution and varying visual appearance. Most existing methods rely on domain randomization or data generation but struggle to capture the underlying scene distribution, resulting in the loss of useful semantic information. Inspired by the diffusion model's capability to generate diverse variations within a given scene context, we consider harnessing its rich prior knowledge of scene distribution to tackle the challenging DGSS task. In this paper, we propose a novel agent \textbf{Query}-driven learning framework based on \textbf{Diff}usion model guidance for DGSS, named QueryDiff. Our recipe comprises three key ingredients: (1) generating agent queries from segmentation features to aggregate semantic information about instances within the scene; (2) learning the inherent semantic distribution of the scene through agent queries guided by diffusion features; (3) refining segmentation features using optimized agent queries for robust mask predictions. Extensive experiments across various settings demonstrate that our method significantly outperforms previous state-of-the-art methods. Notably, it enhances the model's ability to generalize effectively to extreme domains, such as cubist art styles.

![Framework](static/images/framework.png)

## Visual Results

![Framework](static/images/qualitatives.png)

## Environment

- Python (3.8.19)
- PyTorch (2.0.1) 
- TorchVision (0.15.2)
- mmcv (2.2.0)
- mmsegmentation (1.2.2)
- mmengine (0.10.5)
- diffusers (0.30.2)
- transformers (4.27.4)

## Installation

```
conda create -n querydiff python=3.8
conda activate querydiff
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install "mmsegmentation>=1.0.0"
pip install "mmdet>=3.0.0"
pip install xformers=='0.0.20'
pip install -r requirements.txt
```

## Dataset Preparation

Cityscapes

- Obtain the files `leftImg8bit_trainvaltest.zip` and `gt_trainvaltest.zip` from the [Cityscapes download page](https://www.cityscapes-dataset.com/downloads/).
- Unpack both archives and place their contents inside: data/cityscapes

Mapillary

- Download the Mapillary Vistas Dataset v1.2 from the [Mapillary Research portal](https://research.mapillary.com/).
- Extract the dataset into the directory: data/mapillary.

GTA

- Fetch the complete set of images and annotations from the [TU Darmstadt “Playing for Data” website](https://download.visinf.tu-darmstadt.de/data/from_games/).
- Unzip everything and move it to: data/gta.

ACDC

- Download the full ACDC dataset (images + annotations) from https://acdc.vision.ee.ethz.ch/.
- Extract everything into: data/acdc.

Dataset Conversion and Preprocessing: When setting up the datasets for the first time, run the following commands to convert formats and prepare validation splits:

```
mkdir data
# Prepare GTA and Cityscapes for evaluation
python tools/convert_datasets/gta.py data/gta           # GTA as source domain
python tools/convert_datasets/cityscapes.py data/cityscapes

# Transform Mapillary into Cityscapes-style labels and generate resized validation data
python tools/convert_datasets/mapillary2cityscape.py \
    data/mapillary \
    data/mapillary/cityscapes_trainIdLabel \
    --train_id

python tools/convert_datasets/mapillary_resize.py \
    data/mapillary/validation/images \
    data/mapillary/cityscapes_trainIdLabel/val/label \
    data/mapillary/half/val_img \
    data/mapillary/half/val_label
```

## Pretrained Model Weights

- Obtaining the weights: 

  - You can download the official DINOv2 ViT-L/14 pretrained checkpoint from the [facebookresearch release page](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth).
  - After downloading, place the file directly inside the project directory without renaming it.

- Converting the weights:

  - Before training or evaluation, convert the checkpoint into the format expected by this codebase:

    ```
    python tools/convert_models/convert_dinov2.py \
        checkpoints/dinov2_vitl14_pretrain.pth \
        checkpoints/dinov2_converted.pth
    ```

  - For high-resolution models (e.g., 1024×1024):

    ```
    python tools/convert_models/convert_dinov2.py \
        checkpoints/dinov2_vitl14_pretrain.pth \
        checkpoints/dinov2_converted_1024x1024.pth \
        --height 1024 --width 1024
    ```

## Evaluation

    python tools/test.py \
        --config <path/to/config_file.py> \
        --checkpoint <path/to/checkpoint.pth> \
        --show-dir <path/to/output_directory>

For example:

```
python tools/test.py \
    --config configs/dinov2/dinov2_mask2former_512x512_bs1x4.py \
    --checkpoint /home/lifan/testmodel/iter_40000.pth \
    --show-dir output/vis_result/
```

## Training

```
python tools/train.py --config <path/to/config_file.py>
```

For example:

```
python tools/train.py --config configs/dinov2/dinov2_mask2former_512x512_bs1x4.py
```

## Demo

    python tools/visualize.py \
        --config <path/to/config/file.py> \
        --checkpoint <path/to/checkpoint.pth> \
        --images <path/to/input/images_or_folder> \
        --save_dir <path/to/output_directory>

For example:

```
python tools/visualize.py \
    --config configs/dinov2/dinov2_mask2former_512x512_bs1x4.py \
    --checkpoint model/iter_40000.pth \
    --images data/mapillary/validation/images/ \
    --save_dir output/vis_result/
```

## Acknowledgements

This repo is built upon these previous works:

- [Rein](https://github.com/w1oves/Rein)
- [Marigold](https://github.com/prs-eth/marigold)
- [Diffusion Hyperfeatures](https://github.com/diffusion-hyperfeatures/diffusion_hyperfeatures)
- [DAFormer](https://github.com/lhoyer/DAFormer)

## Citation

If you find it helpful, you can cite our paper in your work.

    @inproceedings{li2025better,
      title={Better to Teach than to Give: Domain Generalized Semantic Segmentation via Agent Queries with Diffusion Model Guidance},
      author={Li, Fan and Wang, Xuan and Qi, Min and Zhang, Zhaoxiang and Xu, Yuelei},
      booktitle={Proceedings of the 42nd International Conference on Machine Learning},
      pages={36129--36139},
      year={2025},
      volume={267},
      series={Proceedings of Machine Learning Research},
      month={13--19 Jul},
      publisher={PMLR},
      }








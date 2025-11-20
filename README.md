# QueryDiff
[ICML 2025] Official implement of &lt;Better to Teach than to Give: Domain Generalized Semantic Segmentation via Agent Queries with Diffusion Model Guidance>
- [x] Evaluation
- [ ] Training

## Setup Environments

To set up your environment, execute the following commands:

```bash
conda create -n querydiff -y
conda activate querydiff
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install "mmsegmentation>=1.0.0"
pip install "mmdet>=3.0.0"
pip install xformers=='0.0.20' # optional for DINOv2
pip install -r requirements.txt
```

## Dataset Preparation

The preparation process follows a similar procedure to that of [DDB](https://github.com/xiaoachen98/DDB).

**GTA:** Please, download all image and label packages from [here](https://download.visinf.tu-darmstadt.de/data/from_games/) and extract them to `data/gta`.

**Cityscapes:** Please, download leftImg8bit_trainvaltest.zip and gt_trainvaltest.zip from [here](https://www.cityscapes-dataset.com/downloads/) and extract them to `data/cityscapes`.

**Mapillary:** Please, download MAPILLARY v1.2 from [here](https://research.mapillary.com/) and extract them to `data/mapillary`.

**ACDC:** Download all image and label packages from [ACDC](https://acdc.vision.ee.ethz.ch/) and extract them to `data/acdc`.

## Pretraining Weights

The stable diffusion model weights can be obtained from https://huggingface.co/stabilityai/stable-diffusion-2-1.

DINOv2 weights can be obtained from https://github.com/facebookresearch/dinov2.

## Evaluation

Run the evaluation:

```
python tools/test.py /path/to/cfg /path/to/checkpoint --backbone /path/to/backbone
```

## Acknowledgment

This repo is built upon these previous works:

* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [Marigold](https://github.com/prs-eth/marigold)
* [Diffusion Hyperfeatures](https://github.com/diffusion-hyperfeatures/diffusion_hyperfeatures)
* [DDB](https://github.com/xiaoachen98/DDB)
* [Rein](https://github.com/w1oves/Rein)

## Citation

If you find it helpful, you can cite our paper in your work.

```bibtex
@inproceedings{li2025better,
  title={Better to Teach than to Give: Domain Generalized Semantic Segmentation via Agent Queries with Diffusion Model Guidance},
  author={Li, Fan and Wang, Xuan and Qi, Min and Zhang, Zhaoxiang and Xu, Yuelei},
  booktitle = 	 {Proceedings of the 42nd International Conference on Machine Learning},
  pages = 	 {36129--36139},
  year = 	 {2025},
  volume = 	 {267},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {13--19 Jul},
  publisher =    {PMLR},
  }
```


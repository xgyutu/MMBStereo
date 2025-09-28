<p align="center">
  <h1 align="center">Mamba-Based Multi-Branch Cost Aggregation for Stereo Matching</h1>
  <p align="center">

### The paper was published in Applied Soft Computing journal.

**Paper**: [Mamba-based multi-branch cost aggregation for stereo matching](https://doi.org/10.1016/j.asoc.2025.113973)

**Journal**: Applied Soft Computing, 2025

# How to use

## Environment
* Python 3.8
* PyTorch 1.10+

## Install

### Create a virtual environment and activate it.

```bash
conda create -n mmbstereo python=3.8
conda activate mmbstereo
```

### Dependencies

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia
pip install opencv-python
pip install scikit-image
pip install tensorboard
pip install matplotlib 
pip install tqdm
pip install timm
pip install basicsr
```

### Install causal-conv1d

```bash
cd causal-conv1d
python setup.py install
```

### Install mamba

```bash
cd mamba
python setup.py install
```

## Data Preparation

Download the following datasets:
- [Scene Flow Datasets](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
- [KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo)
- [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

## Train

### Train on Scene Flow Dataset

First, train the network for 26 epochs on Scene Flow:

```bash
python main_sceneflow.py --logdir ./checkpoints/sceneflow/mmbstereo
```

### Train on KITTI Dataset

Use the following command to train model on KITTI (using pretrained model on Scene Flow):

```bash
python main_kitti.py --loadckpt ./checkpoints/sceneflow/mmbstereo/checkpoint_000025.ckpt --logdir ./checkpoints/kitti/mmbstereo
```

## Evaluation

### Generate disparity images for KITTI test set

```bash
python save_disp.py --loadckpt ./checkpoints/kitti/mmbstereo/checkpoint_000499.ckpt
```

### Submit to KITTI benchmarks

```bash
python save_disp.py --loadckpt ./checkpoints/kitti/mmbstereo/checkpoint_000499.ckpt
```

## Model Architecture

The MMBStereo framework consists of:

- **MMBA (Mamba-based Multi-branch Cost Aggregation)**: The core aggregation network that uses Mamba blocks for efficient cost volume processing
- **Multi-branch Architecture**: Separate branches for spatial context and edge information
- **MobileNetV2 Backbone**: Lightweight feature extraction network

## Citation

If you use this code in your research, please cite:

```bibtex
@article{LU2025113973,
title = {Mamba-based multi-branch cost aggregation for stereo matching},
journal = {Applied Soft Computing},
pages = {113973},
year = {2025},
issn = {1568-4946},
doi = {https://doi.org/10.1016/j.asoc.2025.113973},
url = {https://www.sciencedirect.com/science/article/pii/S1568494625012864},
author = {Xingyuan Lu and Yanbing Xue and Leida Li and Shiyin Li and Zan Gao},
keywords = {Stereo matching, Cost aggregation, Mamba},
abstract = {This study presents Mamba-Based Multi-Branch Cost Aggregation for Stereo Matching (MMBStereo), an innovative real-time stereo matching framework with high performance. The core innovation lies in the Mamba-based multi-branch cost aggregation network, which uses a unique three-branch aggregation strategy. The Mamba Aggregation Branch integrates the State Space Model from the Mamba structure, replacing conventional convolution and Transformer methods, significantly enhancing network performance and efficiency. The Spatial Aggregation Branch addresses the loss of spatial texture information, improving the scene's contextual representation. Meanwhile, the Edge Aggregation Branch enhances edge responses, improving object boundary detection accuracy. Through a carefully designed multi-branch fusion strategy, the framework improves disparity prediction accuracy while maintaining real-time inference. Our method achieves competitive accuracy with non-real-time stereo matching frameworks, surpassing existing lightweight solutions in the widely recognized KITTI benchmark tests.}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

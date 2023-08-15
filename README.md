# A Dual-Prototypes Network For Few Shot Action Recognition Library

This repo contains reimplementations of a dual-prototypes network for few-shot action recognition in Pytorch based on TRX

I intend to keep it up to date so there's a common resource for people interested in this topic, and it should be a good codebase to start from if you want to implement your own method. 

Feature/method/pull requests are welcome, along with any suggestions, help or questions.

### Key Features
- Scripts to extract/format datasets using public splits
- Easy to use data loader for creating episodic tasks (just uses folders of images so is much easier to get working than the TRX zip version)
- Train/val/test framework for running everything

### Reimplementations

- Episodic TSN baseline using norm squared or cosine distance (as proposed in OTAM)
- [TRX](https://arxiv.org/abs/2101.06184) (CVPR 2021)
  

### Todo list

- Reimplement a dual-prototypes network
- Tensorboard logging in addition to the current logfile
- Any other suggestions you think would be useful


### Datasets supported

- Something-Something V2 ([splits from OTAM](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cao_Few-Shot_Video_Classification_via_Temporal_Alignment_CVPR_2020_paper.pdf))
- UCF101 ([splits from ARN](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500511.pdf))
- HMDB51 ([splits from ARN](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500511.pdf))
- Kinetics ([splits from CMN](https://openaccess.thecvf.com/content_ECCV_2018/papers/Linchao_Zhu_Compound_Memory_Networks_ECCV_2018_paper.pdf)

# Instructions

## Installation

Conda is recommended. 

### Requirements

- python >= 3.6
- pytorch >= 1.8
- einops
- ffmpeg (for extracting data)

### Hardware

To use a ResNet 50 backbone you'll need at least a 2 x RTX3090 machine. You can fit everything all on one GPU using a ResNet 18 backbone.


## Data preparation

Download the datasets from their original locations:

- [Something-Something V2](https://20bn.com/datasets/something-something#download)
- [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)
- [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads)
- [Kinetics](https://github.com/ffmpbgrnn/CMN)

Once you've downloaded the datasets, you can use the extract scripts to extract frames and put them in train/val/test folders. You'll need to modify the paths at the top of the scripts.
To remove unnecessary frames and save space (e.g. just leave 8 uniformly sampled frames), you can use shrink_dataset.py. Again, modify the paths at the top of the sctipt.

## Running

Use run.py. Example arguments for some training runs are in the scripts folder. You might need to modify the distribute functions in model.py to suit your system depending on your GPU configuration.

## Implementing your own method

Inherit the class CNN_FSHead in model.py, and add the option to use it in run.py. That's it! You can see how the other methods do this in model.py.



# Acnkowledgements

We based code on [TRX](https://arxiv.org/abs/2101.06184) . We use [torch_videovision](https://github.com/hassony2/torch_videovision) for video transforms.






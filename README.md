# AdaAct: An Adaptive Method Stabilizing Activations for Enhanced Generalization

This repository contains the source code for the implementation of **AdaAct**, an optimization algorithm developed to stabilize activation outputs, improve convergence, and enhance generalization in neural networks. The algorithm was introduced in the paper:

> **"An Adaptive Method Stabilizing Activations for Enhanced Generalization"**  
> *Authors: Hyunseok Seung, Jaewoo Lee, Hyunsuk Ko*  
> *Presented at the IEEE ICDM Workshop 2024*

## Overview

**AdaAct** introduces a novel approach to adaptively adjust learning rates based on activation variance. Unlike traditional adaptive optimizers (e.g., Adam) that focus on gradient variance, AdaAct dynamically scales updates according to activation stability. This method combines the fast convergence of adaptive gradient methods with the generalization strengths of SGD.

Key features:
- Enhanced generalization by stabilizing neuron outputs
- Efficient computation without requiring large covariance matrices
- Improved performance on standard benchmarks (CIFAR, ImageNet) with popular architectures (ResNet, DenseNet, ViT)

## Getting Started

### Installation

```bash
conda env create -n adaact -f environments.yml
conda activate adaact
```

### Running the Code

To run experiments with AdaAct on CIFAR-10:
```bash
python main.py --model resnet20 --optim adaact --lr 0.1 --beta1 0.9 --beta2 0.999 --eps 1e-8 --weight_decay 0.002 --epoch 200 --run 0;
```
For ImageNet:
```bash
torchrun --nproc_per_node=2 ./train.py --model resnet50 --sched cosine --epochs 100 --opt adaact \
--lr 4.0 --opt-betas 0.9 0.999 --opt-eps 1e-8 --weight-decay 1e-4 --workers 16 \
--warmup-epochs 0 --warmup-lr 0.4 --min-lr 0.0 --batch-size 256 --grad-accum-steps 4 --amp \
--aug-repeats 0 --aa rand-m7-mstd0.5-inc1 --smoothing 0.0 --remode pixel --crop-pct 0.95 \
--reprob 0.0 --drop 0.0 --drop-path 0.05 --mixup 0.1 --cutmix 1.0;

torchrun --nproc_per_node=2 ./train.py --model deit_small_patch16_224 --sched cosine --epochs 150 --opt adaact \
--lr 4.0 --opt-betas 0.9 0.999 --opt-eps 1e-8 --weight-decay 2e-7 --workers 16 \
--warmup-epochs 5 --warmup-lr 0.4 --min-lr 0.0004 --batch-size 256 --grad-accum-steps 4 --amp \
--aug-repeats 0 --aa rand-m7-mstd0.5-inc1 --smoothing 0.1 --remode pixel --reprob 0.25 \
--drop 0.0 --drop-path 0.1 --mixup 0.8 --cutmix 1.0;
```

## Citation
```bash
@inproceedings{seung2024adaact,
  title={An Adaptive Method Stabilizing Activations for Enhanced Generalization},
  author={Seung, Hyunseok and Lee, Jaewoo and Ko, Hyunsuk},
  booktitle={IEEE International Conference on Data Mining Workshop},
  year={2024}
}
```

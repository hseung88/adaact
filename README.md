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
git clone https://github.com/hseung88/adaact.git
cd adaact
pip install -r requirements.txt
```

### Running the Code

To run experiments with AdaAct on CIFAR-10:
```bash
python main.py --dataset cifar10 --model resnet20 --optimizer adaact
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

# Fashion-MNIST Image Classification: MLP vs CNN

Training and comparing a multilayer perceptron and a convolutional neural network for clothing image classification using PyTorch on the Fashion-MNIST dataset, with TensorBoard training monitoring.

## Overview

This project implements two neural network architectures from scratch in PyTorch for 10-class clothing classification. A fully connected MLP is trained first as a baseline, followed by a CNN that leverages spatial structure in the images to achieve higher accuracy within the same 100,000-parameter budget. Training is monitored via TensorBoard.

## Dataset

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist): 70,000 grayscale 28×28 images across 10 clothing categories (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle Boot). Split: 80% train / 20% validation from the official training set; 10,000-image official test set held out for final evaluation.

## Model 1: Multilayer Perceptron

```
Flatten (784) → Linear(784→100) → ReLU → Linear(100→10) → logits
```

**Parameters:**
- Hidden layer: (784 + 1) × 100 = 78,500
- Output layer: (100 + 1) × 10 = 1,010
- **Total: 79,510**

**Training:** SGD, `lr=0.01`, `batch_size=64`, `epochs=9`, Cross-Entropy Loss

**Result: ~83% validation accuracy**

## Model 2: Convolutional Neural Network

```
Conv2d(1→5, 5×5, same) → ReLU → MaxPool2d(2×2) → Flatten(980) → Linear(980→100) → ReLU → Linear(100→10) → logits
```

**Architecture decisions:**
- 5×5 kernel captures broad spatial features; `padding='same'` preserves spatial dimensions
- MaxPool2d(2×2, stride=2) halves spatial resolution (28×28 → 14×14), reducing parameters in downstream layers
- 5 output channels after pooling: 14 × 14 × 5 = 980 features fed into the linear classifier

**Parameters:**
- Conv layer: ((5×5×1) + 1) × 5 = 130
- Hidden layer: (980 + 1) × 100 = 98,100
- Output layer: (100 + 1) × 10 = 1,010
- **Total: 99,240**

**Training:** SGD, `lr=0.01`, `batch_size=64`, `epochs=25`, Cross-Entropy Loss, monitored with TensorBoard

**Result: >88% validation accuracy**

## Key Findings

The CNN outperforms the MLP with a nearly identical parameter count by exploiting the spatial structure of images. Convolutional filters learn local feature detectors (edges, textures) that are applied across the entire image, whereas the MLP must learn each pixel's contribution independently — a far less parameter-efficient representation.

## Tech Stack

- Python 3
- PyTorch (`torch.nn`, `torch.optim`, `torchvision`)
- TensorBoard (`torch.utils.tensorboard`)
- Matplotlib

## How to Run

```bash
pip install torch torchvision matplotlib jupyter tensorboard
jupyter notebook recognition.ipynb
```

The Fashion-MNIST dataset downloads automatically via `torchvision.datasets`. GPU acceleration is supported via CUDA or MPS if available.

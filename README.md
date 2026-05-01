# Face Image Generation with DCGAN on CelebA

## Overview

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) to generate synthetic 64×64 face images. The Generator learns to produce realistic faces from random noise vectors, while the Discriminator learns to distinguish real from generated images. Both networks are trained in an adversarial framework on the CelebA dataset.

## Dataset

**CelebA** (CelebFaces Attributes Dataset) from Kaggle: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset

- 202,599 aligned celebrity face images at 178×218 resolution
- A subset of 50,000 images is used for training
- Images are resized to 64×64 and normalized to [-1, 1]

Download and extract so the structure is:

```
dataset/
└── img_align_celeba/
    ├── 000001.jpg
    ├── 000002.jpg
    └── ...
```

## Setup

```bash
uv venv
source .venv/bin/activate
uv sync
```

## Running the Notebook

```bash
jupyter notebook generative_model.ipynb
```

Run all cells top-to-bottom. Training 100 epochs takes approximately 27 minutes on an NVIDIA 4070 Ti.

## Results

The DCGAN generates plausible face structures with correct feature positioning and moderate diversity. Three training stabilization techniques are used: spectral normalization on the Discriminator, label smoothing (0.9), and a 2:1 Generator-to-Discriminator training ratio. Key metrics:

- Generator: 3,576,704 parameters
- Discriminator: 2,765,568 parameters
- Training: 100 epochs, batch size 128, Adam optimizer (lr=0.0002)

## Bias Awareness

The CelebA dataset consists of celebrity photographs that are not demographically balanced — it disproportionately represents young, attractive, light-skinned individuals. The generated faces reflect these biases. A model trained on this data should not be deployed for any application that could impact real people without addressing these representation gaps and ensuring diverse training data.

## Dependencies

See `requirements.txt` for the full list. Key libraries:

- Python 3.11+
- PyTorch + torchvision
- NumPy, pandas, matplotlib, seaborn
- Jupyter

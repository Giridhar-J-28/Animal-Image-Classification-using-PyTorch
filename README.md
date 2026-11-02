# Animal Image Classification using PyTorch

A convolutional neural network (CNN) built with PyTorch that classifies animal face images into three classes: **cat**, **dog**, and **wild**.

## Contents
- `main.ipynb`: the full workflow (loading data, building the model, training, evaluating, and predicting on new images)
- `cat.jpg`, `dog.jpeg`, `wild.jpeg`: sample images for trying out predictions

## Dataset
This project uses the [Animal Faces (AFHQ)](https://www.kaggle.com/datasets/andrewmvd/animal-faces) dataset from Kaggle. It isn't included in this repository because of its size (~730 MB).

Download it and place it next to the notebook with this layout:
```
animal-faces/
├── train/
│   ├── cat/
│   ├── dog/
│   └── wild/
└── val/
    ├── cat/
    ├── dog/
    └── wild/
```
Then update the dataset path in `main.ipynb` to match your machine.

## Model
- 3 convolutional blocks (Conv2d → MaxPool2d → ReLU) with 32, 64 and 128 filters
- Flatten → Linear(32768 → 128) → ReLU → Linear(128 → 3)
- Input images are resized to 128×128
- Loss: CrossEntropyLoss, optimizer: Adam
- Device: CUDA, Apple Silicon (MPS), or CPU

## Results
- Validation accuracy: ~95%
- Test accuracy: ~94.6%

## Requirements
```
torch torchvision torchinfo scikit-learn pandas numpy matplotlib pillow
```

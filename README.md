# Soccer Ball Detection with Faster R-CNN

A computer vision project for detecting soccer balls in video frames using deep learning techniques.

## 📋 Project Overview

This project implements an object detection pipeline to automatically identify and localize soccer balls in video footage from soccer matches. The solution uses **Faster R-CNN** (ResNet-50 backbone with FPN) to achieve high accuracy in detecting balls across different game scenarios.

### Key Features
- **Dataset Preparation**: XML annotations converted to COCO format
- **Data Augmentation**: Custom transformations to handle class imbalance
- **Model**: Faster R-CNN with ResNet-50 + Feature Pyramid Network
- **Training & Validation**: Complete training pipeline with loss tracking
- **Post-processing**: Video generation with annotated bounding boxes

## 🎯 Motivation for Faster R-CNN

The project employs Faster R-CNN for several compelling reasons:

1. **Region Proposal Network (RPN)**: Faster R-CNN integrates an RPN that generates region proposals directly from the network, eliminating costly external methods like Selective Search used in earlier R-CNN variants. This dramatically improves speed while maintaining high accuracy.

2. **Accurate Proposal Generation**: The RPN produces highly precise and accurate region proposals, significantly reducing false detections and irrelevant proposals compared to traditional methods.

3. **Multi-class Detection Support**: The architecture naturally supports detecting multiple object categories simultaneously.

## 📊 Dataset

### Processing Pipeline
1. **XML Parsing**: Extract ball annotations from XML track files
2. **Aggregation**: Handle multiple ball instances per frame
3. **COCO Conversion**: Transform to COCO dataset format with bounding boxes
4. **Class Balance**: Address dataset imbalance (frames with/without balls)

### Dataset Statistics
- Total frames from 2 soccer videos: ~11,000 frames
- Custom Dataset class with PyTorch compatibility

### Data Augmentation
To address class imbalance, the following augmentations are applied:

- **Random Horizontal Flip**: Probability-based horizontal flipping with bbox adjustment
- **Random Vertical Flip**: Probability-based vertical flipping with bbox adjustment
- **Random Both Flips**: Combined horizontal and vertical flipping
- **Normalization**: ImageNet-standard mean and std normalization

## 🏗️ Architecture

### Model Configuration
- **Backbone**: ResNet-50 pre-trained on COCO
- **Feature Extractor**: Feature Pyramid Network (FPN)
- **Region Proposal Network**: Generates region proposals
- **RoI Heads**: Classification and bounding box regression
- **Output Classes**: 2 (ball + background)

### Custom Components
- **CustomDataset**: PyTorch Dataset wrapper for COCO format with image cropping
- **Transform Pipeline**: Custom transformations preserving bbox coordinates during augmentation
- **Collate Function**: Handles variable-sized batches

## 🚀 Training & Evaluation

### Training Loop
- **Optimizer**: SGD with momentum (lr=0.001, momentum=0.9)
- **Epochs**: 10
- **Batch Size**: 4
- **Train/Val Split**: 80/20
- **Loss Tracking**: Per-batch and per-epoch loss monitoring

### Key Metrics
- Training loss with data augmentation
- Validation loss on held-out dataset
- Model selection based on validation performance (Epoch 5 chosen as optimal)

## 📁 Project Structure

```
├── train-notebook.ipynb          # Main training pipeline
├── test-notebook.ipynb           # Inference and video generation
├── ID_5_256942_252050.json       # Training data
├── ID_6_256942_252050.json       # Test data
└── README.md                     
```

## 🔗 Resources

### Kaggle Links
All project materials are hosted on Kaggle:

- **Training Notebook**: https://www.kaggle.com/code/stekaggle/train-notebook
- **Testing Notebook**: https://www.kaggle.com/code/stekaggle/test-notebook
  - *Contains post-processing and video generation*
- **Dataset**: https://www.kaggle.com/datasets/stekaggle/computer-vision-futbal
- **Trained Model**: https://www.kaggle.com/models/stekaggle/cv_project_faster

## 💡 Key Insights

- The model achieves optimal performance at Epoch 5, avoiding overfitting observed in later epochs
- Data augmentation is crucial due to class imbalance in the dataset
- Custom transformation functions ensure bbox coordinates are properly adjusted during augmentation
- The COCO dataset format provides a standard interface for detector training and evaluation

## 🛠️ Technologies Used

- **Deep Learning**: PyTorch, TorchVision
- **Data Processing**: Python, JSON, XML parsing
- **Visualization**: Matplotlib
- **COCO Tools**: pycocotools
- **Hardware**: GPU acceleration support

## 📝 Course Context

This is a Computer Vision course project demonstrating end-to-end deep learning pipeline implementation from data preparation to model deployment.


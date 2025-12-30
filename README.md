# Knowledge Discovery and Data Mining (KDDM) - Final Project

This repository contains the final project for CS 513-A, demonstrating various machine learning and data mining techniques applied to real-world datasets.

## Project Overview

This project implements multiple clustering and classification algorithms across four distinct problems:
1. **Hierarchical Clustering** - Income analysis of KY/NJ zip codes
2. **K-means Clustering** - Geographic income segmentation
3. **Multi-Layer Perceptron (MLP)** - Heart attack severity prediction
4. **Random Forest Classification** - Heart attack classification
5. **Neural Network Backpropagation** - Manual gradient descent implementation

## Datasets

### 1. KY-NJ Zip Code Dataset
- **File**: `KY_NJ_Zip_2025F_CS_513-A.csv`
- **Features**: Income tax return percentages (6 brackets) + total returns
- **Size**: 1,328 ZIP codes across Kentucky and New Jersey

### 2. Heart Attack Dataset
- **File**: `Heart_Attack_2025F_CS_513-A.csv`
- **Features**: RestHR, MaxHR, RecHR (Recovery Heart Rate), BP (Blood Pressure)
- **Target**: Heart attack severity (Light, Mild, Massive)

## Implemented Algorithms

### Problem 1: Hierarchical Clustering
- **Method**: Single linkage clustering
- **Clusters**: 3
- **Preprocessing**: StandardScaler normalization
- **Results**: 
  - Cluster 1: 1,241 ZIP codes (93.5%)
  - Cluster 2: 1 ZIP code (outlier - KY 41833)
  - Cluster 3: 1 ZIP code (outlier - KY 40171)

### Problem 2: K-means Clustering
- **Method**: K-means with k=5
- **Parameters**: 
  - `n_estimators=500`
  - `random_state=42`
  - `n_init=10`
- **Results**: More balanced cluster distribution across income levels

### Problem 3: Multi-Layer Perceptron Classifier
- **Architecture**: 3 hidden layers (256, 128, 64 neurons)
- **Activation**: ReLU
- **Optimizer**: Adam
- **Performance**:
  - **Test Accuracy**: 98.89%
  - Near-perfect confusion matrix with only 1 misclassification

### Problem 4: Random Forest Classifier
- **Parameters**:
  - `n_estimators=500`
  - `max_depth=None`
  - `n_jobs=-1` (parallel processing)
- **Performance**:
  - **Test Accuracy**: 94.44%
  - **Most Important Feature**: RecHR (Recovery Heart Rate) - 46.8%

### Problem 5: Neural Network Backpropagation
- Manual implementation of forward and backward propagation
- Demonstrates weight updates using gradient descent
- Learning rate: 0.1

## Key Findings

### Heart Attack Classification
The **Recovery Heart Rate (RecHR)** emerged as the most critical predictor of heart attack severity (46.8% importance), indicating that slower cardiac recovery post-stress strongly correlates with severe cardiac events.

**Feature Importance Ranking**:
1. RecHR: 46.8%
2. BP: 39.4%
3. RestHR: 10.3%
4. MaxHR: 3.4%

### Model Comparison
| Model | Accuracy | Best For |
|-------|----------|----------|
| MLP | 98.89% | Overall best performance |
| Random Forest | 94.44% | Feature interpretability |

## Installation & Requirements


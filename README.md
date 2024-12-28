# Handwritten-Digit-Recognition-using-LeNet-5-CNN-Architecture-on-MNIST-Digit-Database
Overview

This project implements a digit recognition system using a modified version of the LeNet-5 Convolutional Neural Network (CNN) architecture. The model is trained and tested on the MNIST dataset, which consists of handwritten digits (0-9). The system demonstrates the capability of CNNs to classify grayscale images into their respective digit categories.

Features

Data preprocessing including normalization and reshaping.

A CNN model with multiple convolutional, pooling, and dense layers.

Model training with validation to monitor performance.

Visualization of training accuracy and validation accuracy.

Evaluation of test accuracy.

Display of predictions alongside actual labels.

Dataset

The MNIST dataset is used, which contains:

60,000 training images: Grayscale images of handwritten digits.

10,000 test images: Grayscale images for evaluation.

Model Architecture

The modified LeNet-5 model includes:

Convolutional Layer 1: 8 filters, kernel size (3x3), ReLU activation.

MaxPooling Layer 1: Pool size (2x2).

Convolutional Layer 2: 32 filters, kernel size (3x3), ReLU activation.

MaxPooling Layer 2: Pool size (2x2).

Flatten Layer: Converts 2D data into 1D.

Fully Connected Layer 1: 128 neurons, ReLU activation.

Fully Connected Layer 2: 64 neurons, ReLU activation.

Output Layer: 10 neurons (for digits 0-9), softmax activation.

Dependencies

Python 3.x

TensorFlow

NumPy

Matplotlib

Training and Evaluation

The model is trained for 7 epochs with a batch size of 64. The validation split is set to 10% of the training data. The test accuracy achieved is displayed after evaluation.

Results

The model achieves a test accuracy of approximately 95-99%, depending on the specific training run.

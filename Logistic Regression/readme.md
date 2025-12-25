Bank Churn Prediction using PyTorch
This project implements a Logistic Regression model using PyTorch to predict customer churn based on banking data. It includes a complete end-to-end pipeline covering data preprocessing, model training, intermediate evaluation, and performance visualization.

Table of Contents
Overview

Dataset

Prerequisites

Project Structure

Model Architecture

Training & Evaluation

Results

Overview
The goal of this project is to classify whether a bank customer will churn (leave the bank) or not based on various demographic and account-related features. The solution utilizes PyTorch for model construction and Scikit-Learn for robust data preprocessing.

Key features:

Data Preprocessing: Handling categorical variables (One-Hot Encoding) and scaling numerical features.

PyTorch Model: Custom nn.Module implementation of Logistic Regression.

Live Monitoring: Evaluates validation loss and accuracy every 10 epochs.

Visualization: Generates plots for Loss and Accuracy trends.

Dataset
The model is trained on bank_churn.csv.

Target Variable:

churn: 1 if the client has left the bank, 0 otherwise.

Input Features:

credit_score (Numerical)

country (Categorical: Encoded via One-Hot Encoding)

gender (Binary)

age (Numerical)

tenure (Numerical)

products_number (Numerical)

> Note: The ID column is dropped during preprocessing as it provides no predictive value.

Prerequisites
The code is designed to run in a Python 3 environment (like Google Colab). You will need the following libraries:

Bash

pip install torch pandas numpy scikit-learn matplotlib
🚀 Project Structure
The solution is contained within a single script/notebook that performs the following steps:

Data Loading: Reads bank_churn.csv.

Preprocessing: - Splits data into Features (X) and Target (y).

Applies StandardScaler to numerical columns.

Applies OneHotEncoder to categorical columns (e.g., country).

Splits data into Training (80%) and Validation (20%) sets.

Model Definition: Defines a single-layer Linear network.

Training Loop: Optimizes the model using Adam optimizer and BCEWithLogitsLoss.

Visualization: Exports a chart showing training progress.

Model Architecture
Type: Logistic Regression (implemented as a Neural Network).

Input Layer: Dimension depends on preprocessed feature count.

Output Layer: 1 unit (Logits).

Activation: Sigmoid (applied implicitly via the loss function during training, explicitly during inference).

Training & Evaluation
The model is trained using the following hyperparameters:

Epochs: 500

Batch Size: Full batch (Gradient Descent)

Learning Rate: 0.01

Optimizer: Adam

Loss Function: Binary Cross Entropy with Logits (BCEWithLogitsLoss)

Intermediate evaluation is performed every 10 epochs on the validation set to monitor for overfitting.

Results
Upon completion of training (500 epochs), the model typically achieves:

Validation Accuracy: ~71.6%

Validation Loss: ~0.60

Performance Visualization
The script generates a plot training_metrics.png displaying:

Left Graph: Training vs. Validation Loss.

Right Graph: Validation Accuracy over time.

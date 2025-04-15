# [Emotion Recognition in Speech Using Deep Neural Networks](https://ieeexplore.ieee.org/abstract/document/9721504)
[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.png?v=103)](https://github.com/ellerbrock/open-source-badges/)

This project aims to perform **Emotion Recognition in Speech** using **Deep Neural Networks (DNNs)**. The implemented model leverages a **Convolutional Neural Network (CNN)** architecture to classify speech emotions based on acoustic features extracted using **OpenSMILE**. The emotions identified by the model include **anger**, **surprise**, **happiness**, **sadness**, and **neutral**.

## Project Overview

The goal of this project is to explore the effectiveness of **Convolutional Neural Networks (CNNs)** in emotion classification tasks based on speech signals. The project utilizes acoustic features like **eGeMAPS** and **ComParE**, which are commonly used for emotion recognition in speech processing. These features provide a detailed representation of the audio, capturing different aspects like pitch, energy, and formants, which are useful for emotion recognition.

For more information, please the [paper](https://ieeexplore.ieee.org/abstract/document/9721504).

## Objective

Emotion recognition from speech has applications in areas such as **human-computer interaction**, **customer service**, **mental health** analysis, and **assistive technologies** for people with disabilities. The primary goal of this project is to investigate how well a deep learning model can classify emotions based on audio features.

## Model Architecture

The implemented model uses a **Convolutional Neural Network (CNN)** for classifying emotions from speech. The CNN consists of several convolutional layers, pooling layers, and normalization layers to capture the features of the input speech. The final classification layer outputs predictions for 5 classes corresponding to the emotions: **anger**, **surprise**, **happiness**, **sadness**, and **neutral**.

## Features

1. **Deep Learning Model**: A convolutional neural network (CNN) is used to learn from acoustic features and classify speech emotions.
2. **Cross-Validation**: The model is trained using **K-Fold Cross-Validation** to provide more reliable performance metrics and reduce the risk of overfitting.
3. **Custom Loss Function**: The model uses **sparse categorical cross-entropy** as the loss function and custom evaluation metrics, including **Accuracy** and **Unweighted Average Recall (UAR)**, to handle imbalanced data.
4. **Data Normalization**: Before training, the data is standardized using **StandardScaler** to improve model performance.
5. **Early Stopping**: The training process is monitored for early stopping to prevent overfitting and save time.
6. **Model Checkpoints**: The best model weights are saved during training to avoid losing the optimal model due to early stopping.

## Evaluation Metrics

- **Accuracy**: Measures the overall percentage of correct classifications.
- **Unweighted Average Recall (UAR)**: This metric is crucial for evaluating models on imbalanced datasets. It computes the average recall across all classes without considering the class distribution. UAR is particularly useful in emotion recognition tasks, where emotions may not be evenly distributed in the dataset.

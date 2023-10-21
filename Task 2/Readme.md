# Real-Time Face Mask Detection

Real-Time Face Mask Detection is a machine learning project developed by [Ahmed Hisham Fathy Hassabou](https://github.com/AhmedHishamFathy) as part of the internship at [SYNC Interns](https://github.com/SYNC-Interns). This project aims to detect whether a person is wearing a face mask in real-time using computer vision and machine learning techniques.

## Table of Contents
- [General Information](#general-information)
- [Getting Started](#getting-started)
- [Building the Model](#building-the-model)
- [Testing Individual Images](#testing-individual-images)
- [Live Face Mask Detection](#live-face-mask-detection)
- [Requirements](#requirements)
- [Usage](#usage)

## General Information

- **Organization:** SYNC Interns
- **Track:** Machine Learning
- **Role:** Internship
- **Author:** Ahmed Hisham Fathy Hassabou
- **Task No:** 2
- **Task Name:** Real Time Face Mask Detection

## Getting Started

This project includes code for real-time face mask detection using computer vision and machine learning. It can be used to detect whether a person is wearing a mask or not.

## Building the Model

The machine learning model for face mask detection is built using a convolutional neural network (CNN). It is trained to classify whether a person is wearing a face mask or not in real-time.

## Testing Individual Images

The code allows you to load and test individual images for face mask detection. It loads a trained model and uses it to predict whether a person in a given image is wearing a mask or not.

## Live Face Mask Detection

The core part of the project is real-time face mask detection. It uses the pre-trained model to detect faces in real-time using a webcam. If a person is detected without a mask, a red rectangle is drawn around their face, and the label "NO MASK" is displayed. If a mask is detected, a green rectangle is drawn with the label "MASK."

## Requirements

To run this code, you need to have the following libraries and tools installed:

- Python
- Keras
- OpenCV
- NumPy

## Usage

1. Clone this repository.
```bash
git clone https://github.com/Ahmedhh1218/Sync-Machine-Learning-Playground.git
```
2. Install the required libraries.
```bash
pip install opencv-python numpy tensorflow
```
3. Open Task 2 Folder
```bash
cd Task 2
```
4. If you want to experience the training of the model run the code as it is.
```bash
python3 Sync_Machine_Learning_Task_2.py
```
5. To perform real-time face mask detection using a webcam without training the model(as it is pretrained)
- comment step 2 in the code
- run the code as in point 4
# Welcome to the ML Engineer Task Solution at Nafith Logistics International! 🚀

<br>

## Table of Contents 📚

1. Introduction
2. Getting Started
3. How to Run the Scripts
4. Evaluation
5. Google Drive Link for all the files
6. FAQs

<br>


## Introduction 📝
This repository contains the solution for a machine learning task provided by Nafith Logistics International. The task involves developing a Convolutional Neural Network (CNN) classifier from scratch or using a pre-trained model, training it on a provided dataset, and then applying it to classify objects in a video.

<br>


## Getting Started 🏁
To get started with this project, follow these steps:

1. Clone the repository using `git clone https://github.com/baselhusam/nafith`.
2. Install the required packages using `pip install -r requirements.txt`.
Now you’re ready to run the scripts!

<br>

## How to Run the Scripts 🖥️

Here’s how you can run each script:

- `read_and_split.py`: Run using `python3 read_and_split.py --img_size 224`. If you don’t parse an argument for `img_size`, it will be 224 by default.

- `train.py`: Run using `python3 train.py --epochs 100 --learning_rate 0.001 --is_transfer_learning`. The number of epochs is 100 by default, and the learning rate is 0.001 by default. If `is_transfer_learning` is given as an argument, it will train the VGG16 pre-trained model; otherwise, it will train the custom model.

- `test.py`: Run using `python3 test.py --is_transfer_learning`. If `is_transfer_learning` is given as an argument, it will evaluate the VGG16 pre-trained model; otherwise, it will evaluate the custom model.
  
- `classifier.py`: Run using `python3 classifier.py --video_path demo.mkv`. The `video_path` argument is required for the path of the video for running inference on it.

<br>

### All Together 🔥



```bash

# Clone the Repo
git clone https://github.com/baselhusam/nafith.git

# Install the Requirements
pip install -r requirements.txt

# Run the read_and_split.py script
python3 read_and_split.py --img_size 224

# Run the `train.py` script for the custom model
python3 train.py

# Run the `train.py` script for the VGG16 model
python3 train.py --is_transfer_learning

# Run the `test.py` script for the custom model
python3 test.py

# Run the `test.py` scripts for the VGG16 model
python3 test.py --is_transfer_learning

# Run the `classifier.py` script
python3 classifier.py

```

<br>

## Evaluation 📊
The evaluation metrics for both models are as follows

| Model / Metric | Custom Model | VGG16 |
| --------------- | ----------- | ------ |
| Train Set | --- |  -- |
| Accuracy | 0.77 | 0.9 | 
| Precision | 0.77 | 0.89 |
| Recall | 0.79 | 0.89 |
| F1-Score | 0.76 | 0.89 |
| Validation Set | --- | --- |
| Accuracy | 0.69 | 0.88 | 
| Precision | 0.70 | 0.88 |
| Recall | 0.71 | 0.88 |
| F1-Score | 0.69 | 0.88 |
| Test Set | --- | --- |
| Accuracy | 0.71 | 0.85 | 
| Precision | 0.72 | 0.85 |
| Recall | 0.73 | 0.84 |
| F1-Score | 0.71 | 0.84 |

<br>

## Google Drive Link for all the files 💾
Due to GitHub’s file size limit of 100MB, not all files are included in this repository. The missing files can be found in this Google Drive link. The missing files include:

- `assets` Folder: This folder contains numpy arrays and saved history and weights for both models.
- `dataset` Folder: This folder is not included in Google Drive because it’s assumed that you already have it on your machine.

<br>

## FAQs ❓
If you have any questions about this project, please check out our FAQ section or feel free to open an issue.




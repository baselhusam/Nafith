# Solution for the ML Engineer Task at Nafith Logisitics International

## Introduction

The task provided by Nafith Logistics International involves developing a machine learning solution for classifying and counting objects in a video. The solution involves creating a Convolutional Neural Network (CNN) classifier from scratch or using a pre-trained model, training it on a provided dataset, and then applying it to classify objects in a video. The solution is implemented in Python and is designed to run on Linux.

## Approach

The approach to solvig this tasks involved several steps:

1. **Data Preparation** (`read_and_split.py`): This script reads the images, stores them as numpy arrays, splits them into training, testing, and validation sets, and finally saves these numpy arrays.

2. **Model Training** (`train.py`): This script trains a custom CNN model from scratch or uses transfer learning (VGG16) to classify different types of vehicles (car, bus, and truck).

3. **Model Evaluation** (`test.py`): This script evaluates the trained model (either the custom model or the pre-trained VGG16 model). It displays the confusion matrix for the training, validation, and testing sets and saves the images. It also calculates the Accuracy, Precision, Recall, and F1-Score for these sets.

4. **Classification** (`classifier.py`): This script takes the trained model and applies it to count objects from a video. It displays the count for each class, the frames per second (FPS), the majority prediction for every object, and the prediction for the opened frame.

All these scripts are built using classes defined in `utils.py`, which contains all their functionalities.

<br> 


## Files and Folders

The solution includes several files and folders:

- `read_and_split.py`, `train.py`, `test.py`, `classifier.py`, `utils.py`: These are Python scripts that perform various tasks as described above.

- `main.ipynb`: This Jupyter notebook compares the custom model with the VGG16 pre-trained model using flow_from_directory and X and Y Numpy Arrays approaches. It also compares hard disk size (in MB) and time taken to build the models (in seconds).

- `requirements.txt`: This file lists all required Python packages needed to run the solution.

- `Predictions` Folder: Contains saved images from predictions for each object within its supposed folder.

- `assets` Folder: Contains three main sub-folders:
  1. `Data`: Contains numpy arrays.
  2. `Model`: Contains saved history and weights for the custom model.
  3. `Model_Trans`: Contains saved history and weights for the VGG16 pre-trained model.
 
<br>

## Dataset

The dataset provided consists of 2531 images belonging to three different classes: car (1087 images), bus (866 images), and truck (578 images).

<br>

## Python Version
The solution is implemented in Python 3.11.4.

<br>

## Future Enhancements

In future iterations of this project, we could consider implementing additional features such as real-time object detection and tracking, improving classification accuracy through advanced techniques like data augmentation or fine-tuning, or expanding the solution to handle more classes of objects.







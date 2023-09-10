import os
import cv2
import json

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from collections import Counter
from sewar.full_ref import sam
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from tensorflow.keras.applications import VGG16



class ImageProcessor:

    """
    This class is used to process the images.
    It has all the functions needed to process the images.

    The functions are:
    - get_imgs_path
    - preprocess_y
    - imgs_as_array
    - split_data
    - save_npy
    """

    def __init__(self, dir_path='dataset'):
        """
        Initialize the ImageProcessor class

        Parameters
        ----------
        dir_path : str
            Path of the directory where the folders of the classes are

        Returns
        -------
        None
        """
        self.dir_path = dir_path

    def get_imgs_path(self, class_name):
        """
        Get the path of the images of a class

        Parameters
        ----------
        class_name : str
            Name of the class 
        dir_path : str
            Path of the directory where the folders of the classes are

        Returns
        -------
        imgs_path : list
            List of the path of the images of the class
        
        """

        return os.listdir(f'{self.dir_path}/' + class_name)

    @staticmethod
    def preprocess_y(y):
        """
        Preprocess the labels from categorical to numbers (one hot encoded)

        Parameters
        ----------
        y : list
            List of the labels

        Returns
        -------
        y : list
            List of the labels
        
        """

        le = LabelEncoder()
        y = le.fit_transform(y)
        return to_categorical(y)

    def imgs_as_array(self, df, number_of_imgs, img_size):

        """
        Save the images of a dataframe as a numpy array

        This is useful when you have a lot of images and you want to train a model with them.
        It is faster to load a numpy array than to load each image.

        This technique is better, faster and more efficient than using flow_from_directory or flow_from_dataframe.

        Parameters
        ----------
        df : pandas dataframe
            Dataframe with the images path and their labels
        number_of_imgs : int
            Number of images
        img_size : int
            Size of the images
        save_path : str
            Path to save the numpy array

        Returns
        -------
        X : numpy array
            Numpy array with the images
        y : numpy array
            Numpy array with the labels
        
        """

        X = np.empty((number_of_imgs, img_size, img_size, 3), dtype=np.uint8)
        for i, img_path in enumerate(tqdm(df["image"])):
            img = cv2.imread('dataset/' + df["label"][i] + '/' + img_path)
            img = cv2.resize(img, (img_size, img_size))
            X[i] = img
        y = self.preprocess_y(df["label"].values)
        return X, y

    @staticmethod
    def split_data(X, y, val_size, test_size):

        """
        Split the data into train, validation and test sets

        Parameters
        ----------
        X : numpy array
            Numpy array with the images
        y : numpy array
            Numpy array with the labels
        train_size : float
            Percentage of the train set
        val_size : float
            Percentage of the validation set
        test_size : float
            Percentage of the test set
        

        Returns
        -------
        X_train : numpy array
            Numpy array with the images of the train set
        X_val : numpy array
            Numpy array with the images of the validation set
        X_test : numpy array
            Numpy array with the images of the test set
        y_train : numpy array
            Numpy array with the labels of the train set
        y_val : numpy array
            Numpy array with the labels of the validation set
        y_test : numpy array
            Numpy array with the labels of the test set
        
        """

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size,stratify=y_train,
                                                          random_state=42)
        return X_train, X_val, X_test, y_train, y_val, y_test

    @staticmethod
    def save_npy(X_train, X_val, X_test, y_train, y_val, y_test):

        """
        Save the numpy arrays

        Parameters
        ----------
        X_train : numpy array
            Numpy array with the images of the train set
        X_val : numpy array
            Numpy array with the images of the validation set
        X_test : numpy array
            Numpy array with the images of the test set
        y_train : numpy array
            Numpy array with the labels of the train set
        y_val : numpy array
            Numpy array with the labels of the validation set
        y_test : numpy array
            Numpy array with the labels of the test set

        Returns
        -------
        None

        Note: This function saves the numpy arrays in the assets folder.
        
        """

        np.save('assets/Data/X_train.npy', X_train)
        np.save('assets/Data/X_val.npy', X_val)
        np.save('assets/Data/X_test.npy', X_test)
        np.save('assets/Data/y_train.npy', y_train)
        np.save('assets/Data/y_val.npy', y_val)
        np.save('assets/Data/y_test.npy', y_test)



class ModelBuilder:

    """
    This class is used to build the model. 
    It has all the functions needed to build the model, 
    train it and save it.

    The functions are:
    - build_model
    - build_model_vgg
    - create_generators
    - train_model
    - save_model
    - save_history
    """


    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):

        """
        Initialize the ModelBuilder class
        
        Parameters
        ----------
        X_train : numpy array
            Numpy array with the images of the train set
        X_val : numpy array
            Numpy array with the images of the validation set
        y_train : numpy array
            Numpy array with the labels of the train set
        y_val : numpy array
            Numpy array with the labels of the validation set
        X_test : numpy array
            Numpy array with the images of the test set
        y_test : numpy array
            Numpy array with the labels of the test set

        Returns
        -------
        None
        """

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test


    def build_model(self):

        """
        Build the model using CNN Architecture. The model is simple, 
        it is multiple layers of Conv2D and MaxPooling2D, followed by Dense layers.

        Parameters
        ----------
        None

        Returns
        -------
        model : keras model
            The model built with keras

        """

        model = Sequential([

            Conv2D(16, (3, 3), activation='relu',  input_shape=(self.X_train.shape[1], self.X_train.shape[2], self.X_train.shape[3])),
            MaxPooling2D((2, 2)),

            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),

            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),

            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),

            Conv2D(256, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),

            Flatten(),

            Dense(512, activation='relu'),
            Dropout(0.3),

            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(3, activation='softmax')
        ])

        return model
    

    def build_model_vgg(self):

        """
        Build the model using Transfer Learning. The model is VGG16 with some extra layers.

        Parameters
        ----------
        None

        Returns
        -------
        model : keras model
            The model built with keras

        Note: This function uses the VGG16 model from keras, with the weights of imagenet.
        """

        base_model = VGG16(weights='imagenet', 
                        include_top=False, 
                        input_shape=(self.X_train.shape[1], self.X_train.shape[2], self.X_train.shape[3]))

        # Make sure the base model layers are not trainable
        for layer in base_model.layers:
            layer.trainable = False

        # Create a new model on top
        model = Sequential([

            base_model,

            Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=base_model.output_shape[1:]),
            MaxPooling2D((2, 2)),

            Conv2D(256, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),

            Flatten(),

            Dense(512, activation='relu'),
            Dropout(0.3),

            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(3, activation='softmax')
        ])

        return model

    def create_generators(self):

        """
        Create the generators for the train, validation and test sets

        Parameters
        ----------
        None

        Returns
        -------
        train_gen : keras generator
            Generator for the train set
        val_gen : keras generator
            Generator for the validation set
        test_gen : keras generator
            Generator for the test set
        """

        # Create the generators with the transformations
        train_generator = ImageDataGenerator(rescale=1./255,
                                             rotation_range=30,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             horizontal_flip=True,
                                             fill_mode='nearest',
                                             zoom_range=0.15,)
        
        val_generator = ImageDataGenerator(rescale=1./255)
        test_generator = ImageDataGenerator(rescale=1./255)

        # Create the generators after applying the transformations
        train_gen = train_generator.flow(self.X_train, self.y_train, batch_size=32)
        val_gen = val_generator.flow(self.X_val, self.y_val, batch_size=32, shuffle=False)
        test_gen = test_generator.flow(self.X_test, self.y_test, batch_size=32, shuffle=False)

        return train_gen, val_gen, test_gen

    def train_model(self, model, genrators, epochs=100, learning_rate=0.0001):

        """
        Train the model using the generators and the parameters passed

        Parameters
        ----------
        model : keras model
            The model built with keras
        genrators : list
            List with the generators for the train, validation and test sets
        epochs : int
            Number of epochs
        learning_rate : float
            Learning rate

        Returns
        -------
        history : keras history
            History of the model training
        """

        model.compile(loss='categorical_crossentropy', 
                      optimizer=Adam(learning_rate = learning_rate), 
                      metrics=['accuracy'])
        
        early_stopping = EarlyStopping(monitor='val_loss', 
                                       patience=20)
        
        history = model.fit(genrators[0],
                            steps_per_epoch=len(self.X_train) // 32,
                            epochs=epochs,
                            validation_data=genrators[1],
                            validation_steps=len(self.X_val) // 32,
                            callbacks=[early_stopping])
        return history

    def save_model(self,model, changed_path):

        """
        Save the model in .h5 format 
        
        Parameters
        ----------
        model : keras model
            The model built with keras
        changed_path : string
            String to add to the name of the model (if transfer or not)

        Returns
        -------
        None

        Note: This function saves the model in the assets folder.
        """

        path = "Model" + changed_path
        model.save(f"assets/{path}/model.h5")

    def save_history(self,history, changed_path):

        """
        Save the history of the model in a json file

        Parameters
        ----------
        history : keras history
            History of the model training

        Returns
        -------
        None

        Note: This function saves the history of the model in the assets folder.
        """

        path = "Model" + changed_path
        with open(f"assets/{path}/history.json", "w") as f:
            json.dump(history.history, f)


class ModelTester:

    """
    This class is used to evaluate the model.
    It has all the functions needed to evaluate the model.
    The metrics used are: Accuracy, Precision, Recall and F1-score.
    Also, it plot and save the confusion matrix.

    All the metrics and the confusion matrix are saved in a json file.

    The functions are:
    - evaluate_set
    - save_metrics
    
    """

    def __init__(self, model):
        self.model = model
        

    def evaluate_set(self, X, y, set_name, changed_path):
        
        """
        Evaluate the model using the metrics: Accuracy, Precision, Recall and F1-score

        Parameters
        ----------
        X : numpy array
            Numpy array with the images of the set
        y : numpy array
            Numpy array with the labels of the set
        set_name : string
            Name of the set

        Returns
        -------
        accuracy : float
            Accuracy of the model
        precision : float
            Precision of the model
        recall : float
            Recall of the model
        f1 : float
            F1-score of the model

        Note: This function plots and saves the confusion matrix in the assets folder.
        """

        # Load weights
        if changed_path:
            self.model.load_weights("assets/Model_Trans/model.h5")
        else:
            self.model.load_weights("assets/Model/model.h5")


        # Make predictions
        y_pred = self.model.predict(X)
        y_pred = np.round(y_pred)

        # Calculate the metrics
        accuracy =  accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='macro')
        recall = recall_score(y, y_pred, average='macro')
        f1 = f1_score(y, y_pred, average='macro')

        # Print the metrics
        print(f'Accuracy {set_name}: {accuracy}')
        print(f'Precision {set_name}: {precision}')
        print(f'Recall {set_name}: {recall}')
        print(f'F1-score {set_name}: {f1}')

        # Plot and save the confusion matrix
        cm = confusion_matrix(y.argmax(axis=1), y_pred.argmax(axis=1))
        
        plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)

        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        plt.title(f'Confusion Matrix {set_name}')
        plt.tight_layout()
        
        path = "Model" + changed_path
        plt.savefig(f'assets/{path}/confusion_matrix_{set_name}.png')
        plt.show()
        
        return accuracy, precision, recall, f1

    def save_metrics(self, train_metrics, val_metrics, test_metrics, changed_path):

        """
        
        Save the metrics in a json file
        
        Parameters
        ----------
        train_metrics : list
            List with the metrics of the train set
        val_metrics : list
            List with the metrics of the validation set
        test_metrics : list
            List with the metrics of the test set
        changed_path : string
            String to add to the name of the model (if transfer or not)

        Returns
        -------
        None

        Note: This function saves the metrics in the assets folder.
    
        """

        metrics = {
            'Train': {
                'Accuracy': train_metrics[0],
                'Precision': train_metrics[1],
                'Recall': train_metrics[2],
                'F1': train_metrics[3]
            },
            'Validation': {
                'Accuracy': val_metrics[0],
                'Precision': val_metrics[1],
                'Recall': val_metrics[2],
                'F1': val_metrics[3]
            },
            'Test': {
                'Accuracy': test_metrics[0],
                'Precision': test_metrics[1],
                'Recall': test_metrics[2],
                'F1': test_metrics[3]
            }
        }

        path = "Model" + changed_path
        with open(f'assets/{path}/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)


def load_data():

    """
    Load the numpy arrays

    Parameters
    ----------
    None

    Returns
    -------
    X_train : numpy array
        Numpy array with the images of the train set
    y_train : numpy array
        Numpy array with the labels of the train set
    X_val : numpy array
        Numpy array with the images of the validation set
    y_val : numpy array
        Numpy array with the labels of the validation set
    X_test : numpy array
        Numpy array with the images of the test set
    y_test : numpy array
        Numpy array with the labels of the test set
    """

    X_train = np.load("assets/Data/X_train.npy")
    y_train = np.load("assets/Data/y_train.npy")
    X_val = np.load("assets/Data/X_val.npy")
    y_val = np.load("assets/Data/y_val.npy")
    X_test = np.load("assets/Data/X_test.npy")
    y_test = np.load("assets/Data/y_test.npy")

    return X_train, y_train, X_val, y_val, X_test, y_test




class ObjectCounter:

    """
    This class is used to count the objects in a video.
    It has all the functions needed to count the objects.
    The model used is the one trained in the ModelBuilder class.

    The functions are:
    - count_objects
    """

    def __init__(self, video_path, class_dict):

        """
        Initialize the ObjectCounter class
        
        Parameters
        ----------
        video_path : str
            Path of the video
        class_dict : dict
            Dictionary with the classes and their labels

        Returns
        -------
        None
        """

        # Load the class dictionary
        self.video_path = video_path
        self.class_dict = class_dict
        self.counter_dict = {value: 0 for key, value in class_dict.items()}
        
        # Load the model with the weights
        model_builder = ModelBuilder(*load_data())
        self.model = model_builder.build_model_vgg()
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
        self.model.load_weights('assets/Model_Trans/model.h5')
        
        # Load the video
        self.video = cv2.VideoCapture(video_path)

        # Initialize the variables
        self.frame_number = 0
        self.prev_frame = None
        self.prev_predictions = []

    def count_objects(self):

        """
        Count the objects in the video and display the predictions
        The Counting is done using the SAM (Spectral Angle Mapper) algorithm.
        The SAM algorithm is used to compare two images and calculate the difference between them.
        If the difference is more than 0.45, it means that the images are different.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Note: This function saves the predictions in the Predictions folder.

        Note2: This function shows the predictions in a window. 
        """

        while True:

            # Read the frame
            success, frame = self.video.read()
            if not success:
                break
                
            # Prepare the image for the model
            model_input = cv2.resize(frame, (224, 224))
            model_input = np.expand_dims(model_input, axis=0)
            model_input = model_input / 255.0

            # Predict
            prediction = self.model.predict(model_input)
            prediction = np.argmax(prediction)
            prediction = self.class_dict[prediction]

            # Compare the current frame with the previous one
            if self.prev_frame is not None:

                # Convert the images to grayscale
                grayA = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
                grayB = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Calculate the SAM (Spectral Angle Mapper) --> if SAM > 0.45, the images are different
                new_score = sam(grayA, grayB)
                if new_score > 0.45:

                    # Calculate the majority prediction and add it to the counter dictionary
                    majority_prediction = Counter(self.prev_predictions).most_common(1)[0][0]
                    self.counter_dict[majority_prediction] += 1

                    # Save the previous frame to the Prediction folder/ Class folder, and reset the previous predictions
                    cv2.imwrite(f'Predictions/{majority_prediction}/{self.counter_dict[majority_prediction]}.jpg', self.prev_frame)
                    self.prev_predictions = [prediction]

                # If the images are the same, add the prediction to the previous predictions list
                else:
                    self.prev_predictions.append(prediction)



            # Update the variables
            self.prev_frame = frame
            
            # Draw the counter dictionary
            y_position = 20
            for key in self.counter_dict.keys():
                cv2.putText(frame,f"{key}: {self.counter_dict[key]}",(10,y_position),cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,(0,0,255),2)
                y_position += 30


            # Draw Rectangle Behind Prediction Text
            cv2.rectangle(frame, (1080 // 2 - 10, 0), (1080 // 2 + 200, 50), (0,0,0), -1)

            # Draw the Prediction 
            cv2.putText(frame,f"Prediction: {prediction}",(1080 // 2,20),cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,(0,0,255),2)

            # Draw the frame number
            fps = self.video.get(cv2.CAP_PROP_FPS)
            fps = round(fps, 2)
            cv2.putText(frame,f"FPS: {fps}",(1080,50),cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,(0,0,255),2)

            # Draw the majority prediction
            if len(self.prev_predictions) > 0:
                maj_pred_disp = Counter(self.prev_predictions).most_common(1)[0][0]
                cv2.putText(frame,f"Maj Pred: {maj_pred_disp}",(1080,80),cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,(0,0,255),2)

            # Show the frame and update the frame number
            cv2.imshow('Counter', frame)
            self.frame_number += 1

            # If the user presses q, the program stops
            if cv2.waitKey(1) & 0xFF == ord('q'):
              break

        # Release the video and destroy all the windows
        cv2.destroyAllWindows()

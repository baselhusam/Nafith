import argparse
from utils import ImageProcessor,ObjectCounter


class Classifier:

    """
    This class is used to classify objects in a video.
    It creates an ImageProcessor object and an ObjectCounter object.
    Then, it classifies the objects in the video.
    """

    def __init__(self):
        self.processor=ImageProcessor()

    def classify_objects(self):

        """
        Classify the objects in a video.
        It creates an argument parser and a dictionary with the classes.
        Then, it creates an ObjectCounter object and classifies the objects in the video.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Note: This function is meant to be run from the terminal.
                Before running it, make sure you have the video in the assets folder.

        Note2: This functions will display the video and the number of objects in the video.
        """
        
        # Create an argument parser
        parser=argparse.ArgumentParser(description='Classify objects in a video.')
        parser.add_argument('--video_path',type=str, help='Path to the video file')
        args=parser.parse_args()
        
        # Create a dictionary with the classes and create an ObjectCounter object
        class_dict={0:"Bus",1:"Car",2:"Truck"}
        counter=ObjectCounter(args.video_path, class_dict)
        
        # Classify the objects
        counter.count_objects()



if __name__=="__main__":
    classifier=Classifier()
    classifier.classify_objects()





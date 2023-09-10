import pandas as pd
import argparse

from utils import ImageProcessor

class ImageReader:
    def __init__(self):
        self.processor = ImageProcessor()

    def read_and_split(self):

        """
        Read the images and split them into train, validation and test sets
        Also save the numpy arrays and the dataframe as csv file.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        Note: This function is meant to be run from the terminal.
              Before running it, make sure you have the images in the assets folder.
        """

        # Create an argument parser
        parser = argparse.ArgumentParser(description='Read images and split them into train/validation/test sets.')
        parser.add_argument('--img_size', type=int, help='Size of the images')
        
        # Parse the arguments and get the image size
        args = parser.parse_args()
        args.img_size = args.img_size if args.img_size else 224
        
        # Get the images path
        car_imgs = self.processor.get_imgs_path("car")
        bus_imgs = self.processor.get_imgs_path("bus")
        truck_imgs = self.processor.get_imgs_path("truck")

        # Create a dataframe with the images and their labels
        df = pd.DataFrame(columns=['image', 'label'])
        df['image'] = bus_imgs + car_imgs + truck_imgs
        df['label'] = ['bus'] * len(bus_imgs) + ['car'] * len(car_imgs) + ['truck'] * len(truck_imgs)

        # Save the dataframe as a csv file and read it
        df.to_csv("assets/Data/data.csv", index=False)
        df = pd.read_csv("assets/Data/data.csv")

        # Save the images as a numpy array
        X, y = self.processor.imgs_as_array(df, len(df), args.img_size)

        # Split the data into train, validation and test sets
        X_train,X_val,X_test,y_train,y_val,y_test=self.processor.split_data(X,y,val_size=0.1,test_size=0.1)

        # Save the numpy arrays
        self.processor.save_npy(X_train,X_val,X_test,y_train,y_val,y_test)

if __name__ == "__main__":
    reader = ImageReader()
    reader.read_and_split()

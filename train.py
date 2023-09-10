import numpy as np
import argparse

from utils import ModelBuilder, load_data

class ModelTrainer:

    """
    This class is used to train the model.
    It loads the data and creates a ModelBuilder object.
    Then, it builds the model, create the generators and train the model.
    Finally, it saves the model and the history.

    Note: This class is meant to be run from the terminal.
            Before running it, make sure you have the numpy arrays in the assets/Data folder.
    """

    def __init__(self):
        
        # Load the data
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = load_data()

        # Create a ModelBuilder object
        self.ModelBuilder = ModelBuilder(self.X_train,self.y_train,self.X_val,self.y_val, self.X_test, self.y_test)

    def train_model(self):

        """
        Train the model.
        It creates an argument parser.
        Then, it builds the model, create the generators and train the model.
        Finally, it saves the model and the history.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Note: This function is meant to be run from the terminal.
                Before running it, make sure you have the numpy arrays in the assets/Data folder.
        """

        # Create an argument parser
        parser = argparse.ArgumentParser(description='Train a model.')
        parser.add_argument('--epochs', type=int, help='Number of epochs')
        parser.add_argument('--learning_rate', type=float, help='Learning rate')
        parser.add_argument('--is_transfer_learning', action='store_true', help='Is transfer learning')

        
        # Parse the arguments and get the number of epochs and the learning rate
        args = parser.parse_args()
        args.epochs = args.epochs if args.epochs else 100
        args.learning_rate = args.learning_rate if args.learning_rate else 0.0001
        
        # Build the model
        builder = self.ModelBuilder
        model = builder.build_model() if not args.is_transfer_learning else builder.build_model_vgg()

        # Create the generators and Train the model
        train_gen, val_gen, _ = builder.create_generators()
        history = builder.train_model(model, (train_gen, val_gen), args.epochs, args.learning_rate)

        # Save the model and the history
        changed_path = "_Trans" if args.is_transfer_learning else ""
        builder.save_model(model, changed_path)
        builder.save_history(history, changed_path)



if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_model()

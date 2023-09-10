import numpy as np
import argparse
from utils import ModelBuilder, ModelTester, load_data


class ModelEvaluator:
    def __init__(self):
        
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = load_data()


    def evaluate_model(self):

        """
        Evaluate the model.
        It loads the data and creates a ModelBuilder object.
        Then, it builds the model and evaluate the sets. Train, validation and test.
        Finally, it saves the metrics in a json file.

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
        parser.add_argument('--is_transfer_learning', action='store_true', help='Is transfer learning')
        args = parser.parse_args()

        # Create a ModelBuilder object and build the model 
        builder = ModelBuilder(self.X_train,self.y_train,self.X_val,self.y_val, self.X_test, self.y_test)
        model = builder.build_model() if not args.is_transfer_learning else builder.build_model_vgg()
        
        # Create a ModelTester object and evaluate the model
        tester = ModelTester(model)
        
        # Check if the model is transfer learning or not and change the path
        changed_path = "_Trans" if args.is_transfer_learning else ""

        # Evaluate the sets. Train, validation and test.
        train_metrics=tester.evaluate_set(self.X_train,self.y_train,'Train', changed_path)
        val_metrics=tester.evaluate_set(self.X_val,self.y_val,'Validation', changed_path)
        test_metrics=tester.evaluate_set(self.X_test,self.y_test,'Test', changed_path)
        
        # Save the metrics in a json file
        tester.save_metrics(train_metrics,val_metrics,test_metrics, changed_path)



if __name__ == "__main__":
    evaluator=ModelEvaluator()
    evaluator.evaluate_model()

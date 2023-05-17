#---Imports--------------+
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np
import os
import shutil
import json


"""

"""
class regression_GP():
    def __init__(self, X_train_hist, X_train_cat, y_train, X_test_hist, \
            X_test_cat, y_test, X_val_hist, X_val_cat, y_val, model_name="regression_GP"):
        """

        """
        self.model_name = model_name

        self.X_train, self.y_train, self.X_test, self.y_test = self._prepare_input_sets(X_train_hist, \
                X_train_cat, y_train,  X_test_hist, X_test_cat, y_test, \
                X_val_hist, X_val_cat, y_val)
        self.model = self._create_model()
         
    
    def _prepare_input_sets(self, X_train_hist, X_train_cat, y_train, X_test_hist, X_test_cat, \
            y_test, X_val_hist, X_val_cat, y_val):
        """

        """
        # Reshape the input data
        X_train_reshaped = np.reshape(X_train_hist, (X_train_hist.shape[0], -1))
        X_test_reshaped = np.reshape(X_test_hist, (X_test_hist.shape[0], -1))
        X_val_reshaped = np.reshape(X_val_hist, (X_val_hist.shape[0], -1))

        X_train_reshaped = np.concatenate((X_train_reshaped, np.reshape(X_train_cat, (-1, 1))), axis=1)
        X_test_reshaped = np.concatenate((X_test_reshaped, np.reshape(X_test_cat, (-1, 1))), axis=1)
        X_val_reshaped = np.concatenate((X_val_reshaped, np.reshape(X_val_cat, (-1, 1))), axis=1)
        
        # Stack the testing and validation set together
        X_test_val = np.vstack((X_test_reshaped, X_val_reshaped))
        y_test_val = np.hstack((y_test, y_val))
        return X_train_reshaped, y_train, X_test_val, y_test_val 


    def _create_model(self):
        """

        """
        kernel = RBF()
        model = GaussianProcessRegressor(kernel=kernel)
        return model 
        
        
    def train(self):
        """

        """
        self.model.fit(self.X_train, self.y_train)


    def evaluate(self, print_perf=True, save_stats=True):
        """
        
        """
        y_pred = self.model.predict(self.X_test)
        train_score = self.model.score(self.X_train, self.y_train)
        test_score = self.model.score(self.X_test, self.y_test)

        if print_perf:
            print(f"Model score on training data: {train_score}\t")
            print(f"Model score on testing data: {test_score}\t")

        stats = {
            'train': train_score, 
            'test': test_score,
            'prediction': y_pred.tolist(),
            'y_test': self.y_test.tolist()
        }

        dirname_here = os.getcwd()
        if save_stats:
            with open(self.model_name + "_val_data" + '.json', 'w') as f:
                json.dump(stats, f)
            try:
                shutil.move(dirname_here + "/" + self.model_name + "_val_data" + ".json", \
                        dirname_here + "/val_stats/" + self.model_name + ".json") 
            except FileNotFoundError as e:
                print(f"Could not save validation statistics. Error: {e}")
       
 

 
        

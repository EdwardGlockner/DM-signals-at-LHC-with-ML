#---Imports--------------+
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np


"""

"""

class regression_GP():
    def __init__(self, X_train_hist, X_train_cat, y_train, X_test_hist, \
            X_test_cat, y_test, X_val_hist, X_val_cat, y_val):
        """

        """
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val

        self.X_train, self.X_test, self.X_val = self._prepare_input_sets(X_train_hist, \
                X_train_cat, X_test_hist, X_test_cat, X_val_hist, X_val_cat)
        self.model = self._create_model 
         
    
    def _prepare_input_sets(self, X_train_hist, X_train_cat, X_test_hist, X_test_cat, \
            X_val_hist, X_val_cat):
        """

        """
        # Reshape the input data
        X_train_reshaped = np.reshape(X_train_hist, (X_train_hist.shape[0], -1))
        X_test_reshaped = np.reshape(X_test_hist, (X_test_hist.shape[0], -1))
        X_val_reshaped = np.reshape(X_val_hist, (X_val_hist.shape[0], -1))

        X_train_reshaped = np.concatenate((X_train_reshaped, np.reshape(X_train_cat, (-1, 1))), axis=1)
        X_test_reshaped = np.concatenate((X_test_reshaped, np.reshape(X_test_cat, (-1, 1))), axis=1)
        X_val_reshaped = np.concatenate((X_val_reshaped, np.reshape(X_val_cat, (-1, 1))), axis=1)

        return X_train_reshaped, X_test_reshaped, X_val_reshaped


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


    def evaluate(self, print_perf=True):
        """
        
        """
        y_pred = self.model.predict(self.X_test)
        train_score = self.model.score(self.X_train, self.y_train)
        test_score = self.model.score(self.X_test, self.y_test)
        val_score = self.model.score(self.X_val, self.y_val)

        if print_perf:
            print(f"Model score on training data: {train_score}\t")
            print(f"Model score on testing data: {test_score}\t")
            print(f"Model score on validation data: \t {val_score}")
        


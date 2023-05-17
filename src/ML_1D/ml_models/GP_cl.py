#---Imports--------------+
from sklearn.gaussian_process import GaussianProcessClassifier
import numpy as np


"""

"""

class classification_GP():
    def __init__(self, X_train, y_train, X_test, y_test, X_val, y_val):
        """

        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val

        self._prepare_sets()
        self.model = self._create_model 
         

    
    def _prepare_sets(self):
        """

        """
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], -1))
        self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], -1))
        self.X_val = np.reshape(self.X_val, (self.X_val.shape[0], -1))
        

    def _create_model(self):
        """

        """
        model = GaussianProcessClassifier()
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
        


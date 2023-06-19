#-b--Imports--------------+
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.gaussian_process import GaussianProcessClassifier
import numpy as np
import os
import shutil
import json

"""

"""
class classification_GP():
    def __init__(self, X_train, y_train, X_test, y_test, model_name="classification_GP"):
        """

        """
        self.model_name = model_name
        
        self.X_train, self.y_train, self.X_test, self.y_test = \
                self._prepare_sets(X_train, y_train, X_test, y_test)
        self.model = self._create_model()

    
    def _prepare_sets(self, X_train, y_train, X_test, y_test):
        """

        """
        X_train = np.reshape(X_train, (X_train.shape[0], -1))
        X_test = np.reshape(X_test, (X_test.shape[0], -1))
        
        return X_train, y_train, X_test, y_test

    def _create_model(self):
        """

        """
        model = GaussianProcessClassifier()
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
        auc_score = roc_auc_score(self.y_test, y_pred)

        # Calculate recall
        recall = recall_score(self.y_test, y_pred)

        if print_perf:
            print(f"Model score on training data: {train_score}\t")
            print(f"Model score on validation data: {test_score}\t")
        
        stats = {
            'train': train_score, 
            'test': test_score,
            'auc': auc_score,
            'recall':recall,
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
       
 


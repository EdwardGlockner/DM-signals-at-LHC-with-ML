from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import os
import shutil
import json

class regression_GP():
    def __init__(self, X_train, y_train, X_test, y_test, model_name="regression_GP"):
        self.model_name = model_name
        print(X_train.shape)
        print(y_train.shape)
        print(X_test.shape)
        print(y_test.shape)
        self.X_train, self.y_train, self.X_test, self.y_test = \
            self._prepare_sets(X_train, y_train, X_test, y_test)
        self.model = self._create_model()

    def _prepare_sets(self, X_train, y_train, X_test, y_test):
        X_train = np.reshape(X_train, (X_train.shape[0], -1))
        X_test = np.reshape(X_test, (X_test.shape[0], -1))

        return X_train, y_train, X_test, y_test

    def _create_model(self):
        model = GaussianProcessRegressor()
        return model

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self, print_perf=True, save_stats=True):
        y_pred = self.model.predict(self.X_test)
        train_score = self.model.score(self.X_train, self.y_train)  # Replace with appropriate regression metric
        test_score = self.model.score(self.X_test, self.y_test)  # Replace with appropriate regression metric
        # Calculate MAPE
        mape = np.mean(np.abs((self.y_test - y_pred) / self.y_test)) * 100

        # Calculate MAE
        mae = mean_absolute_error(self.y_test, y_pred)

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))

        if print_perf:
            print(f"Model score on training data: {train_score}\t")
            print(f"Model score on validation data: {test_score}\t")

        stats = {
            'train': train_score,
            'test': test_score,
            'mape': mape,
            'mae':mae,
            'rmse':rmse,
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


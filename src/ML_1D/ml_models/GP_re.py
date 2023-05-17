#---Imports--------------+
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
import numpy as np
import os
import shutil
import json
import torch
import gpytorch
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood

"""

"""
class regression_GP(ExactGP):
    def __init__(self, X_train_hist, X_train_cat, y_train, X_test_hist, X_test_cat, y_test, X_val_hist, X_val_cat, y_val, likelihood, model_name="regression_GP"):
        super().__init__(X_train_hist, y_train, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

        # Prepare input sets
        self.X_train = self._prepare_input(X_train_hist, X_train_cat)
        self.y_train = y_train
        self.X_test = self._prepare_input(X_test_hist, X_test_cat)
        self.y_test = y_test
        self.X_val = self._prepare_input(X_val_hist, X_val_cat)
        self.y_val = y_val

    def forward(self, x):
        with gpytorch.settings.fast_pred_var():
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def train_model(self, likelihood, num_iterations, lr):
        self.train()
        likelihood.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, self)

        for i in range(num_iterations):
            optimizer.zero_grad()
            output = self(self.X_train)
            loss = -mll(output, self.y_train)
            loss.backward()
            optimizer.step()

    def _prepare_input(self, X_hist, X_cat):
        X_reshaped = X_hist.reshape(X_hist.shape[0], -1)
        X_cat_reshaped = X_cat.reshape(-1, 1)
        return torch.from_numpy(np.concatenate((X_reshaped, X_cat_reshaped), axis=1)).float()
 
    """ 
    def train(self):

        self.model.fit(self.X_train, self.y_train)


    def evaluate(self, print_perf=True, save_stats=True):
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
    """ 
 

 
        
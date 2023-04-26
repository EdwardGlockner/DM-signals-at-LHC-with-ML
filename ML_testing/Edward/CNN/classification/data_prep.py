import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import os
import tensorflow as tf

def create_datasets(dataset):
    # Load the dataset
    if dataset == "mnist":
        mnist = tf.keras.datasets.mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

    elif dataset == "mnistfashion":
        mnist_fashion = tf.keras.datasets.fashion_mnist
        (X_train, y_train), (X_test, y_test) = mnist_fashion.load_data()

    else:
        return None

    # Reshape and normalize
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_train = X_train / 255

    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    X_test = X_test / 255

    # Label encoding
    y_train = tf.one_hot(y_train.astype(np.int32), depth=10) 
    y_test = tf.one_hot(y_test.astype(np.int32), depth=10)
    
    return X_train, y_train, X_test, y_test 

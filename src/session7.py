# generic tools
import numpy as np

# tools from sklearn
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# tools from tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model

# matplotlib
import matplotlib.pyplot as plt

def download_data():
    data, labels = fetch_openml('mnist_784', version=1, return_X_y=True)

    # normalise data
    data = data.astype("float")/255.0

    # split data
    (X_train, X_test, y_train, y_test) = train_test_split(data,
                                                        labels, 
                                                        test_size=0.2)
    return (X_train, X_test, y_train, y_test)

def convert_labels(y_train, y_test):
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)
    return y_train, y_train

def main():
    X_train, X_test, y_train, y_test = download_data()
    y_train, y_train = convert_labels(y_train, y_test)

if __name__ =="__main__":
    main()
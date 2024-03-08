import os
import sys
sys.path.append("../../..")
# openCV
import cv2

# cifar10 dataset
from tensorflow.keras.datasets import cifar10


def logisticregression():
    # load the Cifar10 dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # Preprocess the data
    # converting to greyscale
    grayed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # normalize
    hist = cv2.calcHist([grayed_image], [0,1,2], None, [255,255,255], [0,256, 0,256, 0,256])
    normalized_hist = cv2.normalize(hist, hist, 0, 1.0, cv2.NORM_MINMAX)
    # reshape
    X_train_scaled.reshape(-1, 1024)
    X_test_scaled.reshape(-1, 1024)

    # train a classifier on the data
    
    # a logistic regression classifier and a neural network classifier
    
    # save a classification report
    
    # save a plot of the loss curve during training

# np.array(list)
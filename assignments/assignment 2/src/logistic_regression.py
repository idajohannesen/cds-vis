import os
import sys
sys.path.append("../../..")
# openCV
import cv2
# numpy
import numpy as np
# matplotlib
import matplotlib.pyplot as plt
# class util functions
from utils.imutils import jimshow as show
from utils.imutils import jimshow_channel as show_channel
import utils.classifier_utils as clf_util

# Import sklearn metrics
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# cifar10 dataset
from tensorflow.keras.datasets import cifar10

def logisticregression():
    # loading the cifar10 dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    # Preprocess the data
    X_train_list=[]
    X_test_list=[]

    for image in X_train: # preprocessing the training images
        gray_train = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to grayscale
        normalized = gray_train/255 # normalize
        X_train_list.append(normalized) # add to list
    X_preprocess_train_array = np.array(X_train_list) # turn list back into an array
    # reshape
    X_preprocess_train = X_preprocess_train_array.reshape(-1, 1024)

    for image in X_test: # preprocessing the testing images
        gray_test = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to grayscale
        normalized = gray_test/255 # normalize
        X_test_list.append(normalized) # add to list
    X_preprocess_test_array = np.array(X_test_list) # turn list back into an array
    # reshape
    X_preprocess_test = X_preprocess_test_array.reshape(-1, 1024)

    # making a logistic regression classifier
    clf = LogisticRegression(tol=0.1, 
                            solver='saga',
                            multi_class='multinomial').fit(X_preprocess_train, y_train)
    
    # defining labels
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # testing accuracy and making a classification report for the logistic regression classifier
    y_pred = clf.predict(X_preprocess_test)
    cm = metrics.classification_report(y_test, y_pred, target_names=classes)
    print(cm)

    # saving classification report as a .txt file
    text_file = open(r'../output/LR_classification_report.txt', 'w')
    text_file.write(cm)
    text_file.close()

if __name__ =="__main__":
    logisticregression()
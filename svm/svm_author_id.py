#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score

# clf = SVC(kernel="linear")
# clf.fit(features_train, labels_train)
# prd = clf.predict(features_test)
# acc = accuracy_score(prd, labels_test)
# print(acc)
#########################################################

#########################################################
### your code goes here ###
features_train = features_train[:int(len(features_train)/100)]
labels_train = labels_train[:int(len(labels_train)/100)]

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

clf = SVC(C=10000, kernel="rbf")
clf.fit(features_train, labels_train)
prd = clf.predict(features_test)
acc = accuracy_score(prd, labels_test)
print(acc)
for i in [10, 26, 50]:
    print('[{}]: {}'.format(i, prd[i]))

from collections import Counter

c = Counter(prd)
print("No of predictions for Chris(1):", c[1])
#########################################################

#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

clf_knn = KNeighborsClassifier()
clf_knn.fit(features_train, labels_train)
prd_knn = clf_knn.predict(features_test)
print("Accuracy kNN:", accuracy_score(labels_test, prd_knn))

prettyPicture(clf_knn, features_test, labels_test, 'knn-test.png')

from sklearn.ensemble import AdaBoostClassifier

clf_ada = AdaBoostClassifier()
clf_ada.fit(features_train, labels_train)
prd_ada = clf_ada.predict(features_test)
print("Accuracy AdaBoost:", accuracy_score(labels_test, prd_ada))

prettyPicture(clf_ada, features_test, labels_test, 'ada-test.png')

from sklearn.ensemble import RandomForestClassifier

clf_forest = RandomForestClassifier()
clf_forest.fit(features_train, labels_train)
prd_forest = clf_forest.predict(features_test)
print("Accuracy RandomForest:", accuracy_score(labels_test, prd_forest))

prettyPicture(clf_forest, features_test, labels_test, 'forest-test.png')
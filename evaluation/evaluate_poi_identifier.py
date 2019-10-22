#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

### add more features to features_list!
features_list = ["poi", "salary"]


data = featureFormat(data_dict, features_list, sort_keys = '../tools/python2_lesson14_keys.pkl')
labels, features = targetFeatureSplit(data)

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)


### your code goes here 
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
prd = clf.predict(features_test)
print('score -', clf.score(features_test, labels_test))

count_prd_test, count_poi_test = 0, 0
for l in prd:
    if l == 1:
        count_prd_test += 1
for l in labels_test:
    if l == 1:
        count_poi_test += 1

# How many POIs are predicted for the test set for your POI identifier? - 4
print('poi predicted in test set -', count_prd_test)
# How many people total are in your test set?
print('total in test set -', len(prd))
# If your identifier predicted 0. (not POI) for everyone in the test set, what would its accuracy be?
print('total poi in test set -', count_poi_test, ', accuracy -', float(len(labels_test) - count_poi_test) / len(labels_test)) # 4, 0.8620689655172413

# Do you get any true positives?
true_positive = 0
for i in range(len(labels_test)):
    if labels_test[i] and prd[i]:
        true_positive += 1

print('true_positive -', true_positive) # 0

from sklearn.metrics import precision_score, recall_score

print('precision_score -', precision_score(labels_test, prd))
print('recall_score -', recall_score(labels_test, prd))

# 34. Quiz. How many true positives?
def true_positives():
    predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
    labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

    true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0
    for i in range(len(predictions)):
        if predictions[i]:
            if labels[i]:
                true_positive += 1
            else:
                false_positive += 1
        if not predictions[i]:
            if not labels[i]:
                true_negative += 1
            else:
                false_negative += 1
    print('true_positive -', true_positive, ', true_negative -', true_negative)
    print('false_positive -', false_positive, ', false_negative -', false_negative)

    print('precision_score - ', precision_score(labels, predictions)) # float(true_positive) / (true_positive + false_positive)
    print('recall_score - ', recall_score(labels, predictions)) # float(true_positive) / (true_positive + false_negative)

true_positives()

'''
Q: “My true positive rate is high, which means that when a ___ is present in the test data, I am good at flagging him or her.”

A: POI
'''

'''
Q: “My identifier doesn’t have great _, but it does have good _. That means that, nearly every time a POI shows up in my test set, I am able to identify him or her. The cost of this is that I sometimes get some false positives, where non-POIs get flagged.”

A: precision, recall
'''

'''
Q: “My identifier doesn’t have great _, but it does have good ____. That means that whenever a POI gets flagged in my test set, I know with a lot of confidence that it’s very likely to be a real POI and not a false alarm. On the other hand, the price I pay for this is that I sometimes miss real POIs, since I’m effectively reluctant to pull the trigger on edge cases.”

A: recall, precision
'''

'''
Q: “My identifier has a really great _.
This is the best of both worlds. Both my false positive and false negative rates are _, which means that I can identify POI’s reliably and accurately. If my identifier finds a POI then the person is almost certainly a POI, and if the identifier does not flag someone, then they are almost certainly not a POI.”

A: F1 Score / low
'''
#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)

### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load(open(words_file, "rb"))
authors = pickle.load(open(authors_file, "rb"))

### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()
words = vectorizer.get_feature_names()

### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]

### your code goes here
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(min_samples_split=40)
clf.fit(features_train, labels_train)
prd = clf.predict(features_test)

from sklearn.metrics import accuracy_score
print('accuracy_score - ', accuracy_score(prd, labels_test))
# Signature words 제거 전 - 0.9476678043230944 Overfit 하지 않는 듯 하다! 이러면 이상한 데이터.
# Signature words 제거 후 - 0.8162684869169511 이제 슬슬 Overfit 하는 군.
idx_imps = []
for i in range(len(clf.feature_importances_)):
    idx_imps.append([i, clf.feature_importances_[i]])
    if clf.feature_importances_[i] > 0.2:
        print('importance - {}, word - {}'.format(clf.feature_importances_[i], words[i]))

idx_imps = sorted(idx_imps, key=lambda idx_imp: 0-idx_imp[1])
print('most important index - {}, word - {}'.format(idx_imps[0], words[idx_imps[0][0]])) 
# 33614, 0.7647058823529412
# sshacklensf

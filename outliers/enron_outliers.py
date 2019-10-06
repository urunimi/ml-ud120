#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot as plt

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "rb"))

### Outlier 인 Total 제거하고 시작
data_dict.pop('TOTAL',0)

features = ["salary", "bonus"]
data = featureFormat(data_dict, features)
salaries, bonuses = [], []

### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    salaries += [salary]
    bonuses += [bonus]
    plt.scatter(salary, bonus)

from sklearn.linear_model import LinearRegression
import numpy

salaries = numpy.reshape( numpy.array(salaries), (len(salaries), 1))
bonuses  = numpy.reshape( numpy.array(bonuses), (len(bonuses), 1))

from sklearn.model_selection import train_test_split
salaries_train, salaries_test, bonuses_train, bonuses_test = train_test_split(salaries, bonuses, test_size=0.1, random_state=42)

reg = LinearRegression()
reg.fit(salaries_train, bonuses_train)

plt.plot(salaries_test, reg.predict(salaries_test), color="b")
print('Score - ', reg.score(salaries_test, bonuses_test))

plt.xlabel("salary")
plt.ylabel("bonus")
plt.savefig('enron_outliers.png')

# We would argue that there’s 4 more outliers to investigate; let's look at a couple of them. Two people made bonuses of at least 5 million dollars, and a salary of over 1 million dollars; in other words, they made out like bandits. What are the names associated with those points?
sal_names = [(v['salary'], k) for k, v in data_dict.items()]
sal_names_sorted = sorted(sal_names, key=lambda sal_name: 0 if type(sal_name[0]) is str else sal_name[0])
print('bandits - ', sal_names_sorted[-2:])
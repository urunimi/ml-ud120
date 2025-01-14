#!/usr/bin/python

import pickle
import sys

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

sys.path.append("../tools/")


def get_outliers(data_dict) -> []:
    return ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']
    # outliers = []
    # import matplotlib.pyplot as plt
    # data_wo_outliers = {}
    # for name, features in data_dict.items():
    #     salary = 0 if features['salary'] == 'NaN' else features['salary']
    #     bonus = 0 if features['bonus'] == 'NaN' else features['bonus']
    #     total_payments = 0 if type(features['total_payments']) == str else features['total_payments']
    #     exed_stock_options = 0 if features['exercised_stock_options'] == 'NaN' else features['exercised_stock_options']
    #     if salary == 0 and bonus == 0 and exed_stock_options == 0 and total_payments == 0:
    #         outliers.append(name)
    #     else:
    #         data_wo_outliers[name] = features
    #         plt.scatter(salary, bonus)
    #     # plt.annotate(name, (salary, bonus))
    # print(outliers)
    # plt.show()

def get_results(features, labels, clf):
    # Provided to give you a starting point. Try a variety of classifiers.

    ### Task 5: Tune your classifier to achieve better than .3 precision and recall 
    ### using our testing script. Check the tester.py script in the final project
    ### folder for details on the evaluation method, especially the test_classifier
    ### function. Because of the small size of the dataset, the script uses
    ### stratified shuffle split cross validation. For more info: 
    ### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

    # Example starting point. Try investigating other evaluation techniques!
    from sklearn.model_selection import train_test_split
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    clf.fit(features_train, labels_train)
    acc_score = clf.score(features_test, labels_test)
    print('accuracy_score -', acc_score)
    
    from sklearn.metrics import precision_score, recall_score
    prd = clf.predict(features_test)
    print('precision_score -', precision_score(labels_test, prd))
    print('recall_score -', recall_score(labels_test, prd))
    return clf, acc_score

def find_poi():
    ### Task 1: Select what features you'll use.
    ### features_list is a list of strings, each of which is a feature name.
    ### The first feature must be "poi".
    financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] # (Units = USD) 

    features = ['poi',
        'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 
        'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 
        'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 
        'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

    # financial_features = ['poi', 'salary', 'total_payments', 'bonus', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock']
    #'email_address',
    email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] # units = number of emails messages; except ‘email_address’, which is a text string
    # email_features = ['poi', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

    ### Load the dictionary containing the dataset
    with open("final_project_dataset.pkl", "rb") as data_file:
        data_dict = pickle.load(data_file)

    ### Task 2: Remove outliers
    for name in get_outliers(data_dict):
        data_dict.pop(name)

    ### Task 3: Create new feature(s)
    ### Store to my_dataset for easy export below.
    my_dataset = data_dict

    ### Extract features and labels from dataset for local testing
    data = featureFormat(my_dataset, features, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    ### Task 4: Try a varity of classifiers
    ### Please name your classifier clf for easy export below.
    ### Note that if you want to do PCA or other multi-stage operations,
    ### you'll need to use Pipelines. For more info:
    ### http://scikit-learn.org/stable/modules/pipeline.html

    from sklearn.naive_bayes import GaussianNB
    clf, acc_score = get_results(features, labels, GaussianNB())

    from sklearn.tree import DecisionTreeClassifier
    d_clf, d_acc_score = get_results(features, labels, DecisionTreeClassifier(min_samples_split=5))
    if d_acc_score > acc_score:
        clf = d_clf

    # from sklearn.model_selection import GridSearchCV
    # from sklearn.svm import SVC
    # param_grid = {
    #      'C': [1e3, 5e3, 1e4, 5e4, 1e5],
    #       'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
    #       }
    # clf = get_results(features, labels, GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid))

    ### Task 6: Dump your classifier, dataset, and features_list so anyone can
    ### check your results. You do not need to change anything below, but make sure
    ### that the version of poi_id.py that you submit can be run on its own and
    ### generates the necessary .pkl files for validating your results.
    dump_classifier_and_data(clf, my_dataset, features)

find_poi()
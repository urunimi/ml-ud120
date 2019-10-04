#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import re

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

num_persons = len(enron_data)
print('num_persons - {}'.format(num_persons))

num_of_poi = 0
for name, prop in enron_data.items():
    if prop['poi']:
        num_of_poi += 1

print('num_of_poi in data - {}'.format(num_of_poi))

num_of_poi_all = 0
with open("../final_project/poi_names.txt") as f:
    for line in f.readlines():
        if re.match('\((y|n)\)', line):
            num_of_poi_all += 1

print('num_of_poi in all - {}'.format(num_of_poi_all))

'''
Like any dict of dicts, individual people/features can be accessed like so:

enron_data["LASTNAME FIRSTNAME"]["feature_name"]
or, sometimes
enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"]["feature_name"]

What is the total value of the stock belonging to James Prentice?
'''
stocks = 0
james_name = 'Prentice James'.upper() 
for name, feature in enron_data.items():
    if james_name in name:
        stocks += feature['total_stock_value']

print('The total value of the stock belonging to James Prentice - ', stocks)

name_chairman = 'Lay Kenneth L'.upper()
name_ceo = 'Skilling Jeffrey K'.upper()
name_cfo = 'Fastow Andrew S'.upper()

# How many email messages do we have from Wesley Colwell to persons of interest?
print('Num of email messages from Wesley Colwell to persons of interest - ', enron_data['COLWELL WESLEY']['from_this_person_to_poi'])

# What’s the value of stock options exercised by Jeffrey K Skilling?
stock_exercised = enron_data[name_ceo]['exercised_stock_options']
print('The value of stock options exercised by Jeffrey K Skilling - ', stock_exercised)

# Who was CFO (chief financial officer) of Enron during most of the time that fraud was going on?
print('total_payments of {} - {}'.format(name_cfo, enron_data[name_cfo]['total_payments']))

most_fraud_amount = 0
most_fraud_name = ''
for name in [name_chairman, name_ceo, name_cfo]:
    payment = enron_data[name]['total_payments']
    if most_fraud_amount < payment:
        most_fraud_amount = payment
        most_fraud_name = name

frauds = dict((name, enron_data[name]['total_payments']) for name in [name_chairman, name_ceo, name_cfo])
max_name = max(frauds, key=frauds.get)
print('The most fraud - {}, amount: - {}'.format(max_name, enron_data[max_name]['total_payments']))

# How many folks in this dataset have a quantified salary? 
# What about a known email address?
# How many POIs in the E+F dataset have “NaN” for their total payments? What percentage of POI’s as a whole is this?

num_salary = 0
num_email = 0
num_total_payments_nan = 0
num_total_payments_poi_nan = 0
nan = 'NaN'
for name, feature in enron_data.items():
    if feature['salary'] != nan:
        num_salary += 1
    if feature['email_address'] != nan:
        num_email += 1
    
    if feature['total_payments'] == nan:
        num_total_payments_nan += 1
        if feature['poi']:
            num_total_payments_poi_nan += 1


print('N of salary - {}, N of email - {}'.format(num_salary, num_email))
print('Percentage of NaN payments - {}%'.format(100.0 * num_total_payments_nan/num_persons))
print('Percentage of NaN payments for POI - {}%'.format(100.0 * num_total_payments_poi_nan/num_of_poi))

'''
If you added in, say, 10 more data points which were all POI’s, and put “NaN” for the total payments for those folks, the numbers you just calculated would change.
What is the new number of people of the dataset? What is the new number of folks with “NaN” for total payments?
'''
# 156, 31

'''
What is the new number of POI’s in the dataset? What is the new number of POI’s with NaN for total_payments?
'''
# 28, 10

'''
Once the new data points are added, do you think a supervised classification algorithm might interpret “NaN” for total_payments as a clue that someone is a POI?
'''
# Yes. 0% -> 36% 로 점프 (Non POI 는 20%) 했기 때문에 POI 가 total_payment 가 NaN 일 가능성이 높아짐

'''
Adding in the new POI’s in this example, none of whom we have financial information for, has introduced a subtle problem, that our lack of financial information about them can be picked up by an algorithm as a clue that they’re POIs. Another way to think about this is that there’s now a difference in how we generated the data for our two classes--non-POIs all come from the financial spreadsheet, while many POIs get added in by hand afterwards. That difference can trick us into thinking we have better performance than we do--suppose you use your POI detector to decide whether a new, unseen person is a POI, and that person isn’t on the spreadsheet. Then all their financial data would contain “NaN” but the person is very likely not a POI (there are many more non-POIs than POIs in the world, and even at Enron)--you’d be likely to accidentally identify them as a POI, though!
This goes to say that, when generating or augmenting a dataset, you should be exceptionally careful if your data are coming from different sources for different classes. It can easily lead to the type of bias or mistake that we showed here. There are ways to deal with this, for example, you wouldn’t have to worry about this problem if you used only email data--in that case, discrepancies in the financial data wouldn’t matter because financial features aren’t being used. There are also more sophisticated ways of estimating how much of an effect these biases can have on your final answer; those are beyond the scope of this course.
For now, the takeaway message is to be very careful about introducing features that come from different sources depending on the class! It’s a classic way to accidentally introduce biases and mistakes.
'''
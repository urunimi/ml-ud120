#!/usr/bin/python

import math


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    import operator

    ### your code goes here
    errors = [abs(a-b) for a,b in zip(predictions, net_worths)]
    all_data = zip(predictions, net_worths, errors)
    all_data_sorted = sorted(all_data, key=operator.itemgetter(2))
    cleaned_data = all_data_sorted[:int(len(predictions)*0.9)]
    return cleaned_data

if __name__ == '__main__':
    predictions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    net_worths = [1, 2, 3, 4, 5, 3, 7, 8, 6, 2]
    outlierCleaner(predictions, ages, net_worths)
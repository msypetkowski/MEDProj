"""
Author: Michał Sypetkowski

Functions for loading various datasets into pandas dataframes.
"""

import pandas as pd


def load_iris():
    return pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
    # return pd.read_csv('iris.csv')


def load_adult():
    return pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
    # return pd.read_csv('adult.data',
        names=[ "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
            "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
            "Hours per week", "Country", "Target"],
            sep=r'\s*,\s*',
            engine='python',
            na_values="?"
    )


def load_ctg():
    ret = pd.read_csv('http://staff.ii.pw.edu.pl/~gprotazi/dydaktyka/dane/cardioto_all_corr.csv')
    # ret = pd.read_csv('cardioto_all_corr.csv')
    ret = ret.drop('Unnamed: 0', 1)
    return ret


def load_cars():
    return pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data',
    # return pd.read_csv('car.data')
            names=[ "class", "buying", "maint", "doors", "persons", "lug_boot", "safety" ])

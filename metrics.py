"""
Author: Michał Sypetkowski

Additional clustering metrics.
"""

import pandas as pd
import numpy as np


def purity(labels_true, labels_pred):
    """ Description: https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
    Each cluster is assigned to the class which is most frequent in the cluster,
    and then the accuracy of this assignment is measured by counting
    the number of correctly assigned documents and dividing by N.
    """
    tab = pd.crosstab(labels_true, labels_pred)
    mapping = np.argmax(tab.values, axis=0)
    mapping = tab.index[mapping]
    unique = np.unique(labels_pred)
    unique.sort()
    unique_mapping = np.zeros(unique.max() + 1, dtype=unique.dtype)
    for i, v in enumerate(unique):
        unique_mapping[v] = i
    labels_pred = unique_mapping[labels_pred]
    predictions = mapping[labels_pred]
    correct = predictions == labels_true
    return np.mean(correct)

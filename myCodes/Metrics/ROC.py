"""ROC Curve

AUC-ROC (Area Under Curve - Receiver Operating Characteristics) curve

https://stackabuse.com/understanding-roc-curves-with-python/
https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# roc curve and auc score
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


def train(model):
    # Fit a model on the train data.
    model = model()
    model.fit(trainX, trainy)

    # Predict probabilities for the test data.
    probs = model.predict_proba(testX)

    # Keep Probabilities of the positive class only.
    probs = probs[:, 1]

    # Compute the AUC Score.
    auc = roc_auc_score(testy, probs)
    label = '{} - AUC: {:.2f}'.format(model.__class__.__name__, auc)
    print(label)

    # Get the ROC Curve.
    fpr, tpr, thresholds = roc_curve(testy, probs)
    return fpr, tpr, thresholds, label


if __name__ == "__main__":
    # Generate sample data
    data_X, class_label = make_classification(
        n_samples=1000, n_classes=2, weights=[1, 1], random_state=1)

    # Split the data into train and test sub-datasets.
    trainX, testX, trainy, testy = train_test_split(
        data_X, class_label, test_size=0.3, random_state=1)

    fpr_1, tpr_1, thresholds, label_1 = train(KNeighborsClassifier)
    fpr_2, tpr_2, thresholds, label_2 = train(RandomForestClassifier)

    # Plot ROC Curve using our defined function
    fig, ax = plt.subplots(1, 1)

    fig.suptitle('Receiver Operating Characteristic (ROC) Curve', fontsize=10)
    ax.set_axisbelow(True)
    ax.plot([0, 1], [0, 1], color='k', linestyle='--')

    ax.plot(fpr_1, tpr_1, label=label_1)
    ax.plot(fpr_2, tpr_2, label=label_2)
    

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    ax.grid()

    plt.show()

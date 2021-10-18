import numpy as np
from joblib import dump, load
import warnings
import matplotlib.pyplot as plt
import mne
from time import time
from mne import Epochs
from mne.datasets import eegbci
from mne.io.edf import read_raw_edf
from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from mne.decoding import CSP
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

def main_train():
    scores = []
    epochs_data = epochs.get_data()
    epochs_data_train = epochs_train.get_data()
    cv = ShuffleSplit(10, test_size=0.2, random_state=45)
    cv_split = cv.split(epochs_data_train)

    # Assemble a classifier
    lda = LinearDiscriminantAnalysis()
    csp = PCA(n_components=3, reg=None, log=True, norm_trace=False)

    # Use scikit-learn Pipeline with cross_val_score function
    clf = Pipeline([('PCA', csp), ('LDA', lda)])
    scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1, scoring='f1')
    # Printing the results
    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1. - class_balance)
    print("Classification f1: %f / Chance level: %f" % (np.mean(scores), class_balance))

    for train_i, test_i in cv_split:
        x_test, y_test = epochs_data_train[train_i], labels[train_i]
        break
    clf.fit(x_test, y_test)
    dump(clf, 'clf.joblibs')
    clf2 = load('clf.joblibs')
    res = clf2.predict(x_test)

    round(clf.score(x_test, y_test), 2)

if __name__ == '__main__':
    main_train()

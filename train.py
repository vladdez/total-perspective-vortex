import numpy as np
import argparse
import mne
from mne.decoding import Vectorizer
from joblib import dump, load
from preprocessing import main_preproc
from mne.decoding import LinearModel
from sklearn.preprocessing import StandardScaler
import sys
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from mne.decoding import CSP
from reductors import my_CSP, my_PCA, my_SPoC
from classifiers import my_LinearDiscriminantAnalysis

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='eegbci')
    parser.add_argument('--testee', default=1)
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--clear', action="store_false")
    parser.add_argument('--algo', default='CSP')
    parser.add_argument('--class', default='LDA')
    args = parser.parse_args()

    return args.__dict__

def main_train():
    args = parse_args()
    testee = int(args['testee'])
    if testee > 109:
        testee = 109
        print('Warning! There is no testee number higher 109. It was set to 109.')
    if testee < 1:
        testee = 1
        print('Warning! There is no testee number lower 1. It was set to 1.')
    learned = []
    predicted = []
    lda = my_LinearDiscriminantAnalysis()
    classif = ('LDA', lda)
    if args['data'] == 'eegbci':
        conditions = ['imagery feet-hand', 'execution feet-hand', 'imagery left-right', 'execution left-right']
        # Assemble a classifier
        csp = my_CSP(n_components=3)
        pca = my_PCA(n_components=2)
        spoc = my_SPoC(n_components=3)

        #for t in conditions:
        for t in ['imagery feet-hand']:
            epochs = main_preproc(data_name=args['data'], testee=testee, task=t, viz=args['viz'], clear=False)
            epochs_train = epochs.copy().crop(tmin=0., tmax=2.)
            labels = epochs.events[:, -1] - 2
            # epochs_data = epochs.get_data()
            epochs_data_train = epochs_train.get_data()
            cv = ShuffleSplit(10, test_size=0.2, random_state=45)
            cv_split = cv.split(epochs_data_train)

            # Use scikit-learn Pipeline with cross_val_score function
            if args['algo'] == "CSP":
                clf = Pipeline([('CSP', csp), classif])
            elif args['algo'] == "SPOC":
                clf = Pipeline([('SPoC', spoc), classif])
            elif args['algo'] == "PCA":
                sh = epochs_data_train.shape
                epochs_data_train = epochs_data_train.reshape(sh[0], sh[1] * sh[2])
                clf = Pipeline([('PCA', pca), classif])
            else:
                sys.exit('Specify correct algo')
            scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1, scoring='f1')
            for train_i, test_i in cv_split:
                x_test, y_test = epochs_data_train[train_i], labels[train_i]
                break
            clf.fit(x_test, y_test)
            res = clf.predict(x_test)

            learned.append(np.mean(scores))
            predicted.append(clf.score(x_test, y_test))
            if args['verbose'] == True:
                print(f"... The predictions of {t} runs for participant {testee}: {res}")

        print("Mean training f1:", round(np.mean(learned), 2))
        print('Mean prediction f1', round(np.mean(predicted), 2))
    elif args['data'] == 'sample':
        conditions = ['auditory/left', 'visual/left']
        epochs = main_preproc(data_name=args['data'], testee=testee, task=None, viz=args['viz'], clear=False)
        X = epochs.get_data()
        labels = epochs.events[:, -1]
        csp = my_CSP(n_components=3)

        clf = Pipeline([('CSP', csp), ('LDA', lda)])
        y = epochs.events[:, 2]
        cv = ShuffleSplit(10, test_size=0.2, random_state=45)
        cv_split = cv.split(X)
        scores = cross_val_score(clf, X, labels, cv=cv, n_jobs=1, scoring='f1')

        for train_i, test_i in cv_split:
            x_test, y_test = X, y
            break
        clf.fit(x_test, y_test)
        res = clf.predict(x_test)
        print(f"... The predictions of runs:", res)
        print("Mean training f1:", round(np.mean(scores), 2))
        print('Mean prediction f1', round(clf.score(x_test, y_test), 2))



if __name__ == '__main__':
    main_train()

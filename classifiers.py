"""
Linear Discriminant Analysis and Quadratic Discriminant Analysis
"""

# Authors: Clemens Brunner
#          Martin Billinger
#          Matthieu Perrot
#          Mathieu Blondel

# License: BSD 3-Clause

import warnings
import numpy as np
from scipy import linalg
from scipy.special import expit

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.covariance import ledoit_wolf, empirical_covariance, shrunk_covariance
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import softmax
from sklearn.preprocessing import StandardScaler

def _class_means(X, y):
    classes, y = np.unique(y, return_inverse=True)
    cnt = np.bincount(y)
    means = np.zeros(shape=(len(classes), X.shape[1]))
    np.add.at(means, y, X)
    means /= cnt[:, None]
    return means

class my_LinearDiscriminantAnalysis(LinearClassifierMixin,
                                 TransformerMixin,
                                 BaseEstimator):
    def __init__(self):
        self.priors = None
        self.tol = 1e-4

    def _solve_svd(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        self.means_ = _class_means(X, y)
        Xc = []
        for idx, group in enumerate(self.classes_):
            Xg = X[y == group, :]
            Xc.append(Xg - self.means_[idx])

        self.xbar_ = np.dot(self.priors_, self.means_)

        Xc = np.concatenate(Xc, axis=0)

        # 1) within (univariate) scaling by with classes std-dev
        std = Xc.std(axis=0)
        # avoid division by zero in normalization
        std[std == 0] = 1.
        fac = 1. / (n_samples - n_classes)

        # 2) Within variance scaling
        X = np.sqrt(fac) * (Xc / std)
        # SVD of centered (within)scaled data
        U, S, Vt = linalg.svd(X, full_matrices=False)

        rank = np.sum(S > self.tol)
        # Scaling of within covariance is: V' 1/S
        scalings = (Vt[:rank] / std).T / S[:rank]

        # 3) Between variance scaling
        # Scale weighted centers
        X = np.dot(((np.sqrt((n_samples * self.priors_) * fac)) *
                    (self.means_ - self.xbar_).T).T, scalings)
        # Centers are living in a space with n_classes-1 dim (maximum)
        # Use SVD to find projection in the space spanned by the
        # (n_classes) centers
        _, S, Vt = linalg.svd(X, full_matrices=0)

        self.explained_variance_ratio_ = (S**2 / np.sum(
            S**2))[:self._max_components]
        rank = np.sum(S > self.tol * S[0])
        self.scalings_ = np.dot(scalings, Vt.T[:, :rank])
        coef = np.dot(self.means_ - self.xbar_, self.scalings_)
        self.intercept_ = (-0.5 * np.sum(coef ** 2, axis=1) +
                           np.log(self.priors_))
        self.coef_ = np.dot(coef, self.scalings_.T)
        self.intercept_ -= np.dot(self.xbar_, self.coef_.T)

    def fit(self, X, y):
        X, y = self._validate_data(X, y, ensure_min_samples=2, estimator=self,
                                   dtype=[np.float64, np.float32])
        self.classes_ = unique_labels(y)
        n_samples, _ = X.shape
        n_classes = len(self.classes_)

        if n_samples == n_classes:
            raise ValueError("The number of samples must be more "
                             "than the number of classes.")

        _, y_t = np.unique(y, return_inverse=True)  # non-negative ints
        self.priors_ = np.bincount(y_t) / float(len(y))

        if (self.priors_ < 0).any():
            raise ValueError("priors must be non-negative")
        if not np.isclose(self.priors_.sum(), 1.0):
            warnings.warn("The priors do not sum to 1. Renormalizing",
                          UserWarning)
            self.priors_ = self.priors_ / self.priors_.sum()

        # Maximum number of components no matter what n_components is
        # specified:
        max_components = min(len(self.classes_) - 1, X.shape[1])

        self._max_components = max_components

        self._solve_svd(X, y)

        if self.classes_.size == 2:  # treat binary case as a special case
            self.coef_ = np.array(self.coef_[1, :] - self.coef_[0, :], ndmin=2,
                                  dtype=X.dtype)
            self.intercept_ = np.array(self.intercept_[1] - self.intercept_[0],
                                       ndmin=1, dtype=X.dtype)
        return self

    def transform(self, X):
        check_is_fitted(self)

        X = check_array(X)
        X_new = np.dot(X - self.xbar_, self.scalings_)

        return X_new[:, :self._max_components]




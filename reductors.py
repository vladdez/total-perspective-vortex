import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin
from mne.fixes import pinv
from mne.utils import _check_option, _validate_type, check_random_state
from mne.cov import _regularized_covariance
from sklearn.base import BaseEstimator
from sklearn.utils.extmath import randomized_svd
from sklearn.utils.validation import check_is_fitted


# https://github.com/mne-tools/mne-python/blob/main/mne/decoding/csp.py
class my_CSP(TransformerMixin, BaseEstimator):
    def __init__(self, n_components=4, reg=None, cov_est='concat',
                 transform_into='average_power', norm_trace=False,
                 cov_method_params=None, rank=None):
        # Init default CSP
        if not isinstance(n_components, int):
            raise ValueError('n_components must be an integer.')
        self.n_components = n_components
        self.rank = rank
        self.reg = reg

        # Init default cov_est
        if not (cov_est == "concat" or cov_est == "epoch"):
            raise ValueError("unknown covariance estimation method")
        self.cov_est = cov_est

        # Init default transform_into
        self.transform_into = _check_option('transform_into', transform_into,
                                            ['average_power', 'csp_space'])

        _validate_type(norm_trace, bool, 'norm_trace')
        self.norm_trace = norm_trace
        self.cov_method_params = cov_method_params

    def _check_Xy(self, X, y=None):
        """Check input data."""
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be of type ndarray (got %s)."
                             % type(X))
        if y is not None:
            if len(X) != len(y) or len(y) < 1:
                raise ValueError('X and y must have the same length.')
        if X.ndim < 3:
            raise ValueError('X must have at least 3 dimensions.')

    def fit(self, X, y):
        self._check_Xy(X, y)
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2.")

        covs, sample_weights = self._compute_covariance_matrices(X, y)
        eigen_vectors, eigen_values = self._decompose_covs(covs,
                                                           sample_weights)
        ix = np.argsort(np.abs(eigen_values - 0.5))[::-1]

        eigen_vectors = eigen_vectors[:, ix]

        self.filters_ = eigen_vectors.T
        self.patterns_ = pinv(eigen_vectors)

        pick_filters = self.filters_[:self.n_components]
        X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])

        # compute features (mean power)
        X = (X ** 2).mean(axis=2)

        # To standardize features
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

        return self

    def transform(self, X):
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be of type ndarray (got %s)." % type(X))
        if self.filters_ is None:
            raise RuntimeError('No filters available. Please first fit CSP '
                               'decomposition.')

        pick_filters = self.filters_[:self.n_components]
        X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])

        # compute features (mean band power)
        if self.transform_into == 'average_power':
            X = (X ** 2).mean(axis=2)
            X = np.log(X)
        return X

    def _compute_covariance_matrices(self, X, y):
        _, n_channels, _ = X.shape
        cov_estimator = self._concat_cov

        covs = []
        sample_weights = []
        for this_class in self._classes:
            cov, weight = cov_estimator(X[y == this_class])

            if self.norm_trace:
                cov /= np.trace(cov)

            covs.append(cov)
            sample_weights.append(weight)

        return np.stack(covs), np.array(sample_weights)

    def _concat_cov(self, x_class):
        """Concatenate epochs before computing the covariance."""
        _, n_channels, _ = x_class.shape

        x_class = np.transpose(x_class, [1, 0, 2])
        x_class = x_class.reshape(n_channels, -1)
        cov = _regularized_covariance(
            x_class, reg=self.reg, method_params=self.cov_method_params,
            rank=self.rank)
        weight = x_class.shape[0]
        return cov, weight

    def _decompose_covs(self, covs, sample_weights):
        n_classes = len(covs)
        if n_classes == 2:
            eigen_values, eigen_vectors = linalg.eigh(covs[0], covs.sum(0))
        else:
            raise ValueError("Only 2 n_classes are allowed")
        return eigen_vectors, eigen_values

class my_SPoC(my_CSP):
    """
    #Source Power Comodulation (SPoC) allows to
    extract spatial filters and
    patterns by using a target (continuous) variable in the decomposition
    process in order to give preference to components whose power correlates
    with the target variable.
    SPoC can be seen as an extension of the CSP driven by a continuous
    variable rather than a discrete variable. Typical applications include
    extraction of motor patterns using EMG power or audio patterns using sound
    envelope.
    """

    def __init__(self, n_components=4, reg=None,
                 transform_into='average_power', cov_method_params=None,
                 rank=None):
        """Init of SPoC."""
        super(my_SPoC, self).__init__(n_components=n_components, reg=reg,
                                   cov_est="epoch", norm_trace=False,
                                   transform_into=transform_into, rank=rank,
                                   cov_method_params=cov_method_params)
        # Covariance estimation have to be done on the single epoch level,
        # unlike CSP where covariance estimation can also be achieved through
        # concatenation of all epochs from the same class.
        delattr(self, 'cov_est')
        delattr(self, 'norm_trace')

    def fit(self, X, y):
        # Estimate the SPoC decomposition on epochs.
        from scipy import linalg
        self._check_Xy(X, y)

        if len(np.unique(y)) < 2:
            raise ValueError("y must have at least two distinct values.")

        # The following code is directly copied from pyRiemann

        # Normalize target variable
        target = y.astype(np.float64)
        target -= target.mean()
        target /= target.std()

        n_epochs, n_channels = X.shape[:2]

        # Estimate single trial covariance
        covs = np.empty((n_epochs, n_channels, n_channels))
        for ii, epoch in enumerate(X):
            covs[ii] = _regularized_covariance(
                epoch, reg=self.reg, method_params=self.cov_method_params,
                rank=self.rank)

        C = covs.mean(0)
        Cz = np.mean(covs * target[:, np.newaxis, np.newaxis], axis=0)

        # solve eigenvalue decomposition
        evals, evecs = linalg.eigh(Cz, C)
        evals = evals.real
        evecs = evecs.real
        # sort vectors
        ix = np.argsort(np.abs(evals))[::-1]

        # sort eigenvectors
        evecs = evecs[:, ix].T

        # spatial patterns
        self.patterns_ = linalg.pinv(evecs).T  # n_channels x n_channels
        self.filters_ = evecs  # n_channels x n_channels

        pick_filters = self.filters_[:self.n_components]
        X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])

        # compute features (mean band power)
        X = (X ** 2).mean(axis=-1)

        # To standardize features
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        return self

    def transform(self, X):
        # Estimate epochs sources given the SPoC filters.
        return super(my_SPoC, self).transform(X)




class _BasePCA(TransformerMixin, BaseEstimator):
    def transform(self, X):
        check_is_fitted(self)
        X = self._validate_data(X, dtype=[np.float64, np.float32], reset=False)
        if self.mean_ is not None:
            X = X - self.mean_
        X_transformed = np.dot(X, self.components_.T)
        return X_transformed


class my_PCA(_BasePCA):
    def __init__(self, n_components=None, *, copy=True, svd_solver="auto", random_state=None):
        self.n_components = n_components
        self.copy = copy
        self.svd_solver = svd_solver
        self.random_state = random_state

    def fit(self, X, y=None):
        self._fit(X)
        return self

    def fit_transform(self, X, y=None):
        U, S, Vt = self._fit(X)
        U = U[:, : self.n_components_]
        U *= S[: self.n_components_]
        return U

    def _fit(self, X):

        X = self._validate_data(X, dtype=[np.float64, np.float32], ensure_2d=True, copy=self.copy)
        # Handle n_components==None
        if self.n_components is None:
            n_components = min(X.shape)
        else:
            n_components = self.n_components

        # Handle svd_solver
        self._fit_svd_solver = self.svd_solver
        if self._fit_svd_solver == "auto":
            # Small problem or n_components == 'mle', just call full PCA
            if max(X.shape) <= 500 or n_components == "mle":
                self._fit_svd_solver = "full"
            elif n_components >= 1 and n_components < 0.8 * min(X.shape):
                self._fit_svd_solver = "randomized"
            # This is also the case of n_components in (0,1)
            else:
                self._fit_svd_solver = "full"

        # Call different fits for either full or truncated SVD
        if self._fit_svd_solver == "full":
            return self._fit_full(X, n_components)
        elif self._fit_svd_solver in ["randomized"]:
            return self._fit_truncated(X, n_components)
        else:
            raise ValueError(
                "Unrecognized svd_solver='{0}'".format(self._fit_svd_solver)
            )

    def _fit_truncated(self, X, n_components):
        n_samples, n_features = X.shape
        random_state = check_random_state(self.random_state)

        # Center data
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_
        U, S, Vt = randomized_svd(
            X, n_components=n_components,n_iter="auto",
            flip_sign=True, random_state=random_state,
        )

        self.n_samples_, self.n_features_ = n_samples, n_features
        self.components_ = Vt
        self.n_components_ = n_components

        # Get variance explained by singular values
        self.explained_variance_ = (S ** 2) / (n_samples - 1)
        self.singular_values_ = S.copy()  # Store the singular values.

        return U, S, Vt
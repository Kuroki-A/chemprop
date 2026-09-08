import warnings
from typing import Any, List, Optional

import numpy as np


class StandardScaler:
    """A :class:`StandardScaler` normalizes the features of a dataset.

    When it is fit on a dataset, the :class:`StandardScaler` learns the mean and standard deviation across the 0th axis.
    When transforming a dataset, the :class:`StandardScaler` subtracts the means and divides by the standard deviations.
    """

    def __init__(self, means: np.ndarray = None, stds: np.ndarray = None, replace_nan_token: Any = None):
        """
        :param means: An optional 1D numpy array of precomputed means.
        :param stds: An optional 1D numpy array of precomputed standard deviations.
        :param replace_nan_token: A token to use to replace NaN entries in the features.
        """
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    @staticmethod
    def _as_numeric_matrix(X, operation: str) -> np.ndarray:
        """Converts scaler input into a rectangular 2-D finite-or-NaN matrix."""
        try:
            array = np.asarray(X, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f'StandardScaler.{operation} requires a rectangular 2-D numeric array.'
            ) from error
        if array.ndim != 2 or array.shape[1] == 0:
            raise ValueError(
                f'StandardScaler.{operation} requires a non-empty feature axis; '
                f'got shape {array.shape}.'
            )
        if np.any(np.isinf(array)):
            raise ValueError(f'StandardScaler.{operation} input contains an infinite value.')
        return array

    def _validated_parameters(self) -> tuple:
        """Returns numeric fitted parameters after validating checkpoint state."""
        if self.means is None or self.stds is None:
            raise ValueError('StandardScaler must be fitted before transformation.')
        try:
            means = np.asarray(self.means, dtype=float)
            stds = np.asarray(self.stds, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError('StandardScaler parameters must be numeric arrays.') from error
        if means.ndim != 1 or stds.ndim != 1 or means.shape != stds.shape or means.size == 0:
            raise ValueError(
                'StandardScaler means and standard deviations must be non-empty '
                f'1-D arrays with the same shape; got {means.shape} and {stds.shape}.'
            )
        if not np.all(np.isfinite(means)):
            raise ValueError('StandardScaler means must contain only finite values.')
        if not np.all(np.isfinite(stds)) or np.any(stds <= 0):
            raise ValueError('StandardScaler standard deviations must be finite and positive.')
        self.means, self.stds = means, stds
        return means, stds

    def fit(self, X: List[List[Optional[float]]]) -> 'StandardScaler':
        """
        Learns means and standard deviations across the 0th axis of the data :code:`X`.

        :param X: A list of lists of floats (or None).
        :return: The fitted :class:`StandardScaler` (self).
        """
        X = self._as_numeric_matrix(X, 'fit')
        if X.shape[0] == 0:
            raise ValueError('StandardScaler.fit requires at least one sample.')
        # Entirely missing columns are supported and map to mean=0/std=1, but
        # NumPy otherwise emits two RuntimeWarnings while computing them.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            self.means = np.nanmean(X, axis=0)
            self.stds = np.nanstd(X, axis=0)
        self.means = np.where(np.isnan(self.means), np.zeros(self.means.shape), self.means)
        self.stds = np.where(np.isnan(self.stds), np.ones(self.stds.shape), self.stds)
        self.stds = np.where(self.stds == 0, np.ones(self.stds.shape), self.stds)

        return self

    def transform(self, X: List[List[Optional[float]]]) -> np.ndarray:
        """
        Transforms the data by subtracting the means and dividing by the standard deviations.

        :param X: A list of lists of floats (or None).
        :return: The transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = self._as_numeric_matrix(X, 'transform')
        means, stds = self._validated_parameters()
        if X.shape[1] != means.size:
            raise ValueError(
                f'StandardScaler.transform feature width {X.shape[1]} does not '
                f'match fitted width {means.size}.'
            )
        transformed_with_nan = (X - means) / stds
        transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none

    def inverse_transform(self, X: List[List[Optional[float]]]) -> np.ndarray:
        """
        Performs the inverse transformation by multiplying by the standard deviations and adding the means.

        :param X: A list of lists of floats.
        :return: The inverse transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = self._as_numeric_matrix(X, 'inverse_transform')
        means, stds = self._validated_parameters()
        if X.shape[1] != means.size:
            raise ValueError(
                f'StandardScaler.inverse_transform feature width {X.shape[1]} '
                f'does not match fitted width {means.size}.'
            )
        transformed_with_nan = X * stds + means
        transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none

class AtomBondScaler(StandardScaler):
    """A :class:`AtomBondScaler` normalizes the features of a dataset.

    When it is fit on a dataset, the :class:`AtomBondScaler` learns the mean and standard deviation across the 0th axis.
    When transforming a dataset, the :class:`AtomBondScaler` subtracts the means and divides by the standard deviations.
    """

    def __init__(self, means: np.ndarray = None, stds: np.ndarray = None, replace_nan_token: Any = None, n_atom_targets = None, n_bond_targets = None):
        super().__init__(means, stds, replace_nan_token)
        self.n_atom_targets = n_atom_targets
        self.n_bond_targets = n_bond_targets

    def fit(self, X: List[List[Optional[float]]]) -> 'AtomBondScaler':
        scalers = []
        for i in range(self.n_atom_targets):
            scaler = StandardScaler().fit(X[i])
            scalers.append(scaler)
        for i in range(self.n_bond_targets):
            scaler = StandardScaler().fit(X[i+self.n_atom_targets])
            scalers.append(scaler)

        self.means = np.array([s.means for s in scalers])
        self.stds = np.array([s.stds for s in scalers])

        return self

    def transform(self, X: List[List[Optional[float]]]) -> List[np.ndarray]:
        """
        Transforms the data by subtracting the means and dividing by the standard deviations.

        :param X: A list of lists of floats (or None).
        :return: The transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        transformed_results = []
        for i in range(self.n_atom_targets):
            Xi = np.array(X[i]).astype(float)
            transformed_with_nan = (Xi - self.means[i]) / self.stds[i]
            transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)
            transformed_results.append(transformed_with_none.tolist())
        for i in range(self.n_bond_targets):
            Xi = np.array(X[i+self.n_atom_targets]).astype(float)
            transformed_with_nan = (Xi - self.means[i+self.n_atom_targets]) / self.stds[i+self.n_atom_targets]
            transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)
            transformed_results.append(transformed_with_none.tolist())

        return transformed_results

    def inverse_transform(self, X: List[List[Optional[float]]]) -> List[np.ndarray]:
        """
        Performs the inverse transformation by multiplying by the standard deviations and adding the means.

        :param X: A list of lists of floats.
        :return: The inverse transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        transformed_results = []
        for i in range(self.n_atom_targets):
            Xi = np.array(X[i]).astype(float)
            transformed_with_nan = Xi * self.stds[i] + self.means[i]
            transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)
            transformed_results.append(transformed_with_none.tolist())
        for i in range(self.n_bond_targets):
            Xi = np.array(X[i+self.n_atom_targets]).astype(float)
            transformed_with_nan = Xi * self.stds[i+self.n_atom_targets] + self.means[i+self.n_atom_targets]
            transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)
            transformed_results.append(transformed_with_none.tolist())

        return transformed_results

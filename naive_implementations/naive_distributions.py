from typing import Tuple, Union

import numpy as np

from distributions import BaseDistribution


class NaiveGaussian(BaseDistribution):
    def __init__(self, mu: np.ndarray, covariance: np.ndarray):
        self.n = len(mu)
        self.mu = mu.reshape(
            -1,
        )
        self.covariance = covariance
        super().__init__()

    @property
    def inv_covariance(self):
        return np.linalg.inv(self.covariance)

    def log_p_tilda(self, x: np.ndarray) -> float:
        return -0.5 * np.dot(
            np.dot(np.subtract(x, self.mu).T, self.inv_covariance),
            np.subtract(x, self.mu),
        )

    @property
    def log_z(self) -> float:
        return np.multiply(self.n / 2, np.log(2 * np.pi)) + 0.5 * np.log(
            np.linalg.det(self.covariance)
        )

    def sample(self, size: Union[int, Tuple[int]]) -> np.ndarray:
        return np.random.multivariate_normal(self.mu.flatten(), self.covariance, size)

    def dlog_p_dx(self, x: np.ndarray) -> np.ndarray:
        return -np.dot(self.inv_covariance, np.subtract(x, self.mu))

    def dlog_p_dx_dx(self, x: np.ndarray) -> np.ndarray:
        return -self.inv_covariance

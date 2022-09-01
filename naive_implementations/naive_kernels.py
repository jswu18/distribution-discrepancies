import numpy as np

from distributions import BaseDistribution
from kernels import BaseKernel, _l2_squared


class NaiveInverseMultiQuadraticKernel(BaseKernel):
    def __init__(self, c: float, beta: float):
        assert c > 0, f"c > 0, {c=}"
        self.c = c

        assert (-1 < beta) & (beta < 0), f"beta must be in (-1, 0), {beta=}"
        self.beta = beta

    def _k_pre_exponent(self, x: np.ndarray, y: np.ndarray) -> float:
        return self.c**2 + _l2_squared(x, y)

    def k(self, x: np.ndarray, y: np.ndarray) -> float:
        return self._k_pre_exponent(x, y) ** self.beta

    def dk_dx(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 2 * self.beta * (self._k_pre_exponent(x, y) ** (self.beta - 1)) * (x - y)

    def dk_dy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 2 * self.beta * (self._k_pre_exponent(x, y) ** (self.beta - 1)) * (y - x)

    def dk_dx_dy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        k_pre_exponent = self._k_pre_exponent(x, y)
        return (
            2
            * self.beta
            * (
                2
                * (self.beta - 1)
                * (k_pre_exponent ** (self.beta - 2))
                * np.dot((y - x).reshape(-1, 1), (x - y).reshape(1, -1))
                - k_pre_exponent ** (self.beta - 1) * np.eye(len(x))
            )
        )


class NaiveSteinKernel:
    def __init__(self, distribution: BaseDistribution, kernel: BaseKernel):
        self.distribution = distribution
        self.kernel = kernel

    def k(self, x: np.ndarray, y: np.ndarray) -> float:
        d = len(x)
        a1 = self.kernel.k(x, y) * np.dot(
            self.distribution.dlog_p_dx(x).T, self.distribution.dlog_p_dx(y)
        )
        a2 = np.dot(self.distribution.dlog_p_dx(y).T, self.kernel.dk_dx(x, y))
        a3 = np.dot(self.distribution.dlog_p_dx(x).T, self.kernel.dk_dy(x, y))
        a4 = np.trace(self.kernel.dk_dx_dy(x, y).reshape(d, d))
        return a1 + a2 + a3 + a4

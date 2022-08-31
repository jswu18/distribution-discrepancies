from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import jax.numpy as jnp
import numpy as np
from jax import jacfwd


class BaseDistribution(ABC):
    """
    Base Distribution class

    It is assumed that distributions are of the form:
        p(x; theta) = p̃(x)/z(theta)
    where theta are the parameters of the distribution.
    """

    @abstractmethod
    def log_p_tilda(self, x: np.ndarray) -> float:
        """
        Computes log(p̃(x)).

        :param x: ndarray of shape (n_dimensions, )
        :return: float being the logarithm of p̃(x)
        """
        raise NotImplemented

    @property
    @abstractmethod
    def log_z(self) -> float:
        """
        Computes log(z(theta)).

        :return: float being the logarithm of z(theta)
        """
        raise NotImplemented

    @abstractmethod
    def sample(self, size: Union[int, Tuple[int]]) -> np.ndarray:
        """
        Samples the distribution.

        :param size: either an integer or shape indicating the number of samples desired
        :return: ndarray of samples from the distribution
        """
        raise NotImplemented

    @abstractmethod
    def dlog_p_dx(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the gradient vector of log p(x) where the ith
        element is dlog(p)/dx_i evaluated at x.

        :param x: ndarray of shape (n_dimensions, )
        :return: ndarray of shape (n_dimensions, ), the gradient vector
        """
        raise NotImplemented

    @abstractmethod
    def dlog_p_tilda_dx(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the gradient vector of log p̃(x) where the ith
        element is dlog(p̃)/dx_i evaluated at x.

        :param x: ndarray of shape (n_dimensions, )
        :return: ndarray of shape (n_dimensions, ), the gradient vector
        """
        raise NotImplemented

    @abstractmethod
    def dlog_p_tilda_dx_dx(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the Hessian matrix of log p̃(x) where the element (i, j) is the
        derivative of log(p̃) with respect to x_i and x_j, d^2log(p̃)/(dx_i, d_x_j) evaluated at x.

        :param x: ndarray of shape (n_dimensions, )
        :return: ndarray of shape (n_dimensions, n_dimensions), the Hessian matrix
        """
        raise NotImplemented

    @property
    def z(self) -> float:
        """
        Computes z(theta).

        :return: float being z(theta)
        """
        return jnp.exp(self.log_z)

    def p_tilda(self, x: np.ndarray) -> float:
        """
        Computes p̃(x).

        :param x: ndarray of shape (n_dimensions, )
        :return: float being p̃(x)
        """
        return jnp.exp(self.log_p_tilda(x))

    def p(self, x: np.ndarray) -> float:
        """
        Computes p(x).

        :param x: ndarray of shape (n_dimensions, )
        :return: float being p(x)
        """
        return jnp.divide(1, self.z) * self.p_tilda(x)

    def log_p(self, x: np.ndarray) -> float:
        """
        Computes log p(x).

        :param x: ndarray of shape (n_dimensions, )
        :return: float being the logarithm of p(x)
        """
        return jnp.subtract(jnp.log(1), self.log_z) * self.log_p_tilda(x)

    def score(self, x: np.ndarray) -> np.ndarray:
        """
        Alias for dlog_p_tilda_dx

        :param x: ndarray of shape (n_dimensions, )
        :return: ndarray of shape (n_dimensions, ), the score vector
        """
        return self.dlog_p_tilda_dx(x)

    def d_score_dx(self, x: np.ndarray) -> np.ndarray:
        """
        Alias for dlog_p_tilda_dx_dx

        :param x: ndarray of shape (n_dimensions, )
        :return: ndarray of shape (n_dimensions, n_dimensions), the Hessian matrix
        """
        return self.dlog_p_tilda_dx_dx(x)


class BaseAutoDiffDistribution(BaseDistribution, ABC):
    """
    Base Distribution class which implements the derivatives using auto differentiation
    """

    def dlog_p_dx(self, x: np.ndarray) -> np.ndarray:
        return jacfwd(self.log_p, argnums=0)(x)

    def dlog_p_tilda_dx(self, x: np.ndarray) -> np.ndarray:
        return jacfwd(self.log_p_tilda, argnums=0)(x)

    def dlog_p_tilda_dx_dx(self, x: np.ndarray) -> np.ndarray:
        return jnp.squeeze(jacfwd(jacfwd(self.log_p_tilda))(x))


class Gaussian(BaseAutoDiffDistribution):
    """
    The Gaussian Distribution defined as:
        p(x) = p̃(x)/z$$
    with:
        z = 2π^(n/2) det(Sigma)^(1/2)
    and
        p̃(x) = exp(-(1/2)(x-mu)^T Sigma^(-1) (x-mu))

    where:
        n is the number of dimensionality of the distribution
    and
        mu is the mean vector of the Gaussian (n, 1)
    and
        Sigma is the covariance matrix (n, n).
    """

    def __init__(self, mu: np.ndarray, covariance: np.ndarray):
        self.n = len(mu)
        self.mu = mu.reshape(
            -1,
        )
        self.covariance = covariance
        super().__init__()

    @property
    def inv_covariance(self):
        return jnp.linalg.inv(self.covariance)

    def log_p_tilda(self, x: np.ndarray) -> float:
        return -0.5 * jnp.dot(
            jnp.dot(jnp.subtract(x, self.mu).T, self.inv_covariance),
            jnp.subtract(x, self.mu),
        )

    @property
    def log_z(self) -> float:
        return jnp.multiply(self.n / 2, jnp.log(2 * jnp.pi)) + 0.5 * jnp.log(
            jnp.linalg.det(self.covariance)
        )

    def sample(self, size: Union[int, Tuple[int]]) -> np.ndarray:
        return np.random.multivariate_normal(self.mu.flatten(), self.covariance, size)


class Mixture(BaseAutoDiffDistribution):
    """
    A Mixture Distribution being the weighted sum of n distributions:
        p(x) = sum_{i=1}^n w_i p_i(x)
    where:
        p_i is the ith distribution
    and
        w_i is the weight of the p_i
    and
        sum_{i=1}^n w_i = 1
    """

    def __init__(self, weights: List[float], distributions: List[BaseDistribution]):
        assert len(weights) == len(
            distributions
        ), f"{len(weights)=} != {len(distributions)=}"
        assert np.sum(weights) == 1, f"{np.sum(weights)=} != 1"

        self.weights = weights
        self.distributions = distributions

    def log_p_tilda(self, x: np.ndarray) -> float:
        return jnp.log(
            jnp.dot(
                jnp.array(self.weights), jnp.array([d.p(x) for d in self.distributions])
            )
        )

    @property
    def log_z(self) -> float:
        return 0

    def sample(self, size: Union[int, Tuple[int]]) -> np.ndarray:
        if isinstance(size, int):
            size = (size, 1)
        return np.vectorize(
            lambda _: np.random.choice(
                a=[float(p_i.sample(1)) for p_i in self.distributions],
                p=self.weights,
            )
        )(np.empty(size))

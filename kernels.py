import math
from abc import ABC, abstractmethod

import jax.numpy as jnp
import numpy as np
from jax import jacfwd

from distributions import BaseDistribution


class BaseKernel(ABC):
    """
    Base Kernel class
    """

    @abstractmethod
    def k(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the kernel between x and y

        :param x: ndarray of shape (n_dimensions, )
        :param y: ndarray of shape (n_dimensions, )
        :return: float being the kernel computed between x and y
        """
        raise NotImplemented

    @abstractmethod
    def dk_dx(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Computes the kernel gradient vector where element i is the partial
        derivative of k with respect to x_i, dk/dx_i

        :param x: ndarray of shape (n_dimensions, )
        :param y: ndarray of shape (n_dimensions, )
        :return: ndarray of shape (n_dimensions, ), the gradient vector
        """
        raise NotImplemented

    @abstractmethod
    def dk_dy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Computes the kernel gradient vector where element i is the partial
        derivative of k with respect to y_i, dk/dy_i

        :param x: ndarray of shape (n_dimensions, )
        :param y: ndarray of shape (n_dimensions, )
        :return: ndarray of shape (n_dimensions, ), the gradient vector
        """
        raise NotImplemented

    @abstractmethod
    def dk_dx_dy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Computes the kernel Hessian matrix where element (i, j) is the partial
        derivative of k with respect o x_i and y_j, dk/(dx_i, dy_j)

        :param x: ndarray of shape (n_dimensions, )
        :param y: ndarray of shape (n_dimensions, )
        :return: ndarray of shape (n_dimensions, n_dimensions), the Hessian matrix
        """
        raise NotImplemented


class BaseAutoDiffKernel(BaseKernel, ABC):
    """
    Base Kernel class which implements the derivatives using auto differentiation
    """

    def dk_dx(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return jacfwd(self.k, argnums=0)(x, y)

    def dk_dy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return jacfwd(self.k, argnums=1)(x, y)

    def dk_dx_dy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return jnp.squeeze(jacfwd(jacfwd(self.k, argnums=0), argnums=1)(x, y))


def _l2_squared(x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes the L2 norm, ||x-y||_2^2

    :param x: ndarray of shape (n_dimensions, )
    :param y: ndarray of shape (n_dimensions, )
    :return: the L2 norm of x-y
    """
    xx = jnp.dot(x.T, x)
    xy = jnp.dot(x.T, y)
    yy = jnp.dot(y.T, y)
    return (xx - 2 * xy + yy).reshape()


class PolynomialKernel(BaseAutoDiffKernel):
    """
    The Polynomial Kernel defined as:
        k(x, y) = (⟨x, y⟩ + 1)^p
    where p is a natural number.
    """

    def __init__(self, p: int):
        self.p = p

    def k(self, x: np.ndarray, y: np.ndarray) -> float:
        n = len(x)
        return ((1 / n) * jnp.dot(x.T, y) + 1) ** self.p


class GaussianKernel(BaseAutoDiffKernel):
    """
    The Gaussian Kernel defined as:
        (x, y) = exp(-sigma||x-y||_2^2)
    where sigma>0.
    """

    def __init__(self, sigma: float):
        self.sigma = sigma

    def k(self, x: np.ndarray, y: np.ndarray) -> float:
        return jnp.exp(-self.sigma * _l2_squared(x, y))


class LaplacianKernel(BaseAutoDiffKernel):
    """
    The Laplacian Kernel is defined as:
        k(x, y) = exp(-sigma||x-y||_1)
    where sigma>0.
    """

    def __init__(self, sigma: float):
        self.sigma = sigma

    def k(self, x: np.ndarray, y: np.ndarray) -> float:
        return jnp.exp(-self.sigma * jnp.sum(jnp.abs(x - y)).reshape())


class InverseMultiQuadraticKernel(BaseAutoDiffKernel):
    """
    The Inverse Multi Quadratic Kernel is defined as:
        k(x, y) =(c^2+||x-y||_2^2)^beta
    where c>0 and beta is in (-1, 0).
    """

    def __init__(self, c: float, beta: float):
        assert c > 0, f"c > 0, {c=}"
        self.c = c

        assert (-1 < beta) & (beta < 0), f"beta must be in (-1, 0), {beta=}"
        self.beta = beta

    def _k_pre_exponent(self, x: np.ndarray, y: np.ndarray) -> float:
        return self.c**2 + _l2_squared(x, y)

    def k(self, x: np.ndarray, y: np.ndarray) -> float:
        return self._k_pre_exponent(x, y) ** self.beta


class SteinKernel(BaseAutoDiffKernel):
    """
    The Stein Kernel, k_p(x, y), is defined as:
        k_p(x, y) = ∇x log p(x)^T ∇y log p(y)^T k(x, y)
                  + ∇y p(y)^T ∇x k(x, y)
                  + ∇x p(x)^T ∇y k(x, y)
                  + < ∇x k(x, •), ∇y k(•, y)>
    where:
        k(x, y) is a seed kernel function (i.e. the Inverse Multi Quadratic Kernel)
    and
        p(x) is a distribution
    and
        < ∇x k(x, •), ∇y k(•, y)> = sum_{i=1}^d dk_dxi_dyi(x,y)}
                                  = Trace(∇x∇y k(x, y))
    """

    def __init__(self, distribution: BaseDistribution, kernel: BaseKernel):
        self.distribution = distribution
        self.kernel = kernel

    def k(self, x: np.ndarray, y: np.ndarray) -> float:
        n = len(x)
        a1 = self.kernel.k(x, y) * jnp.dot(
            self.distribution.dlog_p_dx(x).T, self.distribution.dlog_p_dx(y)
        )
        a2 = jnp.dot(self.distribution.dlog_p_dx(y).T, self.kernel.dk_dx(x, y))
        a3 = jnp.dot(self.distribution.dlog_p_dx(x).T, self.kernel.dk_dy(x, y))
        a4 = jnp.trace(self.kernel.dk_dx_dy(x, y).reshape(n, n))
        return a1 + a2 + a3 + a4

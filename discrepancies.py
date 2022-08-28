from typing import Callable

import jax.numpy as jnp
import numpy as np
from jax import vmap

from kernels import BaseKernel, SteinKernel
from distributions import BaseDistribution


def _gram(
    kernel: Callable[[np.ndarray, np.ndarray], np.ndarray], x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """
    Computes the gram matrix having shape (n, n) where:
        g_(i, j) = kernel(x_i, y_i).
    :param kernel: the kernel function
    :param x: ndarray of shape (n_dimensions, )
    :param y: ndarray of shape (n_dimensions, )
    :return: matrix of shape (n_dimensions, n_dimensions)
    """
    return vmap(
        lambda x_i: vmap(lambda y_i: kernel(x_i, y_i))(y),
    )(x)


def _remove_diagonal(x: np.ndarray) -> np.ndarray:
    """
    Removes the diagonal elements of a matrix x.
    For a matrix of shape (n, n), this will return a matrix of shape (n, n-1).

    :param x: ndarray of shape (n_dimensions, n_dimensions)
    :return: matrix of shape (n_dimensions, n_dimensions-1)
    """
    return x[~np.eye(x.shape[0], dtype=bool)].reshape(x.shape[0], -1)


class MaximumMeanDiscrepancy:
    def __init__(self, kernel: BaseKernel):
        """
        Methods relating to the Maximum Mean Discrepancy

        :param kernel: the kernel function
        """
        self.kernel = kernel

    def compute(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the Maximum Mean Discrepancy defined as:
            MMD^2 = E[k(X,X')] - 2E[k(X,Y)] + E[k(Y,Y')]
        where:
            E is the expectation
        and
            k is the kernel function
        and
            X, X' ~ P distribution
        and
            Y, Y' ~ Q distribution.

        :param x: ndarray of shape (n_samples, n_dimensions)
        :param y: ndarray of shape (n_samples, n_dimensions)
        :return: the unbiased estimate of the maximum mean discrepancy
        """
        xx = _gram(self.kernel.k, x, x)
        xy = _gram(self.kernel.k, x, y)
        yy = _gram(self.kernel.k, y, y)
        return (
            jnp.mean(_remove_diagonal(xx))
            + jnp.mean(_remove_diagonal(yy))
            - 2 * jnp.mean(xy)
        )

    def witness_function(self, x: np.ndarray, y: np.ndarray, t: np.ndarray):
        """
        The witness function f* of the MMD between two distribution P and Q:

            f*(t) proportional <phi(t), mu_P - mu_Q> = E_P[k(X, t)] - E_Q[k(Y, t)]

        where:
            E is the expectation
        and
            k is the kernel function
        and
            X ~ P distribution
        and
            Y ~ Q distribution.

        :param x: ndarray of shape (n_samples, n_dimensions)
        :param y: ndarray of shape (n_samples, n_dimensions)
        :param t: ndarray of shape (n_samples, n_dimensions)
        :return: the estimated values of the witness function f(t)
        """
        return vmap(
            lambda t_i: jnp.mean(vmap(lambda x_i: self.kernel.k(x_i, t_i))(x))
            - jnp.mean(vmap(lambda y_i: self.kernel.k(y_i, t_i))(y))
        )(t)


class KernelSteinDiscrepancy:
    def __init__(self, stein_kernel: SteinKernel):
        """
        Methods relating to the Kernel Stein Discrepancy

        :param stein_kernel: the stein kernel function
        """
        self.stein_kernel = stein_kernel

    def compute(self, x: np.ndarray) -> float:
        """
        Computes the Kernel Stein Discrepancy defined as:
            KSD^2 = E[k_p(X,X')]
        where:
            E is the expectation
        and
            k_p is the Stein kernel function
        and
            X, X' ~ Q distribution.

        :param x: ndarray of shape (n_samples, n_dimensions)
        :return: the unbiased estimate of the kernel stein discrepancy
        """
        xx = _gram(self.stein_kernel.k, x, x)
        return jnp.mean(_remove_diagonal(xx)).reshape()


class FisherDivergence:
    def __init__(self, p: BaseDistribution):
        """
        Methods relating to the Fisher Divergence

        :param p: a probability distribution
        """
        self.p = p

    def compute(self, x: np.ndarray) -> float:
        """
        Computes the Fisher Divergence defined as:
            J = 0.5*E[||score_p(X)-score_q(X)||^2]
        where:
            E is the expectation
        and
            score_p, score_q are the score functions of the distributions p and q respectively
        and
            X ~ Q distribution.

        :param x: ndarray of shape (n_samples, n_dimensions)
        :return: the unbiased estimate of the fisher divergence
        """
        d = x.shape[1]
        return jnp.mean(
            vmap(
                lambda x_i: jnp.sum(
                    jnp.diag(self.p.d_score_dx(x_i).reshape(d, d))
                    + 0.5 * jnp.square(self.p.score(x_i))
                )
            )(x)
        )

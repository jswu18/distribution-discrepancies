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


def mmd(kernel: BaseKernel, x: np.ndarray, y: np.ndarray) -> float:
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

    :param kernel: the kernel function
    :param x: ndarray of shape (n_samples, n_dimensions)
    :param y: ndarray of shape (n_samples, n_dimensions)
    :return: the unbiased estimate of the maximum mean discrepancy
    """
    xx = _gram(kernel.k, x, x)
    xy = _gram(kernel.k, x, y)
    yy = _gram(kernel.k, y, y)
    return (
        jnp.mean(_remove_diagonal(xx))
        + jnp.mean(_remove_diagonal(yy))
        - 2 * jnp.mean(xy)
    )


def ksd(stein_kernel: SteinKernel, x: np.ndarray) -> float:
    """
    Computes the Kernel Stein Discrepancy defined as:
        KSD^2 = E[k_p(X,X')]
    where:
        E is the expectation
    and
        k_p is the Stein kernel function
    and
        X, X' ~ Q distribution.

    :param stein_kernel: the kernel function
    :param x: ndarray of shape (n_samples, n_dimensions)
    :return: the unbiased estimate of the kernel stein discrepancy
    """
    xx = _gram(stein_kernel.k, x, x)
    return jnp.mean(_remove_diagonal(xx)).reshape()


def fisher_divergence(p: BaseDistribution, x: np.ndarray) -> float:
    """
    Computes the Fisher Divergence defined as:
        J = 0.5*E[||score_p(X)-score_q(X)||^2]
    where:
        E is the expectation
    and
        score_p, score_q are the score functions of the distributions p and q respectively
    and
        X ~ Q distribution.

    :param p: the base distribution that the divergence from samples from X ~ Q will be computed
    :param x: ndarray of shape (n_samples, n_dimensions)
    :return: the unbiased estimate of the fisher divergence
    """
    d = x.shape[1]
    return jnp.mean(
        vmap(
            lambda x_i: jnp.sum(
                jnp.diag(p.d_score_dx(x_i).reshape(d, d))
                + 0.5 * jnp.square(p.score(x_i))
            )
        )(x)
    )

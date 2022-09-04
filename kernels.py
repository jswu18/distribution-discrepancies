from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import jax.numpy as jnp
import numpy as np
from jax import jacfwd, jit, tree_util

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
        raise NotImplementedError("Needs to implement k")

    @abstractmethod
    def dk_dx(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Computes the kernel gradient vector where element i is the partial
        derivative of k with respect to x_i, dk/dx_i

        :param x: ndarray of shape (n_dimensions, )
        :param y: ndarray of shape (n_dimensions, )
        :return: ndarray of shape (n_dimensions, ), the gradient vector
        """
        raise NotImplementedError("Needs to implement k")

    @abstractmethod
    def dk_dy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Computes the kernel gradient vector where element i is the partial
        derivative of k with respect to y_i, dk/dy_i

        :param x: ndarray of shape (n_dimensions, )
        :param y: ndarray of shape (n_dimensions, )
        :return: ndarray of shape (n_dimensions, ), the gradient vector
        """
        raise NotImplementedError("Needs to implement dk_dy")

    @abstractmethod
    def dk_dx_dy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Computes the kernel Hessian matrix where element (i, j) is the partial
        derivative of k with respect to x_i and y_j, dk/(dx_i, dy_j)

        :param x: ndarray of shape (n_dimensions, )
        :param y: ndarray of shape (n_dimensions, )
        :return: ndarray of shape (n_dimensions, n_dimensions), the Hessian matrix
        """
        raise NotImplementedError("Needs to implement dk_dx_dy")


class BaseAutoDiffKernel(BaseKernel, ABC):
    """
    Base Kernel class which implements the derivatives using auto differentiation
    """

    def dk_dx(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return jacfwd(self.k, argnums=0)(x, y)

    def dk_dy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return jacfwd(self.k, argnums=1)(x, y)

    def dk_dx_dy(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return jacfwd(jacfwd(self.k, argnums=0), argnums=1)(x, y)
        # return jnp.squeeze(jacfwd(jacfwd(self.k, argnums=0), argnums=1)(x, y))

    @abstractmethod
    def tree_flatten(self) -> Tuple[Tuple, Dict[str, Any]]:
        """
        To have JIT-compiled class methods by registering the type as a custom PyTree object.
        As referenced in:
        https://jax.readthedocs.io/en/latest/faq.html#strategy-3-making-customclass-a-pytree

        :return: A tuple containing dynamic and a dictionary containing static values
        """
        raise NotImplementedError("Needs to implement tree_flatten")

    @classmethod
    @abstractmethod
    def tree_unflatten(
        cls, aux_data: Dict[str, Any], children: Tuple
    ) -> BaseAutoDiffKernel:
        """
        To have JIT-compiled class methods by registering the type as a custom PyTree object.
        As referenced in:
        https://jax.readthedocs.io/en/latest/faq.html#strategy-3-making-customclass-a-pytree

        :param aux_data: tuple containing dynamic values
        :param children: dictionary containing dynamic values
        :return: Class instance
        """
        raise NotImplementedError("Needs to implement tree_unflatten")


@jit
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

    @jit
    def k(self, x: np.ndarray, y: np.ndarray) -> float:
        n = len(x)
        return ((1 / n) * jnp.dot(x.T, y) + 1) ** self.p

    def tree_flatten(self) -> Tuple[Tuple, Dict[str, Any]]:
        children = ()
        aux_data = {"p": self.p}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls, aux_data: Dict[str, Any], children: Tuple
    ) -> PolynomialKernel:
        return cls(*children, **aux_data)


class GaussianKernel(BaseAutoDiffKernel):
    """
    The Gaussian Kernel defined as:
        (x, y) = exp(-sigma||x-y||_2^2)
    where sigma>0.
    """

    def __init__(self, sigma: float):
        self.sigma = sigma

    @jit
    def k(self, x: np.ndarray, y: np.ndarray) -> float:
        return jnp.exp(-self.sigma * _l2_squared(x, y))

    def tree_flatten(self) -> Tuple[Tuple, Dict[str, Any]]:
        children = ()
        aux_data = {"sigma": self.sigma}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls, aux_data: Dict[str, Any], children: Tuple
    ) -> GaussianKernel:
        return cls(*children, **aux_data)


class LaplacianKernel(BaseAutoDiffKernel):
    """
    The Laplacian Kernel is defined as:
        k(x, y) = exp(-sigma||x-y||_1)
    where sigma>0.
    """

    def __init__(self, sigma: float):
        self.sigma = sigma

    @jit
    def k(self, x: np.ndarray, y: np.ndarray) -> float:
        return jnp.exp(-self.sigma * jnp.sum(jnp.abs(x - y)).reshape())

    def tree_flatten(self) -> Tuple[Tuple, Dict[str, Any]]:
        children = ()
        aux_data = {"sigma": self.sigma}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls, aux_data: Dict[str, Any], children: Tuple
    ) -> LaplacianKernel:
        return cls(*children, **aux_data)


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

    @jit
    def k(self, x: np.ndarray, y: np.ndarray) -> float:
        return self._k_pre_exponent(x, y) ** self.beta

    def tree_flatten(self) -> Tuple[Tuple, Dict[str, Any]]:
        children = ()
        aux_data = {"c": self.c, "beta": self.beta}
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls, aux_data: Dict[str, Any], children: Tuple
    ) -> InverseMultiQuadraticKernel:
        return cls(*children, **aux_data)


class SteinKernel(BaseAutoDiffKernel):
    """
    The Stein Kernel, k_p(x, y), is defined as:
        k_p(x, y) = ∇x log p̃(x)^T ∇y log p̃(y)^T k(x, y)
                  + ∇y log p̃(y)^T ∇x k(x, y)
                  + ∇x log p̃(x)^T ∇y k(x, y)
                  + < ∇x k(x, •), ∇y k(•, y)>
    where:
        k(x, y) is a seed kernel function (i.e. the Inverse Multi Quadratic Kernel)
    and
        p(x) is a distribution of form p̃(x)/z
    and
        < ∇x k(x, •), ∇y k(•, y)> = sum_{i=1}^d dk_dxi_dyi(x,y)}
                                  = Trace(∇x∇y k(x, y))
    """

    def __init__(self, distribution: BaseDistribution, kernel: BaseKernel):
        self.distribution = distribution
        self.kernel = kernel

    @jit
    def k(self, x: np.ndarray, y: np.ndarray) -> float:
        a1 = self.kernel.k(x, y) * jnp.dot(
            self.distribution.score(x).T, self.distribution.score(y)
        )
        a2 = jnp.dot(self.distribution.score(y).T, self.kernel.dk_dx(x, y))
        a3 = jnp.dot(self.distribution.score(x).T, self.kernel.dk_dy(x, y))
        a4 = jnp.trace(self.kernel.dk_dx_dy(x, y))
        return a1 + a2 + a3 + a4

    def tree_flatten(self) -> Tuple[Tuple, Dict[str, Any]]:
        children = ()
        aux_data = {
            "distribution": self.distribution,
            "kernel": self.kernel,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: Dict[str, Any], children: Tuple) -> SteinKernel:
        return cls(*children, **aux_data)


for KernelClass in [
    PolynomialKernel,
    GaussianKernel,
    LaplacianKernel,
    InverseMultiQuadraticKernel,
    SteinKernel,
]:
    tree_util.register_pytree_node(
        KernelClass,
        KernelClass.tree_flatten,
        KernelClass.tree_unflatten,
    )

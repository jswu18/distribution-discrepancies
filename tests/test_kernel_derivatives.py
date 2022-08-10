import numpy as np
import pytest

from kernels import InverseMultiQuadraticKernel
from naive_implementations.naive_kernels import NaiveInverseMultiQuadraticKernel


@pytest.mark.parametrize(
    "c,beta,n_dimensions",
    [
        [0.1, -0.5, 2],
        [0.2, -0.1, 3],
        [0.3, -0.2, 5],
    ],
)
def test_inverse_multi_quadratic_kernel_dk_dx(
    c: float,
    beta: float,
    n_dimensions: int,
):
    x = np.random.rand(n_dimensions, 1)
    y = np.random.rand(n_dimensions, 1)
    kernel = InverseMultiQuadraticKernel(c, beta)
    naive_kernel = NaiveInverseMultiQuadraticKernel(c, beta)
    np.testing.assert_allclose(kernel.dk_dx(x, y), naive_kernel.dk_dx(x, y), rtol=1e-4)


@pytest.mark.parametrize(
    "c,beta,n_dimensions",
    [
        [0.1, -0.5, 2],
        [0.2, -0.1, 3],
        [0.3, -0.2, 5],
    ],
)
def test_inverse_multi_quadratic_kernel_dk_dy(
    c: float,
    beta: float,
    n_dimensions: int,
):
    x = np.random.rand(n_dimensions, 1)
    y = np.random.rand(n_dimensions, 1)
    kernel = InverseMultiQuadraticKernel(c, beta)
    naive_kernel = NaiveInverseMultiQuadraticKernel(c, beta)
    np.testing.assert_allclose(kernel.dk_dy(x, y), naive_kernel.dk_dy(x, y), rtol=1e-4)


@pytest.mark.parametrize(
    "c,beta,n_dimensions",
    [
        [0.1, -0.5, 2],
        [0.2, -0.1, 3],
        [0.3, -0.2, 5],
    ],
)
def test_inverse_multi_quadratic_kernel_dk_dx_dy(
    c: float,
    beta: float,
    n_dimensions: int,
):
    x = np.random.rand(
        n_dimensions,
    )
    y = np.random.rand(
        n_dimensions,
    )
    kernel = InverseMultiQuadraticKernel(c, beta)
    naive_kernel = NaiveInverseMultiQuadraticKernel(c, beta)
    np.testing.assert_allclose(
        kernel.dk_dx_dy(x, y), naive_kernel.dk_dx_dy(x, y), rtol=1e-4
    )

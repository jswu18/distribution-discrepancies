import numpy as np
import pytest

from kernels import InverseMultiQuadraticKernel
from naive_implementations.naive_kernels import NaiveInverseMultiQuadraticKernel


@pytest.mark.parametrize(
    "c,beta,x,y,dk_dx",
    [
        [
            0.1,
            -0.5,
            np.array([3, 2, 3]).astype(float),
            np.array([6, 1, 5]).astype(float),
            np.array([0.057208960, -0.019069654, 0.038139308]),
        ],
        [
            0.2,
            -0.1,
            np.array([-3, -7]).astype(float),
            np.array([1, 5]).astype(float),
            np.array([0.00300911, 0.00902734]),
        ],
        [
            0.3,
            -0.2,
            np.array([3]).astype(float),
            np.array([0]).astype(float),
            np.array([-0.08489938]),
        ],
    ],
)
def test_inverse_multi_quadratic_kernel_dk_dx(
    c: float,
    beta: float,
    x: np.ndarray,
    y: np.ndarray,
    dk_dx: np.ndarray,
):
    kernel = InverseMultiQuadraticKernel(c, beta)
    np.testing.assert_array_almost_equal(kernel.dk_dx(x, y), dk_dx)


@pytest.mark.parametrize(
    "c,beta,x,y",
    [
        [
            0.1,
            -0.5,
            np.array([3, 2, 3]).astype(float),
            np.array([6, 1, 5]).astype(float),
        ],
        [
            0.2,
            -0.1,
            np.array([-3, -7]).astype(float),
            np.array([1, 5]).astype(float),
        ],
        [
            0.3,
            -0.2,
            np.array([3]).astype(float),
            np.array([0]).astype(float),
        ],
    ],
)
def test_naive_inverse_multi_quadratic_kernel_dk_dx(
    c: float,
    beta: float,
    x: np.ndarray,
    y: np.ndarray,
):
    kernel = InverseMultiQuadraticKernel(c, beta)
    naive_kernel = NaiveInverseMultiQuadraticKernel(c, beta)
    np.testing.assert_array_almost_equal(kernel.dk_dx(x, y), naive_kernel.dk_dx(x, y))


@pytest.mark.parametrize(
    "c,beta,x,y,dk_dy",
    [
        [
            0.1,
            -0.5,
            np.array([3, 2, 3]).astype(float),
            np.array([6, 1, 5]).astype(float),
            np.array([-0.05720896, 0.01906965, -0.03813931]),
        ],
        [
            0.2,
            -0.1,
            np.array([-3, -7]).astype(float),
            np.array([1, 5]).astype(float),
            np.array([-0.00300911, -0.00902734]),
        ],
        [
            0.3,
            -0.2,
            np.array([3]).astype(float),
            np.array([0]).astype(float),
            np.array([0.08489938]),
        ],
    ],
)
def test_inverse_multi_quadratic_kernel_dk_dy(
    c: float,
    beta: float,
    x: np.ndarray,
    y: np.ndarray,
    dk_dy: np.ndarray,
):
    kernel = InverseMultiQuadraticKernel(c, beta)
    np.testing.assert_array_almost_equal(kernel.dk_dy(x, y), dk_dy)


@pytest.mark.parametrize(
    "c,beta,x,y",
    [
        [
            0.1,
            -0.5,
            np.array([3, 2, 3]).astype(float),
            np.array([6, 1, 5]).astype(float),
        ],
        [0.2, -0.1, np.array([-3, -7]).astype(float), np.array([1, 5]).astype(float)],
        [0.3, -0.2, np.array([3]).astype(float), np.array([0]).astype(float)],
    ],
)
def test_naive_inverse_multi_quadratic_kernel_dk_dy(
    c: float,
    beta: float,
    x: np.ndarray,
    y: np.ndarray,
):
    kernel = InverseMultiQuadraticKernel(c, beta)
    naive_kernel = NaiveInverseMultiQuadraticKernel(c, beta)
    np.testing.assert_array_almost_equal(kernel.dk_dy(x, y), naive_kernel.dk_dy(x, y))


@pytest.mark.parametrize(
    "c,beta,x,y,dk_dx_dy",
    [
        [
            0.1,
            -0.5,
            np.array([3, 2, 3]).astype(float),
            np.array([6, 1, 5]).astype(float),
            np.array(
                [
                    [-0.01768128, 0.01225031, -0.02450062],
                    [0.01225031, 0.01498622, 0.00816687],
                    [-0.02450062, 0.00816687, 0.0027359],
                ],
            ),
        ],
        [
            0.2,
            -0.1,
            np.array([-3, -7]).astype(float),
            np.array([1, 5]).astype(float),
            np.array([[0.00058682, -0.00049638], [-0.00049638, -0.00073686]]),
        ],
        [
            0.3,
            -0.2,
            np.array([3]).astype(float),
            np.array([0]).astype(float),
            np.array([[-0.03894725]]),
        ],
    ],
)
def test_inverse_multi_quadratic_kernel_dk_dx_dy(
    c: float,
    beta: float,
    x: np.ndarray,
    y: np.ndarray,
    dk_dx_dy: np.ndarray,
):
    kernel = InverseMultiQuadraticKernel(c, beta)
    np.testing.assert_array_almost_equal(kernel.dk_dx_dy(x, y), dk_dx_dy)


@pytest.mark.parametrize(
    "c,beta,x,y",
    [
        [
            0.1,
            -0.5,
            np.array([3, 2, 3]).astype(float),
            np.array([6, 1, 5]).astype(float),
        ],
        [
            0.2,
            -0.1,
            np.array([-3, -7]).astype(float),
            np.array([1, 5]).astype(float),
        ],
        [
            0.3,
            -0.2,
            np.array([3]).astype(float),
            np.array([0]).astype(float),
        ],
    ],
)
def test_naive_inverse_multi_quadratic_kernel_dk_dx_dy(
    c: float,
    beta: float,
    x: np.ndarray,
    y: np.ndarray,
):
    kernel = InverseMultiQuadraticKernel(c, beta)
    naive_kernel = NaiveInverseMultiQuadraticKernel(c, beta)
    np.testing.assert_array_almost_equal(
        kernel.dk_dx_dy(x, y), naive_kernel.dk_dx_dy(x, y)
    )

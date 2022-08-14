import numpy as np
import pytest
from scipy.stats import multivariate_normal

from distributions import Gaussian
from naive_implementations.naive_distributions import NaiveGaussian


@pytest.mark.parametrize(
    "mu,covariance,x,p_x",
    [
        [
            np.array([0]).astype(float),
            np.array([[1]]).astype(float),
            np.array([4]).astype(float),
            0.00013383022,
        ],
        [
            np.array([4, 5]).astype(float),
            np.array([[2, 0.1], [0.1, 1]]).astype(float),
            np.array([2, 1]).astype(float),
            1.9892645e-05,
        ],
        [
            np.array([-3]).astype(float),
            np.array([[2]]).astype(float),
            np.array([-2]).astype(float),
            0.21969566,
        ],
    ],
)
def test_gaussian_pdf(
    mu: np.ndarray, covariance: np.ndarray, x: np.ndarray, p_x: float
):
    gaussian = Gaussian(mu, covariance)
    assert gaussian.p(x) == p_x


@pytest.mark.parametrize(
    "mu,covariance,x",
    [
        [
            np.array([0]).astype(float),
            np.array([[1]]).astype(float),
            np.array([4]).astype(float),
        ],
        [
            np.array([4, 5]).astype(float),
            np.array([[2, 0.1], [0.1, 1]]).astype(float),
            np.array([2, 1]).astype(float),
        ],
        [
            np.array([-3]).astype(float),
            np.array([[2]]).astype(float),
            np.array([-2]).astype(float),
        ],
    ],
)
def test_scipy_gaussian_pdf(mu: np.ndarray, covariance: np.ndarray, x: np.ndarray):
    gaussian = Gaussian(mu, covariance)
    scipy_gaussian = multivariate_normal(mean=mu, cov=covariance)
    np.testing.assert_almost_equal(gaussian.p(x), scipy_gaussian.pdf(x))


@pytest.mark.parametrize(
    "mu,covariance,x,dlog_p_dx",
    [
        [
            np.array([0]).astype(float),
            np.array([[1]]).astype(float),
            np.array([4]).astype(float),
            np.array([3.675753984]),
        ],
        [
            np.array([1, 2]).astype(float),
            np.array([[0.5, 0.1], [0.1, 1]]).astype(float),
            np.array([2, 1]).astype(float),
            np.array([3.3251474, -1.8137169]),
        ],
        [
            np.array([-3]).astype(float),
            np.array([[2]]).astype(float),
            np.array([-2]).astype(float),
            np.array([0.63275605]),
        ],
    ],
)
def test_gaussian_dlog_p_dx(
    mu: np.ndarray, covariance: np.ndarray, x: np.ndarray, dlog_p_dx: np.ndarray
):
    gaussian = Gaussian(mu, covariance)
    np.testing.assert_almost_equal(gaussian.dlog_p_dx(x), dlog_p_dx)


@pytest.mark.parametrize(
    "mu,covariance,x",
    [
        [
            np.array([0]).astype(float),
            np.array([[1]]).astype(float),
            np.array([4]).astype(float),
        ],
        [
            np.array([1, 2]).astype(float),
            np.array([[2, 0.1], [0.1, 1]]).astype(float),
            np.array([2, 1]).astype(float),
        ],
        [
            np.array([-3]).astype(float),
            np.array([[2]]).astype(float),
            np.array([-2]).astype(float),
        ],
    ],
)
def test_naive_gaussian_dlog_p_dx(
    mu: np.ndarray, covariance: np.ndarray, x: np.ndarray
):
    gaussian = Gaussian(mu, covariance)
    naive_gaussian = NaiveGaussian(mu, covariance)
    np.testing.assert_almost_equal(gaussian.dlog_p_dx(x), naive_gaussian.dlog_p_dx(x))

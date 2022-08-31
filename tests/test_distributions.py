from typing import List

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from distributions import BaseDistribution, Gaussian, Mixture
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
    "weights,distributions,x,p_x",
    [
        [
            [0.1, 0.2, 0.7],
            [
                Gaussian(mu=np.array([2, 0]), covariance=np.eye(2)),
                Gaussian(mu=np.array([1, 2]), covariance=np.eye(2)),
                Gaussian(mu=np.array([-1, -2]), covariance=np.eye(2)),
            ],
            np.array([0.2, 0.1]),
            0.012914319,
        ],
        [
            [1.0],
            [Gaussian(mu=np.array([-1, -3]), covariance=2 * np.eye(2))],
            np.array([0.2]),
            0.0042919075,
        ],
        [
            [0.2, 0.8],
            [
                Gaussian(mu=np.array([-1, -3]), covariance=2 * np.eye(2)),
                Mixture(
                    weights=[0.5, 0.5],
                    distributions=[
                        Gaussian(mu=np.array([2, 0]), covariance=np.eye(2)),
                        Gaussian(mu=np.array([1, 2]), covariance=np.eye(2)),
                    ],
                ),
            ],
            np.array([-0.2, 0.1]),
            0.011956636,
        ],
    ],
)
def test_mixture_pdf(
    weights: List[float], distributions: List[BaseDistribution], x, p_x
):
    mixture = Mixture(
        weights=weights,
        distributions=distributions,
    )
    assert mixture.p(x) == p_x


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


@pytest.mark.parametrize(
    "mu,covariance,x,score",
    [
        [
            np.array([0]).astype(float),
            np.array([[1]]).astype(float),
            np.array([4]).astype(float),
            np.array([-4]),
        ],
        [
            np.array([1, 2]).astype(float),
            np.array([[0.5, 0.1], [0.1, 1]]).astype(float),
            np.array([2, 1]).astype(float),
            np.array([-2.2448978, 1.2244898]),
        ],
        [
            np.array([-3]).astype(float),
            np.array([[2]]).astype(float),
            np.array([-2]).astype(float),
            np.array([-0.5]),
        ],
    ],
)
def test_gaussian_score(
    mu: np.ndarray, covariance: np.ndarray, x: np.ndarray, score: np.ndarray
):
    gaussian = Gaussian(mu, covariance)
    np.testing.assert_almost_equal(gaussian.score(x), score)


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
def test_naive_gaussian_score(mu: np.ndarray, covariance: np.ndarray, x: np.ndarray):
    gaussian = Gaussian(mu, covariance)
    naive_gaussian = NaiveGaussian(mu, covariance)
    np.testing.assert_almost_equal(gaussian.score(x), naive_gaussian.score(x))


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
            np.array([[0.5, 0.1], [0.1, 1]]).astype(float),
            np.array([2, 1]).astype(float),
        ],
        [
            np.array([-3]).astype(float),
            np.array([[2]]).astype(float),
            np.array([-2]).astype(float),
        ],
    ],
)
def test_gaussian_d_score_dx(mu: np.ndarray, covariance: np.ndarray, x: np.ndarray):
    gaussian = Gaussian(mu, covariance)
    np.testing.assert_almost_equal(gaussian.d_score_dx(x), -np.linalg.inv(covariance))


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
def test_naive_gaussian_d_score_dx(
    mu: np.ndarray, covariance: np.ndarray, x: np.ndarray
):
    gaussian = Gaussian(mu, covariance)
    naive_gaussian = NaiveGaussian(mu, covariance)
    np.testing.assert_almost_equal(gaussian.d_score_dx(x), naive_gaussian.d_score_dx(x))

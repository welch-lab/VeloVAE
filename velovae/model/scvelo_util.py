"""ScVelo Utility Functions.
     Notice:
     All the functions in this file comes directly from scVelo.
     They are some utility function, which are not included in the scVelo API.
     However, during software development, we found these functions quite useful
     in the initialization stage, since we adopt scVelo's initialization method.
     Therefore, we directly copied the code here and formally cite the work:

     Reference:
     Bergen, V., Lange, M., Peidli, S., Wolf, F. A., & Theis, F. J. (2020).
     Generalizing RNA velocity to transient cell states through dynamical modeling.
     Nature biotechnology, 38(12), 1408-1414.

     We thank the authors for providing these handy tools.
"""
############################################################################


import numpy as np
from numpy import exp
from scipy.sparse import csr_matrix, coo_matrix, issparse, spmatrix

import warnings
from typing import Union

from ..scvelo_preprocessing.moments import get_moments
from ..scvelo_preprocessing.neighbors import (
    get_n_neighs,
    get_neighs,
    neighbors,
    pca,
    verify_neighbors,
    compute_connectivities_umap
)
from sklearn.neighbors import NearestNeighbors


def prod_sum_obs(A, B):
    # dot product and sum over axis 0 (obs) equivalent to np.sum(A * B, 0)

    if issparse(A):
        return A.multiply(B).sum(0).A1
    else:
        return np.einsum("ij, ij -> j", A, B) if A.ndim > 1 else (A * B).sum()


def R_squared(residual, total):
    # Clipping added by GYC: remove warning
    r2 = np.ones(residual.shape[1]) - prod_sum_obs(
        residual, residual
    ) / np.clip(prod_sum_obs(total, total), a_min=1e-6, a_max=None)
    r2[np.isnan(r2)] = 0
    return r2


def test_bimodality(x, bins=30, kde=True):
    # Test for bimodal distribution.

    from scipy.stats import gaussian_kde, norm

    lb, ub = np.min(x), np.percentile(x, 99.9)
    grid = np.linspace(lb, ub if ub <= lb else np.max(x), bins)
    kde_grid = (
        gaussian_kde(x)(grid) if kde else np.histogram(x, bins=grid, density=True)[0]
    )

    idx = int(bins / 2) - 2
    idx += np.argmin(kde_grid[idx: idx + 4])

    peak_0 = kde_grid[:idx].argmax()
    peak_1 = kde_grid[idx:].argmax()
    kde_peak = kde_grid[idx:][
        peak_1
    ]  # min(kde_grid[:idx][peak_0], kde_grid[idx:][peak_1])
    kde_mid = kde_grid[idx:].mean()  # kde_grid[idx]

    t_stat = (kde_peak - kde_mid) / np.clip(np.std(kde_grid) / np.sqrt(bins), 1, None)
    p_val = norm.sf(t_stat)

    grid_0 = grid[:idx]
    grid_1 = grid[idx:]
    means = [
        (grid_0[peak_0] + grid_0[min(peak_0 + 1, len(grid_0) - 1)]) / 2,
        (grid_1[peak_1] + grid_1[min(peak_1 + 1, len(grid_1) - 1)]) / 2,
    ]

    return t_stat, p_val, means  # ~ t_test (reject unimodality if t_stat > 3)


def get_weight(x, y=None, perc=95):
    xy_norm = np.array(x.A if issparse(x) else x)
    if y is not None:
        if issparse(y):
            y = y.A
        xy_norm = xy_norm / np.clip(np.max(xy_norm, axis=0), 1e-3, None)
        xy_norm += y / np.clip(np.max(y, axis=0), 1e-3, None)
    if isinstance(perc, int):
        weights = xy_norm >= np.percentile(xy_norm, perc, axis=0)
    else:
        lb, ub = np.percentile(xy_norm, perc, axis=0)
        weights = (xy_norm <= lb) | (xy_norm >= ub)
    return weights


def sum(a, axis=None):
    """Sum array elements over a given axis.

    Arguments
    ---------
    a
        Elements to sum.
    axis
        Axis along which to sum elements. If `None`, all elements will be summed.
        Defaults to `None`.

    Returns
    -------
    ndarray
        Sum of array along given axis.
    """
    if a.ndim == 1:
        axis = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return a.sum(axis=axis).A1 if issparse(a) else a.sum(axis=axis)


def leastsq_NxN(x, y, fit_offset=False, perc=None, constraint_positive_offset=True):
    # Solves least squares X*b=Y for b.

    if perc is not None:
        if not fit_offset and isinstance(perc, (list, tuple)):
            perc = perc[1]
        weights = csr_matrix(get_weight(x, y, perc=perc)).astype(bool)
        x, y = weights.multiply(x).tocsr(), weights.multiply(y).tocsr()
    else:
        weights = None
    
    xx_ = prod_sum_obs(x, x)
    xy_ = prod_sum_obs(x, y)

    if fit_offset:
        n_obs = x.shape[0] if weights is None else sum(weights)
        x_ = sum(x) / n_obs
        y_ = sum(y) / n_obs
        gamma = (xy_ / n_obs - x_ * y_) / (xx_ / n_obs - x_ ** 2)
        offset = y_ - gamma * x_

        # fix negative offsets:
        if constraint_positive_offset:
            idx = offset < 0
            if gamma.ndim > 0:
                gamma[idx] = xy_[idx] / xx_[idx]
            else:
                gamma = xy_ / xx_
            offset = np.clip(offset, 0, None)
    else:
        gamma = xy_ / xx_
        offset = np.zeros(x.shape[1]) if x.ndim > 1 else 0

    nans_offset, nans_gamma = np.isnan(offset), np.isnan(gamma)
    if np.any([nans_offset, nans_gamma]):
        offset[np.isnan(offset)], gamma[np.isnan(gamma)] = 0, 0
    return offset, gamma


def inv(x):
    x_inv = 1 / x * (x != 0)
    return x_inv


def log(x, eps=1e-6):  # to avoid invalid values for log.
    return np.log(np.clip(x, eps, 1 - eps))


def unspliced(tau, u0, alpha, beta):
    expu = exp(-beta * tau)
    return u0 * expu + alpha / beta * (1 - expu)


def spliced(tau, s0, u0, alpha, beta, gamma):
    c = (alpha - u0 * beta) * inv(gamma - beta)
    expu, exps = exp(-beta * tau), exp(-gamma * tau)

    return s0 * exps + alpha / gamma * (1 - exps) + c * (exps - expu)


def mRNA(tau, u0, s0, alpha, beta, gamma):
    expu, exps = exp(-beta * tau), exp(-gamma * tau)
    expus = (alpha - u0 * beta) * inv(gamma - beta) * (exps - expu)
    u = u0 * expu + alpha / beta * (1 - expu)
    s = s0 * exps + alpha / gamma * (1 - exps) + expus

    return u, s


def vectorize(t, t_, alpha, beta, gamma=None, alpha_=0, u0=0, s0=0, sorted=False):
    o = np.array(t < t_, dtype=int)
    tau = t * o + (t - t_) * (1 - o)

    u0_ = unspliced(t_, u0, alpha, beta)
    s0_ = spliced(t_, s0, u0, alpha, beta, gamma if gamma is not None else beta / 2)

    # vectorize u0, s0 and alpha
    u0 = u0 * o + u0_ * (1 - o)
    s0 = s0 * o + s0_ * (1 - o)
    alpha = alpha * o + alpha_ * (1 - o)

    if sorted:
        idx = np.argsort(t)
        tau, alpha, u0, s0 = tau[idx], alpha[idx], u0[idx], s0[idx]
    return tau, alpha, u0, s0


def tau_inv(u, s=None, u0=None, s0=None, alpha=None, beta=None, gamma=None):

    inv_u = (gamma >= beta) if gamma is not None else True
    inv_us = np.invert(inv_u)

    any_invu = np.any(inv_u) or s is None
    any_invus = np.any(inv_us) and s is not None

    if any_invus:  # tau_inv(u, s)
        beta_ = beta * inv(gamma - beta)
        xinf = alpha / gamma - beta_ * (alpha / beta)
        tau = -1 / gamma * log((s - beta_ * u - xinf) / (s0 - beta_ * u0 - xinf))

    if any_invu:  # tau_inv(u)
        uinf = alpha / beta
        tau_u = -1 / beta * log((u - uinf) / (u0 - uinf))
        tau = tau_u * inv_u + tau * inv_us if any_invus else tau_u
    return tau

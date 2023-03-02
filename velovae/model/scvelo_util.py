############################################################################
"""
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
"""
############################################################################


import numpy as np
from numpy import exp
from scipy.sparse import csr_matrix, issparse

import warnings

import numpy as np
import os
from typing import Union

from scipy.sparse import coo_matrix, issparse, spmatrix

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
        n_obs = x.shape[0] if weights is None else sum_obs(weights)
        x_ = sum_obs(x) / n_obs
        y_ = sum_obs(y) / n_obs
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


############################################################
#  Velocity Embedding
############################################################


def vals_to_csr(vals, rows, cols, shape, split_negative=False):
    graph = coo_matrix((vals, (rows, cols)), shape=shape)

    if split_negative:
        graph_neg = graph.copy()

        graph.data = np.clip(graph.data, 0, 1)
        graph_neg.data = np.clip(graph_neg.data, -1, 0)

        graph.eliminate_zeros()
        graph_neg.eliminate_zeros()

        return graph.tocsr(), graph_neg.tocsr()

    else:
        return graph.tocsr()


def l2_norm(x: Union[np.ndarray, spmatrix], axis: int = 1) -> Union[float, np.ndarray]:
    """Calculate l2 norm along a given axis.
    Arguments
    ---------
    x
        Array to calculate l2 norm of.
    axis
        Axis along which to calculate l2 norm.
    Returns
    -------
    Union[float, ndarray]
        L2 norm along a given axis.
    """

    if issparse(x):
        return np.sqrt(x.multiply(x).sum(axis=axis).A1)
    elif x.ndim == 1:
        return np.sqrt(np.einsum("i, i -> ", x, x))
    elif axis == 0:
        return np.sqrt(np.einsum("ij, ij -> j", x, x))
    elif axis == 1:
        return np.sqrt(np.einsum("ij, ij -> i", x, x))


def cosine_correlation(dX, Vi):
    dx = dX - dX.mean(-1)[:, None]
    # dx = dX
    Vi_norm = l2_norm(Vi, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if Vi_norm == 0:
            result = np.zeros(dx.shape[0])
        else:
            result = (
                np.einsum("ij, j", dx, Vi) / (l2_norm(dx, axis=1) * Vi_norm)[None, :]
            )
    return result


def get_indices(dist, n_neighbors=None, mode_neighbors="distances"):

    D = dist.copy()
    D.data += 1e-6

    n_counts = np.sum(D > 0, axis=1)
    n_neighbors = (
        n_counts.min() if n_neighbors is None else min(n_counts.min(), n_neighbors)
    )
    rows = np.where(n_counts > n_neighbors)[0]
    cumsum_neighs = np.array(np.insert(n_counts.cumsum(), 0, 0)).squeeze()
    dat = D.data
    for row in rows:
        n0, n1 = cumsum_neighs[row], cumsum_neighs[row + 1]
        rm_idx = n0 + dat[n0:n1].argsort()[n_neighbors:]
        dat[rm_idx] = 0
    D.eliminate_zeros()

    D.data -= 1e-6
    if mode_neighbors == "distances":
        indices = D.indices.reshape((-1, n_neighbors))
    elif mode_neighbors == "connectivities":
        knn_indices = D.indices.reshape((-1, n_neighbors))
        knn_distances = D.data.reshape((-1, n_neighbors))
        _, conn = compute_connectivities_umap(
            knn_indices, knn_distances, D.shape[0], n_neighbors
        )
        indices = get_indices_from_csr(conn)
    return indices, D


def get_iterative_indices(
    indices,
    index,
    n_recurse_neighbors=2,
    max_neighs=None,
):
    def iterate_indices(indices, index, n_recurse_neighbors):
        if n_recurse_neighbors > 1:
            index = iterate_indices(indices, index, n_recurse_neighbors - 1)
        ix = np.append(index, indices[index])  # direct and indirect neighbors
        if np.isnan(ix).any():
            ix = ix[~np.isnan(ix)]
        return ix.astype(int)

    indices = np.unique(iterate_indices(indices, index, n_recurse_neighbors))
    if max_neighs is not None and len(indices) > max_neighs:
        indices = np.random.choice(indices, max_neighs, replace=False)
    return indices


def normalize(X):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if issparse(X):
            return X.multiply(csr_matrix(1.0 / np.abs(X).sum(1)))
        else:
            return X / X.sum(1)


def knnx0_neighbor_index(t, z, n_neighbors, n_bins):
    ############################################################
    # Given cell time and state, find KNN for each cell in a time 
    # window ahead of it.
    ############################################################
    indices = np.empty((len(t), n_neighbors))
    tmax, tmin = t.max(), t.min()
    delta_t = (tmax-tmin+1e-6)/(n_bins+1)
    idx_sort = np.argsort(t)
    bin_size = len(t) // n_bins

    for i in range(n_bins-1):
        knn_model = NearestNeighbors(n_neighbors=n_neighbors)
        knn_model.fit(z[idx_sort[(i+1)*bin_size:(i+2)*bin_size]])
        dist, ind = knn_model.kneighbors(z[idx_sort[i*bin_size:(i+1)*bin_size]])
        indices[idx_sort[i*bin_size:(i+1)*bin_size]] = idx_sort[(i+1)*bin_size:(i+2)*bin_size][ind] 

    # Build KNN within the last two bins
    knn_model = NearestNeighbors(n_neighbors=n_neighbors)
    knn_model.fit(z[idx_sort[-bin_size:]])
    dist, ind = knn_model.kneighbors(z[idx_sort[(n_bins-1)*bin_size:n_bins*bin_size]])
    indices[idx_sort[(n_bins-1)*bin_size:n_bins*bin_size]] = idx_sort[-bin_size:][ind]

    dist, ind = knn_model.kneighbors(z[idx_sort[n_bins*bin_size:]])
    indices[idx_sort[n_bins*bin_size:]] = idx_sort[-bin_size:][ind]
    return indices


def knnx0_neighbor_index_cellwise(t, z, n_neighbors, bin_size=100, n_shift=100):
    ############################################################
    # Given cell time and state, find KNN for each cell in a time 
    # window ahead of it. 
    ############################################################
    indices = np.empty((len(t), n_neighbors))
    idx_sort = np.argsort(t)

    knn_model = NearestNeighbors(n_neighbors=n_neighbors)
    for i in range(len(t) - n_shift - n_neighbors):
        start = i+n_shift
        end = min(i+n_shift+bin_size, len(t)-1)
        knn_model.fit(z[idx_sort[start:end]])
        dist, ind = knn_model.kneighbors(z[[idx_sort[i]]])
        indices[idx_sort[i]] = idx_sort[start:end][ind[0]]

    #Build KNN within the last two bins
    knn_model = NearestNeighbors(n_neighbors=n_neighbors)
    knn_model.fit(z[idx_sort[-n_shift - n_neighbors:]])
    dist, ind = knn_model.kneighbors(z[idx_sort[-n_shift - n_neighbors:]])
    indices[idx_sort[-n_shift - n_neighbors:]] = idx_sort[-n_shift - n_neighbors:][ind]

    return indices


class VelocityGraph:
    def __init__(
        self,
        adata,
        vkey="velocity",
        xkey="Ms",
        tkey=None,
        state_key=None,
        basis=None,
        n_neighbors=None,
        n_bins=10,
        n_shift=None,
        sqrt_transform=None,
        n_recurse_neighbors=None,
        random_neighbors_at_max=None,
        gene_subset=None,
        approx=None,
        report=False,
        compute_uncertainties=None,
        mode_neighbors="distances",
    ):

        subset = np.ones(adata.n_vars, bool)
        if gene_subset is not None:
            var_names_subset = adata.var_names.isin(gene_subset)
            subset &= var_names_subset if len(var_names_subset) > 0 else gene_subset
        elif f"{vkey}_genes" in adata.var.keys():
            subset &= np.array(adata.var[f"{vkey}_genes"].values, dtype=bool)

        xkey = xkey if xkey in adata.layers.keys() else "spliced"

        X = np.array(
            adata.layers[xkey].A[:, subset]
            if issparse(adata.layers[xkey])
            else adata.layers[xkey][:, subset]
        )
        V = np.array(
            adata.layers[vkey].A[:, subset]
            if issparse(adata.layers[vkey])
            else adata.layers[vkey][:, subset]
        )

        nans = np.isnan(np.sum(V, axis=0))
        if np.any(nans):
            X = X[:, ~nans]
            V = V[:, ~nans]

        if approx is True and X.shape[1] > 100:
            X_pca, PCs, _, _ = pca(X, n_comps=30, svd_solver="arpack", return_info=True)
            self.X = np.array(X_pca, dtype=np.float32)
            self.V = (V - V.mean(0)).dot(PCs.T)
            self.V[V.sum(1) == 0] = 0
        else:
            self.X = np.array(X, dtype=np.float32)
            self.V = np.array(V, dtype=np.float32)
        self.V_raw = np.array(self.V)

        self.sqrt_transform = sqrt_transform
        uns_key = f"{vkey}_params"
        if self.sqrt_transform is None:
            if uns_key in adata.uns.keys() and "mode" in adata.uns[uns_key]:
                self.sqrt_transform = adata.uns[uns_key]["mode"] == "stochastic"
        if self.sqrt_transform:
            self.V = np.sqrt(np.abs(self.V)) * np.sign(self.V)
        self.V -= np.nanmean(self.V, axis=1)[:, None]

        self.n_recurse_neighbors = n_recurse_neighbors
        if self.n_recurse_neighbors is None:
            if n_neighbors is not None or mode_neighbors == "connectivities":
                self.n_recurse_neighbors = 1
            else:
                self.n_recurse_neighbors = 2

        if "neighbors" not in adata.uns.keys():
            neighbors(adata)
        if np.min((get_neighs(adata, "distances") > 0).sum(1).A1) == 0:
            raise ValueError(
                "Your neighbor graph seems to be corrupted. "
                "Consider recomputing via pp.neighbors."
            )
        # if n_neighbors is None or n_neighbors <= get_n_neighs(adata):

        self.indices = get_indices(
            dist=get_neighs(adata, "distances"),
            n_neighbors=n_neighbors,
            mode_neighbors=mode_neighbors,
        )[0]
        """
        print("Apply timed KNN")
        t = adata.obs[tkey].to_numpy()
        z = adata.obsm[state_key]
        bin_size = adata.n_obs//n_bins+1
        if(n_shift is None):
            n_shift = bin_size * 2
        #self.indices =  knnx0_neighbor_index(t, z, n_neighbors, n_bins)
        self.indices = knnx0_neighbor_index_cellwise(t, z, n_neighbors, bin_size, n_shift)
        """

        self.max_neighs = random_neighbors_at_max

        gkey, gkey_ = f"{vkey}_graph", f"{vkey}_graph_neg"
        self.graph = adata.uns[gkey] if gkey in adata.uns.keys() else []
        self.graph_neg = adata.uns[gkey_] if gkey_ in adata.uns.keys() else []

        self.compute_uncertainties = compute_uncertainties
        self.uncertainties = None
        self.self_prob = None
        self.report = report
        self.adata = adata

    def compute_cosines(self):
        n_obs = self.X.shape[0]
        vals, rows, cols, uncertainties = [], [], [], []
        if self.compute_uncertainties:
            moments = get_moments(self.adata, np.sign(self.V_raw), second_order=True)

        for obs_id in range(n_obs):
            if self.V[obs_id].max() != 0 or self.V[obs_id].min() != 0:
                neighs_idx = get_iterative_indices(
                    self.indices, obs_id, self.n_recurse_neighbors, self.max_neighs
                )

                dX = self.X[neighs_idx] - self.X[obs_id, None]  # 60% of runtime
                if self.sqrt_transform:
                    dX = np.sqrt(np.abs(dX)) * np.sign(dX)
                val = cosine_correlation(dX, self.V[obs_id])  # 40% of runtime

                if self.compute_uncertainties:
                    dX /= l2_norm(dX)[:, None]
                    uncertainties.extend(
                        np.nansum(dX ** 2 * moments[obs_id][None, :], 1)
                    )

                vals.extend(val)
                rows.extend(np.ones(len(neighs_idx)) * obs_id)
                cols.extend(neighs_idx)

        vals = np.hstack(vals)
        vals[np.isnan(vals)] = 0

        self.graph, self.graph_neg = vals_to_csr(
            vals, rows, cols, shape=(n_obs, n_obs), split_negative=True
        )

        if self.compute_uncertainties:
            uncertainties = np.hstack(uncertainties)
            uncertainties[np.isnan(uncertainties)] = 0
            self.uncertainties = vals_to_csr(
                uncertainties, rows, cols, shape=(n_obs, n_obs), split_negative=False
            )
            self.uncertainties.eliminate_zeros()

        confidence = self.graph.max(1).A.flatten()
        self.self_prob = np.clip(np.percentile(confidence, 98) - confidence, 0, 1)

        return


def velocity_graph(
    data,
    vkey="velocity",
    xkey="Ms",
    tkey='velovae_time',
    state_key='velovae_z',
    basis=None,
    n_neighbors=None,
    n_bins=10,
    n_shift=None,
    n_recurse_neighbors=None,
    random_neighbors_at_max=None,
    sqrt_transform=None,
    variance_stabilization=None,
    gene_subset=None,
    compute_uncertainties=None,
    approx=None,
    mode_neighbors="distances",
    copy=False,
    n_jobs=None,
    backend="loky",
):
    """Computes velocity graph based on cosine similarities.
    The cosine similarities are computed between velocities and potential cell state
    transitions, i.e. it measures how well a corresponding change in gene expression
    :math:`\\delta_{ij} = x_j - x_i` matches the predicted change according to the
    velocity vector :math:`\\nu_i`,
    .. math::
        \\pi_{ij} = \\cos\\angle(\\delta_{ij}, \\nu_i)
        = \\frac{\\delta_{ij}^T \\nu_i}{\\left\\lVert\\delta_{ij}\\right\\rVert
        \\left\\lVert \\nu_i \\right\\rVert}.
    Arguments
    ---------
    data: :class:`~anndata.AnnData`
        Annotated data matrix.
    vkey: `str` (default: `'velocity'`)
        Name of velocity estimates to be used.
    xkey: `str` (default: `'Ms'`)
        Layer key to extract count data from.
    tkey: `str` (default: `velovae_time`)
        Observation key to extract time data from.
    state_key: `str` (default: `velovae_z`)
        Multi-dimensional Observation key to extract cell state from
    basis: `str` (default: `None`)
        Basis / Embedding to use.
    n_neighbors: `int` or `None` (default: None)
        Use fixed number of neighbors or do recursive neighbor search (if `None`).
    n_bins: `int` (default: 10)
        Number of time bins used to compute velocity graph
    n_recurse_neighbors: `int` (default: `None`)
        Number of recursions for neighbors search. Defaults to
        2 if mode_neighbors is 'distances', and 1 if mode_neighbors is 'connectivities'.
    random_neighbors_at_max: `int` or `None` (default: `None`)
        If number of iterative neighbors for an individual cell is higher than this
        threshold, a random selection of such are chosen as reference neighbors.
    sqrt_transform: `bool` (default: `False`)
        Whether to variance-transform the cell states changes
        and velocities before computing cosine similarities.
    gene_subset: `list` of `str`, subset of adata.var_names or `None`(default: `None`)
        Subset of genes to compute velocity graph on exclusively.
    compute_uncertainties: `bool` (default: `None`)
        Whether to compute uncertainties along with cosine correlation.
    approx: `bool` or `None` (default: `None`)
        If True, first 30 pc's are used instead of the full count matrix
    mode_neighbors: 'str' (default: `'distances'`)
        Determines the type of KNN graph used. Options are 'distances' or
        'connectivities'. The latter yields a symmetric graph.
    copy: `bool` (default: `False`)
        Return a copy instead of writing to adata.
    n_jobs: `int` or `None` (default: `None`)
        Number of parallel jobs.
    backend: `str` (default: "loky")
        Backend used for multiprocessing. See :class:`joblib.Parallel` for valid
        options.
    Returns
    -------
    velocity_graph: `.uns`
        sparse matrix with correlations of cell state transitions with velocities
    """

    adata = data.copy() if copy else data
    verify_neighbors(adata)
    if vkey not in adata.layers.keys():
        velocity(adata, vkey=vkey)
    if sqrt_transform is None:
        sqrt_transform = variance_stabilization

    vgraph = VelocityGraph(
        adata,
        vkey=vkey,
        xkey=xkey,
        tkey=tkey,
        state_key=state_key,
        basis=basis,
        n_neighbors=n_neighbors,
        n_bins=n_bins,
        n_shift=n_shift,
        approx=approx,
        n_recurse_neighbors=n_recurse_neighbors,
        random_neighbors_at_max=random_neighbors_at_max,
        sqrt_transform=sqrt_transform,
        gene_subset=gene_subset,
        compute_uncertainties=compute_uncertainties,
        report=True,
        mode_neighbors=mode_neighbors,
    )

    if isinstance(basis, str):
        print(
            f"The velocity graph is computed on {basis} embedding coordinates.\n",
            f"Consider computing the graph in an unbiased manner \n",
            f"on full expression space by not specifying basis.\n"
        )

    vgraph.compute_cosines()

    adata.uns[f"{vkey}_graph"] = vgraph.graph
    adata.uns[f"{vkey}_graph_neg"] = vgraph.graph_neg

    if vgraph.uncertainties is not None:
        adata.uns[f"{vkey}_graph_uncertainties"] = vgraph.uncertainties

    adata.obs[f"{vkey}_self_transition"] = vgraph.self_prob

    if f"{vkey}_params" in adata.uns.keys():
        if "embeddings" in adata.uns[f"{vkey}_params"]:
            del adata.uns[f"{vkey}_params"]["embeddings"]
    else:
        adata.uns[f"{vkey}_params"] = {}
    adata.uns[f"{vkey}_params"]["mode_neighbors"] = mode_neighbors
    adata.uns[f"{vkey}_params"]["n_recurse_neighbors"] = vgraph.n_recurse_neighbors

    print(
        f"Added '{vkey}_graph', sparse matrix with cosine correlations (adata.uns)"
    )

    return adata if copy else None


def transition_matrix(
    adata,
    vkey="velocity",
    basis=None,
    backward=False,
    self_transitions=False,
    scale=10,
    perc=None,
    threshold=None,
    use_negative_cosines=False,
    weight_diffusion=0,
    scale_diffusion=1,
    weight_indirect_neighbors=None,
    n_neighbors=None,
    vgraph=None,
    basis_constraint=None,
):
    """Computes cell-to-cell transition probabilities
    .. math::
        \\tilde \\pi_{ij} = \\frac1{z_i} \\exp( \\pi_{ij} / \\sigma),
    from the velocity graph :math:`\\pi_{ij}`, with row-normalization :math:`z_i` and
    kernel width :math:`\\sigma` (scale parameter :math:`\\lambda = \\sigma^{-1}`).
    Alternatively, use :func:`cellrank.tl.transition_matrix` to account for uncertainty
    in the velocity estimates.
    Arguments
    ---------
    adata: :class:`~anndata.AnnData`
        Annotated data matrix.
    vkey: `str` (default: `'velocity'`)
        Name of velocity estimates to be used.
    basis: `str` or `None` (default: `None`)
        Restrict transition to embedding if specified
    backward: `bool` (default: `False`)
        Whether to use the transition matrix to
        push forward (`False`) or to pull backward (`True`)
    self_transitions: `bool` (default: `True`)
        Allow transitions from one node to itself.
    scale: `float` (default: 10)
        Scale parameter of gaussian kernel.
    perc: `float` between `0` and `100` or `None` (default: `None`)
        Determines threshold of transitions to include.
    use_negative_cosines: `bool` (default: `False`)
        If True, negatively similar transitions are taken into account.
    weight_diffusion: `float` (default: 0)
        Relative weight to be given to diffusion kernel (Brownian motion)
    scale_diffusion: `float` (default: 1)
        Scale of diffusion kernel.
    weight_indirect_neighbors: `float` between `0` and `1` or `None` (default: `None`)
        Weight to be assigned to indirect neighbors (i.e. neighbors of higher degrees).
    n_neighbors:`int` (default: None)
        Number of nearest neighbors to consider around each cell.
    vgraph: csr matrix or `None` (default: `None`)
        Velocity graph representation to use instead of adata.uns[f'{vkey}_graph'].
    Returns
    -------
    Returns sparse matrix with transition probabilities.
    """

    if f"{vkey}_graph" not in adata.uns:
        raise ValueError(
            "You need to run `tl.velocity_graph` first to compute cosine correlations."
        )

    graph_neg = None
    if vgraph is not None:
        graph = vgraph.copy()
    else:
        if hasattr(adata, "obsp") and f"{vkey}_graph" in adata.obsp.keys():
            graph = csr_matrix(adata.obsp[f"{vkey}_graph"]).copy()
            if f"{vkey}_graph_neg" in adata.obsp.keys():
                graph_neg = adata.obsp[f"{vkey}_graph_neg"]
        else:
            graph = csr_matrix(adata.uns[f"{vkey}_graph"]).copy()
            if f"{vkey}_graph_neg" in adata.uns.keys():
                graph_neg = adata.uns[f"{vkey}_graph_neg"]

    if basis_constraint is not None and f"X_{basis_constraint}" in adata.obsm.keys():
        print("basis_constraint")
        from sklearn.neighbors import NearestNeighbors

        neighs = NearestNeighbors(n_neighbors=100)
        neighs.fit(adata.obsm[f"X_{basis_constraint}"])
        basis_graph = neighs.kneighbors_graph(mode="connectivity") > 0
        graph = graph.multiply(basis_graph)

    if self_transitions:
        print("self_transitions")
        confidence = graph.max(1).A.flatten()
        ub = np.percentile(confidence, 98)
        self_prob = np.clip(ub - confidence, 0, 1)
        graph.setdiag(self_prob)

    T = np.expm1(graph * scale)  # equivalent to np.exp(graph.A * scale) - 1
    if graph_neg is not None:
        print("graph_neg")
        graph_neg = adata.uns[f"{vkey}_graph_neg"]
        if use_negative_cosines:
            T -= np.expm1(-graph_neg * scale)
        else:
            T += np.expm1(graph_neg * scale)
            T.data += 1

    # weight direct and indirect (recursed) neighbors
    if weight_indirect_neighbors is not None and weight_indirect_neighbors < 1:
        print("weight_indirect_neighbors")
        direct_neighbors = get_neighs(adata, "distances") > 0
        direct_neighbors.setdiag(1)
        w = weight_indirect_neighbors
        T = w * T + (1 - w) * direct_neighbors.multiply(T)

    if n_neighbors is not None:
        print("n_neighbors")
        T = T.multiply(
            get_connectivities(
                adata, mode="distances", n_neighbors=n_neighbors, recurse_neighbors=True
            )
        )

    if perc is not None or threshold is not None:
        print("perc or threshold")
        if threshold is None:
            threshold = np.percentile(T.data, perc)
        T.data[T.data < threshold] = 0
        T.eliminate_zeros()

    if backward:
        print("backward")
        T = T.T
    T = normalize(T)

    if f"X_{basis}" in adata.obsm.keys():
        print(f"X_{basis}")
        dists_emb = (T > 0).multiply(squareform(pdist(adata.obsm[f"X_{basis}"])))
        scale_diffusion *= dists_emb.data.mean()

        diffusion_kernel = dists_emb.copy()
        diffusion_kernel.data = np.exp(
            -0.5 * dists_emb.data ** 2 / scale_diffusion ** 2
        )
        T = T.multiply(diffusion_kernel)  # combine velocity kernel & diffusion kernel

        if 0 < weight_diffusion < 1:  # add diffusion kernel (Brownian motion - like)
            diffusion_kernel.data = np.exp(
                -0.5 * dists_emb.data ** 2 / (scale_diffusion / 2) ** 2
            )
            T = (1 - weight_diffusion) * T + weight_diffusion * diffusion_kernel

        T = normalize(T)

    return T

def quiver_autoscale(X_emb, V_emb):
    import matplotlib.pyplot as pl

    #scale_factor = np.abs(X_emb).max()  # just so that it handles very large values
    scale_factor = np.quantile(np.abs(X_emb), 0.95)
    fig, ax = pl.subplots()
    Q = ax.quiver(
        X_emb[:, 0] / scale_factor,
        X_emb[:, 1] / scale_factor,
        V_emb[:, 0],
        V_emb[:, 1],
        angles="xy",
        scale_units="xy",
        scale=None,
    )
    Q._init()
    fig.clf()
    pl.close(fig)
    return Q.scale / scale_factor


def velocity_embedding(
    data,
    basis=None,
    vkey="velocity",
    scale=10,
    self_transitions=True,
    use_negative_cosines=True,
    direct_pca_projection=None,
    retain_scale=False,
    autoscale=True,
    all_comps=True,
    T=None,
    copy=False,
):
    """Projects the single cell velocities into any embedding.
    Given normalized difference of the embedding positions
    :math:
    `\\tilde \\delta_{ij} = \\frac{x_j-x_i}{\\left\\lVert x_j-x_i \\right\\rVert}`.
    the projections are obtained as expected displacements with respect to the
    transition matrix :math:`\\tilde \\pi_{ij}` as
    .. math::
        \\tilde \\nu_i = E_{\\tilde \\pi_{i\\cdot}} [\\tilde \\delta_{i \\cdot}]
        = \\sum_{j \\neq i} \\left( \\tilde \\pi_{ij} - \\frac1n \\right) \\tilde \\
        delta_{ij}.
    Arguments
    ---------
    data: :class:`~anndata.AnnData`
        Annotated data matrix.
    basis: `str` (default: `'tsne'`)
        Which embedding to use.
    vkey: `str` (default: `'velocity'`)
        Name of velocity estimates to be used.
    scale: `int` (default: 10)
        Scale parameter of gaussian kernel for transition matrix.
    self_transitions: `bool` (default: `True`)
        Whether to allow self transitions, based on the confidences of transitioning to
        neighboring cells.
    use_negative_cosines: `bool` (default: `True`)
        Whether to project cell-to-cell transitions with negative cosines into
        negative/opposite direction.
    direct_pca_projection: `bool` (default: `None`)
        Whether to directly project the velocities into PCA space,
        thus skipping the velocity graph.
    retain_scale: `bool` (default: `False`)
        Whether to retain scale from high dimensional space in embedding.
    autoscale: `bool` (default: `True`)
        Whether to scale the embedded velocities by a scalar multiplier,
        which simply ensures that the arrows in the embedding are properly scaled.
    all_comps: `bool` (default: `True`)
        Whether to compute the velocities on all embedding components.
    T: `csr_matrix` (default: `None`)
        Allows the user to directly pass a transition matrix.
    copy: `bool` (default: `False`)
        Return a copy instead of writing to `adata`.
    Returns
    -------
    velocity_umap: `.obsm`
        coordinates of velocity projection on embedding (e.g., basis='umap')
    """

    adata = data.copy() if copy else data

    if basis is None:
        keys = [
            key for key in ["pca", "tsne", "umap"] if f"X_{key}" in adata.obsm.keys()
        ]
        if len(keys) > 0:
            basis = "pca" if direct_pca_projection else keys[-1]
        else:
            raise ValueError("No basis specified")

    if f"X_{basis}" not in adata.obsm_keys():
        raise ValueError("You need to compute the embedding first.")

    if direct_pca_projection and "pca" in basis:
        print(
            "Directly projecting velocities into PCA space is for exploratory analysis ",
            "on principal components.\n",
            "It does not reflect the actual velocity field from high ",
            "dimensional gene expression space.\n",
            "To visualize velocities, consider applying ",
            "`direct_pca_projection=False`.\n"
        )

    print("Computing velocity embedding")

    V = np.array(adata.layers[vkey])
    vgenes = np.ones(adata.n_vars, dtype=bool)
    if f"{vkey}_genes" in adata.var.keys():
        vgenes &= np.array(adata.var[f"{vkey}_genes"], dtype=bool)
    vgenes &= ~np.isnan(V.sum(0))
    V = V[:, vgenes]

    if direct_pca_projection and "pca" in basis:
        PCs = adata.varm["PCs"] if all_comps else adata.varm["PCs"][:, :2]
        PCs = PCs[vgenes]

        X_emb = adata.obsm[f"X_{basis}"]
        V_emb = (V - V.mean(0)).dot(PCs)

    else:
        X_emb = (
            adata.obsm[f"X_{basis}"] if all_comps else adata.obsm[f"X_{basis}"][:, :2]
        )
        V_emb = np.zeros(X_emb.shape)

        T = (
            transition_matrix(
                adata,
                vkey=vkey,
                scale=scale,
                self_transitions=self_transitions,
                use_negative_cosines=use_negative_cosines,
            )
            if T is None
            else T
        )
        T.setdiag(0)
        T.eliminate_zeros()

        densify = adata.n_obs < 1e4
        TA = T.A if densify else None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(adata.n_obs):
                indices = T[i].indices
                dX = X_emb[indices] - X_emb[i, None]  # shape (n_neighbors, 2)
                if not retain_scale:
                    dX /= l2_norm(dX)[:, None]
                dX[np.isnan(dX)] = 0  # zero diff in a steady-state
                probs = TA[i, indices] if densify else T[i].data
                V_emb[i] = probs.dot(dX) - probs.mean() * dX.sum(0)

        if retain_scale:
            print("retain_scale")
            X = (
                adata.layers["Ms"]
                if "Ms" in adata.layers.keys()
                else adata.layers["spliced"]
            )
            delta = T.dot(X[:, vgenes]) - X[:, vgenes]
            if issparse(delta):
                delta = delta.A
            cos_proj = (V * delta).sum(1) / l2_norm(delta)
            V_emb *= np.clip(cos_proj[:, None] * 10, 0, 1)

    if autoscale:
        V_emb /= 3 * quiver_autoscale(X_emb, V_emb)

    if f"{vkey}_params" in adata.uns.keys():
        adata.uns[f"{vkey}_params"]["embeddings"] = (
            []
            if "embeddings" not in adata.uns[f"{vkey}_params"]
            else list(adata.uns[f"{vkey}_params"]["embeddings"])
        )
        adata.uns[f"{vkey}_params"]["embeddings"].extend([basis])

    vkey += f"_{basis}"
    adata.obsm[vkey] = V_emb

    print(f"Added '{vkey}', embedded velocity vectors (adata.obsm)")

    return adata if copy else None

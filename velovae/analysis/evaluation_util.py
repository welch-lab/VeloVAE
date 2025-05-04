from typing import Dict, Iterable, List, Literal, Optional, Tuple
from anndata import AnnData
import numpy as np
from scipy.stats import spearmanr, poisson, norm, mannwhitneyu
from scipy.special import loggamma
import scanpy as sc
from sklearn.metrics.pairwise import pairwise_distances
import hnswlib
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from ..model.model_util import pred_su_numpy, ode_numpy, ode_br_numpy, scv_pred, scv_pred_single


def get_mse(
    U: np.ndarray,
    S: np.ndarray,
    Uhat: np.ndarray,
    Shat: np.ndarray,
    axis: Optional[int] = None
) -> float:
    """
    Compute the mean squared error between the original and estimated arrays.

    The mean squared error is computed as the average of the squared differences 
    between the elements of U and Uhat, and between S and Shat. If an axis is specified,
    the mean is computed along that axis; otherwise, it is computed over the entire array.

    Args:
        U (np.ndarray): Original array U.
        S (np.ndarray): Original array S.
        Uhat (np.ndarray): Estimated array U.
        Shat (np.ndarray): Estimated array S.
        axis (Optional[int], optional): Axis along which to compute the mean. Defaults to None.

    Returns:
        float: The computed mean squared error.
    """
    if axis is not None:
        return np.nanmean((U-Uhat)**2+(S-Shat)**2, axis=axis)
    return np.nanmean((U-Uhat)**2+(S-Shat)**2)


def get_mae(
    U: np.ndarray,
    S: np.ndarray,
    Uhat: np.ndarray,
    Shat: np.ndarray,
    axis: Optional[int] = None
) -> float:
    """ Calculates the MAE in the similar fashion as get_mse. """
    if axis is not None:
        return np.nanmean((U-Uhat)**2+(S-Shat)**2, axis=axis)
    return np.nanmean(np.abs(U-Uhat)+np.abs(S-Shat))


def time_corr(t1: np.ndarray, t2: np.ndarray) -> float:
    """ Spearman correlation coefficient """
    return spearmanr(t1, t2)


def poisson_log_likelihood(mu: np.ndarray, obs: np.ndarray) -> float:
    return -mu+obs*np.log(mu)-loggamma(obs)


def cell_state(
    adata: AnnData,
    method: Literal[
        'scVelo',
        'Vanilla VAE',
        'VeloVAE',
        'FullVB',
        'Discrete VeloVAE',
        'Discrete FullVB',
        'VeloVI'
    ],
    key: str,
    gene_indices: Optional[np.ndarray] = None,
    **kwargs
) -> np.ndarray:
    """
    Assigns cells to one of three states: 'off', 'induction' or 'repression'.

    Args:
        adata (:class:`anndata.AnnData`):
            AnnData Object
        method (Literal):
            Model name.
            Now supports 'scVelo', 'Vanilla VAE', 'VeloVAE', 'FullVB',
            'Discrete VeloVAE', 'Discrete FullVB' and 'VeloVI'.
        key (str):
            Key for extracting model outputs.
        gene_indices (:class:`numpy.ndarray`, optional):
            . Defaults to None.

    Returns:
        :class:`numpy.ndarray`:
            Cell state assignment.
    """
    cell_state = None
    if gene_indices is None:
        gene_indices = np.array(np.range(adata.n_vars))
    if method == 'scVelo':
        t = adata.layers[f"{key}_t"][:, gene_indices]
        toff = adata.var[f"{key}_t_"].to_numpy()[gene_indices]
        cell_state = (t > toff)
    elif method == 'Vanilla VAE':
        t = adata.obs[f"{key}_time"].to_numpy()
        toff = adata.var[f"{key}_toff"].to_numpy()
        ton = adata.var[f"{key}_ton"].to_numpy()
        toff = toff[gene_indices]
        ton = ton[gene_indices]
        cell_state = (t.reshape(-1, 1) > toff) + (t.reshape(-1, 1) < ton)*2
    elif method in ['VeloVAE', 'FullVB', 'Discrete VeloVAE', 'Discrete FullVB']:
        rho = adata.layers[f"{key}_rho"][:, gene_indices]
        t = adata.obs[f"{key}_time"].to_numpy()
        mask_induction = rho > 0.01
        mask_repression = np.empty(mask_induction.shape, dtype=bool)
        mask_off = np.empty(mask_induction.shape, dtype=bool)
        for i in range(len(gene_indices)):
            if np.any(mask_induction[:, i]):
                ton = np.quantile(t[mask_induction[:, i]], 1e-3)
                mask_repression[:, i] = (rho[:, i] <= 0.1) & (t >= ton)
                mask_off[:, i] = (rho[:, i] <= 0.1) & (t < ton)
            else:
                mask_repression[:, i] = True
                mask_off[:, i] = False
        cell_state = mask_repression * 1 + mask_off * 2
    elif method == 'VeloVI':
        t = adata.layers["fit_t"][:, gene_indices]
        toff = adata.var[f"{key}_t_"].to_numpy()[gene_indices]
        cell_state = (t > toff)
    else:
        cell_state = np.array([3 for i in range(adata.n_obs)])
    return cell_state


# scVelo


def get_err_scv(
    adata: AnnData, key: str = 'fit'
) -> Tuple[float, Optional[float], float, Optional[float], float, Optional[float]]:
    """
    Get performance metrics from scVelo results.
    """
    Uhat, Shat = scv_pred(adata, key)
    mse = get_mse(adata.layers['Mu'], adata.layers['Ms'], Uhat, Shat)
    mae = get_mae(adata.layers['Mu'], adata.layers['Ms'], Uhat, Shat)
    logp = np.sum(np.log(adata.var[f"{key}_likelihood"]))
    adata.var[f'{key}_mse'] = get_mse(adata.layers['Mu'], adata.layers['Ms'], Uhat, Shat, axis=0)
    adata.var[f'{key}_mae'] = get_mae(adata.layers['Mu'], adata.layers['Ms'], Uhat, Shat, axis=0)

    return mse, None, mae, None, logp, None


def get_pred_scv_demo(
    adata: AnnData,
    key: str = 'fit',
    genes: Optional[Iterable[str]] = None,
    N: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get prediction from scVelo for plotting.
    """
    if genes is None:
        genes = adata.var_names
    alpha = adata.var[f"{key}_alpha"].to_numpy()
    beta = adata.var[f"{key}_beta"].to_numpy()
    gamma = adata.var[f"{key}_gamma"].to_numpy()
    toff = adata.var[f"{key}_t_"].to_numpy()
    T = adata.layers[f"{key}_t"]
    scaling = adata.var[f"{key}_scaling"].to_numpy()
    Uhat, Shat = np.zeros((2*N, len(genes))), np.zeros((2*N, len(genes)))
    T_demo = np.zeros((2*N, len(genes)))
    for i, gene in enumerate(genes):
        idx = np.where(adata.var_names == gene)[0][0]
        t_1 = np.linspace(0, toff[idx], N)
        t_2 = np.linspace(toff[idx], max(T[:, idx].max(), toff[i]+T[:, idx].max()*0.01), N)
        t_demo = np.concatenate((t_1, t_2))
        T_demo[:, i] = t_demo
        uhat, shat = scv_pred_single(
            t_demo,
            alpha[idx],
            beta[idx],
            gamma[idx],
            toff[idx],
            scaling=scaling[idx],
            uinit=0,
            sinit=0
        )
        Uhat[:, i] = uhat
        Shat[:, i] = shat
    return T_demo, Uhat, Shat


# Vanilla VAE


def get_pred_vanilla(adata: AnnData, key: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get prediction from vanilla VAE.
    """
    # Vanilla VAE
    alpha = adata.var[f"{key}_alpha"].to_numpy()
    beta = adata.var[f"{key}_beta"].to_numpy()
    gamma = adata.var[f"{key}_gamma"].to_numpy()
    toff = adata.var[f"{key}_toff"].to_numpy()
    ton = adata.var[f"{key}_ton"].to_numpy()
    t = adata.obs[f"{key}_time"].to_numpy()
    scaling = adata.var[f"{key}_scaling"].to_numpy()

    if (f"{key}_uhat" not in adata.layers) or (f"{key}_shat" not in adata.layers):
        Uhat, Shat = ode_numpy(t.reshape(-1, 1), alpha, beta, gamma, ton, toff, scaling)
    else:
        Uhat, Shat = adata.layers[f"{key}_uhat"], adata.layers[f"{key}_shat"]

    return Uhat, Shat


def get_err_vanilla(
    adata: AnnData, key: str, gene_mask: Optional[Iterable[str]] = None
) -> Tuple[float, float, float, float, float, float]:
    """
    Get performance metrics from vanilla VAE results.
    """
    U, S = adata.layers["Mu"], adata.layers["Ms"]
    Uhat, Shat = get_pred_vanilla(adata, key)
    train_idx, test_idx = adata.uns[f"{key}_train_idx"], adata.uns[f"{key}_test_idx"]

    if gene_mask is None:
        gene_mask = np.ones((adata.n_vars)).astype(bool)

    sigma_u = adata.var[f"{key}_sigma_u"].to_numpy()[gene_mask]
    sigma_s = adata.var[f"{key}_sigma_s"].to_numpy()[gene_mask]
    dist_u_train = np.abs(U[train_idx][:, gene_mask]-Uhat[train_idx][:, gene_mask])
    dist_s_train = np.abs(S[train_idx][:, gene_mask]-Shat[train_idx][:, gene_mask])
    dist_u_test = np.abs(U[test_idx][:, gene_mask]-Uhat[test_idx][:, gene_mask])
    dist_s_test = np.abs(S[test_idx][:, gene_mask]-Shat[test_idx][:, gene_mask])

    logp_train = -dist_u_train**2/(2*sigma_u**2) \
        - dist_s_train**2/(2*sigma_s**2) \
        - np.log(sigma_u) - np.log(sigma_s) - np.log(2*np.pi)
    logp_test = -dist_u_test**2/(2*sigma_u**2) \
        - dist_s_test**2/(2*sigma_s**2) \
        - np.log(sigma_u) - np.log(sigma_s) - np.log(2*np.pi)
    mse_train = np.nanmean(dist_u_train**2+dist_s_train**2)
    mse_test = np.nanmean(dist_u_test**2+dist_s_test**2)
    mae_train = np.nanmean(dist_u_train+dist_s_train)
    mae_test = np.nanmean(dist_u_test+dist_s_test)
    adata.var[f'{key}_mse_train'] = np.nanmean(dist_u_train**2+dist_u_train**2, 0)
    adata.var[f'{key}_mae_train'] = np.nanmean(dist_u_train+dist_u_train, 0)
    adata.var[f'{key}_mse_test'] = np.nanmean(dist_u_test**2+dist_u_test**2, 0)
    adata.var[f'{key}_mae_test'] = np.nanmean(dist_u_test+dist_u_test, 0)

    logp_train = np.nanmean(np.sum(logp_train, 1))
    logp_test = np.nanmean(np.sum(logp_test, 1))
    adata.var[f'{key}_likelihood_train'] = np.nanmean(logp_train, 0)
    adata.var[f'{key}_likelihood_test'] = np.nanmean(logp_test, 0)

    return mse_train, mse_test, mae_train, mae_test, logp_train, logp_test


def get_pred_vanilla_demo(
    adata: AnnData,
    key: str,
    genes: Iterable[str] = None,
    N: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get prediction from scVelo for plotting.
    """
    alpha = adata.var[f"{key}_alpha"].to_numpy()
    beta = adata.var[f"{key}_beta"].to_numpy()
    gamma = adata.var[f"{key}_gamma"].to_numpy()
    toff = adata.var[f"{key}_toff"].to_numpy()
    ton = adata.var[f"{key}_ton"].to_numpy()
    scaling = adata.var[f"{key}_scaling"].to_numpy()
    t = adata.obs[f"{key}_time"].to_numpy()

    t_demo = np.linspace(0, t.max(), N)
    if genes is None:
        Uhat_demo, Shat_demo = ode_numpy(t_demo.reshape(-1, 1), alpha, beta, gamma, ton, toff, scaling)
    else:
        gene_indices = np.array([np.where(adata.var_names == x)[0][0] for x in genes])
        Uhat_demo, Shat_demo = ode_numpy(t_demo.reshape(-1, 1),
                                         alpha[gene_indices],
                                         beta[gene_indices],
                                         gamma[gene_indices],
                                         ton[gene_indices],
                                         toff[gene_indices],
                                         scaling[gene_indices])

    return t_demo, Uhat_demo, Shat_demo


# VeloVAE
def get_pred_velovae(
    adata: AnnData,
    key: str,
    scv_key: Optional[str] = None,
    full_vb: bool = False,
    discrete: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get velocity predictions from a VeloVAE model stored in an AnnData object.

    Args:
        adata (AnnData): Annotated data matrix containing the VeloVAE results.
        key (str): Key in `adata.uns` that stores the VeloVAE model or results.
        scv_key (Optional[str], optional): Optional key to use for comparison or additional velocity estimation 
            (e.g., scVelo velocities). Defaults to None.
        full_vb (bool, optional): If True, use full variational Bayes inference to get velocities, 
            otherwise use default point estimates. Defaults to False.
        discrete (bool, optional): If True, return discrete velocity predictions for cell states, 
            otherwise continuous velocities are returned. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Predicted unspliced and spliced counts
    """
    if (f"{key}_uhat" not in adata.layers) or (f"{key}_shat" not in adata.layers):
        rho = adata.layers[f"{key}_rho"]
        alpha = (adata.var[f"{key}_alpha"].to_numpy() if not full_vb else
                 np.exp(adata.var[f"{key}_logmu_alpha"].to_numpy()))
        beta = (adata.var[f"{key}_beta"].to_numpy() if not full_vb else
                np.exp(adata.var[f"{key}_logmu_beta"].to_numpy()))
        gamma = (adata.var[f"{key}_gamma"].to_numpy() if not full_vb else
                 np.exp(adata.var[f"{key}_logmu_gamma"].to_numpy()))
        t = adata.obs[f"{key}_time"].to_numpy()
        scaling = adata.var[f"{key}_scaling"].to_numpy()

        u0, s0 = adata.layers[f"{key}_u0"], adata.layers[f"{key}_s0"]
        t0 = adata.obs[f"{key}_t0"].to_numpy()

        Uhat, Shat = pred_su_numpy((t-t0).reshape(-1, 1), u0, s0, rho*alpha, beta, gamma)
        Uhat = Uhat*scaling
    else:
        Uhat, Shat = adata.layers[f"{key}_uhat"], adata.layers[f"{key}_shat"]

    return Uhat, Shat


def get_err_velovae(
    adata: AnnData,
    key: str,
    gene_mask: Optional[np.ndarray] = None,
    full_vb: bool = False,
    discrete: bool = False,
    n_sample: int = 25,
    seed: int = 2022
) -> Tuple[float, float, float, float, float, float]:
    """
    Calculate error metrics to evaluate VeloVAE model performance.

    Args:
        adata (AnnData): Annotated data matrix containing the observed and inferred values.
        key (str): Key in `adata` to identify the velocity or outcome to evaluate.
        gene_mask (Optional[np.ndarray], optional): Subset of genes to consider in error calculation. Defaults to None (all genes used).
        full_vb (bool, optional): Whether to use full variational Bayesian sampling for error estimation. Defaults to False.
        discrete (bool, optional): Whether the data is considered discrete for evaluation. Defaults to False.
        n_sample (int, optional): Number of samples to draw for uncertainty estimation. Defaults to 25.
        seed (int, optional): Random seed for reproducibility of sampling. Defaults to 2022.

    Returns:
        Tuple[float, float, float, float, float, float]: Training and testing error metrics.
    """
    Uhat, Shat = get_pred_velovae(adata, key, full_vb, discrete)
    train_idx, test_idx = adata.uns[f"{key}_train_idx"], adata.uns[f"{key}_test_idx"]
    if gene_mask is None:
        gene_mask = np.ones((adata.n_vars)).astype(bool)
    mse_train_gene = np.ones((adata.n_vars))
    mse_test_gene = np.ones((adata.n_vars))
    mae_train_gene = np.ones((adata.n_vars))
    mae_test_gene = np.ones((adata.n_vars))
    ll_train_gene = np.ones((adata.n_vars))
    ll_test_gene = np.ones((adata.n_vars))
    if discrete:
        U, S = adata.layers["unspliced"].A, adata.layers["spliced"].A
        lu, ls = adata.obs["library_scale_u"].to_numpy(), adata.obs["library_scale_s"].to_numpy()
        Uhat = Uhat*(lu.reshape(-1, 1))
        Shat = Shat*(ls.reshape(-1, 1))
        Uhat = np.clip(Uhat, a_min=1e-2, a_max=None)
        Shat = np.clip(Shat, a_min=1e-2, a_max=None)

        logp_train = np.log(poisson.pmf(U[train_idx][:, gene_mask], Uhat[train_idx][:, gene_mask])+1e-10) \
            + np.log(poisson.pmf(S[train_idx][:, gene_mask], Shat[train_idx][:, gene_mask])+1e-10)
        logp_test = np.log(poisson.pmf(U[test_idx][:, gene_mask], Uhat[test_idx][:, gene_mask])+1e-10) \
            + np.log(poisson.pmf(S[test_idx][:, gene_mask], Shat[test_idx][:, gene_mask])+1e-10)

        # Sample multiple times
        mse_train, mae_train, mse_test, mae_test = 0, 0, 0, 0
        np.random.seed(seed)

        for i in range(n_sample):
            U_sample = poisson.rvs(Uhat)
            S_sample = poisson.rvs(Shat)
            dist_u_train = np.abs(U[train_idx][:, gene_mask]-U_sample[train_idx][:, gene_mask])
            dist_s_train = np.abs(S[train_idx][:, gene_mask]-S_sample[train_idx][:, gene_mask])
            dist_u_test = np.abs(U[test_idx][:, gene_mask]-U_sample[test_idx][:, gene_mask])
            dist_s_test = np.abs(S[test_idx][:, gene_mask]-S_sample[test_idx][:, gene_mask])
            mse_train += np.nanmean(dist_u_train**2+dist_s_train**2)
            mse_test += np.nanmean(dist_u_test**2+dist_s_test**2)
            mae_train += np.nanmean(dist_u_train+dist_s_train)
            mae_test += np.nanmean(dist_u_test+dist_s_test)
            mse_train_gene[gene_mask] = mse_train_gene[gene_mask] + np.nanmean(dist_u_train**2+dist_s_train**2, 0)
            mse_test_gene[gene_mask] = mse_test_gene[gene_mask] + np.nanmean(dist_u_test**2+dist_s_test**2, 0)
            mae_train_gene[gene_mask] = mae_train_gene[gene_mask] + np.nanmean(dist_u_train+dist_s_train, 0)
            mae_test_gene[gene_mask] = mae_test_gene[gene_mask] + np.nanmean(dist_u_test+dist_s_test, 0)
        mse_train /= n_sample
        mse_test /= n_sample
        mae_train /= n_sample
        mae_test /= n_sample
        mse_train_gene = mse_train_gene/n_sample
        mse_test_gene = mse_test_gene/n_sample
        mae_train_gene = mae_train_gene/n_sample
        mae_test_gene = mae_test_gene/n_sample
    else:
        U, S = adata.layers["Mu"], adata.layers["Ms"]
        sigma_u = adata.var[f"{key}_sigma_u"].to_numpy()[gene_mask]
        sigma_s = adata.var[f"{key}_sigma_s"].to_numpy()[gene_mask]

        dist_u_train = np.abs(U[train_idx][:, gene_mask]-Uhat[train_idx][:, gene_mask])
        dist_s_train = np.abs(S[train_idx][:, gene_mask]-Shat[train_idx][:, gene_mask])
        dist_u_test = np.abs(U[test_idx][:, gene_mask]-Uhat[test_idx][:, gene_mask])
        dist_s_test = np.abs(S[test_idx][:, gene_mask]-Shat[test_idx][:, gene_mask])

        logp_train = -dist_u_train**2/(2*sigma_u**2) \
            - dist_s_train**2/(2*sigma_s**2) \
            - np.log(sigma_u) - np.log(sigma_s) - np.log(2*np.pi)
        logp_test = -dist_u_test**2/(2*sigma_u**2) \
            - dist_s_test**2/(2*sigma_s**2) \
            - np.log(sigma_u) - np.log(sigma_s) - np.log(2*np.pi)

        mse_train = np.nanmean(dist_u_train**2+dist_s_train**2)
        mse_test = np.nanmean(dist_u_test**2+dist_s_test**2)
        mae_train = np.nanmean(dist_u_train+dist_s_train)
        mae_test = np.nanmean(dist_u_test+dist_s_test)
        mse_train_gene[gene_mask] = np.nanmean(dist_u_train**2+dist_s_train**2, 0)
        mse_test_gene[gene_mask] = np.nanmean(dist_u_test**2+dist_s_test**2, 0)
        mae_train_gene[gene_mask] = np.nanmean(dist_u_train+dist_s_train, 0)
        mae_test_gene[gene_mask] = np.nanmean(dist_u_test+dist_s_test, 0)

    adata.var[f'{key}_mse_train'] = mse_train_gene
    adata.var[f'{key}_mse_test'] = mse_test_gene
    adata.var[f'{key}_mae_train'] = mae_train_gene
    adata.var[f'{key}_mae_test'] = mae_test_gene
    adata.var[f'{key}_likelihood_train'] = ll_train_gene
    adata.var[f'{key}_likelihood_test'] = ll_test_gene
    ll_train_gene[gene_mask] = np.nanmean(logp_train, 0)
    ll_test_gene[gene_mask] = np.nanmean(logp_test, 0)
    adata.var[f'{key}_likelihood_train'] = ll_train_gene
    adata.var[f'{key}_likelihood_test'] = ll_test_gene
    logp_train = np.nanmean(np.sum(logp_train, 1))
    logp_test = np.nanmean(np.sum(logp_test, 1))

    return mse_train, mse_test, mae_train, mae_test, logp_train, logp_test


def get_pred_velovae_demo(
    adata: AnnData,
    key: str,
    genes: Optional[Iterable[str]] = None,
    full_vb: bool = False,
    discrete: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get prediction from VeloVAE for plotting.

    Args:
        adata (AnnData): Annotated data matrix containing the observed and inferred values.
        key (str): Key in `adata` to identify the velocity or outcome to evaluate.
        gene_mask (Optional[Iterable[str]], optional): Subset of genes to consider in error calculation.
            Defaults to None (all genes used).
        full_vb (bool, optional): Whether to use full variational Bayesian sampling for error estimation.
            Defaults to False.
        discrete (bool, optional): Whether the data is considered discrete for evaluation.
            Defaults to False.
        n_sample (int, optional): Number of samples to draw for uncertainty estimation.
            Defaults to 25.
        seed (int, optional): Random seed for reproducibility of sampling.
            Defaults to 2022.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Predicted unspliced and spliced counts.
    """
    if (f"{key}_uhat" not in adata.layers) or (f"{key}_shat" not in adata.layers):
        alpha = (adata.var[f"{key}_alpha"].to_numpy() if not full_vb else
                 np.exp(adata.var[f"{key}_logmu_alpha"].to_numpy()))
        beta = (adata.var[f"{key}_beta"].to_numpy() if not full_vb else
                np.exp(adata.var[f"{key}_logmu_beta"].to_numpy()))
        gamma = (adata.var[f"{key}_gamma"].to_numpy() if not full_vb else
                 np.exp(adata.var[f"{key}_logmu_gamma"].to_numpy()))
        t = adata.obs[f"{key}_time"].to_numpy()
        scaling = adata.var[f"{key}_scaling"].to_numpy()
        u0, s0 = adata.layers[f"{key}_u0"], adata.layers[f"{key}_s0"]
        t0 = adata.obs[f"{key}_t0"].to_numpy()
        if genes is None:
            rho = adata.layers[f"{key}_rho"]
            Uhat, Shat = pred_su_numpy((t-t0).reshape(-1, 1), u0, s0, alpha, beta, gamma)
            Uhat = Uhat*scaling
        else:
            gene_indices = np.array([np.where(adata.var_names == x)[0][0] for x in genes])
            rho = adata.layers[f"{key}_rho"][:, gene_indices]
            Uhat, Shat = pred_su_numpy(
                (t-t0).reshape(-1, 1),
                u0[:, gene_indices],
                s0[:, gene_indices],
                rho*alpha[gene_indices],
                beta[gene_indices],
                gamma[gene_indices]
            )
            Uhat = Uhat*scaling[gene_indices]
    else:
        if genes is None:
            Uhat, Shat = adata.layers[f"{key}_uhat"], adata.layers[f"{key}_shat"]
        else:
            gene_indices = np.array([np.where(adata.var_names == x)[0][0] for x in genes])
            Uhat, Shat = adata.layers[f"{key}_uhat"][:, gene_indices], adata.layers[f"{key}_shat"][:, gene_indices]
    if discrete:
        lu = adata.obs["library_scale_u"].to_numpy().reshape(-1, 1)
        ls = adata.obs["library_scale_s"].to_numpy().reshape(-1, 1)
        Uhat = Uhat * lu
        Shat = Shat * ls
    return Uhat, Shat


# Branching ODE
def get_pred_brode(adata: AnnData, key: str):
    """
    Get prediction from Branching ODE.
    """
    if (f"{key}_uhat" not in adata.layers) or (f"{key}_shat" not in adata.layers):
        alpha = adata.varm[f"{key}_alpha"]
        beta = adata.varm[f"{key}_beta"]
        gamma = adata.varm[f"{key}_gamma"]
        u0_root, s0_root = adata.varm[f"{key}_u0_root"].T, adata.varm[f"{key}_s0_root"].T
        t_trans = adata.uns[f"{key}_t_trans"]
        scaling = adata.var[f"{key}_scaling"].to_numpy()
        par = np.argmax(adata.uns[f"{key}_w"], 1)
        rate_transition = (adata.varm[f'{key}_rate_transition']
                           if f'{key}_rate_transition' in adata.varm else None)

        t = adata.obs[f"{key}_time"].to_numpy()
        y = adata.obs[f"{key}_label"]

        Uhat, Shat = ode_br_numpy(t.reshape(-1, 1),
                                  y,
                                  par,
                                  alpha=alpha,
                                  beta=beta,
                                  gamma=gamma,
                                  u0_root=u0_root,
                                  s0_root=s0_root,
                                  t_trans=t_trans,
                                  scaling=scaling,
                                  rate_transition=rate_transition)
        Uhat = Uhat*scaling
    else:
        Uhat, Shat = adata.layers[f"{key}_uhat"], adata.layers[f"{key}_shat"]

    return Uhat, Shat


def get_err_brode(
    adata: AnnData, key: str, gene_mask: Optional[np.ndarray] = None
) -> Tuple[float, float, float, float, float, float]:
    """
    Calculate Brode error metrics for evaluating RNA velocity predictions.

    This function computes various error statistics comparing predicted RNA velocity
    against observed data within an AnnData object. It supports applying a mask
    to select specific genes for the evaluation.

    Args:
        adata (AnnData): Annotated data matrix with RNA velocity results.
        key (str): The key in adata layers or obsm where predicted velocities or relevant data are stored.
        gene_mask (Optional[Union[np.ndarray, list, slice]], optional):
            Boolean array or index array to select a subset of genes for error calculation. Defaults to None.

    Returns:
        Tuple[float, float, float, float, float, float]: Six float values representing different error metrics.
    """
    U, S = adata.layers["Mu"], adata.layers["Ms"]
    Uhat, Shat = get_pred_brode(adata, key)
    train_idx, test_idx = adata.uns[f"{key}_train_idx"], adata.uns[f"{key}_test_idx"]

    if gene_mask is None:
        gene_mask = np.ones((adata.n_vars)).astype(bool)
    adata.var[f'{key}_mse_train'] = np.ones((adata.n_vars))*np.nan
    adata.var[f'{key}_mse_test'] = np.ones((adata.n_vars))*np.nan
    adata.var[f'{key}_mae_train'] = np.ones((adata.n_vars))*np.nan
    adata.var[f'{key}_mae_test'] = np.ones((adata.n_vars))*np.nan
    adata.var[f'{key}_ll_train'] = np.ones((adata.n_vars))*np.nan
    adata.var[f'{key}_ll_test'] = np.ones((adata.n_vars))*np.nan

    sigma_u = adata.var[f"{key}_sigma_u"].to_numpy()[gene_mask]
    sigma_s = adata.var[f"{key}_sigma_s"].to_numpy()[gene_mask]
    dist_u_train = np.abs(U[train_idx][:, gene_mask]-Uhat[train_idx][:, gene_mask])
    dist_s_train = np.abs(S[train_idx][:, gene_mask]-Shat[train_idx][:, gene_mask])
    dist_u_test = np.abs(U[test_idx][:, gene_mask]-Uhat[test_idx][:, gene_mask])
    dist_s_test = np.abs(S[test_idx][:, gene_mask]-Shat[test_idx][:, gene_mask])

    logp_train = - dist_u_train**2/(2*sigma_u**2)\
        - dist_s_train**2/(2*sigma_s**2)\
        - np.log(sigma_u) - np.log(sigma_s) - np.log(2*np.pi)
    logp_test = - dist_u_test**2/(2*sigma_u**2)\
        - dist_s_test**2/(2*sigma_s**2)\
        - np.log(sigma_u) - np.log(sigma_s) - np.log(2*np.pi)
    mse_train = np.nanmean(dist_u_train**2+dist_s_train**2)
    mae_train = np.nanmean(dist_u_train+dist_s_train)
    mse_test = np.nanmean(dist_u_test**2+dist_s_test**2)
    mae_test = np.nanmean(dist_u_test+dist_s_test)

    adata.var[f'{key}_mse_train'].values[gene_mask] = np.nanmean(dist_u_train**2+dist_s_train**2, 0)
    adata.var[f'{key}_mse_test'].values[gene_mask] = np.nanmean(dist_u_train+dist_s_train, 0)
    adata.var[f'{key}_mae_train'].values[gene_mask] = np.nanmean(dist_u_test**2+dist_s_test**2, 0)
    adata.var[f'{key}_mae_test'].values[gene_mask] = np.nanmean(dist_u_test+dist_s_test, 0)
    adata.var[f'{key}_ll_train'].values[gene_mask] = np.nanmean(logp_train, 0)
    adata.var[f'{key}_ll_test'].values[gene_mask] = np.nanmean(logp_test, 0)

    logp_train = np.nanmean(np.sum(logp_train, 1))
    logp_test = np.nanmean(np.sum(logp_test, 1))

    return mse_train, mse_test, mae_train, mae_test, logp_train, logp_test


def get_pred_brode_demo(
    adata: AnnData,
    key: str,
    genes: Optional[Iterable[str]] = None,
    N: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions from a Branching Ordinary Differential Equation (ODE) model for visualization.

    Args:
        adata (AnnData): Annotated data object containing gene expression data and model results.
        key (str): Key in `adata` that specifies which Branching ODE model results to use.
        genes (Iterable[str], optional): Subset of genes to include in the prediction.
            If None, all genes will be included.
        N (int, optional): Number of points to sample along the latent time for prediction.
            If None, the default number of points will be used.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Predicted unspliced and spliced counts.
    """
    if isinstance(N, int):
        t_trans = adata.uns[f"{key}_t_trans"]
        t = adata.obs[f"{key}_time"].to_numpy()
        y = adata.obs[f"{key}_label"].to_numpy()  # integer
        par = np.argmax(adata.uns[f"{key}_w"], 1)
        n_type = len(par)
        t_demo = np.concatenate([np.linspace(t_trans[i], t[y == i].max()*1.05, N) for i in range(n_type)])
        y_demo = np.concatenate([(i*np.ones((N))).astype(int) for i in range(n_type)])
        rate_transition = (adata.varm[f'{key}_rate_transition']
                           if f'{key}_rate_transition' in adata.varm else None)

        if genes is None:
            alpha = adata.varm[f"{key}_alpha"].T
            beta = adata.varm[f"{key}_beta"].T
            gamma = adata.varm[f"{key}_gamma"].T
            u0_root, s0_root = adata.varm[f"{key}_u0_root"].T, adata.varm[f"{key}_s0_root"].T
            scaling = adata.var[f"{key}_scaling"].to_numpy()
        else:
            gene_indices = np.array([np.where(adata.var_names == x)[0][0] for x in genes])
            alpha = adata.varm[f"{key}_alpha"][gene_indices].T
            beta = adata.varm[f"{key}_beta"][gene_indices].T
            gamma = adata.varm[f"{key}_gamma"][gene_indices].T
            u0_root = adata.varm[f"{key}_u0_root"][gene_indices].T
            s0_root = adata.varm[f"{key}_s0_root"][gene_indices].T
            scaling = adata.var[f"{key}_scaling"][gene_indices].to_numpy()
            if rate_transition is not None:
                rate_transition = rate_transition[gene_indices]

        Uhat_demo, Shat_demo = ode_br_numpy(
            t_demo.reshape(-1, 1),
            y_demo,
            par,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            t_trans=t_trans,
            u0_root=u0_root,
            s0_root=s0_root,
            scaling=scaling,
            rate_transition=rate_transition
        )

        return t_demo, y_demo, Uhat_demo, Shat_demo
    else:
        if genes is None:
            Uhat, Shat = adata.layers[f"{key}_uhat"], adata.layers[f"{key}_shat"]
        else:
            gene_indices = np.array([np.where(adata.var_names == x)[0][0] for x in genes])
            Uhat, Shat = adata.layers[f"{key}_uhat"][:, gene_indices], adata.layers[f"{key}_shat"][:, gene_indices]
        return Uhat, Shat

# UniTVelo


def get_pred_utv(adata: AnnData, B: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get prediction from UniTVelo.
    """
    t = adata.layers['fit_t']
    o = adata.var['fit_offset'].values
    a0 = adata.var['fit_a'].values
    t0 = adata.var['fit_t'].values
    h0 = adata.var['fit_h'].values
    i = adata.var['fit_intercept'].values
    gamma = adata.var['fit_gamma'].values
    beta = adata.var['fit_beta'].values

    Nb = adata.n_obs//B
    uhat, shat = np.empty(adata.shape), np.empty(adata.shape)
    for i in range(Nb):
        shat[i*B:(i+1)*B] = h0*np.exp(-a0*(t[i*B:(i+1)*B] - t0)**2) + o
        uhat[i*B:(i+1)*B] = (adata.layers['velocity'][i*B:(i+1)*B]+gamma*shat[i*B:(i+1)*B])/beta + i
    if Nb*B < adata.n_obs:
        shat[Nb*B:] = h0*np.exp(-a0*(t[Nb*B:] - t0)**2) + o
        uhat[Nb*B:] = (adata.layers['velocity'][Nb*B:]+gamma*shat[Nb*B:])/beta + i

    return uhat, shat


def get_err_utv(
    adata: AnnData, key: str, gene_mask: Optional[np.ndarray] = None, B: int = 5000
) -> Tuple:
    """
    Calculate performance error metrics for UniTVelo velocity inference results.

    Args:
        adata (AnnData): Annotated data object containing velocity results and associated data.
        key (str): Key within `adata` where velocity results are stored.
        gene_mask (Optional[np.ndarray], optional): Boolean mask to select genes for evaluation. 
            If None, all genes are used. Defaults to None.
        B (int, optional): Number of bootstrap samples for estimating error metrics. Defaults to 5000.

    Returns:
        Tuple: A tuple containing error metrics summarizing the UniTVelo velocity inference accuracy.
    """
    Uhat, Shat = get_pred_utv(adata, B)
    U, S = adata.layers['Mu'], adata.layers['Ms']

    if gene_mask is None:
        gene_mask = np.where(~np.isnan(adata.layers['fit_t'][0]))[0]

    dist_u = np.abs(U[:, gene_mask] - Uhat[:, gene_mask])
    dist_s = np.abs(S[:, gene_mask] - Shat[:, gene_mask])

    # MLE of variance
    var_s = np.var((S[:, gene_mask]-Shat[:, gene_mask]), 0)
    var_u = np.var((U[:, gene_mask]-Uhat[:, gene_mask]), 0)

    mse = np.nanmean(dist_u**2 + dist_s**2)
    mae = np.nanmean(dist_u+dist_s)
    logp = -dist_u**2/(2*var_u)-dist_s**2/(2*var_s) - 0.5*np.log(var_u) - 0.5*np.log(var_s) - np.log(2*np.pi)

    adata.var[f'{key}_mse'] = np.ones((adata.n_vars))*np.nan
    adata.var[f'{key}_mae'] = np.ones((adata.n_vars))*np.nan
    adata.var[f'{key}_likelihood'] = np.ones((adata.n_vars))*np.nan
    adata.var[f'{key}_mse'].values[gene_mask] = np.nanmean(dist_u**2 + dist_s**2, 0)
    adata.var[f'{key}_mae'].values[gene_mask] = np.nanmean(dist_u + dist_s, 0)
    adata.var[f'{key}_likelihood'].values[gene_mask] = np.nanmean(logp, 0)

    return mse, None, mae, None, logp.sum(1).mean(0), None


def get_pred_utv_demo(adata: AnnData, genes: Optional[Iterable[str]] = None, N: int = 100):
    """
    Get prediction from UniTVelo for plotting.
    """
    t = adata.layers['fit_t']

    if genes is None:
        gene_indices = np.array(np.range(adata.n_vars))
    else:
        gene_indices = np.array([np.where(adata.var_names == x)[0][0] for x in genes])

    t_demo = np.linspace(t[:, gene_indices].min(0), t[:, gene_indices].max(0), N)
    o = adata.var['fit_offset'].values[gene_indices]
    a0 = adata.var['fit_a'].values[gene_indices]
    t0 = adata.var['fit_t'].values[gene_indices]
    h0 = adata.var['fit_h'].values[gene_indices]
    i = adata.var['fit_intercept'].values[gene_indices]
    gamma = adata.var['fit_gamma'].values[gene_indices]
    beta = adata.var['fit_beta'].values[gene_indices]

    shat = h0*np.exp(-a0*(t_demo - t0)**2) + o
    vs = shat * (-2*a0*(t_demo - t0))
    uhat = (vs + gamma*shat)/beta + i

    return t_demo, uhat, shat


##########################################################################
# Evaluation utility functions from DeepVelo
# Reference:
# Cui, H., Maan, H., Vladoiu, M.C. et al. DeepVelo: deep learning extends 
# RNA velocity to multi-lineage systems with cell-specific kinetics. 
# Genome Biol 25, 27 (2024).
##########################################################################

def get_neighbor_idx(adata: AnnData, topC: int = 30, topG: int = 20):
    """ DeepVelo helper function """
    Sx_sz = adata.layers["Ms"]
    N_cell, N_gene = Sx_sz.shape

    n_pcas = 30
    pca_ = PCA(
        n_components=n_pcas,
        svd_solver="randomized",
    )
    Sx_sz_pca = pca_.fit_transform(Sx_sz)
    if N_cell < 3000:
        ori_dist = pairwise_distances(Sx_sz_pca, Sx_sz_pca)
        nn_t_idx = np.argsort(ori_dist, axis=1)[:, 1:topC]
    else:
        p = hnswlib.Index(space="l2", dim=n_pcas)
        p.init_index(max_elements=N_cell, ef_construction=200, M=30)
        p.add_items(Sx_sz_pca)
        p.set_ef(max(topC, topG) + 10)
        nn_t_idx = p.knn_query(Sx_sz_pca, k=topC)[0][:, 1:].astype(int)

    return nn_t_idx


def _loss_dv_gene(
    output: torch.Tensor,
    current_state: torch.Tensor,
    idx: torch.LongTensor,
    candidate_states: torch.Tensor,
    n_spliced: int = None,
    l_norm: int = 2,
    *args,
    **kwargs,
):
    """ DeepVelo helper function """
    # Genewise loss function
    batch_size, genes = current_state.shape
    if n_spliced is not None:
        genes = n_spliced
    num_neighbors = idx.shape[1]
    loss = []
    for i in range(num_neighbors):
        ith_neighbors = idx[:, i]
        candidate_state = candidate_states[ith_neighbors, :]  # (batch_size, genes)
        delta = (candidate_state - current_state).detach()  # (batch_size, genes)
        cos_sim = F.cosine_similarity(
            delta[:, :genes], output[:, :genes], dim=1
        )  # (batch_size,)

        # t+1 direction
        candidates = cos_sim.detach() > 0
        if l_norm == 1:
            squared_difference = torch.mean(torch.abs(output - delta), dim=1)
        else:
            squared_difference = torch.mean(torch.pow(output - delta, 2), dim=1)
        squared_difference = squared_difference[candidates]
        loss.append(torch.sum(squared_difference, 0) / len(candidates))

        # t-1 direction, (-output) - delta
        candidates = cos_sim.detach() < 0
        if l_norm == 1:
            squared_difference = torch.mean(torch.pow(output + delta, 1), dim=1)
        else:
            squared_difference = torch.mean(torch.pow(output + delta, 2), dim=1)
        squared_difference = squared_difference[candidates]
        loss.append(torch.sum(squared_difference, 0) / len(candidates))
    loss = torch.stack(loss).mean(0)
    return loss.detach().cpu().numpy()


def _loss_dv(
    output: torch.Tensor,
    current_state: torch.Tensor,
    idx: torch.LongTensor,
    candidate_states: torch.Tensor,
    n_spliced: int = None,
    l_norm: int = 2,
    *args,
    **kwargs,
):
    """ DeepVelo helper function """
    batch_size, genes = current_state.shape
    if n_spliced is not None:
        genes = n_spliced
    num_neighbors = idx.shape[1]
    loss = []
    for i in range(num_neighbors):
        ith_neighbors = idx[:, i]
        candidate_state = candidate_states[ith_neighbors, :]  # (batch_size, genes)
        delta = (candidate_state - current_state).detach()  # (batch_size, genes)
        cos_sim = F.cosine_similarity(
            delta[:, :genes], output[:, :genes], dim=1
        )  # (batch_size,)

        # t+1 direction
        candidates = cos_sim.detach() > 0
        if l_norm == 1:
            squared_difference = torch.mean(torch.abs(output - delta), dim=1)
        else:
            squared_difference = torch.mean(torch.pow(output - delta, 2), dim=1)
        squared_difference = squared_difference[candidates]
        loss.append(torch.sum(squared_difference) / len(candidates))

        # t-1 direction, (-output) - delta
        candidates = cos_sim.detach() < 0
        if l_norm == 1:
            squared_difference = torch.mean(torch.pow(output + delta, 1), dim=1)
        else:
            squared_difference = torch.mean(torch.pow(output + delta, 2), dim=1)
        squared_difference = squared_difference[candidates]
        loss.append(torch.sum(squared_difference) / len(candidates))
    loss = torch.stack(loss).mean()
    return loss.detach().cpu().item()


def get_err_dv(adata: AnnData, key: str, gene_mask: Optional[np.ndarray] = None):
    """
    Get performance metrics from DeepVelo results.
    """
    if gene_mask is None:
        gene_mask = np.array(range(adata.n_vars))
    nn_t_idx = get_neighbor_idx(adata)
    mse_u = _loss_dv(torch.tensor(adata.layers['velocity_unspliced'][:, gene_mask]),
                     torch.tensor(adata.layers['Mu'][:, gene_mask]),
                     torch.tensor(nn_t_idx, dtype=torch.long),
                     torch.tensor(adata.layers['Mu'][:, gene_mask]))
    mse_s = _loss_dv(torch.tensor(adata.layers['velocity'][:, gene_mask]),
                     torch.tensor(adata.layers['Ms'][:, gene_mask]),
                     torch.tensor(nn_t_idx, dtype=torch.long),
                     torch.tensor(adata.layers['Ms'][:, gene_mask]))
    mae_u = _loss_dv(torch.tensor(adata.layers['velocity_unspliced'][:, gene_mask]),
                     torch.tensor(adata.layers['Mu'][:, gene_mask]),
                     torch.tensor(nn_t_idx, dtype=torch.long),
                     torch.tensor(adata.layers['Mu'][:, gene_mask]),
                     l=1)
    mae_s = _loss_dv(torch.tensor(adata.layers['velocity'][:, gene_mask]),
                     torch.tensor(adata.layers['Ms'][:, gene_mask]),
                     torch.tensor(nn_t_idx, dtype=torch.long),
                     torch.tensor(adata.layers['Ms'][:, gene_mask]),
                     l=1)
    adata.var[f'{key}_mse'] = np.ones((adata.n_vars))*np.nan
    mse_u_gene = _loss_dv_gene(torch.tensor(adata.layers['velocity_unspliced'][:, gene_mask]),
                               torch.tensor(adata.layers['Mu'][:, gene_mask]),
                               torch.tensor(nn_t_idx, dtype=torch.long),
                               torch.tensor(adata.layers['Mu'][:, gene_mask]))
    mse_s_gene = _loss_dv_gene(torch.tensor(adata.layers['velocity'][:, gene_mask]),
                               torch.tensor(adata.layers['Ms'][:, gene_mask]),
                               torch.tensor(nn_t_idx, dtype=torch.long),
                               torch.tensor(adata.layers['Ms'][:, gene_mask]))
    adata.var[f'{key}_mse'].values[gene_mask] = mse_u_gene + mse_s_gene

    adata.var[f'{key}_mae'] = np.ones((adata.n_vars))*np.nan
    mae_u_gene = _loss_dv_gene(torch.tensor(adata.layers['velocity_unspliced'][:, gene_mask]),
                               torch.tensor(adata.layers['Mu'][:, gene_mask]),
                               torch.tensor(nn_t_idx, dtype=torch.long),
                               torch.tensor(adata.layers['Mu'][:, gene_mask]),
                               l=1)
    mae_s_gene = _loss_dv_gene(torch.tensor(adata.layers['velocity'][:, gene_mask]),
                               torch.tensor(adata.layers['Ms'][:, gene_mask]),
                               torch.tensor(nn_t_idx, dtype=torch.long),
                               torch.tensor(adata.layers['Ms'][:, gene_mask]),
                               l=1)
    adata.var[f'{key}_mae'].values[gene_mask] = mae_u_gene + mae_s_gene

    return mse_u+mse_s, None, mae_u+mae_s, None, None, None


##########################################################################
# Evaluation utility functions from PyroVelocity
# Reference:
# Qin, Q., Bingham, E., La Manno, G., Langenau, D. M., & Pinello, L. (2022).
# Pyro-Velocity: Probabilistic RNA Velocity inference from single-cell data.
# bioRxiv.
##########################################################################
def get_err_pv(adata: AnnData, key: str, gene_mask: Optional[np.ndarray] = None, discrete: bool = True):
    """
    Get performance metrics from PyroVelocity results.
    """
    if 'err' in adata.uns:
        err_dic = adata.uns['err']
        return (err_dic['MSE Train'], err_dic['MSE Test'],
                err_dic['MAE Train'], err_dic['MAE Test'],
                err_dic['LL Train'], err_dic['LL Test'])
    else:
        train_idx, test_idx = adata.uns[f"{key}_train_idx"], adata.uns[f"{key}_test_idx"]
        adata.var[f'{key}_mse_train'] = np.ones((adata.n_vars))*np.nan
        adata.var[f'{key}_mse_test'] = np.ones((adata.n_vars))*np.nan
        adata.var[f'{key}_mae_train'] = np.ones((adata.n_vars))*np.nan
        adata.var[f'{key}_mae_test'] = np.ones((adata.n_vars))*np.nan
        adata.var[f'{key}_likelihood_train'] = np.ones((adata.n_vars))*np.nan
        adata.var[f'{key}_likelihood_test'] = np.ones((adata.n_vars))*np.nan
        if discrete:
            U, S = adata.layers['unspliced'].A, adata.layers['spliced'].A
            Mu_u, Mu_s = adata.layers[f'{key}_ut'], adata.layers[f'{key}_st']
            Uhat, Shat = adata.layers[f'{key}_u'], adata.layers[f'{key}_s']
        else:
            U, S = adata.layers['Mu'], adata.layers['Ms']
            Uhat, Shat = adata.layers[f'{key}_u'], adata.layers[f'{key}_s']
            # MLE of variance
            var_s_train = np.var((S[train_idx]-Shat[train_idx]), 0)
            var_s_test = np.var((S[test_idx]-Shat[test_idx]), 0)
            var_u_train = np.var((U[train_idx]-Uhat[train_idx]), 0)
            var_u_test = np.var((U[test_idx]-Uhat[test_idx]), 0)

        if gene_mask is None:
            gene_mask = np.ones((adata.n_vars)).astype(bool)

        dist_u_train = np.abs(U[train_idx][:, gene_mask]-Uhat[train_idx][:, gene_mask])
        dist_s_train = np.abs(S[train_idx][:, gene_mask]-Shat[train_idx][:, gene_mask])
        dist_u_test = np.abs(U[test_idx][:, gene_mask]-Uhat[test_idx][:, gene_mask])
        dist_s_test = np.abs(S[test_idx][:, gene_mask]-Shat[test_idx][:, gene_mask])

        if discrete:
            logp_train = np.log(poisson.pmf(U[train_idx], Mu_u[train_idx])+1e-10)\
                + np.log(poisson.pmf(S[train_idx], Mu_s[train_idx])+1e-10)
            logp_test = np.log(poisson.pmf(U[test_idx], Mu_u[test_idx])+1e-10)\
                + np.log(poisson.pmf(S[test_idx], Mu_s[test_idx])+1e-10)
        else:
            logp_train = -dist_u_train**2/(2*var_u_train)-dist_s_train**2/(2*var_s_train)\
                - 0.5*np.log(var_u_train) - 0.5*np.log(var_s_train) - np.log(2*np.pi)
            logp_test = -dist_u_test**2/(2*var_u_test)-dist_s_test**2/(2*var_s_test)\
                - 0.5*np.log(var_u_test) - 0.5*np.log(var_s_test) - np.log(2*np.pi)

        mse_train = np.nanmean(dist_u_train**2+dist_s_train**2)
        mse_test = np.nanmean(dist_u_test**2+dist_s_test**2)
        mae_train = np.nanmean(dist_u_train+dist_s_train)
        mae_test = np.nanmean(dist_u_test+dist_s_test)
        adata.var[f'{key}_mse_train'].values[gene_mask] = np.nanmean(dist_u_train**2+dist_s_train**2, 0)
        adata.var[f'{key}_mse_test'].values[gene_mask] = np.nanmean(dist_u_test**2+dist_s_test**2, 0)
        adata.var[f'{key}_mae_train'].values[gene_mask] = np.nanmean(dist_u_train+dist_s_train, 0)
        adata.var[f'{key}_mae_test'].values[gene_mask] = np.nanmean(dist_u_test+dist_s_test, 0)
        adata.var[f'{key}_likelihood_train'].values[gene_mask] = np.nanmean(logp_train, 0)
        adata.var[f'{key}_likelihood_test'].values[gene_mask] = np.nanmean(logp_test, 0)
        logp_train = np.nanmean(logp_train.sum(1).mean())
        logp_test = np.nanmean(logp_test.sum(1).mean())

    return mse_train, mse_test, mae_train, mae_test, logp_train, logp_test


##########################################################################
# Evaluation utility functions from VeloVI
# Reference:
# Gayoso, Adam, et al. "Deep generative modeling of transcriptional dynamics
# for RNA velocity analysis in single cells." bioRxiv (2022).
##########################################################################
def get_err_velovi(adata: AnnData, key: str, gene_mask: Optional[np.ndarray] = None):
    """
    Get performance metrics from VeloVI results.
    """
    if gene_mask is None:
        gene_mask = np.ones((adata.n_vars)).astype(bool)
    U, S = adata.layers['Mu'][:, gene_mask], adata.layers['Ms'][:, gene_mask]
    Uhat, Shat = adata.layers[f'{key}_uhat'][:, gene_mask], adata.layers[f'{key}_shat'][:, gene_mask]

    try:
        adata.var[f'{key}_mse_train'] = np.ones((adata.n_vars))*np.nan
        adata.var[f'{key}_mse_test'] = np.ones((adata.n_vars))*np.nan
        adata.var[f'{key}_mae_train'] = np.ones((adata.n_vars))*np.nan
        adata.var[f'{key}_mae_test'] = np.ones((adata.n_vars))*np.nan
        train_idx = adata.uns[f'{key}_train_idx']
        test_idx = adata.uns[f'{key}_test_idx']
        dist_u_train = np.abs(U[train_idx]-Uhat[train_idx])
        dist_s_train = np.abs(S[train_idx]-Shat[train_idx])
        dist_u_test = np.abs(U[test_idx]-Uhat[test_idx])
        dist_s_test = np.abs(S[test_idx]-Shat[test_idx])

        mse_train = np.mean(dist_u_train**2+dist_s_train**2)
        mse_test = np.mean(dist_u_test**2+dist_s_test**2)
        mae_train = np.mean(dist_u_train+dist_s_train)
        mae_test = np.mean(dist_u_test+dist_s_test)
        adata.var[f'{key}_mse_train'].values[gene_mask] = np.mean(dist_u_train**2+dist_s_train**2, 0)
        adata.var[f'{key}_mse_test'].values[gene_mask] = np.mean(dist_u_test**2+dist_s_test**2, 0)
        adata.var[f'{key}_mae_train'].values[gene_mask] = np.mean(dist_u_train+dist_s_train, 0)
        adata.var[f'{key}_mae_test'].values[gene_mask] = np.mean(dist_u_test+dist_s_test, 0)
        logp_train = np.mean(adata.obs[f'{key}_likelihood'].to_numpy()[train_idx])
        logp_test = np.mean(adata.obs[f'{key}_likelihood'].to_numpy()[test_idx])
    except KeyError:
        dist_u = np.abs(U-Uhat)
        dist_s = np.abs(S-Shat)
        mse = np.mean(dist_u**2+dist_s**2)
        mae = np.mean(dist_u+dist_s)
        logp = np.mean(adata.obs[f'{key}_likelihood'].to_numpy())

        return mse, None, mae, None, logp, None

    return mse_train, mse_test, mae_train, mae_test, logp_train, logp_test


##########################################################################
# Evaluation Metrics adapted from UniTVelo
# Reference:
# Gao, M., Qiao, C. & Huang, Y. UniTVelo: temporally unified RNA velocity
# reinforces single-cell trajectory inference. Nat Commun 13, 6586 (2022).
# https://doi.org/10.1038/s41467-022-34188-7
##########################################################################


def summary_scores(all_scores: Dict[str, Iterable]):
    """
    Summarize group scores.

    Args:
        all_scores (Dict[str, Iterable]): A dictionary where each key is a group name and the value is an iterable 
            of scores for individual cells in that group.

    Returns:
        tuple: A tuple containing:
            dict[str, float]: Group-wise aggregated mean scores.
            float: Overall aggregated mean score across all groups.
    """
    sep_scores = {k: np.mean(s) for k, s in all_scores.items() if s}
    overal_agg = np.mean([s for k, s in sep_scores.items() if s])
    return sep_scores, overal_agg


def keep_type(adata: AnnData, nodes: Iterable[int], target: str, k_cluster: str):
    """
    Select cells of targeted type

    Args:
        adata (anndata.AnnData):
            Anndata object.
        nodes (Iterable[int]):
            Indexes for cells
        target (str):
            Cluster name.
        k_cluster (str):
            Cluster key in adata.obs dataframe

    Returns:
        list:
             Selected cells.
    """

    return nodes[adata.obs[k_cluster][nodes].values == target]


def remove_type(adata: AnnData, nodes: np.ndarray, target: str, k_cluster: str):
    # Exclude cells of targeted type
    return nodes[adata.obs[k_cluster][nodes].values != target]


def cross_boundary_correctness(
    adata: AnnData,
    k_cluster: str,
    k_velocity: str,
    cluster_edges: List[Tuple[str]],
    return_raw: bool = False,
    x_emb: str = "X_umap",
    gene_mask: Optional[np.ndarray] = None
) -> Tuple[Dict, float]:
    """Cross-Boundary Direction Correctness Score (A->B)
    Args:
        adata (:class:`anndata.AnnData`):
            Anndata object.
        k_cluster (str):
            Key to the cluster column in adata.obs.
        k_velocity (str):
            Key to the velocity matrix in adata.obsm.
        cluster_edges (list[tuple[str]]):
            Pairs of clusters has transition direction A->B
        return_raw (bool, optional):
            Return aggregated or raw scores. Defaults to False.
        x_emb (str, optional):
            Key to x embedding for visualization or a count matrix in adata.layers.
            Defaults to "X_umap".
        gene_mask (:class:`numpy.ndarray`, optional):
            Boolean array to filter out non-velocity genes. Defaults to None.
    Returns:
        tuple:
            - dict: all_scores indexed by cluster_edges or mean scores indexed by cluster_edges
            - float: averaged score over all cells
    """
    scores = {}
    all_scores = {}
    x_emb_name = x_emb
    if x_emb in adata.obsm:
        x_emb = adata.obsm[x_emb]
        if x_emb_name == "X_umap":
            v_emb = adata.obsm['{}_umap'.format(k_velocity)]
        else:
            v_emb = adata.obsm[[key for key in adata.obsm if key.startswith(k_velocity)][0]]
    else:
        x_emb = adata.layers[x_emb]
        v_emb = adata.layers[k_velocity]
        if gene_mask is None:
            gene_mask = ~np.isnan(v_emb[0])
        x_emb = x_emb[:, gene_mask]
        v_emb = v_emb[:, gene_mask]

    # Get the connectivity matrix
    connectivities = adata.obsp[adata.uns['neighbors']['connectivities_key']]
    
    # Convert to CSR format if it's not already
    if not isinstance(connectivities, csr_matrix):
        connectivities = connectivities.tocsr()

    def get_neighbors(idx):
        return connectivities[idx].indices

    for u, v in cluster_edges:
        sel = adata.obs[k_cluster] == u
        sel_indices = np.where(sel)[0]
        x_points = x_emb[sel]
        x_velocities = v_emb[sel]
        type_score = []
        for idx, x_pos, x_vel in zip(sel_indices, x_points, x_velocities):
            nbs = get_neighbors(idx)
            nodes = keep_type(adata, nbs, v, k_cluster)
            if len(nodes) == 0:
                continue
            position_dif = x_emb[nodes] - x_pos
            dir_scores = cosine_similarity(position_dif, x_vel.reshape(1, -1)).flatten()
            type_score.append(np.nanmean(dir_scores))
        if len(type_score) == 0:
            print(f'Warning: cell type transition pair ({u},{v}) does not exist in the KNN graph. Ignored.')
        else:
            scores[f'{u} -> {v}'] = np.nanmean(type_score)
            all_scores[f'{u} -> {v}'] = type_score
    if return_raw:
        return all_scores
    return scores, np.mean([sc for sc in scores.values()])


def _cos_sim_sample(v_sample: np.ndarray, v_neighbors: np.ndarray, dt=None):
    res = cosine_similarity(v_neighbors, v_sample.reshape(1, -1)).flatten()
    if dt is not None:
        res = -np.abs(res) * (dt < 0) + res * (dt >= 0)
    return res


def gen_cross_boundary_correctness(
    adata: AnnData,
    k_cluster: str,
    k_velocity: str,
    cluster_edges: List[Tuple[str]],
    tkey: Optional[str] = None,
    k_hop: int = 5,
    dir_test: bool = False,
    x_emb: str = "X_umap",
    gene_mask: Optional[np.ndarray] = None,
    n_prune: int = 30,
    random_state: int = 2022
) -> Tuple[Dict, np.ndarray]:
    """
    Generalized Cross-Boundary Direction Correctness Score (A->B).

    Args:
        adata (anndata.AnnData): Anndata object.
        k_cluster (str): Key to the cluster column in adata.obs.
        k_velocity (str): Key to the velocity matrix in adata.obsm.
        cluster_edges (list[tuple[str]]): Pairs of clusters that have transition direction A->B.
        tkey (str, optional): Key to the cell time in adata.obs. Defaults to None.
        k_hop (int, optional): Number of steps to consider.
            CBDir will be computed for 1 to k-step neighbors. Defaults to 5.
        dir_test (bool, optional): Whether to subtract CBDir of random walk from CBDir of desired direction.
            Defaults to False.
        x_emb (str, optional): Low dimensional embedding in adata.obsm or original count matrix in adata.layers.
            Defaults to "X_umap".
        gene_mask (np.ndarray, optional): Boolean array to filter out non-velocity genes. Defaults to None.
        n_prune (int, optional): Maximum number of neighbors to keep.
            This is necessary because the number of neighbors grows exponentially. Defaults to 30.
        random_state (int, optional): Seed for random walk sampling. Defaults to 2022.

    Returns:
        tuple: A tuple containing:
            - dict: All scores indexed by cluster_edges or mean scores indexed by cluster_edges.
            - np.ndarray: Average score over all cells for all step numbers.
    """
      # Use k-hop neighbors
    scores = {}
    x_emb_name = x_emb
    if x_emb in adata.obsm:
        x_emb = adata.obsm[x_emb]
        if x_emb_name == "X_umap":
            v_emb = adata.obsm['{}_umap'.format(k_velocity)]
        else:
            v_emb = adata.obsm[[key for key in adata.obsm if key.startswith(k_velocity)][0]]
    else:
        x_emb = adata.layers[x_emb]
        v_emb = adata.layers[k_velocity]
        if gene_mask is None:
            gene_mask = ~np.isnan(v_emb[0])
        x_emb = x_emb[:, gene_mask]
        v_emb = v_emb[:, gene_mask]
    
    
    # Get the connectivity matrix
    connectivities = adata.obsp[adata.uns['neighbors']['connectivities_key']]
    
    # Convert to CSR format if it's not already
    if not isinstance(connectivities, csr_matrix):
        connectivities = connectivities.tocsr()

    def get_neighbors(idx):
        return connectivities[idx].indices

    t = adata.obs[tkey].to_numpy()
    np.random.seed(random_state)
    for u, v in cluster_edges:
        sel = adata.obs[k_cluster] == u
        sel_indices = np.where(sel)[0]
        x_points = x_emb[sel]
        x_velocities = v_emb[sel]
        t_points = t[sel]

        type_score = [[] for i in range(k_hop)]
        type_score_null = [[] for i in range(k_hop)]
        for idx, x_pos, x_vel, t_i in zip(sel_indices, x_points, x_velocities, t_points):
            all_nodes = get_neighbors(idx)
            nodes = keep_type(adata, all_nodes, v, k_cluster)
            if len(nodes) == 0:
                continue
            position_dif = x_emb[nodes] - x_pos
            dt = t[nodes] - t_i
            dir_scores = _cos_sim_sample(x_vel, position_dif, dt)

            nodes_null = all_nodes if len(all_nodes) < n_prune else np.random.choice(all_nodes, n_prune)
            position_dif_null = x_emb[nodes_null] - x_pos
            dt_null = t[nodes_null] - t_i
            dir_scores_null = _cos_sim_sample(x_vel, position_dif_null, dt_null)

            # save 1-hop results
            type_score[0].append(np.nanmean(dir_scores))
            type_score_null[0].append(np.nanmean(dir_scores_null))
            # deal with k-hop neighbors when k > 1
            for k in range(k_hop-1):
                nodes = np.unique(np.concatenate([get_neighbors(n) for n in nodes]))
                nodes = keep_type(adata, nodes, v, k_cluster)

                position_dif = x_emb[nodes] - x_pos
                dt = t[nodes] - t_i
                dir_scores = _cos_sim_sample(x_vel, position_dif, dt)

                if len(nodes) > n_prune:
                    idx_sort = np.argsort(dir_scores)
                    nodes = nodes[idx_sort[-n_prune:]]
                    dir_scores = dir_scores[idx_sort[-n_prune:]]
                type_score[k+1].append(np.nanmean(dir_scores))
            # Compute the same k-hop metric for neighbors not in the descent v
            if dir_test and len(nodes_null) > 0:
                for k in range(k_hop-1):
                    nodes_null = np.unique(np.concatenate([get_neighbors(n) for n in nodes_null]))
                    nodes_null = nodes_null if len(nodes_null) < n_prune else np.random.choice(nodes_null, n_prune)
                    position_dif_null = x_emb[nodes_null] - x_pos
                    dt_null = t[nodes_null] - t_i
                    dir_scores_null = _cos_sim_sample(x_vel, position_dif_null, dt_null)
                    if len(nodes_null) > n_prune:
                        idx_sort = np.argsort(dir_scores_null)
                        nodes_null = nodes_null[idx_sort[-n_prune:]]
                        dir_scores_null = dir_scores_null[idx_sort[-n_prune:]]
                    type_score_null[k+1].append(np.nanmean(dir_scores_null))
        mean_type_score = np.array([np.nanmean(type_score[i]) for i in range(k_hop)])
        mean_type_score_null = np.array([np.nanmean(type_score_null[i]) for i in range(k_hop)])
        mean_type_score_null[np.isnan(mean_type_score_null)] = 0.0
        scores[f'{u} -> {v}'] = (mean_type_score - mean_type_score_null if dir_test else mean_type_score)

    return scores, np.mean(np.stack([sc for sc in scores.values()]), 0)


def gen_cross_boundary_correctness_test(
    adata,
    k_cluster,
    k_velocity,
    cluster_edges,
    tkey,
    k_hop=5,
    x_emb="X_umap",
    gene_mask=None,
    n_prune=30,
    random_state=2022
) -> Tuple[Dict, float, Dict, float]:
    """
    Perform Mann-Whitney U Test of RNA velocity.

    Args:
        adata (anndata.AnnData): Anndata object.
        k_cluster (str): Key to the cluster column in adata.obs.
        k_velocity (str): Key to the velocity matrix in adata.obsm.
        cluster_edges (list[tuple[str]]): Pairs of clusters with transition direction A->B.
        tkey (str): Key to the cell time in adata.obs.
        k_hop (int, optional): Number of steps to consider.
            CBDir will be computed for 1 to k-step neighbors. Defaults to 5.
        x_emb (str, optional): Low dimensional embedding in adata.obsm or original count matrix in adata.layers.
            Defaults to "X_umap".
        gene_mask (numpy.ndarray, optional): Boolean array to filter out non-velocity genes. Defaults to None.
        n_prune (int, optional):
            Maximum number of neighbors to keep; necessary because number of neighbors grows exponentially. Defaults to 30.
        random_state (int, optional): Seed for random walk sampling. Defaults to 2022.

    Returns:
        Tuple containing:
            dict: Proportion of cells with correct velocity flow (velocity accuracy) indexed by cluster_edges.
            numpy.ndarray: Average velocity accuracy over all cells for all step numbers.
            dict: Mann-Whitney U test statistics indexed by cluster_edges.
            numpy.ndarray: Average Mann-Whitney U test statistics over all cells for all step numbers.
    """
    # Use k-hop neighbors
    test_stats, accuracy = {}, {}
    x_emb_name = x_emb
    if x_emb in adata.obsm:
        x_emb = adata.obsm[x_emb]
        if x_emb_name == "X_umap":
            v_emb = adata.obsm['{}_umap'.format(k_velocity)]
        else:
            v_emb = adata.obsm[[key for key in adata.obsm if key.startswith(k_velocity)][0]]
    else:
        x_emb = adata.layers[x_emb]
        v_emb = adata.layers[k_velocity]
        if gene_mask is None:
            gene_mask = ~np.isnan(v_emb[0])
        x_emb = x_emb[:, gene_mask]
        v_emb = v_emb[:, gene_mask]

    # Get the connectivity matrix
    connectivities = adata.obsp[adata.uns['neighbors']['connectivities_key']]
    
    # Convert to CSR format if it's not already
    if not isinstance(connectivities, csr_matrix):
        connectivities = connectivities.tocsr()

    def get_neighbors(idx):
        return connectivities[idx].indices

    t = adata.obs[tkey].to_numpy()
    np.random.seed(random_state)
    for u, v in cluster_edges:
        sel = adata.obs[k_cluster] == u
        sel_indices = np.where(sel)[0]
        x_points = x_emb[sel]
        x_velocities = v_emb[sel]
        t_points = t[sel]

        type_stats = [[] for i in range(k_hop)]
        type_pval = [[] for i in range(k_hop)]
        for idx, x_pos, x_vel, t_i in zip(sel_indices, x_points, x_velocities, t_points):
            all_nodes = get_neighbors(idx)
            nodes = keep_type(adata, all_nodes, v, k_cluster)
            if len(nodes) < 2:
                continue
            position_dif = x_emb[nodes] - x_pos
            dt = t[nodes] - t_i
            dir_scores = _cos_sim_sample(x_vel, position_dif, dt)

            nodes_null = all_nodes if len(all_nodes) < n_prune else np.random.choice(all_nodes, n_prune)
            position_dif_null = x_emb[nodes_null] - x_pos
            dt_null = t[nodes_null] - t_i
            dir_scores_null = _cos_sim_sample(x_vel, position_dif_null, dt_null)
            # Perform Mann-Whitney U test
            res = mannwhitneyu(dir_scores, dir_scores_null, alternative='greater')
            type_stats[0].append(res[0])
            type_pval[0].append(res[1])
            # deal with k-hop neighbors when k > 1
            for k in range(k_hop-1):
                nodes = np.unique(np.concatenate([get_neighbors(n) for n in nodes]))
                nodes = keep_type(adata, nodes, v, k_cluster)
                if len(nodes) < 2:
                    continue

                position_dif = x_emb[nodes] - x_pos
                dt = t[nodes] - t_i
                dir_scores = _cos_sim_sample(x_vel, position_dif, dt)

                # Compute the same k-hop metric for neighbors not in the descent v
                nodes_null = np.unique(np.concatenate([get_neighbors(n) for n in nodes_null]))
                nodes_null = nodes_null if len(nodes_null) < n_prune else np.random.choice(nodes_null, n_prune)
                if len(nodes_null) < 2:
                    continue
                position_dif_null = x_emb[nodes_null] - x_pos
                dt_null = t[nodes_null] - t_i
                dir_scores_null = _cos_sim_sample(x_vel, position_dif_null, dt_null)

                if len(nodes) > n_prune:
                    idx_sort = np.argsort(dir_scores)
                    nodes = nodes[idx_sort[-n_prune:]]
                    dir_scores = dir_scores[idx_sort[-n_prune:]]

                if len(nodes_null) > n_prune:
                    idx_sort = np.argsort(dir_scores_null)
                    nodes_null = nodes_null[idx_sort[-n_prune:]]
                    dir_scores_null = dir_scores_null[idx_sort[-n_prune:]]

                # Perform Mann-Whitney U test
                res = mannwhitneyu(dir_scores, dir_scores_null, alternative='greater')
                type_stats[k+1].append(res[0])
                type_pval[k+1].append(res[1])

        test_stats[f'{u} -> {v}'] = [np.nanmean(type_stats[k]) for k in range(k_hop)]
        # check whether p values are less than 0.05
        accuracy[f'{u} -> {v}'] = [np.nanmean(np.array(type_pval[k]) < 0.05) for k in range(k_hop)]
    return (accuracy, np.nanmean(np.stack([p for p in accuracy.values()]), 0),
            test_stats, np.nanmean(np.stack([sc for sc in test_stats.values()]), 0))


def _combine_scores(scores: Dict[str, float], cell_types: Iterable[str]):
    # scores contains flow from any cell type to other cell types
    # This function determines the flow direction for each pair
    # of cell types
    scores_combined = {}
    for i in range(len(cell_types)):
        u = cell_types[i]
        for j in range(i):
            v = cell_types[j]
            if not (f'{u} -> {v}' in scores and f'{v} -> {u}' in scores):
                continue
            if scores[f'{u} -> {v}'] > scores[f'{v} -> {u}']:
                scores_combined[f'{u} -> {v}'] = scores[f'{u} -> {v}']
            else:
                scores_combined[f'{v} -> {u}'] = scores[f'{v} -> {u}']
    return scores_combined


def calibrated_cross_boundary_correctness(
    adata: AnnData,
    k_cluster: str,
    k_velocity: str,
    k_time: str,
    cluster_edges: Optional[List[Tuple[str]]]=None,
    return_raw: bool = False,
    sum_up: bool = False,
    x_emb: str = "X_umap",
    gene_mask: Optional[np.ndarray] = None,
    k_std_t: Optional[str] = None
) -> Tuple[Dict, float, Dict, float]:
    """Calibrated Cross-Boundary Direction Correctness Score (A->B).

    Args:
        adata (anndata.AnnData): Anndata object.
        k_cluster (str): Key to the cluster column in adata.obs DataFrame.
        k_velocity (str): Key to the velocity matrix in adata.obsm.
        k_time (str): Key to the cell time in adata.obs.
        cluster_edges (List[Tuple[str]], optional): Pairs of clusters with transition direction A->B. Defaults to None.
        return_raw (bool, optional): Return aggregated or raw scores. Defaults to False.
        sum_up (bool, optional): Whether to sum or take the mean (default). Defaults to False.
        x_emb (str, optional): Key to x embedding for visualization. Defaults to "X_umap".
        gene_mask (np.ndarray, optional): Boolean array to filter out non-velocity genes. Defaults to None.
        k_std_t (str, optional): Key to time standard deviation. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - dict: All scores indexed by cluster_edges or mean scores indexed by cluster_edges.
            - float: Averaged score over all cells.
            - dict: Time score (proportion of cells with correct time order in a cell type transition).
            - float: Averaged time score.
    """
    scores = {}
    all_scores = {}
    p_fw = {}

    eval_all = cluster_edges is None
    cell_types = np.unique(adata.obs[k_cluster].to_numpy())
    if eval_all:
        cluster_edges = [(u, v) for v in cell_types for u in cell_types]
    x_emb_name = x_emb
    if x_emb in adata.obsm:
        x_emb = adata.obsm[x_emb]
        if x_emb_name == "X_umap":
            v_emb = adata.obsm['{}_umap'.format(k_velocity)]
        else:
            v_emb = adata.obsm[[key for key in adata.obsm if key.startswith(k_velocity)][0]]
    else:
        x_emb = adata.layers[x_emb]
        v_emb = adata.layers[k_velocity]
        if gene_mask is None:
            gene_mask = ~np.isnan(v_emb[0])
        x_emb = x_emb[:, gene_mask]
        v_emb = v_emb[:, gene_mask]
    t = adata.obs[k_time].to_numpy()
    std_t = None if k_std_t is None else adata.obs[k_std_t].to_numpy()

    # Get the connectivity matrix
    connectivities = adata.obsp[adata.uns['neighbors']['connectivities_key']]
    
    # Convert to CSR format if it's not already
    if not isinstance(connectivities, csr_matrix):
        connectivities = connectivities.tocsr()

    def get_neighbors(idx):
        return connectivities[idx].indices

    for u, v in cluster_edges:
        n1, n2 = 0, 0
        sel = adata.obs[k_cluster] == u
        sel_indices = np.where(sel)[0]
        x_points = x_emb[sel]
        x_velocities = v_emb[sel]

        type_score = []
        for idx, x_pos, x_vel in zip(sel_indices, x_points, x_velocities):
            nbs = get_neighbors(idx)
            nodes = keep_type(adata, nbs, v, k_cluster)
            if len(nodes) == 0:
                continue
            if std_t is None:
                p1 = t[nodes] >= t[idx]
                p2 = 1 - p1
            else:
                p1 = norm.cdf(np.zeros((len(nodes))),
                              loc=t[idx]-t[nodes],
                              scale=np.sqrt(std_t[nodes]**2+std_t[idx]**2))
                p2 = 1 - p1
            n1 += np.sum(p1)
            n2 += np.sum(p2)

            position_dif = (x_emb[nodes] - x_pos) * (np.sign(t[nodes] - t[idx]).reshape(-1, 1))
            dir_scores = cosine_similarity(position_dif, x_vel.reshape(1, -1)).flatten()
            type_score.extend(dir_scores)

        if len(type_score) == 0:
            # print(f'Warning: cell type transition pair ({u},{v}) does not exist in the KNN graph. Ignored.')
            pass
        else:
            scores[f'{u} -> {v}'] = np.nansum(type_score) if sum_up else np.nanmean(type_score)
            all_scores[f'{u} -> {v}'] = type_score
            p_fw[f'{u} -> {v}'] = n1 / (n1 + n2)

    if return_raw:
        return all_scores
    scores_combined = _combine_scores(scores, cell_types) if eval_all else scores
    return scores, np.mean([sc for sc in scores_combined.values()]), p_fw, np.mean([p for p in p_fw.values()])


def _encode_type(cell_types_raw: Iterable[str]):
    #######################################################################
    # Use integer to encode the cell types
    # Each cell type has one unique integer label.
    #######################################################################

    # Map cell types to integers
    label_dic = {}
    label_dic_rev = {}
    for i, type_ in enumerate(cell_types_raw):
        label_dic[type_] = i
        label_dic_rev[i] = type_

    return label_dic, label_dic_rev


def _edge2adj(cell_types: Iterable[str], cluster_edges: List[Tuple[str]]):
    label_dic, label_dic_rev = _encode_type(cell_types)
    adj_mtx = np.zeros((len(cell_types), len(cell_types)))
    for u, v in cluster_edges:
        i, j = label_dic[u], label_dic[v]
        adj_mtx[i, j] = 1
        # child of child
        child_v = np.where(adj_mtx[j] > 0)[0]
        if len(child_v) > 0:
            adj_mtx[i, child_v] = 1
        # parent of parent
        par_u = np.where(adj_mtx[:, i] > 0)[0]
        if len(par_u) > 0:
            adj_mtx[par_u, j] = 1
    return label_dic, label_dic_rev, adj_mtx


def time_score(
    adata: AnnData, tkey: str, cluster_key: str, cluster_edges: List[Tuple[str]]
) -> Tuple[Dict, float]:
    """
    Calculates the Time Accuracy Score, defined as the average proportion of descendant cells 
    that appear after their progenitor cells.

    Args:
        adata (AnnData): AnnData object.
        tkey (str): Key for the inferred cell time.
        cluster_key (str): Key for cell type annotations.
        cluster_edges (List[Tuple[str]]): Pairs of clusters representing transition directions A->B.

    Returns:
        tuple: A tuple containing:
            - dict: Time Accuracy Score for each transition pair.
            - float: Mean Time Accuracy Score.
    """
    # Compute time inference accuracy based on
    # progenitor-descendant pairs
    t = adata.obs[tkey].to_numpy()
    cell_labels = adata.obs[cluster_key]
    cell_types = np.unique(cell_labels)
    label_dic, label_dic_rev, adj_mtx = _edge2adj(cell_types, cluster_edges)
    cell_labels_int = np.array([label_dic[x] for x in cell_labels])
    tscore = {}
    for i in range(adj_mtx.shape[0]):
        children = np.where(adj_mtx[i] > 0)[0]
        for j in children:
            p = np.mean(t[cell_labels_int == i] <= t[cell_labels_int == j].reshape(-1, 1))
            tscore[f'{label_dic_rev[i]} -> {label_dic_rev[j]}'] = p
    tscore_out = {}  # only return directly connected cell types in cluster_edges
    for u, v in cluster_edges:
        if (u, v) in cluster_edges:
            tscore_out[f'{u} -> {v}'] = tscore[f'{u} -> {v}']
    return tscore_out, np.mean([sc for sc in tscore.values()])


def inner_cluster_coh(
    adata: AnnData,
    k_cluster: str,
    k_velocity: str,
    gene_mask: Optional[np.ndarray] = None,
    return_raw: bool = False
) -> Tuple[Dict, float]:
    """
    In-Cluster Coherence.

    Measures the average consistency of RNA velocity in each distinct cell type.

    Args:
        adata (anndata.AnnData): AnnData object.
        k_cluster (str): Key to the cluster column in adata.obs DataFrame.
        k_velocity (str): Key to the velocity matrix in adata.layers.
        gene_mask (Optional[np.ndarray], optional): Boolean array to filter out genes. Defaults to None.
        return_raw (bool, optional): Return aggregated or raw scores. Defaults to False.

    Returns:
        Tuple[Dict, float]: 
            - Dict: all_scores indexed by cluster_edges mean scores indexed by cluster_edges.
            - float: Average score over all cells.
    """
    clusters = np.unique(adata.obs[k_cluster])
    scores = {}
    all_scores = {}

    # Get the connectivity matrix
    connectivities = adata.obsp[adata.uns['neighbors']['connectivities_key']]
    
    # Convert to CSR format if it's not already
    if not isinstance(connectivities, csr_matrix):
        connectivities = connectivities.tocsr()

    def get_neighbors(idx):
        return connectivities[idx].indices

    for cat in clusters:
        sel = adata.obs[k_cluster] == cat
        sel_indices = np.where(sel)[0]
        
        velocities = adata.layers[k_velocity]
        nan_mask = ~np.isnan(velocities[0]) if gene_mask is None else gene_mask
        velocities = velocities[:, nan_mask]
        
        cat_vels = velocities[sel]
        
        cat_score = []
        for ith, idx in enumerate(sel_indices):
            nbs = get_neighbors(idx)
            same_cat_nodes = keep_type(adata, nbs, cat, k_cluster)
            if len(same_cat_nodes) > 0:
                score = cosine_similarity(cat_vels[[ith]], velocities[same_cat_nodes]).mean()
                cat_score.append(score)
        
        all_scores[cat] = cat_score
        scores[cat] = np.mean(cat_score)

    if return_raw:
        return all_scores

    return scores, np.mean([sc for sc in scores.values()])



def _pearson_corr(v: np.ndarray, v_neighbor: np.ndarray):
    return np.corrcoef(v, v_neighbor)[0, 1:]


def velocity_consistency(
    adata: AnnData, vkey: str, gene_mask: Optional[np.ndarray] = None
) -> float:
    """Velocity Consistency as reported in scVelo paper

    Args:
        adata (:class:`anndata.AnnData`):
            Anndata object.
        vkey (str):
            key to the velocity matrix in adata.obsm.
        gene_mask (:class:`numpy.ndarray`, optional):
            Boolean array to filter out genes. Defaults to None.

    Returns:
        float: Average score over all cells.
    """
    # Get the connectivity matrix
    connectivities = adata.obsp[adata.uns['neighbors']['connectivities_key']]
    
    # Convert to CSR format if it's not already
    if not isinstance(connectivities, csr_matrix):
        connectivities = connectivities.tocsr()
    
    velocities = adata.layers[vkey]
    nan_mask = ~np.isnan(velocities[0]) if gene_mask is None else gene_mask
    velocities = velocities[:, nan_mask]
    
    def _get_neighbors(i):
        neighbors = connectivities[i].indices
        return neighbors
    
    consistency_score = [_pearson_corr(velocities[ith], velocities[_get_neighbors(ith)]).mean()
                         for ith in range(adata.n_obs)]
    
    adata.obs[f'{vkey}_consistency'] = consistency_score
    return np.mean(consistency_score)


##########################################################################
# End of Reference
##########################################################################


# Cell Cycle Assignment based on scanpy
# Reference: https://github.com/scverse/scanpy_usage/blob/master/180209_cell_cycle/cell_cycle.ipynb
S_GENES_HUMAN = ['MCM5', 'PCNA', 'TYMS', 'FEN1', 'MCM2', 'MCM4', 'RRM1', 'UNG', 'GINS2', 'MCM6',
                 'CDCA7', 'DTL', 'PRIM1', 'UHRF1', 'MLF1IP', 'HELLS', 'RFC2', 'RPA2', 'NASP', 'RAD51AP1',
                 'GMNN', 'WDR76', 'SLBP', 'CCNE2', 'UBR7', 'POLD3', 'MSH2', 'ATAD2', 'RAD51', 'RRM2',
                 'CDC45', 'CDC6', 'EXO1', 'TIPIN', 'DSCC1', 'BLM', 'CASP8AP2', 'USP1', 'CLSPN', 'POLA1',
                 'CHAF1B', 'BRIP1', 'E2F8']

G2M_GENES_HUMAN = ['HMGB2', 'CDK1', 'NUSAP1', 'UBE2C', 'BIRC5', 'TPX2', 'TOP2A', 'NDC80', 'CKS2', 'NUF2',
                   'CKS1B', 'MKI67', 'TMPO', 'CENPF', 'TACC3', 'FAM64A', 'SMC4', 'CCNB2', 'CKAP2L', 'CKAP2',
                   'AURKB', 'BUB1', 'KIF11', 'ANP32E', 'TUBB4B', 'GTSE1', 'KIF20B', 'HJURP', 'CDCA3', 'HN1',
                   'CDC20', 'TTK', 'CDC25C', 'KIF2C', 'RANGAP1', 'NCAPD2', 'DLGAP5', 'CDCA2', 'CDCA8', 'ECT2',
                   'KIF23', 'HMMR', 'AURKA', 'PSRC1', 'ANLN', 'LBR', 'CKAP5', 'CENPE', 'CTCF', 'NEK2', 'G2E3',
                   'GAS2L3', 'CBX5', 'CENPA']

S_GENES_MOUSE = ['Mcm5', 'Pcna', 'Tyms', 'Fen1', 'Mcm2', 'Mcm4', 'Rrm1', 'Ung', 'Gins2', 'Mcm6',
                 'Cdca7', 'Dtl', 'Prim1', 'Uhrf1', 'Mlf1ip', 'Hells', 'Rfc2', 'Rpa2', 'Nasp', 'Rad51ap1',
                 'Gmnn', 'Wdr76', 'Slbp', 'Ccne2', 'Ubr7', 'Pold3', 'Msh2', 'Atad2', 'Rad51', 'Rrm2',
                 'Cdc45', 'Cdc6', 'Exo1', 'Tipin', 'Dscc1', 'Blm', 'Casp8ap2', 'Usp1', 'Clspn', 'Pola1',
                 'Chaf1b', 'Brip1', 'E2f8']

G2M_GENES_MOUSE = ['Hmgb2', 'Cdk1', 'Nusap1', 'Ube2c', 'Birc5', 'Tpx2', 'Top2a', 'Ndc80',
                   'Cks2', 'Nuf2', 'Cks1b', 'Mki67', 'Tmpo', 'Cenpf', 'Tacc3', 'Fam64a',
                   'Smc4', 'Ccnb2', 'Ckap2l', 'Ckap2', 'Aurkb', 'Bub1', 'Kif11', 'Anp32e',
                   'Tubb4b', 'Gtse1', 'Kif20b', 'Hjurp', 'Cdca3', 'Hn1', 'Cdc20', 'Ttk',
                   'Cdc25c', 'Kif2c', 'Rangap1', 'Ncapd2', 'Dlgap5', 'Cdca2', 'Cdca8',
                   'Ect2', 'Kif23', 'Hmmr', 'Aurka', 'Psrc1', 'Anln', 'Lbr', 'Ckap5',
                   'Cenpe', 'Ctcf', 'Nek2', 'G2e3', 'Gas2l3', 'Cbx5', 'Cenpa']


def assign_phase(
    adata: AnnData,
    model: Literal['human', 'mouse'] = 'human',
    embed: str = 'umap',
    save: Optional[str] = None
):
    """
    Assign cell cycle phases.

    Args:
        adata (AnnData): Annotated data matrix.
        model (str, optional): Species model to use ('human' or 'mouse'). Defaults to 'human'.
        embed (str, optional): Low-dimensional embedding for visualization. Defaults to 'umap'.
        save (str, optional): File path to save the figure. If None, figure is not saved. Defaults to None.
    """
    if model == 'human':
        s_genes, g2m_genes = S_GENES_HUMAN, G2M_GENES_HUMAN
    elif model == 'mouse':
        s_genes, g2m_genes = S_GENES_MOUSE, G2M_GENES_MOUSE
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
    cell_cycle_genes = s_genes + g2m_genes
    cell_cycle_genes = np.array([x in cell_cycle_genes for x in adata.var_names])
    adata_cc_genes = adata[:, cell_cycle_genes]
    sc.tl.pca(adata_cc_genes)
    sc.pl.scatter(adata_cc_genes, basis=embed, color='phase', save=save)

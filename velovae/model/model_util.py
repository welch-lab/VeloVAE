"""Model utility functions
"""
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import loggamma
from .scvelo_util import mRNA, vectorize, tau_inv, R_squared, test_bimodality, leastsq_NxN
from sklearn.neighbors import NearestNeighbors
import pynndescent
from tqdm.autonotebook import trange
from sklearn.cluster import SpectralClustering, KMeans
from scipy.stats import dirichlet, bernoulli, kstest, linregress
from scipy.linalg import svdvals

###################################################################################
# Dynamical Model
# Reference:
# Bergen, V., Lange, M., Peidli, S., Wolf, F. A., & Theis, F. J. (2020).
# Generalizing RNA velocity to transient cell states through dynamical modeling.
# Nature biotechnology, 38(12), 1408-1414.
###################################################################################


def scv_pred_single(t, alpha, beta, gamma, ts, scaling=1.0, uinit=0, sinit=0):
    # Predicts u and s using the dynamical model.
    beta = beta*scaling
    tau, alpha, u0, s0 = vectorize(t, ts, alpha, beta, gamma, u0=uinit, s0=sinit)
    tau = np.clip(tau, a_min=0, a_max=None)
    ut, st = mRNA(tau, u0, s0, alpha, beta, gamma)
    ut = ut*scaling
    return ut.squeeze(), st.squeeze()


def scv_pred(adata, key, glist=None):
    # Reproduce the full prediction of scvelo dynamical model

    n_gene = len(glist) if glist is not None else adata.n_vars
    n_cell = adata.n_obs
    ut, st = np.ones((n_cell, n_gene))*np.nan, np.ones((n_cell, n_gene))*np.nan
    if glist is None:
        glist = adata.var_names.to_numpy()

    for i in range(n_gene):
        idx = np.where(adata.var_names == glist[i])[0][0]
        item = adata.var.loc[glist[i]]
        if len(item) == 0:
            print('Gene '+glist[i]+' not found!')
            continue

        alpha, beta, gamma = item[f'{key}_alpha'], item[f'{key}_beta'], item[f'{key}_gamma']
        scaling = item[f'{key}_scaling']
        ts = item[f'{key}_t_']
        t = adata.layers[f'{key}_t'][:, idx]
        if np.isnan(alpha):
            continue
        u_g, s_g = scv_pred_single(t, alpha, beta, gamma, ts, scaling)
        ut[:, i] = u_g
        st[:, i] = s_g

    return ut, st

#End of Reference


############################################################
# Shared among all VAEs
############################################################


def hist_equal(t, tmax, perc=0.95, n_bin=101):
    # Perform histogram equalization across all local times.
    t_ub = np.quantile(t, perc)
    t_lb = t.min()
    delta_t = (t_ub - t_lb)/(n_bin-1)
    bins = [t_lb+i*delta_t for i in range(n_bin)]+[t.max()]
    pdf_t, edges = np.histogram(t, bins, density=True)
    pt, edges = np.histogram(t, bins, density=False)

    # Perform histogram equalization
    cdf_t = np.concatenate(([0], np.cumsum(pt)))
    cdf_t = cdf_t/cdf_t[-1]
    t_out = np.zeros((len(t)))
    for i in range(n_bin):
        mask = (t >= bins[i]) & (t < bins[i+1])
        t_out[mask] = (cdf_t[i] + (t[mask]-bins[i])*pdf_t[i])*tmax
    return t_out

############################################################
# Basic utility function to compute ODE solutions for all models
############################################################


def pred_su_numpy(tau, u0, s0, alpha, beta, gamma):
    ############################################################
    # (Numpy Version)
    # Analytical solution of the ODE
    # tau: [B x 1] or [B x 1 x 1] time duration starting from the switch-on time of each gene.
    # u0, s0: [G] or [N type x G] initial conditions
    # alpha, beta, gamma: [G] or [N type x G] generation, splicing and degradation rates
    ############################################################
    unstability = (np.abs(beta-gamma) < 1e-6)
    expb, expg = np.exp(-beta*tau), np.exp(-gamma*tau)

    upred = u0*expb + alpha/beta*(1-expb)
    spred = s0*expg + alpha/gamma*(1-expg) \
        + (alpha-beta*u0)/(gamma-beta+1e-6)*(expg-expb)*(1-unstability) \
        - (alpha-beta*u0)*tau*expg*unstability
    return np.clip(upred, a_min=0, a_max=None), np.clip(spred, a_min=0, a_max=None)


def pred_su(tau, u0, s0, alpha, beta, gamma):
    ############################################################
    # (PyTorch Version)
    # Analytical solution of the ODE
    # tau: [B x 1] or [B x 1 x 1] time duration starting from the switch-on time of each gene.
    # u0, s0: [G] or [N type x G] initial conditions
    # alpha, beta, gamma: [G] or [N type x G] generation, splicing and degradation rates
    ############################################################

    expb, expg = torch.exp(-beta*tau), torch.exp(-gamma*tau)
    eps = 1e-6
    unstability = (torch.abs(beta-gamma) < eps).long()

    upred = u0*expb + alpha/beta*(1-expb)
    spred = s0*expg + alpha/gamma*(1-expg) \
        + (alpha-beta*u0)/(gamma-beta+eps)*(expg-expb)*(1-unstability) \
        - (alpha-beta*u0)*tau*expg*unstability
    return nn.functional.relu(upred), nn.functional.relu(spred)


def pred_su_back(tau, u1, s1, alpha, beta, gamma):
    ############################################################
    # (PyTorch Version)
    # Analytical solution of the ODE
    # tau: [B x 1] or [B x 1 x 1] time duration starting from the switch-on time of each gene.
    # u0, s0: [G] or [N type x G] initial conditions
    # alpha, beta, gamma: [G] or [N type x G] generation, splicing and degradation rates
    ############################################################
    expb, expg = torch.exp(beta*tau), torch.exp(gamma*tau)
    eps = 1e-6
    unstability = (torch.abs(beta-gamma) < eps).long()

    upred = u1*expb - alpha/beta*(expb-1)
    spred = s1*expg - alpha/gamma*(expg-1) \
        - (alpha-beta*u1)/(gamma-beta+eps)*(expb-expg)*(1-unstability) \
        + (alpha-beta*u1)*expb*tau*unstability
    return nn.functional.relu(upred), nn.functional.relu(spred)


###################################################################################
# Initialization Methods
# Reference:
# Bergen, V., Lange, M., Peidli, S., Wolf, F. A., & Theis, F. J. (2020).
# Generalizing RNA velocity to transient cell states through dynamical modeling.
# Nature biotechnology, 38(12), 1408-1414.
###################################################################################


def scale_by_gene(U, S, train_idx=None, mode='scale_u'):
    # mode
    #   'auto' means to scale the one with a smaller range
    #   'scale_u' means to match std(u) with std(s)
    #   'scale_s' means to match std(s) with std(u)
    G = U.shape[1]
    scaling_u = np.ones((G))
    scaling_s = np.ones((G))
    std_u, std_s = np.ones((G)), np.ones((G))
    for i in range(G):
        if train_idx is None:
            si, ui = S[:, i], U[:, i]
        else:
            si, ui = S[train_idx, i], U[train_idx, i]
        sfilt, ufilt = si[(si > 0) & (ui > 0)], ui[(si > 0) & (ui > 0)]  # Use only nonzero data points
        if len(sfilt) > 3 and len(ufilt) > 3:
            std_u[i] = np.std(ufilt)
            std_s[i] = np.std(sfilt)
    mask_u, mask_s = (std_u == 0), (std_s == 0)
    std_u = std_u + (mask_u & (~mask_s))*std_s + (mask_u & mask_s)*1
    std_s = std_s + ((~mask_u) & mask_s)*std_u + (mask_u & mask_s)*1
    if mode == 'auto':
        scaling_u = np.max(np.stack([scaling_u, (std_u/std_s)]), 0)
        scaling_s = np.max(np.stack([scaling_s, (std_s/std_u)]), 0)
    elif mode == 'scale_u':
        scaling_u = std_u/std_s
    elif mode == 'scale_s':
        scaling_s = std_s/std_u
    return U/scaling_u, S/scaling_s, scaling_u, scaling_s


def get_gene_scale(U, S, train_idx=None, mode='scale_u'):
    # mode
    #   'auto' means to scale the one with a smaller range
    #   'scale_u' means to match std(u) with std(s)
    #   'scale_s' means to match std(s) with std(u)
    G = U.shape[1]
    scaling_u = np.ones((G))
    scaling_s = np.ones((G))
    std_u, std_s = np.ones((G)), np.ones((G))
    for i in range(G):
        if train_idx is None:
            si, ui = S[:, i], U[:, i]
        else:
            si, ui = S[train_idx, i], U[train_idx, i]
        sfilt, ufilt = si[(si > 0) & (ui > 0)], ui[(si > 0) & (ui > 0)]  # Use only nonzero data points
        if len(sfilt) > 3 and len(ufilt) > 3:
            std_u[i] = np.std(ufilt)
            std_s[i] = np.std(sfilt)
    mask_u, mask_s = (std_u == 0), (std_s == 0)
    std_u = std_u + (mask_u & (~mask_s))*std_s + (mask_u & mask_s)*1
    std_s = std_s + ((~mask_u) & mask_s)*std_u + (mask_u & mask_s)*1
    if mode == 'auto':
        scaling_u = np.max(np.stack([scaling_u, (std_u/std_s)]), 0)
        scaling_s = np.max(np.stack([scaling_s, (std_s/std_u)]), 0)
    elif mode == 'scale_u':
        scaling_u = std_u/std_s
    elif mode == 'scale_s':
        scaling_s = std_s/std_u
    return scaling_u, scaling_s


def compute_scaling_bound(cell_scale):
    # Compute the upper and lower bound for scaling factor thresholding
    log_scale = np.log(cell_scale)
    q3, q1 = np.quantile(log_scale, 0.75), np.quantile(log_scale, 0.25)
    iqr = q3 - q1
    ub, lb = q3 + 1.5*iqr, q1 - 1.5*iqr
    return np.exp(ub), np.exp(lb)


def clip_cell_scale(lu, ls):
    # Remove extreme values
    lu_max, lu_min = compute_scaling_bound(lu)
    ls_max, ls_min = compute_scaling_bound(ls)
    lu = np.clip(lu, a_min=lu_min, a_max=lu_max)
    ls = np.clip(ls, a_min=ls_min, a_max=ls_max)
    return lu, ls


def scale_by_cell(U, S, train_idx=None, separate_us_scale=True, q=50):
    nu, ns = U.sum(1, keepdims=True), S.sum(1, keepdims=True)
    if separate_us_scale:
        norm_count = ((np.percentile(nu, q), np.percentile(ns, q)) if train_idx is None else
                      (np.percentile(nu[train_idx], q), np.percentile(ns[train_idx], q)))
        lu = nu/norm_count[0]
        ls = ns/norm_count[1]
    else:
        norm_count = np.percentile(nu+ns, q) if train_idx is None else np.percentile(nu[train_idx]+ns[train_idx], q)
        lu = (nu+ns)/norm_count
        ls = lu
    # Remove extreme values
    print(f"Detecting zero scaling factors: {np.sum(lu==0)}, {np.sum(ls==0)}")
    lu[lu == 0] = np.min(lu[lu > 0])
    ls[ls == 0] = np.min(ls[ls > 0])
    return U/lu, S/ls, lu, ls


def get_cell_scale(U, S, train_idx=None, separate_us_scale=True, q=0.5):
    nu, ns = U.sum(1, keepdims=True), S.sum(1, keepdims=True)
    if separate_us_scale:
        norm_count = ((np.percentile(nu, q), np.percentile(ns, q)) if train_idx is None else
                      (np.percentile(nu[train_idx], q), np.percentile(ns[train_idx], q)))
        lu = nu/norm_count[0]
        ls = ns/norm_count[1]
    else:
        norm_count = np.percentile(nu+ns, q) if train_idx is None else np.percentile(nu[train_idx]+ns[train_idx], q)
        lu = (nu+ns)/norm_count
        ls = lu
    # Remove extreme values
    print(f"Detecting zero scaling factors: {np.sum(lu==0)}, {np.sum(ls==0)}")
    lu[lu == 0] = np.min(lu[lu > 0])
    ls[ls == 0] = np.min(ls[ls > 0])
    return lu, ls


def get_dispersion(U, S, clip_min=1e-3, clip_max=1000):
    mean_u, mean_s = np.clip(U.mean(0), 1e-6, None), np.clip(S.mean(0), 1e-6, None)
    var_u, var_s = U.var(0), S.var(0)
    dispersion_u, dispersion_s = var_u/mean_u, var_s/mean_s
    dispersion_u = np.clip(dispersion_u, a_min=clip_min, a_max=clip_max)
    dispersion_s = np.clip(dispersion_s, a_min=clip_min, a_max=clip_max)
    return mean_u, mean_s, dispersion_u, dispersion_s


def linreg(u, s):
    # Linear regression (helper function)
    q = np.sum(s*s)
    r = np.sum(u*s)
    k = r/q
    if np.isinf(k) or np.isnan(k):
        k = 1.0+np.random.rand()
    return k


def init_gene(s, u, percent, fit_scaling=False, Ntype=None):
    # Adopted from scvelo
    std_u, std_s = np.std(u), np.std(s)
    scaling = std_u / std_s if fit_scaling else 1.0
    u = u/scaling

    # Pick Quantiles
    # initialize beta and gamma from extreme quantiles of s
    mask_s = s >= np.percentile(s, percent, axis=0)
    mask_u = u >= np.percentile(u, percent, axis=0)
    mask = mask_s & mask_u
    if not np.any(mask):
        mask = mask_s

    # Initialize alpha, beta and gamma
    beta = 1
    gamma = linreg(u[mask], s[mask]) + 1e-6
    if gamma < 0.05 / scaling:
        gamma *= 1.2
    elif gamma > 1.5 / scaling:
        gamma /= 1.2
    u_inf, s_inf = u[mask].mean(), s[mask].mean()
    u0_, s0_ = u_inf, s_inf
    alpha = u_inf*beta
    # initialize switching from u quantiles and alpha from s quantiles
    tstat_u, pval_u, means_u = test_bimodality(u, kde=True)
    tstat_s, pval_s, means_s = test_bimodality(s, kde=True)
    pval_steady = max(pval_u, pval_s)
    steady_u = means_u[1]
    if pval_steady < 1e-3:
        u_inf = np.mean([u_inf, steady_u])
        alpha = gamma * s_inf
        beta = alpha / u_inf
        u0_, s0_ = u_inf, s_inf
    t_ = tau_inv(u0_, s0_, 0, 0, alpha, beta, gamma)  # time to reach steady state
    tau = tau_inv(u, s, 0, 0, alpha, beta, gamma)  # induction
    tau = np.clip(tau, 0, t_)
    tau_ = tau_inv(u, s, u0_, s0_, 0, beta, gamma)  # repression
    tau_ = np.clip(tau_, 0, np.max(tau_[s > 0]))
    ut, st = mRNA(tau, 0, 0, alpha, beta, gamma)
    ut_, st_ = mRNA(tau_, u0_, s0_, 0, beta, gamma)
    distu, distu_ = (u - ut), (u - ut_)
    dists, dists_ = (s - st), (s - st_)
    res = np.array([distu ** 2 + dists ** 2, distu_ ** 2 + dists_ ** 2])
    t = np.array([tau, tau_+np.ones((len(tau_)))*t_])
    o = np.argmin(res, axis=0)
    t_latent = np.array([t[o[i], i] for i in range(len(tau))])

    return alpha, beta, gamma, t_latent, u0_, s0_, t_, scaling


def init_params(data, percent, fit_offset=False, fit_scaling=True, eps=1e-3):
    # Adopted from SCVELO
    # Use the steady-state model to estimate alpha, beta,
    # gamma and the latent time
    # data: ncell x (2*ngene) tensor
    # percent: percentage limit to pick the data
    # Output: a ncellx4 2D array of parameters

    ngene = data.shape[1]//2
    u = data[:, :ngene]
    s = data[:, ngene:]

    params = np.ones((ngene, 4))  # four parameters: alpha, beta, gamma, scaling
    params[:, 0] = np.random.rand((ngene))*np.max(u, 0)
    params[:, 2] = np.clip(np.random.rand((ngene))*np.max(u, 0)/(np.max(s, 0)+1e-10), eps, None)
    T = np.zeros((ngene, len(s)))
    Ts = np.zeros((ngene))
    U0, S0 = np.zeros((ngene)), np.zeros((ngene))  # Steady-1 State

    print('Estimating ODE parameters...')
    for i in trange(ngene):
        si, ui = s[:, i], u[:, i]
        sfilt, ufilt = si[(si > 0) & (ui > 0)], ui[(si > 0) & (ui > 0)]  # Use only nonzero data points
        if len(sfilt) > 3 and len(ufilt) > 3:
            alpha, beta, gamma, t, u0_, s0_, ts, scaling = init_gene(sfilt, ufilt, percent, fit_scaling)
            params[i, :] = np.array([alpha, beta, np.clip(gamma, eps, None), scaling])
            T[i, (si > 0) & (ui > 0)] = t
            U0[i] = u0_
            S0[i] = s0_
            Ts[i] = ts
        else:
            U0[i] = np.max(u)
            S0[i] = np.max(s)

    # Filter out genes
    min_r2 = 0.01
    offset, gamma = leastsq_NxN(s, u, fit_offset, perc=[100-percent, percent])
    gamma = np.clip(gamma, eps, None)
    residual = u-gamma*s
    if fit_offset:
        residual -= offset

    r2 = R_squared(residual, total=u-u.mean(0))
    velocity_genes = (r2 > min_r2) & (r2 < 0.95) & (gamma > 0.01) & (np.max(s > 0, 0) > 0) & (np.max(u > 0, 0) > 0)
    print(f'Detected {np.sum(velocity_genes)} velocity genes.')

    dist_u, dist_s = np.zeros(u.shape), np.zeros(s.shape)
    print('Estimating the variance...')
    assert np.all(params[:, 2] > 0)
    for i in trange(ngene):
        upred, spred = scv_pred_single(T[i],
                                       params[i, 0],
                                       params[i, 1],
                                       params[i, 2],
                                       Ts[i],
                                       params[i, 3])  # upred has the original scale
        dist_u[:, i] = u[:, i] - upred
        dist_s[:, i] = s[:, i] - spred

    sigma_u = np.clip(np.std(dist_u, 0), 0.1, None)
    sigma_s = np.clip(np.std(dist_s, 0), 0.1, None)
    sigma_u[np.isnan(sigma_u)] = 0.1
    sigma_s[np.isnan(sigma_s)] = 0.1

    # Make sure all genes get the same total relevance score
    gene_score = velocity_genes * 1.0 + (1 - velocity_genes) * 0.25

    return params[:, 0], params[:, 1], params[:, 2], params[:, 3], Ts, U0, S0, sigma_u, sigma_s, T.T, gene_score


###################################################################################
# Reinitialization based on the global time
###################################################################################


def get_ts_global(tgl, U, S, perc):
    # Initialize the transition time in the original ODE model.
    tsgl = np.zeros((U.shape[1]))
    for i in range(U.shape[1]):
        u, s = U[:, i], S[:, i]
        zero_mask = (u > 0) & (s > 0)
        mask_u, mask_s = u >= np.percentile(u, perc), s >= np.percentile(s, perc)
        mask = mask_u & mask_s & zero_mask
        if not np.any(mask):
            mask = (mask_u | mask_s) & zero_mask
        # edge case: all u or all s are zero
        if not np.any(mask):
            mask = (mask_u | mask_s) & ((u > 0) | (s > 0))
        # edge case: all u and all s are zero
        if not np.any(mask):
            mask = np.ones((len(u))).astype(bool)
        tsgl[i] = np.median(tgl[mask])
        if np.isnan(tsgl[i]):
            tsgl[i] = np.median(tgl)

    assert not np.any(np.isnan(tsgl))
    return tsgl


def reinit_gene(u, s, t, ts, eps=1e-6, max_val=1e4):
    # Applied to the regular ODE
    # Initialize the ODE parameters (alpha,beta,gamma,t_on) from
    # input data and estimated global cell time.
    # u1, u2: picked from induction
    q = 0.95
    mask1_u = u > np.quantile(u, q)
    mask1_s = s > np.quantile(s, q)
    # edge case handling
    while not np.any(mask1_u | mask1_s) and q > 0.05:
        q = q - 0.05
        mask1_u = u > np.quantile(u, q)
        mask1_s = s > np.quantile(s, q)
    if not np.any(mask1_u | mask1_s):
        mask1_u = u >= np.min(u)
        mask1_s = s >= np.min(s)
    assert np.any(mask1_u | mask1_s)
    u1, s1 = np.median(u[mask1_u | mask1_s]), np.median(s[mask1_s | mask1_u])
    if u1 == 0 or np.isnan(u1):
        u1 = np.max(u)
    if s1 == 0 or np.isnan(s1):
        s1 = np.max(s)

    t1 = np.median(t[mask1_u | mask1_s])
    if t1 <= 0:
        tm = np.max(t[mask1_u | mask1_s])
        t1 = tm if tm > 0 else 1.0

    mask2_u = (u >= u1*0.49) & (u <= u1*0.51) & (t <= ts)
    mask2_s = (s >= s1*0.49) & (s <= s1*0.51) & (t <= ts)
    if np.any(mask2_u):
        t2 = np.median(t[mask2_u | mask2_s])
        u2 = np.median(u[mask2_u])
        t0 = np.log(np.clip((u1-u2)/(u1*np.exp(-t2)-u2*np.exp(-t1)+eps), a_min=1.0, a_max=None))
    else:
        t0 = 0
    beta = 1
    alpha = u1/(1-np.exp(t0-t1)) if u1 > 0 else 0.1*np.random.rand()
    alpha = np.clip(alpha, None, max_val)
    if alpha <= 0 or np.isnan(alpha) or np.isinf(alpha):
        alpha = u1
    p = 0.95
    s_inf = np.quantile(s, p)
    while s_inf == 0 and p < 1.0:
        p = p + 0.01
        s_inf = np.quantile(s, p)
    gamma = alpha/np.clip(s_inf, a_min=eps, a_max=None)
    if gamma <= 0 or np.isnan(gamma) or np.isinf(gamma):
        gamma = 2.0
    gamma = np.clip(gamma, None, max_val)
    return alpha, beta, gamma, t0


def reinit_params(U, S, t, ts):
    print('Reinitialize the regular ODE parameters based on estimated global latent time.')
    G = U.shape[1]
    alpha, beta, gamma, ton = np.zeros((G)), np.zeros((G)), np.zeros((G)), np.zeros((G))
    for i in trange(G):
        alpha_g, beta_g, gamma_g, ton_g = reinit_gene(U[:, i], S[:, i], t, ts[i])
        alpha[i] = alpha_g
        beta[i] = beta_g
        gamma[i] = gamma_g
        ton[i] = ton_g
    assert not np.any(np.isnan(alpha))
    assert not np.any(np.isnan(gamma))
    return alpha, beta, gamma, ton


def find_dirichlet_param(mu, std):
    alpha_1 = ((mu/std)*((1-mu)/std) - 1) * mu
    return np.array([alpha_1, (1-mu)/mu*alpha_1])


def assign_gene_mode_binary(adata, w_noisy, thred=0.05):
    Cs = np.corrcoef(adata.layers['Ms'].T)
    Cu = np.corrcoef(adata.layers['Mu'].T)
    C = 1+Cs*0.5+Cu*0.5
    C[np.isnan(C)] = 0.0
    spc = SpectralClustering(2, affinity='precomputed', assign_labels='discretize')
    y = spc.fit_predict(C)
    adata.var['init_mode'] = y

    alpha_1, alpha_2 = find_dirichlet_param(0.6, 0.2), find_dirichlet_param(0.4, 0.2)
    w = dirichlet(alpha_1).rvs(adata.n_vars)

    # Perform Kolmogorov-Smirnov Test
    w1, w2 = w_noisy[y == 0], w_noisy[y == 1]

    w_neutral = dirichlet.rvs([12, 12], size=adata.n_vars, random_state=42)
    res, pval = kstest(w1, w2)
    if pval > 0.05:
        print('Two modes are indistuiguishable.')
        return w_neutral

    res_1, pval_1 = kstest(w1, w2, alternative='greater', method='asymp')
    res_2, pval_2 = kstest(w1, w2, alternative='less', method='asymp')
    if pval_1 >= thred:  # Take the null hypothesis that values of w1 are greater
        adata.varm['alpha_w'] = (y.reshape(-1, 1) == 0) * alpha_1 + (y.reshape(-1, 1) == 1) * alpha_2
        return (y == 0) * w[:, 0] + (y == 1) * w[:, 1]
    elif pval_2 >= thred:
        adata.varm['alpha_w'] = (y.reshape(-1, 1) == 1) * alpha_1 + (y.reshape(-1, 1) == 0) * alpha_2
        return (y == 0) * w[:, 1] + (y == 1) * w[:, 0]

    return w_neutral


def get_nclusters(C, noise=1.0, n_cluster_thred=3):
    # determine the number of clusters based an affinity matrix
    # thred: minimal number of clusters
    v = svdvals(C)
    n_clusters = 0
    while n_clusters < n_cluster_thred and noise > 0:
        thred = 4 / np.sqrt(3) * np.sqrt(C.shape[0]) * noise
        n_clusters = np.sum((v > thred))
        noise = noise - 0.005
    print(f'{n_clusters} clusters detected based on gene co-expression.')
    return n_clusters


def sample_dir_mix(w, yw, std_prior):
    # Sample from a mixture of dirichlet distributions
    mu_0, mu_1 = np.mean(w[yw == 0]), np.mean(w[yw == 1])
    alpha_w_0 = find_dirichlet_param(mu_0, std_prior)
    alpha_w_1 = find_dirichlet_param(mu_1, std_prior)
    np.random.seed(42)
    q1 = dirichlet.rvs(alpha_w_0, size=len(w))[:, 0]
    np.random.seed(42)
    q2 = dirichlet.rvs(alpha_w_1, size=len(w))[:, 0]
    wq = np.sum(yw == 1)/len(yw)
    np.random.seed(42)
    b = bernoulli.rvs(wq, size=len(w))
    q = (b == 0)*q1 + (b == 1)*q2
    print(f'({1-wq:.2f}, {mu_0}), ({wq:.2f}, {mu_1})')
    return q


def assign_gene_mode_auto(adata,
                          w_noisy,
                          thred=0.05,
                          std_prior=0.1,
                          n_cluster_thred=3):
    # Compute gene correlation matrix
    Cs = np.corrcoef(adata.layers['Ms'].T)
    Cu = np.corrcoef(adata.layers['Mu'].T)
    C = 1+Cs*0.5+Cu*0.5
    C[np.isnan(C)] = 0.0
    # Spectral clustering
    spc = SpectralClustering(get_nclusters(C, n_cluster_thred),
                             affinity='precomputed',
                             assign_labels='discretize',
                             random_state=42)
    y = spc.fit_predict(C)
    adata.var['init_mode'] = y

    # Sample weights from Dirichlet(mu=0.5, std=std_prior)
    alpha_neutral = find_dirichlet_param(0.5, std_prior)
    q_neutral = dirichlet.rvs(alpha_neutral, size=adata.n_vars)[:, 0]
    w = np.empty((adata.n_vars))
    pval_ind = []
    pval_rep = []
    cluster_type = np.zeros((y.max()+1))
    alpha_ind, alpha_rep = find_dirichlet_param(0.6, std_prior), find_dirichlet_param(0.4, std_prior)

    # Perform Komogorov-Smirnov Test
    for i in range(y.max()+1):
        n = np.sum(y == i)
        res_1, pval_1 = kstest(w_noisy[y == i], q_neutral, alternative='greater', method='asymp')
        res_2, pval_2 = kstest(w_noisy[y == i], q_neutral, alternative='less', method='asymp')
        pval_ind.append(pval_1)
        pval_rep.append(pval_2)
        if pval_1 < thred and pval_2 < thred:  # uni/bi-modal dirichlet
            cluster_type[i] = 0
            res_3, pval_3 = kstest(w_noisy[y == i], q_neutral)
            if pval_3 < thred:
                km = KMeans(2, n_init='auto')
                yw = km.fit_predict(w_noisy[y == i].reshape(-1, 1))
                w[y == i] = sample_dir_mix(w_noisy[y == i], yw, std_prior)
            else:
                np.random.seed(42)
                w[y == i] = dirichlet.rvs(alpha_neutral, size=n)[:, 0]

        elif pval_1 >= 0.05:  # induction
            cluster_type[i] = 1
            np.random.seed(42)
            w[y == i] = dirichlet.rvs(alpha_ind, size=n)[:, 0]
        elif pval_2 >= 0.05:  # repression
            cluster_type[i] = 2
            np.random.seed(42)
            w[y == i] = dirichlet.rvs(alpha_rep, size=n)[:, 0]

    pval_ind = np.array(pval_ind)
    pval_rep = np.array(pval_rep)
    print(f'KS-test result: {cluster_type}')
    # If no repressive cluster is found, pick the one with the highest p value
    if np.all(cluster_type == 1):
        ymax = np.argmax(pval_rep)
        print(f'Assign cluster {ymax} to repressive')
        np.random.seed(42)
        w[y == ymax] = dirichlet.rvs(alpha_rep, size=np.sum(y == ymax))[:, 0]

    # If no inductive cluster is found, pick the one with the highest p value
    if np.all(cluster_type == 2):
        ymax = np.argmax(pval_ind)
        print(f'Assign cluster {ymax} to inductive')
        np.random.seed(42)
        w[y == ymax] = dirichlet.rvs(alpha_ind, size=np.sum(y == ymax))[:, 0]

    return w


def assign_gene_mode(adata,
                     w_noisy,
                     assign_type='binary',
                     thred=0.05,
                     std_prior=0.1,
                     n_cluster_thred=3):
    # Assign one of ('inductive', 'repressive', 'mixture') to gene clusters
    # `assign_type' specifies which strategy to use
    if assign_type == 'binary':
        return assign_gene_mode_binary(adata, w_noisy, thred)
    elif assign_type == 'auto':
        return assign_gene_mode_auto(adata, w_noisy, thred, std_prior, n_cluster_thred)
    elif assign_type == 'inductive':
        alpha_ind = find_dirichlet_param(0.8, std_prior)
        np.random.seed(42)
        return dirichlet.rvs(alpha_ind, size=adata.n_vars)[:, 0]
    elif assign_type == 'repressive':
        alpha_rep = find_dirichlet_param(0.2, std_prior)
        np.random.seed(42)
        return dirichlet.rvs(alpha_rep, size=adata.n_vars)[:, 0]


def assign_gene_mode_tprior(adata, tkey, train_idx, std_prior=0.05):
    # Same as assign_gene_mode, but uses the informative time prior
    # to determine inductive and repressive genes
    tprior = adata.obs[tkey].to_numpy()[train_idx]
    alpha_ind, alpha_rep = find_dirichlet_param(0.75, std_prior), find_dirichlet_param(0.25, std_prior)
    w = np.empty((adata.n_vars))
    slope = np.empty((adata.n_vars))
    for i in range(adata.n_vars):
        slope_u, intercept_u, r_u, p_u, se = linregress(tprior, adata.layers['Mu'][train_idx, i])
        slope_s, intercept_s, r_s, p_s, se = linregress(tprior, adata.layers['Ms'][train_idx, i])
        slope[i] = (slope_u*0.5+slope_s*0.5)
    np.random.seed(42)
    w[slope >= 0] = dirichlet.rvs(alpha_ind, size=np.sum(slope >= 0))[:, 0]
    np.random.seed(42)
    w[slope < 0] = dirichlet.rvs(alpha_rep, size=np.sum(slope < 0))[:, 0]
    # return 1/(1+np.exp(-slope))
    return w

############################################################
# Vanilla VAE
############################################################


"""
ODE Solution, with both numpy (for post-training analysis or plotting) and pytorch versions (for training)
"""


def pred_steady_numpy(ts, alpha, beta, gamma):
    ############################################################
    # (Numpy Version)
    # Predict the steady states.
    # ts: [G] switching time, when the kinetics enters the repression phase
    # alpha, beta, gamma: [G] generation, splicing and degradation rates
    ############################################################
    eps = 1e-6
    unstability = np.abs(beta-gamma) < eps

    ts_ = ts.squeeze()
    expb, expg = np.exp(-beta*ts_), np.exp(-gamma*ts_)
    u0 = alpha/(beta+eps)*(1.0-expb)
    s0 = alpha/(gamma+eps)*(1.0-expg)+alpha/(gamma-beta+eps)*(expg-expb)*(1-unstability)-alpha*ts_*expg*unstability
    return u0, s0


def pred_steady(tau_s, alpha, beta, gamma):
    ############################################################
    # (PyTorch Version)
    # Predict the steady states.
    # tau_s: [G] time duration from ton to toff
    # alpha, beta, gamma: [G] generation, splicing and degradation rates
    ############################################################
    eps = 1e-6
    unstability = (torch.abs(beta - gamma) < eps).long()

    expb, expg = torch.exp(-beta*tau_s), torch.exp(-gamma*tau_s)
    u0 = alpha/(beta+eps)*(torch.tensor([1.0]).to(alpha.device)-expb)
    s0 = alpha/(gamma+eps)*(torch.tensor([1.0]).to(alpha.device)-expg) \
        + alpha/(gamma-beta+eps)*(expg-expb)*(1-unstability)-alpha*tau_s*expg*unstability

    return u0, s0


def ode_numpy(t, alpha, beta, gamma, to, ts, scaling=None, k=10.0):
    """(Numpy Version) ODE solution with fixed rates

    Args:
        t (:class:`numpy.ndarray`): Cell time, (N,1)
        alpha (:class:`numpy.ndarray`): Transcription rates
        beta (:class:`numpy.ndarray`): Splicing rates
        gamma (:class:`numpy.ndarray`): Degradation rates
        to (:class:`numpy.ndarray`): switch-on time
        ts (:class:`numpy.ndarray`): switch-off time (induction to repression)
        scaling (:class:numpy array, optional): Scaling factor (u / s). Defaults to None.
        k (float, optional): Parameter for a smooth clip of tau. Defaults to 10.0.

    Returns:
        tuple:
            returns the unspliced and spliced counts predicted by the ODE
    """
    eps = 1e-6
    unstability = (np.abs(beta - gamma) < eps)
    o = (t <= ts).astype(int)
    # Induction
    tau_on = F.softplus(torch.tensor(t-to), beta=k).numpy()
    assert np.all(~np.isnan(tau_on))
    expb, expg = np.exp(-beta*tau_on), np.exp(-gamma*tau_on)
    uhat_on = alpha/(beta+eps)*(1.0-expb)
    shat_on = alpha/(gamma+eps)*(1.0-expg) \
        + alpha/(gamma-beta+eps)*(expg-expb)*(1-unstability) - alpha*tau_on*unstability

    # Repression
    u0_, s0_ = pred_steady_numpy(np.clip(ts-to, 0, None), alpha, beta, gamma)  # tensor shape: (G)
    if ts.ndim == 2 and to.ndim == 2:
        u0_ = u0_.reshape(-1, 1)
        s0_ = s0_.reshape(-1, 1)
    # tau_off = np.clip(t-ts,a_min=0,a_max=None)
    tau_off = F.softplus(torch.tensor(t-ts), beta=k).numpy()
    assert np.all(~np.isnan(tau_off))
    expb, expg = np.exp(-beta*tau_off), np.exp(-gamma*tau_off)
    uhat_off = u0_*expb
    shat_off = s0_*expg+(-beta*u0_)/(gamma-beta+eps)*(expg-expb)*(1-unstability)

    uhat, shat = (uhat_on*o + uhat_off*(1-o)), (shat_on*o + shat_off*(1-o))
    if scaling is not None:
        uhat *= scaling
    return uhat, shat


def ode(t, alpha, beta, gamma, to, ts, neg_slope=0.0):
    """(PyTorch Version) ODE Solution
    Parameters are the same as the numpy version, with arrays replaced with
    tensors. Additionally, neg_slope is used for time clipping.
    """
    eps = 1e-6
    unstability = (torch.abs(beta - gamma) < eps).long()
    o = (t <= ts).int()

    # Induction
    tau_on = F.leaky_relu(t-to, negative_slope=neg_slope)
    expb, expg = torch.exp(-beta*tau_on), torch.exp(-gamma*tau_on)
    uhat_on = alpha/(beta+eps)*(torch.tensor([1.0]).to(alpha.device)-expb)
    shat_on = alpha/(gamma+eps)*(torch.tensor([1.0]).to(alpha.device)-expg) \
        + (alpha/(gamma-beta+eps)*(expg-expb)*(1-unstability) - alpha*tau_on*expg * unstability)

    # Repression
    u0_, s0_ = pred_steady(F.relu(ts-to), alpha, beta, gamma)

    tau_off = F.leaky_relu(t-ts, negative_slope=neg_slope)
    expb, expg = torch.exp(-beta*tau_off), torch.exp(-gamma*tau_off)
    uhat_off = u0_*expb
    shat_off = s0_*expg+(-beta*u0_)/(gamma-beta+eps)*(expg-expb) * (1-unstability)

    return (uhat_on*o + uhat_off*(1-o)), (shat_on*o + shat_off*(1-o))


############################################################
# Branching ODE
############################################################


def encode_type(cell_types_raw):
    """Use integer to encode the cell types

    Args:
        cell_types_raw (array like):
            unique cell types in a dataset

    Returns:
        tuple containing:

            - label_dic (dict): mapping from cell types to integers
            - label_dic_rev (dict): inverse mapping from integers to cell types
    """
    # Map cell types to integers
    label_dic = {}
    label_dic_rev = {}
    for i, type_ in enumerate(cell_types_raw):
        label_dic[type_] = i
        label_dic_rev[i] = type_

    return label_dic, label_dic_rev


def str2int(cell_labels_raw, label_dic):
    """Convert cell type annotations to integers

    Args:
        cell_labels_raw (array like):
            Cell type annotations
        label_dic (dict):
            mapping from cell types to integers

    Returns:
        :class:`numpy.ndarray`:
            Integer encodings of cell type annotations.
    """
    return np.array([label_dic[cell_labels_raw[i]] for i in range(len(cell_labels_raw))])


def int2str(cell_labels, label_dic_rev):
    """Convert integer encodings to original cell type annotations

    Args:
        cell_labels (array like):
            Integer encodings of cell type annotations
        label_dic (dict):
            mapping from integers to cell types

    Returns:
        :class:`numpy.ndarray`:
            Original cell type annotations.
    """
    return np.array([label_dic_rev[cell_labels[i]] for i in range(len(cell_labels))])


def linreg_mtx(u, s):
    ############################################################
    # Performs linear regression ||U-kS||_2 while
    # U and S are matrices and k is a vector.
    # Handles divide by zero by returninig some default value.
    ############################################################
    Q = np.sum(s*s, axis=0)
    R = np.sum(u*s, axis=0)
    k = R/Q
    if np.isinf(k) or np.isnan(k):
        k = 1.5
    return k


def reinit_type_params(U, S, t, ts, cell_labels, cell_types, init_types):
    ############################################################
    # Applied under branching ODE
    # Use the steady-state model and estimated cell time to initialize
    # branching ODE parameters.
    ############################################################
    Ntype = len(cell_types)
    G = U.shape[1]
    alpha, beta, gamma = np.ones((Ntype, G)), np.ones((Ntype, G)), np.ones((Ntype, G))
    u0, s0 = np.zeros((len(init_types), G)), np.zeros((len(init_types), G))

    for i, type_ in enumerate(cell_types):
        mask_type = cell_labels == type_
        # Determine induction or repression
        t_head = np.quantile(t[mask_type], 0.05)
        t_mid = (t_head+np.quantile(t[mask_type], 0.95))*0.5

        u_head = np.mean(U[(t >= t[mask_type].min()) & (t < t_head), :], axis=0)
        u_mid = np.mean(U[(t >= t_mid*0.98) & (t <= t_mid*1.02), :], axis=0)

        s_head = np.mean(S[(t >= t[mask_type].min()) & (t < t_head), :], axis=0)
        s_mid = np.mean(S[(t >= t_mid*0.98) & (t <= t_mid*1.02), :], axis=0)

        o = u_head + s_head < u_mid + s_mid

        # Determine ODE parameters
        U_type, S_type = U[cell_labels == type_], S[cell_labels == type_]

        for g in range(G):
            p_low, p_high = 0.05, 0.95
            u_low = np.quantile(U_type[:, g], p_low)
            s_low = np.quantile(S_type[:, g], p_low)
            u_high = np.quantile(U_type[:, g], p_high)
            s_high = np.quantile(S_type[:, g], p_high)

            # edge cases
            while (u_high == 0 or s_high == 0) and p_high < 1.0:
                p_high += 0.01
                u_high = np.quantile(U_type[:, g], p_high)
                s_high = np.quantile(S_type[:, g], p_high)
            if u_high == 0:
                gamma[type_, g] = 0.01
                continue
            elif s_high == 0:
                gamma[type_, g] = 1.0
                continue

            mask_high = (U_type[:, g] >= u_high) | (S_type[:, g] >= s_high)
            mask_low = (U_type[:, g] <= u_low) | (S_type[:, g] <= s_low)
            mask_q = mask_high | mask_low
            u_q = U_type[mask_q, g]
            s_q = S_type[mask_q, g]
            slope = linreg_mtx(u_q-U_type[:, g].min(), s_q-S_type[:, g].min())
            if slope == 1:
                slope = 1 + 0.1*np.random.rand()
            gamma[type_, g] = np.clip(slope, 0.01, None)

        alpha[type_] = (np.quantile(U_type, 0.95, axis=0) - np.quantile(U_type, 0.05, axis=0)) * o \
            + (np.quantile(U_type, 0.95, axis=0) - np.quantile(U_type, 0.05, axis=0)) \
            * (1-o) * np.random.rand(G) * 0.001 + 1e-10
    for i, type_ in enumerate(init_types):
        mask_type = cell_labels == type_
        t_head = np.quantile(t[mask_type], 0.03)
        u0[i] = np.mean(U[(t >= t[mask_type].min()) & (t <= t_head)], axis=0)+1e-10
        s0[i] = np.mean(S[(t >= t[mask_type].min()) & (t <= t_head)], axis=0)+1e-10

    return alpha, beta, gamma, u0, s0


def get_x0_tree(par, neg_slope=0.0, eps=1e-6, **kwargs):
    # Compute initial conditions by sequentially traversing the tree
    # Returns scaled u0
    alpha, beta, gamma = kwargs['alpha'], kwargs['beta'], kwargs['gamma']  # tensor shape: (N type, G)
    t_trans = kwargs['t_trans']
    scaling = kwargs["scaling"]

    n_type, G = alpha.shape
    u0 = torch.empty(n_type, G, dtype=torch.float32, device=alpha.device)
    s0 = torch.empty(n_type, G, dtype=torch.float32, device=alpha.device)
    self_idx = torch.tensor(range(n_type), dtype=par.dtype, device=alpha.device)
    roots = torch.where(par == self_idx)[0]  # the parent of any root is itself
    u0_root, s0_root = kwargs['u0_root'], kwargs['s0_root']  # tensor shape: (n roots, G), u0 unscaled
    u0[roots] = u0_root/scaling
    s0[roots] = s0_root
    par[roots] = -1  # avoid revisiting the root in the while loop
    count = len(roots)
    progenitors = roots

    while count < n_type:
        cur_level = torch.cat([torch.where(par == x)[0] for x in progenitors])
        tau0 = F.leaky_relu(t_trans[cur_level] - t_trans[par[cur_level]], neg_slope).view(-1, 1)

        u0_hat, s0_hat = pred_su(tau0,
                                 u0[par[cur_level]],
                                 s0[par[cur_level]],
                                 alpha[par[cur_level]],
                                 beta[par[cur_level]],
                                 gamma[par[cur_level]])
        u0[cur_level] = u0_hat
        s0[cur_level] = s0_hat
        progenitors = cur_level
        count += len(cur_level)
    par[roots] = roots
    return u0, s0


def ode_br(t, y, par, neg_slope=0.0, eps=1e-6, **kwargs):
    """(PyTorch Version) Branching ODE solution.
    See the documentation of ode_br_numpy for details.
    """
    alpha, beta, gamma = kwargs['alpha'], kwargs['beta'], kwargs['gamma']  # tensor shape: (N type, G)
    t_trans = kwargs['t_trans']
    scaling = kwargs["scaling"]

    u0, s0 = get_x0_tree(par, neg_slope, **kwargs)
    # For cells with time violation, we use its parent type
    par_batch = par[y]
    mask = (t >= t_trans[y].view(-1, 1)).float()
    tau = F.leaky_relu(t - t_trans[y].view(-1, 1), neg_slope) * mask \
        + F.leaky_relu(t - t_trans[par_batch].view(-1, 1), neg_slope) * (1-mask)
    u0_batch = u0[y] * mask + u0[par_batch] * (1-mask)
    s0_batch = s0[y] * mask + s0[par_batch] * (1-mask)  # tensor shape: (N type, G)
    uhat, shat = pred_su(tau,
                         u0_batch,
                         s0_batch,
                         alpha[y] * mask + alpha[par_batch] * (1-mask),
                         beta[y] * mask + beta[par_batch] * (1-mask),
                         gamma[y] * mask + gamma[par_batch] * (1-mask))
    return uhat * scaling, shat


def get_x0_tree_numpy(par, eps=1e-6, **kwargs):
    # Compute initial conditions by sequentially traversing the tree
    # Returns scaled u0
    alpha, beta, gamma = kwargs['alpha'], kwargs['beta'], kwargs['gamma']  # tensor shape: (N type, G)
    t_trans = kwargs['t_trans']
    scaling = kwargs["scaling"]

    n_type, G = alpha.shape
    u0 = np.empty((n_type, G))
    s0 = np.empty((n_type, G))
    self_idx = np.array(range(n_type))
    roots = np.where(par == self_idx)[0]  # the parent of any root is itself
    u0_root, s0_root = kwargs['u0_root'], kwargs['s0_root']  # tensor shape: (n roots, G), u0 unscaled
    u0[roots] = u0_root/scaling
    s0[roots] = s0_root
    par[roots] = -1
    count = len(roots)
    progenitors = roots

    while count < n_type:
        cur_level = np.concatenate([np.where(par == x)[0] for x in progenitors])
        tau0 = np.clip(t_trans[cur_level] - t_trans[par[cur_level]], 0, None).reshape(-1, 1)
        u0_hat, s0_hat = pred_su_numpy(tau0,
                                       u0[par[cur_level]],
                                       s0[par[cur_level]],
                                       alpha[par[cur_level]],
                                       beta[par[cur_level]],
                                       gamma[par[cur_level]])
        u0[cur_level] = u0_hat
        s0[cur_level] = s0_hat
        progenitors = cur_level
        count += len(cur_level)
    par[roots] = roots
    return u0, s0


def ode_br_numpy(t, y, par, eps=1e-6, **kwargs):
    """
    (Numpy Version)
    Branching ODE solution.

    Args:
        t (:class:`numpy.ndarray`):
            Cell time, (N,1)
        y (:class:`numpy.ndarray`):
            Cell type, encoded in integer, (N,)
        par (:class:`numpy.ndarray`):
            Parent cell type in the transition graph, (N_type,)

    kwargs:
        alpha (:class:`numpy.ndarray`):
            Transcription rates, (cell type by gene ).
        beta (:class:`numpy.ndarray`):
            Splicing rates, (cell type by gene ).
        gamma (:class:`numpy.ndarray`):
            Degradation rates, (cell type by gene ).
        t_trans (:class:`numpy.ndarray`):
            Start time of splicing dynamics of each cell type.
        scaling (:class:`numpy.ndarray`):
            Genewise scaling factor between unspliced and spliced counts.

    Returns:
        tuple containing:

            - :class:`numpy.ndarray`: Predicted u values, (N, G)
            - :class:`numpy.ndarray`: Predicted s values, (N, G)
    """
    alpha, beta, gamma = kwargs['alpha'], kwargs['beta'], kwargs['gamma']  # array shape: (N type, G)
    t_trans = kwargs['t_trans']
    scaling = kwargs["scaling"]

    u0, s0 = get_x0_tree_numpy(par, **kwargs)
    n_type, G = alpha.shape
    uhat, shat = np.zeros((len(y), G)), np.zeros((len(y), G))
    for i in range(n_type):
        mask = (t[y == i] >= t_trans[i])
        tau = np.clip(t[y == i].reshape(-1, 1) - t_trans[i], 0, None) * mask \
            + np.clip(t[y == i].reshape(-1, 1) - t_trans[par[i]], 0, None) * (1-mask)
        uhat_i, shat_i = pred_su_numpy(tau,
                                       u0[i]*mask+u0[par[i]]*(1-mask),
                                       s0[i]*mask+s0[par[i]]*(1-mask),
                                       alpha[i]*mask+alpha[[par[i]]]*(1-mask),
                                       beta[i]*mask+beta[[par[i]]]*(1-mask),
                                       gamma[i]*mask+gamma[[par[i]]]*(1-mask))
        uhat[y == i] = uhat_i
        shat[y == i] = shat_i
    return uhat*scaling, shat


############################################################
#  KNN-Related Functions
############################################################
def _hist_equal(t, t_query, perc=0.95, n_bin=51):
    # Perform histogram equalization across all local times.
    tmax = t.max() - t.min()
    t_ub = np.quantile(t, perc)
    t_lb = t.min()
    delta_t = (t_ub - t_lb)/(n_bin-1)
    bins = [t_lb+i*delta_t for i in range(n_bin)]+[t.max()+0.01]
    pdf_t, edges = np.histogram(t, bins, density=True)
    pt, edges = np.histogram(t, bins, density=False)

    # Perform histogram equalization
    cdf_t = np.concatenate(([0], np.cumsum(pt)))
    cdf_t = cdf_t/cdf_t[-1]
    t_out = np.zeros((len(t)))
    t_out_query = np.zeros((len(t_query)))
    for i in range(n_bin):
        mask = (t >= bins[i]) & (t < bins[i+1])
        t_out[mask] = (cdf_t[i] + (t[mask]-bins[i])*pdf_t[i])*tmax
        mask_q = (t_query >= bins[i]) & (t_query < bins[i+1])
        t_out_query[mask_q] = (cdf_t[i] + (t_query[mask_q]-bins[i])*pdf_t[i])*tmax
    return t_out, t_out_query


def knnx0(U, S,
          t,
          z,
          t_query,
          z_query,
          dt,
          k,
          u0_init=None,
          s0_init=None,
          adaptive=0.0,
          std_t=None,
          forward=False,
          hist_eq=False):
    ############################################################
    # Given cell time and state, find KNN for each cell in a time window ahead of
    # it. The KNNs are used to compute the initial condition for the ODE of
    # the cell.
    # 1-2.    U,S [2D array (N,G)]
    #         Unspliced and Spliced count matrix
    # 3-4.    t,z [1D array (N)]
    #         Latent cell time and state used to build KNN
    # 5-6.    t_query [1D array (N)]
    #         Query cell time and state
    # 7.      dt [float tuple]
    #         Time window coefficient
    # 8.      k [int]
    #         Number of neighbors
    # 9-10.   u0_init, s0_init [1D array]
    #         Default initial condition when a cell has very few neighbors
    #         in its time window
    # 9.      adaptive [float]
    #         When set to positive value, neighbors will be chosen from
    #         [t-adaptive*std_t, t-adaptive*std_t+delta_t]
    # 10.     std_t [1D array (N)]
    #         Posterior standard deviation of cell time
    # 11.     forward [bool]
    #         Whether to look for ancestors or descendants
    # 12.     hist_eq [bool]
    #         Whether to perform histogram equalization to time.
    #         The purpose is to preserve more resolution in
    #         densely populated time intervals.
    ############################################################
    Nq = len(t_query)
    u0 = (np.zeros((Nq, U.shape[1])) if u0_init is None
          else np.tile(u0_init, (Nq, 1)))
    s0 = (np.zeros((Nq, S.shape[1])) if s0_init is None
          else np.tile(s0_init, (Nq, 1)))
    t0 = np.ones((Nq))*(t.min() - dt[0])
    t_knn = t

    n1 = 0
    len_avg = 0
    if hist_eq:  # time histogram equalization
        t, t_query = _hist_equal(t, t_query)
    # Used as the default u/s counts at the final time point
    t_98 = np.quantile(t, 0.98)
    p = 0.98
    while not np.any(t >= t_98) and p > 0.01:
        p = p - 0.01
        t_98 = np.quantile(t, p)
    u_end, s_end = U[t >= t_98].mean(0), S[t >= t_98].mean(0)
    for i in trange(Nq):  # iterate through every cell
        if adaptive > 0:
            dt_r, dt_l = adaptive*std_t[i], adaptive*std_t[i] + (dt[1]-dt[0])
        else:
            dt_r, dt_l = dt[0], dt[1]
        if forward:
            t_ub, t_lb = t_query[i] + dt_l, t_query[i] + dt_r
        else:
            t_ub, t_lb = t_query[i] - dt_r, t_query[i] - dt_l
        indices = np.where((t >= t_lb) & (t < t_ub))[0]  # filter out cells in the bin
        k_ = len(indices)
        delta_t = dt[1] - dt[0]  # increment / decrement of the time window boundary
        while k_ < k and t_lb > t.min() - (dt[1] - dt[0]) and t_ub < t.max() + (dt[1] - dt[0]):
            if forward:
                t_lb = t_query[i]
                t_ub = t_ub + delta_t
            else:
                t_lb = t_lb - delta_t
                t_ub = t_query[i]
            indices = np.where((t >= t_lb) & (t < t_ub))[0]  # filter out cells in the bin
            k_ = len(indices)
        len_avg = len_avg + k_
        if k_ > 0:
            k_neighbor = k if k_ > k else max(1, k_//2)
            knn_model = NearestNeighbors(n_neighbors=k_neighbor)
            knn_model.fit(z[indices])
            dist, ind = knn_model.kneighbors(z_query[i:i+1])
            u0[i] = np.mean(U[indices[ind.squeeze()].astype(int)], 0)
            s0[i] = np.mean(S[indices[ind.squeeze()].astype(int)], 0)
            t0[i] = np.mean(t_knn[indices[ind.squeeze()].astype(int)])
        else:
            if forward:
                u0[i] = u_end
                s0[i] = s_end
                t0[i] = t_98 + (t_98-t.min()) * 0.01
            n1 = n1+1
    print(f"Percentage of Invalid Sets: {n1/Nq:.3f}")
    print(f"Average Set Size: {len_avg//Nq}")
    return u0, s0, t0


def knnx0_index(t,
                z,
                t_query,
                z_query,
                dt,
                k,
                adaptive=0.0,
                std_t=None,
                forward=False,
                hist_eq=False):
    ############################################################
    # Same functionality as knnx0, but returns the neighbor index
    ############################################################
    Nq = len(t_query)
    n1 = 0
    len_avg = 0
    if hist_eq:
        t, t_query = _hist_equal(t, t_query)
    neighbor_index = []
    for i in trange(Nq):
        if adaptive > 0:
            dt_r, dt_l = adaptive*std_t[i], adaptive*std_t[i] + (dt[1]-dt[0])
        else:
            dt_r, dt_l = dt[0], dt[1]
        if forward:
            t_ub, t_lb = t_query[i] + dt_l, t_query[i] + dt_r
        else:
            t_ub, t_lb = t_query[i] - dt_r, t_query[i] - dt_l
        indices = np.where((t >= t_lb) & (t < t_ub))[0]
        k_ = len(indices)
        delta_t = dt[1] - dt[0]  # increment / decrement of the time window boundary
        while k_ < k and t_lb > t.min() - (dt[1] - dt[0]) and t_ub < t.max() + (dt[1] - dt[0]):
            if forward:
                t_lb = t_query[i]
                t_ub = t_ub + delta_t
            else:
                t_lb = t_lb - delta_t
                t_ub = t_query[i]
            indices = np.where((t >= t_lb) & (t < t_ub))[0]  # filter out cells in the bin
            k_ = len(indices)
        len_avg = len_avg + k_
        if k_ > 1:
            k_neighbor = k if k_ > k else max(1, k_//2)
            knn_model = NearestNeighbors(n_neighbors=k_neighbor)
            knn_model.fit(z[indices])
            dist, ind = knn_model.kneighbors(z_query[i:i+1])
            if isinstance(ind, int):
                ind = np.array([int])
            neighbor_index.append(indices[ind.flatten()].astype(int))
        elif k_ == 1:
            neighbor_index.append(indices)
        else:
            neighbor_index.append([])
            n1 = n1+1
    print(f"Percentage of Invalid Sets: {n1/Nq:.3f}")
    print(f"Average Set Size: {len_avg//Nq}")
    return neighbor_index


def get_x0(U,
           S,
           t,
           dt,
           neighbor_index,
           u0_init=None,
           s0_init=None,
           forward=False):
    N = len(neighbor_index)  # training + validation
    u0 = (np.zeros((N, U.shape[1])) if u0_init is None
          else np.tile(u0_init, (N, 1)))
    s0 = (np.zeros((N, S.shape[1])) if s0_init is None
          else np.tile(s0_init, (N, 1)))
    t0 = np.ones((N))*(t.min() - dt[0])
    # Used as the default u/s counts at the final time point
    t_98 = np.quantile(t, 0.98)
    p = 0.98
    while not np.any(t >= t_98) and p > 0.01:
        p = p - 0.01
        t_98 = np.quantile(t, p)
    u_end, s_end = U[t >= t_98].mean(0), S[t >= t_98].mean(0)

    for i in range(N):
        if len(neighbor_index[i]) > 0:
            u0[i] = U[neighbor_index[i]].mean(0)
            s0[i] = S[neighbor_index[i]].mean(0)
            t0[i] = t[neighbor_index[i]].mean()
        elif forward:
            u0[i] = u_end
            s0[i] = s_end
            t0[i] = t_98 + (t_98-t.min()) * 0.01
    return u0, s0, t0


def knn_transition_prob(t,
                        z,
                        t_query,
                        z_query,
                        cell_labels,
                        n_type,
                        dt,
                        k,
                        soft_assign=True):
    ############################################################
    # Compute the frequency of cell type transition based on windowed KNN.
    # Used in transition graph construction.
    ############################################################
    N, Nq = len(t), len(t_query)
    P = np.zeros((n_type, n_type))
    t0 = np.zeros((n_type))
    sigma_t = np.zeros((n_type))

    for i in range(n_type):
        t0[i] = np.quantile(t[cell_labels == i], 0.01)
        sigma_t[i] = t[cell_labels == i].std()
    if soft_assign:
        A = np.empty((N, N))
        for i in range(Nq):
            t_ub, t_lb = t_query[i] - dt[0], t_query[i] - dt[1]
            indices = np.where((t >= t_lb) & (t < t_ub))[0]
            k_ = len(indices)
            if k_ > 0:
                if k_ <= k:
                    A[i, indices] = 1
                else:
                    knn_model = NearestNeighbors(n_neighbors=k)
                    knn_model.fit(z[indices])
                    dist, ind = knn_model.kneighbors(z_query[i:i+1])
                    A[i, indices[ind.squeeze()]] = 1
        for i in range(n_type):
            for j in range(n_type):
                P[i, j] = A[cell_labels == i][:, cell_labels == j].sum()
    else:
        A = np.empty((N, n_type))
        for i in range(Nq):
            t_ub, t_lb = t_query[i] - dt[0], t_query[i] - dt[1]
            indices = np.where((t >= t_lb) & (t < t_ub))[0]
            k_ = len(indices)
            if k_ > 0:
                if k_ <= k:
                    knn_model = NearestNeighbors(n_neighbors=min(k, k_))
                    knn_model.fit(z[indices])
                    dist, ind = knn_model.kneighbors(z_query[i:i+1])
                    knn_label = cell_labels[indices][ind.squeeze()]
                else:
                    knn_label = cell_labels[indices]
                n_par = np.array([np.sum(knn_label == i) for i in range(n_type)])
                A[i, np.argmax(n_par)] = 1
        for i in range(n_type):
            P[i] = A[cell_labels == i].sum(0)
    psum = P.sum(1)
    psum[psum == 0] = 1
    return P/(psum.reshape(-1, 1))

############################################################
#  ELBO term related to categorical variables in BasisVAE
#  Referece:
#  Märtens, K. &amp; Yau, C.. (2020). BasisVAE: Translation-
#  invariant feature-level clustering with Variational Autoencoders.
#  Proceedings of the Twenty Third International Conference on
#  Artificial Intelligence and Statistics, in Proceedings of
#  Machine Learning Research</i> 108:2928-2937
#  Available from https://proceedings.mlr.press/v108/martens20b.html.
############################################################


def elbo_collapsed_categorical(logits_phi, alpha, K, N):
    phi = torch.softmax(logits_phi, dim=1)

    if alpha.ndim == 1:
        sum_alpha = alpha.sum()
        pseudocounts = phi.sum(dim=0)  # n basis
        term1 = torch.lgamma(sum_alpha) - torch.lgamma(sum_alpha + N)
        term2 = (torch.lgamma(alpha + pseudocounts) - torch.lgamma(alpha)).sum()
    else:
        sum_alpha = alpha.sum(axis=-1)  # n vars
        pseudocounts = phi.sum(dim=0)
        term1 = (torch.lgamma(sum_alpha) - torch.lgamma(sum_alpha + N)).mean(0)
        term2 = (torch.lgamma(alpha + pseudocounts) - torch.lgamma(alpha)).mean(0).sum()

    E_q_logq = (phi * torch.log(phi + 1e-16)).sum()

    return -term1 - term2 + E_q_logq


def entropy(logits_phi):
    phi = torch.softmax(logits_phi, dim=1)
    return (phi * torch.log(phi + 1e-16)).sum()

############################################################
# Other Auxilliary Functions
############################################################


def get_gene_index(genes_all, gene_list):
    gind = []
    gremove = []
    for gene in gene_list:
        matches = np.where(genes_all == gene)[0]
        if len(matches) == 1:
            gind.append(matches[0])
        elif len(matches) == 0:
            print(f'Warning: Gene {gene} not found! Ignored.')
            gremove.append(gene)
        else:
            gind.append(matches[0])
            print('Warning: Gene {gene} has multiple matches. Pick the first one.')
    gene_list = list(gene_list)
    for gene in gremove:
        gene_list.remove(gene)
    return gind, gene_list


def convert_time(t):
    """Convert the time in sec into the format: hour:minute:second
    """
    hour = int(t//3600)
    minute = int((t - hour*3600)//60)
    second = int(t - hour*3600 - minute*60)

    return f"{hour:3d} h : {minute:2d} m : {second:2d} s"


def sample_genes(adata, n, key, mode='top', q=0.5):
    if mode == 'random':
        return np.random.choice(adata.var_names, n, replace=False)
    val_sorted = adata.var[key].sort_values(ascending=False)
    genes_sorted = val_sorted.index.to_numpy()
    if mode == 'quantile':
        N = np.sum(val_sorted.to_numpy() >= q)
        return np.random.choice(genes_sorted[:N], min(n, N), replace=False)
    return genes_sorted[:n]


def add_capture_time(adata, tkey, save_key="tprior"):
    capture_time = adata.obs[tkey].to_numpy()
    if isinstance(capture_time[0], str):
        j = 0
        while not (capture_time[0][j] >= '0' and capture_time[0][j] >= '9'):
            j = j+1
        tprior = np.array([float(x[1:]) for x in capture_time])
    else:
        tprior = capture_time
    tprior = tprior - tprior.min() + 0.01
    adata.obs["tprior"] = tprior


def add_cell_cluster(adata, cluster_key, save_key="clusters"):
    cell_labels = adata.obs[cluster_key].to_numpy()
    adata.obs["clusters"] = np.array([str(x) for x in cell_labels])

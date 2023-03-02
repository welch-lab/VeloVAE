import numpy as np
from scipy.ndimage import gaussian_filter1d
from .model_util import ode_numpy, ode_br_numpy, pred_su_numpy


def rna_velocity_vanillavae(adata,
                            key,
                            use_raw=False,
                            use_scv_genes=False,
                            k=10,
                            return_copy=False):
    """Compute the velocity based on:
       du/dt = alpha - beta * u, ds/dt = beta * u - gamma * s

    Arguments
    ---------
    adata : :class:`anndata.AnnData`
    key : str
        key used for extracting ODE parameters
    use_raw : bool, optional
        whether to use the (noisy) input count to compute the velocity
    use_scv_genes : bool, optional
        whether to compute velocity only for genes scVelo fits

    Returns (only if `return_copy`=True)
    -------
    Vu : `numpy array`
        velocity of u
    V : `numpy array`
        velocity of s
    U : `numpy array`
        predicted u values
    S : `numpy array`
        predicted s values
    """
    alpha = adata.var[f"{key}_alpha"].to_numpy()
    beta = adata.var[f"{key}_beta"].to_numpy()
    gamma = adata.var[f"{key}_gamma"].to_numpy()
    t = adata.obs[f"{key}_time"].to_numpy()
    ton = adata.var[f"{key}_ton"].to_numpy()
    toff = adata.var[f"{key}_toff"].to_numpy()

    if use_raw:
        U, S = adata.layers['Mu'], adata.layers['Ms']
    else:
        scaling = adata.var[f"{key}_scaling"].to_numpy()
        if f"{key}_uhat" in adata.layers and f"{key}_shat" in adata.layers:
            U, S = adata.layers[f"{key}_uhat"], adata.layers[f"{key}_shat"]
            U = U/scaling
        else:
            U, S = ode_numpy(t.reshape(-1, 1),
                             alpha,
                             beta,
                             gamma,
                             ton,
                             toff,
                             None)  # don't need scaling here
            adata.layers["Uhat"] = U*scaling
            adata.layers["Shat"] = S
    # smooth transition at the switch-on time
    soft_coeff = 1/(1+np.exp(-(t.reshape(-1, 1) - ton)*k))

    Vu = alpha * ((t.reshape(-1, 1) >= ton) & (t.reshape(-1, 1) <= toff)) \
        - beta * U
    V = (beta * U - gamma * S)*soft_coeff

    adata.layers[f"{key}_velocity_u"] = Vu
    adata.layers[f"{key}_velocity"] = V
    if use_scv_genes:
        gene_mask = np.isnan(adata.var['fit_scaling'].to_numpy())
        Vu[:, gene_mask] = np.nan
        V[:, gene_mask] = np.nan
    if return_copy:
        return Vu, V, U, S


def rna_velocity_vae(adata,
                     key,
                     use_raw=False,
                     use_scv_genes=False,
                     sigma=None,
                     approx=False,
                     full_vb=False,
                     return_copy=False):
    """Compute the velocity based on:
       du/dt = rho * alpha - beta * u, ds/dt = beta * u - gamma * s

    Arguments
    ---------
    adata : :class:`anndata.AnnData`
    key : str
        key used for extracting ODE parameters
    use_raw : bool, optional
        whether to use the (noisy) input count to compute the velocity
    use_scv_genes : bool, optional
        whether to compute velocity only for genes scVelo fits
    sigma : float, optional
        Parameter used in Gaussian filtering of velocity values.
    apprx : bool, optional
        Whether to use linear approximation to compute velocity
    full_vb : bool, optional
        Whether the model is full VB

    Returns (only if `return_copy`=True)
    -------
    Vu : `numpy array`
        velocity of u
    V : `numpy array`
        velocity of s
    U : `numpy array`
        predicted u values
    S : `numpy array`
        predicted s values
    """
    alpha = np.exp(adata.var[f"{key}_logmu_alpha"].to_numpy()) if full_vb\
        else adata.var[f"{key}_alpha"].to_numpy()
    rho = adata.layers[f"{key}_rho"]
    beta = np.exp(adata.var[f"{key}_logmu_beta"].to_numpy()) if full_vb\
        else adata.var[f"{key}_beta"].to_numpy()
    gamma = np.exp(adata.var[f"{key}_logmu_gamma"].to_numpy()) if full_vb\
        else adata.var[f"{key}_gamma"].to_numpy()
    t = adata.obs[f"{key}_time"].to_numpy()
    t0 = adata.obs[f"{key}_t0"].to_numpy()
    U0 = adata.layers[f"{key}_u0"]
    S0 = adata.layers[f"{key}_s0"]

    if use_raw:
        U, S = adata.layers['Mu'], adata.layers['Ms']
    else:
        scaling = adata.var[f"{key}_scaling"].to_numpy()
        if f"{key}_uhat" in adata.layers and f"{key}_shat" in adata.layers:
            U, S = adata.layers[f"{key}_uhat"], adata.layers[f"{key}_shat"]
            U = U/scaling
        else:
            U, S = pred_su_numpy(np.clip(t-t0, 0, None).reshape(-1, 1),
                                 U0/scaling,
                                 S0,
                                 alpha*rho,
                                 beta, gamma)
            U, S = np.clip(U, 0, None), np.clip(S, 0, None)
            adata.layers["Uhat"] = U * scaling
            adata.layers["Shat"] = S
    if approx:
        V = (S - S0)/((t - t0).reshape(-1, 1))
        Vu = (U - U0)/((t - t0).reshape(-1, 1))
    else:
        V = (beta * U - gamma * S)
        Vu = rho * alpha - beta * U
    if sigma is not None:
        time_order = np.argsort(t)
        V[time_order] = gaussian_filter1d(V[time_order], sigma,
                                          axis=0, mode="nearest")
        Vu[time_order] = gaussian_filter1d(Vu[time_order], sigma,
                                           axis=0, mode="nearest")
    adata.layers[f"{key}_velocity"] = V
    adata.layers[f"{key}_velocity_u"] = Vu
    if use_scv_genes:
        gene_mask = np.isnan(adata.var['fit_scaling'].to_numpy())
        V[:, gene_mask] = np.nan
    if return_copy:
        return Vu, V, U, S


def rna_velocity_brode(adata, key, use_raw=False, use_scv_genes=False, k=10.0):
    """Compute the velocity based on: ds/dt = beta * u - gamma * s where
    u and s are predicted by branching ODE

    Arguments
    ---------
    adata : :class:`anndata.AnnData`
    key : str
        key used for extracting ODE parameters
    use_raw : bool, optional
        whether to use the (noisy) input count to compute the velocity
    use_scv_genes : bool, optional
        whether to compute velocity only for genes scVelo fits
    k : float, optional
        Parameter used in soft clipping of time duration

    Returns
    -------
    V : `numpy array`
        velocity
    U : `numpy array`
        predicted u values
    S : `numpy array`
        predicted s values
    """
    alpha = adata.varm[f"{key}_alpha"].T
    beta = adata.varm[f"{key}_beta"].T
    gamma = adata.varm[f"{key}_gamma"].T
    t_trans = adata.uns[f"{key}_t_trans"]
    u0 = adata.varm[f"{key}_u0"].T
    s0 = adata.varm[f"{key}_s0"].T
    scaling = adata.var[f"{key}_scaling"].to_numpy()
    w = adata.uns[f"{key}_w"]
    parents = np.argmax(w, 1)

    t = adata.obs[f"{key}_time"].to_numpy()
    y = adata.obs[f"{key}_label"].to_numpy()

    if use_raw:
        U, S = adata.layers['Mu'], adata.layers['Ms']
    else:
        if f"{key}_uhat" in adata.layers and f"{key}_shat" in adata.layers:
            U, S = adata.layers[f"{key}_uhat"], adata.layers[f"{key}_shat"]
            U = U/scaling
        else:
            U, S = ode_br_numpy(t.reshape(-1, 1),
                                y,
                                np.argmax(w, 1),
                                alpha=alpha,
                                beta=beta,
                                gamma=gamma,
                                t_trans=t_trans,
                                u0=u0,
                                s0=s0)
            adata.layers["Uhat"] = U
            adata.layers["Shat"] = S

    V = np.zeros(S.shape)
    for i in range(alpha.shape[0]):
        denom = 1+np.exp(-(t[y == i].reshape(-1, 1) - t_trans[parents[i]])*k)
        soft_coeff = 1 / denom  # smooth transition at the switch-on time
        V[y == i] = (beta[i]*U[y == i] - gamma[i]*S[y == i]) * soft_coeff
    adata.layers[f"{key}_velocity"] = V
    if use_scv_genes:
        gene_mask = np.isnan(adata.var['fit_scaling'].to_numpy())
        V[:, gene_mask] = np.nan
    return V, U, S


def rna_velocity_cyclevae(adata,
                          key,
                          use_raw=False,
                          use_scv_genes=False,
                          k=10,
                          return_copy=False):
    """Compute the velocity based on:
       du/dt = alpha - beta * u, ds/dt = beta * u - gamma * s

    Arguments
    ---------
    adata : :class:`anndata.AnnData`
    key : str
        key used for extracting ODE parameters
    use_raw : bool, optional
        whether to use the (noisy) input count to compute the velocity
    use_scv_genes : bool, optional
        whether to compute velocity only for genes scVelo fits

    Returns (only if `return_copy`=True)
    -------
    Vu : `numpy array`
        velocity of u
    V : `numpy array`
        velocity of s
    U : `numpy array`
        predicted u values
    S : `numpy array`
        predicted s values
    """
    alpha = adata.var[f"{key}_alpha"].to_numpy()
    beta = adata.var[f"{key}_beta"].to_numpy()
    gamma = adata.var[f"{key}_gamma"].to_numpy()
    theta = adata.obs[f"{key}_phase"].to_numpy()
    theta_on = adata.var[f"{key}_phase_on"].to_numpy()
    theta_off = adata.var[f"{key}_phase_off"].to_numpy()

    if use_raw:
        U, S = adata.layers['Mu'], adata.layers['Ms']
    else:
        scaling = adata.var[f"{key}_scaling"].to_numpy()
        if f"{key}_uhat" in adata.layers and f"{key}_shat" in adata.layers:
            U, S = adata.layers[f"{key}_uhat"], adata.layers[f"{key}_shat"]
            U = U/scaling
        else:
            t = adata.uns['tmax'] * \
                (1 - np.cos(np.clip(theta.reshape(-1, 1)-theta_on, 0, None))) \
                / 2
            toff = adata.uns['tmax'] * \
                (1 - np.cos(np.clip(theta_off-theta_on, 0, None))) / 2
            U, S = ode_numpy(t,
                             alpha,
                             beta,
                             gamma,
                             0,
                             toff,
                             None)  # don't need scaling here
            adata.layers["Uhat"] = U*scaling
            adata.layers["Shat"] = S
    # smooth transition at the switch-on time
    soft_coeff = 1/(1+np.exp(-(theta.reshape(-1, 1) - theta_on)*k))

    dt_dtheta = adata.uns['tmax']\
        * np.sin(np.clip(theta.reshape(-1, 1)-theta_on, 0, None))*0.5
    alpha_mask = (theta.reshape(-1, 1) >= theta_on)\
        & (theta.reshape(-1, 1) <= theta_off)
    Vu = (alpha * alpha_mask - beta * U) * dt_dtheta
    V = (beta * U - gamma * S) * soft_coeff * dt_dtheta

    adata.layers[f"{key}_velocity_u"] = Vu
    adata.layers[f"{key}_velocity"] = V
    if use_scv_genes:
        gene_mask = np.isnan(adata.var['fit_scaling'].to_numpy())
        Vu[:, gene_mask] = np.nan
        V[:, gene_mask] = np.nan
    if return_copy:
        return Vu, V, U, S


def smooth_vel(v, t, W=5):
    order_t = np.argsort(t)
    h = np.ones((W))*(1/W)
    v_ret = np.zeros((len(v)))
    v_ret[order_t] = np.convolve(v[order_t], h, mode='same')
    return v_ret

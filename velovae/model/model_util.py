from copy import deepcopy
import numpy as np
import os
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from .scvelo_util import mRNA, vectorize, tau_inv, R_squared, test_bimodality, leastsq_NxN
from sklearn.neighbors import NearestNeighbors
import pynndescent

"""
Dynamical Model

Reference: 
Bergen, V., Lange, M., Peidli, S., Wolf, F. A., & Theis, F. J. (2020). 
Generalizing RNA velocity to transient cell states through dynamical modeling. 
Nature biotechnology, 38(12), 1408-1414.
"""
def scv_pred_single(t,alpha,beta,gamma,ts,scaling=1.0, uinit=0, sinit=0):
    #Predicts u and s using the dynamical model.
    
    beta = beta*scaling
    tau, alpha, u0, s0 = vectorize(t, ts, alpha, beta, gamma, u0=uinit, s0=sinit)
    tau = np.clip(tau,a_min=0,a_max=None)
    ut, st = mRNA(tau, u0, s0, alpha, beta, gamma)
    ut = ut*scaling
    return ut.squeeze(), st.squeeze()

def scv_pred(adata, key, glist=None):
	#Reproduce the full prediction of scvelo dynamical model
	
	ngene = len(glist) if glist is not None else adata.n_vars
	ncell = adata.n_obs
	ut, st = np.ones((adata.n_obs,ngene))*np.nan, np.ones((adata.n_obs,ngene))*np.nan
	if(glist is None):
		glist = adata.var_names.to_numpy()
	
	for i in range(ngene):
		idx = np.where(adata.var_names==glist[i])[0][0]
		item = adata.var.loc[glist[i]]
		if(len(item)==0):
			print('Gene '+glist[i]+' not found!')
			continue
			
		alpha, beta, gamma, scaling = item[f'{key}_alpha'], item[f'{key}_beta'], item[f'{key}_gamma'], item[f'{key}_scaling']
		ts = item[f'{key}_t_']
		scaling = item[f'{key}_scaling']
		t = adata.layers[f'{key}_t'][:,idx]
		if(np.isnan(alpha)):
			continue
		u_g, s_g = scv_pred_single(t,alpha,beta,gamma,ts,scaling)
		
		ut[:,i] = u_g
		st[:,i] = s_g
    
	return ut, st

############################################################
#Shared among all VAEs
############################################################
def hist_equal(t, Tmax, perc=0.95, Nbin=101):
    #Perform histogram equalization across all local times.
    
    t_ub = np.quantile(t, perc)
    t_lb = t.min()
    delta_t = (t_ub - t_lb)/(Nbin-1)
    bins = [t_lb+i*delta_t for i in range(Nbin)]+[t.max()] 
    pdf_t, edges = np.histogram(t, bins, density=True)
    pt, edges = np.histogram(t, bins, density=False)
    
    #Perform histogram equalization
    cdf_t = np.concatenate(([0], np.cumsum(pt)))
    cdf_t = cdf_t/cdf_t[-1]
    t_out = np.zeros((len(t)))
    for i in range(Nbin):
        mask = (t>=bins[i]) & (t<bins[i+1])
        t_out[mask] = (cdf_t[i] + (t[mask]-bins[i])*pdf_t[i])*Tmax
    return t_out

############################################################
#Basic utility function to compute ODE solutions for all models
############################################################
def pred_su_numpy(tau, u0, s0, alpha, beta, gamma):
    ############################################################
    #(Numpy Version)
    #Analytical solution of the ODE
    #tau: [B x 1] or [B x 1 x 1] time duration starting from the switch-on time of each gene.
    #u0, s0: [G] or [N type x G] initial conditions
    #alpha, beta, gamma: [G] or [N type x G] generation, splicing and degradation rates
    ############################################################
    
    unstability = (np.abs(beta-gamma) < 1e-6)
    expb, expg = np.exp(-beta*tau), np.exp(-gamma*tau)
    
    upred = u0*expb+alpha/beta*(1-expb)
    spred = s0*expg+alpha/gamma*(1-expg)+(alpha-beta*u0)/(gamma-beta+1e-6)*(expg-expb)*(1-unstability)-(alpha-beta*u0)*tau*expg*unstability
    return np.clip(upred, a_min=0, a_max=None), np.clip(spred, a_min=0, a_max=None)


def pred_su(tau, u0, s0, alpha, beta, gamma):
    ############################################################
    #(PyTorch Version)
    #Analytical solution of the ODE
    #tau: [B x 1] or [B x 1 x 1] time duration starting from the switch-on time of each gene.
    #u0, s0: [G] or [N type x G] initial conditions
    #alpha, beta, gamma: [G] or [N type x G] generation, splicing and degradation rates
    ############################################################
    
    expb, expg = torch.exp(-beta*tau), torch.exp(-gamma*tau)
    eps = 1e-6
    unstability = (torch.abs(beta-gamma) < eps).long()
    
    upred = u0*expb+alpha/beta*(1-expb)
    spred = s0*expg+alpha/gamma*(1-expg)+(alpha-beta*u0)/(gamma-beta+eps)*(expg-expb)*(1-unstability)-(alpha-beta*u0)*tau*expg*unstability
    return nn.functional.relu(upred), nn.functional.relu(spred)

"""
Initialization Methods

Reference: 
Bergen, V., Lange, M., Peidli, S., Wolf, F. A., & Theis, F. J. (2020). 
Generalizing RNA velocity to transient cell states through dynamical modeling. 
Nature biotechnology, 38(12), 1408-1414.
"""
def scale_by_gene(U,S,train_idx=None,mode='scale_u'):
    #mode
    #   'auto' means to scale the one with a smaller range
    #   'scale_u' means to match std(u) with std(s)
    #   'scale_s' means to match std(s) with std(u)
    G = U.shape[1]
    scaling_u = np.ones((G))
    scaling_s = np.ones((G))
    std_u, std_s = np.ones((G)),np.ones((G))
    for i in range(G):
        if(train_idx is None):
            si, ui = S[:,i], U[:,i]
        else:
            si, ui = S[train_idx,i], U[train_idx,i]
        sfilt, ufilt = si[(si>0) & (ui>0)], ui[(si>0) & (ui>0)] #Use only nonzero data points
        if(len(sfilt)>3 and len(ufilt)>3):
            std_u[i] = np.std(ufilt)
            std_s[i] = np.std(sfilt)
    mask_u, mask_s = (std_u==0), (std_s==0)
    std_u = std_u + (mask_u & (~mask_s))*std_s + (mask_u & mask_s)*1
    std_s = std_s + ((~mask_u) & mask_s)*std_u + (mask_u & mask_s)*1
    if(mode=='auto'):
        scaling_u = np.max(np.stack([scaling_u,(std_u/std_s)]),0)
        scaling_s = np.max(np.stack([scaling_s,(std_s/std_u)]),0) 
    elif(mode=='scale_u'):
        scaling_u = std_u/std_s
    elif(mode=='scale_s'):
        scaling_s = std_s/std_u
    return U/scaling_u, S/scaling_s, scaling_u, scaling_s

def get_gene_scale(U,S,train_idx=None,mode='scale_u'):
    #mode
    #   'auto' means to scale the one with a smaller range
    #   'scale_u' means to match std(u) with std(s)
    #   'scale_s' means to match std(s) with std(u)
    G = U.shape[1]
    scaling_u = np.ones((G))
    scaling_s = np.ones((G))
    std_u, std_s = np.ones((G)),np.ones((G))
    for i in range(G):
        if(train_idx is None):
            si, ui = S[:,i], U[:,i]
        else:
            si, ui = S[train_idx,i], U[train_idx,i]
        sfilt, ufilt = si[(si>0) & (ui>0)], ui[(si>0) & (ui>0)] #Use only nonzero data points
        if(len(sfilt)>3 and len(ufilt)>3):
            std_u[i] = np.std(ufilt)
            std_s[i] = np.std(sfilt)
    mask_u, mask_s = (std_u==0), (std_s==0)
    std_u = std_u + (mask_u & (~mask_s))*std_s + (mask_u & mask_s)*1
    std_s = std_s + ((~mask_u) & mask_s)*std_u + (mask_u & mask_s)*1
    if(mode=='auto'):
        scaling_u = np.max(np.stack([scaling_u,(std_u/std_s)]),0)
        scaling_s = np.max(np.stack([scaling_s,(std_s/std_u)]),0) 
    elif(mode=='scale_u'):
        scaling_u = std_u/std_s
    elif(mode=='scale_s'):
        scaling_s = std_s/std_u
    return scaling_u, scaling_s

def scale_by_cell(U,S,train_idx=None,separate_us_scale=True):
    N = U.shape[0]
    nu, ns = U.sum(1, keepdims=True), S.sum(1, keepdims=True)
    if(separate_us_scale):
        norm_count = (np.median(nu), np.median(ns)) if train_idx is None else (np.median(nu[train_idx]), np.median(ns[train_idx]))
        lu = nu/norm_count[0]
        ls = ns/norm_count[1]
    else:
        norm_count = np.median(nu+ns) if train_idx is None else np.median(nu[train_idx]+ns[train_idx])
        lu = (nu+ns)/norm_count
        ls = lu
    return U/lu, S/ls, lu, ls

def get_cell_scale(U,S,train_idx=None,separate_us_scale=True):
    N = U.shape[0]
    nu, ns = U.sum(1, keepdims=True), S.sum(1, keepdims=True)
    if(separate_us_scale):
        norm_count = (np.median(nu), np.median(ns)) if train_idx is None else (np.median(nu[train_idx]), np.median(ns[train_idx]))
        lu = nu/norm_count[0]
        ls = ns/norm_count[1]
    else:
        norm_count = np.median(nu+ns) if train_idx is None else np.median(nu[train_idx]+ns[train_idx])
        lu = (nu+ns)/norm_count
        ls = lu
    return lu, ls

def linreg(u, s):
    q = np.sum(s*s)
    r = np.sum(u*s)
    k = r/q
    if np.isinf(k) or np.isnan(k):
        k = 1.0+np.random.rand()
    return k
    

def init_gene(s,u,percent,fit_scaling=False,Ntype=None):
    #Adopted from scvelo
    
    std_u, std_s = np.std(u), np.std(s)
    scaling = std_u / std_s if fit_scaling else 1.0
    u = u/scaling
    
    #Pick Quantiles
    # initialize beta and gamma from extreme quantiles of s
    mask_s = s >= np.percentile(s, percent, axis=0)
    mask_u = u >= np.percentile(u, percent, axis=0)
    mask = mask_s & mask_u
    if(not np.any(mask)):
        mask = mask_s
    
    
    #Initialize alpha, beta and gamma
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
    steady_s = means_s[1]
    if pval_steady < 1e-3:
        u_inf = np.mean([u_inf, steady_u])
        alpha = gamma * s_inf
        beta = alpha / u_inf
        u0_, s0_ = u_inf, s_inf
    t_ = tau_inv(u0_, s0_, 0, 0, alpha, beta, gamma) #time to reach steady state
    tau = tau_inv(u, s, 0, 0, alpha, beta, gamma) #induction
    tau = np.clip(tau, 0, t_)
    tau_ = tau_inv(u, s, u0_, s0_, 0, beta, gamma) #repression
    tau_ = np.clip(tau_, 0, np.max(tau_[s > 0]))
    ut, st = mRNA(tau, 0, 0, alpha, beta, gamma)
    ut_, st_ = mRNA(tau_, u0_, s0_, 0, beta, gamma)
    distu, distu_ = (u - ut), (u - ut_)
    dists, dists_ = (s - st), (s - st_)
    res = np.array([distu ** 2 + dists ** 2, distu_ ** 2 + dists_ ** 2])
    t = np.array([tau,tau_+np.ones((len(tau_)))*t_])
    o = np.argmin(res, axis=0)
    t_latent = np.array([t[o[i],i] for i in range(len(tau))])
    
    
    return alpha, beta, gamma, t_latent, u0_, s0_, t_, scaling
    
def init_params(data, percent,fit_offset=False,fit_scaling=True):
    #Adopted from SCVELO
    #Use the steady-state model to estimate alpha, beta,
    #gamma and the latent time
    #data: ncell x (2*ngene) tensor
    #percent: percentage limit to pick the data
    #Output: a ncellx4 2D array of parameters
    
    ngene = data.shape[1]//2
    u = data[:,:ngene]
    s = data[:,ngene:]
    
    
    params = np.ones((ngene,4)) #four parameters: alpha, beta, gamma, scaling
    params[:,0] = np.random.rand((ngene))*np.max(u,0)
    params[:,2] = np.random.rand((ngene))*np.max(u,0)/(np.max(s,0)+1e-10)
    T = np.zeros((ngene, len(s)))
    Ts = np.zeros((ngene))
    U0, S0 = np.zeros((ngene)), np.zeros((ngene)) #Steady-1 State
    
    for i in range(ngene):
        si, ui = s[:,i], u[:,i]
        sfilt, ufilt = si[(si>0) & (ui>0)], ui[(si>0) & (ui>0)] #Use only nonzero data points
        if(len(sfilt)>3 and len(ufilt)>3):
            alpha, beta, gamma, t, u0_, s0_, ts, scaling = init_gene(sfilt,ufilt,percent,fit_scaling)
            params[i,:] = np.array([alpha,beta,gamma,scaling])
            T[i, (si>0) & (ui>0)] = t
            U0[i] = u0_
            S0[i] = s0_
            Ts[i] = ts
        else:
            U0[i] = np.max(u)
            S0[i] = np.max(s)
    
    #Filter out genes
    min_r2 = 0.01
    offset, gamma = leastsq_NxN(s,u,fit_offset,perc=[100-percent,percent])
    residual = u-gamma*s
    if(fit_offset):
        residual -= offset
    r2 = R_squared(residual, total=u-u.mean(0))
    velocity_genes = (r2>min_r2) & (gamma>0.01) & (np.max(s > 0, 0) > 0) & (np.max(u > 0, 0) > 0)
    
    dist_u, dist_s = np.zeros(u.shape),np.zeros(s.shape)
    for i in range(ngene):
        upred, spred = scv_pred_single(T[i],params[i,0],params[i,1],params[i,2],Ts[i],params[i,3]) #upred has the original scale
        dist_u[:,i] = u[:,i] - upred
        dist_s[:,i] = s[:,i] - spred
    
    sigma_u = np.clip( np.std(dist_u, 0), 0.1, None)
    sigma_s = np.clip( np.std(dist_s, 0), 0.1, None)
    sigma_u[np.isnan(sigma_u)] = 0.1
    sigma_s[np.isnan(sigma_s)] = 0.1
    
    #Make sure all genes get the same total relevance score
    Rscore = ((u>0) & (s>0))*np.ones(u.shape) + ((u==0) & (s==0))*np.ones(u.shape)*0.02 + ((u==0) & (s>0))*np.ones(u.shape)*0.1 + ((u>0) & (s==0))*np.ones(u.shape)*0.1
     
    return params[:,0], params[:,1], params[:,2], params[:,3], Ts, U0, S0, sigma_u, sigma_s, T.T, Rscore

"""
Initialization for raw read counts
"""
from scipy.special import loggamma
def logp_poisson(x, rate):
    return x*np.log(rate)-rate-loggamma(x+1)

def init_gene_raw(s,u,percent,fit_scaling=False,poisson_model=True,kde=True):
    #Adopted from scvelo
    
    std_u, std_s = np.std(u), np.std(s)
    scaling = std_u / std_s if fit_scaling else 1.0
    u = u/scaling
    
    
    #Pick Quantiles
    # initialize beta and gamma from extreme quantiles of s
    mask_s = s >= max(1, np.percentile(s, percent, axis=0))
    mask_u = u >= max(1, np.percentile(u, percent, axis=0))
    mask = mask_s & mask_u
    if(not np.any(mask)):
        mask = mask_u | mask_s
    
    
    #Initialize alpha, beta and gamma
    beta = 1
    gamma = linreg(u[mask], s[mask]) + 1e-6
    if gamma < 0.05 / scaling:
        gamma *= 1.2
    elif gamma > 1.5 / scaling:
        gamma /= 1.2
    
    u_inf, s_inf = u[mask].mean(), s[mask].mean()
    u0_, s0_ = u_inf, s_inf
    alpha = u_inf*beta
    if(alpha==0):
        print(np.percentile(s, percent, axis=0), np.percentile(u, percent, axis=0))
        print(u[mask], s[mask])
        
    # initialize switching from u quantiles and alpha from s quantiles
    tstat_u, pval_u, means_u = test_bimodality(u, kde=kde)
    tstat_s, pval_s, means_s = test_bimodality(s, kde=kde)
    pval_steady = max(pval_u, pval_s)
    steady_u = means_u[1]
    steady_s = means_s[1]
    if pval_steady < 1e-3:
        u_inf = np.mean([u_inf, steady_u])
        alpha = gamma * s_inf
        beta = alpha / u_inf
        u0_, s0_ = u_inf, s_inf
    t_ = tau_inv(u0_, s0_, 0, 0, alpha, beta, gamma) #time to reach steady state
    tau = tau_inv(u, s, 0, 0, alpha, beta, gamma) #induction
    tau = np.clip(tau, 0, t_)
    tau_ = tau_inv(u, s, u0_, s0_, 0, beta, gamma) #repression
    tau_ = np.clip(tau_, 0, np.max(tau_[s > 0]))
    t = np.array([tau,tau_+np.ones((len(tau_)))*t_])
    
    ut, st = mRNA(tau, 0, 0, alpha, beta, gamma)
    ut_, st_ = mRNA(tau_, u0_, s0_, 0, beta, gamma)
    
    if(poisson_model):
        logp_u, logp_u_ = logp_poisson(u, ut), logp_poisson(u, ut_)
        logp_s, logp_s_ = logp_poisson(s, st), logp_poisson(s, st_)
        logp = np.array([logp_u+logp_s, logp_u_+logp_s_])
        o = np.argmax(logp, axis=0)
    else:
        distu, distu_ = (u - ut), (u - ut_)
        dists, dists_ = (s - st), (s - st_)
        res = np.array([distu ** 2 + dists ** 2, distu_ ** 2 + dists_ ** 2])
        o = np.argmin(res, axis=0)
    
    t_latent = np.array([t[o[i],i] for i in range(len(tau))])
    
    
    return alpha, beta, gamma, t_latent, u0_, s0_, t_, scaling
    
def init_params_raw(data, percent, fit_offset=False, fit_scaling=True):
    #Adopted from SCVELO
    #Use the steady-state model to estimate alpha, beta,
    #gamma and the latent time
    #data: ncell x (2*ngene) tensor
    #percent: percentage limit to pick the data
    #Output: a ncellx4 2D array of parameters
    
    ngene = data.shape[1]//2
    u = data[:,:ngene]
    s = data[:,ngene:]
    
    
    params = np.ones((ngene,4)) #four parameters: alpha, beta, gamma, scaling
    params[:,0] = np.random.rand((ngene))*np.max(u,0)
    params[:,2] = np.random.rand((ngene))*np.max(u,0)/(np.max(s,0)+1e-10)
    T = np.zeros((ngene, len(s)))
    Ts = np.zeros((ngene))
    U0, S0 = np.zeros((ngene)), np.zeros((ngene)) #Steady-1 State
    
    for i in range(ngene):
        si, ui = s[:,i], u[:,i]
        sfilt, ufilt = si[(si>0) & (ui>0)], ui[(si>0) & (ui>0)] #Use only nonzero data points
        if(len(sfilt)>3 and len(ufilt)>3):
            alpha, beta, gamma, t, u0_, s0_, ts, scaling = init_gene_raw(si,ui,percent,fit_scaling)
            params[i,:] = np.array([alpha,beta,gamma,scaling])
            T[i] = t
            U0[i] = u0_
            S0[i] = s0_
            Ts[i] = ts
        else:
            U0[i] = np.max(u)
            S0[i] = np.max(s)
     
    return params[:,0], params[:,1], params[:,2], params[:,3], Ts, U0, S0, T.T




    
"""
Reinitialization based on the global time
"""
def get_ts_global(tgl, U, S, perc):
    #Initialize the transition time in the original ODE model.
    
    tsgl = np.zeros((U.shape[1]))
    for i in range(U.shape[1]):
        u,s = U[:,i],S[:,i]
        zero_mask = (u>0) & (s>0)
        mask_u, mask_s = u>=np.percentile(u,perc),s>=np.percentile(s,perc)
        tsgl[i] = np.median(tgl[mask_u & mask_s & zero_mask])
        if(np.isnan(tsgl[i])):
            tsgl[i] = np.median(tgl[(mask_u | mask_s) & zero_mask])
        if(np.isnan(tsgl[i])):
            tsgl[i] = np.median(tgl)
    assert not np.any(np.isnan(tsgl))
    return tsgl



def reinit_gene(u,s,t,ts):
    #Applied to the regular ODE 
    #Initialize the ODE parameters (alpha,beta,gamma,t_on) from
    #input data and estimated global cell time.
    
    #u1, u2: picked from induction
    mask1_u = u>np.quantile(u,0.95)
    mask1_s = s>np.quantile(s,0.95)
    u1, s1 = np.median(u[mask1_u | mask1_s]), np.median(s[mask1_s | mask1_u])
    
    if(u1 == 0 or np.isnan(u1)):
        u1 = np.max(u)
    if(s1 == 0 or np.isnan(s1)):
        s1 = np.max(s)
    
    
    t1 = np.median(t[mask1_u | mask1_s])
    if(t1 <= 0):
        tm = np.max(t[mask1_u | mask1_s])
        t1 = tm if tm>0 else 1.0
    
    mask2_u = (u>=u1*0.49)&(u<=u1*0.51)&(t<=ts) 
    mask2_s = (s>=s1*0.49)&(s<=s1*0.51)&(t<=ts) 
    if(np.any(mask2_u) or np.any(mask2_s)):
        t2 = np.median(t[mask2_u | mask2_s])
        u2, s2 = np.median(u[mask2_u]), np.median(s[mask2_s])
        t0 = max(0,np.log((u1-u2)/(u1*np.exp(-t2)-u2*np.exp(-t1))))
    else:
        t0 = 0
    beta = 1
    alpha = u1/(1-np.exp(t0-t1)) if u1>0 else 0.1*np.random.rand()
    if(alpha <= 0 or np.isnan(alpha) or np.isinf(alpha)):
        alpha = u1
    gamma = alpha/np.quantile(s,0.95)
    if(gamma <= 0 or np.isnan(gamma) or np.isinf(gamma)):
        gamma = 2.0
    return alpha,beta,gamma,t0
    
def reinit_params(U, S, t, ts):
    #Reinitialize the regular ODE parameters based on estimated global latent time.
    
    G = U.shape[1]
    alpha, beta, gamma, ton = np.zeros((G)), np.zeros((G)), np.zeros((G)), np.zeros((G))
    for i in range(G):
        alpha_g, beta_g, gamma_g, ton_g = reinit_gene(U[:,i], S[:,i], t, ts[i])
        alpha[i] = alpha_g
        beta[i] = beta_g
        gamma[i] = gamma_g
        ton[i] = ton_g
    return alpha, beta, gamma, ton
    

############################################################
#Vanilla VAE
############################################################
"""
ODE Solution, with both numpy (for post-training analysis or plotting) and pytorch versions (for training)
"""
def pred_steady_numpy(ts,alpha,beta,gamma):
    ############################################################
    #(Numpy Version)
    #Predict the steady states.
    #ts: [G] switching time, when the kinetics enters the repression phase
    #alpha, beta, gamma: [G] generation, splicing and degradation rates
    ############################################################
    
    alpha_, beta_, gamma_ = np.clip(alpha,a_min=0,a_max=None), np.clip(beta,a_min=0,a_max=None), np.clip(gamma,a_min=0,a_max=None)
    eps = 1e-6
    unstability = np.abs(beta-gamma) < eps
    
    ts_ = ts.squeeze()
    expb, expg = np.exp(-beta*ts_), np.exp(-gamma*ts_)
    u0 = alpha/(beta+eps)*(1.0-expb)
    s0 = alpha/(gamma+eps)*(1.0-expg)+alpha/(gamma-beta+eps)*(expg-expb)*(1-unstability)-alpha*ts_*expg*unstability
    return u0,s0
    
def pred_steady(tau_s, alpha, beta, gamma):
    ############################################################
    #(PyTorch Version)
    #Predict the steady states.
    #tau_s: [G] time duration from ton to toff
    #alpha, beta, gamma: [G] generation, splicing and degradation rates
    ############################################################
    
    eps = 1e-6
    unstability = (torch.abs(beta - gamma) < eps).long()
    
    expb, expg = torch.exp(-beta*tau_s), torch.exp(-gamma*tau_s)
    u0 = alpha/(beta+eps)*(torch.tensor([1.0]).to(alpha.device)-expb)
    s0 = alpha/(gamma+eps)*(torch.tensor([1.0]).to(alpha.device)-expg)+alpha/(gamma-beta+eps)*(expg-expb)*(1-unstability)-alpha*tau_s*expg*unstability
    
    return u0,s0

def ode_numpy(t,alpha,beta,gamma,to,ts,scaling=None, k=10.0):
    """(Numpy Version) ODE solution with fixed rates
    
    Arguments
    ---------
    
    t : `numpy array`
        Cell time, (N,1)
    alpha, beta, gamma : `numpy array`
        Generation, splicing and degradation rates, (G,)
    to, ts : `numpy array` 
        Switch-on and -off time, (G,)
    scaling : `numpy array`, optional
        Scaling factor
    k : float, optional
        Parameter for a smooth clip of tau.
    
    Returns
    -------
    uhat, shat : `numpy array`
        Predicted u and s values
    """
    eps = 1e-6
    unstability = (np.abs(beta - gamma) < eps)
    
    o = (t<=ts).astype(int)
    #Induction
    #tau_on = np.clip(t-to,a_min=0,a_max=None)
    tau_on = F.softplus(torch.tensor(t-to), beta=k).numpy()
    assert np.all(~np.isnan(tau_on))
    expb, expg = np.exp(-beta*tau_on), np.exp(-gamma*tau_on)
    uhat_on = alpha/(beta+eps)*(1.0-expb)
    shat_on = alpha/(gamma+eps)*(1.0-expg)+alpha/(gamma-beta+eps)*(expg-expb)*(1-unstability) - alpha*tau_on*unstability
    
    #Repression
    u0_,s0_ = pred_steady_numpy(np.clip(ts-to,0,None),alpha,beta,gamma) #[G]
    if(ts.ndim==2 and to.ndim==2):
        u0_ = u0_.reshape(-1,1)
        s0_ = s0_.reshape(-1,1)
    #tau_off = np.clip(t-ts,a_min=0,a_max=None)
    tau_off = F.softplus(torch.tensor(t-ts), beta=k).numpy()
    assert np.all(~np.isnan(tau_off))
    expb, expg = np.exp(-beta*tau_off), np.exp(-gamma*tau_off)
    uhat_off = u0_*expb
    shat_off = s0_*expg+(-beta*u0_)/(gamma-beta+eps)*(expg-expb)*(1-unstability)
    
    uhat, shat = (uhat_on*o + uhat_off*(1-o)),(shat_on*o + shat_off*(1-o))
    if(scaling is not None):
        uhat *= scaling
    return uhat, shat 

def ode(t, alpha, beta, gamma, to, ts, neg_slope=0.0):
    """(PyTorch Version) ODE Solution
    Parameters are the same as the numpy version, with arrays replaced with 
    tensors. Additionally, neg_slope is used for time clipping.
    """
    eps = 1e-6
    unstability = (torch.abs(beta - gamma) < eps).long()
    o = (t<=ts).int()
    
    #Induction
    tau_on = F.leaky_relu(t-to, negative_slope=neg_slope)
    expb, expg = torch.exp(-beta*tau_on), torch.exp(-gamma*tau_on)
    uhat_on = alpha/(beta+eps)*(torch.tensor([1.0]).to(alpha.device)-expb)
    shat_on = alpha/(gamma+eps)*(torch.tensor([1.0]).to(alpha.device)-expg)+ (alpha/(gamma-beta+eps)*(expg-expb)*(1-unstability) - alpha*tau_on*expg * unstability)
    
    #Repression
    u0_,s0_ = pred_steady(F.relu(ts-to),alpha,beta,gamma)
    
    tau_off = F.leaky_relu(t-ts, negative_slope=neg_slope)
    expb, expg = torch.exp(-beta*tau_off), torch.exp(-gamma*tau_off)
    uhat_off = u0_*expb
    shat_off = s0_*expg+(-beta*u0_)/(gamma-beta+eps)*(expg-expb) * (1-unstability)
    
    return (uhat_on*o + uhat_off*(1-o)),(shat_on*o + shat_off*(1-o)) 




    

############################################################
#Branching ODE
############################################################
def encode_type(cell_types_raw):
    ############################################################
    #Use integer to encode the cell types
    #Each cell type has one unique integer label.
    ############################################################
    
    #Map cell types to integers 
    label_dic = {}
    label_dic_rev = {}
    for i, type_ in enumerate(cell_types_raw):
        label_dic[type_] = i
        label_dic_rev[i] = type_
        
    return label_dic, label_dic_rev

def str2int(cell_labels_raw, label_dic):
    return np.array([label_dic[cell_labels_raw[i]] for i in range(len(cell_labels_raw))])
    
def int2str(cell_labels, label_dic_rev):
    return np.array([label_dic_rev[cell_labels[i]] for i in range(len(cell_labels))])
    
def linreg_mtx(u,s):
    ############################################################
    #Performs linear regression ||U-kS||_2 while 
    #U and S are matrices and k is a vector.
    #Handles divide by zero by returninig some default value.
    ############################################################
    Q = np.sum(s*s, axis=0)
    R = np.sum(u*s, axis=0)
    k = R/Q
    if np.isinf(k) or np.isnan(k):
        k = 1.5
    #k[np.isinf(k) | np.isnan(k)] = 1.5
    return k

def reinit_type_params(U, S, t, ts, cell_labels, cell_types, init_types):
    ############################################################
    #Applied under branching ODE
    #Use the steady-state model and estimated cell time to initialize
    #branching ODE parameters.
    ############################################################
    Ntype = len(cell_types)
    G = U.shape[1]
    alpha, beta, gamma = np.ones((Ntype,G)), np.ones((Ntype,G)), np.ones((Ntype,G))
    u0, s0 = np.zeros((len(init_types),G)), np.zeros((len(init_types),G))

    for i, type_ in enumerate(cell_types):
        mask_type = cell_labels == type_
        #Determine induction or repression
        
        t_head = np.quantile(t[mask_type],0.02)
        t_mid = (t_head+np.quantile(t[mask_type],0.98))*0.5
    
        u_head = np.mean(U[(t>=t[mask_type].min()) & (t<t_head),:],axis=0)
        u_mid = np.mean(U[(t>=t_mid*0.98) & (t<=t_mid*1.02),:],axis=0)
    
        s_head = np.mean(S[(t>=t[mask_type].min()) & (t<t_head),:],axis=0)
        s_mid = np.mean(S[(t>=t_mid*0.98) & (t<=t_mid*1.02),:],axis=0)
    
        o = u_head + s_head < u_mid + s_mid
        
        #Determine ODE parameters
        U_type, S_type = U[(cell_labels==type_)], S[(cell_labels==type_)]
        
        for g in range(G):
            u_low = np.min(U_type[:,g])
            s_low = np.min(S_type[:,g])
            u_high = np.quantile(U_type[:,g],0.95)
            s_high = np.quantile(S_type[:,g],0.95)
            mask_high =  (U_type[:,g]>u_high) | (S_type[:,g]>s_high)
            mask_low = (U_type[:,g]<u_low) | (S_type[:,g]<s_low)
            mask_q = mask_high | mask_low
            u_q = U_type[mask_q,g]
            s_q = S_type[mask_q,g]
            slope = linreg_mtx(u_q-U_type[:,g].min(), s_q-S_type[:,g].min())
            if(slope == 1):
                slope = 1 + 0.1*np.random.rand()
            gamma[type_, g] = np.clip(slope, 0.01, None)
        
        alpha[type_] = (np.quantile(U_type,0.95,axis=0) - np.quantile(U_type,0.05,axis=0)) * o \
                        + (np.quantile(U_type,0.95,axis=0) - np.quantile(U_type,0.05,axis=0)) * (1-o) * np.random.rand(G) * 0.001+1e-10
            
            
    for i, type_ in enumerate(init_types):
        mask_type = cell_labels == type_
        t_head = np.quantile(t[mask_type],0.03)
        u0[i] = np.mean(U[(t>=t[mask_type].min()) & (t<=t_head)],axis=0)+1e-10
        s0[i] = np.mean(S[(t>=t[mask_type].min()) & (t<=t_head)],axis=0)+1e-10
        
     
    return alpha,beta,gamma,u0,s0


def ode_br(t, y, par, neg_slope=0.0, **kwargs):
    """(PyTorch Version) Branching ODE solution.
    """
    alpha,beta,gamma = kwargs['alpha'], kwargs['beta'], kwargs['gamma'] #[N type x G]
    t_trans = kwargs['t_trans']
    u0,s0 = kwargs['u0'], kwargs['s0'] #[N type x G]
    scaling=kwargs["scaling"]
    
    Ntype, G = alpha.shape
    N = t.shape[0]
    
    tau0 = F.leaky_relu((t_trans - t_trans[par]).view(-1,1), neg_slope)
    u0_hat, s0_hat = pred_su(tau0, u0[par], s0[par], alpha[par], beta[par], gamma[par]) 
    
    #For cells with time violation, we use its parent type 
    mask = (t >= t_trans[y].view(-1,1)).float()
    par_batch = par[y]
    u0_batch = u0_hat[y] * mask + u0_hat[par_batch] * (1-mask)
    s0_batch = s0_hat[y] * mask + s0_hat[par_batch] * (1-mask) #[N x G]
    tau = F.leaky_relu(t - t_trans[y].view(-1,1), neg_slope) * mask + F.leaky_relu(t - t_trans[par_batch].view(-1,1), neg_slope) * (1-mask)
    uhat, shat = pred_su(tau,
                         u0_batch,
                         s0_batch,
                         alpha[y] * mask + alpha[par_batch] * (1-mask),
                         beta[y] * mask + beta[par_batch] * (1-mask),
                         gamma[y] * mask + gamma[par_batch] * (1-mask))
    return uhat * scaling, shat

def ode_br_numpy(t, y, par, neg_slope=0.0, **kwargs):
    """
    (Numpy Version)
    Branching ODE solution.
    
    Arguments
    ---------
    t : `numpy array`
        Cell time, (N,1)
    y : `numpy array`
        Cell type, encoded in integer, (N,)
    par : `numpy array`
        Parent cell type in the transition graph, (N_type,)
    
    Returns
    -------
    uhat, shat : `numpy array`
        Predicted u and s values, (N,G)
    """
    alpha,beta,gamma = kwargs['alpha'], kwargs['beta'], kwargs['gamma'] #[N type x G]
    t_trans = kwargs['t_trans']
    u0,s0 = kwargs['u0'], kwargs['s0'] #[N type x G]
    scaling=kwargs["scaling"]
    
    Ntype, G = alpha.shape
    N = t.shape[0]
    
    tau0 = np.clip((t_trans - t_trans[par]).reshape(-1,1), 0, None)
    u0_hat, s0_hat = pred_su_numpy(tau0, u0[par], s0[par], alpha[par], beta[par], gamma[par]) #[Ntype x G]
    
    
    uhat, shat = np.zeros((N,G)), np.zeros((N,G))
    for i in range(Ntype):
        mask = (t[y==i] >= t_trans[i])
        tau = np.clip(t[y==i].reshape(-1,1) - t_trans[i], 0, None) * mask + np.clip(t[y==i].reshape(-1,1) - t_trans[par[i]], 0, None) * (1-mask)
        uhat_i, shat_i = pred_su_numpy(tau,
                                     u0_hat[i]*mask+u0_hat[par[i]]*(1-mask),
                                     s0_hat[i]*mask+s0_hat[par[i]]*(1-mask),
                                     alpha[i],
                                     beta[i],
                                     gamma[i])
        uhat[y==i] = uhat_i
        shat[y==i] = shat_i
    return uhat*scaling, shat





############################################################
#  Optimal Transport
############################################################


"""
Geoffrey Schiebinger, Jian Shu, Marcin Tabaka, Brian Cleary, Vidya Subramanian, 
  Aryeh Solomon, Joshua Gould, Siyan Liu, Stacie Lin, Peter Berube, Lia Lee, 
  Jenny Chen, Justin Brumbaugh, Philippe Rigollet, Konrad Hochedlinger, Rudolf Jaenisch, Aviv Regev, Eric S. Lander,
  Optimal-Transport Analysis of Single-Cell Gene Expression Identifies Developmental Trajectories in Reprogramming,
  Cell,
  Volume 176, Issue 4,
  2019,
  Pages 928-943.e22,
  ISSN 0092-8674,
  https://doi.org/10.1016/j.cell.2019.01.006.
"""
# @ Lénaïc Chizat 2015 - optimal transport
def fdiv(l, x, p, dx):
    return l * np.sum(dx * (x * (np.log(x / p)) - x + p))


def fdivstar(l, u, p, dx):
    return l * np.sum((p * dx) * (np.exp(u / l) - 1))


def primal(C, K, R, dx, dy, p, q, a, b, epsilon, lambda1, lambda2):
    I = len(p)
    J = len(q)
    F1 = lambda x, y: fdiv(lambda1, x, p, y)
    F2 = lambda x, y: fdiv(lambda2, x, q, y)
    with np.errstate(divide='ignore'):
        return F1(np.dot(R, dy), dx) + F2(np.dot(R.T, dx), dy) \
               + (epsilon * np.sum(R * np.nan_to_num(np.log(R)) - R + K) \
                  + np.sum(R * C)) / (I * J)


def dual(C, K, R, dx, dy, p, q, a, b, epsilon, lambda1, lambda2):
    I = len(p)
    J = len(q)
    F1c = lambda u, v: fdivstar(lambda1, u, p, v)
    F2c = lambda u, v: fdivstar(lambda2, u, q, v)
    return - F1c(- epsilon * np.log(a), dx) - F2c(- epsilon * np.log(b), dy) \
           - epsilon * np.sum(R - K) / (I * J)


# end @ Lénaïc Chizat

def optimal_transport_duality_gap(C, G, lambda1, lambda2, epsilon, batch_size, tolerance, tau,
                                  epsilon0, max_iter, **ignored):
    """
    Compute the optimal transport with stabilized numerics, with the guarantee that the duality gap is at most `tolerance`
    Code is from `the work by Schiebinger et al. <https://www.cell.com/cell/fulltext/S0092-8674(19)30039-X?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS009286741930039X%3Fshowall%3Dtrue>`
    
    Arguments
    ----------
    C : 2-D ndarray
        The cost matrix. C[i][j] is the cost to transport cell i to cell j
    G : 1-D array_like
        Growth value for input cells.
    lambda1 : float, optional
        Regularization parameter for the marginal constraint on p
    lambda2 : float, optional
        Regularization parameter for the marginal constraint on q
    epsilon : float, optional
        Entropy regularization parameter.
    batch_size : int, optional
        Number of iterations to perform between each duality gap check
    tolerance : float, optional
        Upper bound on the duality gap that the resulting transport map must guarantee.
    tau : float, optional
        Threshold at which to perform numerical stabilization
    epsilon0 : float, optional
        Starting value for exponentially-decreasing epsilon
    max_iter : int, optional
        Maximum number of iterations. Print a warning and return if it is reached, even without convergence.
    
    Returns
    -------
    transport_map : 2-D ndarray
        The entropy-regularized unbalanced transport map
    """
    C = np.asarray(C, dtype=np.float64)
    epsilon_scalings = 5
    scale_factor = np.exp(- np.log(epsilon) / epsilon_scalings)

    I, J = C.shape
    dx, dy = np.ones(I) / I, np.ones(J) / J

    p = G
    q = np.ones(C.shape[1]) * np.average(G)

    u, v = np.zeros(I), np.zeros(J)
    a, b = np.ones(I), np.ones(J)

    epsilon_i = epsilon0 * scale_factor
    current_iter = 0

    for e in range(epsilon_scalings + 1):
        duality_gap = np.inf
        u = u + epsilon_i * np.log(a)
        v = v + epsilon_i * np.log(b)  # absorb
        epsilon_i = epsilon_i / scale_factor
        _K = np.exp(-C / epsilon_i)
        alpha1 = lambda1 / (lambda1 + epsilon_i)
        alpha2 = lambda2 / (lambda2 + epsilon_i)
        K = np.exp((np.array([u]).T - C + np.array([v])) / epsilon_i)
        a, b = np.ones(I), np.ones(J)
        old_a, old_b = a, b
        threshold = tolerance if e == epsilon_scalings else 1e-6

        while duality_gap > threshold:
            for i in range(batch_size if e == epsilon_scalings else 5):
                current_iter += 1
                old_a, old_b = a, b
                a = (p / (K.dot(np.multiply(b, dy)))) ** alpha1 * np.exp(-u / (lambda1 + epsilon_i))
                b = (q / (K.T.dot(np.multiply(a, dx)))) ** alpha2 * np.exp(-v / (lambda2 + epsilon_i))

                # stabilization
                if (max(max(abs(a)), max(abs(b))) > tau):
                    u = u + epsilon_i * np.log(a)
                    v = v + epsilon_i * np.log(b)  # absorb
                    K = np.exp((np.array([u]).T - C + np.array([v])) / epsilon_i)
                    a, b = np.ones(I), np.ones(J)

                if current_iter >= max_iter:
                    print("Reached max_iter with duality gap still above threshold. Returning")
                    return (K.T * a).T * b

            # The real dual variables. a and b are only the stabilized variables
            _a = a * np.exp(u / epsilon_i)
            _b = b * np.exp(v / epsilon_i)

            # Skip duality gap computation for the first epsilon scalings, use dual variables evolution instead
            if e == epsilon_scalings:
                R = (K.T * a).T * b
                pri = primal(C, _K, R, dx, dy, p, q, _a, _b, epsilon_i, lambda1, lambda2)
                dua = dual(C, _K, R, dx, dy, p, q, _a, _b, epsilon_i, lambda1, lambda2)
                duality_gap = (pri - dua) / abs(pri)
            else:
                duality_gap = max(
                    np.linalg.norm(_a - old_a * np.exp(u / epsilon_i)) / (1 + np.linalg.norm(_a)),
                    np.linalg.norm(_b - old_b * np.exp(v / epsilon_i)) / (1 + np.linalg.norm(_b)))

    if np.isnan(duality_gap):
        raise RuntimeError("Overflow encountered in duality gap computation, please report this incident")
    return R / C.shape[1]

"""
Pytorch Version
"""
def fdiv_ts(l, x, p, dx):
    return l * torch.sum(dx * (x * (torch.log(x / p)) - x + p))


def fdivstar_ts(l, u, p, dx):
    return l * torch.sum((p * dx) * (torch.exp(u / l) - 1))


def primal_ts(C, K, R, dx, dy, p, q, a, b, epsilon, lambda1, lambda2):
    I = len(p)
    J = len(q)
    F1 = lambda x, y: fdiv_ts(lambda1, x, p, y)
    F2 = lambda x, y: fdiv_ts(lambda2, x, q, y)
    with np.errstate(divide='ignore'):
        return F1(torch.sum(R*dy, 1), dx) + F2(torch.sum(R.T*dx, 1), dy) \
               + (epsilon * torch.sum(R * torch.nan_to_num(torch.log(R)) - R + K) \
                  + torch.sum(R * C)) / (I * J)


def dual_ts(C, K, R, dx, dy, p, q, a, b, epsilon, lambda1, lambda2):
    I = len(p)
    J = len(q)
    F1c = lambda u, v: fdivstar_ts(lambda1, u, p, v)
    F2c = lambda u, v: fdivstar_ts(lambda2, u, q, v)
    return - F1c(- epsilon * torch.log(a), dx) - F2c(- epsilon * torch.log(b), dy) \
           - epsilon * torch.sum(R - K) / (I * J)


# end @ Lénaïc Chizat

def optimal_transport_duality_gap_ts(C, G, lambda1, lambda2, epsilon, batch_size, tolerance, tau,
                                  epsilon0, max_iter, **ignored):
    #C = C.double()
    epsilon_scalings = 5
    scale_factor = np.exp(- np.log(epsilon) / epsilon_scalings)

    I, J = C.shape
    dx, dy = torch.ones(I, device=C.device, dtype=C.dtype) / I, torch.ones(J, device=C.device, dtype=C.dtype) / J

    p = G
    q = torch.ones(C.shape[1], device=C.device, dtype=C.dtype) * (G.mean())

    u, v = torch.zeros(I, device=C.device, dtype=C.dtype), torch.zeros(J, device=C.device, dtype=C.dtype)
    a, b = torch.ones(I, device=C.device, dtype=C.dtype), torch.ones(J, device=C.device, dtype=C.dtype)

    epsilon_i = epsilon0 * scale_factor
    current_iter = 0

    for e in range(epsilon_scalings + 1):
        duality_gap = np.inf
        u = u + epsilon_i * torch.log(a)
        v = v + epsilon_i * torch.log(b)  # absorb
        epsilon_i = epsilon_i / scale_factor
        _K = torch.exp(-C / epsilon_i)
        alpha1 = lambda1 / (lambda1 + epsilon_i)
        alpha2 = lambda2 / (lambda2 + epsilon_i)
        K = torch.exp((u.view(-1,1) - C + v.view(1,-1)) / epsilon_i)
        a, b = torch.ones(I, device=C.device, dtype=C.dtype), torch.ones(J, device=C.device, dtype=C.dtype)
        old_a, old_b = a, b
        threshold = tolerance if e == epsilon_scalings else 1e-6

        while duality_gap > threshold:
            for i in range(batch_size if e == epsilon_scalings else 5):
                current_iter += 1
                old_a, old_b = a, b
                a = (p / ( torch.sum(K * (b*dy), 1))).pow(alpha1) * torch.exp(-u / (lambda1 + epsilon_i))
                b = (q / ( torch.sum(K.T*(a*dx), 1))).pow(alpha2) * torch.exp(-v / (lambda2 + epsilon_i))

                # stabilization
                if (max(torch.abs(a).max(), torch.abs(b).max()) > tau):
                    u = u + epsilon_i * torch.log(a)
                    v = v + epsilon_i * torch.log(b)  # absorb
                    K = torch.exp((u.view(-1,1) - C + v.view(1,-1)) / epsilon_i)
                    a, b = torch.ones(I, device=C.device, dtype=C.dtype), torch.ones(J, device=C.device, dtype=C.dtype)

                if current_iter >= max_iter:
                    print(f"Reached max_iter with duality gap still above threshold ({duality_gap:.5f}). Returning")
                    return (K.T * a).T * b

            # The real dual variables. a and b are only the stabilized variables
            _a = a * torch.exp(u / epsilon_i)
            _b = b * torch.exp(v / epsilon_i)

            # Skip duality gap computation for the first epsilon scalings, use dual variables evolution instead
            if e == epsilon_scalings:
                R = (K.T * a).T * b
                pri = primal_ts(C, _K, R, dx, dy, p, q, _a, _b, epsilon_i, lambda1, lambda2)
                dua = dual_ts(C, _K, R, dx, dy, p, q, _a, _b, epsilon_i, lambda1, lambda2)
                duality_gap = (pri - dua) / abs(pri)
            else:
                duality_gap = max(
                    torch.norm(_a - old_a * torch.exp(u / epsilon_i)) / (1 + torch.norm(_a)),
                    torch.norm(_b - old_b * torch.exp(v / epsilon_i)) / (1 + torch.norm(_b)))

    if torch.isnan(duality_gap):
        raise RuntimeError("Overflow encountered in duality gap computation, please report this incident")
    return R / C.shape[1]



############################################################
#  KNN-Related Functions
############################################################
def knnx0_alt(U, S, t, z, t_query, z_query, dt, k):
    N, Nq = len(t), len(t_query)
    u0 = np.zeros((Nq, U.shape[1]))
    s0 = np.zeros((Nq, S.shape[1]))
    t0 = np.ones((Nq))*(t.min() - dt[0])
    
    order_idx = np.argsort(t)
    _t = t[order_idx]
    _z = z[order_idx]
    _U = U[order_idx]
    _S = S[order_idx]
    
    order_query = np.argsort(t_query)
    _t_query = t_query[order_query]
    _z_query = z_query[order_query]
    
    knn = np.ones((Nq,k))*np.nan
    D = np.ones((Nq,k))*np.nan
    ptr = 0
    left, right = 0, 0 #pointer in the query sequence
    i = 0
    while(left<Nq and i<N): #i as initial point x0
        #Update left, right
        if(_t[i]+dt[0]>=_t_query[-1]):
            break;
        for l in range(left, Nq):
            if(_t_query[l]>=_t[i]+dt[0]):
                left = l
                break
        for l in range(right, Nq):
            if(_t_query[l]>=_t[i]+dt[1]):
                right = l
                break
        
        #Update KNN
        for j in range(left, right): #j is the set of cell with i in the range [tj-dt,tj-dt/2]
            dist = np.linalg.norm(_z[i]-_z_query[j])
            pos_nan = np.where(np.isnan(knn[j]))[0]
            if(len(pos_nan)>0): #there hasn't been k nearest neighbors for j yet
                knn[j,pos_nan[0]] = i
                D[j,pos_nan[0]] = dist
            else:
                idx_largest = np.argmax(D[j])
                if(dist<D[j,idx_largest]):
                    D[j,idx_largest] = dist
                    knn[j,idx_largest] = i
        i += 1
    #Calculate initial time and conditions
    for i in range(Nq):
        if(np.all(np.isnan(knn[i]))):
            continue
        pos = np.where(~np.isnan(knn[i]))[0]
        u0[order_query[i]] = _U[knn[i,pos].astype(int)].mean(0)
        s0[order_query[i]] = _S[knn[i,pos].astype(int)].mean(0)
        t0[order_query[i]] = _t[knn[i,pos].astype(int)].mean()
    
    return u0,s0,t0

def knnx0(U, S, t, z, t_query, z_query, dt, k, adaptive=False, std_t=None):
    ############################################################
    #Given cell time and state, find KNN for each cell in a time window ahead of
    #it. The KNNs are used to compute the initial condition for the ODE of
    #the cell.
    #1-2.    U,S [2D array (N,G)]
    #        Unspliced and Spliced count matrix
    #3-4.    t,z [1D array (N)]
    #        Latent cell time and state used to build KNN
    #5-6.    t_query [1D array (N)]
    #        Query cell time and state
    #7.      dt [float tuple]
    #        Time window coefficient
    #8.      k [int]
    #        Number of neighbors
    #9.      adaptive [bool]
    #        Whether to use adaptive time window based on time uncertainty
    #10.     std_t [1D array (N)]
    #        Posterior standard deviation of cell time
    ############################################################
    N, Nq = len(t), len(t_query)
    u0 = np.zeros((Nq, U.shape[1]))
    s0 = np.zeros((Nq, S.shape[1]))
    t0 = np.ones((Nq))*(t.min() - dt[0])
    
    n1 = 0
    len_avg = 0
    for i in range(Nq):
        if(adaptive):
            dt_r, dt_l = std_t[i], std_t[i] + (dt[1]-dt[0])
        else:
            dt_r, dt_l = dt[0], dt[1]
        t_ub, t_lb = t_query[i] - dt_r, t_query[i] - dt_l
        indices = np.where((t>=t_lb) & (t<t_ub))[0]
        k_ = len(indices)
        len_avg = len_avg+k_
        if(k_>0):
            if(k_<k):
                u0[i] = U[indices].mean(0)
                s0[i] = S[indices].mean(0)
                t0[i] = t[indices].mean()
                n1 = n1+1
            else:
                knn_model = NearestNeighbors(n_neighbors=k)
                knn_model.fit(z[indices])
                dist, ind = knn_model.kneighbors(z_query[i:i+1])
                u0[i] = np.mean( U[indices[ind.squeeze()].astype(int)], 0)
                s0[i] = np.mean( S[indices[ind.squeeze()].astype(int)], 0)
                t0[i] = np.mean( t[indices[ind.squeeze()].astype(int)] )
        else:
            n1 = n1+1
    print(f"Percentage of Invalid Sets: {n1/Nq:.3f}")
    print(f"Average Set Size: {len_avg//Nq}")
    return u0,s0,t0

def knnx0_quantile(U, S, t, z, t_query, z_query, dt, k, q=0.95):
    ############################################################
    #Given cell time and state, find KNN for each cell in a time window ahead of
    #it. The KNNs are used to compute the initial condition for the ODE of
    #the cell.
    #1-2.    U,S [2D array (N,G)]
    #        Unspliced and Spliced count matrix
    #3-4.    t,z [1D array (N)]
    #        Latent cell time and state used to build KNN
    #5-6.    t_query [1D array (N)]
    #        Query cell time and state
    #7.      dt [float tuple]
    #        Time window coefficient
    #8.      k [int]
    #        Number of neighbors
    ############################################################
    N, Nq = len(t), len(t_query)
    u0 = np.zeros((Nq, U.shape[1]))
    s0 = np.zeros((Nq, S.shape[1]))
    t0 = np.ones((Nq))*(t.min() - dt[0])
    
    n1 = 0
    len_avg = 0
    for i in range(Nq):
        t_ub, t_lb = t_query[i] - dt[0], t_query[i] - dt[1]
        indices = np.where((t>=t_lb) & (t<t_ub))[0]
        k_ = len(indices)
        len_avg = len_avg+k_
        if(k_>0):
            if(k_<k):
                u0[i] = U[indices].mean(0)
                s0[i] = S[indices].mean(0)
                t0[i] = t[indices].mean()
                n1 = n1+1
            else:
                knn_model = NearestNeighbors(n_neighbors=k)
                knn_model.fit(z[indices])
                dist, ind = knn_model.kneighbors(z_query[i:i+1])
                u0[i] = np.quantile( U[indices[ind.squeeze()].astype(int)], q, 0)
                s0[i] = np.quantile( S[indices[ind.squeeze()].astype(int)], q, 0)
                t0[i] = np.mean( t[indices[ind.squeeze()].astype(int)] )
        else:
            n1 = n1+1
    print(f"Percentage of Invalid Sets: {n1/Nq:.3f}")
    print(f"Average Set Size: {len_avg//Nq}")
    return u0,s0,t0


def knnx0_bin(U, 
              S, 
              t, 
              z, 
              t_query, 
              z_query, 
              dt, 
              k=None, 
              n_graph=10, 
              pruning_degree_multiplier=1.5, 
              diversify_prob=1.0, 
              max_bin_size=10000):
    ############################################################
    #Same functionality as knnx0, but with a different algorithm. Instead of computing
    #a KNN graph for each cell, we divide the time line into several bins and compute
    #a KNN for each bin. The parent of each cell is chosen from its previous bin. 
    ############################################################
    tmin = min(t.min(), t_query.min())
    N, Nq = len(t), len(t_query)
    u0 = np.zeros((Nq, U.shape[1]))
    s0 = np.zeros((Nq, S.shape[1]))
    t0 = np.ones((Nq))*(t.min() - dt[0])
    
    delta_t = (np.quantile(t,0.99)-tmin+1e-6)/(n_graph+1)
    
    order_t = np.argsort(t)
    order_t_query = np.argsort(t_query)
    
    #First Time Interval: Use the average of initial data points.
    indices = np.where(t<tmin+delta_t)[0]
    if(len(indices) > max_bin_size):
        indices = np.random.choice(indices, max_bin_size, replace=False)
    mask_init = t<np.quantile(t[indices], 0.2)
    u_init = U[mask_init].mean(0)
    s_init = S[mask_init].mean(0)
    indices_query = np.where(t_query<tmin+delta_t)[0]
    u0[indices_query] = u_init
    s0[indices_query] = s_init
    t0[indices_query] = tmin
    
    for i in range(n_graph):
        t_ub, t_lb = tmin+(i+1)*delta_t, tmin+i*delta_t
        indices = np.where((t>=t_lb) & (t<t_ub))[0]
        if(len(indices) > max_bin_size):
            indices = np.random.choice(indices, max_bin_size, replace=False)
        k_ = len(indices)
        if(k_==0):
            continue
        if(k is None):
            k = max(1, len(indices)//20)
        knn_model = pynndescent.NNDescent(z[indices], n_neighbors=k+1, pruning_degree_multiplier=pruning_degree_multiplier, diversify_prob=diversify_prob)
        #The query points are in the next time interval
        indices_query = np.where((t_query>=t_ub) & (t_query<t_ub+delta_t))[0] if i<n_graph-1 else np.where(t_query>=t_ub)[0]
        if(len(indices_query)==0):
            continue
        try:
            ind, dist = knn_model.query(z_query[indices_query], k=k)
            ind = ind.astype(int)
        except ValueError:
            knn_model = NearestNeighbors(n_neighbors=min(k,len(indices)))
            knn_model.fit(z[indices])
            dist, ind = knn_model.kneighbors(z_query[indices_query])
        
        for j in range(len(indices_query)):
            neighbor_idx = indices[ind[j]]
            u0[indices_query[j]] = np.mean( U[ neighbor_idx ], 0)
            s0[indices_query[j]] = np.mean( S[ neighbor_idx ], 0)
            t0[indices_query[j]] = np.mean( t[ neighbor_idx ] )
    return u0,s0,t0



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
    #Compute the frequency of cell type transition based on windowed KNN.
    #Used in transition graph construction.
    ############################################################
    N, Nq = len(t), len(t_query)
    P = np.zeros((n_type, n_type))
    t0 = np.zeros((n_type))
    sigma_t = np.zeros((n_type))
    
    for i in range(n_type):
        t0[i] = np.quantile(t[cell_labels==i], 0.01)
        sigma_t[i] = t[cell_labels==i].std()
    if(soft_assign):
        A = csr_matrix((N, N))
        for i in range(Nq):
            t_ub, t_lb = t_query[i] - dt[0], t_query[i] - dt[1]
            indices = np.where((t>=t_lb) & (t<t_ub))[0]
            k_ = len(indices)
            if(k_>0):
                if(k_<=k):
                    A[i, indices] = 1 #(np.sqrt(np.sum((z_query[i] - z[indices])**2,1)) < dist_thred).astype(int) #np.exp(-((t[i] - t0[cell_labels[i]])/sigma_t[cell_labels[i]])**2)
                else:
                    knn_model = NearestNeighbors(n_neighbors=k)
                    knn_model.fit(z[indices])
                    dist, ind = knn_model.kneighbors(z_query[i:i+1])
                    A[i,indices[ind.squeeze()]] = 1 #(dist < dist_thred).astype(int) #np.exp(-((t[i] - t0[cell_labels[i]])/sigma_t[cell_labels[i]])**2)
        for i in range(n_type):
            for j in range(n_type):
                P[i,j] = A[cell_labels==i][:,cell_labels==j].sum()
    else:
        A = csr_matrix((N, n_type))
        for i in range(Nq):
            t_ub, t_lb = t_query[i] - dt[0], t_query[i] - dt[1]
            indices = np.where((t>=t_lb) & (t<t_ub))[0]
            k_ = len(indices)
            if(k_>0):
                if(k_<=k):
                    knn_model = NearestNeighbors(n_neighbors=min(k,k_))
                    knn_model.fit(z[indices])
                    dist, ind = knn_model.kneighbors(z_query[i:i+1])
                    knn_label = cell_labels[indices][ind.squeeze()]
                else:
                    knn_label = cell_labels[indices]
                n_par = np.array([np.sum(knn_label==i) for i in range(n_type)])
                A[i, np.argmax(n_par)] = 1
        for i in range(n_type):
            P[i] = A[cell_labels==i].sum(0)
    psum = P.sum(1)
    psum[psum==0] = 1
    return P/(psum.reshape(-1,1))


############################################################
#Other Auxilliary Functions
############################################################
def make_dir(file_path):
    if(os.path.exists(file_path)):
        return
    else:
        directories = file_path.split('/')
        cur_path = ''
        for directory in directories:
            if(directory==''):
                continue
            cur_path += directory
            cur_path += '/'
            if(not (directory=='.' or directory == '..') ):
                if not os.path.exists(cur_path):
                    os.mkdir(cur_path)
    

def get_gene_index(genes_all, gene_list):
    gind = []
    gremove = []
    for gene in gene_list:
        matches = np.where(genes_all==gene)[0]
        if(len(matches)==1):
            gind.append(matches[0])
        elif(len(matches)==0):
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

def sample_genes(adata, n, key, mode='top',q=0.5):
    if(mode=='random'):
        return np.random.choice(adata.var_names, n, replace=False)
    val_sorted = adata.var[key].sort_values(ascending=False)
    genes_sorted = val_sorted.index.to_numpy()
    if(mode=='threshold'):
        N = np.sum(val_sorted.to_numpy()>=q)
        return np.random.choice(genes_sorted[:N], min(n,N), replace=False)
    return genes_sorted[:n]

def add_capture_time(adata, tkey, save_key="tprior"):
    capture_time = adata.obs[tkey].to_numpy()
    if(isinstance(capture_time[0], str)):
        j = 0
        while(not (capture_time[0][j]>='0' and capture_time[0][j]>='9') ):
            j = j+1
        tprior = np.array([float(x[1:]) for x in capture_time])
    else:
        tprior = capture_time
    tprior = tprior - tprior.min() + 0.01
    adata.obs["tprior"] = tprior

def add_cell_cluster(adata, cluster_key, save_key="clusters"):
    cell_labels = adata.obs[cluster_key].to_numpy()
    adata.obs["clusters"] = np.array([str(x) for x in cell_labels])
    
def count_peak_expression(adata, cluster_key = "clusters"):
    def encodeType(cell_types_raw):
        #Use integer to encode the cell types
        #Each cell type has one unique integer label.
        
        #Map cell types to integers 
        label_dic = {}
        label_dic_rev = {}
        for i, type_ in enumerate(cell_types_raw):
            label_dic[type_] = i
            label_dic_rev[i] = type_
            
        return label_dic, label_dic_rev
    cell_labels = adata.obs[cluster_key]
    cell_types = np.unique(cell_labels)
    label_dic, label_dic_rev = encodeType(cell_types)
    cell_labels = np.array([label_dic[x] for x in cell_labels])
    n_type = len(cell_types)
    peak_hist = np.zeros((n_type))
    peak_val_hist = [[] for i in range(n_type)]
    peak_gene = [[] for i in range(n_type)]
    for i in range(adata.n_vars):
        peak_expression = [np.quantile(adata.layers["Ms"][cell_labels==j,i],0.9) for j in range(n_type)]
        peak_hist[np.argmax(peak_expression)] = peak_hist[np.argmax(peak_expression)]+1
        peak_gene[np.argmax(peak_expression)].append(i)
        for j in range(n_type):
            peak_val_hist[j].append(peak_expression[j])
    out = {}
    out_val = {}
    out_peak_gene = {}
    for i in range(n_type):
        out[label_dic_rev[i]] = peak_hist[i]
        out_val[label_dic_rev[i]] = peak_val_hist[i]
        out_peak_gene[label_dic_rev[i]] = peak_gene[i]
    return out,out_val,out_peak_gene

def check_init_cond(adata, tkey, init_type=None, q=0.01):
    if(init_type is None):
        t = adata.obs[tkey].to_numpy()
        t_start = np.quantile(t, q)
        u0 = adata.layers["Mu"][t<=t_start].mean(0)
        s0 = adata.layers["Ms"][t<=t_start].mean(0)
        u_top = np.quantile(adata.layers["Mu"], 0.99, 0)+1e-10
        s_top = np.quantile(adata.layers["Ms"], 0.99, 0)+1e-10
    else:
        cell_labels = adata.obs["clusters"].to_numpy()
        mask = cell_labels==init_type
        u0 = np.quantile(adata.layers["Mu"][mask],0.05,0)
        s0 = np.quantile(adata.layers["Ms"][mask],0.05,0)
        u_top = np.quantile(adata.layers["Mu"][~mask], 0.99, 0)+1e-10
        s_top = np.quantile(adata.layers["Ms"][~mask], 0.99, 0)+1e-10
    
    
    return
    

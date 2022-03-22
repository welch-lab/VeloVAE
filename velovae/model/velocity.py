import numpy as np
import scipy as sp
import scipy.sparse as spr
from scipy.spatial.distance import cosine as cosdist
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import gaussian_filter1d
import pynndescent
from sklearn.preprocessing import normalize
from .model_util import odeNumpy, odeBrNumpy, initAllPairsNumpy, predSUNumpy

def rnaVelocityVanillaVAE(adata, key, use_raw=False, use_scv_genes=False, k=10):
    """
    Compute the velocity based on:
    ds/dt = beta * u - gamma * s
    """
    alpha = adata.var[f"{key}_alpha"].to_numpy()
    beta = adata.var[f"{key}_beta"].to_numpy()
    gamma = adata.var[f"{key}_gamma"].to_numpy()
    t = adata.obs[f"{key}_time"].to_numpy()
    ton = adata.var[f"{key}_ton"].to_numpy()
    toff = adata.var[f"{key}_toff"].to_numpy()
    scaling = adata.var[f"{key}_scaling"].to_numpy()
    if(use_raw):
        U, S = adata.layers['Mu'], adata.layers['Ms']
    else:
        if(f"{key}_uhat" in adata.layers and f"{key}_shat" in adata.layers):
            U, S = adata.layers[f"{key}_uhat"], adata.layers[f"{key}_shat"]
            U = U/scaling
        else:
            U, S = odeNumpy(t.reshape(-1,1),alpha,beta,gamma,ton,toff, None) #don't need scaling here
            adata.layers["Uhat"] = U*scaling
            adata.layers["Shat"] = S
    
    soft_coeff = 1/(1+np.exp(-(t.reshape(-1,1) - ton)*k))
    
    V = (beta * U - gamma * S)*soft_coeff
    adata.layers[f"{key}_velocity"] = V
    if(use_scv_genes):
        gene_mask = np.isnan(adata.var['fit_scaling'].to_numpy())
        V[:, gene_mask] = np.nan
    return V, U, S

def rnaVelocityBrODE(adata, key, use_raw=False, use_scv_genes=False):
    """
    Compute the velocity based on:
    ds/dt = beta * u - gamma * s
    """
    alpha = adata.varm[f"{key}_alpha"].T
    beta = adata.varm[f"{key}_beta"].T
    gamma = adata.varm[f"{key}_gamma"].T
    ts = adata.varm[f"{key}_ts"].T
    t_trans = adata.uns[f"{key}_t_trans"]
    u0 = adata.varm[f"{key}_u0"].T
    s0 = adata.varm[f"{key}_s0"].T
    sigma_u = adata.var[f"{key}_sigma_u"].to_numpy()
    sigma_s = adata.var[f"{key}_sigma_s"].to_numpy()
    scaling = adata.var[f"{key}_scaling"].to_numpy()
    w = adata.uns[f"{key}_w"]
    parents = np.argmax(w,1)
    
    t = adata.obs[f"{key}_time"].to_numpy()
    y = adata.obs[f"{key}_label"].to_numpy()
    """
    y_onehot = np.zeros((adata.n_obs,alpha.shape[0]))
    w_onehot = np.zeros((adata.n_obs,alpha.shape[0]))
    for i in range(alpha.shape[0]):
        y_onehot[y==i, i] = 1
        w_onehot[y==i, np.argmax(w[i])] = 1
    """
    if(use_raw):
        U, S = adata.layers['Mu'], adata.layers['Ms']
    else:
        U, S = odeBrNumpy(t.reshape(-1,1),
                          y,
                          w,
                          alpha=alpha,
                          beta=beta,
                          gamma=gamma,
                          t_trans=t_trans,
                          ts=ts,
                          u0=u0,
                          s0=s0)
        adata.layers["Uhat"] = U
        adata.layers["Shat"] = S
    
    V = np.zeros(S.shape)
    for i in range(alpha.shape[0]):
        V[y==i] = (beta[i]*U[y==i] - gamma[i]*S[y==i])
    adata.layers[f"{key}_velocity"] = V
    if(use_scv_genes):
        gene_mask = np.isnan(adata.var['fit_scaling'].to_numpy())
        V[:, gene_mask] = np.nan
    return V, U, S
    
def rnaVelocityVAE(adata, key, use_raw=False, use_scv_genes=False, sigma=None, approx=False, full_vb=False):
    """
    Compute the velocity based on:
    ds/dt = beta * u - gamma * s
    """
    alpha = np.exp(adata.var[f"{key}_logmu_alpha"].to_numpy()) if full_vb else adata.var[f"{key}_alpha"].to_numpy()
    rho = adata.layers[f"{key}_rho"]
    beta = np.exp(adata.var[f"{key}_logmu_beta"].to_numpy()) if full_vb else adata.var[f"{key}_beta"].to_numpy()
    gamma = np.exp(adata.var[f"{key}_logmu_gamma"].to_numpy()) if full_vb else adata.var[f"{key}_gamma"].to_numpy()
    t = adata.obs[f"{key}_time"].to_numpy()
    t0 = adata.obs[f"{key}_t0"].to_numpy()
    U0 = adata.layers[f"{key}_u0"]
    S0 = adata.layers[f"{key}_s0"]
    scaling = adata.var[f"{key}_scaling"].to_numpy()
    if(use_raw):
        U, S = adata.layers['Mu'], adata.layers['Ms']
    else:
        if(f"{key}_uhat" in adata.layers and f"{key}_shat" in adata.layers):
            U, S = adata.layers[f"{key}_uhat"], adata.layers[f"{key}_shat"]
            U = U/scaling
        else:
            U, S = predSUNumpy(np.clip(t-t0,0,None).reshape(-1,1),U0/scaling,S0,alpha*rho,beta,gamma)
            U, S = np.clip(U, 0, None), np.clip(S, 0, None)
            adata.layers["Uhat"] = U * scaling
            adata.layers["Shat"] = S
    if(approx):
        V = (S - S0)/((t - t0).reshape(-1,1))
    else:
        V = (beta * U - gamma * S)
    if(sigma is not None):
        time_order = np.argsort(t)
        V[time_order] = gaussian_filter1d(V[time_order], sigma, axis=0, mode="nearest")
    adata.layers[f"{key}_velocity"] = V
    if(use_scv_genes):
        gene_mask = np.isnan(adata.var['fit_scaling'].to_numpy())
        V[:, gene_mask] = np.nan
    return V, U, S

def smoothVel(v, t, W=5):
    order_t = np.argsort(t)
    h = np.ones((W))*(1/W)
    v_ret = np.zeros((len(v)))
    v_ret[order_t] = np.convolve(v[order_t], h, mode='same')
    return v_ret
    

"""
Reference:

Bergen, V., Lange, M., Peidli, S., Wolf, F. A., & Theis, F. J. (2020). 
Generalizing RNA velocity to transient cell states through dynamical modeling. 
Nature biotechnology, 38(12), 1408-1414.
"""
def compute_vscore(v, delta_s, sigma_v=None):
    """
    v: (N, G)
    v_neighbors: (N, N neighbors, G)
    """
    #normalize velocity
    N,k,G = delta_s.shape
    v_norm = np.linalg.norm(v,axis=1)
    v_norm[v_norm==0] = 1.0
    v_ = v/v_norm.reshape(N,1)
    
    delta_s_norm = np.linalg.norm(delta_s,axis=2)
    delta_s_norm[delta_s_norm==0] = 1.0
    delta_s_ = delta_s/delta_s_norm.reshape(N,k,1)
    
    cos_dist = 1 - np.einsum('ik,ijk->ij', v_, delta_s_)
    if(sigma_v is None):
        sigma_v = np.quantile(cos_dist, 0.25, 1).reshape(-1,1)
    return np.exp(-(cos_dist/sigma_v))

def velocity_embedding(adata, 
                       vkey, 
                       tkey,
                       n_neighbors,
                       embedding="umap",
                       eps_t = None):
    """
    Project the RNA velocity onto low-dimensional embedding.
    
    Arguments:
    1. adata: AnnData object
    2. vkey: key for extracting RNA velocity
    3. tkey: key for extracting time
    4. n_neighbors: number of neighbors in KNN graph build on the embedding
    5. embedding: type of low-dimensional embedding
    6. eps_t: time margin when considering cell transition
    """
    print("---     Computing velocity embedding...     ---")
    v = adata.layers[vkey]
    s = adata.layers["Ms"]
    gene_mask = ~np.isnan(v[0])
    t = adata.obs[tkey].to_numpy()
    if(~np.all(gene_mask)):
        v = v[:,gene_mask]
        s = s[:,gene_mask]
    x_umap = adata.obsm[f"X_{embedding}"]
    knn_model = pynndescent.NNDescent(x_umap, n_neighbors=n_neighbors)
    neighbors, dist = knn_model.neighbor_graph
    neighbors = neighbors.astype(int)
    
    if(eps_t is None):
        eps_t = np.quantile(np.clip(t[neighbors]-t.reshape(-1,1),0,None), 0.05, 1).reshape(-1,1)
    tscore = t.reshape(-1,1)<=t[neighbors]-eps_t
    delta_s = np.stack([s[neighbors[i]]-s[i] for i in range(v.shape[0])])
    vscore = compute_vscore(v, delta_s)
    score = tscore*vscore
    
    vx = (x_umap[:,0][neighbors] - x_umap[:,0].reshape(-1,1))
    vy = (x_umap[:,1][neighbors] - x_umap[:,1].reshape(-1,1))
    #vx = vx/(np.linalg.norm(vx).reshape(-1,1)+1e-8)
    #vy = vy/(np.linalg.norm(vx).reshape(-1,1)+1e-8)
    vx = (vx*score).sum(1)/score.sum(1)
    vy = (vy*score).sum(1)/score.sum(1)
    print("---                Finished.                ---")
    
    adata.obsm[f"{vkey}_{embedding}"] = np.stack([vx,vy]).T
    return
    

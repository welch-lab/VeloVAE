import numpy as np
import scipy as sp
import scipy.sparse as spr
from scipy.spatial.distance import cosine as cosdist
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import normalize
from .model_util import odeNumpy, odeBrNumpy, initAllPairsNumpy, predSUNumpy

def rnaVelocityVAE(adata, key, use_raw=False, use_scv_genes=False, k=10):
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

def rnaVelocityBrVAE(adata, key, use_raw=False, use_scv_genes=False):
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
    
def rnaVelocityVAEpp(adata, key, use_raw=False, use_scv_genes=False):
    """
    Compute the velocity based on:
    ds/dt = beta * u - gamma * s
    """
    alpha = adata.var[f"{key}_alpha"].to_numpy()
    rho = adata.layers[f"{key}_rho"]
    beta = adata.var[f"{key}_beta"].to_numpy()
    gamma = adata.var[f"{key}_gamma"].to_numpy()
    t = adata.obs[f"{key}_time"].to_numpy()
    t0 = adata.obs[f"{key}_t0"].to_numpy()
    u0 = adata.layers[f"{key}_u0"]
    s0 = adata.layers[f"{key}_s0"]
    scaling = adata.var[f"{key}_scaling"].to_numpy()
    if(use_raw):
        U, S = adata.layers['Mu'], adata.layers['Ms']
    else:
        if(f"{key}_uhat" in adata.layers and f"{key}_shat" in adata.layers):
            U, S = adata.layers[f"{key}_uhat"], adata.layers[f"{key}_shat"]
            U = U/scaling
        else:
            U, S = predSUNumpy(np.clip(t-t0,0,None).reshape(-1,1),u0/scaling,s0,alpha*rho,beta,gamma)
            U, S = np.clip(U, 0, None), np.clip(S, 0, None)
            adata.layers["Uhat"] = U * scaling
            adata.layers["Shat"] = S
    
    V = (beta * U - gamma * S)
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
def transitionMtx(adata, key, scale=10, **kwargs):
    #Step 1: Build a cosine similarity matrix
    S = adata.layers["Ms"]
    N = S.shape[0]
    #KNN performed on the PCA Space 
    try:
        knn_ind = adata.uns["neighbors"]["indices"]
        connectivities = adata.uns["neighbors"]["connectivities"]
        k = knn_ind.shape[1]
    except KeyError:
        print('Neighborhood graph not found! Please run the preprocessing function!')
    
    try:
        if key=="fit":
            V = adata.layers["velocity"]
        else:
            V = adata.layers[f"{key}_velocity"]  
    except KeyError:
        print('Please compute the RNA velocity first!')
    mask = ~np.isnan(V[0])
    V = V[:,mask]

    #Velocity graph based on cosine similarity
    A = spr.csr_matrix((N, N), dtype=float)
    for i in range(N):
        #Difference in expression
        ds =  S[knn_ind[i]][:,mask] - S[i,mask] #n neighbor x ngene
        ds -= np.mean(ds,-1)[:, None]
        #Cosine Similarity
        norm_v = np.linalg.norm(V[i])
        norm_v += norm_v==0
        norm_ds = np.linalg.norm(ds, axis=1)
        norm_ds += norm_ds==0
        A[i, knn_ind[i]] = np.einsum('ij,j',ds,V[i])/(norm_ds*norm_v)[None,:]
    
    #Transition Matrix
    A_pos, A_neg = A, A.copy()
    A_pos.data = np.clip(A_pos.data, 0, 1)
    A_neg.data = np.clip(A_neg.data, -1, 0)
    A_pos.eliminate_zeros()
    A_neg.eliminate_zeros()
    A_pos = A_pos.tocsr()
    A_neg = A_neg.tocsr()
    Pi = np.expm1(A_pos * scale)
    Pi -= np.expm1(-A_neg * scale)
    #Pi.data += 1
    Pi = normalize(Pi, norm='l1', axis=1)
    Pi.eliminate_zeros()
    
    #Smoothing from scVelo
    basis = kwargs.pop('basis', 'umap')
    scale_diffusion = 1.0
    weight_diffusion = 0.0
    if f"X_{basis}" in adata.obsm.keys():
        dists_emb = (Pi > 0).multiply(squareform(pdist(adata.obsm[f"X_{basis}"])))
        scale_diffusion *= dists_emb.data.mean()

        diffusion_kernel = dists_emb.copy()
        diffusion_kernel.data = np.exp(
            -0.5 * dists_emb.data ** 2 / scale_diffusion ** 2
        )
        Pi = Pi.multiply(diffusion_kernel)  # combine velocity kernel & diffusion kernel

        if 0 < weight_diffusion < 1:  # add diffusion kernel (Brownian motion - like)
            diffusion_kernel.data = np.exp(
                -0.5 * dists_emb.data ** 2 / (scale_diffusion / 2) ** 2
            )
            Pi = (1 - weight_diffusion) * Pi + weight_diffusion * diffusion_kernel

        Pi = normalize(Pi, norm='l1', axis=1)
    
    return Pi, k



def rnaVelocityEmbed(adata, key, **kwargs):
    """
    Compute the velocity in the low-dimensional embedding
    """
    X_umap = adata.obsm['X_umap']
    
    P, k = transitionMtx(adata, key, **kwargs)
    assert not np.any(np.isnan(P.data))
    Vembed = np.zeros(X_umap.shape)
    for i in range(Vembed.shape[0]):
        neighbor_idx = P[i].indices
        if(len(neighbor_idx)==0):
            continue
        Delta_x = X_umap[neighbor_idx] - X_umap[i]
        norm_dx = np.linalg.norm(Delta_x,1)
        norm_dx += norm_dx==0
        Delta_x = Delta_x/norm_dx.reshape(-1,1)
        Vembed[i] = P[i, neighbor_idx] @ (Delta_x) - Delta_x.mean(0)
    
    return Vembed[:,0], Vembed[:,1], P
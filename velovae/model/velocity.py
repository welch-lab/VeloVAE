import numpy as np
import scipy as sp
import scipy.sparse as spr


def rnaVelocityVanilla(adata, key, use_scv_genes=False):
    """
    Compute the velocity based on:
    ds/dt = beta * u - gamma * s
    """
    U, S = adata.layers['Mu'], adata.layers['Ms']
    alpha = adata.var[f"{key}_alpha"].to_numpy()
    beta = adata.var[f"{key}_beta"].to_numpy()
    gamma = adata.var[f"{key}_gamma"].to_numpy()
    t = adata.obs[f"{key}_t"].to_numpy()
    to = adata.var[f"{key}_to"].to_numpy()
    
    V = (beta * U - gamma * S)*(t.reshape(-1,1) >= to)
    adata.layers[f"{key}_velocity"] = V
    if(use_scv_genes):
        gene_mask = np.isnan(adata.var['fit_alpha'].to_numpy())
        V[:,gene_mask] = np.nan
    return V
    
def smoothVel(v, t, W=5):
    order_t = np.argsort(t)
    h = np.ones((W))*(1/W)
    v_ret = np.zeros((len(v)))
    v_ret[order_t] = np.convolve(v[order_t], h, mode='same')
    return v_ret

def rnaVelocityBranch(adata, key, graph, init_types, use_scv_genes=False):
    """
    Compute the velocity based on
    ds/dt = beta_y * u - gamma_y * s, where y is the cell type
    """
    Ntype = len(graph.keys())
    alpha = adata.varm[f"{key}_alpha"].T
    beta = adata.varm[f"{key}_beta"].T
    gamma = adata.varm[f"{key}_gamma"].T
    t_trans = adata.uns[f"{key}_t_trans"]
    ts = adata.varm[f"{key}_t_"].T
    t = adata.obs[f"{key}_t"].to_numpy()
    scaling = adata.var[f"{key}_scaling"].to_numpy()
    u0 = adata.varm[f"{key}_u0"].T
    s0 = adata.varm[f"{key}_s0"].T
    cell_labels = adata.obs[f"{key}_label"].to_numpy()
    Uhat, Shat =  odeFullNumpy(t.reshape(len(t),1),
                               graph,
                               init_types,
                               alpha=alpha,
                               beta=beta,
                               gamma=gamma,
                               t_trans=t_trans,
                               ts=ts,
                               scaling=scaling,
                               u0=u0,
                               s0=s0,
                               cell_labels=cell_labels,
                               train_mode=False)
    t_trans_orig, ts_orig = recoverTransitionTime(t_trans, ts, graph, init_types)
    tmask = (t.reshape(-1,1)>=ts_orig[cell_labels]) 
    V = (beta[cell_labels]*Uhat - gamma[cell_labels]*Shat)*tmask
    
    if(use_scv_genes):
        gene_mask = np.isnan(adata.var['fit_alpha'].to_numpy())
        V[:,gene_mask] = np.nan
    adata.layers[f"{key}_velocity"] = V
    return V
    

"""
Reference:

Bergen, V., Lange, M., Peidli, S., Wolf, F. A., & Theis, F. J. (2020). 
Generalizing RNA velocity to transient cell states through dynamical modeling. 
Nature biotechnology, 38(12), 1408-1414.
"""
def transitionMtx(adata, key, **kwargs):
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
    
    if key=="fit":
        V = adata.layers["velocity"]
    else:
        vkey = f"{key}_velocity"
        if vkey in adata.layers:
            V = adata.layers[key]  
        elif("graph" in kwargs and "init_types" in kwargs):
            V = rnaVelocityBranch(adata,key,kwargs["graph"],kwargs["init_types"],use_scv_genes=False)
        else:
            V = rnaVelocity(adata,key)
    mask = ~np.isnan(V[0])
    V = V[:,mask]
    Pi = spr.csr_matrix((N, N), dtype=float)
    sigma = np.zeros((N))
    for i in range(N):
        #Difference in expression
        ds = S[i,mask]-S[knn_ind[i]][:,mask] #n neighbor x ngene
        #ds -= np.mean(ds,1)[:, None]
        #Cosine Similarity
        cosine_sim = np.sum(ds*V[i],1)/(np.linalg.norm(ds,axis=1)*np.linalg.norm(V[i]))
        cosine_sim[0] = 0
        sigma[i] = np.max(np.abs(cosine_sim))
        Pi[i, knn_ind[i]] = cosine_sim/sigma[i]
    
    #Step 2: Apply the Gaussian Kernel
    #sigma = np.clip(np.var(knn_cosine, axis=1).reshape(-1,1), a_min=0.01, a_max=None)
    Pi_til = Pi.expm1()
    Z = np.array(Pi_til.sum(1)).squeeze()+k
    I, J = Pi_til.nonzero()
    val = []
    for i,j in list(zip(I,J)):
        val.append((Pi_til[i,j]+1)/Z[i])
    
    T = spr.csr_matrix((np.array(val),(I,J)))
    
    return T, k



def rnaVelocityEmbed(adata, key, **kwargs):
    """
    Compute the velocity in the low-dimensional embedding
    """
    def dist_v(x, y):
        D = len(x)
        dx = y[:D//2]-x[:D//2]
        v = x[D//2:]
        return 1-dx.dot(v)/np.linalg.norm(dx)/np.linalg.norm(v)
    I, J = np.mgrid[0:adata.n_obs,0:adata.n_obs]
    X_umap = adata.obsm['X_umap']
    
    Pi_til, k = transitionMtx(adata, key, **kwargs)
    
    Delta_x = X_umap[:,0].reshape(-1,1) - X_umap[:,0]
    Delta_y = X_umap[:,1].reshape(-1,1) - X_umap[:,1]
    norm = np.sqrt(Delta_x**2+Delta_y**2)
    Delta_x /= norm
    Delta_y /= norm
    
    Vx = np.array((Pi_til.multiply(Delta_x)).sum(1)).squeeze()
    Vy = np.array((Pi_til.multiply(Delta_y)).sum(1)).squeeze()
    
    return Vx, Vy
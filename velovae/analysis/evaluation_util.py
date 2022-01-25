import numpy as np
from scipy.stats import spearmanr
from ..model.model_util import initParams, predSUNumpy, odeNumpy, odeBrNumpy, scvPred, scvPredSingle

def getMSE(U,S,Uhat,Shat):
    return np.mean((U-Uhat)**2+(S-Shat)**2)

def getMAE(U,S,Uhat,Shat):
    return np.mean(np.abs(U-Uhat)+np.abs(S-Shat))

def getLL(U,S,Uhat,Shat,sigma_u,sigma_s):
    return np.mean(np.sum(-(U-Uhat)**2/(2*sigma_u**2)-(S-Shat)**2/(2*sigma_s**2) - np.log(sigma_u) - np.log(sigma_s) - np.log(2*np.pi),1))
    
def timeCorr(t1, t2):
    return spearmanr(t1,t2)

def cellState(adata, method, key, gene_indices=None):
    if(method=='scvelo'):
        t = adata.layers[f"{key}_time"]
        toff = adata.var[f"{key}_toff"].to_numpy()
        cell_state = (t.reshape(-1,1) > toff)
    else:
        t = adata.obs[f"{key}_time"].to_numpy()
        toff = adata.var[f"{key}_t_"].to_numpy()
        ton = adata.var[f"{key}_ton"].to_numpy()
        cell_state = (t.reshape(-1,1) > toff) + (t.reshape(-1,1) < ton)*2
    if(gene_indices is not None):
        return cell_state[:, gene_indices]
    return cell_state

def getPredictionSCV(adata, key='fit'):
    Uhat, Shat = scvPred(adata, key)
    logp = np.sum(np.log(adata.var[f"{key}_likelihood"]))
    return Uhat, Shat, logp

def getPredictionSCVDemo(adata, key='fit', genes=None, N=100):
    if(genes is None):
        genes = adata.var_names
    alpha, beta, gamma = adata.var[f"{key}_alpha"].to_numpy(),adata.var[f"{key}_beta"].to_numpy(),adata.var[f"{key}_gamma"].to_numpy()
    toff = adata.var[f"{key}_t_"].to_numpy()
    T = adata.layers[f"{key}_t"]
    scaling = adata.var[f"{key}_scaling"].to_numpy()
    Uhat, Shat = np.zeros((2*N,len(genes))), np.zeros((2*N,len(genes)))
    T_demo = np.zeros((2*N, len(genes)))
    for i, gene in enumerate(genes):
        idx = np.where(adata.var_names==gene)[0][0]
        t_demo = np.concatenate((np.linspace(0,toff[idx],N), np.linspace(toff[idx], max(T[:,idx].max(), toff[i]+T[:,idx].max()*0.01),N)))
        T_demo[:,i] = t_demo
        uhat, shat = scvPredSingle(t_demo,alpha[idx],beta[idx],gamma[idx],toff[idx],scaling=scaling[idx], uinit=0, sinit=0)
        Uhat[:,i] = uhat
        Shat[:,i] = shat
    return T_demo, Uhat, Shat

def getPredictionVanilla(adata, key, scv_key=None):
    U, S = adata.layers["Mu"], adata.layers["Ms"]
    #Vanilla VAE
    alpha = adata.var[f"{key}_alpha"].to_numpy()
    beta = adata.var[f"{key}_beta"].to_numpy()
    gamma = adata.var[f"{key}_gamma"].to_numpy()
    toff = adata.var[f"{key}_toff"].to_numpy()
    ton = adata.var[f"{key}_ton"].to_numpy()
    t = adata.obs[f"{key}_time"].to_numpy()
    scaling = adata.var[f"{key}_scaling"].to_numpy()
    sigma_u, sigma_s = adata.var[f"{key}_sigma_u"].to_numpy(), adata.var[f"{key}_sigma_s"].to_numpy()
    
    if( (f"{key}_uhat" not in adata.layers) or (f"{key}_shat" not in adata.layers)):
        Uhat, Shat = odeNumpy(t.reshape(-1,1), alpha, beta, gamma, ton, toff, scaling)
    else:
        Uhat, Shat = adata.layers[f"{key}_uhat"], adata.layers[f"{key}_shat"]
    
    train_idx, test_idx = adata.uns[f"{key}_train_idx"], adata.uns[f"{key}_test_idx"]
    logp_train = -(U[train_idx]-Uhat[train_idx])**2/(2*sigma_u**2)-(S[train_idx]-Shat[train_idx])**2/(2*sigma_s**2) - np.log(sigma_u) - np.log(sigma_s) - np.log(2*np.pi)
    logp_test = -(U[test_idx]-Uhat[test_idx])**2/(2*sigma_u**2)-(S[test_idx]-Shat[test_idx])**2/(2*sigma_s**2) - np.log(sigma_u) - np.log(sigma_s) - np.log(2*np.pi)
    if(scv_key is None):
        logp_train = np.nanmean(np.sum(logp_train,1))
        logp_test = np.nanmean(np.sum(logp_test,1))
    else:
        scv_mask = ~np.isnan(adata.var[f"{scv_key}_alpha"].to_numpy())
        logp_train = np.nanmean(np.sum(logp_train[:,scv_mask],1))
        logp_test = np.nanmean(np.sum(logp_test[:,scv_mask],1))
    
    return Uhat, Shat, logp_train, logp_test

def getPredictionVanillaDemo(adata, key, genes=None, N=100):
    alpha = adata.var[f"{key}_alpha"].to_numpy()
    beta = adata.var[f"{key}_beta"].to_numpy()
    gamma = adata.var[f"{key}_gamma"].to_numpy()
    toff = adata.var[f"{key}_toff"].to_numpy()
    ton = adata.var[f"{key}_ton"].to_numpy()
    scaling = adata.var[f"{key}_scaling"].to_numpy()
    t = adata.obs[f"{key}_time"].to_numpy()
    
    t_demo = np.linspace(0, t.max(), N)
    if(genes is None):
        Uhat_demo, Shat_demo = odeNumpy(t_demo.reshape(-1,1), alpha, beta, gamma, ton, toff, scaling)
    else:
        gene_indices = np.array([np.where(adata.var_names==x)[0][0] for x in genes])
        Uhat_demo, Shat_demo = odeNumpy(t_demo.reshape(-1,1), alpha[gene_indices], beta[gene_indices], gamma[gene_indices], ton[gene_indices], toff[gene_indices], scaling[gene_indices])
    
    return t_demo, Uhat_demo, Shat_demo
    



def getPredictionBranching(adata, key, scv_key=None):
    U, S = adata.layers["Mu"], adata.layers["Ms"]
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
    
    if( (f"{key}_uhat" not in adata.layers) or (f"{key}_shat" not in adata.layers)):
        Uhat, Shat = odeBrNumpy(t.reshape(-1,1),
                                y,
                                w,
                                alpha=alpha,
                                beta=beta,
                                gamma=gamma,
                                t_trans=t_trans,
                                ts=ts,
                                u0=u0,
                                s0=s0,
                                scaling=scaling)
    else:
        Uhat, Shat = adata.layers[f"{key}_uhat"], adata.layers[f"{key}_shat"]
        
    train_idx, test_idx = adata.uns[f"{key}_train_idx"], adata.uns[f"{key}_test_idx"]
    logp_train = -(U[train_idx]-Uhat[train_idx])**2/(2*sigma_u**2)-(S[train_idx]-Shat[train_idx])**2/(2*sigma_s**2) - np.log(sigma_u) - np.log(sigma_s) - np.log(2*np.pi)
    logp_test = -(U[test_idx]-Uhat[test_idx])**2/(2*sigma_u**2)-(S[test_idx]-Shat[test_idx])**2/(2*sigma_s**2) - np.log(sigma_u) - np.log(sigma_s) - np.log(2*np.pi)
    
    if(scv_key is None):
        logp_train = np.nanmean(np.sum(logp_train,1))
        logp_test = np.nanmean(np.sum(logp_test,1))
    else:
        scv_mask = ~np.isnan(adata.var[f"{scv_key}_alpha"].to_numpy())
        logp_train = np.nanmean(np.sum(logp_train[:,scv_mask],1))
        logp_test = np.nanmean(np.sum(logp_test[:,scv_mask],1))
    return Uhat, Shat, logp_train, logp_test

def getPredictionBranchingDemo(adata, key, genes=None, N=100):
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
    parents = np.argmax(w, 1)
    
    t = adata.obs[f"{key}_time"].to_numpy()
    y = adata.obs[f"{key}_label"].to_numpy()
    
    Ntype = alpha.shape[0]
    t_demo = np.zeros((Ntype*N))
    y_demo = np.zeros((Ntype*N))
    #w_onehot = np.zeros((Ntype*N, Ntype))
    #y_onehot = np.zeros((Ntype*N, Ntype))
    
    for i in range(Ntype):
        if(parents[i]==i):
            tmin = 0
        else:
            tmin = t_trans[i]
        if(np.any(parents==i)):
            tmax = max(np.max(t_trans[np.where(parents==i)[0]]), t[y==i].max())
        else:
            tmax = t[y==i].max()
        t_demo[i*N:(i+1)*N] = np.linspace(tmin, tmax, N)
        y_demo[i*N:(i+1)*N] = i
        #y_onehot[i*N:(i+1)*N, i] = 1
        #w_onehot[i*N:(i+1)*N, parents[i]] = 1
    if(genes is None):
        Uhat, Shat = odeBrNumpy(t_demo.reshape(-1,1),
                                y_demo,
                                w,
                                alpha=alpha,
                                beta=beta,
                                gamma=gamma,
                                t_trans=t_trans,
                                ts=ts,
                                u0=u0,
                                s0=s0,
                                scaling=scaling)
    else:
        gene_indices = np.array([np.where(adata.var_names==x)[0][0] for x in genes])
        Uhat, Shat = odeBrNumpy(t_demo.reshape(-1,1),
                                y_demo,
                                w,
                                alpha=alpha[:,gene_indices],
                                beta=beta[:,gene_indices],
                                gamma=gamma[:,gene_indices],
                                t_trans=t_trans,
                                ts=ts[:,gene_indices],
                                u0=u0[:,gene_indices],
                                s0=s0[:,gene_indices],
                                scaling=scaling[gene_indices])
        
    return t_demo, y_demo, Uhat, Shat

def getPredictionVAEpp(adata, key, scv_key=None):
    U, S = adata.layers["Mu"], adata.layers["Ms"]
    alpha = adata.var[f"{key}_alpha"].to_numpy()
    beta = adata.var[f"{key}_beta"].to_numpy()
    gamma = adata.var[f"{key}_gamma"].to_numpy()
    t = adata.obs[f"{key}_time"].to_numpy()
    scaling = adata.var[f"{key}_scaling"].to_numpy()
    sigma_u, sigma_s = adata.var[f"{key}_sigma_u"].to_numpy(), adata.var[f"{key}_sigma_s"].to_numpy()
    u0, s0 = adata.layers[f"{key}_u0"], adata.layers[f"{key}_s0"]
    t0 = adata.obs[f"{key}_t0"].to_numpy()
    
    if( (f"{key}_uhat" not in adata.layers) or (f"{key}_shat" not in adata.layers)):
        Uhat, Shat = predSUNUmpy((t-t0).reshape(-1,1), u0, s0, alpha, beta, gamma)
        Uhat = Uhat*scaling
    else:
        Uhat, Shat = adata.layers[f"{key}_uhat"], adata.layers[f"{key}_shat"]
    
    train_idx, test_idx = adata.uns[f"{key}_train_idx"], adata.uns[f"{key}_test_idx"]
    logp_train = -(U[train_idx]-Uhat[train_idx])**2/(2*sigma_u**2)-(S[train_idx]-Shat[train_idx])**2/(2*sigma_s**2) - np.log(sigma_u) - np.log(sigma_s) - np.log(2*np.pi)
    logp_test = -(U[test_idx]-Uhat[test_idx])**2/(2*sigma_u**2)-(S[test_idx]-Shat[test_idx])**2/(2*sigma_s**2) - np.log(sigma_u) - np.log(sigma_s) - np.log(2*np.pi)
    
    if(scv_key is None):
        logp_train = np.nanmean(np.sum(logp_train,1))
        logp_test = np.nanmean(np.sum(logp_test,1))
    else:
        scv_mask = ~np.isnan(adata.var[f"{scv_key}_alpha"].to_numpy())
        logp_train = np.nanmean(np.sum(logp_train[:,scv_mask],1))
        logp_test = np.nanmean(np.sum(logp_test[:,scv_mask],1))
    return Uhat, Shat, logp_train, logp_test
    
import numpy as np
from scipy.stats import spearmanr
from ..model.model_util import initParams, predSUNumpy, odeNumpy, odeWeightedNumpy, scvPred, scvPredSingle

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
        toff = adata.var[f"{key}_t_"].to_numpy()
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
        t_demo = np.concatenate((np.linspace(0,toff[i],N), np.linspace(toff[i],T[:,idx].max()+0.01,N)))
        T_demo[:,i] = t_demo
        uhat, shat = scvPredSingle(t_demo,alpha[idx],beta[idx],gamma[idx],toff[idx],scaling=scaling[idx], uinit=0, sinit=0)
        Uhat[:,i] = uhat
        Shat[:,i] = shat
    return T_demo, Uhat, Shat

def getPredictionVanilla(adata, key='vanilla'):
    U, S = adata.layers["Mu"], adata.layers["Ms"]
    #Vanilla VAE
    alpha = adata.var[f"{key}_alpha"].to_numpy()
    beta = adata.var[f"{key}_beta"].to_numpy()
    gamma = adata.var[f"{key}_gamma"].to_numpy()
    toff = adata.var[f"{key}_t_"].to_numpy()
    ton = adata.var[f"{key}_ton"].to_numpy()
    t = adata.obs[f"{key}_time"].to_numpy()
    scaling = adata.var[f"{key}_scaling"].to_numpy()
    sigma_u, sigma_s = adata.var[f"{key}_sigma_u"].to_numpy(), adata.var[f"{key}_sigma_s"].to_numpy()
    
    Uhat, Shat = odeNumpy(t.reshape(-1,1), alpha, beta, gamma, ton, toff, scaling)
    
    logp = -(U-Uhat)**2/(2*sigma_u**2)-(S-Shat)**2/(2*sigma_s**2) - np.log(sigma_u) - np.log(sigma_s) - np.log(2*np.pi)
    logp = np.nanmean(np.sum(logp,1))
    
    return Uhat, Shat, logp

def getPredictionVanillaDemo(adata, key='vanilla', genes=None, N=100):
    alpha = adata.var[f"{key}_alpha"].to_numpy()
    beta = adata.var[f"{key}_beta"].to_numpy()
    gamma = adata.var[f"{key}_gamma"].to_numpy()
    toff = adata.var[f"{key}_t_"].to_numpy()
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
    



def getPredictionMix(adata, key):
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
    y_onehot = np.zeros(adata.obsm[f"{key}_ptype"].shape)
    w_onehot = np.zeros(adata.obsm[f"{key}_ptype"].shape)
    for i in range(alpha.shape[0]):
        y_onehot[y==i, i] = 1
        w_onehot[y==i, i] = 1
    
    Uhat, Shat = odeWeightedNumpy(t,
                                  y_onehot,
                                  w_onehot,
                                  alpha=alpha,
                                  beta=beta,
                                  gamma=gamma,
                                  t_trans=t_trans,
                                  ts=ts,
                                  u0=u0,
                                  s0=s0,
                                  scaling=scaling)
    logp = -(adata.layers['Mu']-Uhat)**2/(2*sigma_u**2)-(adata.layers['Ms']-Shat)**2/(2*sigma_s**2) - np.log(sigma_u) - np.log(sigma_s) - np.log(2*np.pi)
    logp = np.nanmean(logp.sum(1))
    return Uhat, Shat, logp

def getPredictionMixDemo(adata, key, genes=None, N=100):
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
    w_onehot = np.zeros((Ntype*N, Ntype))
    y_onehot = np.zeros((Ntype*N, Ntype))
    
    for i in range(Ntype):
        tmin = t_trans[i]
        if(np.any(parents==i)):
            tmax = np.max(t_trans[np.where(parents==i)[0]])
        else:
            tmax = t[y==i].max()
        t_demo[i*N:(i+1)*N] = np.linspace(tmin, tmax, N)
        y_demo[i*N:(i+1)*N] = i
        y_onehot[y_demo==i, i] = 1
        w_onehot[y_demo==i, parents[i]] = 1
    
    Uhat, Shat = odeWeightedNumpy(t_demo.reshape(-1,1),
                                  y_onehot,
                                  w_onehot,
                                  alpha=alpha,
                                  beta=beta,
                                  gamma=gamma,
                                  t_trans=t_trans,
                                  ts=ts,
                                  u0=u0,
                                  s0=s0,
                                  scaling=scaling)
    return t_demo, y_demo, Uhat, Shat
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import pairwise_distances
from ..model.model_util import initParams, predSUNumpy, odeNumpy, ode_br_numpy, scvPred, scvPredSingle, optimal_transport_duality_gap

def getMSE(U,S,Uhat,Shat):
    return np.mean((U-Uhat)**2+(S-Shat)**2)

def getMAE(U,S,Uhat,Shat):
    return np.mean(np.abs(U-Uhat)+np.abs(S-Shat))
    
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
    #store gene likelihood
    adata.var[f"{key}_likelihood_train"] = logp_train.mean(0)
    adata.var[f"{key}_likelihood_test"] = logp_test.mean(0)
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

def transition_prob_util(x_embed, t, cell_labels, nbin=20, epsilon = 0.05, batch_size = 5, lambda1 = 1, lambda2 = 50, max_iter = 2000, q = 0.01):
    cell_types = np.unique(cell_labels)
    Ntype = len(cell_types)
    dt = (np.quantile(t,0.999)-t.min())/(nbin) #time resolution
    
    P = np.zeros((Ntype, Ntype))
    t_trans = []
    for i, x in enumerate(cell_types): #child type
        mask = cell_labels==x
        t0 = np.quantile(t[mask], q) #estimated transition time
        t_trans.append(t0)
        
        mask1 = (t>=t0-dt) & (t<t0) 
        mask2 = (t>=t0) & (t<t0+dt) & mask
        
        if(np.any(mask1) and np.any(mask2)):
            x1, x2 = x_embed[mask1], x_embed[mask2]
            C = pairwise_distances(x1, x2, metric='sqeuclidean', n_jobs=-1)
            C = C/np.median(C)
            g = np.power(np.sum(mask2)/np.sum(mask1), 1/dt)
            G = np.ones((C.shape[0]))*g
            print(g)
            
            Pi = optimal_transport_duality_gap(C,G,lambda1, lambda2, epsilon, 5, 0.01, 10000, 1, max_iter)
            Pi[np.isnan(Pi)] = 0
            Pi[np.isinf(Pi)] = 0
            
            #Sum the weights of each cell type
            cell_labels_1 = cell_labels[mask1]
            cell_labels_2 = cell_labels[mask2]
            
            for j, y in enumerate(cell_types): #parent
                if(np.any(cell_labels_1==y) and np.any(cell_labels_2==x)):
                    P[i,j] = np.sum(np.array(Pi[cell_labels_1==y]))
                    
        sum_p = P[i].sum()
        sum_p = sum_p + (sum_p==0)
        P[i] = P[i]/sum_p
    
    return P, cell_types, t_trans
    
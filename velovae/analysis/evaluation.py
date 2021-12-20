import numpy as np
import torch 
import torch.nn as nn
import anndata
import scvelo as scv
import scanpy
from pandas import DataFrame, Index
from ..model.model_util import scvPred, makeDir, predSUNumpy, odeWeightedNumpy
from .evaluation_util import *
from velovae.plotting import plotPhaseGrid, plotSigGrid, plotClusterGrid, plotTimeGrid




def getMetric(adata, method, key, scv_mask=True):
    """
    Get specific metrics given a method.
    """
    stats = {}
    if method=='scvelo':
        Uhat, Shat, logp = getPredictionSCV(adata, key)
    elif method=='vanilla':
        Uhat, Shat, logp = getPredictionVanilla(adata, key)
    elif method=='branching':
        Uhat, Shat, logp = getPredictionMix(adata, key)
    U, S = adata.layers['Mu'], adata.layers['Ms']
    
    if(scv_mask):
        try:
            gene_mask = ~np.isnan(adata.var['fit_alpha'].to_numpy())
            stats['MSE'] = np.nanmean((U[:,gene_mask]-Uhat[:,gene_mask])**2+(S[:,gene_mask]-Shat[:,gene_mask])**2)
            stats['MAE'] = np.nanmean(np.abs(U[:,gene_mask]-Uhat[:,gene_mask])+np.abs(S[:,gene_mask]-Shat[:,gene_mask]))
        except KeyError:
            print('Warning: scvelo fitting not found! Compute the full MSE/MAE instead.')
            stats['MSE'] = np.nanmean((U-Uhat)**2+(S-Shat)**2)
            stats['MAE'] = np.nanmean(np.abs(U-Uhat)+np.abs(S-Shat))
    else:
        stats['MSE'] = np.nanmean((U-Uhat)**2+(S-Shat)**2)
        stats['MAE'] = np.nanmean(np.abs(U-Uhat)+np.abs(S-Shat))
    stats['LL'] = logp
    if('latent_time' in adata.obs):
        tscv = adata.obs['latent_time'].to_numpy()
        if(method=='scvelo'):
            T_scv = adata.layers["fit_t"]
            mask = ~(np.isnan(adata.var['fit_alpha'].to_numpy()))
            corr, pval = 0, 0
            for i in range(adata.n_vars):
                if(mask[i]):
                    c, p = spearmanr(T_scv[:,i], tscv)
                    corr += c
                    pval += p
            corr = corr / np.sum(mask)
            pval = pval / np.sum(mask)
        else:
            t = adata.obs[f"{key}_time"]
            corr, pval = spearmanr(t, tscv)
        stats['corr'] = corr
    return stats

def testAllTransition(adata, key, k=2):
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
    
    t = adata.obs[f"{key}_time"].to_numpy()
    y = adata.obs[f"{key}_label"].to_numpy()
    
    
    #Get the best k parents
    Ntype, G = alpha.shape
    E = np.eye(Ntype)
    y_onehot = E[y]
    parents = np.zeros((Ntype, k))
    for i in range(Ntype):
        idx_w = np.flip(np.argsort(w[i]))
        parents[i] = idx_w[:k]
    print(parents)
    
    #Try every possible graph and print out the log likelihood
    logp = np.zeros((int(k**Ntype)))
    for i in range(int(k**Ntype)):
        w_onehot = np.zeros((Ntype, Ntype))
        m = i
        for j in range(Ntype):
            idx = m % 2
            w_onehot[j, int(parents[j, idx])] = 1
            m = m//2
        w_onehot = w_onehot[y]
        Uhat, Shat = odeWeightedNumpy(t, y_onehot,
                                      w_onehot,
                                      alpha=alpha,
                                      beta=beta,
                                      gamma=gamma,
                                      t_trans=t_trans,
                                      ts=ts,
                                      u0=u0,
                                      s0=s0,
                                      sigma_u = sigma_u,
                                      sigma_s = sigma_s,
                                      scaling=scaling)
        logp[i] = np.sum(np.mean((U-Uhat)**2/(2*sigma_u**2)+(S-Shat)**2/(2*sigma_s**2)+np.log(sigma_u)+np.log(sigma_s)+np.log(2*np.pi), 0))
        print(i,'\t', logp[i])
    idx_p = np.flip(np.argsort(logp))
    print(idx_p[:10])
    
    return

def postAnalysis(adata, methods, keys, genes=[], plot_type=["signal"], Nplot=500, embed="umap", grid_size=(1,1), save_path="figures"):
    """
    Main function for post analysis.
    adata: anndata object
    methods: list of strings containing the methods to compare with
    keys: key of each method (to extract parameters from anndata)
    genes: genes to plot
    plot_type: currently supports phase, signal (u/s vs. t), time and cell type
    """
    makeDir(save_path)
    U, S = adata.layers["Mu"], adata.layers["Ms"]
    X_embed = adata.obsm[f"X_{embed}"]
    cell_labels_raw = adata.obs["clusters"].to_numpy()
    cell_types_raw = np.unique(cell_labels_raw)
    label_dic = {}
    for i, x in enumerate(cell_types_raw):
        label_dic[x] = i
    cell_labels = np.array([label_dic[x] for x in cell_labels_raw])
    gene_indices = np.array([np.where(adata.var_names==x)[0][0] for x in genes])
    
    Ntype = len(cell_types_raw)
    
    methods = np.unique(methods)
    stats = {}
    Uhat, Shat = {},{}
    That, Yhat = {},{}
    
    for i, method in enumerate(methods):
        stats_i = getMetric(adata, method, keys[i])
        stats[method] = stats_i
        if(method=='scvelo'):
            t_i, Uhat_i, Shat_i = getPredictionSCVDemo(adata, keys[i], genes, Nplot)
            Yhat[method] = np.concatenate((np.zeros((Nplot)), np.ones((Nplot))))
        elif(method=='vanilla'):
            t_i, Uhat_i, Shat_i = getPredictionVanillaDemo(adata, keys[i], genes, Nplot)
            Yhat[method] = None
        elif(method=='branching'):
            t_i, y_i, Uhat_i, Shat_i = getPredictionMixDemo(adata, keys[i], genes, Nplot)
            Yhat[method] = y_i
        That[method] = t_i
        Uhat[method] = Uhat_i
        Shat[method] = Shat_i
    
    print("---     Post Analysis     ---")
    for method in stats:
        metrics = list(stats[method].keys())
        break
    stats_df = DataFrame({}, index=Index(metrics))
    for i, method in enumerate(methods):
        stats_df.insert(i, method, [stats[method][x] for x in metrics])
    print(stats_df)
    
    print("---   Plotting  Results   ---")
    
    if("type" in plot_type or "all" in plot_type):
        p_given = np.zeros((len(cell_labels),Ntype))
        for i in range(Ntype):
            p_given[cell_labels==i, i] = 1
        Y = {"Labels": p_given}
        
        for i, method in enumerate(methods):
            if(f"{keys[i]}_ptype" in adata.obsm):
                Y[method] = adata.obsm[f"{keys[i]}_ptype"]
            else:
                Y[method] = p_given
            
        
        plotClusterGrid(X_embed, 
                        Y,
                        cell_types_raw, 
                        False, 
                        True,
                        save_path)
    
    if("time" in plot_type or "all" in plot_type):
        T = {}
        std_t = {}
        for i, method in enumerate(methods):
            if(method=='scvelo'):
                T[method] = adata.obs["latent_time"].to_numpy()
                std_t[method] = np.zeros((adata.n_obs))
            else:
                T[method] = adata.obs[f"{keys[i]}_time"].to_numpy()
                std_t[method] = adata.obs[f"{keys[i]}_std_t"].to_numpy()
        plotTimeGrid(T,
                     std_t,
                     X_embed,
                     savefig=True,  
                     path=save_path)
    
    if(len(genes)==0):
        return
    
    if("phase" in plot_type or "all" in plot_type):
        Labels_phase = {}
        Legends_phase = {}
        for i, method in enumerate(methods):
            if(method=='vanilla' or method=='scvelo'):
                Labels_phase[method] = cellState(adata, method, keys[i], gene_indices)
                Legends_phase[method] = ['Induction', 'Repression', 'Off']
            else:
                Labels_phase[method] = adata.obs[f"{keys[i]}_label"]
                Legends_phase[method] = adata.var_names
        plotPhaseGrid(grid_size[0], 
                      grid_size[1],
                      genes,
                      U[:,gene_indices], 
                      S[:,gene_indices],
                      Labels_phase,
                      Legends_phase,
                      Uhat, 
                      Shat,
                      Yhat,
                      savefig=True,
                      path=save_path,
                      figname='test')
    
    if("signal" in plot_type or "all" in plot_type):
        T = {}
        Labels_sig = {}
        Legends_sig = {}
        for i, method in enumerate(methods):
            if(method=='scvelo'):
                methods_ = np.concatenate((methods,['scVelo Global']))
                T[method] = adata.layers[f"{keys[i]}_t"][:,gene_indices]
                T['scvelo Global'] = adata.obs["latent_time"].to_numpy()*20
                Labels_sig[method] = np.array([label_dic[x] for x in adata.obs["clusters"].to_numpy()])
                Labels_sig['scvelo Global'] = Labels_sig[method]
                Legends_sig[method] = cell_types_raw
                Legends_sig['scvelo Global'] = cell_types_raw
            elif(method=='vanilla'):
                T[method] = adata.obs[f"{keys[i]}_time"].to_numpy()
                Labels_sig[method] = np.array([label_dic[x] for x in adata.obs["clusters"].to_numpy()])
                Legends_sig[method] = cell_types_raw
            else:
                T[method] = adata.obs[f"{keys[i]}_time"].to_numpy()
                Labels_sig[method] = adata.obs[f"{keys[i]}_label"].to_numpy()
                Legends_sig[method] = cell_types_raw

        plotSigGrid(grid_size[0], 
                    grid_size[1], 
                    genes,
                    T,
                    U[:,gene_indices], 
                    S[:,gene_indices], 
                    Labels_sig,
                    Legends_sig,
                    That,
                    Uhat, 
                    Shat, 
                    Yhat,
                    savefig=True,  
                    path=save_path, 
                    figname="test")
    
    return
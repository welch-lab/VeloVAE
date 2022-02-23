import numpy as np
import pandas as pd
from pandas import DataFrame, Index
from ..model.model_util import makeDir
from .evaluation_util import *
from velovae.plotting import plot_phase_grid, plot_sig_grid, plot_cluster_grid, plot_time_grid




def getMetric(adata, method, key, scv_key=None, scv_mask=True):
    """
    Get specific metrics given a method.
    key: key for extracting the ODE parameters learned by the model
    scv_key (optional): key of the scvelo fitting, used for filtering the genes (only effective if scv_mask=True)
    scv_mask: whether to filter out the genes not fitted by scvelo (used for fairness in the comparison of different methods)
    """
    stats = {}
    if method=='scVelo':
        Uhat, Shat, logp_train = getPredictionSCV(adata, key)
        logp_test = "N/A"
    elif method=='Vanilla VAE':
        Uhat, Shat, logp_train, logp_test = getPredictionVanilla(adata, key, scv_key)
    elif method=='VeloVAE':
        Uhat, Shat, logp_train, logp_test = getPredictionVAEpp(adata, key, scv_key)
        
    U, S = adata.layers['Mu'], adata.layers['Ms']
    if(method=='scVelo'):
        train_idx = np.array(range(adata.n_obs)).astype(int)
        test_idx = np.array([])
    else:
        train_idx, test_idx = adata.uns[f"{key}_train_idx"], adata.uns[f"{key}_test_idx"]
    if(scv_mask):
        try:
            gene_mask = ~np.isnan(adata.var['fit_alpha'].to_numpy())
            stats['MSE Train'] = np.nanmean((U[train_idx][:,gene_mask]-Uhat[train_idx][:,gene_mask])**2+(S[train_idx][:,gene_mask]-Shat[train_idx][:,gene_mask])**2)
            stats['MAE Train'] = np.nanmean(np.abs(U[train_idx][:,gene_mask]-Uhat[train_idx][:,gene_mask])+np.abs(S[train_idx][:,gene_mask]-Shat[train_idx][:,gene_mask]))
            if(len(test_idx)>0):
                stats['MSE Test'] = np.nanmean((U[test_idx][:,gene_mask]-Uhat[test_idx][:,gene_mask])**2+(S[test_idx][:,gene_mask]-Shat[test_idx][:,gene_mask])**2)
                stats['MAE Test'] = np.nanmean(np.abs(U[test_idx][:,gene_mask]-Uhat[test_idx][:,gene_mask])+np.abs(S[test_idx][:,gene_mask]-Shat[test_idx][:,gene_mask]))
            else:
                stats['MSE Test'] = "N/A"
                stats['MAE Test'] = "N/A"
        except KeyError:
            print('Warning: scvelo fitting not found! Compute the full MSE/MAE instead.')
            stats['MSE Train'] = np.nanmean((U[train_idx]-Uhat[train_idx])**2+(S[train_idx]-Shat[train_idx])**2)
            stats['MAE Train'] = np.nanmean(np.abs(U[train_idx]-Uhat[train_idx])+np.abs(S[train_idx]-Shat[train_idx]))
            if(len(test_idx)>0):
                stats['MSE Test'] = np.nanmean((U[test_idx]-Uhat[test_idx])**2+(S[test_idx]-Shat[test_idx])**2)
                stats['MAE Test'] = np.nanmean(np.abs(U[test_idx]-Uhat[test_idx])+np.abs(S[test_idx]-Shat[test_idx]))
            else:
                stats['MSE Test'] = "N/A"
                stats['MAE Test'] = "N/A"
    else:
        stats['MSE Train'] = np.nanmean((U[train_idx]-Uhat[train_idx])**2+(S[train_idx]-Shat[train_idx])**2)
        stats['MAE Train'] = np.nanmean(np.abs(U[train_idx]-Uhat[train_idx])+np.abs(S[train_idx]-Shat[train_idx]))
        if(len(test_idx)>0):
            stats['MSE Test'] = np.nanmean((U[test_idx]-Uhat[test_idx])**2+(S[test_idx]-Shat[test_idx])**2)
            stats['MAE Test'] = np.nanmean(np.abs(U[test_idx]-Uhat[test_idx])+np.abs(S[test_idx]-Shat[test_idx]))
        else:
            stats['MSE Test'] = "N/A"
            stats['MAE Test'] = "N/A"
    stats['LL Train'] = logp_train
    stats['LL Test'] = logp_test
    if('tprior' in adata.obs):
        tprior = adata.obs['tprior'].to_numpy()
        t = adata.obs["latent_time"].to_numpy() if (method=='scVelo') else adata.obs[f"{key}_time"].to_numpy()
        corr, pval = spearmanr(t, tprior)
        stats['corr'] = corr
    return stats

def postAnalysis(adata, methods, keys, test_id, genes=[], plot_type=["signal"], Nplot=500, frac=0.5, embed="umap", grid_size=(1,1), save_path="figures"):
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
    if(len(genes)>0):
        gene_indices = np.array([np.where(adata.var_names==x)[0][0] for x in genes])
    else:
        print("Warning: No gene names are provided. Randomly select a gene...")
        gene_indices = np.random.choice(adata.n_vars, grid_size[0]*grid_size[1], replace=False).astype(int)
        genes = adata.var_names[gene_indices].to_numpy()
        print(genes)
    
    Ntype = len(cell_types_raw)
    
    stats = {}
    Uhat, Shat, V = {},{},{}
    That, Yhat = {},{}
    
    scv_idx = np.where(methods=='scVelo')[0]
    scv_key = keys[scv_idx[0]] if(len(scv_idx)>0) else None
    for i, method in enumerate(methods):
        stats_i = getMetric(adata, method, keys[i], scv_key, (scv_key is None) )
        stats[method] = stats_i
        if(method=='scVelo'):
            t_i, Uhat_i, Shat_i = getPredictionSCVDemo(adata, keys[i], genes, Nplot)
            Yhat[method] = np.concatenate((np.zeros((Nplot)), np.ones((Nplot))))
            V[method] = adata.layers["velocity"][:,gene_indices]
        elif(method=='Vanilla VAE'):
            t_i, Uhat_i, Shat_i = getPredictionVanillaDemo(adata, keys[i], genes, Nplot)
            Yhat[method] = None
            V[method] = adata.layers[f"{keys[i]}_velocity"][:,gene_indices]
        elif(method=='VeloVAE'):
            Uhat_i, Shat_i, logp_train, logp_test = getPredictionVAEpp(adata, keys[i], None)
            V[method] = adata.layers[f"{keys[i]}_velocity"][:,gene_indices]
            
            t_i = adata.obs[f'{keys[i]}_time'].to_numpy()
            cell_labels_raw = adata.obs["clusters"].to_numpy()
            cell_types_raw = np.unique(cell_labels_raw)
            cell_labels = np.zeros((len(cell_labels_raw)))
            for i in range(len(cell_types_raw)):
                cell_labels[cell_labels_raw==cell_types_raw[i]] = i
            Yhat[method] = cell_labels
        That[method] = t_i
        Uhat[method] = Uhat_i[:,gene_indices] if method=='VeloVAE' else Uhat_i
        Shat[method] = Shat_i[:,gene_indices] if method=='VeloVAE' else Shat_i
    
    print("---     Post Analysis     ---")
    print(f"Dataset Size: {adata.n_obs} cells, {adata.n_vars} genes")
    for method in stats:
        metrics = list(stats[method].keys())
        break
    stats_df = DataFrame({}, index=Index(metrics))
    for i, method in enumerate(methods):
        stats_df.insert(i, method, [stats[method][x] for x in metrics])
    pd.set_option("precision", 4)
    print(stats_df)
    
    print("---   Plotting  Results   ---")
    
    if("type" in plot_type or "all" in plot_type):
        p_given = np.zeros((len(cell_labels),Ntype))
        for i in range(Ntype):
            p_given[cell_labels==i, i] = 1
        Y = {"True": p_given}
        
        for i, method in enumerate(methods):
            if(f"{keys[i]}_ptype" in adata.obsm):
                Y[method] = adata.obsm[f"{keys[i]}_ptype"]
            else:
                Y[method] = p_given
            
        
        plot_cluster_grid(X_embed, 
                          Y,
                          cell_types_raw, 
                          False, 
                          f"{save_path}/cluster_{test_id}.png")
    
    if("time" in plot_type or "all" in plot_type):
        T = {}
        std_t = {}
        capture_time = adata.obs["tprior"].to_numpy() if "tprior" in adata.obs else None
        for i, method in enumerate(methods):
            if(method=='scVelo'):
                T[method] = adata.obs["latent_time"].to_numpy()
                std_t[method] = np.zeros((adata.n_obs))
            else:
                T[method] = adata.obs[f"{keys[i]}_time"].to_numpy()
                std_t[method] = adata.obs[f"{keys[i]}_std_t"].to_numpy()
        plot_time_grid(T,
                       X_embed,
                       capture_time,
                       None,
                       down_sample = max(1,adata.n_obs//5000),
                       figname=f"{save_path}/time_{test_id}.png")
    
    if(len(genes)==0):
        return
    
    if("phase" in plot_type or "all" in plot_type):
        Labels_phase = {}
        Legends_phase = {}
        for i, method in enumerate(methods):
            if(method=='VeloVAE' or method=='scVelo'):
                Labels_phase[method] = cellState(adata, method, keys[i], gene_indices)
                Legends_phase[method] = ['Induction', 'Repression', 'Off']
            else:
                Labels_phase[method] = adata.obs[f"{keys[i]}_label"]
                Legends_phase[method] = adata.var_names
        plot_phase_grid(grid_size[0], 
                        grid_size[1],
                        genes,
                        U[:,gene_indices], 
                        S[:,gene_indices],
                        Labels_phase,
                        Legends_phase,
                        Uhat, 
                        Shat,
                        Yhat,
                        path=save_path,
                        figname=test_id)
    
    if("signal" in plot_type or "all" in plot_type):
        T = {}
        Labels_sig = {}
        Legends_sig = {}
        for i, method in enumerate(methods):
            if(method=='scVelo'):
                methods_ = np.concatenate((methods,['scVelo Global']))
                T[method] = adata.layers[f"{keys[i]}_t"][:,gene_indices]
                T['scVelo Global'] = adata.obs["latent_time"].to_numpy()*20
                Labels_sig[method] = np.array([label_dic[x] for x in adata.obs["clusters"].to_numpy()])
                Labels_sig['scVelo Global'] = Labels_sig[method]
                Legends_sig[method] = cell_types_raw
                Legends_sig['scVelo Global'] = cell_types_raw
            elif(method=='Vanilla VAE' or method=='VeloVAE'):
                T[method] = adata.obs[f"{keys[i]}_time"].to_numpy()
                Labels_sig[method] = np.array([label_dic[x] for x in adata.obs["clusters"].to_numpy()])
                Legends_sig[method] = cell_types_raw
            else:
                T[method] = adata.obs[f"{keys[i]}_time"].to_numpy()
                Labels_sig[method] = adata.obs[f"{keys[i]}_label"].to_numpy()
                Legends_sig[method] = cell_types_raw

        plot_sig_grid(grid_size[0], 
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
                      V,
                      Yhat,
                      frac=frac,
                      down_sample=max(1,adata.n_obs//5000),
                      path=save_path, 
                      figname=test_id)
    
    return stats_df
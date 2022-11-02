import numpy as np
import pandas as pd
from pandas import DataFrame, Index
from ..model.model_util import make_dir
from .evaluation_util import *
from velovae.plotting import plot_cluster, plot_phase_grid, plot_sig_grid, plot_time_grid




def get_metric(adata, method, key, scv_key=None, scv_mask=True):
    """Get specific metrics given a method.
    
    Arguments
    ---------
    adata : :class:`anndata.AnnData`
    key : str
       Key in .var or .varm for extracting the ODE parameters learned by the model
    scv_key : str, optional
       Key for scvelo fitting. Used for filtering the genes (only effective if scv_mask=True)
    scv_mask : bool, optional
       Whether to filter out the genes not fitted by scvelo (used for fairness in the comparison with scVelo)
    
    Returns
    -------
    stats : :class:`pandas.DataFrame`
        Stores the performance metrics. Rows are metric names and columns are method names
    """
    
    stats = {}  #contains the performance metrics
    if method=='scVelo':
        Uhat, Shat, logp_train = get_pred_scv(adata, key)
        logp_test = "N/A"
    elif method=='Vanilla VAE':
        Uhat, Shat, logp_train, logp_test = get_pred_vanilla(adata, key, scv_key)
    elif method=='VeloVAE' or method=="FullVB":
        Uhat, Shat, logp_train, logp_test = get_pred_velovae(adata, key, scv_key, method=="FullVB")
    elif(method=='BrODE'):
        Uhat, Shat, logp_train, logp_test = get_pred_brode(adata, key, scv_key)

    U, S = adata.layers['Mu'], adata.layers['Ms']
    if(method=='scVelo'):
        train_idx = np.array(range(adata.n_obs)).astype(int)
        test_idx = np.array([])
    else:
        train_idx, test_idx = adata.uns[f"{key}_train_idx"], adata.uns[f"{key}_test_idx"]
    
    if(scv_mask):
        try:
            gene_mask = ~np.isnan(adata.var['fit_alpha'].to_numpy()) #For all genes not fitted, scVelo sets the ODE parameters to nan.
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

def transition_prob(adata, embed_key, tkey, label_key, nbin=20, epsilon = 0.05, batch_size = 5, lambda1 = 1, lambda2 = 50, max_iter = 2000, q = 0.01):
    x_embed = adata.obsm[embed_key]
    t = adata.obs[tkey].to_numpy()
    cell_labels = np.array([str(x) for x in adata.obs[label_key].to_numpy()])
    
    p_trans, cell_types, t_trans = transition_prob_util(x_embed,
                                                        t,
                                                        cell_labels,
                                                        nbin, 
                                                        epsilon, 
                                                        batch_size, 
                                                        lambda1, 
                                                        lambda2, 
                                                        max_iter, 
                                                        q)
    return p_trans, cell_types, t_trans

def post_analysis(adata, 
                  test_id,
                  methods, 
                  keys, 
                  compute_metrics=False,
                  genes=[], 
                  plot_type=["signal"], 
                  cluster_key="clusters",
                  nplot=500, 
                  frac=0.0, 
                  embed="umap", 
                  grid_size=(1,1),
                  save_path="figures",
                  **kwargs):
    """Main function for post analysis. This function computes performance metrics and generates plots based on user input.
    
    Arguments
    ---------
    adata : :class:`anndata.AnnData`
    test_id : str
        Used for naming the figures. 
        For example, it can be set as the name of the dataset.
    methods : string list
        Contains the methods to compare with. 
        Valid methods are "scVelo", "Vanilla VAE", "VeloVAE" and "BrODE".
    keys : string list
        Used for extracting ODE parameters from .var or .varm from anndata
        It should be of the same length as methods.
    compute_metrics : bool, optional
        Whether to compute the performance metrics for the methods
    genes : string list, optional 
        Genes to plot. Used when plot_type contains "phase" or "signal"
    plot_type : string list, optional
        Type of plots to generate.
        Currently supports phase, signal (u/s/v vs. t), time and cell type
    cluster_key : str, optional
        Key in .obs containing the cell type labels
    nplot : int, optional
        (Optional) Number of data points in the prediction (or for each cell type in VeloVAE and BrODE).
        This is to save computation. For plotting the prediction, we don't need 
        as many points as the original dataset contains.
    frac : float in (0,1), optional
        Parameter for the loess plot. 
        A higher value means larger time window and the resulting fitted line will
        be smoother. 
    embed : str, optional
        2D embedding used for visualization of time and cell type.
        The true key for the embedding is f"X_{embed}" in .obsm
    grid_size : int tuple, optional
        Grid size for plotting the genes.
        n_row*n_col >= len(genes)
    save_path : str, optional
        Path to save the figures.
    
    Returns
    -------
    stats_df : :class:`pandas.DataFrame`
        Contains the performance metrics of all methods.
        Saves the figures to 'save_path'.
    """
    make_dir(save_path)
    U, S = adata.layers["Mu"], adata.layers["Ms"]
    X_embed = adata.obsm[f"X_{embed}"]
    cell_labels_raw = adata.obs[cluster_key].to_numpy()
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
    
    scv_idx = np.where(np.array(methods)=='scVelo')[0]
    scv_key = keys[scv_idx[0]] if(len(scv_idx)>0) else None
    for i, method in enumerate(methods):
        if(compute_metrics):
            stats_i = get_metric(adata, method, keys[i], scv_key, (scv_key is not None) )
            stats[method] = stats_i
        
        if(method=='scVelo'):
            t_i, Uhat_i, Shat_i = get_pred_scv_demo(adata, keys[i], genes, nplot)
            Yhat[method] = np.concatenate((np.zeros((nplot)), np.ones((nplot))))
            V[method] = adata.layers["velocity"][:,gene_indices]
        elif(method=='Vanilla VAE'):
            t_i, Uhat_i, Shat_i = get_pred_vanilla_demo(adata, keys[i], genes, nplot)
            Yhat[method] = None
            V[method] = adata.layers[f"{keys[i]}_velocity"][:,gene_indices]
        elif(method=='VeloVAE' or method=='FullVB'):
            Uhat_i, Shat_i = get_pred_velovae_demo(adata, keys[i], genes, method=='FullVB')
            V[method] = adata.layers[f"{keys[i]}_velocity"][:,gene_indices]
            t_i = adata.obs[f'{keys[i]}_time'].to_numpy()
            cell_labels_raw = adata.obs[cluster_key].to_numpy()
            cell_types_raw = np.unique(cell_labels_raw)
            cell_labels = np.zeros((len(cell_labels_raw)))
            for i in range(len(cell_types_raw)):
                cell_labels[cell_labels_raw==cell_types_raw[i]] = i
            Yhat[method] = cell_labels
        elif(method=='BrODE'):
            t_i, y_brode, Uhat_i, Shat_i = get_pred_brode_demo(adata, keys[i], genes)
            V[method] = adata.layers[f"{keys[i]}_velocity"][:,gene_indices]
            Yhat[method] = y_brode
        That[method] = t_i
        t_brode = t_i
        Uhat[method] = Uhat_i
        Shat[method] = Shat_i
    
    if(compute_metrics):
        print("---     Post Analysis     ---")
        print(f"Dataset Size: {adata.n_obs} cells, {adata.n_vars} genes")
        for method in stats:
            metrics = list(stats[method].keys())
            break
        stats_df = DataFrame({}, index=Index(metrics))
        for i, method in enumerate(methods):
            stats_df.insert(i, method, [stats[method][x] for x in metrics])
        pd.set_option("display.precision", 4)
        print(stats_df)
    
    print("---   Plotting  Results   ---")
    if('cluster' in plot_type or "all" in plot_type):
        plot_cluster(adata.obsm[f"X_{embed}"], adata.obs[cluster_key].to_numpy(), embed=embed, save=f"{save_path}/{test_id}_umap.png")
    
    if("time" in plot_type or "all" in plot_type):
        T = {}
        capture_time = adata.obs["tprior"].to_numpy() if "tprior" in adata.obs else None
        for i, method in enumerate(methods):
            if(method=='scVelo'):
                T[method] = adata.obs["latent_time"].to_numpy()
            else:
                T[method] = adata.obs[f"{keys[i]}_time"].to_numpy()
        plot_time_grid(T,
                       X_embed,
                       capture_time,
                       None,
                       down_sample = min(10, max(1,adata.n_obs//5000)),
                       save=f"{save_path}/{test_id}_time.png")
    
    if(len(genes)==0):
        return
    
    format = kwargs["format"] if "format"  in kwargs else "png"
    if("phase" in plot_type or "all" in plot_type):
        Labels_phase = {}
        Legends_phase = {}
        for i, method in enumerate(methods):
            Labels_phase[method] = cell_state(adata, method, keys[i], gene_indices)
            Legends_phase[method] = ['Induction', 'Repression', 'Off']
            
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
                        figname=test_id,
                        format=format)
    
    if("signal" in plot_type or "all" in plot_type):
        T = {}
        Labels_sig = {}
        Legends_sig = {}
        
        for i, method in enumerate(methods):
            Legends_sig[method] = cell_types_raw
            if(method=='scVelo'):
                methods_ = np.concatenate((methods,['scVelo Global']))
                T[method] = adata.layers[f"{keys[i]}_t"][:,gene_indices]
                T['scVelo Global'] = adata.obs["latent_time"].to_numpy()*20
                Labels_sig[method] = np.array([label_dic[x] for x in adata.obs[cluster_key].to_numpy()])
                Labels_sig['scVelo Global'] = Labels_sig[method]
                Legends_sig['scVelo Global'] = cell_types_raw
            else:
                T[method] = adata.obs[f"{keys[i]}_time"].to_numpy()
                Labels_sig[method] = np.array([label_dic[x] for x in adata.obs[cluster_key].to_numpy()])
        sparsity_correction = kwargs["sparsity_correction"] if "sparsity_correction" in kwargs else False 
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
                      down_sample=min(10, max(1,adata.n_obs//5000)),
                      sparsity_correction=sparsity_correction,
                      path=save_path, 
                      figname=test_id,
                      format=format)
    if(compute_metrics):
        return stats_df
    return

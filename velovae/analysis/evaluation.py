import numpy as np
import pandas as pd
from pandas import DataFrame, Index
from ..model.model_util import make_dir
from .evaluation_util import *
from velovae.plotting import get_colors, plot_cluster, plot_phase_grid, plot_sig_grid, plot_time_grid




def get_metric(adata, 
               method, 
               key, 
               vkey,  
               cluster_key="clusters", 
               scv_key=None, 
               scv_mask=True, 
               cluster_edges=[], 
               embed='umap'):
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
    gene_mask = adata.var[scv_key] if scv_key in adata.var else None
    
    if method=='scVelo':
        mse_train, mse_test, mae_train, mae_test, logp_train, logp_test = get_err_scv(adata)
    elif method=='Vanilla VAE':
        mse_train, mse_test, mae_train, mae_test, logp_train, logp_test = get_err_vanilla(adata, key, gene_mask)
    elif method=='VeloVAE' or method=='FullVB':
        mse_train, mse_test, mae_train, mae_test, logp_train, logp_test = get_err_velovae(adata, key, gene_mask, 'FullVB' in method)
    elif(method=='BrODE'):
        mse_train, mse_test, mae_train, mae_test, logp_train, logp_test = get_err_brode(adata, key, gene_mask)
    elif(method=='Discrete VeloVAE' or method=='Discrete FullVB'):
        mse_train, mse_test, mae_train, mae_test, logp_train, logp_test = get_err_velovae(adata, key, gene_mask, 'FullVB' in method, True)
    elif(method=='UniTVelo'):
        mse_train, mse_test, mae_train, mae_test, logp_train, logp_test = get_err_utv(adata, gene_mask)
    elif(method=='DeepVelo'):
        mse_train, mse_test, mae_train, mae_test, logp_train, logp_test = get_err_dv(adata, gene_mask)
    elif('PyroVelocity' in method):
        if('err' in adata.uns):
            mse_train, mse_test, mae_train, mae_test, logp_train, logp_test  = adata.uns['err']['MSE Train'], adata.uns['err']['MSE Test'],\
                                                                               adata.uns['err']['MAE Train'], adata.uns['err']['MAE Test'],\
                                                                               adata.uns['err']['LL Train'], adata.uns['err']['LL Test']
        else:
            mse_train, mse_test, mae_train, mae_test, logp_train, logp_test = get_err_pv(adata, key, gene_mask, not 'Continuous' in method)
    elif(method=='VeloVI'):
        mse_train, mse_test, mae_train, mae_test, logp_train, logp_test = get_err_velovi(adata, key, gene_mask)
    
    if(method in ['scVelo', 'UniTVelo', 'DeepVelo']):
        logp_test = 'N/A'
        mse_test = 'N/A'
        logp_test = 'N/A'
        mae_test = 'N/A'
    
    if(method == 'DeepVelo'):
        logp_train = 'N/A'

    stats['MSE Train'] = mse_train
    stats['MSE Test'] = mse_test
    stats['MAE Train'] = mae_train
    stats['MAE Test'] = mae_test
    stats['LL Train'] = logp_train
    stats['LL Test'] = logp_test
    
    if('tprior' in adata.obs):
        if(method == 'DeepVelo'):
            stats['corr'] = 'N/A'
        else:
            tprior = adata.obs['tprior'].to_numpy()
            t = adata.obs["latent_time"].to_numpy() if (method in ['scVelo','UniTVelo']) else adata.obs[f"{key}_time"].to_numpy()
            corr, pval = spearmanr(t, tprior)
            stats['corr'] = corr
    
    if(not f"{vkey}_{embed}" in adata.obsm):
        print("Computing velocity embedding using scVelo")
        try:
            from scvelo.tl import velocity_graph, velocity_embedding
            velocity_graph(adata, vkey=vkey)
            velocity_embedding(adata, vkey=vkey, basis=embed)
        except ImportError:
            print("Please install scVelo to compute velocity embedding.\nSkipping metrics 'Cross-Boundary Direction Correctness' and 'In-Cluster Coherence'.")
    
    #Filter extreme values
    v_embed = adata.obsm[f'{vkey}_{embed}']
    idx_extreme = np.where(np.any(np.isnan(v_embed),1))[0]
    if(len(idx_extreme)>0):
        v_filt = np.stack([np.nanmean(v_embed[:,0][adata.uns['neighbors']['indices'][idx_extreme]],1), \
                           np.nanmean(v_embed[:,1][adata.uns['neighbors']['indices'][idx_extreme]],1)]).T
        if(np.any(np.isnan(v_filt)) or np.any(np.isinf(v_filt))):
            v_filt[np.where(np.isnan(v_filt) or np.isinf(v_filt))[0]] = 0
        v_embed[idx_extreme] = v_filt 
    
    #Compute velocity metrics
    if(len(cluster_edges)>0):
        _, mean_cbdir = cross_boundary_correctness(adata, cluster_key, vkey, cluster_edges, x_emb=f"X_{embed}")
        stats[f'Cross-Boundary Direction Correctness (embed)'] = mean_cbdir
        _, mean_cbdir = cross_boundary_correctness(adata, cluster_key, vkey, cluster_edges, x_emb="Ms")
        stats['Cross-Boundary Direction Correctness'] = mean_cbdir
    _, mean_iccoh = inner_cluster_coh(adata, cluster_key, vkey)
    stats['In-Cluster Coherence'] = mean_iccoh
    
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
                  scv_key=None,
                  compute_metrics=True,
                  raw_count=False,
                  genes=[], 
                  plot_type=[], 
                  cluster_key="clusters",
                  cluster_edges=[],
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
    #Retrieve data
    if(raw_count):
        U, S = adata.layers["unspliced"].A, adata.layers["spliced"].A
    else:
        U, S = adata.layers["Mu"], adata.layers["Ms"]
    X_embed = adata.obsm[f"X_{embed}"]
    cell_labels_raw = adata.obs[cluster_key].to_numpy()
    cell_types_raw = np.unique(cell_labels_raw)
    label_dic = {}
    for i, x in enumerate(cell_types_raw):
        label_dic[x] = i
    cell_labels = np.array([label_dic[x] for x in cell_labels_raw])
    
    #Get gene indices
    if(len(genes)>0):
        gene_indices = []
        gene_rm = []
        for gene in genes:
            idx = np.where(adata.var_names==gene)[0]
            if(len(idx)>0):
                gene_indices.append(idx[0])
            else:
                print(f"Warning: gene name {gene} not found in AnnData. Removed.")
                gene_rm.append(gene)
        for gene in gene_rm:
            genes.remove(gene)
        
        if(len(gene_indices)==0):
            print("Warning: No gene names found. Randomly select genes...")
            gene_indices = np.random.choice(adata.n_vars, grid_size[0]*grid_size[1], replace=False).astype(int)
            genes = adata.var_names[gene_indices].to_numpy()
    else:
        print("Warning: No gene names are provided. Randomly select genes...")
        gene_indices = np.random.choice(adata.n_vars, grid_size[0]*grid_size[1], replace=False).astype(int)
        genes = adata.var_names[gene_indices].to_numpy()
        print(genes)
    
    Ntype = len(cell_types_raw)
    
    stats = {}
    Uhat, Shat, V = {},{},{}
    That, Yhat = {},{}
    
    scv_idx = np.where(np.array(methods)=='scVelo')[0]
    vkeys = []
    for i, method in enumerate(methods):
        vkey = 'velocity' if method in ['scVelo','UniTVelo','DeepVelo'] else f'{keys[i]}_velocity'
        vkeys.append(vkey)
    
    # Compute metrics and generate plots for each method
    for i, method in enumerate(methods):
        if(compute_metrics):
            stats_i = get_metric(adata, 
                                 method, 
                                 keys[i], 
                                 vkeys[i], 
                                 cluster_key, 
                                 scv_key, 
                                 (scv_key in adata.var),
                                 cluster_edges,
                                 embed)
            method_ = f"{method} ({keys[i]})" if method in stats else method# avoid duplicate methods with different keys
            stats[method_] = stats_i
        #Compute prediction for the purpose of plotting (a fixed number of plots)
        if('phase' in plot_type or 'signal' in plot_type or 'all' in plot_type):
            #Integer-encoded cell type
            cell_labels_raw = adata.obs[cluster_key].to_numpy()
            cell_types_raw = np.unique(cell_labels_raw)
            cell_labels = np.zeros((len(cell_labels_raw)))
            for j in range(len(cell_types_raw)):
                cell_labels[cell_labels_raw==cell_types_raw[j]] = j
            
            if(method=='scVelo'):
                t_i, Uhat_i, Shat_i = get_pred_scv_demo(adata, keys[i], genes, nplot)
                Yhat[method] = np.concatenate((np.zeros((nplot)), np.ones((nplot))))
                V[method] = adata.layers["velocity"][:,gene_indices]
            elif(method=='Vanilla VAE'):
                t_i, Uhat_i, Shat_i = get_pred_vanilla_demo(adata, keys[i], genes, nplot)
                Yhat[method] = None
                V[method] = adata.layers[f"{keys[i]}_velocity"][:,gene_indices]
            elif(method in ['VeloVAE','FullVB','Discrete VeloVAE','Discrete FullVB']):
                Uhat_i, Shat_i = get_pred_velovae_demo(adata, keys[i], genes, 'FullVB' in method, 'Discrete' in method)
                V[method] = adata.layers[f"{keys[i]}_velocity"][:,gene_indices]
                t_i = adata.obs[f'{keys[i]}_time'].to_numpy()
                Yhat[method] = cell_labels
            elif(method=='BrODE'):
                t_i, y_brode, Uhat_i, Shat_i = get_pred_brode_demo(adata, keys[i], genes)
                V[method] = adata.layers[f"{keys[i]}_velocity"][:,gene_indices]
                Yhat[method] = y_brode
            elif(method=="UniTVelo"):
                t_i, Uhat_i, Shat_i = get_pred_utv_demo(adata, genes, nplot)
                V[method] = adata.layers["velocity"][:,gene_indices]
                Yhat[method] = None
            elif(method=="DeepVelo"):
                t_i = adata.obs[f'{keys[i]}_time'].to_numpy()
                V[method] = adata.layers["velocity"][:,gene_indices]
                Uhat_i = adata.layers["Mu"][:,gene_indices]+adata.layers["velocity_unspliced"][:,gene_indices]
                Shat_i = adata.layers["Mu"][:,gene_indices]+V[method]
                Yhat[method] = None
            elif(method in ["PyroVelocity","Continuous PyroVelocity"]):
                t_i = adata.obs[f'{keys[i]}_time'].to_numpy()
                Uhat_i, Shat_i = adata.layers[f'{keys[i]}_u'][:,gene_indices], adata.layers[f'{keys[i]}_s'][:,gene_indices]
                V[method] = adata.layers[f"{keys[i]}_velocity"][:,gene_indices]
                Yhat[method] = cell_labels
            elif(method=="VeloVI"):
                t_i = adata.layers['fit_t'][:,gene_indices]
                Uhat_i, Shat_i = adata.layers[f'{keys[i]}_uhat'][:,gene_indices], adata.layers[f'{keys[i]}_shat'][:,gene_indices]
                V[method] = adata.layers[f"{keys[i]}_velocity"][:,gene_indices]
                Yhat[method] = cell_labels
            
            That[method] = t_i
            t_brode = t_i
            Uhat[method] = Uhat_i
            Shat[method] = Shat_i

    if(compute_metrics):
        print("---     Computing Peformance Metrics     ---")
        print(f"Dataset Size: {adata.n_obs} cells, {adata.n_vars} genes")
        stats_df = pd.DataFrame(stats)
        pd.set_option("display.precision", 3)
    
    print("---   Plotting  Results   ---")
    if('cluster' in plot_type or "all" in plot_type):
        plot_cluster(adata.obsm[f"X_{embed}"], adata.obs[cluster_key].to_numpy(), embed=embed, save=f"{save_path}/{test_id}_umap.png")
    
    #Generate plots
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
        Labels_phase_demo = {}
        for i, method in enumerate(methods):
            Labels_phase[method] = cell_state(adata, method, keys[i], gene_indices)
            Legends_phase[method] = ['Induction', 'Repression', 'Off', 'Unknown']
            Labels_phase_demo[method] = None
        plot_phase_grid(grid_size[0], 
                        grid_size[1],
                        genes,
                        U[:,gene_indices], 
                        S[:,gene_indices],
                        Labels_phase,
                        Legends_phase,
                        Uhat, 
                        Shat,
                        Labels_phase_demo,
                        path=save_path,
                        figname=test_id,
                        format=format)
    
    if('signal' in plot_type or 'all' in plot_type):
        T = {}
        Labels_sig = {}
        Legends_sig = {}
        for i, method in enumerate(methods):
            Labels_sig[method] = np.array([label_dic[x] for x in adata.obs[cluster_key].to_numpy()])
            Legends_sig[method] = cell_types_raw
            if(method=='scVelo'):
                methods_ = np.concatenate((methods,['scVelo Global']))
                T[method] = adata.layers[f"{keys[i]}_t"][:,gene_indices]
                T['scVelo Global'] = adata.obs['latent_time'].to_numpy()*20
                Labels_sig['scVelo Global'] = Labels_sig[method]
                Legends_sig['scVelo Global'] = cell_types_raw
            elif(method=='VeloVI'):
                T[method] = adata.layers[f"fit_t"][:,gene_indices]
            else:
                T[method] = adata.obs[f"{keys[i]}_time"].to_numpy()
        
        sparsity_correction = kwargs['sparsity_correction'] if 'sparsity_correction' in kwargs else False 
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
    
    if('stream' in plot_type or 'all' in plot_type):
        try:
            from scvelo.tl import velocity_graph
            from scvelo.pl import velocity_embedding_stream
            colors = get_colors(len(cell_types_raw))
            for i, vkey in enumerate(vkeys):
                if(not f"{vkey}_graph" in adata.uns):
                    velocity_graph(adata, vkey=vkey)
                velocity_embedding_stream(adata, 
                                          basis=embed,
                                          vkey=vkey,
                                          title="", 
                                          palette=colors,
                                          legend_fontsize=15,
                                          dpi=150,
                                          show=False,
                                          save=f'{save_path}/{test_id}_{keys[i]}_stream.png')
        except ImportError:
            print('Please install scVelo in order to generate stream plots')
            pass
    
    if(compute_metrics):
        stats_df.to_csv(f"{save_path}/metrics_{test_id}.csv",sep='\t')
        return stats_df
    return

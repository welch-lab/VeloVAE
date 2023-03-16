import numpy as np
import pandas as pd
from ..model.model_util import make_dir
from .evaluation_util import *
from velovae.plotting import get_colors, plot_cluster, plot_phase_grid, plot_sig_grid, plot_time_grid
from multiprocessing import cpu_count
from scipy.stats import spearmanr


def get_n_cpu(n_cell):
    # used for scVelo parallel jobs
    return int(min(cpu_count(), max(1, n_cell/3000)))


def get_metric(adata,
               method,
               key,
               vkey,
               cluster_key="clusters",
               gene_key='velocity_genes',
               cluster_edges=[],
               embed='umap',
               n_jobs=None):
    """Get specific metrics given a method.

    Arguments
    ---------
    adata : :class:`anndata.AnnData`
    key : str
       Key in .var or .varm for extracting the ODE parameters learned by the model
    vkey : str
        Key in .layers for extracting rna velocity
    cluster_key : str
        Key in .obs for extracting cell type annotation
    gene_key : str, optional
       Key for filtering the genes.
    cluster_edges : str, optional
        List of tuples. Each tuple contains the progenitor cell type and its descendant cell type.
    embed : str, optional
        Low-dimensional embedding name.
    n_jobs : int, optional
        Number of parallel jobs. Used in scVelo velocity graph computation.
    Returns
    -------
    stats : :class:`pandas.DataFrame`
        Stores the performance metrics. Rows are metric names and columns are method names
    """

    stats = {}  # contains the performance metrics
    gene_mask = adata.var[gene_key].to_numpy() if gene_key in adata.var else None

    if method == 'scVelo':
        (mse_train, mse_test,
         mae_train, mae_test,
         logp_train, logp_test,
         run_time) = get_err_scv(adata)
    elif method == 'Vanilla VAE':
        (mse_train, mse_test,
         mae_train, mae_test,
         logp_train, logp_test,
         run_time) = get_err_vanilla(adata, key, gene_mask)
    elif method == 'Cycle VAE':
        (mse_train, mse_test,
         mae_train, mae_test,
         logp_train, logp_test,
         run_time) = get_err_cycle(adata, key, gene_mask)
    elif method == 'VeloVAE' or method == 'FullVB':
        (mse_train, mse_test,
         mae_train, mae_test,
         logp_train, logp_test,
         run_time) = get_err_velovae(adata, key, gene_mask, 'FullVB' in method)
    elif method == 'BrODE':
        (mse_train, mse_test,
         mae_train, mae_test,
         logp_train, logp_test,
         run_time) = get_err_brode(adata, key, gene_mask)
    elif method == 'Discrete VeloVAE' or method == 'Discrete FullVB':
        (mse_train, mse_test,
         mae_train, mae_test,
         logp_train, logp_test,
         run_time) = get_err_velovae(adata, key, gene_mask, 'FullVB' in method, True)
    elif method == 'UniTVelo':
        (mse_train, mse_test,
         mae_train, mae_test,
         logp_train, logp_test,
         run_time) = get_err_utv(adata, key, gene_mask)
    elif method == 'DeepVelo':
        (mse_train, mse_test,
         mae_train, mae_test,
         logp_train, logp_test,
         run_time) = get_err_dv(adata, key, gene_mask)
    elif 'PyroVelocity' in method:
        if 'err' in adata.uns:
            mse_train, mse_test = adata.uns['err']['MSE Train'], adata.uns['err']['MSE Test']
            mae_train, mae_test = adata.uns['err']['MAE Train'], adata.uns['err']['MAE Test']
            logp_train, logp_test = adata.uns['err']['LL Train'], adata.uns['err']['LL Test']
            run_time = adata.uns[f'{key}_run_time'] if f'{key}_run_time' in adata.uns else np.nan
        else:
            (mse_train, mse_test,
             mae_train, mae_test,
             logp_train, logp_test,
             run_time) = get_err_pv(adata, key, gene_mask, 'Continuous' not in method)
    elif method == 'VeloVI':
        (mse_train, mse_test,
         mae_train, mae_test,
         logp_train, logp_test,
         run_time) = get_err_velovi(adata, key, gene_mask)

    if method in ['scVelo', 'UniTVelo', 'DeepVelo']:
        logp_test = 'N/A'
        mse_test = 'N/A'
        logp_test = 'N/A'
        mae_test = 'N/A'

    if method == 'DeepVelo':
        logp_train = 'N/A'

    stats['MSE Train'] = mse_train
    stats['MSE Test'] = mse_test
    stats['MAE Train'] = mae_train
    stats['MAE Test'] = mae_test
    stats['LL Train'] = logp_train
    stats['LL Test'] = logp_test
    stats['Training Time'] = run_time

    if 'tprior' in adata.obs:
        if method == 'DeepVelo':
            stats['corr'] = 'N/A'
        else:
            tprior = adata.obs['tprior'].to_numpy()
            t = (adata.obs["latent_time"].to_numpy()
                 if (method in ['scVelo', 'UniTVelo']) else
                 adata.obs[f"{key}_time"].to_numpy())
            corr, pval = spearmanr(t, tprior)
            stats['corr'] = corr

    print("Computing velocity embedding using scVelo")
    try:
        from scvelo.tl import velocity_graph, velocity_embedding
        n_jobs = get_n_cpu(adata.n_obs) if n_jobs is None else n_jobs
        velocity_graph(adata, vkey=vkey, n_jobs=n_jobs)
        velocity_embedding(adata, vkey=vkey, basis=embed)
    except ImportError:
        print("Please install scVelo to compute velocity embedding.\n"
              "Skipping metrics 'Cross-Boundary Direction Correctness' and 'In-Cluster Coherence'.")

    # Filter extreme values
    v_embed = adata.obsm[f'{vkey}_{embed}']
    idx_extreme = np.where(np.any(np.isnan(v_embed), 1))[0]
    if len(idx_extreme) > 0:
        v_filt = np.stack([np.nanmean(v_embed[:, 0][adata.uns['neighbors']['indices'][idx_extreme]], 1),
                           np.nanmean(v_embed[:, 1][adata.uns['neighbors']['indices'][idx_extreme]], 1)]).T
        if np.any(np.isnan(v_filt)) or np.any(np.isinf(v_filt)):
            v_filt[np.where(np.isnan(v_filt) or np.isinf(v_filt))[0]] = 0
        v_embed[idx_extreme] = v_filt

    # Compute velocity metrics
    if len(cluster_edges) > 0:
        _, mean_cbdir = cross_boundary_correctness(adata,
                                                   cluster_key,
                                                   vkey,
                                                   cluster_edges,
                                                   x_emb=f"X_{embed}")
        stats['Cross-Boundary Direction Correctness (embed)'] = mean_cbdir
        _, mean_cbdir = cross_boundary_correctness(adata,
                                                   cluster_key,
                                                   vkey,
                                                   cluster_edges,
                                                   x_emb="Ms",
                                                   gene_mask=gene_mask)
        stats['Cross-Boundary Direction Correctness'] = mean_cbdir
    _, mean_iccoh = inner_cluster_coh(adata, cluster_key, vkey)
    stats['In-Cluster Coherence'] = mean_iccoh

    return stats


def post_analysis(adata,
                  test_id,
                  methods,
                  keys,
                  gene_key='velocity_genes',
                  compute_metrics=True,
                  raw_count=False,
                  genes=[],
                  plot_type=[],
                  cluster_key="clusters",
                  cluster_edges=[],
                  nplot=500,
                  frac=0.0,
                  embed="umap",
                  grid_size=(1, 1),
                  save_path="figures",
                  **kwargs):
    """Main function for post analysis.
    This function computes performance metrics and generates plots based on user input.

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
    gene_key : string, optional
        Key in .var for gene filtering. Usually set to select velocity genes.
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
    # Retrieve data
    if raw_count:
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

    # Get gene indices
    if len(genes) > 0:
        gene_indices = []
        gene_rm = []
        for gene in genes:
            idx = np.where(adata.var_names == gene)[0]
            if len(idx) > 0:
                gene_indices.append(idx[0])
            else:
                print(f"Warning: gene name {gene} not found in AnnData. Removed.")
                gene_rm.append(gene)
        for gene in gene_rm:
            genes.remove(gene)

        if len(gene_indices) == 0:
            print("Warning: No gene names found. Randomly select genes...")
            gene_indices = np.random.choice(adata.n_vars, grid_size[0]*grid_size[1], replace=False).astype(int)
            genes = adata.var_names[gene_indices].to_numpy()
    else:
        print("Warning: No gene names are provided. Randomly select genes...")
        gene_indices = np.random.choice(adata.n_vars, grid_size[0]*grid_size[1], replace=False).astype(int)
        genes = adata.var_names[gene_indices].to_numpy()
        print(genes)

    stats = {}
    Uhat, Shat, V = {}, {}, {}
    That, Yhat = {}, {}
    vkeys = []
    for i, method in enumerate(methods):
        vkey = 'velocity' if method in ['scVelo', 'UniTVelo', 'DeepVelo'] else f'{keys[i]}_velocity'
        vkeys.append(vkey)

    # Compute metrics and generate plots for each method
    for i, method in enumerate(methods):
        if compute_metrics:
            stats_i = get_metric(adata,
                                 method,
                                 keys[i],
                                 vkeys[i],
                                 cluster_key,
                                 gene_key,
                                 cluster_edges,
                                 embed,
                                 n_jobs=kwargs['n_jobs'] if 'n_jobs' in kwargs else None)
            # avoid duplicate methods with different keys
            method_ = f"{method} ({keys[i]})" if method in stats else method
            stats[method_] = stats_i
        # Compute prediction for the purpose of plotting (a fixed number of plots)
        if 'phase' in plot_type or 'signal' in plot_type or 'all' in plot_type:
            # avoid duplicate methods with different keys
            method_ = f"{method} ({keys[i]})" if method in V else method
            # Integer-encoded cell type
            cell_labels_raw = adata.obs[cluster_key].to_numpy()
            cell_types_raw = np.unique(cell_labels_raw)
            cell_labels = np.zeros((len(cell_labels_raw)))
            for j in range(len(cell_types_raw)):
                cell_labels[cell_labels_raw == cell_types_raw[j]] = j

            if method == 'scVelo':
                t_i, Uhat_i, Shat_i = get_pred_scv_demo(adata, keys[i], genes, nplot)
                Yhat[method_] = np.concatenate((np.zeros((nplot)), np.ones((nplot))))
                V[method_] = adata.layers["velocity"][:, gene_indices]
            elif method == 'Vanilla VAE':
                t_i, Uhat_i, Shat_i = get_pred_vanilla_demo(adata, keys[i], genes, nplot)
                Yhat[method_] = None
                V[method_] = adata.layers[f"{keys[i]}_velocity"][:, gene_indices]
            elif method == 'Cycle VAE':
                t_i, Uhat_i, Shat_i = get_pred_cycle_demo(adata, keys[i], genes, nplot)
                Yhat[method_] = None
                V[method_] = adata.layers[f"{keys[i]}_velocity"][:, gene_indices]
            elif method in ['VeloVAE', 'FullVB', 'Discrete VeloVAE', 'Discrete FullVB']:
                Uhat_i, Shat_i = get_pred_velovae_demo(adata, keys[i], genes, 'FullVB' in method, 'Discrete' in method)
                V[method_] = adata.layers[f"{keys[i]}_velocity"][:, gene_indices]
                t_i = adata.obs[f'{keys[i]}_time'].to_numpy()
                Yhat[method_] = cell_labels
            elif method == 'BrODE':
                t_i, y_brode, Uhat_i, Shat_i = get_pred_brode_demo(adata, keys[i], genes)
                V[method_] = adata.layers[f"{keys[i]}_velocity"][:, gene_indices]
                Yhat[method_] = y_brode
            elif method == "UniTVelo":
                t_i, Uhat_i, Shat_i = get_pred_utv_demo(adata, genes, nplot)
                V[method_] = adata.layers["velocity"][:, gene_indices]
                Yhat[method_] = None
            elif method == "DeepVelo":
                t_i = adata.obs[f'{keys[i]}_time'].to_numpy()
                V[method_] = adata.layers["velocity"][:, gene_indices]
                Uhat_i = adata.layers["Mu"][:, gene_indices]+adata.layers["velocity_unspliced"][:, gene_indices]
                Shat_i = adata.layers["Mu"][:, gene_indices]+V[method]
                Yhat[method_] = None
            elif method in ["PyroVelocity", "Continuous PyroVelocity"]:
                t_i = adata.obs[f'{keys[i]}_time'].to_numpy()
                Uhat_i = adata.layers[f'{keys[i]}_u'][:, gene_indices]
                Shat_i = adata.layers[f'{keys[i]}_s'][:, gene_indices]
                V[method_] = adata.layers[f"{keys[i]}_velocity"][:, gene_indices]
                Yhat[method_] = cell_labels
            elif method == "VeloVI":
                t_i = adata.layers['fit_t'][:, gene_indices]
                Uhat_i = adata.layers[f'{keys[i]}_uhat'][:, gene_indices]
                Shat_i = adata.layers[f'{keys[i]}_shat'][:, gene_indices]
                V[method_] = adata.layers[f"{keys[i]}_velocity"][:, gene_indices]
                Yhat[method_] = cell_labels

            That[method_] = t_i
            Uhat[method_] = Uhat_i
            Shat[method_] = Shat_i

    if compute_metrics:
        print("---     Computing Peformance Metrics     ---")
        print(f"Dataset Size: {adata.n_obs} cells, {adata.n_vars} genes")
        stats_df = pd.DataFrame(stats)
        pd.set_option("display.precision", 3)

    print("---   Plotting  Results   ---")
    if 'cluster' in plot_type or "all" in plot_type:
        plot_cluster(adata.obsm[f"X_{embed}"],
                     adata.obs[cluster_key].to_numpy(),
                     embed=embed,
                     save=f"{save_path}/{test_id}_umap.png")

    # Generate plots
    if "time" in plot_type or "all" in plot_type:
        T = {}
        capture_time = adata.obs["tprior"].to_numpy() if "tprior" in adata.obs else None
        for i, method in enumerate(methods):
            # avoid duplicate methods with different keys
            method_ = f"{method} ({keys[i]})" if method in T else method
            if method == 'scVelo':
                T[method_] = adata.obs["latent_time"].to_numpy()
            else:
                T[method_] = adata.obs[f"{keys[i]}_time"].to_numpy()
        plot_time_grid(T,
                       X_embed,
                       capture_time,
                       None,
                       down_sample=min(10, max(1, adata.n_obs//5000)),
                       save=f"{save_path}/{test_id}_time.png")

    if len(genes) == 0:
        return

    format = kwargs["format"] if "format" in kwargs else "png"
    if "phase" in plot_type or "all" in plot_type:
        Labels_phase = {}
        Legends_phase = {}
        Labels_phase_demo = {}
        for i, method in enumerate(methods):
            # avoid duplicate methods with different keys
            method_ = f"{method} ({keys[i]})" if method in Labels_phase else method
            Labels_phase[method_] = cell_state(adata, method, keys[i], gene_indices)
            Legends_phase[method_] = ['Induction', 'Repression', 'Off', 'Unknown']
            Labels_phase_demo[method] = None
        plot_phase_grid(grid_size[0],
                        grid_size[1],
                        genes,
                        U[:, gene_indices],
                        S[:, gene_indices],
                        Labels_phase,
                        Legends_phase,
                        Uhat,
                        Shat,
                        Labels_phase_demo,
                        path=save_path,
                        figname=test_id,
                        format=format)

    if 'signal' in plot_type or 'all' in plot_type:
        T = {}
        Labels_sig = {}
        Legends_sig = {}
        for i, method in enumerate(methods):
            # avoid duplicate methods with different keys
            method_ = f"{method} ({keys[i]})" if method in Labels_sig else method
            Labels_sig[method_] = np.array([label_dic[x] for x in adata.obs[cluster_key].to_numpy()])
            Legends_sig[method_] = cell_types_raw
            if method == 'scVelo':
                T[method_] = adata.layers[f"{keys[i]}_t"][:, gene_indices]
                T['scVelo Global'] = adata.obs['latent_time'].to_numpy()*20
                Labels_sig['scVelo Global'] = Labels_sig[method]
                Legends_sig['scVelo Global'] = cell_types_raw
            elif method == 'UniTVelo':
                T[method_] = adata.layers["fit_t"][:, gene_indices]
            elif method == 'VeloVI':
                T[method_] = adata.layers["fit_t"][:, gene_indices]
            else:
                T[method_] = adata.obs[f"{keys[i]}_time"].to_numpy()

        sparsity_correction = kwargs['sparsity_correction'] if 'sparsity_correction' in kwargs else False
        plot_sig_grid(grid_size[0],
                      grid_size[1],
                      genes,
                      T,
                      U[:, gene_indices],
                      S[:, gene_indices],
                      Labels_sig,
                      Legends_sig,
                      That,
                      Uhat,
                      Shat,
                      V,
                      Yhat,
                      frac=frac,
                      down_sample=min(20, max(1, adata.n_obs//2500)),
                      sparsity_correction=sparsity_correction,
                      path=save_path,
                      figname=test_id,
                      format=format)

    if 'stream' in plot_type or 'all' in plot_type:
        try:
            from scvelo.tl import velocity_graph
            from scvelo.pl import velocity_embedding_stream
            colors = get_colors(len(cell_types_raw))
            for i, vkey in enumerate(vkeys):
                if f"{vkey}_graph" not in adata.uns:
                    velocity_graph(adata, vkey=vkey, n_jobs=get_n_cpu(adata.n_obs))
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

    if compute_metrics:
        stats_df.to_csv(f"{save_path}/metrics_{test_id}.csv", sep='\t')
        return stats_df
    return

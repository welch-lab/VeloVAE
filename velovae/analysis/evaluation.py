"""Evaluation Module
Performs performance evaluation for various RNA velocity models and generates figures.
"""
import numpy as np
import pandas as pd
from os import makedirs
from .evaluation_util import *
from .evaluation_util import time_score
from velovae.plotting import get_colors, plot_cluster, plot_phase_grid, plot_sig_grid, plot_time_grid
from multiprocessing import cpu_count
from scipy.stats import spearmanr


def get_n_cpu(n_cell):
    # used for scVelo parallel jobs
    return int(min(cpu_count(), max(1, n_cell/2000)))


def get_velocity_metric_placeholder(cluster_edges):
    # Convert tuples to a single string
    cluster_edges_ = []
    for pair in cluster_edges:
        cluster_edges_.append(f'{pair[0]} -> {pair[1]}')
    cbdir_embed = dict.fromkeys(cluster_edges_)
    cbdir = dict.fromkeys(cluster_edges_)
    tscore = dict.fromkeys(cluster_edges_)
    iccoh = dict.fromkeys(cluster_edges_)
    return (iccoh, np.nan,
            cbdir_embed, np.nan,
            cbdir, np.nan,
            tscore, np.nan,
            np.nan)


def get_velocity_metric(adata,
                        key,
                        vkey,
                        cluster_key,
                        cluster_edges,
                        gene_mask=None,
                        embed='umap',
                        n_jobs=None):
    """
    Computes Cross-Boundary Direction Correctness and In-Cluster Coherence.
    The function calls scvelo.tl.velocity_graph.

    Args:
        adata (:class:`anndata.AnnData`):
            AnnData object.
        key (str):
            Key for cell time in the form of f'{key}_time'.
        vkey (str):
            Key for velocity in adata.obsm.
        cluster_key (str):
            Key for cell type annotations.
        cluster_edges (list[tuple[str]]):
            List of ground truth cell type transitions.
            Each transition is of the form (A, B) where A is a progenitor
            cell type and B is a descendant type.
        gene_mask (:class:`np.ndarray`, optional):
            Boolean array to filter out velocity genes. Defaults to None.
        embed (str, optional):
            Low-dimensional embedding. Defaults to 'umap'.
        n_jobs (_type_, optional):
            Number of parallel jobs. Defaults to None.

    Returns:
        tuple

            - dict: In-Cluster Coherence per cell type transition
            - float: Mean In-Cluster Coherence
            - dict: CBDir per cell type transition
            - float: Mean CBDir
            - dict: CBDir (embedding) per cell type transition
            - float: Mean CBDir (embedding)
            - dict: Time Accuracy Score per cell type transition
            - float: Mean Time Accuracy Score
            - float: Velocity Consistency
    """
    mean_constcy_score = velocity_consistency(adata, vkey, gene_mask)
    if cluster_edges is not None:
        try:
            from scvelo.tl import velocity_graph, velocity_embedding
            n_jobs = get_n_cpu(adata.n_obs) if n_jobs is None else n_jobs
            gene_subset = adata.var_names if gene_mask is None else adata.var_names[gene_mask]
            velocity_graph(adata, vkey=vkey, gene_subset=gene_subset, n_jobs=n_jobs)
            velocity_embedding(adata, vkey=vkey, basis=embed)
        except ImportError:
            print("Please install scVelo to compute velocity embedding.\n"
                  "Skipping metrics 'Cross-Boundary Direction Correctness' and 'In-Cluster Coherence'.")
        iccoh, mean_iccoh = inner_cluster_coh(adata, cluster_key, vkey, gene_mask)
        cbdir_embed, mean_cbdir_embed = cross_boundary_correctness(adata,
                                                                   cluster_key,
                                                                   vkey,
                                                                   cluster_edges,
                                                                   x_emb=f"X_{embed}")
        cbdir, mean_cbdir = cross_boundary_correctness(adata,
                                                       cluster_key,
                                                       vkey,
                                                       cluster_edges,
                                                       x_emb="Ms",
                                                       gene_mask=gene_mask)
        tscore, mean_tscore = time_score(adata, f'{key}_time', cluster_key, cluster_edges)
    else:
        mean_cbdir_embed = np.nan
        mean_cbdir = np.nan
        mean_tscore = np.nan
        mean_iccoh = np.nan
        cbdir_embed = dict.fromkeys([])
        cbdir = dict.fromkeys([])
        tscore = dict.fromkeys([])
        iccoh = dict.fromkeys([])
    return (iccoh, mean_iccoh,
            cbdir_embed, mean_cbdir_embed,
            cbdir, mean_cbdir,
            tscore, mean_tscore,
            mean_constcy_score)


def get_metric(adata,
               method,
               key,
               vkey,
               cluster_key="clusters",
               gene_key='velocity_genes',
               cluster_edges=None,
               embed='umap',
               n_jobs=None):
    """
    Get performance metrics given a method.

    Args:
        adata (:class:`anndata.AnnData`):
            AnnData object.
        method (str):
            Model name. The velovae package also provides evaluation for other RNA velocity methods.
        key (str):
            Key in .var or .varm for extracting the ODE parameters learned by the model.
        vkey (str):
            Key in .layers for extracting rna velocity.
        cluster_key (str, optional):
            Key in .obs for extracting cell type annotation. Defaults to "clusters".
        gene_key (str, optional):
            Key for filtering the genes.. Defaults to 'velocity_genes'.
        cluster_edges (list[tuple[str]], optional):
            List of ground truth cell type transitions.
            Each transition is of the form (A, B) where A is a progenitor
            cell type and B is a descendant type.
            Defaults to None.
        embed (str, optional):
            Low-dimensional embedding name.. Defaults to 'umap'.
        n_jobs (int, optional):
            Number of parallel jobs. Used in scVelo velocity graph computation.
            By default, it is automatically determined based on dataset size.
            Defaults to None.

    Returns:
        stats (:class:`pandas.DataFrame`):
            Stores the performance metrics. Rows are metric names and columns are method names
    """
    stats = {
        'MSE Train': np.nan,
        'MSE Test': np.nan,
        'MAE Train': np.nan,
        'MAE Test': np.nan,
        'LL Train': np.nan,
        'LL Test': np.nan,
        'Training Time': np.nan
    }  # contains the performance metrics
    if gene_key is not None and gene_key in adata.var:
        gene_mask = adata.var[gene_key].to_numpy()
    else:
        gene_mask = None

    if method == 'scVelo':
        (mse_train, mse_test,
         mae_train, mae_test,
         logp_train, logp_test) = get_err_scv(adata)
    elif method == 'Vanilla VAE':
        (mse_train, mse_test,
         mae_train, mae_test,
         logp_train, logp_test) = get_err_vanilla(adata, key, gene_mask)
    elif method == 'Cycle VAE':
        (mse_train, mse_test,
         mae_train, mae_test,
         logp_train, logp_test) = get_err_cycle(adata, key, gene_mask)
    elif method == 'VeloVAE' or method == 'FullVB':
        (mse_train, mse_test,
         mae_train, mae_test,
         logp_train, logp_test) = get_err_velovae(adata, key, gene_mask, 'FullVB' in method)
    elif method == 'BrODE':
        (mse_train, mse_test,
         mae_train, mae_test,
         logp_train, logp_test) = get_err_brode(adata, key, gene_mask)
    elif method == 'Discrete VeloVAE' or method == 'Discrete FullVB':
        (mse_train, mse_test,
         mae_train, mae_test,
         logp_train, logp_test) = get_err_velovae(adata, key, gene_mask, 'FullVB' in method, True)
    elif method == 'UniTVelo':
        (mse_train, mse_test,
         mae_train, mae_test,
         logp_train, logp_test) = get_err_utv(adata, key, gene_mask)
    elif method == 'DeepVelo':
        (mse_train, mse_test,
         mae_train, mae_test,
         logp_train, logp_test) = get_err_dv(adata, key, gene_mask)
    elif 'PyroVelocity' in method:
        if 'err' in adata.uns:
            mse_train, mse_test = adata.uns['err']['MSE Train'], adata.uns['err']['MSE Test']
            mae_train, mae_test = adata.uns['err']['MAE Train'], adata.uns['err']['MAE Test']
            logp_train, logp_test = adata.uns['err']['LL Train'], adata.uns['err']['LL Test']
        else:
            (mse_train, mse_test,
             mae_train, mae_test,
             logp_train, logp_test) = get_err_pv(adata, key, gene_mask, 'Continuous' not in method)
    elif method == 'VeloVI':
        (mse_train, mse_test,
         mae_train, mae_test,
         logp_train, logp_test) = get_err_velovi(adata, key, gene_mask)
    else:
        mse_train, mse_test = np.nan, np.nan
        mae_train, mae_test = np.nan, np.nan
        logp_train, logp_test = np.nan, np.nan

    stats['MSE Train'] = mse_train
    stats['MSE Test'] = mse_test
    stats['MAE Train'] = mae_train
    stats['MAE Test'] = mae_test
    stats['LL Train'] = logp_train
    stats['LL Test'] = logp_test

    if 'tprior' in adata.obs:
        tprior = adata.obs['tprior'].to_numpy()
        t = (adata.obs["latent_time"].to_numpy()
             if (method in ['scVelo', 'UniTVelo']) else
             adata.obs[f"{key}_time"].to_numpy())
        corr, pval = spearmanr(t, tprior)
        stats['corr'] = corr
    else:
        stats['corr'] = np.nan

    print("Computing velocity embedding using scVelo")
    # Compute velocity metrics using a subset of genes defined by gene_mask
    if gene_mask is not None:
        (iccoh_sub, mean_iccoh_sub,
         cbdir_sub_embed, mean_cbdir_sub_embed,
         cbdir_sub, mean_cbdir_sub,
         tscore_sub, mean_tscore_sub,
         mean_vel_consistency_sub) = get_velocity_metric(adata,
                                                         key,
                                                         vkey,
                                                         cluster_key,
                                                         cluster_edges,
                                                         gene_mask,
                                                         embed,
                                                         n_jobs)
    else:
        (iccoh_sub, mean_iccoh_sub,
         cbdir_sub_embed, mean_cbdir_sub_embed,
         cbdir_sub, mean_cbdir_sub,
         tscore_sub, mean_tscore_sub,
         mean_vel_consistency_sub) = get_velocity_metric_placeholder(cluster_edges)
    stats['CBDir (Embed, Velocity Genes)'] = mean_cbdir_sub_embed
    stats['CBDir (Velocity Genes)'] = mean_cbdir_sub
    stats['In-Cluster Coherence (Velocity Genes)'] = mean_iccoh_sub
    stats['Vel Consistency (Velocity Genes)'] = mean_vel_consistency_sub

    # Compute velocity metrics on all genes
    (iccoh, mean_iccoh,
     cbdir_embed, mean_cbdir_embed,
     cbdir, mean_cbdir,
     tscore, mean_tscore,
     mean_vel_consistency) = get_velocity_metric(adata,
                                                 key,
                                                 vkey,
                                                 cluster_key,
                                                 cluster_edges,
                                                 None,
                                                 embed,
                                                 n_jobs)
    stats['CBDir (Embed)'] = mean_cbdir_embed
    stats['CBDir'] = mean_cbdir
    stats['Time Score'] = mean_tscore
    stats['In-Cluster Coherence'] = mean_iccoh
    stats['Vel Consistency'] = mean_vel_consistency
    stats_type = pd.concat([pd.DataFrame.from_dict(cbdir_sub, orient='index'),
                            pd.DataFrame.from_dict(cbdir_sub_embed, orient='index'),
                            pd.DataFrame.from_dict(cbdir, orient='index'),
                            pd.DataFrame.from_dict(cbdir_embed, orient='index'),
                            pd.DataFrame.from_dict(tscore, orient='index')],
                           axis=1).T
    stats_type.index = pd.Index(['CBDir (Velocity Genes)',
                                 'CBDir (Embed, Velocity Genes)',
                                 'CBDir',
                                 'CBDir (Embed)',
                                 'Time Score'])
    return stats, stats_type


def post_analysis(adata,
                  test_id,
                  methods,
                  keys,
                  gene_key='velocity_genes',
                  compute_metrics=True,
                  raw_count=False,
                  genes=[],
                  plot_type=['time', 'gene', 'stream'],
                  cluster_key="clusters",
                  cluster_edges=[],
                  nplot=500,
                  frac=0.0,
                  embed="umap",
                  grid_size=(1, 1),
                  sparsity_correction=True,
                  figure_path=None,
                  save=None,
                  **kwargs):
    """High-level API for method evaluation and plotting after training.
    This function computes performance metrics and generates plots based on user input.

    Args:
        adata (:class:`anndata.AnnData`):
            AnnData object.
        test_id (str):
            Used for naming the figures.
            For example, it can be set as the name of the dataset.
        methods (list[str]):
            Contains the methods to compare with.
            Now supports {'scVelo', 'UniTVelo', 'DeepVelo', 'cellDancer', 'VeloVI', 'PyroVelocity',
            'VeloVAE', 'FullVB', 'Discrete VeloVAE', 'Discrete FullVB', 'BrODE'}.
        keys (list[str]):
            Used for extracting ODE parameters from .var or .varm from anndata
            It should be of the same length as methods.
        gene_key (str, optional):
            Key in .var for gene filtering. Usually set to select velocity genes.
            Defaults to 'velocity_genes'.
        compute_metrics (bool, optional):
            Whether to compute the performance metrics for the methods. Defaults to True.
        raw_count (bool, optional):
            Whether to plot raw count numbers for discrete models. Defaults to False.
        genes (list[str], optional):
            Genes to plot. Used when plot_type contains "phase" or "gene".
            If not provided, gene(s) will be randomly sampled for plotting. Defaults to [].
        plot_type (list, optional):
            Type of plots to generate.
            Now supports {'time', 'gene', 'stream', 'phase', 'cluster'}.
            Defaults to ['time', 'gene', 'stream'].
        cluster_key (str, optional):
            Key in .obs containing the cell type annotations. Defaults to "clusters".
        cluster_edges (list[str], optional):
            List of ground-truth cell type ancestor-descendant relations, e.g. (A, B)
            means cell type A is the ancestor of type B. This is used for computing
            velocity metrics. Defaults to [].
        nplot (int, optional):
            Number of data points in the line prediction.
            This is to save memory. For plotting line predictions, we don't need
            as many points as the original dataset contains. Defaults to 500.
        frac (float, optional):
            Parameter for the loess plot.
            A higher value means larger time window and the resulting fitted line will
            be smoother. Disabled if set to 0.
            Defaults to 0.0.
        embed (str, optional):
            2D embedding used for visualization of time and cell type.
            The true key for the embedding is f"X_{embed}" in .obsm.
            Defaults to "umap".
        grid_size (tuple[int], optional):
            Grid size for plotting the genes.
            n_row * n_col >= len(genes). Defaults to (1, 1).
        sparsity_correction (bool, optional):
            Whether to sample cells non-uniformly across time and count values so
            that regions with sparser data point distributions will not be missed
            in gene plots due to sampling. Default to True.
        figure_path (str, optional):
            Path to save the figures.. Defaults to None.
        save (str, optional):
            Path + output file name to save the AnnData object to a .h5ad file.
            Defaults to None.

    kwargs:
        random_state (int):
            Random number seed. Default to 42.
        n_jobs (int):
            Number of CPU cores used for parallel computing in scvelo.tl.velocity_graph.
        format (str):
            Figure format. Default to 'png'.


    Returns:
        tuple

            - :class:`pandas.DataFrame`: Contains the dataset-wise performance metrics of all methods.
            - :class:`pandas.DataFrame`: Contains the performance metrics of each pair of ancestor and desendant cell types.

        Saves the figures to 'figure_path'.

        Notice that the two output dataframes will be None if 'compute_metrics' is set to False.
    """
    # set the random seed
    random_state = 42 if not 'random_state' in kwargs else kwargs['random_state']
    np.random.seed(random_state)

    if figure_path is not None:
        makedirs(figure_path, exist_ok=True)
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
    stats_type_list = []
    Uhat, Shat, V = {}, {}, {}
    That, Yhat = {}, {}
    vkeys = []
    for i, method in enumerate(methods):
        vkey = 'velocity' if method in ['scVelo', 'UniTVelo', 'DeepVelo'] else f'{keys[i]}_velocity'
        vkeys.append(vkey)

    # Compute metrics and generate plots for each method
    for i, method in enumerate(methods):
        if compute_metrics:
            stats_i, stats_type_i = get_metric(adata,
                                               method,
                                               keys[i],
                                               vkeys[i],
                                               cluster_key,
                                               gene_key,
                                               cluster_edges,
                                               embed,
                                               n_jobs=(kwargs['n_jobs']
                                                       if 'n_jobs' in kwargs
                                                       else None))
            stats_type_list.append(stats_type_i)
            # avoid duplicate methods with different keys
            method_ = f"{method} ({keys[i]})" if method in stats else method
            stats[method_] = stats_i
        # Compute prediction for the purpose of plotting (a fixed number of plots)
        if 'phase' in plot_type or 'gene' in plot_type or 'all' in plot_type:
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
                t_i, y_i, Uhat_i, Shat_i = get_pred_brode_demo(adata, keys[i], genes, N=100)
                V[method_] = adata.layers[f"{keys[i]}_velocity"][:, gene_indices]
                #t_i = adata.obs[f'{keys[i]}_time'].to_numpy()
                #Yhat[method_] = cell_labels
                Yhat[method_] = y_i
            elif method == "UniTVelo":
                t_i, Uhat_i, Shat_i = get_pred_utv_demo(adata, genes, nplot)
                V[method_] = adata.layers["velocity"][:, gene_indices]
                Yhat[method_] = None
            elif method == "DeepVelo":
                t_i = adata.obs[f'{keys[i]}_time'].to_numpy()
                V[method_] = adata.layers["velocity"][:, gene_indices]
                Uhat_i = adata.layers["Mu"][:, gene_indices]
                Shat_i = adata.layers["Ms"][:, gene_indices]
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
            elif method == "cellDancer":
                t_i = adata.obs[f'{keys[i]}_time'].to_numpy()
                Uhat_i = adata.layers["Mu"][:, gene_indices]
                Shat_i = adata.layers["Ms"][:, gene_indices]
                V[method_] = adata.layers[f"{keys[i]}_velocity"][:, gene_indices]
                Yhat[method_] = cell_labels

            That[method_] = t_i
            Uhat[method_] = Uhat_i
            Shat[method_] = Shat_i

    if compute_metrics:
        print("---     Computing Peformance Metrics     ---")
        print(f"Dataset Size: {adata.n_obs} cells, {adata.n_vars} genes")
        stats_df = pd.DataFrame(stats)
        stats_type_df = pd.concat(stats_type_list,
                                  axis=1,
                                  keys=methods,
                                  names=['Model'])
        pd.set_option("display.precision", 3)

    print("---   Plotting  Results   ---")
    if 'cluster' in plot_type or "all" in plot_type:
        plot_cluster(adata.obsm[f"X_{embed}"],
                     adata.obs[cluster_key].to_numpy(),
                     embed=embed,
                     save=(None if figure_path is None else 
                           f"{figure_path}/{test_id}_umap.png"))

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
                       save=(None if figure_path is None else
                             f"{figure_path}/{test_id}_time.png"))

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
                        path=figure_path,
                        figname=test_id,
                        format=format)

    if 'gene' in plot_type or 'all' in plot_type:
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
                      down_sample=min(20, max(1, adata.n_obs//5000)),
                      sparsity_correction=sparsity_correction,
                      path=figure_path,
                      figname=test_id,
                      format=format)

    if 'stream' in plot_type or 'all' in plot_type:
        try:
            from scvelo.tl import velocity_graph
            from scvelo.pl import velocity_embedding_stream
            colors = get_colors(len(cell_types_raw))
            for i, vkey in enumerate(vkeys):
                if methods[i] in ['scVelo', 'UniTVelo', 'DeepVelo']:
                    gene_subset = adata.var_names[adata.var['velocity_genes'].to_numpy()]
                else:
                    gene_subset = adata.var_names[~np.isnan(adata.layers[vkey][0])]
                velocity_graph(adata, vkey=vkey, gene_subset=gene_subset, n_jobs=get_n_cpu(adata.n_obs))
                velocity_embedding_stream(adata,
                                          basis=embed,
                                          vkey=vkey,
                                          title="",
                                          palette=colors,
                                          legend_fontsize=np.clip(15 - np.clip(len(colors)-10, 0, None), 8, None),
                                          legend_loc='on data' if len(colors) <= 10 else 'right margin',
                                          dpi=300,
                                          show=True,
                                          save=(None if figure_path is None else
                                                f'{figure_path}/{test_id}_{keys[i]}_stream.png'))
        except ImportError:
            print('Please install scVelo in order to generate stream plots')
            pass
    if save is not None:
        adata.write_h5ad(save)
    if compute_metrics:
        if figure_path is not None:
            stats_df.to_csv(f"{figure_path}/metrics_{test_id}.csv", sep='\t')
        return stats_df, stats_type_df

    return None, None

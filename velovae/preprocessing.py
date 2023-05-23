import scanpy
import numpy as np
from .scvelo_preprocessing import *
from tqdm import tqdm


def count_peak_expression(adata, cluster_key="clusters"):
    """Figure out genes with the highest expression in each cell type.

    Args:
        adata (:class:`anndata.AnnData`):
            AnnData object.
        cluster_key (str, optional):
            Key for cell type annotation. Defaults to "clusters".

    Returns:
        dict: Number of genes with peak expression in each cell type.
        dict: Peak gene expression levels corresponding to each cell type.
        dict: Peak gene indices corresponding to each cell type.
    """
    # Count the number of genes with peak expression in each cell type.
    def encodeType(cell_types_raw):
        # Use integer to encode the cell types.
        # Each cell type has one unique integer label.
        # Map cell types to integers
        label_dic = {}
        label_dic_rev = {}
        for i, type_ in enumerate(cell_types_raw):
            label_dic[type_] = i
            label_dic_rev[i] = type_

        return label_dic, label_dic_rev
    cell_labels = adata.obs[cluster_key]
    cell_types = np.unique(cell_labels)
    label_dic, label_dic_rev = encodeType(cell_types)
    cell_labels = np.array([label_dic[x] for x in cell_labels])
    n_type = len(cell_types)

    X = np.array(adata.layers["spliced"].A+adata.layers["unspliced"].A)
    peak_expression = np.stack([np.quantile(X[cell_labels == j], 0.9, 0) for j in range(n_type)])
    peak_type = np.argmax(peak_expression, 0)  # cell type with the highest expression level for all genes
    peak_hist = np.array([np.sum(peak_type == i) for i in range(n_type)])  # gene count
    peak_val_hist = [peak_expression[:, peak_type == i][i] for i in range(n_type)]  # peak expression
    peak_gene = [np.where(peak_type == i)[0] for i in range(n_type)]  # peak gene index list

    out_peak_count = {}
    out_peak_expr = {}
    out_peak_gene = {}
    for i in range(n_type):
        out_peak_count[label_dic_rev[i]] = peak_hist[i]
        out_peak_expr[label_dic_rev[i]] = peak_val_hist[i]
        out_peak_gene[label_dic_rev[i]] = np.array(peak_gene[i])

    return out_peak_count, out_peak_expr, out_peak_gene


def balanced_gene_selection(adata, n_gene, cluster_key):
    """select the same number of genes for each cell type.

    Args:
        adata (:class:`anndata.AnnData`):
            AnnData object.
        n_gene (int):
            Number of genes to select.
        cluster_key (str):
            Key for cell type annotation.
    """
    if n_gene > adata.n_vars:
        return
    cell_labels = adata.obs[cluster_key].to_numpy()
    cell_types = np.unique(cell_labels)
    n_type = len(cell_types)
    count, peak_expr, peak_gene = count_peak_expression(adata, cluster_key)
    length_list = [len(peak_gene[x]) for x in cell_types]
    order_length = np.argsort(length_list)
    k = 0
    s = 0
    while s+length_list[order_length[k]]*(n_type-k) < n_gene:
        s = s+length_list[order_length[k]]
        k = k+1

    gene_list = []
    # Cell types with all peak genes picked
    for i in range(k):
        gene_list.extend(peak_gene[cell_types[order_length[i]]])
    n_gene_per_type = (n_gene - s)//(n_type-k)
    for i in range(k, n_type-1):
        gene_idx_order = np.flip(np.argsort(peak_expr[cell_types[order_length[i]]]))
        gene_list.extend(peak_gene[cell_types[order_length[i]]][gene_idx_order[:n_gene_per_type]])
    if k < n_type-1:
        gene_idx_order = np.flip(np.argsort(peak_expr[cell_types[order_length[-1]]]))
        n_res = n_gene - s - n_gene_per_type*(n_type-k-1)
        gene_list.extend(peak_gene[cell_types[order_length[-1]]][gene_idx_order[:n_res]])
    gene_subsel = np.zeros((adata.n_vars), dtype=bool)
    gene_subsel[np.array(gene_list).astype(int)] = True
    adata._inplace_subset_var(gene_subsel)
    return


def filt_gene_sparsity(adata, thred_u=0.99, thred_s=0.99):
    N, G = adata.n_obs, adata.n_vars
    sparsity_u = np.zeros((G))
    sparsity_s = np.zeros((G))
    for i in tqdm(range(G)):
        sparsity_u[i] = np.sum(adata.layers["unspliced"][:, i].A.squeeze() == 0)/N
        sparsity_s[i] = np.sum(adata.layers["spliced"][:, i].A.squeeze() == 0)/N
    gene_subset = (sparsity_u < thred_u) & (sparsity_s < thred_s)
    print(f"Kept {np.sum(gene_subset)} genes after sparsity filtering")
    adata._inplace_subset_var(gene_subset)


def rank_gene_selection(adata, cluster_key, **kwargs):
    """Select genes using wilcoxon test.

    Args:
        adata (:class:`anndata.AnnData`):
            AnnData object.
        cluster_key (str):
            Key for cell type annotation.
    """
    if "cell_types" not in kwargs:
        cell_types = np.unique(adata.obs[cluster_key].to_numpy())
    else:
        cell_types = kwargs["cell_types"]
    use_raw = kwargs["use_raw"] if "use_raw" in kwargs else False
    layer = kwargs["layer"] if "layer" in kwargs else None
    scanpy.tl.rank_genes_groups(adata,
                                groupby=cluster_key,
                                use_raw=use_raw,
                                layer=layer,
                                method='wilcoxon',
                                pts=True)
    min_in_group_fraction = kwargs["min_in_group_fraction"] if "min_in_group_fraction" in kwargs else 0.1
    min_fold_change = kwargs["min_fold_change"] if "min_fold_change" in kwargs else 1.5
    max_out_group_fraction = kwargs["max_out_group_fraction"] if "max_out_group_fraction" in kwargs else 0.5
    compare_abs = kwargs["compare_abs"] if "compare_abs" in kwargs else False
    scanpy.tl.filter_rank_genes_groups(adata,
                                       groupby=cluster_key,
                                       use_raw=False,
                                       min_in_group_fraction=min_in_group_fraction,
                                       min_fold_change=min_fold_change,
                                       max_out_group_fraction=max_out_group_fraction,
                                       compare_abs=compare_abs)
    gene_subset = np.zeros((adata.n_vars), dtype=bool)
    # Build a gene index mapping
    gene_dic = {}
    for i, x in enumerate(adata.var_names):
        gene_dic[x] = i
    gene_set = set()
    for ctype in cell_types:
        names = adata.uns['rank_genes_groups_filtered']['names'][ctype].astype(str)
        adata.uns['rank_genes_groups_filtered']['names'][ctype] = names
        gene_set = gene_set.union(set(names))
    for gene in gene_set:
        if gene != 'nan':
            gene_subset[gene_dic[gene]] = True
    print(f"Picked {len(gene_set)-1} genes")
    adata._inplace_subset_var(gene_subset)
    del adata.uns['rank_genes_groups']['pts']
    del adata.uns['rank_genes_groups']['pts_rest']
    del adata.uns['rank_genes_groups_filtered']['pts']
    del adata.uns['rank_genes_groups_filtered']['pts_rest']


def preprocess(adata,
               n_gene=1000,
               cluster_key="clusters",
               tkey=None,
               selection_method="scv",
               min_count_per_cell=0,
               min_genes_expressed=None,
               min_shared_counts=10,
               min_shared_cells=10,
               min_counts_s=None,
               min_cells_s=None,
               max_counts_s=None,
               max_cells_s=None,
               min_counts_u=None,
               min_cells_u=None,
               max_counts_u=None,
               max_cells_u=None,
               npc=30,
               n_neighbors=30,
               genes_retain=None,
               perform_clustering=False,
               resolution=1.0,
               compute_umap=False,
               umap_min_dist=0.5,
               keep_raw=True,
               **kwargs):
    """Run the entire preprocessing pipeline using scanpy

    Args:
        adata (:class:`anndata.AnnData`):
            AnnData object.
        n_gene (int, optional):
            Number of genes to keep. Defaults to 1000.
        cluster_key (str, optional):
            Key for cell type annotations. Defaults to "clusters".
        tkey (str, optional):
            Key in adata.obs containing the capture time. Defaults to None.
        selection_method (str, optional):
            {'scv','balanced','wilcoxon'}.
            If set to 'balanced', the function will call balanced_gene_selection.
            If set to 'wilcoxon', the function will call rank_gene_selection.
            Otherwise, it uses scanpy to pick highly variable genes.
            Defaults to "scv".
        min_count_per_cell (int, optional):
            Minimum total count per cell. Defaults to 0.
        min_genes_expressed (int, optional):
            Defaults to None.
        min_shared_counts (int, optional):
            Defaults to 10.
        min_shared_cells (int, optional):
            Defaults to 10.
        min_counts_s (int, optional):
            Defaults to None.
        min_cells_s (int, optional):
            Defaults to None.
        max_counts_s (int, optional):
            Defaults to None.
        max_cells_s (int, optional):
            Defaults to None.
        min_counts_u (int, optional):
            Defaults to None.
        min_cells_u (int, optional):
            Defaults to None.
        max_counts_u (int, optional):
            Defaults to None.
        max_cells_u (int, optional):
            Defaults to None.
        npc (int, optional):
            Number of PCA dimensions. Defaults to 30.
        n_neighbors (int, optional):
            Number of neighbors in KNN. Defaults to 30.
        genes_retain (array like, optional):
            Preprocessing will pick these exact genes
            regardless of their counts and gene selection method.
            Defaults to None.
        perform_clustering (bool, optional):
            Whether to perform leiden clustering. Defaults to False.
        resolution (float, optional):
            Leiden clustering hyperparameter. Defaults to 1.0.
        compute_umap (bool, optional):
            Whether to compute 2D UMAP. Defaults to False.
        umap_min_dist (float, optional):
            UMAP hyperparameter. Defaults to 0.5.
        keep_raw (bool, optional):
            Whether to keep the original raw counts (without normalization).
            Defaults to True.
    """
    # Preprocessing
    # 1. Cell, Gene filtering and data normalization
    n_cell = adata.n_obs
    if min_count_per_cell is None:
        min_count_per_cell = n_gene * 0.5
    if min_genes_expressed is None:
        min_genes_expressed = n_gene // 50
    scanpy.pp.filter_cells(adata, min_counts=min_count_per_cell)
    scanpy.pp.filter_cells(adata, min_genes=min_genes_expressed)
    if n_cell - adata.n_obs > 0:
        print(f"Filtered out {n_cell - adata.n_obs} cells with low counts.")

    if keep_raw:
        gene_names_all = np.array(adata.var_names)
        U_raw = adata.layers["unspliced"]
        S_raw = adata.layers["spliced"]

    if n_gene > 0:
        flavor = kwargs["flavor"] if "flavor" in kwargs else "seurat"
        if selection_method == "balanced":
            print("Balanced gene selection.")
            filter_genes(adata,
                         min_counts=min_counts_s,
                         min_cells=min_cells_s,
                         max_counts=max_counts_s,
                         max_cells=max_cells_s,
                         min_counts_u=min_counts_u,
                         min_cells_u=min_cells_u,
                         max_counts_u=max_counts_u,
                         max_cells_u=max_cells_u,
                         retain_genes=genes_retain)
            balanced_gene_selection(adata, n_gene, cluster_key)
            normalize_per_cell(adata)
        elif selection_method == "wilcoxon":
            print("Marker gene selection using Wilcoxon test.")
            filter_genes(adata,
                         min_counts=min_counts_s,
                         min_cells=min_cells_s,
                         max_counts=max_counts_s,
                         max_cells=max_cells_s,
                         min_counts_u=min_counts_u,
                         min_cells_u=min_cells_u,
                         max_counts_u=max_counts_u,
                         max_cells_u=max_cells_u,
                         retain_genes=genes_retain)
            normalize_per_cell(adata)
            log1p(adata)
            if adata.n_vars > n_gene:
                filter_genes_dispersion(adata,
                                        n_top_genes=n_gene,
                                        retain_genes=genes_retain,
                                        flavor=flavor)
            rank_gene_selection(adata, cluster_key, **kwargs)
        else:
            filter_and_normalize(adata,
                                 min_shared_counts=min_shared_counts,
                                 min_shared_cells=min_shared_cells,
                                 min_counts_u=min_counts_u,
                                 n_top_genes=n_gene,
                                 retain_genes=genes_retain,
                                 flavor=flavor)
    elif genes_retain is not None:
        gene_subset = np.zeros(adata.n_vars, dtype=bool)
        for i in range(len(genes_retain)):
            indices = np.where(adata.var_names == genes_retain[i])[0]
            if len(indices) == 0:
                continue
            gene_subset[indices[0]] = True
        adata._inplace_subset_var(gene_subset)
        normalize_per_cell(adata)
        log1p(adata)

    # second round of gene filter in case genes in genes_retain don't fulfill
    # minimal count requirement
    if genes_retain is not None:
        filter_genes(adata,
                     min_counts=min_counts_s,
                     min_cells=min_cells_s,
                     max_counts=max_counts_s,
                     max_cells=max_cells_s,
                     min_counts_u=min_counts_u,
                     min_cells_u=min_cells_u,
                     max_counts_u=max_counts_u,
                     max_cells_u=max_cells_u,
                     retain_genes=genes_retain)

    # 2. KNN Averaging
    # remove_duplicate_cells(adata)
    moments(adata, n_pcs=npc, n_neighbors=n_neighbors)

    if keep_raw:
        print("Keep raw unspliced/spliced count data.")
        gene_idx = np.array([np.where(gene_names_all == x)[0][0] for x in adata.var_names])
        adata.layers["unspliced"] = U_raw[:, gene_idx].astype(int)
        adata.layers["spliced"] = S_raw[:, gene_idx].astype(int)

    # 3. Obtain cell clusters
    if perform_clustering:
        scanpy.tl.leiden(adata, key_added='clusters', resolution=resolution)
    # 4. Obtain Capture Time (If available)
    if tkey is not None:
        capture_time = adata.obs[tkey].to_numpy()
        if isinstance(capture_time[0], str):
            tprior = np.array([float(x[1:]) for x in capture_time])
        else:
            tprior = capture_time
        tprior = tprior - tprior.min() + 0.01
        adata.obs["tprior"] = tprior

    # 5. Compute Umap coordinates for visulization
    if compute_umap:
        print("Computing UMAP coordinates.")
        if "X_umap" in adata.obsm:
            print("Warning: Overwriting existing UMAP coordinates.")
        scanpy.tl.umap(adata, min_dist=umap_min_dist)

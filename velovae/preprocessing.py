import anndata
import scanpy
import numpy as np
from .scvelo_preprocessing import *

def count_peak_expression(adata, cluster_key = "clusters"):
    """
    Count the number of genes with peak expression in each cell type
    """
    def encodeType(cell_types_raw):
        """
        Use integer to encode the cell types
        Each cell type has one unique integer label.
        """
        #Map cell types to integers 
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
    
    X=adata.layers["spliced"].todense()+adata.layers["unspliced"].todense()
    peak_expression = np.stack([np.quantile(X[cell_labels==j],0.9,0) for j in range(n_type)])
    peak_type = np.argmax(peak_expression, 0)
    peak_hist = np.array([np.sum(peak_type==i) for i in range(n_type)]) #gene count 
    peak_val_hist = [peak_expression[:,peak_type==i][i] for i in range(n_type)] #peak expression 
    peak_gene = [np.where(peak_type==i)[0] for i in range(n_type)] #peak gene index list
    
    out_peak_count = {}
    out_peak_expr = {}
    out_peak_gene = {}
    for i in range(n_type):
        out_peak_count[label_dic_rev[i]] = peak_hist[i]
        out_peak_expr[label_dic_rev[i]] = peak_val_hist[i]
        out_peak_gene[label_dic_rev[i]] = np.array(peak_gene[i])
    
    return out_peak_count,out_peak_expr,out_peak_gene

def balanced_gene_selection(adata, n_gene, cluster_key):
    if(n_gene>adata.n_vars):
        return 
    cell_labels = adata.obs[cluster_key].to_numpy()
    cell_types = np.unique(cell_labels)
    n_type = len(cell_types)
    count, peak_expr, peak_gene = count_peak_expression(adata, cluster_key)
    length_list = [len(peak_gene[x]) for x in cell_types]
    order_length = np.argsort(length_list)
    k = 0
    s = 0
    while(s+length_list[order_length[k]]*(n_type-k)<n_gene):
        s = s+length_list[order_length[k]]
        k = k+1
    
    gene_list = []
    #Cell types with all peak genes picked
    for i in range(k):
        gene_list.extend(peak_gene[cell_types[order_length[i]]])
    n_gene_per_type = (n_gene - s)//(n_type-k)
    for i in range(k, n_type-1):
        gene_idx_order = np.flip(np.argsort(peak_expr[cell_types[order_length[i]]]))
        gene_list.extend(peak_gene[cell_types[order_length[i]]][gene_idx_order[:n_gene_per_type]])
    if(k < n_type-1):
        gene_idx_order = np.flip(np.argsort(peak_expr[cell_types[order_length[-1]]]))
        n_res = n_gene - s - n_gene_per_type*(n_type-k-1)
        gene_list.extend(peak_gene[cell_types[order_length[-1]]][gene_idx_order[:n_res]])
    gene_subsel = np.zeros((adata.n_vars), dtype=bool)
    gene_subsel[np.array(gene_list).astype(int)] = True
    adata._inplace_subset_var(gene_subsel)
    return 

def preprocess(adata, 
               Ngene=1000,
               cluster_key="clusters",
               tkey=None,
               selection_method="scv",
               min_count_per_cell=20,
               min_genes_expressed=20,
               min_shared_count=10,
               min_shared_cells=10,
               min_counts_s=None,
               min_cells_s=None,
               max_counts_s=None,
               max_cells_s=None,
               min_counts_u=None,
               min_cells_u=None,
               max_counts_u=None,
               max_cells_u=None,
               max_proportion_per_cell=0.05,
               npc=30,
               n_neighbors=30,
               umap_min_dist=0.5,
               resolution=1.0,
               genes_retain=None,
               perform_clustering=False,
               compute_umap=False):
    """
    Perform all kinds of preprocessing steps using scanpy
    By setting genes_retain to a specific list of gene names, preprocessing will pick these exact genes regardless of their counts and gene selection method.
    """
    #Preprocessing
    #1. Cell, Gene filtering and data normalization
    scanpy.pp.filter_cells(adata, min_counts=min_count_per_cell)
    scanpy.pp.filter_cells(adata, min_genes=min_genes_expressed)
    if(genes_retain is None):
        if(selection_method=="balanced"):
            print("Balanced gene selection.")
            filter_and_normalize(adata, min_shared_counts=min_shared_count, min_shared_cells = min_shared_cells,  min_counts_u = min_counts_u, n_top_genes=Ngene*3)
            #filter_genes(adata, min_shared_counts=min_shared_count, min_shared_cells = min_shared_cells)
            #normalize_per_cell(adata, max_proportion_per_cell=max_proportion_per_cell)
            balanced_gene_selection(adata, Ngene, cluster_key)
            #log1p(adata)
        else:
            filter_genes(adata, 
                         min_counts=min_counts_s, 
                         min_cells=min_cells_s, 
                         max_counts=max_counts_s, 
                         max_cells=max_cells_s, 
                         min_counts_u=min_counts_u, 
                         min_cells_u=min_cells_u, 
                         max_counts_u=max_counts_u, 
                         max_cells_u=max_cells_u)
            filter_and_normalize(adata, min_shared_counts=min_shared_count, min_shared_cells = min_shared_cells,  min_counts_u = min_counts_u,  n_top_genes=Ngene)
            
    else:
        gene_index = np.ones(adata.n_vars, dtype=bool)
        for i in range(len(genes_retain)):
            indices = np.where(adata.var_names==genes_retain[i])[0]
            if(len(indices)==0):
                continue
            idx = indices[0]
            gene_index[idx] = False
        
        normalize_per_cell(adata)
        log1p(adata)
    
    #2. KNN Averaging
    moments(adata, n_pcs=npc, n_neighbors=n_neighbors)
    
    
    #3. Obtain cell clusters
    if(perform_clustering):
        scanpy.tl.leiden(adata, key_added='clusters', resolution=resolution)
    #4. Obtain Capture Time (If available)
    if(tkey is not None):
        capture_time = adata.obs[tkey].to_numpy()
        if(isinstance(capture_time[0], str)):
            tprior = np.array([float(x[1:]) for x in capture_time])
        else:
            tprior = capture_time
        tprior = tprior - tprior.min() + 0.01
        adata.obs["tprior"] = tprior
    
    #5. Compute Umap coordinates for visulization
    if(compute_umap):
        if("X_umap" in adata.obsm):
            print("Warning: Overwriting existing UMAP coordinates.")
        scanpy.tl.umap(adata, min_dist=umap_min_dist)

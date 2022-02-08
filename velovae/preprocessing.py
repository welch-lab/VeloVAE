import anndata
import scanpy
from .scvelo_preprocessing import *


def preprocess(adata, 
               Ngene,
               min_count_per_cell=20,
               min_genes_expressed=20,
               min_shared_count=50,
               min_shared_cells=10,
               max_genes=None,
               max_cells=None,
               npc=30,
               n_neighbors=30,
               umap_min_dist=0.5,
               tkey=None):
    """
    Perform all kinds of preprocessing steps using scanpy
    """
    #Preprocessing
    #1. Cell, Gene filtering and data normalization
    scanpy.pp.filter_cells(adata, min_counts=min_count_per_cell)
    scanpy.pp.filter_cells(adata, min_genes=min_genes_expressed)
    filter_and_normalize(adata, min_shared_counts=min_shared_count, min_shared_cells = min_shared_cells,  n_top_genes=Ngene)
    #2. KNN Averaging
    moments(adata, n_pcs=npc, n_neighbors=n_neighbors)
    #3. Obtain cell clusters
    if(not 'clusters' in adata.obs):
        scanpy.tl.leiden(adata, key_added='clusters')
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
    if(not 'X_umap' in adata.obsm):
        scanpy.tl.umap(adata, min_dist=umap_min_dist)
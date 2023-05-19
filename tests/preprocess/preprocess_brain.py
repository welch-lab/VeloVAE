import anndata
import numpy as np
import os
import sys
sys.path.append('../../')
import velovae as vv

def preprocess_brain():
    dataset="Braindev_full"
    adata = anndata.read_h5ad("/nfs/turbo/umms-welchjd/yichen/data/scRNA/Braindev_full/Braindev_full.h5ad")
    root = "/nfs/turbo/umms-welchjd/yichen/data/scRNA/Braindev_full"
    
    cell_labels = adata.obs["Class"].to_numpy()
    adata.obs["clusters"] = cell_labels
    cell_mask = ~( (cell_labels=="Bad cells") | (cell_labels=="Undefined") )
    adata = adata[cell_mask]
    
    #Preprocessing
    vv.preprocess(adata, n_gene=2000, keep_raw=True)
    
    #Add capture time
    capture_time = adata.obs["Age"].to_numpy()
    tprior = np.array([float(x[1:]) for x in capture_time])
    adata.obs["tprior"] = tprior
    
    adata.write_h5ad(f"{root}/{dataset}_pp.h5ad")

preprocess_brain()

import anndata
import numpy as np
import os
import sys
sys.path.append('../../')
import velovae as vv

def preprocess_embryo():
    dataset = "Embryo"
    
    print("Reading file...")
    adata = anndata.read_h5ad("/nfs/turbo/umms-welchjd/yichen/data/scRNA/Embryo/mouse_E9_13.h5ad")
    root = "/nfs/turbo/umms-welchjd/yichen/data/scRNA/Embryo"
    
    import pandas as pd
    cell_df = pd.read_csv("/nfs/turbo/umms-welchjd/yichen/data/scRNA/Embryo/cell_annotate.csv")
    gene_df = pd.read_csv(f"/nfs/turbo/umms-welchjd/yichen/data/scRNA/Embryo/gene_annotate.csv")
    #cell annotation
    cell_id_anno = cell_df['sample'].to_numpy()
    cell_id = adata.obs.index.to_numpy()
    check_dic = {}
    for x in cell_id_anno:
        check_dic[x] = 0
    for x in cell_id:
        check_dic[x] = 0
    for x in cell_id_anno:
        check_dic[x] += 1
    for x in cell_id:
        check_dic[x] += 1
    is_in_anno = np.array([False for i in range(len(cell_id_anno))])
    is_in_adata = np.array([False for i in range(len(cell_id))])
    
    for i in range(len(cell_id)):
        is_in_adata[i] = (check_dic[cell_id[i]]==2)
    for i in range(len(cell_id_anno)):
        is_in_anno[i] = (check_dic[cell_id_anno[i]]==2)
    adata = adata[is_in_adata]
    
    for key in cell_df.keys():
        arr = cell_df[key].to_numpy()
        if(arr.dtype=='object'):
            arr = np.array([str(x) for x in arr])
        adata.obs[key] = arr[is_in_anno]
    
    adata.obsm["X_tsne"] = np.stack([adata.obs["Main_cluster_tsne_1"].to_numpy(),adata.obs["Main_cluster_tsne_2"].to_numpy()]).T
    
    #gene annotation
    gene_id = adata.var.index.to_numpy()
    gene_id_anno_raw = gene_df["gene_id"].to_numpy()
    gene_id = np.array([str(x) for x in gene_id])
    gene_id_anno = np.empty((len(gene_id_anno_raw)), dtype='U18')
    print(len(gene_id), len(gene_id_anno))
    for i,x in enumerate(gene_id_anno_raw):
        gene_id_split = x.split('.')
        gene_id_anno[i] = gene_id_split[0]
    
    check_dic_anndata = {}
    for x in gene_id:
        check_dic_anndata[x] = False
    for x in gene_id_anno:
        if(x in check_dic_anndata):
            check_dic_anndata[x] = True
    
    check_dic_anno = {}
    for x in gene_id_anno:
        check_dic_anno[x] = False
    for x in gene_id:
        if(x in check_dic_anno):
            check_dic_anno[x] = True
    
    is_in_adata = np.array([check_dic_anndata[x] for x in check_dic_anndata])
    is_in_df = np.array([check_dic_anno[x] for x in check_dic_anno])
    
    adata.var["gene_id"] = gene_id
    adata = adata[:,is_in_adata]
    
    gene_type = gene_df["gene_type"][is_in_df].to_numpy()
    gene_type = np.array([str(x) for x in gene_type])
    adata.var["gene_type"] = gene_type
    
    gene_short_name = gene_df["gene_short_name"][is_in_df].to_numpy()
    gene_short_name = np.array([str(x) for x in gene_short_name])
    adata.var.index = pd.Index(gene_short_name)
    
    adata.var_names_make_unique()
    
    cell_labels = adata.obs["Main_cell_type"].to_numpy()
    cell_labels = np.array([str(x) for x in cell_labels])
    nan_mask = ~(cell_labels == 'nan')
    adata = adata[nan_mask]

    #Informative time prior
    capture_time = adata.obs["day"].to_numpy()
    tprior = np.array([float(x[1:]) for x in capture_time])
    adata.obs["tprior"] = tprior - tprior.min()
    
    #Cell Cluster
    cell_labels = adata.obs["Main_cell_type"].to_numpy()
    cell_labels = np.array([str(x) for x in cell_labels])
    adata.obs["clusters"] = cell_labels
    
    #Preprocessing
    vv.preprocess(adata, n_gene=2000, keep_raw=True)
    
    adata.write_h5ad(f"{root}/{dataset}_dpp.h5ad")

preprocess_embryo()
import anndata
import numpy as np
import scvelo as scv
import sys
sys.path.append('../')
import velovae as vv
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import umap

colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'lime', 'grey', \
   'olive', 'cyan', 'pink', 'gold', 'steelblue', 'salmon', 'teal', \
   'magenta', 'rosybrown', 'darkorange', 'yellow', 'greenyellow', 'darkseagreen', 'yellowgreen', 'palegreen', \
   'hotpink', 'navajowhite', 'aqua', 'navy', 'saddlebrown', 'black', 'maroon']

def testVanillaVAE():
    adata = scv.datasets.pancreas()
    vv.preprocess(adata)
    model = vv.VanillaVAE(adata, Tmax)
    model.train(adata, config={'num_epochs':1}, gene_plot=['Cpe'], figure_path='./')
    model.saveModel('./')
    model.saveAnnData(self, './', file_name='pancreas.h5ad')

def testVAEpp(): 
    filename = '../data/Pancreas/pancreas.h5ad'
    adata = anndata.read_h5ad(filename)
    vv.preprocess(adata, Ngene)
    
    figure_path = '/home/gyichen/velovae/figures/Pancreas/HyperTune_1'
    model_path = '/home/gyichen/velovae/checkpoints/Pancreas/HyperTune_1'
    data_path = '../data/Pancreas/Rho'
    
    key = 'hyper1'
    
    Cz = 5
    k = 30
    dt = [0.1,0.12]
    N_knn = 100
    config_vae = {
        'num_epochs':500, 'test_epoch':50, 'save_epoch':50, 
        'learning_rate':2e-4, 'learning_rate_ode':2e-4, 'lambda':1e-3, 
        'neg_slope':0, 'reg_t':1.0, 'reg_z':1.0, 'batch_size':128,
        'knn': k, 'coeff_t1': dt[0], 'coeff_t2': dt[1], 'Cz': Cz, 'N_knn':N_knn
    }
    model = vv.VanillaVAEpp(adata, 20, Cz, device='gpu', hidden_size=(500,250,250,500), tprior=None, coeff_t=dt, n_neighbor=k)
    gene_plot = ['Pcsk2','Dcdc2a','Gng12','Cpe','Smoc1','Tmem163','Ank', 'Ppp3ca']
    model.train(adata, config=config_vae, plot=True,gene_plot=gene_plot, figure_path=figure_path)
    
    with open(figure_path+'/config.txt','w') as f:
        for key in model.config:
            f.write(key+'\t'+str(model.config[key])+'\n')
    
    model.saveModel(model_path, 'encoder_vanillapp', 'decoder_vanillapp')
    model.saveAnnData(adata, key, data_path, file_name='output_vanillapp.h5ad')
    
    
    z = adata.obsm[f'{key}_z']
    t = adata.obs[f'{key}_time'].to_numpy()
    t0 = adata.obs[f'{key}_t0'].to_numpy()
    u0 = adata.layers[f'{key}_u0']
    s0 = adata.layers[f'{key}_s0']
    rho = adata.layers[f'{key}_rho']
    ton = adata.var[f'{key}_ton'].to_numpy()
    toff = adata.var[f'{key}_toff'].to_numpy()
    
    pca = PCA(n_components=3)
    rho_pca = pca.fit_transform(rho)
    
    cell_labels = adata.obs['clusters'].to_numpy()
    cell_types = np.unique(cell_labels)
    fig=plt.figure(figsize=(15,12))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(10, 30)
    for i,x in enumerate((cell_types)):
        ax.scatter(rho_pca[cell_labels==x,0], rho_pca[cell_labels==x,1], rho_pca[cell_labels==x,2], label=x, color=colors[i])
        
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.legend(bbox_to_anchor=(-0.15,1.0), loc='upper right')
    plt.show()
    fig.savefig(figure_path+'/rho_pca.png')
    
    umap_obj = umap.UMAP(n_neighbors=50, n_components=2, min_dist=0.1)
    z_umap = umap_obj.fit_transform(z)
    
    fig=plt.figure(figsize=(10,10))
    for i,x in enumerate((cell_types)):
        plt.scatter(z_umap[cell_labels==x,0], z_umap[cell_labels==x,1], label=x, color=colors[i])
    plt.legend(bbox_to_anchor=(-0.15,1.0), loc='upper right')
    plt.show()
    fig.savefig(figure_path+'/z_umap.png')
    
    scv.tl.velocity_graph(adata, vkey=f'{key}_velocity')
    scv.tl.velocity_embedding(adata, vkey=f'{key}_velocity')
    scv.pl.velocity_embedding_stream(adata, vkey=f'{key}_velocity', figsize=(8,6), save=figure_path+f'/velocity.png')
    
testVAEpp()
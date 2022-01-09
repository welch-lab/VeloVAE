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
def run_scvelo():
    adata = anndata.read_h5ad("../data/Dentategyrus/dentategyrus.h5ad")
    vv.preprocess(adata, 800)
    scv.tl.recover_dynamics(adata)
    
def testVanillaVAE():
    figure_path = '/home/gyichen/velovae/figures/Dentategyrus/train_test'
    adata = anndata.read_h5ad("../data/Dentategyrus/dentategyrus.h5ad")
    #vv.preprocess(adata, 800)
    Tmax = 20
    model = vv.VanillaVAE(adata, Tmax)
    model.train(adata, config={'num_epochs':100}, plot=True, gene_plot=['Cpe'], figure_path=figure_path)
    model.saveAnnData(adata)

def testPipeline(): 
    dataset = 'Pancreas'
    filename = '../data/Pancreas/output.h5ad'
    #filename = '/nfs/turbo/umms-welchjd/yichen/data/scRNA/braindev_pp.h5ad'
    adata = anndata.read_h5ad(filename)
    #vv.preprocess(adata, Ngene)
    #adata.obs["tprior"] = adata.obs.tprior - adata.obs.tprior.min()
    #adata.obs['clusters'] = adata.obs['leiden'].to_numpy()
    
    """
    (Optional) Run scVelo
    """
    scv.tl.recover_dynamics(adata)
    
    """
    VAE
    """
    figure_path = f'../figures/{dataset}/VAE'
    model_path = f'../checkpoints/{dataset}/VAE'
    data_path = f'../data/{dataset}'
    #data_path = '/scratch/blaauw_root/blaauw1/gyichen'
    config = {
        'hidden_size':(500,250),
        'tmax':20,
        'num_epochs':500, 
        'test_epoch':50, 
        'save_epoch':50, 
        'learning_rate':2.5e-4, 
        'learning_rate_ode':5e-4, 
        'neg_slope':0.0,
        'lambda':1e-3, 
        'reg_t':1.0, 
        'batch_size':128,
        'train_test_split':0.7,
        'tprior':None
    }
    
    vae = vv.VanillaVAE(adata, config['tmax'], hidden_size=config['hidden_size'], tprior=config['tprior'], device='cuda:0')
    vae.train(adata, config=config, gene_plot=gene_plot, figure_path=figure_path)
    vae.saveModel(model_path)
    vae.saveAnnData(adata, 'vanilla', data_path, file_name='output.h5ad')
    with open(figure_path+'/config.txt','w') as f:
        for key in vae.config:
            f.write(key+'\t'+str(vae.config[key])+'\n')
    
    """
    VAE++
    """
    figure_path = f'/home/gyichen/velovae/figures/{dataset}/VAEpp'
    model_path = f'/home/gyichen/velovae/checkpoints/{dataset}/VAEpp'
    #data_path = '/nfs/turbo/umms-welchjd/yichen/data/scRNA'
    data_path = f'../data/{dataset}'
    config = {
        'Cz':5,
        'hidden_size':(500,250,250,500),
        'tmax':20,
        'num_epochs':250, 
        'num_epochs_post':250,
        'test_epoch':50, 
        'save_epoch':50, 
        'learning_rate':2e-4, 
        'learning_rate_ode':2e-4, 
        'lambda':1e-3, 
        'neg_slope':0, 
        'reg_t':1.0, 
        'reg_z':1.0, 
        'batch_size':128,
        'train_test_split':1.0,
        'N_knn_update': 25,
        'n_neighbors': 10, 
        'dt': (0.1, 0.15), 
        'tprior':None,
        'tkey':None
    }
    #tmax = adata.obs.vanilla_time.max()
    model = vv.VanillaVAEpp(adata, 
                            config['tmax'], 
                            config['Cz'], 
                            device='cuda:0', 
                            hidden_size=config['hidden_size'], 
                            tprior=config['tprior'], 
                            tkey=config['tkey'])
    model.train(adata, config=config, plot=True, gene_plot=gene_plot, figure_path=figure_path)
    
    with open(figure_path+'/config.txt','w') as f:
        for key in model.config:
            f.write(key+'\t'+str(model.config[key])+'\n')
    
    model.saveModel(model_path, 'encoder_vae++', 'decoder_vae++')
    model.saveAnnData(adata, 'vae++', data_path, file_name='output.h5ad')
    
#testVanillaVAE()    
testPipeline()
#run_scvelo()

#!/usr/bin/env python
# coding: utf-8

# In[1]:

# In[12]:


import sys
sys.path.append('/scratch/pkg/VeloVAE/VeloVAE')
import velovae as vv
import torch
import scanpy as sc
import numpy as np


# In[23]:


#TLS
#discrete model
TLS_adata_path = '/data/processed/TLS.h5ad'
# In[24]:


TLS_adata = sc.read_h5ad(TLS_adata_path)

output_path = '/VeloVAE/velovae_output/full/discrete'


model_path = output_path+'/'+dataset+'_model'
figure_path = output_path+'/'+dataset+'_figure'
data_path = output_path+'/'+dataset+'_data'

#n_gene = 1000
#vv.preprocess(scifate2_adata, 
              #n_gene,
              #compute_umap=True)

# In[29]:


device = 'cuda' if torch.cuda.is_available() else 'cpu' 


# In[30]:


gene_plot = ['Emp1', 'Hoxb9']


torch.manual_seed(42)
np.random.seed(42)
vae = vv.VAE(TLS_adata, 
             tmax=20, 
             dim_z=5,
             discrete=True,
             device=device)



config = {
    #'learning_rate':1e-3,
    #'learning_rate_ode':1e-3,
    #'learning_rate_post':1e-4,
    # You can change any hyperparameters here!
}
vae.train(TLS_adata,
          config=config,
          plot=False,
          gene_plot=gene_plot,
          figure_path=figure_path,
          embed='umap')


# In[45]:


vae.save_model(model_path, 'encoder_vae', 'decoder_vae')
vae.save_anndata(TLS_adata, 'vae', data_path, file_name="TLS_adata_out.h5ad")

# In[26]:

#continuous model
dataset = 'TLS'

device = 'cuda' if torch.cuda.is_available() else 'cpu' 

gene_plot = ['Emp1', 'Hoxb9']
# In[36]:
output_path = '/scratch/welchjd_root/welchjd0/yuxuans/VeloVAE/velovae_output/full/continuous'

TLS_adata = sc.read_h5ad(TLS_adata_path)

model_path = output_path+'/'+dataset+'_model'
figure_path = output_path+'/'+dataset+'_figure'
data_path = output_path+'/'+dataset+'_data'

torch.manual_seed(42)
np.random.seed(42)
vae = vv.VAE(TLS_adata, 
             tmax=20, 
             dim_z=5,
             #discrete=True,
             device=device)


config = {
    #'learning_rate':1e-3,
    #'learning_rate_ode':1e-3,
    #'learning_rate_post':1e-4,
    # You can change any hyperparameters here!
}
vae.train(TLS_adata,
          config=config,
          plot=False,
          gene_plot=gene_plot,
          figure_path=figure_path,
          embed='umap')

vae.save_model(model_path, 'encoder_vae', 'decoder_vae')
vae.save_anndata(TLS_adata, 'vae', data_path, file_name="TLS_adata_out.h5ad')





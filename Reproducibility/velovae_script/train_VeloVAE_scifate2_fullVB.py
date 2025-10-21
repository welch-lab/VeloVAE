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


#scifate2

scifate2_adata_path = '/data/processed/scifate2.h5ad'

#fullVB_continuous
scifate2_adata = sc.read_h5ad(scifate2_adata_path)

dataset = 'scifate2'

output_path = '/scratch/welchjd_root/welchjd0/yuxuans/VeloVAE/velovae_output/full/fullVB_continuous'

# In[40]:


model_path = output_path+'/'+dataset+'_model'
figure_path = output_path+'/'+dataset+'_figure'
data_path = output_path+'/'+dataset+'_data'

device = 'cuda' if torch.cuda.is_available() else 'cpu' 


# In[30]:


gene_plot = ['Emp1', 'Hoxb9']


torch.manual_seed(42)
np.random.seed(42)
vae = vv.VAE(scifate2_adata, 
             tmax=20, 
             dim_z=30,
             full_vb=True,
             device=device)


#1e-3,1e-3,1e-4
config = {
    #'learning_rate':1e-3,
    #'learning_rate_ode':1e-3,
    #'learning_rate_post':1e-4,
    # You can change any hyperparameters here!
}
vae.train(scifate2_adata,
          config=config,
          plot=False,
          gene_plot=gene_plot,
          figure_path=figure_path,
          embed='umap')


# In[45]:


vae.save_model(model_path, 'encoder_vae', 'decoder_vae')
vae.save_anndata(scifate2_adata, 'dfullvb', data_path, file_name="scifate2_adata_out.h5ad")



#fullVB discrete

scifate2_adata = sc.read_h5ad(scifate2_adata_path)

output_path = '/scratch/welchjd_root/welchjd0/yuxuans/VeloVAE/velovae_output/full/fullVB_discrete'

# In[40]:


model_path = output_path+'/'+dataset+'_model'
figure_path = output_path+'/'+dataset+'_figure'
data_path = output_path+'/'+dataset+'_data'


scifate2_adata = sc.read_h5ad(scifate2_adata_path)

# In[26]:


dataset = 'scifate2'

scifate2_adata.layers['spliced'] = scifate2_adata.layers['raw_spliced']
scifate2_adata.layers['unspliced'] = scifate2_adata.layers['raw_unspliced']
torch.manual_seed(42)
np.random.seed(42)
vae = vv.VAE(scifate2_adata, 
             tmax=20, 
             dim_z=30,
             discrete=True,
             full_vb=True,
             device=device)

config = {
    #'learning_rate':1e-3,
    #'learning_rate_ode':1e-3,
    #'learning_rate_post':1e-4,
    # You can change any hyperparameters here!
}
vae.train(scifate2_adata,
          config=config,
          plot=False,
          gene_plot=gene_plot,
          figure_path=figure_path,
          embed='umap')


# In[45]:


vae.save_model(model_path, 'encoder_vae', 'decoder_vae')
vae.save_anndata(scifate2_adata, 'dfullvb', data_path, file_name="scifate2_adata_out.h5ad")

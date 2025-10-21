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

#discrete model

#scifate2
scifate2_adata_path = '/data/processed/scifate2.h5ad'
# In[24]:


scifate2_adata = sc.read_h5ad(scifate2_adata_path)

# In[26]:


dataset = 'scifate2'


# In[36]:

output_path = '/VeloVAE/velovae_output/full/discrete'
# In[40]:


model_path = output_path+'/'+dataset+'_model'
figure_path = output_path+'/'+dataset+'_figure'
data_path = output_path+'/'+dataset+'_data'


# In[29]:


device = 'cuda' if torch.cuda.is_available() else 'cpu' 


# In[30]:


gene_plot = ['Emp1', 'Hoxb9']

torch.manual_seed(42)
np.random.seed(42)

vae = vv.VAE(scifate2_adata, 
             tmax=20, 
             dim_z=30,
             discrete=True,
             device=device)

# In[32]:



# In[33]:

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
vae.save_anndata(scifate2_adata, 'vae', data_path, file_name="scifate2_adata_out.h5ad")

#continuous model
scifate2_adata = sc.read_h5ad(scifate2_adata_path)

output_path = '/VeloVAE/velovae_output/full/continuous'

model_path = output_path+'/'+dataset+'_model'
figure_path = output_path+'/'+dataset+'_figure'
data_path = output_path+'/'+dataset+'_data'


# In[29]:


device = 'cuda' if torch.cuda.is_available() else 'cpu' 


# In[30]:


gene_plot = ['Emp1', 'Hoxb9']

torch.manual_seed(42)
np.random.seed(42)

vae = vv.VAE(scifate2_adata, 
             tmax=20, 
             dim_z=30,
             discrete=False,
             device=device)

# In[32]:



# In[33]:

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
vae.save_anndata(scifate2_adata, 'vae', data_path, file_name="scifate2_adata_out.h5ad")








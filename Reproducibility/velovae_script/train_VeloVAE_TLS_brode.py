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
TLS_adata_path = '/VeloVAE/velovae_output/full/continuous/TLS_data/TLS_adata_out.h5ad'
# In[24]:


TLS_adata = sc.read_h5ad(TLS_adata_path)

# In[26]:


dataset = 'TLS'


# In[36]:

output_path = '/VeloVAE/velovae_output/full/brode'
# In[40]:


model_path = output_path+'/'+dataset+'_model'
figure_path = output_path+'/'+dataset+'_figure'
data_path = output_path+'/'+dataset+'_data'


device = 'cuda' if torch.cuda.is_available() else 'cpu' 


# In[30]:


gene_plot = ['Emp1', 'Hoxb9']


brode = vv.BrODE(TLS_adata,
                 'cell_state',
                 'vae_time',
                 'vae_z',
                 param_key='vae',
                 device=device)


torch.manual_seed(42)
np.random.seed(42)


# In[32]:



# In[33]:

#1e-3,1e-3,1e-4
config = {
    #'learning_rate':1e-3,
    #'learning_rate_ode':1e-3,
    #'learning_rate_post':1e-4,
    # You can change any hyperparameters here!
}
brode.train(TLS_adata,
            'vae_time',
            'cell_state',
            plot=False,
            gene_plot=gene_plot,
            figure_path=figure_path,
            )


# In[45]:


brode.save_model(model_path, 'brode')
brode.save_anndata(TLS_adata, 'brode', data_path, file_name="TLS_adata_out.h5ad")


# In[ ]:





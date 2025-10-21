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
scifate2_adata_path = '/VeloVAE/velovae_output/full/continuous/scifate2_data/scifate2_adata_out.h5ad'
# In[24]:


scifate2_adata = sc.read_h5ad(scifate2_adata_path)

# In[26]:


dataset = 'scifate2'


# In[36]:

output_path = '/VeloVAE/velovae_output/full/brode'
# In[40]:


model_path = output_path+'/'+dataset+'_model'
figure_path = output_path+'/'+dataset+'_figure'
data_path = output_path+'/'+dataset+'_data'


# In[42]:


# In[28]:




device = 'cuda' if torch.cuda.is_available() else 'cpu' 


# In[30]:


gene_plot = ['Emp1', 'Hoxb9']



brode = vv.BrODE(scifate2_adata,
                 'cell_annotation',
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
brode.train(scifate2_adata,
            'vae_time',
            'cell_annotation',
            plot=False,
            gene_plot=gene_plot)


# In[45]:


brode.save_model(model_path, 'brode')
brode.save_anndata(scifate2_adata, 'brode', data_path, file_name="scifate2_adata_out.h5ad")


# In[ ]:





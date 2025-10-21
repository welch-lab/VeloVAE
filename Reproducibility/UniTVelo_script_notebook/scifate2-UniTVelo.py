#!/usr/bin/env python
# coding: utf-8

# In[1]:

# In[12]:


import unitvelo as utv
import anndata
import scvelo as scv
import numpy as np
import os
import argparse
import time


# In[23]:


#scifate2
scifate2_path = '/data/processed/scifate2.h5ad'
# In[24]:


scifate2 = anndata.read_h5ad(scifate2_path)

# Hyperparameters
velo_config = utv.config.Configuration()
velo_config.R2_ADJUST = True
velo_config.IROOT = None
velo_config.FIT_OPTION = '1'
velo_config.ASSIGN_POS_U = True
velo_config.GPU = 0

t_start = time.time()
label = 'cell_annotation'
scifate2 = utv.run_model(scifate2, label, config_file=velo_config)
run_time = time.time() - t_start
print(f"Total run time: {run_time}")

scv.pl.velocity_embedding_stream(scifate2,
                                 color=scifate2.uns['label'],
                                 title='',
                                 basis='umap',
                                 legend_loc='far right',
                                 dpi=100)
scv.tl.latent_time(scifate2, min_likelihood=None)

scifate2.obs['utv_time'] = scifate2.obs['latent_time'].to_numpy()
scifate2.uns['utv_run_time'] = run_time
scv.pl.scatter(scifate2, color="latent_time", basis='umap', cmap="plasma")
scifate2.write_h5ad('/VeloVAE/unitvelo/output/scifate2_unitvelo.h5ad')

# In[ ]:





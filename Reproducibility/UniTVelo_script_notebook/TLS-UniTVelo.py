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


TLS_path = '/VeloVAE/data/processed/TLS.h5ad'

TLS = anndata.read_h5ad(TLS_path)

velo_config = utv.config.Configuration()
velo_config.R2_ADJUST = True
velo_config.IROOT = None
velo_config.FIT_OPTION = '1'
velo_config.ASSIGN_POS_U = True
velo_config.GPU = 0

t_start = time.time()
label = 'cell_state'
TLS.var['gene'] = TLS.var.index.tolist()
TLS = utv.run_model(TLS, label, config_file=velo_config)
run_time = time.time() - t_start
print(f"Total run time: {run_time}")

scv.pl.velocity_embedding_stream(TLS,
                                 color=TLS.uns['label'],
                                 title='',
                                 basis='umap',
                                 legend_loc='far right',
                                 dpi=100)
scv.tl.latent_time(TLS, min_likelihood=None)

TLS.obs['utv_time'] = TLS.obs['latent_time'].to_numpy()
TLS.uns['utv_run_time'] = run_time
scv.pl.scatter(TLS, color="latent_time", basis='umap', cmap="plasma")
TLS.write_h5ad('/VeloVAE/unitvelo/output/TLS_unitvelo.h5ad')

# In[ ]:





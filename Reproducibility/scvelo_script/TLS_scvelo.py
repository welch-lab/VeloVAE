#!/usr/bin/env python
# coding: utf-8

# In[1]:

# In[12]:


import anndata
import numpy as np
import scvelo as scv
import pandas as pd
import sys
sys.path.append('../../../')
import velovae as vv
import time
import os

root = '< your root path >'
dataset = 'TLS'
adata = anndata.read_h5ad(f'{root}/data/{dataset}.h5ad')

t_start = time.time()
scv.tl.recover_dynamics(adata)
run_time = time.time() - t_start
scv.tl.velocity(adata, mode='dynamical')
scv.tl.velocity_graph(adata)
scv.tl.latent_time(adata)
adata.uns['fit_run_time'] = run_time
adata.write_h5ad(f'{data_path}/{dataset}.h5ad')

print(f"Total run time: {run_time}")
print("---------------------------------------------------")






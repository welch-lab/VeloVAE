#!/usr/bin/env python
# coding: utf-8

# In[1]:


import anndata
import numpy as np
import scvelo as scv
import pandas as pd
import sys
sys.path.append('../../../')
import velovae as vv
import time
import os


# In[2]:


root = '< your root path >'
dataset = 'Bonemarrow'
adata = anndata.read_h5ad(f'{root}/data/{dataset}_pp.h5ad')


# In[3]:


data_path = f'{root}/data/scv/{dataset}'
figure_path = f'{root}/figures/{dataset}/scv'
os.makedirs(data_path, exist_ok=True)
os.makedirs(figure_path, exist_ok=True)


# In[4]:


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


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:

# In[12]:

import anndata
import scvelo as scv
import scvi
import numpy as np
from velovi import preprocess_data, VELOVI
import torch
import os
import argparse
import time

# In[23]:


#scifate2
scifate2_path = '/data/processed/scifate2.h5ad'
# In[24]:


scifate2 = anndata.read_h5ad(scifate2_path)

# In[26]:
start = time.time()
VELOVI.setup_anndata(scifate2, spliced_layer="Ms", unspliced_layer="Mu")
torch.set_float32_matmul_precision('highest')
vae = VELOVI(scifate2)
scvi.settings.seed=2022
vae.train(max_epochs=500, accelerator='cuda',devices=1, train_size=1.0)
run_time = time.time() - start

def add_velovi_outputs_to_adata(adata, vae, num_samples=25):
    vae.module = vae.module.to(torch.device('cpu'))
    latent_time = vae.get_latent_time(n_samples=num_samples)
    velocities = vae.get_velocity(n_samples=num_samples, velo_statistic="mean")
    
    
    t = latent_time
    scaling = 20 / t.max(0)

    adata.layers["velovi_velocity"] = velocities / scaling
    adata.layers["latent_time_velovi"] = latent_time
    
    adata.var["velovi_alpha"] = vae.get_rates()["alpha"] / scaling
    adata.var["velovi_beta"] = vae.get_rates()["beta"] / scaling
    adata.var["velovi_gamma"] = vae.get_rates()["gamma"] / scaling
    adata.var["velovi_t_"] = (
        torch.nn.functional.softplus(vae.module.switch_time_unconstr)
        .detach()
        .cpu()
        .numpy()
    ) * scaling
    print(type(latent_time))
    print(type(scaling))
    #latent_time_np = np.asarray(latent_time)
    #adata.layers["fit_t"] = latent_time_np[:, np.newaxis] * scaling[np.newaxis, :]
    #adata.layers["fit_t"] = latent_time.to_numpy()[:, np.newaxis] * scaling[np.newaxis, :]
    scaling=np.array(scaling)
    latent_time=np.array(latent_time)
    #adata.layers["fit_t"] = latent_time.to_numpy() * scaling[np.newaxis, :]
    adata.layers["fit_t"] = latent_time* scaling[np.newaxis, :]
    adata.var['velovi_scaling'] = 1.0

    shat, uhat = vae.get_expression_fit(adata, n_samples=num_samples, return_mean=True)
    likelihood = vae.get_gene_likelihood(adata, n_samples=num_samples, return_mean=True)
    adata.layers["velovi_uhat"] = uhat
    adata.layers["velovi_shat"] = shat
    adata.obs["velovi_likelihood"] = likelihood.sum(1)
    adata.var["fit_likelihood"] = likelihood.mean(0)

    scv.tl.velocity_graph(adata, vkey="velovi_velocity")
    scv.tl.latent_time(adata, vkey="velovi_velocity")
    adata.obs['velovi_time'] = adata.obs['latent_time'].to_numpy()
    del adata.obs['latent_time']
    
    random_state = np.random.RandomState(seed=0)
    permutation = random_state.permutation(adata.n_obs)
    n_val = int(np.floor(adata.n_obs * 0.3))
    adata.uns['velovi_test_idx'] = permutation[:n_val]
    adata.uns['velovi_train_idx'] = permutation[n_val:]
    adata.uns['velovi_run_time'] = run_time
    
add_velovi_outputs_to_adata(scifate2, vae)
scifate2.write_h5ad('/VeloVAE/velovi/output/scifate2_velovi.h5ad')
# In[36]:


# In[ ]:





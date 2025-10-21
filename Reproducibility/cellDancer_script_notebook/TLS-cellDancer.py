#!/usr/bin/env python
# coding: utf-8

# In[1]:


import celldancer as cd

import anndata

TLS_path = '/data/processed/TLS.h5ad'


# In[6]:


TLS = anndata.read_h5ad(TLS_path)
TLS.obs_names_make_unique()

N, G = TLS.shape


# In[10]:


data_path = '/VeloVAE/celldancer/data'


# In[11]:


dataset='TLS'


# In[12]:


df = cd.adata_to_df_with_embed(TLS,
                               cell_type_para='cell_state',
                               embed_para='X_umap',
                               save_path=f"{data_path}/{dataset}.csv")


# In[ ]:


df_loss , df = cd.velocity(df,
                           max_epoches=200,
                           permutation_ratio=0.125,
                           n_jobs=8,
                           save_path=data_path)


# In[ ]:


#n_neigh = TLS.uns['neighbors']['indices'].shape[1]
n_neigh = 30
df = cd.compute_cell_velocity(cellDancer_df=df,
                              projection_neighbor_choice="gene",
                              expression_scale='power10',
                              projection_neighbor_size=n_neigh,
                              speed_up=(100, 100))


# In[ ]:


dt = 0.05
n_repeats = 10
t_total = {dt: int(10/dt)}
df = cd.pseudo_time(cellDancer_df=df,
                    grid=(30, 30),
                    dt=dt,
                    t_total=t_total[dt],
                    n_repeats=n_repeats,
                    speed_up=(100, 100),
                    n_paths=3,
                    plot_long_trajs=True,
                    psrng_seeds_diffusion=[i for i in range(n_repeats)],
                    n_jobs=8)


# In[ ]:


df.to_csv(f"{data_path}/{dataset}_out.csv")


# In[ ]:


import matplotlib.pyplot as plt
import celldancer.cdplt as cdplt
fig, ax = plt.subplots(figsize=(8,6))
im=cdplt.scatter_cell(ax,
                      df,
                      colors='pseudotime',
                      alpha=0.5,
                      velocity=True)
ax.axis('off')
fig.savefig(f"{data_path}/cd_time_{dataset}.png")


# In[ ]:

# In[ ]:


time = df["pseudotime"].to_numpy().reshape(G, N).T

TLS.layers["cd_alpha"] = df["alpha"].to_numpy().reshape(G, N).T
TLS.layers["cd_beta"] = df["beta"].to_numpy().reshape(G, N).T
TLS.layers["cd_gamma"] = df["gamma"].to_numpy().reshape(G, N).T
TLS.layers["cd_velocity_u"] = (df["unsplice_predict"].to_numpy().reshape(G, N)
                                 - df["unsplice"].to_numpy().reshape(G, N)).T
TLS.layers["cd_velocity"] = (df["splice_predict"].to_numpy().reshape(G, N)
                               - df["splice"].to_numpy().reshape(G, N)).T
TLS.obs["cd_time"] = time[:, 0]

TLS.write_h5ad('/VeloVAE/celldancer/out/TLS_out.h5ad')


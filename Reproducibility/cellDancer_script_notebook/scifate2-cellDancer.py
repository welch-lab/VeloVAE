#!/usr/bin/env python
# coding: utf-8

# In[1]:


import celldancer as cd
import anndata

dataset = 'scifate2'
scifate2_path =  '/data/processed/scifate2.h5ad'
scifate2 = anndata.read_h5ad(scifate2_path)


# In[7]:


N, G = scifate2.shape


# In[10]:


data_path = '/VeloVAE/celldancer/data'



# In[12]:


df = cd.adata_to_df_with_embed(scifate2,
                               cell_type_para='cell_annotation',
                               embed_para='X_umap',
                               save_path=f"{data_path}/{dataset}.csv")


# In[ ]:


df_loss , df = cd.velocity(df,
                           max_epoches=200,
                           permutation_ratio=0.125,
                           n_jobs=8,
                           save_path=data_path)


# In[ ]:


#n_neigh = scifate2.uns['neighbors']['indices'].shape[1]
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


scifate2


# In[ ]:


time = df["pseudotime"].to_numpy().reshape(G, N).T

scifate2.layers["cd_alpha"] = df["alpha"].to_numpy().reshape(G, N).T
scifate2.layers["cd_beta"] = df["beta"].to_numpy().reshape(G, N).T
scifate2.layers["cd_gamma"] = df["gamma"].to_numpy().reshape(G, N).T
scifate2.layers["cd_velocity_u"] = (df["unsplice_predict"].to_numpy().reshape(G, N)
                                 - df["unsplice"].to_numpy().reshape(G, N)).T
scifate2.layers["cd_velocity"] = (df["splice_predict"].to_numpy().reshape(G, N)
                               - df["splice"].to_numpy().reshape(G, N)).T
scifate2.obs["cd_time"] = time[:, 0]

scifate2.write_h5ad('/VeloVAE/celldancer/out/scifate2_out.h5ad')


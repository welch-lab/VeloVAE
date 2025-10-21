#!/usr/bin/env python
# coding: utf-8

# In[1]:

# In[12]:

import anndata
import numpy as np
from deepvelo.utils import velocity, velocity_confidence, latent_time, update_dict
from deepvelo.utils.preprocess import autoset_coeff_s
from deepvelo.utils.plot import statplot, compare_plot
from deepvelo import train, Constants
import os
import time

# In[23]:


#TLS
TLS_path = '/VeloVAE/data/processed/TLS.h5ad'
# In[24]:

TLS = anndata.read_h5ad(TLS_path)

save_dir = f"/VeloVAE/deepvelo/output"
configs = {
    "name": "DeepVelo-TLS", # name of the experiment
    "arch":{"args":{"pred_unspliced":True}},
    "loss": {"args": {"coeff_s": autoset_coeff_s(TLS)}},
    "trainer": {"verbosity": 0, "save_dir": save_dir}, # increase verbosity to show training progress
}
configs = update_dict(Constants.default_configs, configs)


t_start = time.time()
velocity(TLS, mask_zero=False)
trainer = train(TLS, configs)
run_time = time.time() - t_start


latent_time(TLS)
TLS.obs['dv_time'] = TLS.obs['latent_time'].to_numpy()
del TLS.obs['latent_time']
TLS.uns['dv_run_time']=run_time
os.makedirs(save_dir, exist_ok=True)
TLS.write_h5ad('/VeloVAE/deepvelo/output/deepvelo_TLS.h5ad')
print(f'Total run time: {run_time}')







# In[ ]:





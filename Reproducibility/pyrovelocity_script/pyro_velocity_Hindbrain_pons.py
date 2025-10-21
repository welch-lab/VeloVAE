#!/usr/bin/env python
# coding: utf-8


# In[3]:


import scvelo as scv
import numpy as np
import torch
import matplotlib.pyplot as plt
import pyrovelocity as pv
from pyrovelocity.tasks.data import download_dataset
from pyrovelocity.tasks.preprocess import preprocess_dataset
from pyrovelocity.tasks.train import train_dataset,train_model
from pyrovelocity.plots._vector_fields import plot_mean_vector_field,plot_vector_field_uncertain
from pyrovelocity.plots._genes import plot_gene_ranking
from pyrovelocity.utils import mae_evaluate
from pyrovelocity.analysis.analyze import vector_field_uncertainty
import anndata
import time
Hindbrain_pons_adata_raw = anndata.read_h5ad('/data/scRNA/Hindbrain_pons/Hindbrain_pons.h5ad')

Hindbrain_pons_adata_processed = preprocess_dataset(data_set_name = 'Hindbrain_pons',
                                                    adata = Hindbrain_pons_adata_raw,
                                                    data_processed_path = '/data/external',
                                                    n_top_genes = 2000)

#model1
start=time.time()
result = train_dataset(Hindbrain_pons_adata_processed,
                       data_set_name = 'Hindbrain_pons',
                       offset=False,
                       guide_type='auto_t0_constraint',
                       patient_improve=1e-3,
                       max_epochs=4000,
                       #models_path = models_path,
                       )
end=time.time()
print(end-start)
#model2
start=time.time()
result = train_dataset(Hindbrain_pons_adata_processed,
                       data_set_name = 'Hindbrain_pons',
                       offset=True,
                       guide_type='auto',
                       patient_improve=1e-3,
                       max_epochs=4000,
                       #models_path = models_path,
                       )

end=time.time()
print(end-start)


# In[6]:





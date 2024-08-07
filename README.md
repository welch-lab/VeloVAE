# VeloVAE - Variational Mixtures of ODEs for Inferring Cellular Gene Expression Dynamics
# JJ fork
## Made fork to fix some issues with Scanpy/Anndata version
### Reference issue [#6](https://github.com/welch-lab/VeloVAE/issues/6)

Current error is that `evaluation_utils.py` calls for `adata.uns['neighbors']['indices']` but this structure is no longer retained in newer versions of Scanpy and Anndata. To address this issue, I made some adjustments to the code as the current output from `adata.uns['neighbors'].keys()` is 
```
dict_keys(['connectivities_key', 'distances_key', 'params'])
```
where the indices are called in the `connectivities_key`. I don't know what version of Scanpy, ScVelo, and Anndata could be causing the issue but I have introduced a temporary fix to the problem until the repository owners can fix it. 

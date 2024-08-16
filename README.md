# VeloVAE - Variational Mixtures of ODEs for Inferring Cellular Gene Expression Dynamics
## Introduction

The rapid growth of scRNA-seq data has spurred many advances in single-cell analysis. A key problem is to understand how gene expression changes during cell development. Recently, RNA velocity provided a new way to study gene expression from a dynamical system point of view. Because cells are destroyed during measurement, any scRNA-seq data are just one or very few static snapshots of mRNA count matrices. This makes it hard to learn the gene expression kinetics, as the time information is lost.

VeloVAE is a deep generative model for learning gene expression dynamics from scRNA-seq data. The purpose of our method is to infer latent cell time and velocity simultaneously. This is achieved by using variational Bayesian inference with neural networks. VeloVAE is based on the biochemical process of converting nascent mRNA molecules to mature ones via splicing. The generative model is constrained by a set of ordinary differential equations. VeloVAE is capable of handling more complex gene expression kinetics compared with previous methods.

## Package Usage
The package depends on several main-stream packages in computational biology and machine learning, including [scanpy](https://scanpy.readthedocs.io/en/stable/), [PyTorch](https://pytorch.org/), [scikit-learn](https://scikit-learn.org/stable/). We suggest using a GPU to accelerate the training process.

A sample jupyter notebook is available [here](notebooks/velovae_example.ipynb). Notice that the dataset in the example is from [scVelo](https://scvelo.readthedocs.io/), so you would need to install scVelo.

The package has not been submitted to PyPI yet, but you can download it and import the package locally by adding the path of the package to the system path:
```python
import sys
sys.path.append(< path to the package >)
import velovae
```
We also plan to make the package available on PyPI soon.

# JJ fork
## Made fork to fix some issues with Scanpy/Anndata version
### Reference issue [#6](https://github.com/welch-lab/VeloVAE/issues/6)

Current error is that `evaluation_utils.py` calls for `adata.uns['neighbors']['indices']` but this structure is no longer retained in newer versions of Scanpy and Anndata. To address this issue, I made some adjustments to the code as the current output from `adata.uns['neighbors'].keys()` is 
```
dict_keys(['connectivities_key', 'distances_key', 'params'])
```
where the indices are called in the `connectivities_key`. I don't know what version of Scanpy, ScVelo, and Anndata could be causing the issue but I have introduced a temporary fix to the problem until the repository owners can fix it. 

Introduction
============

The rapid growth of scRNA-seq data has spurred many advances in single-cell analysis.
A key problem is to understand how gene expression changes during cell development.
Recently, RNA velocity provided a new way to study gene expression from a dynamical
system point of view. Because cells are destroyed during measurement, any scRNA-seq
data are just one or very few static snapshots of mRNA count matrices. This makes it
hard to learn the gene expression kinetics, as the time information is lost.

``VeloVAE`` is a deep generative model for learning gene expression dynamics from scRNA-seq
data. The purpose of our method is to infer latent cell time and velocity simultaneously.
This is achieved by using variational Bayesian inference with neural networks. ``VeloVAE`` is
based on the biochemical process of converting nascent mRNA molecules to mature ones via
splicing. The generative model is constrained by a set of ordinary differential equations.
VeloVAE is capable of handling more complex gene expression kinetics compared with previous methods.

As a computational tool for RNA velocity inference and analysis, this python package contains model
training and evaluation modules. Once trained, a VAE model reconstructs the cell time and infers
rate parameters as specified in previous works (`La\ Manno, et al., Nature 2018 <https://www.nature.com/articles/s41586-018-0414-6>`_,
`Bergen et al., Nature Biotechnology 2020 <https://www.nature.com/articles/s41587-020-0591-3>`_).
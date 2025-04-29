from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='velovae',
    version='0.1.4',
    packages=find_packages(),
    author='Yichen Gu',
    author_email='gyichen@umich.edu',
    description='Bayesian inference of RNA Velocity',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'anndata>=0.8.0, <=0.9.1',
        'hnswlib>=0.6.2',
        'igraph>=0.10.4',
        'ipywidgets',
        'jupyter',
        'loess>=2.1.2',
        'matplotlib>=3.3.0, <=3.7.3',  # to make sure scvelo setting works
        'memory_profiler>=0.61.0',
        'numba>=0.41.0',
        'numpy>=1.17.0, <=1.23.5',  # higher versions have issues with scvelo due to creating a nested array with list of array with different lengths
        'pandas>=0.23.0, <=1.5.3',  # to make sure scvelo plot works
        'pynndescent>=0.5.7',
        'scanpy>=1.5.0',
        'scikit-learn>=0.20.0',
        'scipy>=1.13.0',
        'scvelo>=0.2.3, <=0.2.5',
        'seaborn>=0.10.0',
        'torch>=1.8.0',
        'tqdm<=4.62.3',
        'umap-learn>=0.3.10',
    ]
)

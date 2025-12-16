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
        'hnswlib>=0.6.2',
        'ipywidgets',
        'jupyter',
        'loess>=2.1.2',
        'memory_profiler>=0.61.0',
        'numpy<2',
        'pynndescent>=0.5.7',
        'scipy>=1.13.0',
        'scvelo>=0.3.3',
        'seaborn>=0.10.0',
        'torch>=1.8.0',
        'tqdm<=4.62.3',
        'igraph>=0.11,<1',
        'louvain',
        'leidenalg'
    ]
)

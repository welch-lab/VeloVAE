Examples
=============

Installation
*************
The package is available on PyPI. Users can install it using pip:
::

    pip install velovae

You can download the package in development mode by treating it as a local package:
::

    git clone https://github.com/welch-lab/VeloVAE.git

Then, you can import the package locally by adding the path of the package to the system path:
::

    import sys
    sys.path.append(< path to the package >)
    import velovae

Usage
*************
To train a VAE model, we need an input dataset in the form of an `AnnData <https://anndata.readthedocs.io/en/latest/index.html>`_
object. Any raw count matrix needs to be preprocessed as shown in an scVelo tutorial. This package has a basic preprocessing function,
but we suggest users considering scVelo for more comprehensive preprocessing functionality.
The following code block shows the major steps for using the tool:
::

    adata = anndata.read_h5ad('your_file_name.h5ad')
    n_gene = 2000
    vv.preprocess(adata, n_gene)  # use scvelo.pp alternatively
    vae = vv.VAE(adata, tmax=20, dim_z=5, device='cuda:0')
    config = {
    # You can change any hyperparameters here!
    }
    vae.train(adata, config=config)
    vae.save_model(model_path, 'encoder_vae', 'decoder_vae')
    vae.save_anndata(adata, 'vae', 'your_data_path', file_name='your_output_name.h5ad')

There are some hyper=parameters users can tune before training.
A notebook with illustrative examples can be found `here <https://github.com/welch-lab/VeloVAE/blob/master/notebooks/velovae_example.ipynb>`_.

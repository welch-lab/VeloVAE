import anndata
import scvelo as scv
import numpy as np
from deepvelo.utils import velocity, velocity_confidence, latent_time, update_dict
from deepvelo.utils.preprocess import autoset_coeff_s
from deepvelo.utils.plot import statplot, compare_plot
from deepvelo import train, Constants
import os
import argparse
import time


parser = argparse.ArgumentParser('hyperparam')
parser.add_argument('-i', '--input', type=str, default=None)
parser.add_argument('-o', '--out_folder', type=str, default="dvae")
parser.add_argument('-s', '--save', type=str, default=None)
parser.add_argument('-r', '--root_cell', type=str, default=None)
args = parser.parse_args()


def run_deepvelo(adata):
    # specific configs to overide the default configs
    configs = {
        "name": "DeepVelo", # name of the experiment
        "arch":{"args":{"pred_unspliced":True}},
        "loss": {"args": {"coeff_s": autoset_coeff_s(adata)}},
        "trainer": {"verbosity": 0, "save_dir":args.out_folder}, # increase verbosity to show training progress
    }
    configs = update_dict(Constants.default_configs, configs)
    
    t_start = time.time()
    velocity(adata, mask_zero=False)
    trainer = train(adata, configs)
    run_time = time.time() - t_start
    
    latent_time(adata)
    adata.obs['dv_time'] = adata.obs['latent_time'].to_numpy()
    del adata.obs['latent_time']
    adata.uns['dv_run_time']=run_time
    if(args.save is not None):
        adata.write_h5ad(f'{args.out_folder}/{args.save}')
    print(f'Total run time: {run_time}')
    print("---------------------------------------------------")

adata = anndata.read_h5ad(args.input)
os.makedirs(args.out_folder, exist_ok=True)
run_deepvelo(adata)

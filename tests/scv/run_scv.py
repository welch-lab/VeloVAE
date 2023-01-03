import anndata
import scvelo as scv
import numpy as np
import os
import argparse
import time

parser = argparse.ArgumentParser('hyperparam')
parser.add_argument('-i', '--input', type=str, default=None)
parser.add_argument('-o', '--out_folder', type=str, default="results")
parser.add_argument('-s', '--save', type=str, default=None)
args = parser.parse_args()

def run_scv(adata):
    t_start = time.time()
    scv.tl.recover_dynamics(adata)
    run_time = time.time() - t_start
    scv.tl.velocity(adata, mode='dynamical')
    scv.tl.velocity_graph(adata)
    scv.tl.latent_time(adata)
    if(args.save is not None):
        adata.uns['fit_run_time'] = run_time
        adata.write_h5ad(f'{args.out_folder}/{args.save}')
    
    print(f"Total run time: {run_time}")
    print("---------------------------------------------------")

adata = anndata.read_h5ad(args.input)
os.makedirs(args.out_folder, exist_ok=True)
run_scv(adata)

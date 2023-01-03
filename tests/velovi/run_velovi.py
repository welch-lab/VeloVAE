import anndata
import scvelo as scv
import scvi
import numpy as np
from velovi import preprocess_data, VELOVI
import torch
import os
import argparse
import time

parser = argparse.ArgumentParser('hyperparam')
parser.add_argument('-i', '--input', type=str, default=None)
parser.add_argument('-o', '--out_folder', type=str, default="dvae")
parser.add_argument('-s', '--save', type=str, default=None)
parser.add_argument('--num_samples', type=int, default=25)
parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--train_size', type=float, default=0.7)
args = parser.parse_args()

def add_velovi_outputs_to_adata(adata, vae):
    vae.module = vae.module.to(torch.device('cpu'))
    latent_time = vae.get_latent_time(n_samples=args.num_samples)
    velocities = vae.get_velocity(n_samples=args.num_samples, velo_statistic="mean")
    shat, uhat = vae.get_expression_fit(adata, n_samples=args.num_samples, return_mean=True)
    likelihood = vae.get_gene_likelihood(adata, n_samples=args.num_samples, return_mean=True)
    
    t = latent_time
    scaling = 20 / t.max(0)

    adata.layers["velovi_velocity"] = velocities / scaling
    #adata.layers["velovi_time"] = latent_time
    adata.layers["velovi_uhat"] = uhat
    adata.layers["velovi_shat"] = shat
    
    adata.obs["velovi_likelihood"] = likelihood.sum(1)
    adata.var["fit_likelihood"] = likelihood.mean(0)
    adata.var["velovi_alpha"] = vae.get_rates()["alpha"] / scaling
    adata.var["velovi_beta"] = vae.get_rates()["beta"] / scaling
    adata.var["velovi_gamma"] = vae.get_rates()["gamma"] / scaling
    adata.var["velovi_t_"] = (
        torch.nn.functional.softplus(vae.module.switch_time_unconstr)
        .detach()
        .cpu()
        .numpy()
    ) * scaling
    adata.layers["fit_t"] = latent_time.values * scaling[np.newaxis, :]
    adata.var['velovi_scaling'] = 1.0
    scv.tl.velocity_graph(adata, vkey="velovi_velocity")
    scv.tl.latent_time(adata, vkey="velovi_velocity")
    adata.obs['velovi_time'] = adata.obs['latent_time'].to_numpy()
    del adata.obs['latent_time']
    
    random_state = np.random.RandomState(seed=0)
    permutation = random_state.permutation(adata.n_obs)
    n_val = int(np.floor(adata.n_obs * 0.3))
    adata.uns['velovi_test_idx'] = permutation[:n_val]
    adata.uns['velovi_train_idx'] = permutation[n_val:]


def run_velovi(adata):
    adata = preprocess_data(adata)
    VELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
    
    t_start = time.time()
    vae = VELOVI(adata)
    scvi.settings.seed=2022
    
    vae.train(max_epochs=args.num_epochs, use_gpu='cuda:0', train_size=args.train_size)
    run_time = time.time() - t_start
    add_velovi_outputs_to_adata(adata, vae)
    
    if(args.save is not None):
        adata.uns['velovi_run_time'] = run_time
        adata.write_h5ad(f'{args.out_folder}/{args.save}')
    print(f'Total run time: {run_time}')
    print("---------------------------------------------------")

adata = anndata.read_h5ad(args.input)
os.makedirs(args.out_folder, exist_ok=True)
run_velovi(adata)

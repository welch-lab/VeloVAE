import unitvelo as utv
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
parser.add_argument('-r', '--root_cell', type=str, default=None)
parser.add_argument('-m', '--mode', type=str, default='unified')
parser.add_argument('--r2_adjust', action='store_true')
parser.add_argument('--assign_pos_u', action='store_true')
args = parser.parse_args()

def run_utv(adata):
    print(f'Input: {args.input}')
    #utv.utils.choose_mode(adata, 'leiden')
    t_start = time.time()
    velo_config = utv.config.Configuration()
    velo_config.R2_ADJUST = args.r2_adjust
    velo_config.IROOT = args.root_cell
    velo_config.FIT_OPTION = '2' if args.mode=='independent' else '1' 
    velo_config.ASSIGN_POS_U = args.assign_pos_u
    velo_config.GPU = 0
    
    label = 'clusters'
    #label = 'leiden'
    adata = utv.run_model(adata, label, config_file=velo_config)
    run_time = time.time() - t_start

    scv.pl.velocity_embedding_stream(adata, color=adata.uns['label'], title='', legend_loc='far right', dpi=100, save=f'{args.out_folder}/vel_utv.png')
    scv.tl.latent_time(adata, min_likelihood=None)
    scv.pl.scatter(adata, color="latent_time", cmap="plasma", save=f'{args.out_folder}/time_utv.png')
    
    if(args.mode=='independent'):
        adata.obs['utv_time'] = adata.obs['latent_time'].to_numpy()
    else:
        idx_t = np.where(~np.isnan(adata.layers['fit_t'][0]))[0][0]
        t = adata.layers['fit_t'][:,idx_t]
        adata.obs['utv_time'] = t
    adata.uns['utv_run_time'] = run_time
    if(args.save is not None):
        adata.write_h5ad(f'{args.out_folder}/{args.save}')
    print(f"Total run time: {run_time}")
    print("---------------------------------------------------")

adata = anndata.read_h5ad(args.input)
os.makedirs(args.out_folder, exist_ok=True)
run_utv(adata)

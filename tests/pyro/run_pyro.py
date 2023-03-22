import anndata
import scvelo as scv
import numpy as np
from scipy.stats import poisson
from pyrovelocity.api import train_model
from pyrovelocity.plot import plot_posterior_time, plot_gene_ranking,\
      vector_field_uncertainty, plot_vector_field_uncertain,\
      plot_mean_vector_field, project_grid_points,rainbowplot,denoised_umap,\
      us_rainbowplot, plot_arrow_examples
import os
import argparse
import time
import matplotlib.pyplot as plt
import torch
print(torch.__version__)
parser = argparse.ArgumentParser('hyperparam')
parser.add_argument('-i', '--input', type=str, default=None)
parser.add_argument('-o', '--out_folder', type=str, default="dvae")
parser.add_argument('-s', '--save', type=str, default=None)
parser.add_argument('-r', '--root_cell', type=str, default=None)
parser.add_argument('--num_epochs', type=int, default=4000)
parser.add_argument('--batch_size', type=int, default=-1)
parser.add_argument('--train_size', type=float, default=0.7)
parser.add_argument('--input_type', type=str, default='raw')
parser.add_argument('--embed', type=str, default='umap')
args = parser.parse_args()

def save_fitting(adata, pos_train, train_idx, pos_test=None, test_idx=None):
    split = not ((pos_test is None) or (test_idx is None))
    for key in pos_train:
        k,m,n = pos_train[key].shape
        if(m>1 and n>1):
            adata.layers[f'pv_{key}'] = np.zeros((adata.n_obs,n))
            adata.layers[f'pv_{key}'][train_idx] = pos_train[key].mean(0)
            if(split):
                adata.layers[f'pv_{key}'][test_idx] = pos_test[key].mean(0)
        elif(m>1):
            adata.obs[f'pv_{key}'] = np.zeros((adata.n_obs))
            adata.obs[f'pv_{key}'][train_idx] = pos_train[key].mean(0).squeeze()
            if(split):
                adata.obs[f'pv_{key}'][test_idx] = pos_test[key].mean(0).squeeze()
        elif(n>1):
            adata.var[f'pv_{key}'] = pos_train[key].mean(0).squeeze()
        else:
            adata.uns[f'pv_{key}'] = pos_train[key]
            

def run_pyro(adata):
    adata.layers['raw_spliced']   = adata.layers['spliced']
    adata.layers['raw_unspliced'] = adata.layers['unspliced']
    adata.obs['u_lib_size_raw'] = adata.layers['raw_unspliced'].toarray().sum(-1)
    adata.obs['s_lib_size_raw'] = adata.layers['raw_spliced'].toarray().sum(-1)
    likelihood = 'Poisson' if args.input_type=='raw' else 'Normal'
    
    batch_size=args.batch_size
    if(batch_size==-1 and adata.n_obs>4000):
        batch_size=256
    
    t_start = time.time()
    if(args.train_size<1.0):
        pos_train, pos_test, train_idx, test_idx = train_model(adata,
                                                             max_epochs=args.num_epochs, 
                                                             svi_train=True, 
                                                             log_every=100,
                                                             patient_init=45,
                                                             batch_size=batch_size, 
                                                             use_gpu=0, 
                                                             likelihood=likelihood,
                                                             input_type=args.input_type,
                                                             cell_state='clusters',
                                                             include_prior=True,
                                                             offset=False,
                                                             library_size=True,
                                                             patient_improve=1e-3,
                                                             seed=2022,
                                                             guide_type='auto_t0_constraint',
                                                             train_size=args.train_size)
        run_time = time.time()-t_start
        #Combine the results
        pos = {}
        print(train_idx.shape)
        
        for key in pos_train:
            n_sample = pos_train[key].shape[0]
            if(pos_train[key].shape[1]>1 and pos_train[key].shape[2]>1):
                val = np.zeros((n_sample, adata.n_obs, adata.n_vars))
                val[:,train_idx] = pos_train[key]
                val[:,test_idx] = pos_test[key]
                pos[key] = val
            elif(pos_train[key].shape[2]==1):
                val = np.zeros((n_sample, adata.n_obs, 1))
                val[:,train_idx] = pos_train[key]
                val[:,test_idx] = pos_test[key]
                pos[key] = val
            else:
                pos[key] = pos_train[key]
        
        #Compute velocity
        fig, ax = plt.subplots()
        embed_mean = plot_mean_vector_field(pos, adata, spliced='Ms', ax=ax, basis=args.embed, n_jobs=10)
        scv.pl.velocity_embedding_stream(adata, basis=args.embed, title='', vkey='velocity_pyro', save=f'{args.out_folder}/vel_stream_ms.png')
        fig, ax = plt.subplots()
        embed_mean = plot_mean_vector_field(pos, adata, ax=ax, basis=args.embed, n_jobs=10)
        scv.pl.velocity_embedding_stream(adata, basis=args.embed, title='', vkey='velocity_pyro', save=f'{args.out_folder}/vel_stream.png')
        
        adata.obsm[f"pv_velocity_{args.embed}"] = embed_mean
        del adata.obsm[f"velocity_pyro_{args.embed}"]
        adata.layers["pv_shat"] = adata.layers["spliced_pyro"]
        adata.layers["pv_uhat"] = np.zeros(adata.shape)
        adata.layers["pv_uhat"][train_idx] = pos_train["u"].mean(0)
        adata.layers["pv_uhat"][test_idx] = pos_test["u"].mean(0)
        del adata.layers["spliced_pyro"]
        adata.layers["pv_velocity"] = adata.layers["velocity_pyro"]
        del adata.layers["velocity_pyro"]
        
        adata.obs['pv_time'] = pos['cell_time'][:,:,0].mean(0).squeeze()
        adata.uns['pv_train_idx'] = train_idx
        adata.uns['pv_test_idx'] = test_idx
        save_fitting(adata, pos_train, train_idx, pos_test, test_idx)
        
        err_dic = {}
        mse_train, mse_test, mae_train, mae_test, logp_train, logp_test = 0,0,0,0,0,0
        
        if(args.input_type == 'raw'):
            for i in range(n_sample):
                dist_u_train = np.abs(adata.layers['unspliced'][train_idx].A-pos_train['u'][i])
                dist_s_train = np.abs(adata.layers['spliced'][train_idx].A-pos_train['s'][i])
                dist_u_test = np.abs(adata.layers['unspliced'][test_idx].A-pos_test['u'][i])
                dist_s_test = np.abs(adata.layers['spliced'][test_idx].A-pos_test['s'][i])
                mse_train += np.mean(dist_u_train**2+dist_s_train**2)
                mse_test += np.mean(dist_u_test**2+dist_s_test**2)
                mae_train += np.mean(dist_u_train+dist_s_train)
                mae_test += np.mean(dist_u_test+dist_s_test)
                logp_train += np.log(poisson.pmf(adata.layers['unspliced'][train_idx].A, pos_train['ut'][i])+1e-10).sum(1).mean()\
                             +np.log(poisson.pmf(adata.layers['unspliced'][train_idx].A, pos_train['st'][i])+1e-10).sum(1).mean()
                logp_test += np.log(poisson.pmf(adata.layers['unspliced'][test_idx].A, pos_test['ut'][i])+1e-10).sum(1).mean()\
                             +np.log(poisson.pmf(adata.layers['unspliced'][test_idx].A, pos_test['st'][i])+1e-10).sum(1).mean()
            adata.var['velocity_genes'] = True
            
        else:
            for i in range(pos['u'].shape[0]):
                dist_u_train = np.abs(adata.layers['Mu'][train_idx]-pos_train['u'][i])
                dist_s_train = np.abs(adata.layers['Ms'][train_idx]-pos_train['s'][i])
                dist_u_test = np.abs(adata.layers['Mu'][test_idx]-pos_test['u'][i])
                dist_s_test = np.abs(adata.layers['Ms'][test_idx]-pos_test['s'][i])
                mse_train += np.mean(dist_u_train**2+dist_s_train**2)
                mse_test += np.mean(dist_u_test**2+dist_s_test**2)
                mae_train += np.mean(dist_u_train+dist_s_train)
                mae_test += np.mean(dist_u_test+dist_s_test)
                #MLE of variance
                var_s_train, var_s_test = np.var(adata.layers['Mu'][train_idx]-pos_train['u'][i],0),np.var(adata.layers['Ms'][train_idx]-pos_train['s'][i],0)
                var_u_train, var_u_test = np.var(adata.layers['Mu'][test_idx]-pos_test['u'][i],0),np.var(adata.layers['Ms'][test_idx]-pos_test['s'][i],0)
                logp_train += -dist_u_train**2/(2*var_u_train)-dist_s_train**2/(2*var_s_train) \
                            - 0.5*np.log(var_u_train) - 0.5*np.log(var_s_train) - np.log(2*np.pi)
                logp_test += -dist_u_test**2/(2*var_u_test)-dist_s_test**2/(2*var_s_test) \
                            - 0.5*np.log(var_u_test) - 0.5*np.log(var_s_test) - np.log(2*np.pi)

                if ('u_scale' in pos) and ('s_scale' in pos):
                    scale = pos['u_scale'][i] / pos['s_scale'][i]
                else:
                    scale = 1
                
                ukey = 'ut' if 'ut' in pos else 'u'
                skey = 'st' if 'st' in pos else 's'
                if 'beta_k' in pos:
                    vs_train = pos_train[ukey][i] * pos_train['beta_k'][i] / scale - pos_train[skey][i] * pos_train['gamma_k'][i]
                    vs_test = pos_test[ukey][i] * pos_test['beta_k'][i] / scale - pos_test[skey][i] * pos_test['gamma_k'][i]
                else:
                    vs_train = pos_train['beta'][i] * pos_train[ukey][i] / scale - pos_train['gamma'][i] * pos_train[skey][i]
                    vs_test = pos_test['beta'][i] * pos_test[ukey][i] / scale - pos_test['gamma'][i] * pos_test[skey][i]
                #print(scale.shape, train_idx.shape, vs[train_idx].shape, vs_train.shape)
                vs[train_idx] = vs[train_idx]+vs_train
                vs[test_idx] = vs[test_idx]+vs_test
            
            adata.layers['pv_velocity'] = vs / n_sample
            adata.var['velocity_genes'] = True
        
        
        err_dic['MSE Train'] = mse_train / n_sample
        err_dic['MSE Test'] = mse_test / n_sample
        err_dic['MAE Train'] = mae_train / n_sample
        err_dic['MAE Test'] = mae_test / n_sample
        err_dic['LL Train'] = logp_train / n_sample
        err_dic['LL Test'] = logp_test / n_sample
        adata.uns['err'] = err_dic
    else:
        adata_model_pos = train_model(adata,
                                      max_epochs=args.num_epochs, 
                                      svi_train=True, 
                                      log_every=100,
                                      patient_init=45,
                                      batch_size=args.batch_size, 
                                      use_gpu=0, 
                                      likelihood=likelihood,
                                      input_type=args.input_type,
                                      cell_state='clusters',
                                      include_prior=True,
                                      offset=False,
                                      library_size=True,
                                      patient_improve=1e-4,
                                      seed=2022,
                                      guide_type='auto_t0_constraint',
                                      train_size=1.0)
        run_time = time.time()-t_start
        adata.obs['pv_time'] = adata_model_pos[1]['cell_time'][:,:,0].mean(0)
        fig, ax = plt.subplots()
        embed_mean = plot_mean_vector_field(adata_model_pos[1], adata, spliced='Ms', ax=ax, basis=args.embed, n_jobs=10)
        scv.pl.velocity_embedding_stream(adata, basis=args.embed, title='', vkey='velocity_pyro', save=f'{args.out_folder}/vel_stream_ms.png')
        fig, ax = plt.subplots()
        embed_mean = plot_mean_vector_field(adata_model_pos[1], adata, ax=ax, basis=args.embed, n_jobs=10)
        scv.pl.velocity_embedding_stream(adata, basis=args.embed, title='', vkey='velocity_pyro', save=f'{args.out_folder}/vel_stream.png')
        
        adata.obsm[f"pv_velocity_{args.embed}"] = embed_mean
        del adata.obsm[f"velocity_pyro_{args.embed}"]
        adata.layers["pv_shat"] = adata.layers["spliced_pyro"]
        del adata.layers["spliced_pyro"]
        adata.layers["pv_uhat"] = adata_model_pos[1]['u'].mean(0)
        adata.layers["pv_velocity"] = adata.layers["velocity_pyro"]
        del adata.layers["velocity_pyro"]
        save_fitting(adata, adata_model_pos[1], np.array(range(adata.n_obs)))
        
    if(args.save is not None):
        adata.uns['pv_run_time'] = run_time
        adata.write_h5ad(f'{args.out_folder}/{args.save}')
    print(f'Total run time: {run_time}')
    print("---------------------------------------------------")

adata = anndata.read_h5ad(args.input)
os.makedirs(args.out_folder, exist_ok=True)
run_pyro(adata)

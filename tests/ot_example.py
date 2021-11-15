import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata
import wot
import argparse

##  Argument Parsing    ##
parser = argparse.ArgumentParser('ot')
parser.add_argument('-k','--tkey', type=str, default="vanilla")
parser.add_argument('-b','--nbin', type=int, default=20)
parser.add_argument('-e','--epsilon', type=float, default=0.05)
parser.add_argument('-n','--niter', type=int, default=1)
parser.add_argument('-q','--quantile', type=float, default=0.01)
parser.add_argument('--lambda1', type=float, default=1.0)
parser.add_argument('--lambda2', type=float, default=50.0)
args = parser.parse_args()

def discretizeTime(adata, tkey, Nbin):
    if('day' in adata.obs):
        print("Warning: the key 'day' is already in obs")
    t = adata.obs[f'{tkey}_time'].to_numpy()
    tmin, tmax = t.min(), t.max()
    dt = (tmax-tmin)/Nbin
    pseudo_days = t // dt
    adata.obs['day'] = pseudo_days


def transportMap(adata, tkey, epsilon = 0.05, lambda1 = 1, lambda2 = 50, niter = 1, q = 0.01):
    cell_labels = adata.obs['clusters'].to_numpy()
    cell_types = np.unique(cell_labels)
    t = adata.obs[f'{tkey}_time'].to_numpy()
    pseudo_days = adata.obs['day'].to_numpy()
    P = np.zeros((len(cell_types), len(cell_types)))
    ot_model = wot.ot.OTModel(adata,epsilon = epsilon, lambda1 = lambda1, lambda2 = lambda2, growth_iters=niter) 
    for i, x in enumerate(cell_types): #child type
        mask = cell_labels==x
        t_lb = np.quantile(t[mask], q) #estimated transition time
        idx = np.argmin(np.abs(t-t_lb))   #find the nearest time bin
        d = pseudo_days[idx]
        if(d>0):
            tmap = ot_model.compute_transport_map(d-1,d)
            if(tmap is not None):
                Pi = tmap.X # cell x cell transport matrix
            
                #Sum the weights of each cell type
                cell_labels_1 = cell_labels[pseudo_days==d-1]
                cell_labels_2 = cell_labels[pseudo_days==d]
                for j, y in enumerate(cell_types): #parent
                    if(np.any(cell_labels_1==y) and np.any(cell_labels_2==x)):
                        P[i,j] = np.sum(Pi[cell_labels_1==y][:,cell_labels_2==x])
        if(P[i].sum()==0):
            P[i,i] = 1.0
        P[i] = P[i]/P[i].sum()
    print('***        Transition Probability:        ***')
    P_dic = {}
    for j,y in enumerate(cell_types):
        P_dic[y] = P[:,j]
    w_df = pd.DataFrame(P_dic, index=pd.Index(cell_types))
    with pd.option_context('display.max_rows', None, 
                               'display.max_columns', None, 
                               'display.precision', 3,
                               'display.chop_threshold',1e-3,
                               'display.width', None):
        
        print(w_df)

tkey='vanilla'
#adata = anndata.read_h5ad('../data/Pancreas/output.h5ad')
adata = anndata.read_h5ad('/scratch/blaauw_root/blaauw1/gyichen/output.h5ad')
discretizeTime(adata, args.tkey, args.nbin)
transportMap(adata, args.tkey, args.epsilon, args.lambda1, args.lambda2, args.niter, args.quantile)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata
import wot
import argparse
import sklearn
from velovae import optimal_transport_duality_gap

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
    tmin, tmax = t.min(), t.max()
    dt = (tmax-tmin)/args.nbin
    #pseudo_days = adata.obs['day'].to_numpy()
    P = np.zeros((len(cell_types), len(cell_types)))
    
    for i, x in enumerate(cell_types): #child type
        mask = cell_labels==x
        t0 = np.quantile(t[mask], q) #estimated transition time
        pseudo_days = np.zeros((len(t)))
        pseudo_days[(t>=t0-dt) & (t<t0)] = 1
        pseudo_days[(t>=t0) & (t<t0+dt)] = 2
        adata.obs['day'] = pseudo_days
        #Create ot model and call the ot algorithm
        ot_model = wot.ot.OTModel(adata,epsilon = epsilon, lambda1 = lambda1, lambda2 = lambda2, growth_iters=niter) 
        tmap = ot_model.compute_transport_map(1,2)
        if(tmap is not None):
            Pi = tmap.X # cell x cell transport matrix
        
            #Sum the weights of each cell type
            cell_labels_1 = cell_labels[pseudo_days==1]
            cell_labels_2 = cell_labels[pseudo_days==2]
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
                               'display.precision', 2,
                               'display.chop_threshold',1e-3,
                               'display.width', 200):
        
        print(w_df)

def transportMapCustom(adata, tkey, epsilon = 0.05, lambda1 = 1, lambda2 = 50, niter = 1, q = 0.01):
    cell_labels = adata.obs['clusters'].to_numpy()
    cell_types = np.unique(cell_labels)
    t = adata.obs[f'{tkey}_time'].to_numpy()
    tmin, tmax = t.min(), t.max()
    dt = (tmax-tmin)/args.nbin
    #pseudo_days = adata.obs['day'].to_numpy()
    P = np.zeros((len(cell_types), len(cell_types)))
    
    X_pca = adata.obsm["X_pca"]
    for i, x in enumerate(cell_types): #child type
        mask = cell_labels==x
        t0 = np.quantile(t[mask], q) #estimated transition time
        mask1 = (t>=t0-dt) & (t<t0)
        mask2 = (t>=t0) & (t<t0+dt)
        
        if(np.any(mask1) and np.any(mask2)):
            X1, X2 = X_pca[mask1], X_pca[mask2]
            C = sklearn.metrics.pairwise.pairwise_distances(X1,X2,metric='sqeuclidean', n_jobs=-1)
            C = C/np.median(C)
            G = np.ones((C.shape[0]))
            Pi = optimal_transport_duality_gap(C, G, lambda1, lambda2, epsilon, 5, 1e-8, 10000,
                                  1, 1000)
            
            #Sum the weights of each cell type
            cell_labels_1 = cell_labels[mask1]
            cell_labels_2 = cell_labels[mask2]
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
                               'display.precision', 2,
                               'display.chop_threshold',1e-3,
                               'display.width', 200):
        
        print(w_df)

#adata = anndata.read_h5ad('/home/gyichen/ODE_CellVelocity/ODE_CellVelocity/data/pancreas.h5ad')
adata = anndata.read_h5ad('data/Pancreas/output.h5ad')
#adata = anndata.read_h5ad('/scratch/blaauw_root/blaauw1/gyichen/output.h5ad')
#discretizeTime(adata, args.tkey, args.nbin)
transportMapCustom(adata, args.tkey, args.epsilon, args.lambda1, args.lambda2, args.niter, args.quantile)
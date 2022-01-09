import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata
import wot
import argparse
import sklearn
import velovae as vv
from velovae import optimal_transport_duality_gap, plotTLatent

import torch

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
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'lime', 'grey', \
   'olive', 'cyan', 'maroon', 'pink', 'gold', 'steelblue', 'salmon', 'teal', \
   'magenta', 'rosybrown', 'darkorange', 'yellow', 'greenyellow', 'darkseagreen', 'yellowgreen', 'palegreen', \
   'hotpink', 'navajowhite', 'aqua', 'navy', 'saddlebrown', 'black']

def discretizeTime(adata, tkey, Nbin):
    if('day' in adata.obs):
        print("Warning: the key 'day' is already in obs")
    t = adata.obs[f'{tkey}_time'].to_numpy()
    tmin, tmax = t.min(), t.max()
    dt = (tmax-tmin)/Nbin
    pseudo_days = t // dt
    adata.obs['day'] = pseudo_days

def compute_default_cost_matrix(a, b, eigenvals=None):

        if eigenvals is not None:
            a = a.dot(eigenvals)
            b = b.dot(eigenvals)

        cost_matrix = sklearn.metrics.pairwise.pairwise_distances(a, b, metric='sqeuclidean', n_jobs=-1)
        cost_matrix = cost_matrix / np.median(cost_matrix)
        return cost_matrix

def transportStats(adata, tkey, all_to_one=False, normalize=False, epsilon = 0.05, lambda1 = 1, lambda2 = 50, niter = 1, q = 0.01, **kwargs):
    cell_labels = adata.obs['clusters'].to_numpy()
    cell_types = np.unique(cell_labels)
    t = adata.obs[f'{tkey}_time'].to_numpy()
    tmin, tmax = t.min(), t.max()
    dt = (tmax-tmin)/args.nbin
    P = np.zeros((len(cell_types), len(cell_types)))
    R = np.zeros((len(cell_types), len(cell_types)))
    
    X = np.concatenate((adata.layers['Mu'], adata.layers['Ms']), 1)
    M = 50
    for i, x in enumerate(cell_types): #child type
        mask = cell_labels==x
        t0 = np.quantile(t[mask], q) #estimated transition time
        t1_sorted = np.sort(t[t<t0])
        if(all_to_one):
            t2_sorted = np.sort(t[(t>=t0) & (mask)])
        else:
            t2_sorted = np.sort(t[t>=t0])
        t_lb = t1_sorted[-M] if M<len(t1_sorted) else t1_sorted[0]
        t_ub = t2_sorted[M] if M<len(t2_sorted) else t2_sorted[-1]
        pseudo_days = np.zeros((len(t)))
        pseudo_days[(t>=t_lb) & (t<t0)] = 1
        pseudo_days[(t>=t0) & (t<=t_ub)] = 2  
        adata.obs['day'] = pseudo_days
        #Compute Cost Matrix
        #p0_x, p1_x, pca, mean = wot.ot.compute_pca(X[(t>=t0-dt) & (t<t0)], X[(t>=t0) & (t<t0+dt)], n_comp)
        #eigenvals = np.diag(pca.singular_values_)
        #Create ot model and call the ot algorithm
        ot_model = wot.ot.OTModel(adata,epsilon = epsilon, lambda1 = lambda1, lambda2 = lambda2, growth_iters=niter) 
        #C = compute_default_cost_matrix(p0_x, p1_x, eigenvals)
        tmap = ot_model.compute_transport_map(1,2)
        
        if(tmap is not None):
            Pi = tmap.X # cell x cell transport matrix
        
            #Sum the weights of each cell type
            cell_labels_1 = cell_labels[pseudo_days==1]
            cell_labels_2 = cell_labels[pseudo_days==2]
            for j, y in enumerate(cell_types): #parent
                if(np.any(cell_labels_1==y) and np.any(cell_labels_2==x)):
                    P[i,j] = np.sum(Pi[cell_labels_1==y]) if all_to_one else np.sum(Pi[cell_labels_1==y][:, cell_labels_2==x]) 
                    #R[i,j] = np.sum(Pi[cell_labels_1==y]*C[cell_labels_1==y]) if all_to_one else np.sum(Pi[cell_labels_1==y][:, cell_labels_2==x]*C[cell_labels_1==y][:, cell_labels_2==x]) 
                    if(normalize):
                        P[i,j] /= (len(cell_labels_1)*len(cell_labels_2))
                        #R[i,j] /= (np.sum(cell_labels_1==y)*np.sum(cell_labels_2==x))
    Ntype = len(cell_types)
    fig, ax = plt.subplots(figsize=(1.5*Ntype+3.0,0.9*Ntype+1.0))
    ax.bar(cell_types, P[:, 0], label=cell_types[0], color=colors[0])
    bottom = P[:, 0].copy()
    for i in range(1, Ntype):
        ax.bar(cell_types, P[:, i], bottom=bottom, label=cell_types[i], color=colors[i])
        bottom += P[:, i]
    ax.set_ylabel('Total Transport', fontsize=int(1.25*Ntype))
    ax.legend(bbox_to_anchor=(1.05,1.0), loc='upper left', fontsize=int(1.5*Ntype))
    ax.set_xticklabels(cell_types, rotation=45, fontsize=int(1.25*Ntype))
    plt.tight_layout()
    figname = kwargs.pop('figname', 'transport_bar')
    fig.savefig(f'figures/{figname}.png')
    """
    ig, ax = plt.subplots(figsize=(1.5*Ntype+2.0,0.9*Ntype+1.0))
    ax.bar(cell_types, R[:, 0], label=cell_types[0], color=colors[0])
    bottom = R[:, 0]
    for i in range(1, Ntype):
        ax.bar(cell_types, R[:, i], bottom=bottom, label=cell_types[i], color=colors[i])
        bottom += R[:, i]
    ax.set_ylabel('Total Cost')
    ax.legend(bbox_to_anchor=(1.05,1.0))
    ax.set_xticklabels(cell_types, rotation=45)
    plt.tight_layout()
    fig.savefig('figures/cost_bar.png')
    """
    
    #Normalize to transition probability
    for i in range(Ntype):
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
    return

def transportMapCustom(adata, tkey, epsilon = 0.05, lambda1 = 1, lambda2 = 50, niter = 1, q = 0.01, **kwargs):
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
        mask2 = (t>=t0) & (t<t0+dt) & mask
        
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
                    P[i,j] = np.sum(Pi[cell_labels_1==y])
    
    Ntype = len(cell_types)
    fig, ax = plt.subplots(figsize=(1.5*Ntype+3.0,0.9*Ntype+1.0))
    ax.bar(cell_types, P[:, 0], label=cell_types[0], color=colors[0])
    bottom = P[:, 0].copy()
    for i in range(1, Ntype):
        ax.bar(cell_types, P[:, i], bottom=bottom, label=cell_types[i], color=colors[i])
        bottom += P[:, i]
    ax.set_ylabel('Total Transport', fontsize=int(1.25*Ntype))
    ax.legend(bbox_to_anchor=(1.05,1.0), loc='upper left', fontsize=int(1.5*Ntype))
    ax.set_xticklabels(cell_types, rotation=45, fontsize=int(1.25*Ntype))
    plt.tight_layout()
    figname = kwargs.pop('figname', 'transport_bar')
    fig.savefig(f'figures/{figname}.png')
    
    
    
    #Normalize to transition probability
    for i in range(Ntype):
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

#adata = anndata.read_h5ad('/home/gyichen/ODE_CellVelocity/ODE_CellVelocity/data/dentategyrus.h5ad')
adata = anndata.read_h5ad('data/Dentategyrus/output.h5ad')
print(adata.X.shape)
#transportMap(adata, args.tkey, args.epsilon, args.lambda1, args.lambda2, args.niter, args.quantile)
#transportMapCustom(adata, args.tkey, args.epsilon, args.lambda1, args.lambda2, args.niter, args.quantile)
transportMapCustom(adata, args.tkey, args.epsilon, args.lambda1, args.lambda2, args.niter, args.quantile, figname='transport')
transportMapCustom(adata, args.tkey, args.epsilon, args.lambda1, args.lambda2, args.niter, args.quantile, figname='transport_all2one')
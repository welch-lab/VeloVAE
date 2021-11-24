import numpy as np
import anndata
import scvelo as scv
import argparse

##  Argument Parsing    ##
parser = argparse.ArgumentParser('test')
parser.add_argument('-i','--infile', type=str)
parser.add_argument('-o', '--outfile', type=str, default="output.h5ad")
parser.add_argument('--save_path', type=str, default='data/')
parser.add_argument('-g', '--Ngene', type=int, default=1000)
parser.add_argument('--Nplot', type=int, default=10)
parser.add_argument('--min_shared_counts', type=int, default=20)
parser.add_argument('--min_shared_cells', type=int, default=10)
parser.add_argument('--skip_pp', action='store_true')
args = parser.parse_args()

def run_scv(filename, Ngene, Nplot, min_shared_counts=args.min_shared_counts, min_shared_cells=args.min_shared_cells, skip_pp=False):
    adata = anndata.read_h5ad(filename)
    #Preprocessing
    if(not skip_pp):
        #1. Gene filtering and data normalization
        scv.pp.filter_and_normalize(adata, min_shared_counts=min_shared_counts, min_shared_cells = min_shared_cells,  n_top_genes=Ngene)
        #2. KNN Averaging
        scv.pp.moments(adata,n_pcs=30, n_neighbors=30)
        #3. Obtain cell clusters
        if(not 'clusters' in adata.obs):
            if('Class' in adata.obs):
                adata.obs['clusters'] = adata.obs['Class'].to_numpy()
            else:
                scanpy.tl.leiden(adata, key_added='clusters')
                print(np.unique(adata.obs['clusters'].to_numpy()))
    print(f'{adata.n_obs} Cells, {adata.n_vars} Genes.')
    
    #4. Compute Umap coordinates for visulization
    if(not 'X_umap' in adata.obsm):
        scv.tl.umap(adata)
    #Fit each gene
    scv.tl.recover_dynamics(adata)

    #Compute velocity, time and velocity graph (KNN graph based on velocity)
    scv.tl.velocity(adata, mode='dynamical')
    scv.tl.latent_time(adata)
    scv.tl.velocity_graph(adata, vkey="velocity", tkey="fit_t", gene_subset=adata.var_names[adata.var["velocity_genes"]])
    
    #Plotting
    top_genes = adata.var['fit_likelihood'].sort_values(ascending=False).index
    for i in range(Nplot):
        scv.pl.scatter(adata, basis=[top_genes[i]],linewidth=2.0,figsize=(12,8),add_assignments=True,save=f"{top_genes[i]}.png")
    
    scv.pl.velocity_embedding_stream(adata, basis='umap', vkey="velocity", save="vel-stream.png")
    scv.pl.scatter(adata, color='latent_time', color_map='gnuplot', size=80, colorbar=True, save="time.png")
    
    #Save the output
    adata.write_h5ad(args.save_path+args.outfile)
    
    scv.pl.scatter(adata, basis='X_umap', figsize=(10,10), save="class.png")
    return

def scvPlot(filename, genes):
    adata = anndata.read_h5ad(filename)
    
    scv.pl.scatter(adata, x='latent_time', y=genes, legend_loc='right margin', frameon=False, save="genes_global.png")
    scv.pl.scatter(adata, x='fit_t', y=genes, legend_loc='right margin', frameon=False, save="genes.png")
    #age = adata.obs['Age'].to_numpy()
    #tprior = np.array([float(x[1:]) for x in age])
    #adata.obs['tprior'] = tprior
    #scv.pl.scatter(adata, basis='X_umap', color='tprior', cmap='gnuplot', save='tprior.png')
    
    return


run_scv(args.infile, args.Ngene, args.Nplot, args.skip_pp)
#genes = ['Auts2', 'Dync1i1', 'Gm3764', 'Mapt', 'Nfib', 'Rbfox1', 'Satb2', 'Slc6a13', 'Srrm4', 'Tcf4']
#scvPlot(filename, genes)

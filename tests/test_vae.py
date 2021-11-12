import numpy as np
import scvelo as scv
import ..velovae as vv

def testVanillaVAE():
    adata = scv.datasets.pancreas()
    vv.preprocess(adata)
    model = vv.VanillaVAE(adata, Tmax)
    model.train(adata, config={'num_epochs':1}, gene_plot=['Cpe'], figure_path='./')
    model.saveModel('./')
    model.saveAnnData(self, './', file_name='pancreas.h5ad')
    
testVanillaVAE()
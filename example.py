import anndata
import numpy as np
import scvelo as scv
import velovae as vv
import torch
import torch.nn as nn
import argparse

##  Argument Parsing    ##
parser = argparse.ArgumentParser('test')
parser.add_argument('-t','--test_name', type=str, default="testVanillaVAE")
parser.add_argument('-d','--dataset',type=str)
parser.add_argument('-f','--file',type=str)
parser.add_argument('--save', type=str, default="")
args = parser.parse_args()

gene_plot = { 
            'Pancreas': ['Pcsk2','Dcdc2a','Gng12','Cpe','Smoc1','Tmem163','Ank'],
            'Dentategyrus': ['Ppp3ca','Ak5','Btbd9','Tmsb10','Hn1','Dlg2','Tcea1','Herc2'],
            'Brain10x': ['Ank3','Arpp21','Epha3','Grin2b','Grm5','Plxna4','Robo2','Tle4'],
            'Braindev': ['Auts2', 'Dync1i1', 'Gm3764', 'Mapt', 'Nfib', 'Rbfox1', 'Satb2', 'Slc6a13', 'Srrm4', 'Tcf4']
            }

def testVanillaVAE():
    dataset = args.dataset
    
    adata = anndata.read_h5ad(args.file)
    #vv.preprocess(adata, 1000)
    
    figure_path = f'figures/{dataset}/Default'
    pt_path = f'checkpoints/{dataset}/Default'
    file_path = args.save
    
    config_vae = {'num_epochs':800, 'test_epoch':50, 'save_epoch':50, 'learning_rate':2e-4, 'learning_rate_ode':2e-4, 'lambda':1e-3, 'neg_slope':0, 'reg_t':2.0, 'batch_size':1024}
    model = vv.VanillaVAE(adata, 20, hidden_size=(500, 250), tprior='tprior', device='gpu')
    model.train(adata, config=config_vae, gene_plot=gene_plot[dataset], figure_path=figure_path)
    model.saveModel(pt_path)
    model.saveAnnData(adata, 'vanilla', file_path, file_name='output.h5ad')

def testOTVAE():
    dataset = args.dataset
    
    adata = anndata.read_h5ad(args.file)
    #vv.preprocess(adata, 1000)
    
    figure_path = f'figures/{dataset}/OTVAE'
    pt_path = f'checkpoints/{dataset}/OTVAE'
    file_path = args.save
    
    config_vae = {'num_epochs':800, 'test_epoch':50, 'save_epoch':50, 'learning_rate':2e-4, 'learning_rate_ode':2e-4, 'lambda':1e-3, 'neg_slope':0, 'reg_t':2.0, 'batch_size':1024}
    model = vv.OTVAE(adata, 20, hidden_size=(500, 250), tprior=None, device='gpu')
    model.train(adata, config=config_vae, gene_plot=gene_plot[dataset], figure_path=figure_path)
    model.saveModel(pt_path)
    model.saveAnnData(adata, 'otvae', file_path, file_name='output.h5ad')
    
def testBranchingVAE():
    dataset = args.dataset
    adata = anndata.read_h5ad(args.file)
    #vv.preprocess(adata, 2000)
    
    figure_path = f'figures/{dataset}/BrVAE'
    pt_path = f'checkpoints/{dataset}/BrVAE'
    file_path = args.save
    
    
    config_vae = {'num_epochs':800, 'test_epoch':50, 'save_epoch':50, 'learning_rate':2e-4, 'learning_rate_ode':2e-4, 'lambda':3e-3, 'reg_t':10.0, 'batch_size':1024, 'yprior':True}
    model = vv.BranchingVAE(adata, "braindev", Cz=30, hidden_size=[(500,250), (500,250), (500,250)], tprior='tprior', device='gpu', tkey='vanilla')
    model.encoder.encoder_t.load_state_dict(torch.load(f'checkpoints/{dataset}/Default/encoder_vanilla.pt',map_location=model.device))
    model.train(adata, config=config_vae, gene_plot=gene_plot[dataset], figure_path=figure_path)
    model.saveModel(pt_path)
    model.saveAnnData(adata, 'bvae', file_path, file_name='braindev_part.h5ad')

def testMixtureVAE():
    dataset = args.dataset
    adata = anndata.read_h5ad(args.file)
    #vv.preprocess(adata, 2000)
    
    
    figure_path = f'figures/{dataset}/Default'
    pt_path = f'checkpoints/{dataset}/Default'
    file_path = args.save
    
    t_vanilla =adata.obs['vanilla_time'].to_numpy()
    config_vae = {'num_epochs':800, 'test_epoch':50, 'save_epoch':50, 
                  'learning_rate':2e-4, 'learning_rate_ode':2e-4, 'lambda':2e-3, 'neg_slope':0, 
                  'reg_t':2.0, 'reg_y':100.0, 'informative_y':True}
    model = vv.VAE(adata, Cz=30, hidden_size=[(500,250), (500,250), (500,250)], Tmax=t_vanilla.max(), device='gpu', tkey='vanilla')
    model.encoder.encoder_t.load_state_dict(torch.load(f'checkpoints/{dataset}/Default/encoder_vanilla.pt',map_location=model.device))
    model.debugW(adata)
    #model.train(adata, config=config_vae, gene_plot=gene_plot[dataset], figure_path=figure_path)
    #model.saveModel(pt_path)
    #model.saveAnnData(adata, 'mvae', file_path, file_name='output.h5ad')

def testBranchingVAETwoStage():
    dataset = args.dataset
    adata = anndata.read_h5ad(args.file)
    vv.preprocess(adata, 2000)
    
    figure_path = f'figures/{dataset}/Default'
    pt_path = f'checkpoints/{dataset}/Default'
    file_path = args.save
    
    config_vanilla = {'num_epochs':800, 'test_epoch':50, 'save_epoch':100, 'learning_rate':2e-4, 'learning_rate_ode':2e-4, 'lambda':1e-3}
    model_vanilla = vv.VanillaVAE(adata, Tmax=20, device='gpu')
    model_vanilla.train(adata, config=config_vanilla, gene_plot=gene_plot[dataset], figure_path=figure_path)
    model_vanilla.saveModel(pt_path)
    model_vanilla.saveAnnData(adata, 'vanilla', file_path, file_name='output.h5ad')
    
    
    config_vae = {'num_epochs':800, 'test_epoch':50, 'save_epoch':100, 'learning_rate':2e-4, 'learning_rate_ode':2e-4, 'lambda':1e-3, 'neg_slope':1e-4}
    model = vv.BranchingVAE(adata, "pancreas", Cz=24, device='gpu', tkey='vanilla')
    model.train(adata, config=config_vae, gene_plot=gene_plot[dataset], figure_path=figure_path)
    model.saveModel(pt_path)
    model.saveAnnData(adata, 'vae', file_path, file_name='output.h5ad')
    
def testVAETwoStage():
    dataset = args.dataset
    adata = anndata.read_h5ad(args.file)
    vv.preprocess(adata, 2000)
    
    figure_path = f'figures/{dataset}/Default'
    pt_path = f'checkpoints/{dataset}/Default'
    file_path = args.save
    
    config_vanilla = {'num_epochs':800, 'test_epoch':50, 'save_epoch':100, 'learning_rate':2e-4, 'learning_rate_ode':2e-4, 'neg_slope':0, 'lambda':1e-3}
    model_vanilla = vv.VanillaVAE(adata, Tmax=20, device='gpu', hidden_size=(500,250))
    model_vanilla.train(adata, config=config_vanilla, gene_plot=gene_plot[dataset], figure_path=figure_path)
    model_vanilla.saveModel(pt_path)
    model_vanilla.saveAnnData(adata, 'vanilla', file_path, file_name='output.h5ad')
    
    t_vanilla =adata.obs['vanilla_time'].to_numpy()
    config_vae = {'num_epochs':1000, 'test_epoch':50, 'save_epoch':50, 'learning_rate':2e-4, 'learning_rate_ode':2e-4, 'lambda':1e-3, 'neg_slope':1e-4, 'reg_t':2.0, 'reg_y':50.0, 'informative_y':True}
    model = vv.VAE(adata, Cz=24, hidden_size=[(500,250), (500,250), (500,250)], Tmax=t_vanilla.max(), device='gpu', tkey='vanilla')
    model.encoder.encoder_t.load_state_dict(torch.load('checkpoints/Pancreas/Default/encoder_vanilla.pt',map_location=model.device))
    
    model.train(adata, config=config_vae, gene_plot=gene_plot[dataset], figure_path=figure_path)
    model.saveModel(pt_path)
    model.saveAnnData(adata, 'mvae', file_path, file_name='output.h5ad')
    
    
def debugMixtureVAE():
    dataset = args.dataset
    adata = anndata.read_h5ad(args.file)
    vv.preprocess(adata, 2000)
    
    figure_path = f'figures/{dataset}/Default'
    pt_path = f'checkpoints/{dataset}/Default'
    file_path = args.save
    
    t_vanilla =adata.obs['vanilla_time'].to_numpy()
    config_vae = {'num_epochs':800, 'test_epoch':50, 'save_epoch':50, 
                  'learning_rate':2e-4, 'learning_rate_ode':2e-4, 'lambda':2e-3, 'neg_slope':1e-5, 'ct': 0.5,
                  'reg_t':2.0, 'reg_y':100.0, 'informative_y':True}
    model = vv.VAE(adata, Cz=30, hidden_size=[(500,250), (500,250), (500,250)], Tmax=t_vanilla.max(), device='gpu', tkey='vanilla')
    model.decoder.getTimeDistribution()

def testPlot():
    figure_path = 'figures/Pancreas/Post'
    adata = anndata.read_h5ad('data/Pancreas/output.h5ad')
    methods = ['vanilla']
    keys = ['vanilla']
    vv.postAnalysis(adata, methods, keys, gene_plot[dataset], plot_type=['all'], grid_size=(1,1), save_path=figure_path)
    return


if(args.test_name=='testVanillaVAE'):
    testVanillaVAE()
elif(args.test_name=='testOTVAE'):
    testOTVAE()
elif(args.test_name=='testBranchingVAE'):
    testBranchingVAE()
elif(args.test_name=='testMixtureVAE'):
    testMixtureVAE()
elif(args.test_name=='testBranchingVAETwoStage'):
    testBranchingVAETwoStage()
elif(args.test_name=='testVAETwoStage'):
    testVAETwoStage()
elif(args.test_name=='debugBrVAE'):
    debugBrVAE()
elif(args.test_name=='debugMixtureVAE'):
    debugMixtureVAE()
elif(args.test_name=='testPlot'):
    testPlot()

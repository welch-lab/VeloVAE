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
args = parser.parse_args()

gene_plot = { 
            'Pancreas': ['Pcsk2','Dcdc2a','Gng12','Cpe','Smoc1','Tmem163','Ank'],
            'Dentategyrus': ['Ppp3ca','Ak5','Btbd9','Tmsb10','Hn1','Dlg2','Tcea1','Herc2'],
            'Brain10x': ['Ank3','Arpp21','Epha3','Grin2b','Grm5','Plxna4','Robo2','Tle4'],
            'Braindev': ['Auts2', 'Dync1i1', 'Gm3764', 'Mapt', 'Nfib', 'Rbfox1', 'Satb2', 'Slc6a13', 'Srrm4', 'Tcf4']
            }

def testVanillaVAE():
    dataset = 'Braindev'
    
    adata = anndata.read_h5ad('/scratch/blaauw_root/blaauw1/gyichen/braindev_part.h5ad')
    #adata = anndata.read_h5ad(f'data/{dataset}/dentategyrus.h5ad')
    #vv.preprocess(adata, 1000)
    
    figure_path = f'figures/{dataset}/Default'
    pt_path = f'checkpoints/{dataset}/Default'
    file_path = '/scratch/blaauw_root/blaauw1/gyichen'
    #file_path = f'data/{dataset}/'
    
    config_vae = {'num_epochs':800, 'test_epoch':50, 'save_epoch':50, 'learning_rate':2e-4, 'learning_rate_ode':2e-4, 'lambda':1e-3, 'neg_slope':0, 'reg_t':10.0, 'batch_size':1024}
    model = vv.VanillaVAE(adata, 20, hidden_size=(500, 250), tprior='tprior', device='gpu')
    model.train(adata, config=config_vae, gene_plot=gene_plot[dataset], figure_path=figure_path)
    model.saveModel(pt_path)
    model.saveAnnData(adata, 'vanilla', file_path, file_name='output.h5ad')
    
def testBranchingVAE():
    dataset = 'Braindev'
    #adata = anndata.read_h5ad(f'data/{dataset}/output.h5ad')
    adata = anndata.read_h5ad('/scratch/blaauw_root/blaauw1/gyichen/braindev_part.h5ad')
    #vv.preprocess(adata, 2000)
    
    figure_path = f'figures/{dataset}/BrVAE'
    pt_path = f'checkpoints/{dataset}/BrVAE'
    #file_path = f'data/{dataset}/'
    file_path = '/scratch/blaauw_root/blaauw1/gyichen'
    
    
    config_vae = {'num_epochs':800, 'test_epoch':50, 'save_epoch':50, 'learning_rate':2e-4, 'learning_rate_ode':2e-4, 'lambda':3e-3, 'reg_t':10.0, 'batch_size':1024, 'yprior':True}
    model = vv.BranchingVAE(adata, "braindev", Cz=30, hidden_size=[(1000,500),(1000,500),(1000,500)], tprior='tprior', device='gpu', tkey='vanilla')
    model.encoder.encoder_t.load_state_dict(torch.load(f'checkpoints/{dataset}/Default/encoder_vanilla.pt',map_location=model.device))
    model.train(adata, config=config_vae, gene_plot=gene_plot[dataset], figure_path=figure_path)
    model.saveModel(pt_path)
    model.saveAnnData(adata, 'bvae', file_path, file_name='braindev_part.h5ad')

def testMixtureVAE():
    dataset = 'Pancreas'
    adata = anndata.read_h5ad(f'data/{dataset}/output.h5ad')
    #adata = anndata.read_h5ad('/scratch/blaauw_root/blaauw1/gyichen/braindev_part.h5ad')
    #vv.preprocess(adata, 2000)
    
    
    figure_path = f'figures/{dataset}/MixtureVAE2'
    pt_path = f'checkpoints/{dataset}/MixtureVAE2'
    file_path = f'data/{dataset}/'
    #file_path = '/scratch/blaauw_root/blaauw1/gyichen'
    
    t_vanilla =adata.obs['vanilla_time'].to_numpy()
    config_vae = {'num_epochs':800, 'test_epoch':50, 'save_epoch':50, 
                  'learning_rate':2e-4, 'learning_rate_ode':2e-4, 'lambda':2e-3, 'neg_slope':0, 
                  'reg_t':2.0, 'reg_y':100.0, 'informative_y':True}
    model = vv.VAE(adata, Cz=30, hidden_size=[(500,250), (500,250), (500,250)], Tmax=t_vanilla.max(), device='gpu', tkey='vanilla')
    #model.printWeight(model.decoder.tscore, model.decoder.xscore)
    model.encoder.encoder_t.load_state_dict(torch.load(f'checkpoints/{dataset}/Default/encoder_vanilla.pt',map_location=model.device))
    
    model.train(adata, config=config_vae, gene_plot=gene_plot[dataset], figure_path=figure_path)
    model.saveModel(pt_path)
    model.saveAnnData(adata, 'mvae', file_path, file_name='output.h5ad')

def testBranchingVAETwoStage():
    adata = anndata.read_h5ad('data/Pancreas/pancreas.h5ad')
    vv.preprocess(adata, 2000)
    
    figure_path = 'figures/Pancreas/BrVAETwoStage'
    pt_path = 'checkpoints/Pancreas/BrVAETwoStage'
    file_path = 'data/Pancreas/BrVAETwoStage'
    gene_plot = ['Pcsk2','Dcdc2a','Gng12','Cpe','Smoc1','Tmem163','Ank']
    
    config_vanilla = {'num_epochs':800, 'test_epoch':50, 'save_epoch':100, 'learning_rate':2e-4, 'learning_rate_ode':2e-4, 'lambda':1e-3}
    model_vanilla = vv.VanillaVAE(adata, Tmax=20, device='gpu')
    model_vanilla.train(adata, config=config_vanilla, gene_plot=gene_plot, figure_path=figure_path)
    model_vanilla.saveModel(pt_path)
    model_vanilla.saveAnnData(adata, 'vanilla', file_path, file_name='output.h5ad')
    
    
    config_vae = {'num_epochs':800, 'test_epoch':50, 'save_epoch':100, 'learning_rate':2e-4, 'learning_rate_ode':2e-4, 'lambda':1e-3, 'neg_slope':1e-4}
    model = vv.BranchingVAE(adata, "pancreas", Cz=24, device='gpu', tkey='vanilla')
    model.train(adata, config=config_vae, gene_plot=gene_plot, figure_path=figure_path)
    model.saveModel(pt_path)
    model.saveAnnData(adata, 'vae', file_path, file_name='output.h5ad')
    
def testVAETwoStage():
    adata = anndata.read_h5ad('data/Pancreas/pancreas.h5ad')
    vv.preprocess(adata, 2000)
    
    figure_path = 'figures/Pancreas/TwoStage'
    pt_path = 'checkpoints/Pancreas/TwoStage'
    file_path = 'data/Pancreas'
    gene_plot = ['Pcsk2','Dcdc2a','Gng12','Cpe','Smoc1','Tmem163','Ank']
    
    config_vanilla = {'num_epochs':800, 'test_epoch':50, 'save_epoch':100, 'learning_rate':2e-4, 'learning_rate_ode':2e-4, 'neg_slope':0, 'lambda':1e-3}
    model_vanilla = vv.VanillaVAE(adata, Tmax=20, device='gpu', hidden_size=(500,250))
    model_vanilla.train(adata, config=config_vanilla, gene_plot=gene_plot, figure_path=figure_path)
    model_vanilla.saveModel(pt_path)
    model_vanilla.saveAnnData(adata, 'vanilla', file_path, file_name='output.h5ad')
    
    t_vanilla =adata.obs['vanilla_time'].to_numpy()
    config_vae = {'num_epochs':1000, 'test_epoch':50, 'save_epoch':50, 'learning_rate':2e-4, 'learning_rate_ode':2e-4, 'lambda':1e-3, 'neg_slope':1e-4, 'reg_t':2.0, 'reg_y':50.0, 'informative_y':True}
    model = vv.VAE(adata, Cz=24, hidden_size=[(500,250), (500,250), (500,250)], Tmax=t_vanilla.max(), device='gpu', tkey='vanilla')
    model.encoder.encoder_t.load_state_dict(torch.load('checkpoints/Pancreas/Default/encoder_vanilla.pt',map_location=model.device))
    
    model.train(adata, config=config_vae, gene_plot=gene_plot, figure_path=figure_path)
    model.saveModel(pt_path)
    model.saveAnnData(adata, 'mvae', file_path, file_name='output.h5ad')

def debugBrVAE():
    adata = anndata.read_h5ad('data/Pancreas/pretrained.h5ad')
    U,S = adata.layers["Mu"], adata.layers["Ms"]
    
    figure_path = 'figures/Pancreas/Debug'
    pt_path = 'checkpoints/Pancreas/Debug'
    file_path = 'data/Pancreas/'
    gene_plot = ['Pcsk2','Dcdc2a','Gng12','Cpe','Smoc1']
    key='vae'
    
    model = vv.BranchingVAE(adata, "pancreas", Cz=30, hidden_size=[(2000,1000),(2000,1000),(2000,1000)], device='gpu')
    t = adata.obs["vae_t"].to_numpy()
    cell_labels_raw = adata.obs["clusters"].to_numpy()
    cell_types_raw = np.unique(cell_labels_raw)
    
    #Use the trained parameters
    model.decoder.alpha = nn.Parameter(torch.tensor(np.log(adata.varm[f"{key}_alpha"].T), device=model.device).double())
    model.decoder.beta = nn.Parameter(torch.tensor(np.log(adata.varm[f"{key}_beta"].T), device=model.device).double())
    model.decoder.gamma = nn.Parameter(torch.tensor(np.log(adata.varm[f"{key}_gamma"].T), device=model.device).double())
    
    #Transition time
    model.setMode('eval')
    transgraph = vv.model.TransGraph(cell_types_raw, 'pancreas')
    t_trans = adata.uns[f"{key}_t_trans"]
    ts = adata.varm[f"{key}_t_"].T
    model.decoder.t_trans = nn.Parameter(torch.tensor(np.log(t_trans), device=model.device).double())
    model.decoder.ts = nn.Parameter(torch.tensor(np.log(ts), device=model.device).double())
    t_trans_orig, ts_orig = model.decoder.recoverTransitionTime()
    
    
    #Initial conditions
    model.decoder.u0 = nn.Parameter(torch.tensor(np.log(adata.varm[f"{key}_u0"].T), device=model.device).double())
    model.decoder.s0 = nn.Parameter(torch.tensor(np.log(adata.varm[f"{key}_s0"].T), device=model.device).double())
    
    model.decoder.scaling = nn.Parameter(torch.tensor(np.log(adata.var[f"{key}_scaling"]), device=model.device).double())
    model.decoder.sigma_u = nn.Parameter(torch.tensor(np.log(adata.var[f"{key}_sigma_u"]), device=model.device).double())
    model.decoder.sigma_s = nn.Parameter(torch.tensor(np.log(adata.var[f"{key}_sigma_s"]), device=model.device).double())
    
    tdemo, ydemo, Uhat, Shat = model.decoder.forwardDemo(t.max(), 8, M=100)
    Uhat = Uhat.detach().cpu().numpy()
    Shat = Shat.detach().cpu().numpy()
    tdemo = tdemo.detach().cpu().numpy()
    ydemo = ydemo.detach().cpu().numpy()
    idx = np.where(adata.var_names=='Pcsk2')[0][0]
    
    vv.plotSig(t, 
                U[:,idx], S[:,idx], 
                Uhat[:,idx], Shat[:,idx], 'Pcsk2', 
                True, 
                figure_path, 
                'Pcsk2',
                cell_labels=cell_labels_raw,
                cell_types=cell_types_raw,
                labels_pred = cell_labels_raw,
                sparsify=2,
                t_trans=t_trans,
                ts=ts[:,idx],
                tdemo=tdemo,
                labels_demo = np.array([model.decoder.transgraph.label_dic_rev[x] for x in ydemo]))
    
    
def debugMixtureVAE():
    dataset = 'Pancreas'
    adata = anndata.read_h5ad(f'data/{dataset}/output.h5ad')
    #vv.preprocess(adata, 2000)
    
    figure_path = f'figures/{dataset}/MixtureVAE'
    pt_path = f'checkpoints/{dataset}/MixtureVAE'
    file_path = f'data/{dataset}/'
    gene_plot = { 
                'Pancreas': ['Pcsk2','Dcdc2a','Gng12','Cpe','Smoc1','Tmem163','Ank'],
                'Dentategyrus': ['Ppp3ca','Ak5','Btbd9','Tmsb10','Hn1','Dlg2','Tcea1','Herc2'],
                'Brain10x': ['Ank3','Arpp21','Epha3','Grin2b','Grm5','Plxna4','Robo2','Tle4']
                }
    
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
    gene_plot = ['Pcsk2','Cpe','Dcdc2a','Actn4','Gnao1','Nnat','Pak3','Ppp3ca','Gng12','Smoc1']
    vv.postAnalysis(adata, methods, keys, gene_plot, plot_type=['all'], grid_size=(1,1), save_path=figure_path)
    return


if(args.test_name=='testVanillaVAE'):
    testVanillaVAE()
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from sklearn.metrics import adjusted_rand_score
import time
from velovae.plotting import plotSig, plotTLatent, plotTrainLoss, plotTestLoss, plotCluster, plotLatentEmbedding

from .model_util import  convertTime, histEqual, initParams, getTsGlobal, transitionTime, reinitTypeParams, recoverTransitionTime, odeInitial, odeBranch, makeDir, getGeneIndex
from .VAE import encoder_part, ResBlock
from .TrainingData import SCLabeledData
from .TransitionGraph import TransGraph

############################################################
#KL Divergence
############################################################
def KLz(mu_zx, std_zx):
    """
    KL divergence between a Gaussian random vector with independent entries and a standard Normal random vector
    mu_zx, std_zx: [B x Cz]
    """
    B = mu_zx.shape[0]
    return -0.5 * torch.sum(1 + 2*torch.log(std_zx) - mu_zx.pow(2) - std_zx.pow(2))

def KLy(logit_y_tzx, pi_y):
    """
    KL divergence between two categorical distributions
    logit_y_tzx: [B x N type] posterior
    pi_y: [B x N type] prior
    """
    py_tzx = F.softmax(logit_y_tzx)
    if(pi_y.ndim==1):
        logpi_y = torch.log(pi_y).reshape(1,-1).repeat(py_tzx.shape[0],1)
    else:
        logpi_y = torch.log(pi_y)
    return F.kl_div(logpi_y, py_tzx, reduction='batchmean')

def KLGaussian(mu1, std1, mu2, std2):
    """
    Compute the KL divergence of two Gaussian distributions
    """
    return torch.mean(torch.sum(torch.log(std2/std1)+std1.pow(2)/(2*std2.pow(2))-0.5+(mu1-mu2).pow(2)/(2*std2.pow(2)),1))


############################################################
#Training Objective
############################################################
def VAERisk(u, s,
            uhat, shat, 
            mu_tx, std_tx, 
            mu_t, std_t,
            mu_zx, std_zx,
            mu_z, std_z,
            logit_y_tzx, pi_y, 
            sigma_u, sigma_s,
            weight=None):
    """
    This is the negative evidence lower bound.
    u , s : [B x G] input data
    uhat, shat: [B x G] prediction by the ODE model
    mu_tx, std_tx: [B x 1] encoder output, conditional Gaussian parameters
    mu_t, std_t: [1] Gaussian prior of time
    mu_zx, std_zx: [B x Cz] encoder output, conditional Gaussian parameters
    logit_y_tzx: [B x N type] type probability before softmax operation
    pi_y: cell type prior (conditioned on time and z)
    sigma_u, sigma_s : standard deviation of the Gaussian likelihood (decoder)
    weight: sample weight
    """
    assert not torch.any(torch.isnan(uhat))
    assert not torch.any(torch.isnan(shat))
    #KL divergence
    kld_t = KLGaussian(mu_tx, std_tx, mu_t, std_t)
    kld_z = KLGaussian(mu_zx, std_zx, mu_z, std_z)
    kld_y = KLy(logit_y_tzx, pi_y)
    
    log_gaussian = -(u-uhat).pow(2)/(2*sigma_u**2)-(s-shat).pow(2)/(2*sigma_s**2) - torch.log(sigma_u) - torch.log(sigma_s) - np.log(2*np.pi)
    if( weight is not None):
        log_gaussian = log_gaussian*weight
    err_rec = torch.mean(torch.sum(log_gaussian,1))
    return err_rec, kld_t, kld_z, kld_y



##############################################################
# BranchingVAE: 
#	Encoder learns the cell time and type
#
#	Decoder is a branching ODE
#	
##############################################################
class encoder(nn.Module):
    """
    Encoder Network
    Given the observation, it learns a latent representation z and cell time t.
    """
    def __init__(self, Cin, Ntype, Cz=None, hidden_size=[(1000, 500), (1000, 500), (1000, 500)], device=torch.device('cpu'), **kwargs):
        super(encoder, self).__init__()
        try:
            hidden_t, hidden_z, hidden_y = hidden_size[0], hidden_size[1], hidden_size[2]
        except IndexError:
            print(f'Expect hidden layer sizes of three networks, but got {len(hidden_size)} instead!')
            
        if(Cz is None):
            Cz = Ntype
        self.Cz = Cz
        
        #q(T | X)
        self.encoder_t = encoder_part(Cin, 1, hidden_t[0], hidden_t[1], device)

        #q(Z | X)
        self.encoder_z = encoder_part(Cin, Cz, hidden_z[0], hidden_z[1], device)
        
        #q(Y | Z,T,X)
        self.fw_Y_ZTX = ResBlock(Cz, hidden_y[0], hidden_y[1]).to(device)
        self.fc_yout = nn.Linear(hidden_y[1], Ntype).to(device)
        self.ky = torch.tensor([1e4]).double().to(device)
        
        if('checkpoint' in kwargs):
            self.load_state_dict(torch.load(kwargs['checkpoint'],map_location=device))
        else:
            self.init_weights()
        

    def init_weights(self):
        #Initialize forward blocks
        self.encoder_t.init_weights()
        self.encoder_z.init_weights()
        
        for m in self.fw_Y_ZTX.modules():
            if(isinstance(m, nn.Linear)):
                nn.init.xavier_uniform_(m.weight,np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
            elif(isinstance(m, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in [self.fc_yout]:
                nn.init.xavier_uniform_(m.weight,np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def reparameterize(self, mu, std):
        eps = torch.normal(mean=torch.zeros(mu.shape),std=torch.ones(mu.shape)).to(mu.device)
        return std*eps+mu

    def forward(self, data_in, scaling, t_trans, temp=1.0):
        data_in_scale = torch.cat((data_in[:,:data_in.shape[1]//2]/scaling, data_in[:,data_in.shape[1]//2:]),1).float()
        #q(T | X)
        mu_tx, std_tx = self.encoder_t.forward(data_in_scale)

        #q(Z | X)
        mu_zx, std_zx = self.encoder_z.forward(data_in_scale)

        #Sampling
        t = self.reparameterize(mu_tx, std_tx)
        z = self.reparameterize(mu_zx, std_zx)

        #q(Y | Z,T,X)
        logit_y = self.fc_yout(self.fw_Y_ZTX(z)) - self.ky*F.relu(t_trans - t)
        y = F.gumbel_softmax(logit_y, tau=temp, hard=True) 

        return mu_tx, std_tx, mu_zx, std_zx, logit_y, t, z, y


class decoder(nn.Module):
    """
    The ODE model that recovers the input data
    """
    def __init__(self, adata, Tmax, dataset_name=None, graph=None, init_types=None, device=torch.device('cpu'), p=98, tkey=None):
        super(decoder,self).__init__()
        
        U,S = adata.layers['Mu'], adata.layers['Ms']
        X = np.concatenate((U,S),1)
        cell_labels_raw = adata.obs["clusters"].to_numpy() if "clusters" in adata.obs else None
        cell_types_raw = np.unique(cell_labels_raw)
        
        
        #Transition Graph
        self.transgraph = TransGraph(cell_types_raw, dataset_name, graph, init_types)
        self.cell_labels = self.transgraph.str2int(cell_labels_raw)
        self.cell_types = np.unique(self.cell_labels)
        
        #Dynamical Model Parameters
        
        if(tkey is not None):
            print(f'[:: Decoder ::] Using pretrained time with key {tkey}')
            scaling = adata.var[f"{tkey}_scaling"].to_numpy()
            sigma_u = adata.var[f"{tkey}_sigma_u"].to_numpy()
            sigma_s = adata.var[f"{tkey}_sigma_s"].to_numpy()
            self.t_init = adata.obs[f"{tkey}_time"].to_numpy()
        else:
            alpha, beta, gamma, scaling, ts, u0, s0, sigma_u, sigma_s, T, Rscore = initParams(X,p,fit_scaling=True)
            T = T+np.random.rand(T.shape[0],T.shape[1]) * 1e-3
            T_eq = np.zeros(T.shape)
            Nbin = T.shape[0]//50+1
            for i in range(T.shape[1]):
                T_eq[:, i] = histEqual(T[:, i], Tmax, 0.9, Nbin)
            self.t_init = np.quantile(T_eq,0.5,1)
        
        
        t_trans, ts = transitionTime(self.t_init, self.cell_labels, self.cell_types, self.transgraph.graph, self.transgraph.init_types, adata.n_vars)
        alpha, beta, gamma, u0, s0 = reinitTypeParams(U/scaling, S, self.t_init, ts, self.cell_labels, self.cell_types, self.transgraph.init_types) #The parameters are from the scaled version!
            
        self.alpha = nn.Parameter(torch.tensor(np.log(alpha), device=device).float())
        self.beta = nn.Parameter(torch.tensor(np.log(beta), device=device).float())
        self.gamma = nn.Parameter(torch.tensor(np.log(gamma), device=device).float())
        
        self.scaling = nn.Parameter(torch.tensor(np.log(scaling), device=device).float())
        self.t_trans = nn.Parameter(torch.tensor(np.log(t_trans+1e-10), device=device).float())
        self.ts = nn.Parameter(torch.tensor(np.log(ts+1e-10), device=device).float())
        self.u0 = nn.Parameter(torch.tensor(np.log(u0), device=device).float())
        self.s0 = nn.Parameter(torch.tensor(np.log(s0), device=device).float())
        
        self.sigma_u = nn.Parameter(torch.tensor(np.log(sigma_u), device=device).float())
        self.sigma_s = nn.Parameter(torch.tensor(np.log(sigma_s), device=device).float())
        #self.Rscore = Rscore
        
        self.scaling.requires_grad = False
        self.sigma_u.requires_grad = False
        self.sigma_s.requires_grad = False
        
        
    def recoverTransitionTime(self):
        return recoverTransitionTime(torch.exp(self.t_trans), torch.exp(self.ts), self.transgraph.graph, self.transgraph.init_types) 
    
    def forward(self, t, p, train_mode=False, scale_back=True, neg_slope=1e-4):
        """
        t: [B x 1]
        p: [B x N_type]
        train_mode: determines whether to use leaky_relu 
        scale_back: whether to scale the output back to the original range
        """
        #Call branching ODE
        
        Uhat, Shat = odeBranch(t, self.transgraph.graph, self.transgraph.init_types,\
                               train_mode=train_mode,
                               neg_slope=neg_slope,
                               ptype=p,
                               alpha=torch.exp(self.alpha),
                               beta=torch.exp(self.beta),
                               gamma=torch.exp(self.gamma),
                               t_trans=torch.exp(self.t_trans),
                               ts=torch.exp(self.ts),
                               u0=torch.exp(self.u0),
                               s0=torch.exp(self.s0))
        if(scale_back):
            scaling = torch.exp(self.scaling)
            Uhat = Uhat*scaling
        
        return torch.sum(p.unsqueeze(-1)*Uhat,1), torch.sum(p.unsqueeze(-1)*Shat,1)
    
    def forwardDemo(self, Tmax, Ntype, M=100):
        t = torch.empty(Ntype*M, device=self.alpha.device).double()
        y = torch.empty(Ntype*M, device=self.alpha.device).long()
        t_trans_orig, ts_orig = self.recoverTransitionTime()
        t_trans_orig = t_trans_orig.detach().cpu().numpy()
        i = 0
        
        for i in self.transgraph.graph:
            tmin = t_trans_orig[i]
            if(len(self.transgraph.graph[i])>0):
                tmax = np.max([t_trans_orig[j] for j in self.transgraph.graph[i]])
            else:
                tmax = Tmax
            t[i*M:(i+1)*M] = torch.linspace(tmin, tmax, M)
            y[i*M:(i+1)*M] = i
        p = F.one_hot(y, Ntype)
        
        Uhat, Shat = odeBranch(t.view(-1,1), self.transgraph.graph, self.transgraph.init_types,\
                               train_mode=False,
                               neg_slope=0,
                               ptype=p,
                               alpha=torch.exp(self.alpha),
                               beta=torch.exp(self.beta),
                               gamma=torch.exp(self.gamma),
                               t_trans=torch.exp(self.t_trans),
                               ts=torch.exp(self.ts),
                               u0=torch.exp(self.u0),
                               s0=torch.exp(self.s0))
        scaling = torch.exp(self.scaling)
        Uhat = Uhat*scaling
        
        return t, y, torch.sum(p.unsqueeze(-1)*Uhat,1), torch.sum(p.unsqueeze(-1)*Shat,1)

class BranchingVAE():
    """
    The final VAE object containing all sub-modules
    """
    def __init__(self,
                 adata,
                 dataset_name,
                 graph=None, 
                 init_types=None, 
                 Cz=1,
                 hidden_size=[(1000,500),(1000,500),(1000,500)],
                 Tmax=20.0, 
                 tprior=None,
                 device='cpu',
                 train_scaling=False,
                 **kwargs):
        """
        adata: anndata object
        dataset_name: name of the dataset for loading the default transition graph. If set to None, the transition graph will be set to empty or customized values.
        graph: a dictionary containing the transition graph. Key is a parent node and the values are its children.
        init_types: an array of initial cell types (start of all lineages)
        Tmax: user-defined maximum time for the process
        pretrain: whether the VAE is in the pretraining mode
        fit_scaling: whether to include the scalings in the training parameters
        """
        try:
            U,S = adata.layers['Mu'], adata.layers['Ms']
            cell_labels_raw = adata.obs["clusters"].to_numpy()
        except KeyError:
            print('Please run the preprocessing step!')
        
        self.setDevice(device)
        
        N, G = adata.n_obs, adata.n_vars
        Ntype = len(np.unique(cell_labels_raw))
        self.Ntype = Ntype
        self.encoder = encoder(2*G, Ntype, Cz, device=self.device, **kwargs).float()
        if('tkey' in kwargs):
            tkey = kwargs['tkey']
        else:
            tkey = None
        self.decoder = decoder(adata, Tmax, dataset_name, graph, init_types, device=self.device, tkey=tkey)
        self.Tmax=torch.tensor(Tmax,dtype=torch.float).to(self.device)
        
        #Prior distribution
        if(tprior is None):
            self.mu_t = (torch.ones(U.shape[0],1)*Tmax*0.5).double().to(self.device)
            self.std_t = (torch.ones(U.shape[0],1)*Tmax*0.18).double().to(self.device)
        else:
            t = adata.obs[tprior].to_numpy()
            t = t/t.max()*Tmax
            self.mu_t = torch.tensor(t).view(-1,1).double().to(self.device)
            self.std_t = (torch.ones(self.mu_t.shape)*(np.std(t)/Ntype)).double().to(self.device)

        self.mu_z = torch.zeros(Cz).float().to(self.device)
        self.std_z = torch.ones(Cz).float().to(self.device)
        
        self.prior_y_default = (torch.ones(Ntype)*(1/Ntype)).float().to(self.device)
        
        #Training Configuration
        self.config = {
            "num_epochs":500, "learning_rate":1e-4, "learning_rate_ode":1e-4, "lambda":1e-3,\
            "reg_t":1.0, "reg_z":2.0, "reg_y":2.0, "neg_slope":0, \
            "test_epoch":100, "save_epoch":100, "batch_size":128, "K_alt":0,\
            "train_scaling":False, "train_std":False, "weight_sample":False, "anneal":True, "yprior":False,
        }
    
    def setDevice(self, device, device_number=None):
        if(device=='gpu'):
            if(torch.cuda.is_available()):
                self.device_number = device_number if (isinstance(device_number,int)) else torch.cuda.current_device()
                self.device = torch.device('cuda:'+str(self.device_number))
            else:
                print('Warning: GPU not detected. Using CPU as the device.')
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')
    
    
    def getInformativePriorY(self, label_batch, delta=0.001):
        label_batch = label_batch.squeeze()
        py = F.one_hot(label_batch, self.Ntype)+delta
        for i in range(self.Ntype):
            py[label_batch==i, i] -= (self.Ntype+1)*delta
        return py
    
    def reparameterize(self, mu, std, B):
        eps = torch.normal(mean=0.0, std=1.0, size=(B,1)).float().to(self.device)
        return std*eps+mu
    

    def forward(self, data_in, train_mode, scale_back=True, temp=1.0):
        scaling = torch.exp(self.decoder.scaling)
        t_trans_orig, ts_orig = self.decoder.recoverTransitionTime()
        mu_tx, std_tx, mu_zx, std_zx, logit_y, t, z, y = self.encoder.forward(data_in, scaling, t_trans_orig, temp=temp)

        uhat, shat = self.decoder.forward(t, y, train_mode, scale_back, neg_slope=self.config['neg_slope'])
        return mu_tx, std_tx, mu_zx, std_zx, logit_y, t, y, uhat, shat
        

    def evalModel(self, data_in, scale_back=True):
        """
        Original scale, used for testing and plotting
        """
        scaling = torch.exp(self.decoder.scaling)
        t_trans_orig, ts_orig = self.decoder.recoverTransitionTime()
        mu_tx, std_tx, mu_zx, std_zx, logit_y, t, z, y = self.encoder.forward(data_in, scaling, t_trans_orig, temp=1e-4)
        
        #Determine cell type
        py_tzx = F.softmax(logit_y)
        labels = torch.argmax(logit_y,1)
        p_type_onehot = F.one_hot(labels, self.Ntype).float()
        
        #Call the decoder
        uhat, shat = self.decoder.forward(mu_tx, p_type_onehot, False, scale_back)

        return mu_tx, std_tx, mu_zx, std_zx, py_tzx, z, uhat, shat
    
    def setMode(self,mode):
        if(mode=='train'):
            self.encoder.train()
            self.decoder.train()
        elif(mode=='eval'):
            self.encoder.eval()
            self.decoder.eval()
        else:
            print("Warning: mode not recognized. Must be 'train' or  'test'! ")
    
    def train_epoch(self, 
                    X_loader, 
                    optimizer, 
                    optimizer2, 
                    counter, 
                    anneal=True, 
                    K=2, 
                    reg_t=1.0, 
                    reg_z=2.0, 
                    reg_y=2.0):
        """
        Training in each epoch
        X_loader: Data loader of the input data
        optimizer
        """
        self.setMode('train')
        iterX = iter(X_loader)
        B = len(iterX)
        loss_list = []
        Nupdate = 100
        for i in range(B):
            tau = np.clip(np.exp(-3e-5*((counter+i+1)//Nupdate*Nupdate)), 0.5, None) if anneal else 0.5
            optimizer.zero_grad()
            if(optimizer2 is not None):
                optimizer2.zero_grad()
            batch = iterX.next()
            xbatch, label_batch, weight, idx = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device), batch[3].to(self.device)
            mu_tx, std_tx, mu_zx, std_zx, logit_y, t, y, uhat, shat= self.forward(xbatch, train_mode=True, scale_back=True, temp=tau)
            u, s = xbatch[:,:xbatch.shape[1]//2],xbatch[:,xbatch.shape[1]//2:]
            if(self.config["yprior"]):
                py = self.getInformativePriorY(label_batch)
            else:
                py = self.prior_y_default
            err_rec, kld_t, kld_z, kld_y = VAERisk(u, s, 
                                                   uhat, shat, 
                                                   mu_tx, std_tx,
                                                   self.mu_t[idx], self.std_t[idx],
                                                   mu_zx, std_zx,
                                                   self.mu_z, self.std_z,
                                                   logit_y, py,
                                                   torch.exp(self.decoder.sigma_u), torch.exp(self.decoder.sigma_s), 
                                                   weight=None)
            loss = - err_rec + reg_t * kld_t + reg_z * kld_z + reg_y * kld_y
            loss_list.append(loss.detach().cpu().item())
            loss.backward()
            optimizer.step()
            if( optimizer2 is not None and ((i+1) % K == 0 or i==B-1)):
                optimizer2.step()
        return loss_list, counter+B, tau
        
    def train(self, 
              adata, 
              config={}, 
              plot=True, 
              gene_plot=[], 
              figure_path="figures", 
              embed="umap"):
        #We don't have to specify all the hyperparameters. Just pass the ones we want to modify.
        for key in config:
            if(key in self.config):
                self.config[key] = config[key]
            else:
                print(f"Warning: unknown hyperparameter: {key}")
        if(self.config["train_scaling"]):
            self.decoder.scaling.requires_grad = True
        if(self.config["train_std"]):
            self.decoder.sigma_u.requires_grad = True
            self.decoder.sigma_s.requires_grad = True
        
        print("------------------------- Train a Branching VAE -------------------------")
        #Get data loader
        U,S = adata.layers['Mu'], adata.layers['Ms']
        X = np.concatenate((U,S), 1)
        try:
            Xembed = adata.obsm[f"X_{embed}"]
        except KeyError:
            print("Embedding not found! Please run the corresponding preprocessing step!")
        
        cell_labels_raw = adata.obs["clusters"].to_numpy() if "clusters" in adata.obs else np.array(['Unknown' for i in range(adata.n_obs)])
        cell_types_raw = np.unique(cell_labels_raw)
        cell_labels = self.decoder.transgraph.str2int(cell_labels_raw)
        print("***      Creating  Dataset      ***")
        dataset = SCLabeledData(X, cell_labels, Rscore) if self.config['weight_sample'] else SCLabeledData(X, cell_labels)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True)
        print("***          Finished.          ***")
        
        gind, gene_plot = getGeneIndex(adata.var_names, gene_plot)
        
        if(plot):
            makeDir(figure_path)
            #Plot the initial estimate
            plotTLatent(self.decoder.t_init, Xembed, f"Initial Estimate", plot, figure_path, f"init-BrVAE")
            
            t_demo, y_demo, Uhat_init, Shat_init = self.decoder.forwardDemo(self.decoder.t_init.max(), len(self.decoder.cell_types))
            y_demo_str = self.decoder.transgraph.int2str(y_demo.detach().cpu().numpy())
            for i in range(len(gind)):
                idx = gind[i]
                track_idx = None
                plotSig(self.decoder.t_init, 
                        U[:,idx], S[:,idx], 
                        Uhat_init[:,idx].detach().cpu().numpy(), Shat_init[:,idx].detach().cpu().numpy(), 
                        gene_plot[i], 
                        True, 
                        figure_path, 
                        f"{gene_plot[i]}-init",
                        cell_labels=cell_labels_raw,
                        sparsify=3,
                        tdemo = t_demo,
                        labels_pred = cell_labels_raw,
                        labels_demo = y_demo_str,)
        
    
        #define optimizer
        print("***    Creating  optimizers     ***")
        learning_rate = self.config["learning_rate"]
        param_nn = list(self.encoder.parameters())
        param_ode = [self.decoder.alpha, self.decoder.beta, self.decoder.gamma, \
                    self.decoder.u0, self.decoder.s0,\
                    self.decoder.ts, self.decoder.t_trans] 
        if(self.config["train_scaling"]):
            param_ode = param_ode+[self.decoder.scaling]
        if(self.config["train_std"]):
            param_ode = param_ode+[self.decoder.sigma_u, self.decoder.sigma_s]
    
        optimizer = torch.optim.Adam(param_nn, lr=self.config["learning_rate"], weight_decay=self.config["lambda"])
        optimizer_ode = torch.optim.Adam(param_ode, lr=self.config["learning_rate_ode"])
        print('***           Finished.         ***')
    
        #Optionally load model parameters
        anneal=True
        tau=1.0
        counter = 0
        
        #Main Training Process
        print("***        Start training       ***")
        print(f"Total Number of Iterations Per Epoch: {len(data_loader)}")
        n_epochs, n_save = self.config["num_epochs"], self.config["save_epoch"]
        loss_train, loss_test, err_test, randidx_test = [],[],[],[]
        
        
        start = time.time()
        for epoch in range(n_epochs):
            #Optimize the encoder
            if(self.config["K_alt"]==0):
                loss_list, counter, tau = self.train_epoch(data_loader, optimizer, None, counter, self.config["anneal"], self.config["K_alt"], self.config["reg_t"], self.config["reg_z"], self.config["reg_y"])
                loss_train = loss_train+loss_list
                loss_list, _, tau = self.train_epoch(data_loader, optimizer_ode, None, counter, self.config["anneal"], self.config["K_alt"], self.config["reg_t"], self.config["reg_z"], self.config["reg_y"])
                loss_train = loss_train+loss_list
            else:
                loss_list, counter, tau = self.train_epoch(data_loader, optimizer, optimizer_ode, counter, anneal, self.config["K_alt"], self.config["reg_t"], self.config["reg_z"], self.config["reg_y"])
                loss_train = loss_train+loss_list
            
            if(epoch==0 or (epoch+1) % self.config["test_epoch"] == 0):
                print(f'temperature = {tau}')
                save = (epoch+1)%n_save==0 or epoch==0
                loss, t, err, rand_idx = self.test(torch.tensor(X).float().to(self.device),
                                                   Xembed,
                                                   cell_labels,
                                                   epoch+1, 
                                                   gind, 
                                                   gene_plot,
                                                   save, 
                                                   figure_path,
                                                   cell_labels_raw)
                    
                print(f"Epoch {epoch+1}: Train Loss = {loss_train[-1]:.2f}, Test Loss = {loss:.2f}, err_type = {err:.3f}, rand index = {rand_idx:.3f} \t Total Time = {convertTime(time.time()-start)}")
                loss_test.append(loss)
                err_test.append(err)
                randidx_test.append(rand_idx)
        
        print("***     Finished Training     ***")
        plotTrainLoss(loss_train,[i for i in range(len(loss_train))],True,figure_path,"vae")
        plotTestLoss(loss_test,[1]+[i*self.config["test_epoch"] for i in range(1,len(loss_test))],True,figure_path,"vae")
        plotTestLoss(err_test,[1]+[i*self.config["test_epoch"] for i in range(1,len(err_test))],True,figure_path,"err-vae")
        plotTestLoss(randidx_test,[1]+[i*self.config["test_epoch"] for i in range(1,len(randidx_test))],True,figure_path,"rand_idx-vae")
        return
    
    def predAll(self, data):
        N, G = data.shape[0], data.shape[1]//2
        Uhat, Shat = torch.empty(N,G), torch.empty(N,G)
        t = torch.empty(N)
        y = torch.empty(N)
        z = torch.empty(N, self.encoder.Cz)
        qy = torch.empty(N, self.Ntype)
        
        with torch.no_grad():
            B = self.config["batch_size"]
            Nb = N // B
            for i in range(Nb):
                mu_tx, std_tx, mu_zx, std_zx, py_tzx, z_batch, uhat, shat = self.evalModel(data[i*B:(i+1)*B])
                Uhat[i*B:(i+1)*B] = uhat.cpu()
                Shat[i*B:(i+1)*B] = shat.cpu()
                t[i*B:(i+1)*B] = mu_tx.cpu().squeeze()
                qy[i*B:(i+1)*B] = py_tzx.cpu()
                y[i*B:(i+1)*B] = torch.argmax(py_tzx, 1).cpu()
                z[i*B:(i+1)*B] = z_batch.cpu()
            if(N > B*Nb):
                mu_tx, std_tx, mu_zx, std_zx, py_tzx, z_batch, uhat, shat = self.evalModel(data[B*Nb:])
                Uhat[Nb*B:] = uhat.cpu()
                Shat[Nb*B:] = shat.cpu()
                t[Nb*B:] = mu_tx.cpu().squeeze()
                qy[Nb*B:] = py_tzx.cpu()
                y[Nb*B:] = torch.argmax(py_tzx, 1).cpu()
                z[Nb*B:] = mu_zx.cpu()
        
        return Uhat, Shat, t, qy, y, z
    
    def test(self,
             data, 
             Xembed,
             cell_labels, 
             testid=0, 
             gind=None,
             gene_plot=[],
             plot=False,
             path='figures', 
             cell_labels_raw=None,
             **kwargs):
        """
        Validation and Plotting
        data: ncell x ngene tensor
        """
        self.setMode('eval')
        Uhat, Shat, t, qy, y, z = self.predAll(data)
        t = t.numpy()
        qy = qy.numpy()
        y = y.numpy()
        z = z.numpy()
        
        U,S = data[:,:data.shape[1]//2], data[:,data.shape[1]//2:]
        U,S = U.detach().cpu().numpy(), S.detach().cpu().numpy()
        mse = np.mean((Uhat.detach().cpu().numpy()-U)**2+(Shat.detach().cpu().numpy()-S)**2)
    
        t_trans, ts = self.decoder.recoverTransitionTime()
        
        err = 1.0 - np.sum(y==cell_labels)/len(cell_labels)
        rand_idx = adjusted_rand_score(cell_labels, y)
        if(plot):
            Uhat, Shat = Uhat.detach().cpu().numpy(), Shat.detach().cpu().numpy()
            plotCluster(Xembed,
                        qy,
                        cell_labels_raw,
                        False,
                        path,
                        f"{testid}")
            plotLatentEmbedding(z, 
                                self.Ntype, 
                                y,
                                self.decoder.transgraph.label_dic_rev,
                                path=path,
                                figname=f"{testid}_yhat")
            plotLatentEmbedding(z, 
                                self.Ntype, 
                                self.decoder.cell_labels,
                                self.decoder.transgraph.label_dic_rev,
                                path=path,
                                figname=f"{testid}_y")
            plotTLatent(t, Xembed, f"Training Epoch {testid}", plot, path, f"{testid}")
            for i in range(len(gene_plot)):
                idx = gind[i]
                plotSig(t.squeeze(), 
                        U[:,idx], S[:,idx], 
                        Uhat[:,idx], Shat[:,idx], gene_plot[i], 
                        True, 
                        path, 
                        f"{gene_plot[i]}-{testid}",
                        cell_labels=cell_labels_raw,
                        labels_pred = self.decoder.transgraph.int2str(y),
                        sparsify=3)
                
        return mse, t.squeeze(), err, rand_idx
    
    
    def saveModel(self, file_path, enc_name='encoder', dec_name='decoder'):
        """
        Save the encoder parameters to a .pt file.
        Save the decoder parameters to the anndata object.
        """
        makeDir(file_path)
        torch.save(self.encoder.state_dict(), f"{file_path}/{enc_name}.pt")
        torch.save(self.decoder.state_dict(), f"{file_path}/{dec_name}.pt")
    
    def saveAnnData(self, adata, key, file_path, file_name=None):
        """
        Save the ODE parameters and cell time to the anndata object and write it to disk.
        """
        makeDir(file_path)
        
        U,S = adata.layers['Mu'], adata.layers['Ms']
        X = np.concatenate((U,S), 1)
        
        adata.varm[f"{key}_alpha"] = np.exp(self.decoder.alpha.detach().cpu().numpy()).T
        adata.varm[f"{key}_beta"] = np.exp(self.decoder.beta.detach().cpu().numpy()).T
        adata.varm[f"{key}_gamma"] = np.exp(self.decoder.gamma.detach().cpu().numpy()).T
        
        #Recover the original transition time and switching time
        self.setMode('eval')
        t_trans_orig, ts_orig = self.decoder.recoverTransitionTime() 
        adata.varm[f"{key}_t_"] = ts_orig.detach().cpu().numpy().T
        adata.uns[f"{key}_t_trans"] = t_trans_orig.detach().cpu().numpy().T
        
        #Compute the initial conditions for all cell types
        U0_dic, S0_dic = odeInitial(torch.exp(self.decoder.t_trans), 
                                   torch.exp(self.decoder.ts), 
                                   self.decoder.transgraph.graph, 
                                   self.decoder.transgraph.init_type, 
                                   torch.exp(self.decoder.alpha), 
                                   torch.exp(self.decoder.beta), 
                                   torch.exp(self.decoder.gamma),
                                   torch.exp(self.decoder.u0),
                                   torch.exp(self.decoder.s0),
                                   use_numpy=False)
        U0 = torch.stack([U0_dic[x] for x in U0_dic]).detach().cpu().numpy()
        S0 = torch.stack([S0_dic[x] for x in S0_dic]).detach().cpu().numpy()
        adata.varm[f"{key}_u0"] = U0.T
        adata.varm[f"{key}_s0"] = S0.T
        
        adata.var[f"{key}_scaling"] = np.exp(self.decoder.scaling.detach().cpu().numpy())
        adata.var[f"{key}_sigma_u"] = np.exp(self.decoder.sigma_u.detach().cpu().numpy())
        adata.var[f"{key}_sigma_s"] = np.exp(self.decoder.sigma_s.detach().cpu().numpy())
        
        
        mu_tx, std_tx, mu_zx, std_zx, logit_y, t, z, y = self.encoder.forward(torch.tensor(X).to(self.device), torch.exp(self.decoder.scaling), t_trans_orig, temp=1e-4)
        p_type = F.softmax(logit_y)
        labels = np.argmax(p_type.detach().cpu().numpy(), 1)
        my_t = mu_tx.squeeze().detach().cpu().numpy()
        
        adata.obs[f"{key}_time"] = my_t
        adata.obs[f"{key}_std_t"] = std_tx.squeeze().detach().cpu().numpy()
        adata.obsm[f"{key}_ptype"] = p_type.detach().cpu().numpy()
        adata.obs[f"{key}_label"] = labels
        
        adata.uns[f"{key}_transition_graph"] = self.decoder.transgraph.graph
        adata.uns[f"{key}_init_types"] = self.decoder.transgraph.init_types
        
        if(file_name is not None):
            adata.write_h5ad(f"{file_path}/{file_name}")

##############################################################
# Some basic building blocks of the neural network
##############################################################
class ForwardBlock(nn.Module):
    """
    A subnetwork as the building block of the entire NN in the VAE.
    There are four layers in this block:
    1. Linear layer
    2. Batch normalization
    3. Dropout
    4. Nonlinear Unit
    """
    def __init__(self, Cin, Cout, activation='ReLU'):
        super(ForwardBlock, self).__init__()
        self.fc = nn.Linear(Cin, Cout)
        self.bn = nn.BatchNorm1d(num_features=Cout)
        self.act = nn.ReLU()
        if(activation == 'LeakyReLU'):
            self.act == nn.LeakyReLU()
        elif(activation == 'ReLU'):
            self.act = nn.ReLU()
        elif(activation == 'ELU'):
            self.act = nn.ELU()
        elif(activation == 'tanh'):
            self.act = nn.Tanh() 
        else:
            print('Warning: activation not supported! Pick the default setting (ReLU)')
            self.act = nn.ReLU()
        self.dpt = nn.Dropout(p=0.2)
        self.block = nn.Sequential(self.fc, self.bn, self.act, self.dpt)

    def forward(self, data_in):
        return self.block(data_in)

class ResBlock(nn.Module):
    """
    A subnetwork with skip connections
    """
    def __init__(self, Cin, Cmid, Cout, activation='ReLU'):
        super(ResBlock, self).__init__()
        self.fb1 = ForwardBlock(Cin, Cmid, activation=activation)
        self.fb2 = ForwardBlock(Cmid, Cout, activation=activation)
        self.downsample = Cin > Cout
        self.sampling = nn.AvgPool1d(Cin//Cout) if Cin >= Cout else nn.Upsample(Cout)
    
    def forward(self, data_in):
        yres = self.fb2.forward(self.fb1.forward(data_in))
        #y = torch.mean(data_in,1).unsqueeze(-1)
        y = self.sampling(data_in.unsqueeze(1)).squeeze(1)
        if(y.shape[1] <= yres.shape[1]):
            d = yres.shape[1] - y.shape[1]
            y = F.pad(y,(d//2,d-d//2))
        else:
            d = y.shape[1] - yres.shape[1]
            y = y[:,d//2:y.shape[1]-(d-d//2)]
        return yres+y

class encoder_part(nn.Module):
    def __init__(self, Cin, Cout, N1=500, N2=250, device=torch.device('cpu'), **kwargs):
        super(encoder_part, self).__init__()
        self.fc1 = nn.Linear(Cin, N1).to(device)
        self.bn1 = nn.BatchNorm1d(num_features=N1).to(device)
        self.dpt1 = nn.Dropout(p=0.2).to(device)
        self.fc2 = nn.Linear(N1, N2).to(device)
        self.bn2 = nn.BatchNorm1d(num_features=N2).to(device)
        self.dpt2 = nn.Dropout(p=0.2).to(device)
        
        self.net = nn.Sequential(self.fc1, self.bn1, nn.LeakyReLU(), self.dpt1,
                                 self.fc2, self.bn2, nn.LeakyReLU(), self.dpt2,
                                 )
        
        self.fc_mu, self.spt1 = nn.Linear(N2,Cout).to(device), nn.Softplus()
        self.fc_std, self.spt2 = nn.Linear(N2,Cout).to(device), nn.Softplus()
        
        if('checkpoint' in kwargs):
            self.load_state_dict(torch.load(kwargs['checkpoint'],map_location=device))
        else:
            self.init_weights()

    def init_weights(self):
        for m in self.net.modules():
            if(isinstance(m, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
            elif(isinstance(m, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in [self.fc_mu, self.fc_std]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, data_in):
        z = self.net(data_in)
        mu_zx, std_zx = self.spt1(self.fc_mu(z)), self.spt2(self.fc_std(z))
        return mu_zx, std_zx

#############################################################
# VAE: 
#	Encoder learns the cell time and type
#
#	Decoder is an ODE. Instead of explicitly using a transition
#	graph, the ODE takes a probabilistic view of the predecessor
#   of each cell type.
##############################################################
class encoder_type(nn.Module):
    def __init__(self, Cin, Cout, hidden_size=(500,250), device=torch.device('cpu')):
        super(encoder_type, self).__init__()
        self.fw = ResBlock(Cz, hidden_size[0], hidden_size[1]).to(device)
        self.fc_yout = nn.Linear(hidden_size[1], Cout).to(device)
        self.ky = torch.tensor([1e4]).double().to(device)
        
        self.init_weights()
    
    def init_weights(self):
        for m in self.fw.modules():
            if(isinstance(m, nn.Linear)):
                nn.init.xavier_uniform_(m.weight,np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
            elif(isinstance(m, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in [self.fc_yout]:
                nn.init.xavier_uniform_(m.weight,np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, z, t, t_trans, temp):
        logit_y = self.fc_yout(self.fw(z)) - self.ky*F.relu(t_trans - t)
        y = F.gumbel_softmax(logit_y, tau=temp, hard=True) 
        return logit_y, y
        
class encoder(nn.Module):
    """
    Encoder Network
    Given the observation, it learns a latent representation z and cell time t.
    """
    def __init__(self, Cin, Ntype, Cz=None, hidden_size=[(500,250),(500,250),(500,250)], device=torch.device('cpu'), **kwargs):
        super(encoder, self).__init__()
        try:
            hidden_t, hidden_z, hidden_y = hidden_size[0], hidden_size[1], hidden_size[2]
        except IndexError:
            try:
                hidden_t, hidden_z = hidden_size[1], hidden_size[2]
            except IndexError:
                print(f'Expect hidden layer sizes of at least two networks, but got {len(hidden_size)} instead!')
            
        if(Cz is None):
            Cz = Ntype
        self.Cz = Cz
        
        #q(T | X)
        self.encoder_t = encoder_part(Cin, 1, hidden_t[0], hidden_t[1], device)

        #q(Z | X)
        self.encoder_z = encoder_part(Cin, Cz, hidden_z[0], hidden_z[1], device)

        #q(Y | Z,T,X)
        fix_cell_type = kwargs.pop('fix_cell_type', True)
        self.nn_y = None if fix_cell_type else encoder_type(Cz, Ntype, hidden_y, device)
        
        if('checkpoint' in kwargs):
            self.load_state_dict(torch.load(kwargs['checkpoint'],map_location=device))
        else:
            self.init_weights()
        

    def init_weights(self):
        #Initialize forward blocks
        self.encoder_t.init_weights()
        self.encoder_z.init_weights()
        if(self.nn_y is not None):
            self.nn_y.init_weights()
    
    def reparameterize(self, mu, std):
        eps = torch.normal(mean=torch.zeros(mu.shape),std=torch.ones(mu.shape)).to(mu.device)
        return std*eps+mu

    def forward(self, data_in, scaling, t_trans, temp=1.0):
        data_in_scale = torch.cat((data_in[:,:data_in.shape[1]//2]/scaling, data_in[:,data_in.shape[1]//2:]),1).double()
        #q(T | X)
        mu_tx, std_tx = self.encoder_t.forward(data_in_scale)

        #q(Z | X)
        mu_zx, std_zx = self.encoder_z.forward(data_in_scale)

        #Sampling
        t = self.reparameterize(mu_tx, std_tx)
        z = self.reparameterize(mu_zx, std_zx)

        #q(Y | Z,T,X)
        if(self.nn_y is not None):
            logit_y, y = self.nn_y.forward(z, t, t_trans, temp)
            return mu_tx, std_tx, mu_zx, std_zx, logit_y, t, z, y
        return mu_tx, std_tx, mu_zx, std_zx, None, t, z, None

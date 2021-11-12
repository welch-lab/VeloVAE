import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from sklearn.metrics import adjusted_rand_score
import time
import matplotlib.pyplot as plt

from velovae.plotting import plotSig, plotTLatent, plotTrainLoss, plotTestLoss, plotCluster, plotLatentEmbedding

from .model_util import  convertTime, initParams, getTsGlobal, reinitParams, transitionTime, reinitTypeParams, recoverTransitionTime, predSU, makeDir, getGeneIndex
from .model_util import odeInitial, odeWeighted, computeMixWeight
from .TrainingData import SCLabeledData
from .TransitionGraph import TransGraph, encodeType, str2int, int2str

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
def VAERisk(u,
            s,
            uhat,
            shat,
            w,
            mu_tx, std_tx,
            mu_t, std_t,
            mu_zx, std_zx,
            mu_z, std_z,
            logit_y_tzx, pi_y,
            sigma_u, sigma_s, 
            weight=None):
    """
    This is an upper bound of the negative evidence lower bound.
    1. u,s,uhat,shat: raw and predicted counts
    2. w: [B x Ntype] parent mixture weight
    3. mu_tx, std_tx: [B x 1] encoder output, conditional Gaussian parameters
    4. mu_t, std_t: [1] Gaussian prior of time
    5. mu_zx, std_zx: [B x Cz] encoder output, conditional Gaussian parameters
    6. logit_y_tzx: [B x N type] type probability before softmax operation
    7. pi_y: cell type prior (conditioned on time and z)
    8. sigma_u, sigma_s : standard deviation of the Gaussian likelihood (decoder)
    9. weight: sample weight
    
    
    log(p(x|t,z,y)) = log(\sum_{k} w_k p(x|t,z,y,k))
                   >= \sum_{k}w_k log(p(x|t,z,y,k))
    """
    
    #KL divergence
    kld_t = KLGaussian(mu_tx, std_tx, mu_t, std_t)
    kld_z = KLGaussian(mu_zx, std_zx, mu_z, std_z)
    kld_y = KLy(logit_y_tzx, pi_y)

    log_gaussian = -((uhat-u.unsqueeze(1))/sigma_u).pow(2)-((shat-s.unsqueeze(1))/sigma_s).pow(2)-torch.log(sigma_u)-torch.log(sigma_s*2*np.pi)
    #mse_sc = -((uhat-u.unsqueeze(1))/sigma_u).pow(2)-((shat-s.unsqueeze(1))/sigma_s).pow(2)
    #rectifier = nn.Softplus(threshold=1e-3)
    logpx_zty = (log_gaussian * (w.unsqueeze(-1))).sum(1)
    if( weight is not None):
        logpx_zty = logpx_zty*weight.view(-1,1)
    
    err_rec = torch.mean(torch.sum(logpx_zty, 1))
    return err_rec, kld_t, kld_z, kld_y



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
        data_in_scale = torch.cat((data_in[:,:data_in.shape[1]//2]/scaling, data_in[:,data_in.shape[1]//2:]),1).double()
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
    def __init__(self, adata, Tmax, device=torch.device('cpu'), p=95, tkey=None):
        super(decoder,self).__init__()
        
        U,S = adata.layers['Mu'], adata.layers['Ms']
        X = np.concatenate((U,S),1) 
        self.cell_labels_raw = adata.obs["clusters"].to_numpy() if "clusters" in adata.obs else None
        self.cell_types_raw = np.unique(self.cell_labels_raw)
        N, G = U.shape
        Ntype = len(self.cell_types_raw)
        
        self.label_dic, self.label_dic_rev = encodeType(self.cell_types_raw)
        self.cell_labels = np.array([self.label_dic[self.cell_labels_raw[i]] for i in range(len(self.cell_labels_raw))])
        self.cell_types = np.unique(self.cell_labels)
        self.Tmax = Tmax
        
        #Dynamical Model Parameters
        alpha, beta, gamma, scaling, ts, u0, s0, sigma_u, sigma_s, T, Rscore = initParams(X, p, fit_scaling=True)
        if(tkey is not None):
            print(f'[:: Decoder ::] Using pretrained time with key {tkey}')
            self.t_init = adata.obs[f'{tkey}_time'].to_numpy()
        else:
            t_init = np.quantile(T,0.5,1)
            self.t_init = t_init/np.quantile(t_init, 0.99)*Tmax
        
        t_trans, t_end, dts = np.zeros((Ntype)), np.zeros((Ntype)), np.random.rand(Ntype, G)*0.01
        for y in self.cell_types:
            t_trans[y] = np.quantile(self.t_init[self.cell_labels==y], 0.01)
            t_end[y] = np.quantile(self.t_init[self.cell_labels==y], 0.99)
        ts = t_trans.reshape(-1,1) + dts
        
        alpha, beta, gamma, u0, s0 = reinitTypeParams(U/scaling, S, self.t_init, ts, self.cell_labels, self.cell_types, self.cell_types)
        
        self.device = device
        self.alpha = (torch.tensor(np.log(alpha), device=device).double())
        self.beta = (torch.tensor(np.log(beta), device=device).double())
        self.gamma = (torch.tensor(np.log(gamma), device=device).double())
        
        self.scaling = (torch.tensor(np.log(scaling), device=device).double())
        self.t_trans = (torch.tensor(np.log(t_trans+1e-10), device=device).double())
        self.t_end = (torch.tensor(np.log(t_end+1e-10), device=device).double())
        self.dts = (torch.tensor(np.log(dts+1e-10), device=device).double())
        self.u0 = (torch.tensor(np.log(u0), device=device).double())
        self.s0 = (torch.tensor(np.log(s0), device=device).double())
        
        self.sigma_u = (torch.tensor(np.log(sigma_u), device=device).double())
        self.sigma_s = (torch.tensor(np.log(sigma_s), device=device).double())
        self.Rscore = Rscore
        
        self.scaling.requires_grad = False
        self.sigma_u.requires_grad = False
        self.sigma_s.requires_grad = False
        
        self.eps_t = torch.tensor(Tmax*0.01, device=device).double()
        
        self.updateWeight(self.t_init, self.cell_labels)
            
    def updateWeight(self, t, cell_labels):
        if(not isinstance(t, torch.Tensor)):
            t = torch.tensor(t).double().to(self.device)
        if(not isinstance(cell_labels, torch.Tensor)):
            cell_labels = torch.tensor(cell_labels).long().to(self.device)
        with torch.no_grad():
            logit_w, tscore, xscore = computeMixWeight(t.squeeze(),
                                       cell_labels,
                                       torch.exp(self.alpha),
                                       torch.exp(self.beta),
                                       torch.exp(self.gamma),
                                       torch.exp(self.t_trans),
                                       torch.exp(self.t_end),
                                       torch.exp(self.dts)+torch.exp(self.t_trans.view(-1,1)),
                                       torch.exp(self.u0),
                                       torch.exp(self.s0),
                                       torch.exp(self.sigma_u),
                                       torch.exp(self.sigma_s),
                                       self.eps_t)
            self.w = logit_w
            self.tscore = tscore.detach().cpu().numpy()
            self.xscore = xscore.detach().cpu().numpy()
            #self.u0 = torch.log(U0_hat+eps)
            #self.s0 = torch.log(S0_hat+eps)
    
    def forward(self, t, y_onehot, train_mode=False, neg_slope=1e-4, temp=0.01):
        """
        t: [B x 1]
        y_onehot: [B x Ntype]
        train_mode: determines whether to use leaky_relu 
        """
        #w = F.softmax(torch.sum(self.w*y_onehot.unsqueeze(-1), 1), 1)
        #w = F.gumbel_softmax(torch.sum(self.w*y_onehot.unsqueeze(-1), 1), tau=temp, hard=True)
        Uhat, Shat = odeWeighted(t, y_onehot,
                                 train_mode=False,
                                 neg_slope=neg_slope,
                                 alpha=torch.exp(self.alpha),
                                 beta=torch.exp(self.beta),
                                 gamma=torch.exp(self.gamma),
                                 t_trans=torch.exp(self.t_trans),
                                 ts=torch.exp(self.t_trans.view(-1,1))+torch.exp(self.dts),
                                 u0=torch.exp(self.u0),
                                 s0=torch.exp(self.s0),
                                 sigma_u = torch.exp(self.sigma_u),
                                 sigma_s = torch.exp(self.sigma_s),
                                 scaling=torch.exp(self.scaling))
        return Uhat, Shat
    
    def predSU(self, t, y_onehot, gidx=None):
        Ntype = y_onehot.shape[1]
        
        par = torch.argmax(self.w, 1)
        w = F.one_hot(torch.sum(y_onehot*par, 1).long(), Ntype)
        #w = F.softmax(torch.sum(self.w*y_onehot.unsqueeze(-1), 1), 1)
        if(gidx is None):
            Uhat, Shat, = odeWeighted(t, y_onehot,
                                      train_mode=False,
                                      alpha=torch.exp(self.alpha),
                                      beta=torch.exp(self.beta),
                                      gamma=torch.exp(self.gamma),
                                      t_trans=torch.exp(self.t_trans),
                                      ts=torch.exp(self.t_trans.view(-1,1))+torch.exp(self.dts),
                                      u0=torch.exp(self.u0),
                                      s0=torch.exp(self.s0),
                                      sigma_u = torch.exp(self.sigma_u),
                                      sigma_s = torch.exp(self.sigma_s),
                                      scaling=torch.exp(self.scaling))
        else:
            Uhat, Shat = odeWeighted(t, y_onehot, 
                                     train_mode=False,
                                     alpha=torch.exp(self.alpha[:,gidx]),
                                     beta=torch.exp(self.beta[:,gidx]),
                                     gamma=torch.exp(self.gamma[:,gidx]),
                                     t_trans=torch.exp(self.t_trans),
                                     ts=torch.exp(self.t_trans.view(-1,1))+torch.exp(self.dts[:,gidx]),
                                     u0=torch.exp(self.u0[:,gidx]),
                                     s0=torch.exp(self.s0[:,gidx]),
                                     sigma_u = torch.exp(self.sigma_u[gidx]),
                                     sigma_s = torch.exp(self.sigma_s[gidx]),
                                     scaling=torch.exp(self.scaling[gidx]))
        return (Uhat*w.unsqueeze(-1)).sum(1), (Shat*w.unsqueeze(-1)).sum(1)
    
    def getTimeDistribution(self):
        from scipy.stats import norm
        
        fig, ax = plt.subplots()
        for i in range(self.alpha.shape[0]):
            t_type = self.t_init[self.cell_labels==i]
            mu_t, std_t = np.mean(t_type), np.std(t_type)
            tmin, tmax = t_type.min(), t_type.max()
            x = np.linspace(tmin, tmax, 100)
            ax.plot(x, norm.pdf(x, loc=mu_t, scale=1/std_t),'-', lw=5, alpha=0.5, label=self.label_dic_rev[i])
        handles, labels = ax.get_legend_handles_labels()
        lgd=fig.legend(handles, labels, fontsize=15, markerscale=5, bbox_to_anchor=(1.0,1.0), loc='upper left')
        ax.set_xlabel("Time")
        ax.set_ylabel("PDF")
        ax.set_title("Time Distribution of Cell Types")
        fig.savefig("figures/time_dist.png", bbox_extra_artists=(lgd,), bbox_inches='tight')



class VAE():
    """
    The final VAE object containing all sub-modules
    """
    def __init__(self,
                 adata,
                 Cz=1,
                 hidden_size=[(1000,500),(1000,500),(1000,500)],
                 Tmax=20.0, 
                 device='cpu',
                 train_scaling=False,
                 **kwargs):
        """
        adata: anndata object
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
        #Training Configuration
        self.config = {
            "num_epochs":500, "learning_rate":1e-4, "learning_rate_ode":1e-4, "lambda":1e-3,\
            "reg_t":1.0, "reg_z":2.0, "reg_y":2.0, "reg_w":1.0, "neg_slope":1e-4,\
            "test_epoch":100, "save_epoch":100, "batch_size":128, "K_alt":0,\
            "train_scaling":False, "train_std":False, "weight_sample":False, "anneal":True, "informative_y":False
            
        }
        
        N, G = adata.n_obs, adata.n_vars
        Ntype = len(np.unique(cell_labels_raw))
        self.Ntype = Ntype
        self.encoder = encoder(2*G, Ntype, Cz, device=self.device, **kwargs).double()
        if('tkey' in kwargs):
            tkey = kwargs['tkey']
        else:
            tkey = None
        
        self.decoder = decoder(adata, Tmax, device=self.device, tkey=tkey)
        self.Tmax=torch.tensor(Tmax,dtype=torch.double).to(self.device)
        
        #Prior distribution
        self.mu_t = torch.tensor([Tmax*0.5]).double().to(self.device)
        self.std_t = torch.tensor([Tmax*0.5]).double().to(self.device)

        self.mu_z = torch.zeros(Cz).double().to(self.device)
        self.std_z = torch.ones(Cz).double().to(self.device)
        
        self.use_informative_y = "informative_y" in kwargs
        self.prior_y_default = (torch.ones(Ntype)*(1/Ntype)).double().to(self.device)
        self.prior_w_default = (torch.ones(Ntype,Ntype)*(1/Ntype)).double().to(self.device)
        
        
        
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
        eps = torch.normal(mean=0.0, std=1.0, size=(B,1)).double().to(self.device)
        return std*eps+mu
    

    def forward(self, data_in, train_mode, temp=1.0):
        scaling = torch.exp(self.decoder.scaling)
        
        mu_tx, std_tx, mu_zx, std_zx, logit_y, t, z, y_onehot = self.encoder.forward(data_in, scaling, torch.exp(self.decoder.t_trans.detach()), temp=temp)
        
        uhat, shat = self.decoder.forward(t, y_onehot.float(), train_mode, neg_slope=self.config['neg_slope'], temp=temp)
        
        return mu_tx, std_tx, mu_zx, std_zx, logit_y, t, y_onehot, uhat, shat
    
    def evalModel(self, data_in, gidx=None):
        """
        Run the full model with determinisic parent types.
        """
        scaling = torch.exp(self.decoder.scaling)
        
        mu_tx, std_tx, mu_zx, std_zx, logit_y, t, z, y_onehot = self.encoder.forward(data_in, scaling, torch.exp(self.decoder.t_trans.detach()), temp=1e-4)
        
        #Determine cell type
        py_tzx = F.softmax(logit_y,1)
        labels = torch.argmax(logit_y,1)
        y_onehot_pred = F.one_hot(labels, self.Ntype).float() #B x Ntype
        
        uhat, shat = self.decoder.predSU(mu_tx, y_onehot_pred, gidx)
        
        return uhat, shat, mu_tx, mu_zx, py_tzx, labels
    
    def setMode(self,mode):
        if(mode=='train'):
            self.encoder.train()
            self.decoder.train()
        elif(mode=='eval'):
            self.encoder.eval()
            self.decoder.eval()
        else:
            print("Warning: mode not recognized. Must be 'train' or 'test'! ")
    
    def printWeight(self, tscore=None, xscore=None):
        w = F.softmax(self.decoder.w.detach()).cpu().numpy()
        with pd.option_context('display.max_rows', None, 
                               'display.max_columns', None, 
                               'display.precision', 3,
                               'display.chop_threshold',1e-3,
                               'display.width', None):
            w_dic = {}
            for i in range(self.Ntype):
                w_dic[self.decoder.label_dic_rev[i]] = w[:, i]
            w_df = pd.DataFrame(w_dic, index=pd.Index([self.decoder.label_dic_rev[i] for i in range(self.Ntype)]))
            print(w_df)
            print("-------------------------------------------------")
        
            w_dic = {}
            for i in range(self.Ntype):
                w_dic[self.decoder.label_dic_rev[i]] = self.decoder.w[:, i].detach().cpu().numpy()
            w_df = pd.DataFrame(w_dic, index=pd.Index([self.decoder.label_dic_rev[i] for i in range(self.Ntype)]))
            print("Logit: ")
            print(w_df)
            print("-------------------------------------------------")
        
            if(tscore is not None):
                w_dic = {}
                for i in range(self.Ntype):
                    w_dic[self.decoder.label_dic_rev[i]] = tscore[:, i]
                w_df = pd.DataFrame(w_dic, index=pd.Index([self.decoder.label_dic_rev[i] for i in range(self.Ntype)]))
                print("Time Score: ")
                print(w_df)
                print("-------------------------------------------------")
        
            if(xscore is not None):
                w_dic = {}
                for i in range(self.Ntype):
                    w_dic[self.decoder.label_dic_rev[i]] = xscore[:, i]
                w_df = pd.DataFrame(w_dic, index=pd.Index([self.decoder.label_dic_rev[i] for i in range(self.Ntype)]))
                print("MSE Penalty: ")
                print(w_df)
                print("-------------------------------------------------")
    
    
    def train_epoch(self, 
                    X_loader, 
                    optimizer, 
                    optimizer2, 
                    counter, 
                    anneal=True, 
                    K=2, 
                    reg_t=1.0, 
                    reg_z=1.0, 
                    reg_y=1.0,
                    informative_y=False):
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
            xbatch, label_batch, weight = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
            mu_tx, std_tx, mu_zx, std_zx, logit_y, t, y_onehot, uhat, shat = self.forward(xbatch, True, temp=tau)
            u, s = xbatch[:,:xbatch.shape[1]//2],xbatch[:,xbatch.shape[1]//2:]
            
            if(informative_y):
                py = self.getInformativePriorY(label_batch)
            else:
                py = self.prior_y_default
            w_batch = (F.softmax(self.decoder.w)*y_onehot.unsqueeze(-1)).sum(1)
            err_rec, kld_t, kld_z, kld_y = VAERisk(u, s,
                                                   uhat, shat,
                                                   w_batch,
                                                   mu_tx, std_tx,
                                                   self.mu_t, self.std_t,
                                                   mu_zx, std_zx,
                                                   self.mu_z, self.std_z,
                                                   logit_y, py,
                                                   torch.exp(self.decoder.sigma_u), torch.exp(self.decoder.sigma_s), 
                                                   weight=None)
            loss = - err_rec + reg_t * kld_t + reg_z * kld_z + reg_y * kld_y
            loss_list.append(loss.detach().cpu().item())
            loss.backward()
            optimizer.step()
        
            
        #Update the mixture weight
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
        
        print("------------------------- Train a Mixture VAE -------------------------")
        #Get data loader
        U,S = adata.layers['Mu'], adata.layers['Ms']
        X = np.concatenate((U,S), 1)
        try:
            Xembed = adata.obsm[f"X_{embed}"]
        except KeyError:
            print("Embedding not found! Please run the corresponding preprocessing step!")
        plotTLatent(self.decoder.t_init, Xembed, 'Vanilla VAE', True, figure_path, 'vanilla')
        
        gind, gene_plot = getGeneIndex(adata.var_names, gene_plot)
        
        if(plot):
            makeDir(figure_path)
            
            t = torch.tensor(self.decoder.t_init).to(self.device)
            y_onehot = F.one_hot(torch.tensor(self.decoder.cell_labels)).float().to(self.device)
            #Plot the initial guess
            Uhat_init, Shat_init = self.decoder.predSU(t.unsqueeze(-1), y_onehot, gidx=torch.tensor(gind,device=self.device))
            t_trans = np.exp(self.decoder.t_trans.detach().cpu().numpy())
            ts = t_trans.reshape(-1,1) + np.exp(self.decoder.dts.detach().cpu().numpy())
            for i in range(len(gene_plot)):
                idx = gind[i]
                plotSig(self.decoder.t_init, 
                        U[:,idx], S[:,idx], 
                        Uhat_init[:,i].detach().cpu().numpy(), Shat_init[:,i].detach().cpu().numpy(), gene_plot[i], 
                        True, 
                        figure_path, 
                        f"{gene_plot[i]}-init",
                        cell_labels=self.decoder.cell_labels_raw,
                        cell_types=self.decoder.cell_types_raw,
                        labels_pred = self.decoder.cell_labels_raw,
                        sparsify=2,
                        t_trans=t_trans,
                        ts=ts[:,i])
            
        
        print("***      Creating  Dataset      ***")
        dataset = SCLabeledData(X, self.decoder.cell_labels, Rscore) if self.config['weight_sample'] else SCLabeledData(X, self.decoder.cell_labels)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True)
        print("***          Finished.          ***")
    
        #define optimizer
        print("***    Creating  optimizers     ***")
        learning_rate = self.config["learning_rate"]
        #param_nn = list(self.encoder.encoder_z.parameters())+list(self.encoder.fw_Y_ZTX.parameters())+list(self.encoder.fc_yout.parameters())
        #param_nn_t = self.encoder.encoder_t.parameters()
        param_nn = list(self.encoder.parameters())
        param_ode = [self.decoder.alpha, self.decoder.beta, self.decoder.gamma,
                    self.decoder.t_trans, self.decoder.dts, self.decoder.u0, self.decoder.s0]
        if(self.config["train_scaling"]):
            param_ode = param_ode+[self.decoder.scaling]
        if(self.config["train_std"]):
            param_ode = param_ode+[self.decoder.sigma_u, self.decoder.sigma_s]
    
        optimizer = torch.optim.Adam(param_nn, lr=self.config["learning_rate"], weight_decay=self.config["lambda"])
        #optimizer_t = torch.optim.Adam(param_nn_t, lr=self.config["learning_rate"]*0.05, weight_decay=self.config["lambda"])
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
        print('Before Training')
        self.printWeight()
        
        for epoch in range(n_epochs):
            #Optimize the encoder
            if(self.config["K_alt"]==0):
                loss_list, counter, tau = self.train_epoch(data_loader, 
                                                           optimizer, 
                                                           None, 
                                                           counter, 
                                                           self.config["anneal"], 
                                                           self.config["K_alt"],  
                                                           self.config["reg_t"], 
                                                           self.config["reg_z"], 
                                                           self.config["reg_y"],
                                                           self.config["informative_y"])
                loss_train = loss_train+loss_list
                loss_list, _, tau = self.train_epoch(data_loader, 
                                                     optimizer_ode, 
                                                     None, 
                                                     counter, 
                                                     self.
                                                     config["anneal"], 
                                                     self.config["K_alt"], 
                                                     self.config["reg_t"], 
                                                     self.config["reg_z"], 
                                                     self.config["reg_y"],
                                                     self.config["informative_y"])
                
                
                loss_train = loss_train+loss_list
            else:
                loss_list, counter, tau = self.train_epoch(data_loader, 
                                                           optimizer, 
                                                           optimizer_ode, 
                                                           counter, 
                                                           config["anneal"], 
                                                           self.config["K_alt"], 
                                                           self.config["reg_t"], 
                                                           self.config["reg_z"], 
                                                           self.config["reg_y"],
                                                           self.config["informative_y"])
                loss_train = loss_train+loss_list
            
            if( (epoch+1)%10 == 0):
                #Use the prior cell type to update the weights 
                self.decoder.updateWeight(t, self.decoder.cell_labels)
                print(f'Epoch {epoch+1} ')
                self.printWeight()
            
            if(epoch==0 or (epoch+1) % self.config["test_epoch"] == 0):
                print(f'temperature = {tau}')
                save = (epoch+1)%n_save==0 or epoch==0
                loss, t, err, rand_idx = self.test(torch.tensor(X).double().to(self.device),
                                                   Xembed,
                                                   self.decoder.cell_labels,
                                                   epoch+1, 
                                                   gind, 
                                                   False,
                                                   gene_plot,
                                                   save, 
                                                   figure_path
                                                   )
                    
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
                uhat, shat, mu_tx, mu_zx, py_tzx, labels = self.evalModel(data[i*B:(i+1)*B])
                Uhat[i*B:(i+1)*B] = uhat.cpu()
                Shat[i*B:(i+1)*B] = shat.cpu()
                t[i*B:(i+1)*B] = mu_tx.cpu().squeeze()
                qy[i*B:(i+1)*B] = py_tzx.cpu()
                y[i*B:(i+1)*B] = labels.cpu().squeeze()
                z[i*B:(i+1)*B] = mu_zx.cpu()
            if(N > B*Nb):
                uhat, shat, mu_tx, mu_zx, py_tzx, labels = self.evalModel(data[B*Nb:])
                Uhat[Nb*B:] = uhat.cpu()
                Shat[Nb*B:] = shat.cpu()
                t[Nb*B:] = mu_tx.cpu().squeeze()
                qy[Nb*B:] = py_tzx.cpu()
                y[Nb*B:] = labels.cpu().squeeze()
                z[Nb*B:] = mu_zx.cpu()
        
        return Uhat, Shat, t, qy, y, z
    
    def test(self,
             data, 
             Xembed,
             cell_labels, 
             testid=0, 
             gind=None,
             update_sigma=False,
             gene_plot=[],
             plot=False,
             path='figures', 
             **kwargs):
        """
        data: ncell x ngene tensor
        """
        
        self.setMode('eval')
        U,S = data[:,:data.shape[1]//2], data[:,data.shape[1]//2:]
        Uhat, Shat, t, qy, y, z = self.predAll(data)
        t = t.numpy()
        qy = qy.numpy()
        y = y.numpy()
        z = z.numpy()
        
        U,S = U.detach().cpu().numpy(), S.detach().cpu().numpy()

        mse = np.mean((Uhat.detach().cpu().numpy()-U)**2+(Shat.detach().cpu().numpy()-S)**2)
        if(update_sigma):
            self.decoder.sigma_u = torch.tensor(np.log(np.std(Uhat-U, 0)+1e-10), dtype=self.decoder.alpha.dtype).to(self.device)
            self.decoder.sigma_s = torch.tensor(np.log(np.std(Shat-S, 0)+1e-10), dtype=self.decoder.alpha.dtype).to(self.device)
        
        t_trans = np.exp(self.decoder.t_trans.detach().cpu().numpy())
        
        err = 1.0 - np.sum(y==self.decoder.cell_labels)/len(self.decoder.cell_labels)
        rand_idx = adjusted_rand_score(self.decoder.cell_labels, y)
        if(plot):
            plotCluster(Xembed,
                        qy,
                        self.decoder.cell_labels_raw,
                        False,
                        path,
                        f"{testid}")
            plotLatentEmbedding(z, 
                                self.Ntype, 
                                y,
                                self.decoder.label_dic_rev,
                                path=path,
                                figname=f"{testid}_yhat")
            plotLatentEmbedding(z, 
                                self.Ntype, 
                                self.decoder.cell_labels,
                                self.decoder.label_dic_rev,
                                path=path,
                                figname=f"{testid}_y")
            plotTLatent(t, Xembed, f"Training Epoch {testid}", plot, path, f"{testid}")
            for i in range(len(gene_plot)):
                idx = gind[i]
                plotSig(t, 
                        U[:,idx], S[:,idx], 
                        Uhat[:,idx], Shat[:,idx], gene_plot[i], 
                        True, 
                        path, 
                        f"{gene_plot[i]}-{testid}",
                        cell_labels=self.decoder.cell_labels_raw,
                        cell_types=self.decoder.cell_types_raw,
                        labels_pred = int2str(y, self.decoder.label_dic_rev),
                        sparsify=2,
                        t_trans=t_trans,
                        ts=np.exp(self.decoder.dts[:,idx].detach().cpu().numpy()) + t_trans)
                
        return mse, t, err, rand_idx
    
    
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
        adata.varm[f"{key}_ts"] = np.exp(self.decoder.dts.detach().cpu().numpy()).T + np.exp(self.decoder.t_trans.detach().cpu().numpy())
        adata.uns[f"{key}_t_trans"] = np.exp(self.decoder.t_trans.detach().cpu().numpy())
        adata.varm[f"{key}_u0"] = np.exp(self.decoder.u0.detach().cpu().numpy()).T
        adata.varm[f"{key}_s0"] = np.exp(self.decoder.s0.detach().cpu().numpy()).T
        adata.var[f"{key}_scaling"] = np.exp(self.decoder.scaling.detach().cpu().numpy())
        adata.var[f"{key}_sigma_u"] = np.exp(self.decoder.sigma_u.detach().cpu().numpy())
        adata.var[f"{key}_sigma_s"] = np.exp(self.decoder.sigma_s.detach().cpu().numpy())
        adata.uns[f"{key}_w"] = F.softmax(self.decoder.w.detach()).cpu().numpy()
        
        
        self.setMode('eval')
        mu_tx, std_tx, mu_zx, std_zx, logit_y, t, z, y_onehot = self.encoder.forward(torch.tensor(X).to(self.device), torch.exp(self.decoder.scaling), torch.exp(self.decoder.t_trans), temp=1e-4)
        p_type = F.softmax(logit_y)
        p_type_onehot = F.one_hot(torch.argmax(p_type,1), self.Ntype)
        labels = np.argmax(p_type.detach().cpu().numpy(), 1)
        my_t = mu_tx.squeeze().detach().cpu().numpy()
        
        adata.obs[f"{key}_time"] = my_t
        adata.obs[f"{key}_std_t"] = std_tx.squeeze().detach().cpu().numpy()
        adata.obsm[f"{key}_ptype"] = p_type.detach().cpu().numpy()
        adata.obs[f"{key}_label"] = labels
        
        
        if(file_name is not None):
            adata.write_h5ad(f"{file_path}/{file_name}")
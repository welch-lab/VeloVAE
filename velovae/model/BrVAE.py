import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import sklearn
from sklearn.metrics import adjusted_rand_score
import time
import matplotlib.pyplot as plt

from velovae.plotting import plotSig, plotTLatent, plotTrainLoss, plotTestLoss, plotCluster, plotLatentEmbedding

from .model_util import  histEqual, convertTime, initParams, getTsGlobal, reinitTypeParams, predSU, getGeneIndex
from .model_util import odeBr, optimal_transport_duality_gap, optimal_transport_duality_gap_ts
from .TrainingData import SCData
from .TransitionGraph import TransGraph, encodeType, str2int, int2str
from .velocity import rnaVelocityBrVAE

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


class decoder(nn.Module):
    """
    The ODE model that recovers the input data
    """
    def __init__(self, adata, Tmax, train_idx, device=torch.device('cpu'), p=95, tkey=None, nbin=40, q=0.01):
        super(decoder,self).__init__()
        
        U,S = adata.layers['Mu'][train_idx], adata.layers['Ms'][train_idx]
        X = np.concatenate((U,S),1) 
        N, G = len(train_idx), adata.n_vars
        
        cell_labels = adata.obs["clusters_int"].to_numpy()
        self.cell_labels = cell_labels[train_idx]
        self.cell_types = adata.uns["types_int"]
        Ntype = len(self.cell_types)
        self.Tmax = Tmax
        
        #Dynamical Model Parameters
        alpha, beta, gamma, scaling, ts, u0, s0, sigma_u, sigma_s, T, Rscore = initParams(X, p, fit_scaling=True)
        if(tkey is not None):
            print(f"[:: Decoder ::] Using pretrained time with key '{tkey}'")
            self.t_init = adata.obs[f'{tkey}_time'].to_numpy()
        else:
            T = T+np.random.rand(T.shape[0],T.shape[1]) * 1e-3
            T_eq = np.zeros(T.shape)
            Nbin = T.shape[0]//50+1
            for i in range(T.shape[1]):
                T_eq[:, i] = histEqual(T[:, i], Tmax, 0.9, Nbin)
            self.t_init = np.quantile(T_eq,0.5,1)
        
        t_trans, dts = np.zeros((Ntype)), np.random.rand(Ntype, G)*0.01
        for y in self.cell_types:
            t_trans[y] = np.quantile(self.t_init[self.cell_labels==y], 0.01)
        ts = t_trans.reshape(-1,1) + dts
        
        alpha, beta, gamma, u0, s0 = reinitTypeParams(U/scaling, S, self.t_init, ts, self.cell_labels, self.cell_types, self.cell_types)
        
        self.device = device
        self.alpha = nn.Parameter(torch.tensor(np.log(alpha), device=device).double())
        self.beta = nn.Parameter(torch.tensor(np.log(beta), device=device).double())
        self.gamma = nn.Parameter(torch.tensor(np.log(gamma), device=device).double())
        
        self.scaling = nn.Parameter(torch.tensor(np.log(scaling), device=device).double())
        self.t_trans = nn.Parameter(torch.tensor(np.log(t_trans+1e-10), device=device).double())
        self.dts = nn.Parameter(torch.tensor(np.log(dts+1e-10), device=device).double())
        self.u0 = nn.Parameter(torch.tensor(np.log(u0), device=device).double())
        self.s0 = nn.Parameter(torch.tensor(np.log(s0), device=device).double())
        
        self.sigma_u = (torch.tensor(np.log(sigma_u), device=device).double())
        self.sigma_s = (torch.tensor(np.log(sigma_s), device=device).double())
        
        self.scaling.requires_grad = False
        self.sigma_u.requires_grad = False
        self.sigma_s.requires_grad = False
        self.u0.requires_grad=False
        self.s0.requires_grad=False
        
        self.updateWeight(adata.obsm['X_pca'][train_idx], self.t_init, self.cell_labels, nbin=nbin, q=q)
        
    
    def forward(self, t, y_onehot, train_mode=False, neg_slope=0.0, temp=0.01):
        """
        t: [B x 1]
        y_onehot: [B x Ntype]
        train_mode: determines whether to use leaky_relu 
        """
        w = torch.sum(self.w*y_onehot.unsqueeze(-1), 1)
        w_onehot = F.one_hot(torch.argmax(w, 1), y_onehot.shape[1])
        
        Uhat, Shat = odeBr(t, y_onehot,
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
        return (Uhat*w_onehot.unsqueeze(-1)).sum(1), (Shat*w_onehot.unsqueeze(-1)).sum(1)
    
    def updateWeight(self, X_pca, t, cell_labels, nbin=20, epsilon = 0.05, lambda1 = 1, lambda2 = 50, max_iter = 2000, q = 0.01):
        Ntype = len(self.cell_types)
        dt = (t.max()-t.min())/nbin
        
        P = torch.zeros((Ntype, Ntype), device=self.alpha.device)
        for i, x in enumerate(self.cell_types): #child type
            mask = cell_labels==x
            if(not np.any(mask)):
                P[x,x] = 1.0
                continue
            t0 = np.quantile(t[mask], q) #estimated transition time
            
            mask1 = (t>=t0-dt) & (t<t0) 
            mask2 = (t>=t0) & (t<t0+dt)
            
            if(np.any(mask1) and np.any(mask2)):
                X1, X2 = X_pca[mask1], X_pca[mask2]
                C = sklearn.metrics.pairwise.pairwise_distances(X1,X2,metric='sqeuclidean', n_jobs=-1)
                C = C/np.median(C)
                G = np.ones((C.shape[0]))
                
                Pi = optimal_transport_duality_gap_ts(torch.tensor(C, device=self.alpha.device), 
                                                      torch.tensor(G, device=self.alpha.device), 
                                                      lambda1, lambda2, epsilon, 5, 1e-3, 10000, 1, max_iter)
                
                #Pi_ = optimal_transport_duality_gap(C,G,lambda1, lambda2, epsilon, 5, 0.01, 10000, 1, max_iter)
                
                #Sum the weights of each cell type
                cell_labels_1 = cell_labels[mask1]
                cell_labels_2 = cell_labels[mask2]
                for j, y in enumerate(self.cell_types): #parent
                    if(np.any(cell_labels_1==y) and np.any(cell_labels_2==x)):
                        P[x,y] = torch.sum(Pi[cell_labels_1==y])
            if(P[x].sum()==0):
                P[x,x] = 1.0
            
            P[x] = P[x]/P[x].sum()
        
        self.w = P.to(self.alpha.device)
        return
    
    def predSU(self, t, y_onehot, gidx=None):
        Ntype = y_onehot.shape[1]
        
        w = torch.sum(self.w*y_onehot.unsqueeze(-1), 1)
        w_onehot = F.one_hot(torch.argmax(w, 1), y_onehot.shape[1])
        if(gidx is None):
            Uhat, Shat, = odeBr(t, y_onehot,
                                neg_slope=0.0,
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
            Uhat, Shat = odeBr(t, y_onehot, 
                               neg_slope=0.0,
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
        return (Uhat*w_onehot.unsqueeze(-1)).sum(1), (Shat*w_onehot.unsqueeze(-1)).sum(1)
    
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



class BrVAE():
    """
    The final VAE object containing all sub-modules
    """
    def __init__(self,
                 adata,
                 Cz=1,
                 hidden_size=[(1000,500),(1000,500),(1000,500)],
                 Tmax=20.0, 
                 tprior=None,
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
            self.cell_types_raw = np.unique(cell_labels_raw)
        except KeyError:
            print('Please run the preprocessing step!')
        
        self.setDevice(device)
        
        #Training Configuration
        self.config = {
            "Cz":5,
            'hidden_size':[(500,250), (500,250), (500,250)],
            "num_epochs":800, 
            "learning_rate":1e-4, 
            "learning_rate_ode":1e-4, 
            "lambda":1e-3,
            "reg_t":1.0, 
            "reg_z":1.0, 
            "reg_y":1.0, 
            "neg_slope":0.0,
            "test_epoch":100, 
            "save_epoch":100, 
            "batch_size":128, 
            "train_test_split":0.7,
            "K_alt":0,
            "Nstart_ot":400, 
            "Nupdate_ot":25, 
            "nbin":40, 
            "q_ot":0.01,
            "fix_cell_type":True,
            "train_scaling":False, 
            "train_std":False, 
            "weight_sample":False, 
            "anneal":True, 
            "yprior":None, 
            "tprior":None,
            "sparsify":2
        }
        
        self.label_dic, self.label_dic_rev = encodeType(self.cell_types_raw)
        self.cell_labels = np.array([self.label_dic[x] for x in cell_labels_raw])
        self.cell_types = np.unique(self.cell_labels)
        adata.obs["clusters_int"] = self.cell_labels
        adata.uns["types_int"] = self.cell_types
        adata.uns["label_dic"] = self.label_dic
        adata.uns["label_dic_rev"] = self.label_dic_rev

        self.splitTrainTest(adata.n_obs)
        
        for key in kwargs:
            self.config[key] = kwargs[key]
        
        N, G = adata.n_obs, adata.n_vars
        Ntype = len(np.unique(cell_labels_raw))
        self.Ntype = Ntype
        self.encoder = encoder(2*G, Ntype, Cz, device=self.device, fix_cell_type=self.config["fix_cell_type"], **kwargs).double()
        if('tkey' in kwargs):
            tkey = kwargs['tkey']
        else:
            tkey = None
        
        self.decoder = decoder(adata, Tmax, self.train_idx, device=self.device, tkey=tkey, nbin=self.config['nbin'], q=self.config['q_ot'])
        self.Tmax=Tmax
        
        #Prior distribution
        if(tprior is None):
            self.mu_t = (torch.ones(len(self.train_idx),1)*Tmax*0.5).double().to(self.device)
            self.std_t = (torch.ones(len(self.train_idx),1)*Tmax*0.25).double().to(self.device)
        else:
            print('Using informative time prior.')
            t = adata.obs[tprior].to_numpy()
            t = t[self.train_idx]/t.max()*Tmax
            self.mu_t = torch.tensor(t).view(-1,1).double().to(self.device)
            self.std_t = (torch.ones(self.mu_t.shape)*Tmax*0.25).double().to(self.device)

        self.mu_z = torch.zeros(Cz).double().to(self.device)
        self.std_z = torch.ones(Cz).double().to(self.device)
        
        self.prior_y_default = (torch.ones(Ntype)*(1/Ntype)).double().to(self.device)
    
    def setDevice(self, device):
        if('cuda' in device):
            if(torch.cuda.is_available()):
                self.device = torch.device(device)
            else:
                print('Warning: GPU not detected. Using CPU as the device.')
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')
    
    def splitTrainTest(self, N):
        rand_perm = np.random.permutation(N)
        n_train = int(N*self.config["train_test_split"])
        self.train_idx = rand_perm[:n_train]
        self.test_idx = rand_perm[n_train:]
        
        return
    
    def getInformativePriorY(self, label_batch, delta=0.001):
        label_batch = label_batch.squeeze()
        py = F.one_hot(label_batch, self.Ntype)+delta
        for i in range(self.Ntype):
            py[label_batch==i, i] -= (self.Ntype+1)*delta
        return py
    
    def reparameterize(self, mu, std, B):
        eps = torch.normal(mean=0.0, std=1.0, size=(B,1)).double().to(self.device)
        return std*eps+mu
    
    def forward(self, data_in, y_onehot, temp=1.0):
        scaling = torch.exp(self.decoder.scaling)
        
        mu_tx, std_tx, mu_zx, std_zx, logit_y, t, z, _y_onehot = self.encoder.forward(data_in, scaling, torch.exp(self.decoder.t_trans.detach()), temp=temp)
        if(_y_onehot is None):
            _y_onehot = y_onehot
        
        uhat, shat = self.decoder.forward(t, _y_onehot.float(), True, neg_slope=self.config['neg_slope'], temp=temp)
        
        return mu_tx, std_tx, mu_zx, std_zx, logit_y, t, _y_onehot, uhat, shat
    
    def evalModel(self, data_in, y_onehot, gidx=None):
        """
        Run the full model with determinisic parent types.
        """
        scaling = torch.exp(self.decoder.scaling)
        
        mu_tx, std_tx, mu_zx, std_zx, logit_y, t, z, _y_onehot = self.encoder.forward(data_in, scaling, torch.exp(self.decoder.t_trans.detach()), temp=1e-4)
        
        #Determine cell type
        if(logit_y is not None):
            py_tzx = F.softmax(logit_y,1)
            labels = torch.argmax(logit_y,1)
            y_onehot_pred = F.one_hot(labels, self.Ntype).float() #B x Ntype
        else:
            py_tzx = None
            y_onehot_pred = y_onehot
            labels = torch.argmax(y_onehot, 1)
        
        uhat, shat = self.decoder.predSU(mu_tx, y_onehot_pred, gidx)
        
        return uhat, shat, mu_tx, std_tx, mu_zx, std_zx, py_tzx, labels
    
    def setMode(self,mode):
        if(mode=='train'):
            self.encoder.train()
            self.decoder.train()
        elif(mode=='eval'):
            self.encoder.eval()
            self.decoder.eval()
        else:
            print("Warning: mode not recognized. Must be 'train' or 'test'! ")
    
    def updateWeight(self, data, X_embed):
        N, G = data.shape[0], data.shape[1]//2
        t = torch.empty(N)
        
        with torch.no_grad():
            B = self.config["batch_size"]
            Nb = N // B
            scaling = torch.exp(self.decoder.scaling)
            mu_tx, std_tx, mu_zx, std_zx, logit_y, t, z, y_onehot = self.encoder.forward(data, scaling, torch.exp(self.decoder.t_trans.detach()), temp=1e-4)
                
        t = mu_tx.detach().cpu().numpy().squeeze()
        if(self.config["fix_cell_type"]):
            self.decoder.updateWeight(X_embed, t, self.decoder.cell_labels, nbin=self.config['nbin'], q=self.config['q_ot'])
        else:
            y = torch.argmax(logit_y,1).detach().cpu().numpy()
            self.decoder.updateWeight(X_embed, t, y, nbin=self.config['nbin'], q=self.config['q_ot'])
        
    
    ############################################################
    #Training Objective
    ############################################################
    def VAERisk(self,
                u,
                s,
                uhat,
                shat,
                mu_tx, std_tx,
                mu_t, std_t,
                mu_zx, std_zx,
                mu_z, std_z,
                sigma_u, sigma_s, 
                logit_y_tzx=None, pi_y=None,
                weight=None):
        """
        1. u,s,uhat,shat: raw and predicted counts
        2. w: [B x Ntype] parent mixture weight
        3. mu_tx, std_tx: [B x 1] encoder output, conditional Gaussian parameters
        4. mu_t, std_t: [1] Gaussian prior of time
        5. mu_zx, std_zx: [B x Cz] encoder output, conditional Gaussian parameters
        6. logit_y_tzx: [B x N type] type probability before softmax operation
        7. pi_y: cell type prior (conditioned on time and z)
        8. sigma_u, sigma_s : standard deviation of the Gaussian likelihood (decoder)
        9. weight: sample weight
        """
        
        #KL divergence
        kld_t = KLGaussian(mu_tx, std_tx, mu_t, std_t)
        kld_z = KLGaussian(mu_zx, std_zx, mu_z, std_z)
        kld_y = 0 if ((logit_y_tzx is None) or (pi_y is None)) else KLy(logit_y_tzx, pi_y)
    
        log_gaussian = -((uhat-u)/sigma_u).pow(2)-((shat-s)/sigma_s).pow(2)-torch.log(sigma_u)-torch.log(sigma_s*2*np.pi)
        
        if( weight is not None):
            log_gaussian = log_gaussian*weight.view(-1,1)
        
        err_rec = torch.mean(torch.sum(log_gaussian, 1))
        return err_rec, kld_t, kld_z, kld_y
      
    def train_epoch(self, 
                    X_loader, 
                    optimizer, 
                    optimizer2, 
                    counter, 
                    anneal=True, 
                    K=2, 
                    reg_t=1.0, 
                    reg_z=1.0, 
                    reg_y=1.0):
        """
        Training in each epoch
        X_loader: Data loader of the input data
        optimizer
        """
        self.setMode('train')
        iterX = iter(X_loader)
        B = len(iterX)
        loss_list = []
        Nupdate = B*5
        for i in range(B):
            tau = np.clip(np.exp(-3e-5*((counter+i+1)//Nupdate*Nupdate)), 0.5, None) if anneal else 0.5
            optimizer.zero_grad()
            if(optimizer2 is not None):
                optimizer2.zero_grad()
            batch = iterX.next()
            xbatch, label_batch, weight, idx = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device), batch[3].to(self.device)
            u, s = xbatch[:,:xbatch.shape[1]//2],xbatch[:,xbatch.shape[1]//2:]
            
            y_onehot_fix = F.one_hot(label_batch, self.Ntype)
            mu_tx, std_tx, mu_zx, std_zx, logit_y, t, y_onehot, uhat, shat = self.forward(xbatch, y_onehot_fix, temp=tau)
           
            
            if(logit_y is None):
                err_rec, kld_t, kld_z, kld_y = self.VAERisk(u, s,
                                                            uhat, shat,
                                                            mu_tx, std_tx,
                                                            self.mu_t[idx], self.std_t[idx],
                                                            mu_zx, std_zx,
                                                            self.mu_z, self.std_z,
                                                            torch.exp(self.decoder.sigma_u), torch.exp(self.decoder.sigma_s))
            else:
                if(self.config['yprior'] is not None):
                    prior_y = self.getInformativePriorY(label_batch)
                else:
                    prior_y = self.prior_y_default
                err_rec, kld_t, kld_z, kld_y = self.VAERisk(u, s,
                                                            uhat, shat,
                                                            mu_tx, std_tx,
                                                            self.mu_t[idx], self.std_t[idx],
                                                            mu_zx, std_zx,
                                                            self.mu_z, self.std_z,
                                                            torch.exp(self.decoder.sigma_u), torch.exp(self.decoder.sigma_s), 
                                                            logit_y, prior_y,
                                                            weight=None)
            loss = - err_rec + reg_t * kld_t + reg_z * kld_z + reg_y * kld_y
            loss_list.append(loss.detach().cpu().item())
            loss.backward()
            optimizer.step()
            
        if( optimizer2 is not None and ((i+1) % K == 0 or i==B-1)):
            optimizer2.step()
        
        return loss_list, counter+B, tau
    
    def loadConfig(self, config):
        #We don't have to specify all the hyperparameters. Just pass the ones we want to modify.
        for key in config:
            if(key in self.config):
                self.config[key] = config[key]
            else:
                self.config[key] = config[key]
                print(f"Added new hyperparameter: {key}")
        if(self.config["train_scaling"]):
            self.decoder.scaling.requires_grad = True
        if(self.config["train_std"]):
            self.decoder.sigma_u.requires_grad = True
            self.decoder.sigma_s.requires_grad = True
    
    def printWeight(self):
        w = self.decoder.w.cpu().numpy()
        with pd.option_context('display.max_rows', None, 
                               'display.max_columns', None, 
                               'display.precision', 3,
                               'display.chop_threshold',1e-3,
                               'display.width', 200):
            w_dic = {}
            for i, x in enumerate(self.cell_types):
                w_dic[self.label_dic_rev[x]] = w[:, x]
            w_df = pd.DataFrame(w_dic, index=pd.Index([self.label_dic_rev[x] for x in self.cell_types]))
            print(w_df)
    
    def train(self, 
              adata, 
              config={}, 
              plot=True, 
              gene_plot=[], 
              figure_path="figures", 
              embed="umap"):
        
        self.loadConfig(config)
        
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
        plotTLatent(self.decoder.t_init, Xembed[self.train_idx], 'Vanilla VAE', True, figure_path, 'vanilla')
        
        gind, gene_plot = getGeneIndex(adata.var_names, gene_plot)
        
        print("*** Creating Training/Validation Datasets ***")
        train_set = SCData(X[self.train_idx], self.cell_labels[self.train_idx], self.decoder.Rscore[self.train_idx]) if self.config['weight_sample'] else SCData(X[self.train_idx], self.cell_labels[self.train_idx])
        test_set = None
        if(len(self.test_idx)>0):
            test_set = SCData(X[self.test_idx], self.cell_labels[self.test_idx], self.decoder.Rscore[self.test_idx]) if self.config['weight_sample'] else SCData(X[self.test_idx], self.cell_labels[self.test_idx])
        data_loader = torch.utils.data.DataLoader(train_set, batch_size=self.config["batch_size"], shuffle=True)
        print('***           Finished.         ***')
        
        if(plot):
            os.makedirs(figure_path, exist_ok=True)
    
        #define optimizer
        print("***    Creating  optimizers     ***")
        learning_rate = self.config["learning_rate"]
        param_nn = list(self.encoder.parameters())
        param_ode = [self.decoder.alpha, self.decoder.beta, self.decoder.gamma,
                    self.decoder.t_trans, self.decoder.dts, self.decoder.u0, self.decoder.s0]
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
        loss_train, loss_test, err_train, err_test, randidx_train, randidx_test = [],[],[],[],[],[]
        
        start = time.time()
        
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
                                                           self.config["reg_y"])
                loss_list, _, tau = self.train_epoch(data_loader, 
                                                     optimizer_ode, 
                                                     None, 
                                                     counter, 
                                                     self.
                                                     config["anneal"], 
                                                     self.config["K_alt"], 
                                                     self.config["reg_t"], 
                                                     self.config["reg_z"], 
                                                     self.config["reg_y"])
                
                
            else:
                loss_list, counter, tau = self.train_epoch(data_loader, 
                                                           optimizer, 
                                                           optimizer_ode, 
                                                           counter, 
                                                           config["anneal"], 
                                                           self.config["K_alt"], 
                                                           self.config["reg_t"], 
                                                           self.config["reg_z"], 
                                                           self.config["reg_y"])

            if(epoch==0 or (epoch+1) % self.config["test_epoch"] == 0):
                print(f'temperature = {tau}')
                save = (epoch+1)%n_save==0 or epoch==0
                mse_train, t, err_train, rand_idx_train = self.test(train_set,
                                                                    Xembed[self.train_idx],
                                                                    f"train{epoch+1}", 
                                                                    gind, 
                                                                    False,
                                                                    gene_plot,
                                                                    save, 
                                                                    figure_path)
                mse_test, err_test, rand_idx_test = 'N/A', 'N/A', 'N/A'
                if(test_set is not None):
                    mse_test, t, err_test, rand_idx_test = self.test(test_set,
                                                                     Xembed[self.test_idx],
                                                                     f"test{epoch+1}", 
                                                                     gind, 
                                                                     False,
                                                                     gene_plot,
                                                                     save, 
                                                                     figure_path)
                loss_train.append(mse_train)
                loss_test.append(mse_test)
                if(not self.config["fix_cell_type"]):
                    err_train.append(err_train)
                    randidx_train.append(rand_idx_train)
                    err_test.append(err_test)
                    randidx_test.append(rand_idx_test)
                
                print(f"Epoch {epoch+1}: Train MSE = {mse_train:.2f}, Test MSE = {mse_test:.2f}, \t Total Time = {convertTime(time.time()-start)}")
                if(not self.config["fix_cell_type"]):
                    print(f"Train Type Error: {err_train:.3f}, Test Type Error: {err_test:.3f}")
                
                self.printWeight()
            
            if( (epoch+1)>=self.config['Nstart_ot'] and (epoch+1) % self.config['Nupdate_ot'] == 0):
                self.updateWeight(torch.tensor(X[self.train_idx]).to(self.device), adata.obsm['X_pca'][self.train_idx])
        
        print("***     Finished Training     ***")
        plotTrainLoss(loss_train,[i for i in range(len(loss_train))],True,figure_path,"train-brvae")
        plotTestLoss(loss_test,[1]+[i*self.config["test_epoch"] for i in range(1,len(loss_test))],True,figure_path,"test-brvae")
        if(not self.config["fix_cell_type"]):
            plotTestLoss(err_train,[1]+[i*self.config["test_epoch"] for i in range(1,len(err_train))],True,figure_path,"train-err-brvae")
            plotTestLoss(randidx_train,[1]+[i*self.config["test_epoch"] for i in range(1,len(randidx_train))],True,figure_path,"train-randidx-brvae")
            plotTestLoss(err_test,[1]+[i*self.config["test_epoch"] for i in range(1,len(err_test))],True,figure_path,"test-err-brvae")
            plotTestLoss(randidx_test,[1]+[i*self.config["test_epoch"] for i in range(1,len(randidx_test))],True,figure_path,"test-randidx-brvae")
        return
    
    
    def predAll(self, data, cell_labels):
        N, G = data.shape[0], data.shape[1]//2
        Uhat, Shat = torch.empty(N,G), torch.empty(N,G)
        t = torch.empty(N)
        std_t = torch.empty(N)
        y = torch.empty(N)
        z = torch.empty(N, self.encoder.Cz)
        std_z = torch.empty(N, self.encoder.Cz)
        qy = torch.empty(N, self.Ntype)
        
        y_onehot_fix = F.one_hot(torch.tensor(cell_labels), self.Ntype).to(data.device)
        with torch.no_grad():
            B = min(N//10, 1000)
            Nb = N // B
            for i in range(Nb):
                uhat, shat, mu_tx, std_tx, mu_zx, std_zx, py_tzx, labels = self.evalModel(data[i*B:(i+1)*B], y_onehot_fix[i*B:(i+1)*B])
                Uhat[i*B:(i+1)*B] = uhat.cpu()
                Shat[i*B:(i+1)*B] = shat.cpu()
                t[i*B:(i+1)*B] = mu_tx.cpu().squeeze()
                std_t[i*B:(i+1)*B] = std_tx.cpu().squeeze()
                z[i*B:(i+1)*B] = mu_zx.cpu()
                std_z[i*B:(i+1)*B] = std_zx.cpu()
                if(py_tzx is None):
                    qy[i*B:(i+1)*B] = y_onehot_fix[i*B:(i+1)*B].detach().cpu()
                    y[i*B:(i+1)*B] = torch.argmax(qy[i*B:(i+1)*B],1)
                else:
                    qy[i*B:(i+1)*B] = py_tzx.cpu()
                    y[i*B:(i+1)*B] = labels.cpu().squeeze()
                
            if(N > B*Nb):
                uhat, shat, mu_tx, std_tx, mu_zx, std_zx, py_tzx, labels = self.evalModel(data[B*Nb:], y_onehot_fix[B*Nb:])
                Uhat[Nb*B:] = uhat.cpu()
                Shat[Nb*B:] = shat.cpu()
                t[Nb*B:] = mu_tx.cpu().squeeze()
                std_t[Nb*B:] = std_tx.cpu().squeeze()
                z[Nb*B:] = mu_zx.cpu()
                std_z[Nb*B:] = std_zx.cpu()
                if(py_tzx is None):
                    qy[Nb*B:] = y_onehot_fix[Nb*B:].detach().cpu()
                    y[Nb*B:] = torch.argmax(qy[Nb*B:],1)
                else:
                    qy[Nb*B:] = py_tzx.cpu()
                    y[Nb*B:] = labels.cpu().squeeze()
                
        
        return Uhat, Shat, t, std_t, qy, y, z, std_z
    
    def test(self,
             dataset, 
             Xembed,
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
        data = torch.tensor(dataset.data).double().to(self.device)
        U,S = data[:,:data.shape[1]//2], data[:,data.shape[1]//2:]
        Uhat, Shat, t, std_t, qy, y, z, std_z = self.predAll(data, dataset.labels)
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
        
        err = 1.0 - np.sum(y==dataset.labels)/len(dataset.labels)
        rand_idx = adjusted_rand_score(dataset.labels, y)
        cell_labels_raw = np.array([self.label_dic_rev[x] for x in dataset.labels])
        if(plot):
            if(not self.config["fix_cell_type"]):
                plotCluster(Xembed,
                            qy,
                            cell_labels_raw,
                            False,
                            path,
                            f"{testid}")
                plotLatentEmbedding(z, 
                                    self.Ntype, 
                                    y,
                                    self.label_dic_rev,
                                    path=path,
                                    figname=f"{testid}_yhat")
            plotLatentEmbedding(z, 
                                self.Ntype, 
                                dataset.labels,
                                self.label_dic_rev,
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
                        cell_labels=cell_labels_raw,
                        cell_types=self.cell_types_raw,
                        labels_pred = int2str(y, self.label_dic_rev),
                        sparsify=self.config["sparsify"],
                        t_trans=t_trans,
                        ts=np.exp(self.decoder.dts[:,idx].detach().cpu().numpy()) + t_trans)
                
        return mse, t, err, rand_idx
    
    
    def saveModel(self, file_path, enc_name='encoder', dec_name='decoder'):
        """
        Save the encoder parameters to a .pt file.
        Save the decoder parameters to the anndata object.
        """
        os.makedirs(file_path, exist_ok=True)
        torch.save(self.encoder.state_dict(), f"{file_path}/{enc_name}.pt")
        torch.save(self.decoder.state_dict(), f"{file_path}/{dec_name}.pt")
    
    def saveAnnData(self, adata, key, file_path, file_name=None):
        """
        Save the ODE parameters and cell time to the anndata object and write it to disk.
        """
        os.makedirs(file_path, exist_ok=True)
        
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
        adata.uns[f"{key}_w"] = self.decoder.w.detach().cpu().numpy()
        
        
        self.setMode('eval')
        Uhat, Shat, t, std_t, qy, y, z, std_z = self.predAll(torch.tensor(np.concatenate((adata.layers['Mu'], adata.layers['Ms']),axis=1)).float().to(self.device), adata.obs["clusters_int"].to_numpy())
        adata.obsm[f"{key}_ptype"] = qy.numpy()
        adata.obs[f"{key}_label"] = y.numpy()
        adata.obs[f"{key}_time"] = t.numpy()
        adata.obs[f"{key}_std_t"] = std_t.numpy()
        adata.obsm[f"{key}_z"] = z.numpy()
        adata.obsm[f"{key}_std_z"] = std_z.numpy()
        adata.layers[f"{key}_uhat"] = Uhat.numpy()
        adata.layers[f"{key}_shat"] = Shat.numpy()
        
        adata.uns[f"{key}_train_idx"] = self.train_idx
        adata.uns[f"{key}_test_idx"] = self.test_idx
        
        rnaVelocityBrVAE(adata, key)
        
        if(file_name is not None):
            adata.write_h5ad(f"{file_path}/{file_name}")
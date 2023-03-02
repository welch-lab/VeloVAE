import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.negative_binomial import NegativeBinomial
from torch.distributions.poisson import Poisson
import os
from torch.utils.data import Dataset, DataLoader
import time
from velovae.plotting import plot_sig, plot_sig_, plot_time, plot_train_loss, plot_test_loss, plot_loss_split, plot_vel

from .vae import VAE
from .vae import encoder as encoder_velovae
from .vae import decoder as decoder_velovae
from .model_util import scale_by_gene, scale_by_cell, get_cell_scale, get_gene_scale, get_dispersion
from .model_util import hist_equal, init_params, init_params_raw, get_ts_global, reinit_params, convert_time, get_gene_index
from .model_util import pred_su, ode, knnx0_index, get_x0
from .transition_graph import encode_type
from .training_data import SCData
from .vanilla_vae import VanillaVAE, kl_gaussian, kl_uniform
from .vanilla_vae import encoder as encoder_vanilla
from .vanilla_vae import decoder as decoder_vanilla
from .velocity import rna_velocity_vae



##############################################################
# VAE
##############################################################
class encoder(nn.Module):
    def __init__(self, Cin, dim_z, dim_cond=0, N1=500, N2=250, device=torch.device('cpu'), checkpoint=None):
        super(encoder, self).__init__()
        self.fc1 = nn.Linear(Cin, N1).to(device)
        self.bn1 = nn.BatchNorm1d(num_features=N1).to(device)
        self.dpt1 = nn.Dropout(p=0.2).to(device)
        self.fc2 = nn.Linear(N1, N2).to(device)
        self.bn2 = nn.BatchNorm1d(num_features=N2).to(device)
        self.dpt2 = nn.Dropout(p=0.2).to(device)
        
        self.net = nn.Sequential(self.fc1, self.bn1, nn.ReLU(), self.dpt1,
                                 self.fc2, self.bn2, nn.ReLU(), self.dpt2,
                                 )
        
        self.fc_mu_t, self.spt1 = nn.Linear(N2+dim_cond,1).to(device), nn.Softplus()
        self.fc_std_t, self.spt2 = nn.Linear(N2+dim_cond,1).to(device), nn.Softplus()
        self.fc_mu_z = nn.Linear(N2+dim_cond,dim_z).to(device)
        self.fc_std_z, self.spt3 = nn.Linear(N2+dim_cond,dim_z).to(device), nn.Softplus()
        
        if(checkpoint is not None):
            self.load_state_dict(torch.load(checkpoint,map_location=device))
        else:
            self.init_weights()

    def init_weights(self, data_density=0.1):
        for m in self.net.modules():
            if(isinstance(m, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
            elif(isinstance(m, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in [self.fc_mu_t, self.fc_std_t, self.fc_mu_z, self.fc_std_z]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)
    
    def forward(self, data_in, condition=None):
        h = self.net(data_in)
        if(condition is not None):
            h = torch.cat((h,condition),1)
        mu_tx, std_tx = self.spt1(self.fc_mu_t(h)), self.spt2(self.fc_std_t(h))
        mu_zx, std_zx = self.fc_mu_z(h), self.spt3(self.fc_std_z(h))
        return mu_tx, std_tx, mu_zx, std_zx

class decoder(nn.Module):
    def __init__(self, 
                 adata, 
                 tmax,
                 train_idx,
                 dim_z, 
                 dim_cond=0,
                 N1=250, 
                 N2=500, 
                 p=98, 
                 use_raw=True,
                 init_ton_zero=False,
                 scale_cell=False,
                 separate_us_scale=True,
                 add_noise=False,
                 device=torch.device('cpu'), 
                 init_method="steady", 
                 init_key=None, 
                 init_type=None,
                 checkpoint=None):
        super(decoder,self).__init__()
        G = adata.n_vars
        self.tmax=tmax
            
        #Library Size
        if(use_raw):
            U, S = adata.layers['unspliced'].A.astype(float), adata.layers['spliced'].A.astype(float)
        else:
            U, S = adata.layers["Mu"], adata.layers["Ms"]
        
        # Dispersion
        mean_u, mean_s, dispersion_u, dispersion_s = get_dispersion(U[train_idx], S[train_idx])
        adata.var["mean_u"] = mean_u
        adata.var["mean_s"] = mean_s
        adata.var["dispersion_u"] = dispersion_u
        adata.var["dispersion_s"] = dispersion_s
        
        if scale_cell:
            U, S, lu, ls = scale_by_cell(U, S, train_idx, separate_us_scale, 50)
            adata.obs["library_scale_u"] = lu
            adata.obs["library_scale_s"] = ls
        else:
            lu, ls = get_cell_scale(U, S, train_idx, separate_us_scale, 50)
            adata.obs["library_scale_u"] = lu
            adata.obs["library_scale_s"] = ls
        
        U = U[train_idx]
        S = S[train_idx]
        
        X = np.concatenate((U,S),1)
        if(add_noise):
            noise = np.exp(np.random.normal(size=(len(train_idx), 2*G))*1e-3)
            X = X + noise
        
        if(checkpoint is None):
            alpha, beta, gamma, scaling, toff, u0, s0, T = init_params_raw(X,p,fit_scaling=True)
            
            if(init_method == "random"):
                print("Random Initialization.")
                scaling, scaling_s = get_gene_scale(U,S,None)
                self.alpha = nn.Parameter(torch.normal(0.0, 0.01, size=(G,), device=device).float())
                self.beta =  nn.Parameter(torch.normal(0.0, 0.01, size=(G,), device=device).float())
                self.gamma = nn.Parameter(torch.normal(0.0, 0.01, size=(G,), device=device).float())
                self.ton = torch.nn.Parameter(torch.ones(G, device=device).float()*(-10))
                self.t_init = None
            elif(init_method == "tprior"):
                print("Initialization using prior time.")
                t_prior = adata.obs[init_key].to_numpy()
                t_prior = t_prior[train_idx]
                std_t = (np.std(t_prior)+1e-3)*0.2
                self.t_init = np.random.uniform(t_prior-std_t, t_prior+std_t)
                self.t_init -= self.t_init.min()
                self.t_init = self.t_init
                self.t_init = self.t_init/self.t_init.max()*tmax
                toff = get_ts_global(self.t_init, U, S, 95)
                alpha, beta, gamma, ton = reinit_params(U, S, self.t_init, toff)
                
                self.alpha = nn.Parameter(torch.tensor(np.log(alpha), device=device).float())
                self.beta = nn.Parameter(torch.tensor(np.log(beta), device=device).float())
                self.gamma = nn.Parameter(torch.tensor(np.log(gamma), device=device).float())
                self.ton = nn.Parameter((torch.ones(adata.n_vars, device=device)*(-10)).float()) if init_ton_zero else nn.Parameter(torch.tensor(np.log(ton+1e-10), device=device).float())
            else:
                print("Initialization using the steady-state and dynamical models.")

                if(init_key is not None):
                    self.t_init = adata.obs[init_key].to_numpy()[train_idx]
                else:
                    T = T+np.random.rand(T.shape[0],T.shape[1]) * 1e-3
                    T_eq = np.zeros(T.shape)
                    Nbin = T.shape[0]//50+1
                    for i in range(T.shape[1]):
                        T_eq[:, i] = hist_equal(T[:, i], tmax, 0.9, Nbin)
                    self.t_init = np.quantile(T_eq,0.5,1)
                toff = get_ts_global(self.t_init, U, S, 95)
                alpha, beta, gamma, ton = reinit_params(U, S, self.t_init, toff)
                
                self.alpha = nn.Parameter(torch.tensor(np.log(alpha), device=device).float())
                self.beta = nn.Parameter(torch.tensor(np.log(beta), device=device).float())
                self.gamma = nn.Parameter(torch.tensor(np.log(gamma), device=device).float())
                self.ton = nn.Parameter((torch.ones(adata.n_vars, device=device)*(-10)).float()) if init_ton_zero else nn.Parameter(torch.tensor(np.log(ton+1e-10), device=device).float())
        
        self.scaling = nn.Parameter(torch.tensor(np.log(scaling), device=device).float())
            
        
        if(init_type is None):  
            self.u0 = nn.Parameter(torch.ones(G, device=device).float()*(-10))
            self.s0 = nn.Parameter(torch.ones(G, device=device).float()*(-10))
        elif(init_type == "random"):
            rv_u = stats.gamma(1.0, 0, 4.0)
            rv_s = stats.gamma(1.0, 0, 4.0)
            r_u_gamma = rv_u.rvs(size=(G))
            r_s_gamma = rv_s.rvs(size=(G))
            r_u_bern = stats.bernoulli(0.02).rvs(size=(G))
            r_s_bern = stats.bernoulli(0.02).rvs(size=(G))
            u_top = np.quantile(U, 0.99, 0)
            s_top = np.quantile(S, 0.99, 0)
            
            u0, s0 = u_top*r_u_gamma*r_u_bern, s_top*r_s_gamma*r_s_bern
            self.u0 = nn.Parameter(torch.tensor(np.log(u0+1e-10), device=device).float())
            self.s0 = nn.Parameter(torch.tensor(np.log(s0+1e-10), device=device).float())
        else: #use the mean count of the initial type
            print(f"Setting the root cell to {init_type}")
            cell_labels = adata.obs["clusters"].to_numpy()[train_idx]
            cell_mask = cell_labels==init_type
            self.u0 = nn.Parameter(torch.tensor(np.log((U[cell_mask]/lu[train_idx][cell_mask].reshape(-1,1)).mean(0)+1e-10), device=device).float())
            self.s0 = nn.Parameter(torch.tensor(np.log((S[cell_mask]/ls[train_idx][cell_mask].reshape(-1,1)).mean(0)+1e-10), device=device).float())
            
            tprior = np.ones((adata.n_obs))*tmax*0.5
            tprior[adata.obs["clusters"].to_numpy()==init_type] = 0
            adata.obs['tprior'] = tprior

        self.scaling.requires_grad = False
        self.ton.requires_grad = False
        
        self.fc1 = nn.Linear(dim_z+dim_cond, N1).to(device)
        self.bn1 = nn.BatchNorm1d(num_features=N1).to(device)
        self.dpt1 = nn.Dropout(p=0.2).to(device)
        self.fc2 = nn.Linear(N1, N2).to(device)
        self.bn2 = nn.BatchNorm1d(num_features=N2).to(device)
        self.dpt2 = nn.Dropout(p=0.2).to(device)
        
        self.fc_out1 = nn.Linear(N2, G).to(device)
        
        self.net_rho = nn.Sequential(self.fc1, self.bn1, nn.LeakyReLU(), self.dpt1,
                                     self.fc2, self.bn2, nn.LeakyReLU(), self.dpt2)
        
        self.fc3 = nn.Linear(dim_z+dim_cond, N1).to(device)
        self.bn3 = nn.BatchNorm1d(num_features=N1).to(device)
        self.dpt3 = nn.Dropout(p=0.2).to(device)
        self.fc4 = nn.Linear(N1, N2).to(device)
        self.bn4 = nn.BatchNorm1d(num_features=N2).to(device)
        self.dpt4 = nn.Dropout(p=0.2).to(device)
        
        self.fc_out2 = nn.Linear(N2, G).to(device)
        
        self.net_rho2 = nn.Sequential(self.fc3, self.bn3, nn.LeakyReLU(), self.dpt3,
                                      self.fc4, self.bn4, nn.LeakyReLU(), self.dpt4)
        
        if(checkpoint is not None):
            self.alpha = nn.Parameter(torch.empty(G, device=device).float())
            self.beta = nn.Parameter(torch.empty(G, device=device).float())
            self.gamma = nn.Parameter(torch.empty(G, device=device).float())
            self.scaling = nn.Parameter(torch.empty(G, device=device).float())
            self.ton = nn.Parameter(torch.empty(G, device=device).float())
            self.u0 = nn.Parameter(torch.empty(G, device=device).float())
            self.s0 = nn.Parameter(torch.empty(G, device=device).float())
            self.load_state_dict(torch.load(checkpoint, map_location=device))
        else:
            self.init_weights()
    
    def init_weights(self):
        for m in self.net_rho.modules():
            if(isinstance(m, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
            elif(isinstance(m, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.net_rho2.modules():
            if(isinstance(m, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
            elif(isinstance(m, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        for m in [self.fc_out1, self.fc_out2]:
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.0)
    
    def forward(self, t, z, condition=None, u0=None, s0=None, t0=None, neg_slope=0.0):
        
        
        if(u0 is None or s0 is None or t0 is None):
            if(condition is None):
                rho = torch.sigmoid(self.fc_out1(self.net_rho(z)))
            else:
                rho = torch.sigmoid(self.fc_out1(self.net_rho(torch.cat((z,condition),1))))
            alpha = self.alpha.exp()*rho
            Uhat, Shat = pred_su(F.leaky_relu(t - self.ton.exp(), neg_slope), self.u0.exp()/self.scaling.exp(),  self.s0.exp(), alpha, self.beta.exp(), self.gamma.exp())
        else:
            if(condition is None):
                rho = torch.sigmoid(self.fc_out2(self.net_rho2(z)))
            else:
                rho = torch.sigmoid(self.fc_out2(self.net_rho2(torch.cat((z,condition),1))))
            alpha = self.alpha.exp()*rho
            Uhat, Shat = pred_su(F.leaky_relu(t-t0, neg_slope), u0, s0, alpha, self.beta.exp(), self.gamma.exp())
        Uhat = F.relu(Uhat)*self.scaling.exp()
        Shat = F.relu(Shat)
        vu = alpha - self.beta.exp() * Uhat / torch.exp(self.scaling)
        vs = self.beta.exp() * Uhat / torch.exp(self.scaling) - self.gamma.exp() * Shat
        
        return nn.functional.relu(Uhat), nn.functional.relu(Shat), vu, vs
    
    def pred_rho(self, z, stage, condition=None):
        if(stage==1):
            if(condition is None):
                rho = torch.sigmoid(self.fc_out1(self.net_rho(z)))
            else:
                rho = torch.sigmoid(self.fc_out1(self.net_rho(torch.cat((z,condition),1))))
        else:
            if(condition is None):
                rho = torch.sigmoid(self.fc_out2(self.net_rho2(z)))
            else:
                rho = torch.sigmoid(self.fc_out2(self.net_rho2(torch.cat((z,condition),1))))
        return rho        


class DVAE(VAE):
    """
    Discrete VeloVAE Model
    """
    def __init__(self, 
                 adata, 
                 tmax, 
                 dim_z, 
                 dim_cond=0,
                 device='cpu', 
                 hidden_size=(1000, 500, 500, 1000), 
                 use_raw=True,
                 init_method="steady", 
                 init_key=None,
                 tprior=None, 
                 init_type=None,
                 init_ton_zero=True,
                 add_noise=True,
                 time_distribution="gaussian",
                 count_distribution="Poisson",
                 std_z_prior=0.01,
                 scale_cell=False,
                 separate_us_scale=True,
                 checkpoints=[None, None]):
        
        t_start = time.time()
        self.config = {
            #Model Parameters
            "dim_z":dim_z,
            "hidden_size":hidden_size,
            "tmax":tmax,
            "init_method":init_method,
            "init_key":init_key,
            "tprior":tprior,
            "std_z_prior":std_z_prior,
            "tail":0.01,
            "time_overlap":0.5,
            "n_neighbors":10,
            "dt": (0.03,0.06),
            "knn_adaptive": 0,

            #Training Parameters
            "n_epochs":2000, 
            "n_epochs_post":2000,
            "batch_size":128, 
            "learning_rate":None, 
            "learning_rate_ode":None, 
            "learning_rate_post":None,
            "lambda":1e-3, 
            "lambda_rho":1e-3,
            "kl_t":1.0, 
            "kl_z":1.0, 
            "kl_l":1.0,
            "reg_v":0.0,
            "test_iter":None, 
            "save_epoch":100, 
            "x0_epoch":25,
            "n_warmup":5,
            "early_stop":5,
            "early_stop_thred":adata.n_vars*1e-3,
            "train_test_split":0.7,
            "neg_slope":0.0,
            "k_alt":1, 
            "train_scaling":False, 
            "train_ton":False,
            "train_std":False,
            "weight_sample":False,
            "vel_continuity_loss":False,
            "use_raw":use_raw,
            
            #Normalization Configurations
            "scale_cell":scale_cell,
            "separate_us_scale":separate_us_scale,
            "scale_gene_encoder":True,
            "scale_cell_encoder":False,
            "log1p":True,

            #Plotting
            "sparsify":1
        }
        
        self.set_device(device)
        self.split_train_test(adata.n_obs)
        
        N, G = adata.n_obs, adata.n_vars
        self.dim_z = dim_z
        self.enable_cvae = dim_cond>0
        try:
            self.encoder = encoder(2*G, dim_z, dim_cond, hidden_size[0], hidden_size[1], self.device, checkpoint=checkpoints[0]).float()
        except IndexError:
            print('Please provide two dimensions!')
        
        self.decoder = decoder(adata, 
                               tmax, 
                               self.train_idx,
                               dim_z, 
                               N1=hidden_size[2], 
                               N2=hidden_size[3], 
                               use_raw=self.config['use_raw'],
                               init_ton_zero=init_ton_zero,
                               scale_cell = scale_cell,
                               separate_us_scale = separate_us_scale,
                               add_noise=add_noise,
                               device=self.device, 
                               init_method = init_method,
                               init_key = init_key,
                               init_type = init_type,
                               checkpoint=checkpoints[1]).float()
        
        self.tmax = tmax
        self.time_distribution = time_distribution
        if(init_type is not None):
            tprior = 'tprior'
            self.config['tprior'] = tprior
            self.config['train_ton'] = False
        self.get_prior(adata, time_distribution, tmax, tprior)
        
        self.p_z = torch.stack([torch.zeros(N,dim_z), torch.ones(N,dim_z)*self.config["std_z_prior"]]).float().to(self.device)
        
        self.use_knn = False
        self.neighbor_idx, self.neighbor_idx_fw = None, None
        self.u0 = None
        self.s0 = None
        self.t0 = None
        self.lu_scale = torch.tensor(np.log(adata.obs['library_scale_u'].to_numpy())).unsqueeze(-1).float().to(self.device)
        self.ls_scale = torch.tensor(np.log(adata.obs['library_scale_s'].to_numpy())).unsqueeze(-1).float().to(self.device)
        
        print(f"Library scale (U): Max={np.exp(self.lu_scale.detach().cpu().numpy().max()):.2f}, Min={np.exp(self.lu_scale.detach().cpu().numpy().min()):.2f}, Mean={np.exp(self.lu_scale.detach().cpu().numpy().mean()):.2f}")
        print(f"Library scale (S): Max={np.exp(self.ls_scale.detach().cpu().numpy().max()):.2f}, Min={np.exp(self.ls_scale.detach().cpu().numpy().min()):.2f}, Mean={np.exp(self.ls_scale.detach().cpu().numpy().mean()):.2f}")
        
        #Determine Count Distribution
        dispersion_u = adata.var["dispersion_u"].to_numpy()
        dispersion_s = adata.var["dispersion_s"].to_numpy()
        if(count_distribution=="auto"):
            p_nb = np.sum((dispersion_u>1) & (dispersion_s>1))/adata.n_vars
            if(p_nb > 0.5):
                count_distribution = "NB"
                self.vae_risk = self.vae_risk_nb
            else:
                count_distribution = "Poisson"
                self.vae_risk = self.vae_risk_poisson
            print(f"Mean dispersion: u={dispersion_u.mean():.2f}, s={dispersion_s.mean():.2f}")
            print(f"Over-Dispersion = {p_nb:.2f} => Using {count_distribution} to model count data.")
        elif(count_distribution=="NB"):
            self.vae_risk = self.vae_risk_nb
        else:
            self.vae_risk = self.vae_risk_poisson
        
        mean_u = adata.var["mean_u"].to_numpy()
        mean_s = adata.var["mean_s"].to_numpy()
        dispersion_u[dispersion_u<1] = 1.001
        dispersion_s[dispersion_s<1] = 1.001
        self.eta_u = torch.tensor(np.log(dispersion_u-1)-np.log(mean_u)).float().to(self.device)
        self.eta_s = torch.tensor(np.log(dispersion_s-1)-np.log(mean_s)).float().to(self.device)
        

        #Class attributes for training
        self.loss_train, self.loss_test = [],[]
        self.logp_list, self.kldt_list, self.kldz_list, self.kldparam_list = [],[],[],None
        self.counter = 0 #Count the number of iterations
        self.n_drop = 0 #Count the number of consecutive iterations with little decrease in loss
        self.train_stage = 1
        
        self.timer = time.time()-t_start
    
    def set_mode(self,mode):
        #Set the model to either training or evaluation mode.
        
        if(mode=='train'):
            self.encoder.train()
            self.decoder.train()
            self.decoder.alpha.requires_grad = True
            self.decoder.beta.requires_grad = True
            self.decoder.gamma.requires_grad = True
            self.decoder.u0.requires_grad = True
            self.decoder.s0.requires_grad = True
        elif(mode=='eval'):
            self.encoder.eval()
            self.decoder.eval()
            self.decoder.alpha.requires_grad = False
            self.decoder.beta.requires_grad = False
            self.decoder.gamma.requires_grad = False
            self.decoder.u0.requires_grad = False
            self.decoder.s0.requires_grad = False
        else:
            print("Warning: mode not recognized. Must be 'train' or 'test'! ")
    
    def forward(self, data_in, lu_scale, ls_scale, u0=None, s0=None, t0=None, t1=None, condition=None):
        data_in_scale = data_in
        if(self.config["scale_gene_encoder"]):
            data_in_scale = torch.cat((data_in_scale[:,:data_in_scale.shape[1]//2]/self.decoder.scaling.exp(), data_in_scale[:,data_in_scale.shape[1]//2:]),1)
        if(self.config["scale_cell_encoder"]):
            data_in_scale = torch.cat((data_in_scale[:,:data_in_scale.shape[1]//2]/lu_scale, data_in_scale[:,data_in_scale.shape[1]//2:]/ls_scale),1)
        if(self.config["log1p"]):
            data_in_scale = torch.log1p(data_in_scale)
        mu_t, std_t, mu_z, std_z = self.encoder.forward(data_in_scale, condition)
        t = self.sample(mu_t, std_t)
        z = self.sample(mu_z, std_z)

        uhat, shat, vu, vs = self.decoder.forward(t, z, condition, u0, s0, t0, neg_slope=self.config["neg_slope"])
        if(t1 is not None):
            uhat_fw, shat_fw, vu_fw, vs_fw = self.decoder.forward(t1, z, condition, uhat, shat, t, neg_slope=self.config["neg_slope"])
        else:
            uhat_fw, shat_fw, vu_fw, vs_fw = None, None, None, None

        return mu_t, std_t, mu_z, std_z, t, z, uhat, shat, uhat_fw, shat_fw, vu, vs, vu_fw, vs_fw
    
    def eval_model(self, data_in, lu_scale, ls_scale, u0=None, s0=None, t0=None, t1=None, condition=None, continuous=True):
        data_in_scale = data_in
        if(self.config["scale_gene_encoder"]):
            data_in_scale = torch.cat((data_in_scale[:,:data_in_scale.shape[1]//2]/self.decoder.scaling.exp(), data_in_scale[:,data_in_scale.shape[1]//2:]),1)
        if(self.config["scale_cell_encoder"]):
            data_in_scale = torch.cat((data_in_scale[:,:data_in_scale.shape[1]//2]/lu_scale, data_in_scale[:,data_in_scale.shape[1]//2:]/ls_scale),1)
        if(self.config["log1p"]):
            data_in_scale = torch.log1p(data_in_scale)
        mu_t, std_t, mu_z, std_z = self.encoder.forward(data_in_scale, condition)
        uhat, shat, vu, vs = self.decoder.forward(mu_t, mu_z, condition, u0=u0, s0=s0, t0=t0, neg_slope=0.0)
        
        if(t1 is not None):
            uhat_fw, shat_fw, vu_fw, vs_fw = self.decoder.forward(t1, mu_z, condition, uhat, shat, mu_t, neg_slope=self.config["neg_slope"])
        else:
            uhat_fw, shat_fw, vu_fw, vs_fw = None, None, None, None
        if(not continuous):
            poisson_u = Poisson(F.softplus(uhat, beta=100))
            poisson_s = Poisson(F.softplus(shat, beta=100))
            if(t1 is not None):
                poisson_u = Poisson(F.softplus(uhat_fw, beta=100))
                poisson_s = Poisson(F.softplus(shat_fw, beta=100))
                u_out_fw = poisson_u_fw.sample()
                s_out_fw = poisson_s_fw.sample()
            else:
                u_out_fw, s_out_fw = None, None
            u_out = poisson_u.sample()
            s_out = poisson_s.sample()
            
            return mu_t, std_t, mu_z, std_z, u_out, s_out, u_out_fw, s_out_fw, vu, vs, vu_fw, vs_fw
        return mu_t, std_t, mu_z, std_z, uhat, shat, uhat_fw, shat_fw, vu, vs, vu_fw, vs_fw
    
    def update_x0(self, U, S):
        start = time.time()
        self.set_mode('eval')
        out, elbo = self.pred_all(np.concatenate((U,S),1), self.cell_labels, "both", ["uhat","shat","t","z"], gene_idx=np.array(range(U.shape[1])))
        t, std_t, z = out[4], out[5], out[6]
        #Clip the time to avoid outliers
        t = np.clip(t, 0, np.quantile(t, 0.99))
        dt = (self.config["dt"][0]*(t.max()-t.min()), self.config["dt"][1]*(t.max()-t.min()))
        scaling = np.exp(self.decoder.scaling.detach().cpu().numpy())
        if(self.neighbor_idx is None):
            self.neighbor_idx = knnx0_index(out[0][self.train_idx]/scaling, 
                                            out[2][self.train_idx], 
                                            t[self.train_idx], 
                                            z[self.train_idx], 
                                            t, 
                                            z, 
                                            dt, 
                                            self.config["n_neighbors"], 
                                            adaptive=self.config["knn_adaptive"], 
                                            std_t=std_t)
        u0, s0, t0 = get_x0(out[0][self.train_idx]/scaling, out[2][self.train_idx], t[self.train_idx], self.neighbor_idx)
        u1,s1,t1=None,None,None
        if(self.config["vel_continuity_loss"] and self.neighbor_idx_fw is None):
            self.neighbor_idx_fw = knnx0_index(out[0][self.train_idx]/scaling, 
                                               out[2][self.train_idx], 
                                               t[self.train_idx], 
                                               z[self.train_idx], 
                                               t, 
                                               z, 
                                               dt, 
                                               self.config["n_neighbors"], 
                                               adaptive=self.config["knn_adaptive"], 
                                               std_t=std_t,
                                               forward=True)
            u1, s1, t1 = get_x0(out[0][self.train_idx]/scaling, out[2][self.train_idx], t[self.train_idx], self.neighbor_idx_fw)
            t1 = t1.reshape(-1,1)
        self.set_mode('train')
        return u0, s0, t0.reshape(-1,1), u1, s1, t1
    
    
    def sample_poisson(self, uhat, shat):
        uhat[torch.isnan(uhat)] = 1e-3
        shat[torch.isnan(shat)] = 1e-3
        u_sample = torch.poisson(uhat)
        s_sample = torch.poisson(shat)
        return u_sample.cpu(), s_sample.cpu()
    
    def sample_nb(self, uhat, shat, pu, ps):
        u_nb = NegativeBinomial(uhat*(1-pu)/pu, pu.repeat(uhat.shape[0],1))
        s_nb = NegativeBinomial(shat*(1-ps)/ps, ps.repeat(shat.shape[0],1))
        return u_nb.sample(), s_nb.sample()
    
    def loss_vel(self, x0, xhat, v):
        cossim = nn.CosineSimilarity(dim=1)
        return cossim(xhat-x0, v).mean()
    
    def kl_poisson(self, lamb_1, lamb_2):
        return lamb_1 * (torch.log(lamb_1) - torch.log(lamb_2)) + lamb_2 - lamb_1
    
    def vae_risk_poisson(self, 
                         q_tx, p_t, 
                         q_zx, p_z, 
                         u, s, uhat, shat, 
                         u1=None, s1=None, uhat_fw=None, shat_fw=None,
                         weight=None,
                         eps=1e-2):
        kldt = self.kl_time(q_tx[0], q_tx[1], p_t[0], p_t[1], tail=self.config["tail"])
        kldz = kl_gaussian(q_zx[0], q_zx[1], p_z[0], p_z[1])
        
        #poisson
        mask_u = (torch.isnan(uhat) | torch.isinf(uhat)).float()
        uhat = uhat * (1-mask_u)
        mask_s = (torch.isnan(shat) | torch.isinf(shat)).float()
        shat = shat * (1-mask_s)
        
        poisson_u = Poisson(F.relu(uhat)+1e-2)
        poisson_s = Poisson(F.relu(shat)+1e-2)
        
        logp = poisson_u.log_prob(u) + poisson_s.log_prob(s)
        #velocity continuity loss
        if(uhat_fw is not None and shat_fw is not None):
            logp = logp - self.kl_poisson(u1, uhat_fw) - self.kl_poisson(s1, shat_fw)
        if( weight is not None):
            logp = logp*weight
        
        err_rec = torch.mean(logp.sum(1))
        
        return err_rec, kldt, kldz, None
        
    
    def vae_risk_nb(self, 
                    q_tx, p_t, 
                    q_zx, p_z, 
                    u, s, uhat, shat, 
                    weight=None,
                    eps=1e-6):
        #This is the negative ELBO.
        kldt = self.kl_time(q_tx[0], q_tx[1], p_t[0], p_t[1], tail=self.config["tail"])
        kldz = kl_gaussian(q_zx[0], q_zx[1], p_z[0], p_z[1])
        
        #NB
        p_nb_u = torch.sigmoid(self.eta_u+torch.log(F.relu(uhat)+1e-10))
        p_nb_s = torch.sigmoid(self.eta_s+torch.log(F.relu(shat)+1e-10))
        nb_u = NegativeBinomial((F.relu(uhat)+1e-10)*(1-p_nb_u)/p_nb_u, probs=p_nb_u)
        nb_s = NegativeBinomial((F.relu(shat)+1e-10)*(1-p_nb_s)/p_nb_s, probs=p_nb_s)
        logp = nb_u.log_prob(u) + nb_s.log_prob(s)
        
        if( weight is not None):
            logp = logp*weight
        err_rec = torch.mean(torch.sum(logp,1))

        return err_rec, kldt, kldz, None
    
    def train_epoch(self, train_loader, test_set, optimizer, optimizer2=None, K=1):
        B = len(train_loader)
        self.set_mode('train')
        stop_training = False
            
        for i, batch in enumerate(train_loader):
            if(self.counter == 1 or self.counter % self.config["test_iter"] == 0):
                elbo_test = self.test(test_set, 
                                      None, 
                                      self.counter, 
                                      True)
                if(len(self.loss_test)>0): #update the number of epochs with dropping ELBO
                    if(elbo_test - self.loss_test[-1] <= self.config["early_stop_thred"]):
                        self.n_drop = self.n_drop+1
                    else:
                        self.n_drop = 0
                self.loss_test.append(elbo_test)
                self.set_mode('train')
                
                if(self.n_drop>=self.config["early_stop"] and self.config["early_stop"]>0):
                    stop_training = True
                    break
            
            optimizer.zero_grad()
            if(optimizer2 is not None):
                optimizer2.zero_grad()
            
            xbatch, idx = batch[0].float().to(self.device), batch[3]
            u = xbatch[:,:xbatch.shape[1]//2]
            s = xbatch[:,xbatch.shape[1]//2:]
            
            u0 = torch.tensor(self.u0[self.train_idx[idx]], device=self.device, requires_grad = True) if self.use_knn else None
            s0 = torch.tensor(self.s0[self.train_idx[idx]], device=self.device, requires_grad = True) if self.use_knn else None
            t0 = torch.tensor(self.t0[self.train_idx[idx]], device=self.device, requires_grad = True) if self.use_knn else None
            u1 = torch.tensor(self.u1[self.train_idx[idx]], device=self.device, requires_grad = True) if (self.use_knn and self.config["vel_continuity_loss"]) else None
            s1 = torch.tensor(self.s1[self.train_idx[idx]], device=self.device, requires_grad = True) if (self.use_knn and self.config["vel_continuity_loss"]) else None
            t1 = torch.tensor(self.t1[self.train_idx[idx]], device=self.device, requires_grad = True) if (self.use_knn and self.config["vel_continuity_loss"]) else None
            lu_scale = self.lu_scale[self.train_idx[idx]].exp()
            ls_scale = self.ls_scale[self.train_idx[idx]].exp()
            
            
            condition = F.one_hot(batch[1].to(self.device), self.n_type).float() if self.enable_cvae else None
            mu_tx, std_tx, mu_zx, std_zx, t, z, uhat, shat, uhat_fw, shat_fw, vu, vs, vu_fw, vs_fw = self.forward(xbatch, lu_scale, ls_scale, u0, s0, t0, t1, condition)
            
            
            err_rec, kldt, kldz, kld_param = self.vae_risk((mu_tx, std_tx), self.p_t[:,self.train_idx[idx],:],
                                                            (mu_zx, std_zx), self.p_z[:,self.train_idx[idx],:],
                                                            u.int(),
                                                            s.int(), 
                                                            uhat*lu_scale, shat*ls_scale,
                                                            u1, s1, uhat_fw, shat_fw,
                                                            None)
            loss = - err_rec + self.config["kl_t"]*kldt + self.config["kl_z"]*kldz
            
            if(kld_param is not None):
                loss = loss + self.config["kl_param"]*kld_param
            if(self.use_knn and self.config['reg_v']>0):
                loss = loss - self.config["reg_v"] * (self.loss_vel(u0, uhat, vu) + self.loss_vel(s0, shat, vs))
                if(vu_fw is not None and vs_fw is not None):
                    loss = loss - self.config["reg_v"] * (self.loss_vel(uhat, uhat_fw, vu_fw) + self.loss_vel(shat, shat_fw, vs_fw))
            loss.backward()
            if(K==0):
                optimizer.step()
                if( optimizer2 is not None ):
                    optimizer2.step()
            else:
                if( optimizer2 is not None and ((i+1) % (K+1) == 0 or i==B-1)):
                    optimizer2.step()
                else:
                    optimizer.step()
            
            self.loss_train.append(loss.detach().cpu().item())
            self.counter = self.counter + 1
        return stop_training
    
    def train(self, 
              adata, 
              config={}, 
              plot=False, 
              gene_plot=[], 
              cluster_key = "clusters",
              figure_path = "figures", 
              embed="umap"):
        start = time.time()
        self.train_mse = []
        self.test_mse = []
        
        self.load_config(config)
        if(self.config["learning_rate"] is None):
            p = (np.sum(adata.layers["unspliced"].A>0) + (np.sum(adata.layers["spliced"].A>0)))/adata.n_obs/adata.n_vars/2
            self.config["learning_rate"] = 10**(-8.3*p-2.25)
            self.config["learning_rate_post"] = self.config["learning_rate"]
            self.config["learning_rate_ode"] = 5*self.config["learning_rate"]
        
        print("--------------------------- Train a VeloVAE ---------------------------")
        #Get data loader
        if(self.config['use_raw']):
            U, S = adata.layers['unspliced'].A.astype(float), adata.layers['spliced'].A.astype(float)
        else:
            U, S = adata.layers['Cu'], adata.layers['Cs']
        X = np.concatenate((U,S), 1)
        
        try:
            Xembed = adata.obsm[f"X_{embed}"]
        except KeyError:
            print("Embedding not found! Set to None.")
            Xembed = None
            plot = False
        
        cell_labels_raw = adata.obs[cluster_key].to_numpy() if cluster_key in adata.obs else np.array(['Unknown' for i in range(adata.n_obs)])
        #Encode the labels
        cell_types_raw = np.unique(cell_labels_raw)
        self.label_dic, self.label_dic_rev = encode_type(cell_types_raw)

        self.n_type = len(cell_types_raw)
        self.cell_labels = np.array([self.label_dic[x] for x in cell_labels_raw])
        self.cell_types = np.array([self.label_dic[cell_types_raw[i]] for i in range(self.n_type)])
        
        print("*********        Creating Training/Validation Datasets        *********")
        train_set = SCData(X[self.train_idx], self.cell_labels[self.train_idx], self.decoder.Rscore[self.train_idx]) if self.config['weight_sample'] else SCData(X[self.train_idx], self.cell_labels[self.train_idx])
        test_set = None
        if(len(self.test_idx)>0):
            test_set = SCData(X[self.test_idx], self.cell_labels[self.test_idx], self.decoder.Rscore[self.test_idx]) if self.config['weight_sample'] else SCData(X[self.test_idx], self.cell_labels[self.test_idx])
        data_loader = torch.utils.data.DataLoader(train_set, batch_size=self.config["batch_size"], shuffle=True)
        #Automatically set test iteration if not given
        if(self.config["test_iter"] is None):
            self.config["test_iter"] = len(self.train_idx)//self.config["batch_size"]*2
        print("*********                      Finished.                      *********")
        
        gind, gene_plot = get_gene_index(adata.var_names, gene_plot)
        os.makedirs(figure_path, exist_ok=True)
        
        #define optimizer
        print("*********                 Creating optimizers                 *********")
        param_nn = list(self.encoder.parameters())+list(self.decoder.net_rho.parameters())+list(self.decoder.fc_out1.parameters())
        param_ode = [self.decoder.alpha, self.decoder.beta, self.decoder.gamma, self.decoder.u0, self.decoder.s0] 
        if(self.config['train_ton']):
            self.decoder.ton.requires_grad = True
            param_ode.append(self.decoder.ton)
        if(self.config['train_scaling']):
            self.decoder.scaling.requires_grad = True
            param_ode = param_ode+[self.decoder.scaling]

        optimizer = torch.optim.Adam(param_nn, lr=self.config["learning_rate"], weight_decay=self.config["lambda"])
        optimizer_ode = torch.optim.Adam(param_ode, lr=self.config["learning_rate_ode"])
        print("*********                      Finished.                      *********")
      
        #Main Training Process
        print("*********                    Start training                   *********")
        print("*********                      Stage  1                       *********")
        print(f"Total Number of Iterations Per Epoch: {len(data_loader)}, test iteration: {self.config['test_iter']}")
        
        n_epochs = self.config["n_epochs"]
        
        for epoch in range(n_epochs):
            #Train the encoder
            if(self.config["k_alt"] is None):
                stop_training = self.train_epoch(data_loader, test_set, optimizer)
                
                if(epoch>=self.config["n_warmup"]):
                    stop_training_ode = self.train_epoch(data_loader, test_set, optimizer_ode)
                    if(stop_training_ode):
                        print(f"*********       Stage 1: Early Stop Triggered at epoch {epoch+1}.       *********")
                        break
            else:
                if(epoch>=self.config["n_warmup"]):
                    stop_training = self.train_epoch(data_loader, test_set, optimizer_ode, optimizer, self.config["k_alt"])
                else:
                    stop_training = self.train_epoch(data_loader, test_set, optimizer, None, self.config["k_alt"])
            
            if(plot and (epoch==0 or (epoch+1) % self.config["save_epoch"] == 0)):
                elbo_train = self.test(train_set,
                                       Xembed[self.train_idx],
                                       f"train{epoch+1}", 
                                       False,
                                       gind, 
                                       gene_plot,
                                       plot, 
                                       figure_path)
                self.set_mode('train')
                elbo_test = self.loss_test[-1] if len(self.loss_test)>0 else -np.inf
                print(f"Epoch {epoch+1}: Train ELBO = {elbo_train:.3f}, Test ELBO = {elbo_test:.3f}, \t Total Time = {convert_time(time.time()-start)}")
            
            if(stop_training):
                print(f"*********       Stage 1: Early Stop Triggered at epoch {epoch+1}.       *********")
                break
        
        n_stage1 = epoch+1
        n_test1 = len(self.loss_test)
        
        print("*********                      Stage  2                       *********")
        u0, s0, t0, u1, s1, t1 = self.update_x0(U, S)
        self.u0 = u0
        self.s0 = s0
        self.t0 = t0
        if(self.config['vel_continuity_loss']):
            self.u1 = u1
            self.s1 = s1
            self.t1 = t1
        
        self.use_knn = True
        self.decoder.init_weights()
        self.n_drop = 0
        self.train_stage = 2
        
        #param_ode = [self.decoder.alpha, self.decoder.beta, self.decoder.gamma] 
        #optimizer_ode = torch.optim.Adam(param_ode, lr=self.config["learning_rate_ode"])
        param_post = list(self.decoder.net_rho2.parameters())+list(self.decoder.fc_out2.parameters())
        optimizer_post = torch.optim.Adam(param_post, lr=self.config["learning_rate_post"], weight_decay=self.config["lambda_rho"])
        for epoch in range(self.config["n_epochs_post"]):
            if(self.config["k_alt"] is None):
                stop_training = self.train_epoch(data_loader, test_set, optimizer_post)
                
                if(epoch>=self.config["n_warmup"]):
                    stop_training_ode = self.train_epoch(data_loader, test_set, optimizer_ode)
                    if(stop_training_ode):
                        print(f"*********       Stage 2: Early Stop Triggered at epoch {epoch+n_stage1+1}.       *********")
                        break
            else:
                if(epoch>=self.config["n_warmup"]):
                    stop_training = self.train_epoch(data_loader, test_set, optimizer_post, optimizer_ode, self.config["k_alt"])
                else:
                    stop_training = self.train_epoch(data_loader, test_set, optimizer_post, None, self.config["k_alt"])
            
            if((epoch+1) % self.config["x0_epoch"] == 0):
                u0, s0, t0, u1, s1, t1 = self.update_x0(U, S)
                self.u0 = u0
                self.s0 = s0
                self.t0 = t0
            
            if(plot and (epoch==0 or (epoch+n_stage1+1) % self.config["save_epoch"] == 0)):
                elbo_train = self.test(train_set,
                                       Xembed[self.train_idx],
                                       f"train{epoch+n_stage1+1}", 
                                       False,
                                       gind, 
                                       gene_plot,
                                       plot, 
                                       figure_path)
                self.decoder.train()
                elbo_test = self.loss_test[-1] if len(self.loss_test)>n_test1 else -np.inf
                print(f"Epoch {epoch+n_stage1+1}: Train ELBO = {elbo_train:.3f}, Test ELBO = {elbo_test:.3f}, \t Total Time = {convert_time(time.time()-start)}")
            
            if(stop_training):
                print(f"*********       Stage 2: Early Stop Triggered at epoch {epoch+n_stage1+1}.       *********")
                break
        
        #Plot final results
        elbo_train = self.test(train_set,
                               Xembed[self.train_idx],
                               "final-train", 
                               False,
                               gind, 
                               gene_plot,
                               plot, 
                               figure_path)
        elbo_test = self.test(test_set,
                              Xembed[self.test_idx],
                              "final-test", 
                              True,
                              gind, 
                              gene_plot,
                              plot, 
                              figure_path)
        self.loss_train.append(elbo_train)
        self.loss_test.append(elbo_test)
        if(plot):
            plot_train_loss(self.loss_train, range(1,len(self.loss_train)+1), save=f'{figure_path}/train_loss_velovae.png')
            if(self.config["test_iter"]>0):
                n_skip = len(self.loss_test) // 4
                iters = [i*self.config["test_iter"] for i in range(n_skip+1,len(self.loss_test)+1)]
                plot_test_loss(self.loss_test[n_skip:], iters, save=f'{figure_path}/test_loss_velovae.png')
                if(self.kldparam_list is None):
                    plot_loss_split(self.logp_list[n_skip:], self.kldt_list[n_skip:], self.kldz_list[n_skip:], None, iters, save=f'{figure_path}/test_loss_split_velovae.png')
                else:
                    plot_loss_split(self.logp_list[n_skip:], self.kldt_list[n_skip:], self.kldz_list[n_skip:], self.kldparam_list[n_skip:], iters, save=f'{figure_path}/test_loss_split_fullvb.png')
            plot_train_loss(self.train_mse, range(1,len(self.train_mse)+1), save=f'{figure_path}/train_mse_velovae.png')
            if(self.config["test_iter"]>0):
                iters = [i*self.config["test_iter"] for i in range(1,len(self.test_mse)+1)]
                plot_test_loss(self.test_mse, iters, save=f'{figure_path}/test_mse_velovae.png')
        
        self.timer = self.timer+(time.time()-start)
        print(f"*********              Finished. Total Time = {convert_time(self.timer)}             *********")
        print(f"Final: Train ELBO = {elbo_train:.3f},           Test ELBO = {elbo_test:.3f}")
        print(f"       Training MSE = {self.train_mse[-1]:.3f}, Test MSE = {self.test_mse[-1]:.3f}")
        return elbo_train, elbo_test, self.train_mse[-1], self.test_mse[-1]
    
    def pred_all(self, data, cell_labels, mode='test', output=["uhat", "shat", "t", "z"], gene_idx=None, continuous=True):
        N, G = data.shape[0], data.shape[1]//2
        elbo, mse = 0, 0
        if("uhat" in output):
            Uhat = None if gene_idx is None else np.zeros((N,len(gene_idx)))
            U_sum = np.zeros((N,))
        if("shat" in output):
            Shat = None if gene_idx is None else np.zeros((N,len(gene_idx)))
            S_sum = np.zeros((N,))
        if("t" in output):
            t_out = np.zeros((N))
            std_t_out = np.zeros((N))
        if("z" in output):
            z_out = np.zeros((N, self.dim_z))
            std_z_out = np.zeros((N, self.dim_z))
        
        logp_total = 0
        kldt_total = 0
        kldz_total = 0
        kldparam_total = 0
        
        with torch.no_grad():
            B = min(N//5, 5000)
            Nb = N // B
            has_init_cond = (self.train_stage > 1)
            for i in range(Nb):
                data_in = torch.tensor(data[i*B:(i+1)*B]).float().to(self.device)
                if(mode=="test"):
                    u0 = torch.tensor(self.u0[self.test_idx[i*B:(i+1)*B]], dtype=torch.float, device=self.device) if has_init_cond else None
                    s0 = torch.tensor(self.s0[self.test_idx[i*B:(i+1)*B]], dtype=torch.float, device=self.device) if has_init_cond else None
                    t0 = torch.tensor(self.t0[self.test_idx[i*B:(i+1)*B]], dtype=torch.float, device=self.device) if has_init_cond else None
                    u1 = torch.tensor(self.u1[self.test_idx[i*B:(i+1)*B]], dtype=torch.float, device=self.device) if (has_init_cond and self.config['vel_continuity_loss']) else None
                    s1 = torch.tensor(self.s1[self.test_idx[i*B:(i+1)*B]], dtype=torch.float, device=self.device) if (has_init_cond and self.config['vel_continuity_loss']) else None
                    t1 = torch.tensor(self.t1[self.test_idx[i*B:(i+1)*B]], dtype=torch.float, device=self.device) if (has_init_cond and self.config['vel_continuity_loss']) else None
                    y_onehot = F.one_hot(torch.tensor(cell_labels[self.test_idx[i*B:(i+1)*B]], dtype=int, device=self.device), self.n_type).float() if self.enable_cvae else None
                    p_t = self.p_t[:,self.test_idx[i*B:(i+1)*B],:]
                    p_z = self.p_z[:,self.test_idx[i*B:(i+1)*B],:]
                    lu_scale = self.lu_scale[self.test_idx[i*B:(i+1)*B],:].exp()
                    ls_scale = self.ls_scale[self.test_idx[i*B:(i+1)*B],:].exp()
                    mu_tx, std_tx, mu_zx, std_zx, uhat, shat, uhat_fw, shat_fw, vu, vs, vu_fw, vs_fw = self.eval_model(data_in, lu_scale, ls_scale, u0, s0, t0, t1, y_onehot, continuous)
                elif(mode=="train"):
                    u0 = torch.tensor(self.u0[self.train_idx[i*B:(i+1)*B]], dtype=torch.float, device=self.device) if has_init_cond else None
                    s0 = torch.tensor(self.s0[self.train_idx[i*B:(i+1)*B]], dtype=torch.float, device=self.device) if has_init_cond else None
                    t0 = torch.tensor(self.t0[self.train_idx[i*B:(i+1)*B]], dtype=torch.float, device=self.device) if has_init_cond else None
                    u1 = torch.tensor(self.u1[self.train_idx[i*B:(i+1)*B]], dtype=torch.float, device=self.device) if (has_init_cond and self.config['vel_continuity_loss']) else None
                    s1 = torch.tensor(self.s1[self.train_idx[i*B:(i+1)*B]], dtype=torch.float, device=self.device) if (has_init_cond and self.config['vel_continuity_loss']) else None
                    t1 = torch.tensor(self.t1[self.train_idx[i*B:(i+1)*B]], dtype=torch.float, device=self.device) if (has_init_cond and self.config['vel_continuity_loss']) else None
                    y_onehot = F.one_hot(torch.tensor(cell_labels[self.train_idx[i*B:(i+1)*B]], dtype=int, device=self.device), self.n_type).float() if self.enable_cvae else None
                    p_t = self.p_t[:,self.train_idx[i*B:(i+1)*B],:]
                    p_z = self.p_z[:,self.train_idx[i*B:(i+1)*B],:]
                    lu_scale = self.lu_scale[self.train_idx[i*B:(i+1)*B],:].exp()
                    ls_scale = self.ls_scale[self.train_idx[i*B:(i+1)*B],:].exp()
                    mu_tx, std_tx, mu_zx, std_zx, uhat, shat, uhat_fw, shat_fw, vu, vs, vu_fw, vs_fw = self.eval_model(data_in, lu_scale, ls_scale, u0, s0, t0, t1, y_onehot, continuous)
                else:
                    u0 = torch.tensor(self.u0[i*B:(i+1)*B], dtype=torch.float, device=self.device) if has_init_cond else None
                    s0 = torch.tensor(self.s0[i*B:(i+1)*B], dtype=torch.float, device=self.device) if has_init_cond else None
                    t0 = torch.tensor(self.t0[i*B:(i+1)*B], dtype=torch.float, device=self.device) if has_init_cond else None
                    u1 = torch.tensor(self.u1[i*B:(i+1)*B], dtype=torch.float, device=self.device) if (has_init_cond and self.config['vel_continuity_loss']) else None
                    s1 = torch.tensor(self.s1[i*B:(i+1)*B], dtype=torch.float, device=self.device) if (has_init_cond and self.config['vel_continuity_loss']) else None
                    t1 = torch.tensor(self.t1[i*B:(i+1)*B], dtype=torch.float, device=self.device) if (has_init_cond and self.config['vel_continuity_loss']) else None
                    y_onehot = F.one_hot(torch.tensor(cell_labels[i*B:(i+1)*B], dtype=int, device=self.device), self.n_type).float() if self.enable_cvae else None
                    p_t = self.p_t[:,i*B:(i+1)*B,:]
                    p_z = self.p_z[:,i*B:(i+1)*B,:]
                    lu_scale = self.lu_scale[i*B:(i+1)*B,:].exp()
                    ls_scale = self.ls_scale[i*B:(i+1)*B,:].exp()
                    mu_tx, std_tx, mu_zx, std_zx, uhat, shat, uhat_fw, shat_fw, vu, vs, vu_fw, vs_fw = self.eval_model(data_in, lu_scale, ls_scale, u0, s0, t0, t1, y_onehot, continuous)

                u_raw, s_raw = data_in[:,:G], data_in[:,G:]
                u_sum, s_sum = torch.sum(uhat, 1, keepdim=True), torch.sum(shat, 1, keepdim=True)
                
                logp, kldt, kldz, kld_param = self.vae_risk((mu_tx, std_tx), p_t,
                                                            (mu_zx, std_zx), p_z,
                                                            u_raw.int(), s_raw.int(), 
                                                            uhat*lu_scale, shat*ls_scale, 
                                                            u1, s1, uhat_fw, shat_fw,
                                                            None)
                loss = (- logp + self.config["kl_t"] * kldt + self.config["kl_z"] * kldz).cpu().item()
                if(kld_param is not None):
                    loss = loss+self.config["kl_param"]*kld_param.cpu().item()
                
                mse_batch = np.mean( (u_raw.cpu().numpy() - (uhat*lu_scale).cpu().numpy())**2 + ( s_raw.cpu().numpy() - (shat*ls_scale).cpu().numpy())**2)
                elbo = elbo - (B/N)*loss
                
                mse = mse + (B/N) * mse_batch
                logp_total = logp_total + (B/N) * logp.detach().item()
                kldt_total = kldt_total + (B/N) * kldt.detach().item()
                kldz_total = kldz_total + (B/N) * kldz.detach().item()
                if self.kldparam_list is not None:
                    kldparam_total = kldparam_total + (B/N) * kld_param.detach().item()
                
                if("uhat" in output and gene_idx is not None):
                    Uhat[i*B:(i+1)*B] = uhat[:,gene_idx].cpu().numpy()
                    U_sum[i*B:(i+1)*B] = u_sum.cpu().numpy().squeeze()
                if("shat" in output and gene_idx is not None):
                    Shat[i*B:(i+1)*B] = shat[:,gene_idx].cpu().numpy()
                    S_sum[i*B:(i+1)*B] = s_sum.cpu().numpy().squeeze()
                if("t" in output):
                    t_out[i*B:(i+1)*B] = mu_tx.cpu().squeeze().numpy()
                    std_t_out[i*B:(i+1)*B] = std_tx.cpu().squeeze().numpy()
                if("z" in output):
                    z_out[i*B:(i+1)*B] = mu_zx.cpu().numpy()
                    std_z_out[i*B:(i+1)*B] = std_zx.cpu().numpy()
                
            if(N > B*Nb):
                data_in = torch.tensor(data[Nb*B:]).float().to(self.device)
                if(mode=="test"):
                    u0 = torch.tensor(self.u0[self.test_idx[Nb*B:]], dtype=torch.float, device=self.device) if has_init_cond else None
                    s0 = torch.tensor(self.s0[self.test_idx[Nb*B:]], dtype=torch.float, device=self.device) if has_init_cond else None
                    t0 = torch.tensor(self.t0[self.test_idx[Nb*B:]], dtype=torch.float, device=self.device) if has_init_cond else None
                    u1 = torch.tensor(self.u1[self.test_idx[Nb*B:]], dtype=torch.float, device=self.device) if (has_init_cond and self.config['vel_continuity_loss']) else None
                    s1 = torch.tensor(self.s1[self.test_idx[Nb*B:]], dtype=torch.float, device=self.device) if (has_init_cond and self.config['vel_continuity_loss']) else None
                    t1 = torch.tensor(self.t1[self.test_idx[Nb*B:]], dtype=torch.float, device=self.device) if (has_init_cond and self.config['vel_continuity_loss']) else None
                    y_onehot = F.one_hot(torch.tensor(cell_labels[self.test_idx[Nb*B:]], dtype=int, device=self.device), self.n_type).float() if self.enable_cvae else None
                    p_t = self.p_t[:,self.test_idx[Nb*B:],:]
                    p_z = self.p_z[:,self.test_idx[Nb*B:],:]
                    lu_scale = self.lu_scale[self.test_idx[Nb*B:],:].exp()
                    ls_scale = self.ls_scale[self.test_idx[Nb*B:],:].exp()
                    mu_tx, std_tx, mu_zx, std_zx, uhat, shat, uhat_fw, shat_fw, vu, vs, vu_fw, vs_fw = self.eval_model(data_in, lu_scale, ls_scale, u0, s0, t0, t1, y_onehot)
                elif(mode=="train"):
                    u0 = torch.tensor(self.u0[self.train_idx[Nb*B:]], dtype=torch.float, device=self.device) if has_init_cond else None
                    s0 = torch.tensor(self.s0[self.train_idx[Nb*B:]], dtype=torch.float, device=self.device) if has_init_cond else None
                    t0 = torch.tensor(self.t0[self.train_idx[Nb*B:]], dtype=torch.float, device=self.device) if has_init_cond else None
                    u1 = torch.tensor(self.u1[self.train_idx[Nb*B:]], dtype=torch.float, device=self.device) if (has_init_cond and self.config['vel_continuity_loss']) else None
                    s1 = torch.tensor(self.s1[self.train_idx[Nb*B:]], dtype=torch.float, device=self.device) if (has_init_cond and self.config['vel_continuity_loss']) else None
                    t1 = torch.tensor(self.t1[self.train_idx[Nb*B:]], dtype=torch.float, device=self.device) if (has_init_cond and self.config['vel_continuity_loss']) else None
                    y_onehot = F.one_hot(torch.tensor(cell_labels[self.train_idx[Nb*B:]], dtype=int, device=self.device), self.n_type).float() if self.enable_cvae else None
                    p_t = self.p_t[:,self.train_idx[Nb*B:],:]
                    p_z = self.p_z[:,self.train_idx[Nb*B:],:]
                    lu_scale = self.lu_scale[self.train_idx[Nb*B:],:].exp()
                    ls_scale = self.ls_scale[self.train_idx[Nb*B:],:].exp()
                    mu_tx, std_tx, mu_zx, std_zx, uhat, shat, uhat_fw, shat_fw, vu, vs, vu_fw, vs_fw = self.eval_model(data_in, lu_scale, ls_scale, u0, s0, t0, t1, y_onehot)
                else:
                    u0 = torch.tensor(self.u0[Nb*B:], dtype=torch.float, device=self.device) if has_init_cond else None
                    s0 = torch.tensor(self.s0[Nb*B:], dtype=torch.float, device=self.device) if has_init_cond else None
                    t0 = torch.tensor(self.t0[Nb*B:], dtype=torch.float, device=self.device) if has_init_cond else None
                    u1 = torch.tensor(self.u1[Nb*B:], dtype=torch.float, device=self.device) if (has_init_cond and self.config['vel_continuity_loss']) else None
                    s1 = torch.tensor(self.s1[Nb*B:], dtype=torch.float, device=self.device) if (has_init_cond and self.config['vel_continuity_loss']) else None
                    t1 = torch.tensor(self.t1[Nb*B:], dtype=torch.float, device=self.device) if (has_init_cond and self.config['vel_continuity_loss']) else None
                    y_onehot = F.one_hot(torch.tensor(cell_labels[Nb*B:], dtype=int, device=self.device), self.n_type).float() if self.enable_cvae else None
                    p_t = self.p_t[:,Nb*B:,:]
                    p_z = self.p_z[:,Nb*B:,:]
                    lu_scale = self.lu_scale[Nb*B:,:].exp()
                    ls_scale = self.ls_scale[Nb*B:,:].exp()
                    mu_tx, std_tx, mu_zx, std_zx, uhat, shat, uhat_fw, shat_fw, vu, vs, vu_fw, vs_fw = self.eval_model(data_in, lu_scale, ls_scale, u0, s0, t0, t1, y_onehot)
                u_raw, s_raw = data_in[:,:G], data_in[:,G:]
                u_sum, s_sum = torch.sum(uhat, 1, keepdim=True), torch.sum(shat, 1, keepdim=True)
                logp, kldt, kldz, kld_param = self.vae_risk((mu_tx, std_tx), p_t,
                                                            (mu_zx, std_zx), p_z,
                                                            u_raw.int(), s_raw.int(), 
                                                            uhat*lu_scale, shat*ls_scale,
                                                            u1, s1, uhat_fw, shat_fw,
                                                            None)
                loss = (- logp + self.config["kl_t"] * kldt + self.config["kl_z"] * kldz).cpu().item()
                if(kld_param is not None):
                    loss = loss+self.config["kl_param"]*kld_param.cpu().item()
                #u_sample, s_sample = self.sample_poisson(uhat*lu_scale, shat*ls_scale)
                mse_batch = np.mean( (u_raw.cpu().numpy() - (uhat*lu_scale).cpu().numpy())**2 + ( s_raw.cpu().numpy() - (shat*ls_scale).cpu().numpy())**2)
                elbo = elbo - ((N-B*Nb)/N)*loss
                mse = mse + ((N-B*Nb)/N)*mse_batch
                logp_total = logp_total + ((N-B*Nb)/N) * logp.detach().item()
                kldt_total = kldt_total + ((N-B*Nb)/N) * kldt.detach().item()
                kldz_total = kldz_total + ((N-B*Nb)/N) * kldz.detach().item()
                if self.kldparam_list is not None:
                    kldparam_total = kldparam_total + ((N-B*Nb)/N) * kld_param.detach().item()

                if("uhat" in output and gene_idx is not None):
                    Uhat[Nb*B:] = uhat[:,gene_idx].cpu().numpy()
                    U_sum[Nb*B:] = u_sum.cpu().numpy().squeeze()
                if("shat" in output and gene_idx is not None):
                    Shat[Nb*B:] = shat[:,gene_idx].cpu().numpy()
                    S_sum[Nb*B:] = s_sum.cpu().numpy().squeeze()
                if("t" in output):
                    t_out[Nb*B:] = mu_tx.cpu().squeeze().numpy()
                    std_t_out[Nb*B:] = std_tx.cpu().squeeze().numpy()
                if("z" in output):
                    z_out[Nb*B:] = mu_zx.cpu().numpy()
                    std_z_out[Nb*B:] = std_zx.cpu().numpy()
        
        if(mode=="test"):
            self.logp_list.append(logp_total)
            self.kldt_list.append(kldt_total)
            self.kldz_list.append(kldz_total)
            if self.kldparam_list is not None:
                self.kldparam_list.append(kldparam_total)
         
        out = []
        if("uhat" in output):
            out.append(Uhat)
            out.append(U_sum)
        if("shat" in output):
            out.append(Shat)
            out.append(S_sum)
        if("t" in output):
            out.append(t_out)
            out.append(std_t_out)
        if("z" in output):
            out.append(z_out)
            out.append(std_z_out)
        
        if("mse" in output):
            out.append(mse)
        
        return out, elbo
    
    def pred_rho(self, z, G):
        N = z.shape[0]
        rho = np.zeros((N, G))
        stage = 2 if self.use_knn else 1
        with torch.no_grad():
            B = min(N//5, 1000)
            Nb = N // B
            for i in range(Nb):
                rho_batch = self.decoder.pred_rho(torch.tensor(z[i*B:(i+1)*B]).float().to(self.device),stage)
                rho[i*B:(i+1)*B] = rho_batch.cpu().numpy()
            if(N > Nb*B):
                rho_batch = self.decoder.pred_rho(torch.tensor(z[Nb*B:]).float().to(self.device),stage)
                rho[Nb*B:] = rho_batch.cpu().numpy()
        return rho
    
    def get_mean_param(self, gene_indices=None):
        alpha = np.exp(self.decoder.alpha.detach().cpu().numpy())
        beta = np.exp(self.decoder.beta.detach().cpu().numpy())
        gamma = np.exp(self.decoder.gamma.detach().cpu().numpy())
        scaling = np.exp(self.decoder.scaling.detach().cpu().numpy())
        if(gene_indices is None):
            return alpha, beta, gamma, scaling
        return alpha[gene_indices],beta[gene_indices],gamma[gene_indices],scaling[gene_indices]
    
    def get_gene_vel(self, z, G, uhat, shat, gene_indices):
        rho = self.pred_rho(z, G)
        rho = rho[:,gene_indices]
        alpha, beta, gamma, scaling = self.get_mean_param(gene_indices)
        return rho*alpha - beta*uhat/scaling, beta*uhat/scaling - gamma*shat
    
    def test(self,
             dataset, 
             Xembed,
             testid=0, 
             test_mode=True,
             gind=None, 
             gene_plot=None,
             plot=False, 
             path='figures',
             **kwargs):
        self.set_mode('eval')
        mode = "test" if test_mode else "train"
        with torch.no_grad():
            out, elbo = self.pred_all(dataset.data, self.cell_labels, mode, ["uhat", "shat", "t", "z", "mse"], gind)
            Uhat, U_sum, Shat, S_sum, t, std_t = out[0], out[1], out[2], out[3], out[4], out[5]
            
            
            if(test_mode):
                self.test_mse.append(out[-1])
            else:
               self.train_mse.append(out[-1])
            
            if(plot):
                #Plot Time
                t_ub = np.quantile(t, 0.99)
                plot_time(np.clip(t,None,t_ub), Xembed, save=f"{path}/time-{testid}-velovae.png")
                
                #Plot u/s-t and phase portrait for each gene
                G = dataset.G
                lu_scale = np.exp(self.lu_scale.detach().cpu().numpy().squeeze())
                ls_scale = np.exp(self.ls_scale.detach().cpu().numpy().squeeze())
                if(mode=="test"):
                    lu_scale = lu_scale[self.test_idx]
                    ls_scale = ls_scale[self.test_idx]
                else:
                    lu_scale = lu_scale[self.train_idx]
                    ls_scale = ls_scale[self.train_idx]
                
                for i in range(len(gind)):
                    idx = gind[i]
                    
                    plot_sig(t.squeeze(), 
                             dataset.data[:,idx], dataset.data[:,idx+G], 
                             Uhat[:,i], Shat[:,i], 
                             np.array([self.label_dic_rev[x] for x in dataset.labels]),
                             gene_plot[i], 
                             save=f"{path}/sig-{gene_plot[i]}-{testid}.png",
                             sparsify=self.config['sparsify'])
            
        return elbo
    
    def save_anndata(self, adata, key, file_path, file_name=None):
        self.set_mode('eval')
        os.makedirs(file_path, exist_ok=True)
        
        adata.var[f"{key}_alpha"] = np.exp(self.decoder.alpha.detach().cpu().numpy())
        adata.var[f"{key}_beta"] = np.exp(self.decoder.beta.detach().cpu().numpy())
        adata.var[f"{key}_gamma"] = np.exp(self.decoder.gamma.detach().cpu().numpy())
        adata.var[f"{key}_ton"] = self.decoder.ton.exp().detach().cpu().numpy()
        adata.var[f"{key}_scaling"] = self.decoder.scaling.exp().detach().cpu().numpy()

        if(self.config["use_raw"]):
            U,S = np.array(adata.layers['unspliced'].todense()), np.array(adata.layers['spliced'].todense())
        else:
            U,S = adata.layers["Cu"], adata.layers["Cs"]
        X = np.concatenate((U,S),1)

        out, elbo = self.pred_all(X, self.cell_labels, "both", gene_idx=np.array(range(adata.n_vars)))
        Uhat, U_sum, Shat, S_sum, t, std_t, z, std_z = out
        
        adata.obs[f"{key}_time"] = t
        adata.obs[f"{key}_std_t"] = std_t
        adata.obsm[f"{key}_z"] = z
        adata.obsm[f"{key}_std_z"] = std_z
        
        adata.layers[f"{key}_uhat"] = Uhat
        adata.layers[f"{key}_shat"] = Shat
        
        rho = np.zeros(U.shape)
        with torch.no_grad():
            B = min(U.shape[0]//10, 1000)
            Nb = U.shape[0] // B
            for i in range(Nb):
                rho_batch = torch.sigmoid(self.decoder.fc_out2(self.decoder.net_rho2(torch.tensor(z[i*B:(i+1)*B]).float().to(self.device))))
                rho[i*B:(i+1)*B] = rho_batch.cpu().numpy()
            rho_batch = torch.sigmoid(self.decoder.fc_out2(self.decoder.net_rho2(torch.tensor(z[Nb*B:]).float().to(self.device))))
            rho[Nb*B:] = rho_batch.cpu().numpy()
        
        adata.layers[f"{key}_rho"] = rho
        
        adata.obs[f"{key}_t0"] = self.t0.squeeze()
        adata.layers[f"{key}_u0"] = self.u0
        adata.layers[f"{key}_s0"] = self.s0
        if(self.config["vel_continuity_loss"]):
            adata.obs[f"{key}_t1"] = self.t1.squeeze()
            adata.layers[f"{key}_u1"] = self.u1
            adata.layers[f"{key}_s1"] = self.s1

        adata.uns[f"{key}_train_idx"] = self.train_idx
        adata.uns[f"{key}_test_idx"] = self.test_idx
        adata.uns[f"{key}_run_time"] = self.timer
        
        #scaling_u = adata.var[f"{key}_scaling_u"].to_numpy()
        #scaling_s = adata.var[f"{key}_scaling_s"].to_numpy()
        #adata.layers[f"{key}_velocity"] = adata.var[f"{key}_beta"].to_numpy() * Uhat / scaling_u - adata.var[f"{key}_gamma"].to_numpy() * Shat / scaling_s
        #adata.layers[f"{key}_velocity_u"] = rho * adata.var[f"{key}_alpha"].to_numpy() - adata.var[f"{key}_beta"].to_numpy() * Uhat / scaling_u
        rna_velocity_vae(adata, key, use_raw=False, use_scv_genes=False)
        
        if(file_name is not None):
            adata.write_h5ad(f"{file_path}/{file_name}")





























class decoder_fullvb(decoder):
    def __init__(self, 
                 adata, 
                 tmax,
                 train_idx,
                 dim_z, 
                 dim_cond=0,
                 N1=250, 
                 N2=500, 
                 p=98, 
                 use_raw=True,
                 init_ton_zero=False,
                 scale_cell=False,
                 separate_us_scale=True,
                 add_noise=True,
                 device=torch.device('cpu'), 
                 init_method="steady", 
                 init_key=None, 
                 init_type=None,
                 checkpoint=None):
        super(decoder_fullvb,self).__init__(adata, 
                                            tmax,
                                            train_idx,
                                            dim_z, 
                                            dim_cond,
                                            N1, 
                                            N2, 
                                            p, 
                                            use_raw,
                                            init_ton_zero,
                                            scale_cell,
                                            separate_us_scale,
                                            add_noise,
                                            device, 
                                            init_method, 
                                            init_key, 
                                            init_type,
                                            checkpoint)
        sigma_param = np.log(0.05) * torch.ones(adata.n_vars, device=device)
        if(checkpoint is None):
            self.alpha = nn.Parameter(torch.stack([self.alpha, sigma_param]).float())
            self.beta = nn.Parameter(torch.stack([self.beta, sigma_param]).float())
            self.gamma = nn.Parameter(torch.stack([self.gamma, sigma_param]).float())
    
    def reparameterize_param(self):
        eps = torch.normal(mean=torch.zeros((3,self.alpha.shape[1])),std=torch.ones((3,self.alpha.shape[1]))).to(self.alpha.device)
        alpha = torch.exp(self.alpha[0] + eps[0]*(self.alpha[1].exp()))
        beta = torch.exp(self.beta[0] + eps[1]*(self.beta[1].exp())) 
        gamma = torch.exp(self.gamma[0] + eps[2]*(self.gamma[1].exp()))
        return alpha, beta, gamma
    
    def forward(self, t, z, condition=None, u0=None, s0=None, t0=None, neg_slope=0.0):
        alpha, beta, gamma = self.reparameterize_param()
        if(u0 is None or s0 is None or t0 is None):
            if(condition is None):
                rho = torch.sigmoid(self.fc_out1(self.net_rho(z)))
            else:
                rho = torch.sigmoid(self.fc_out1(self.net_rho(torch.cat((z,condition),1))))
            Uhat, Shat = pred_su(F.leaky_relu(t - self.ton.exp(), neg_slope), self.u0.exp()/self.scaling.exp(),  self.s0.exp(), rho*alpha, beta, gamma)
        else:
            if(condition is None):
                rho = torch.sigmoid(self.fc_out2(self.net_rho2(z)))
            else:
                rho = torch.sigmoid(self.fc_out2(self.net_rho2(torch.cat((z,condition),1))))
            Uhat, Shat = pred_su(F.leaky_relu(t-t0, neg_slope), u0, s0, rho*alpha, beta, gamma)
        Uhat = F.relu(Uhat)*self.scaling.exp()
        Shat = F.relu(Shat)
        vu = alpha - beta * Uhat / torch.exp(self.scaling)
        vs = beta * Uhat / torch.exp(self.scaling) - gamma * Shat
        return Uhat, Shat, vu, vs
        
    def eval_model(self, t, z, condition=None, u0=None, s0=None, t0=None, neg_slope=0.0):
        #Evaluate the decoder. Here, we use the mean instead of randomly sample the ODE parameters.
        
        alpha = self.alpha[0].exp()
        beta = self.beta[0].exp()
        gamma = self.gamma[0].exp()
        if(u0 is None or s0 is None or t0 is None):
            if(condition is None):
                rho = torch.sigmoid(self.fc_out1(self.net_rho(z)))
            else:
                rho = torch.sigmoid(self.fc_out1(self.net_rho(torch.cat((z,condition),1))))
            Uhat, Shat = pred_su(F.leaky_relu(t - self.ton.exp(), neg_slope), 0, 0, rho*alpha, beta, gamma)
        else:
            if(condition is None):
                rho = torch.sigmoid(self.fc_out2(self.net_rho2(z)))
            else:
                rho = torch.sigmoid(self.fc_out2(self.net_rho2(torch.cat((z,condition),1))))
            Uhat, Shat = pred_su(F.leaky_relu(t-t0, neg_slope), u0, s0, rho*alpha, beta, gamma)
        Uhat = F.relu(Uhat)*self.scaling.exp()
        Shat = F.relu(Shat)
        vu = alpha - beta * Uhat / torch.exp(self.scaling)
        vs = beta * Uhat / torch.exp(self.scaling) - gamma * Shat
        return Uhat, Shat, vu, vs


class DVAEFullVB(DVAE):
    def __init__(self, 
                 adata, 
                 tmax, 
                 dim_z, 
                 dim_cond=0,
                 device='cpu', 
                 hidden_size=(1000, 500, 500, 1000), 
                 use_raw=True,
                 init_method="steady", 
                 init_key=None,
                 tprior=None, 
                 init_type=None,
                 init_ton_zero=True,
                 add_noise=True,
                 time_distribution="gaussian",
                 count_distribution="Poisson",
                 std_z_prior=0.01,
                 rate_prior={
                     'alpha':(2.0,1.0),
                     'beta':(1.0,0.5),
                     'gamma':(1.0,0.5)
                 },
                 scale_cell=False,
                 separate_us_scale=True,
                 checkpoints=[None, None]):
        
        t_start = time.time()
        self.config = {
            #Model Parameters
            "dim_z":dim_z,
            "hidden_size":hidden_size,
            "tmax":tmax,
            "init_method":init_method,
            "init_key":init_key,
            "tprior":tprior,
            "std_z_prior":std_z_prior,
            "tail":0.01,
            "time_overlap":0.5,
            "n_neighbors":10,
            "dt": (0.03,0.06),
            "knn_adaptive": 0,
            

            #Training Parameters
            "n_epochs":2000, 
            "n_epochs_post":2000,
            "batch_size":128, 
            "learning_rate":None, 
            "learning_rate_ode":None, 
            "learning_rate_post":None,
            "lambda":1e-3, 
            "lambda_rho":0.1,
            "kl_t":1.0, 
            "kl_z":1.0, 
            "kl_param":1.0,
            "reg_v":0.0,
            "test_iter":None, 
            "save_epoch":100, 
            "x0_epoch":25,
            "n_warmup":5,
            "early_stop":5,
            "early_stop_thred":adata.n_vars*1e-3,
            "train_test_split":0.7,
            "neg_slope":0.0,
            "k_alt":1, 
            "train_scaling":False, 
            "train_ton":False,
            "train_std":False,
            "weight_sample":False,
            "vel_continuity_loss":False,
            "use_raw":use_raw,
            
            #Normalization Configurations
            "scale_cell":scale_cell,
            "separate_us_scale":separate_us_scale,
            "scale_gene_encoder":True,
            "scale_cell_encoder":False,
            "log1p":True,
            
            
            #Plotting
            "sparsify":1
        }
        
        self.set_device(device)
        self.split_train_test(adata.n_obs)
        
        N, G = adata.n_obs, adata.n_vars
        self.dim_z = dim_z
        self.enable_cvae = dim_cond>0
        try:
            self.encoder = encoder_velovae(2*G, dim_z, dim_cond, hidden_size[0], hidden_size[1], self.device, checkpoint=checkpoints[0]).float()
        except IndexError:
            print('Please provide two dimensions!')
        
        self.decoder = decoder_fullvb(adata, 
                                      tmax, 
                                      self.train_idx,
                                      dim_z, 
                                      N1=hidden_size[2], 
                                      N2=hidden_size[3], 
                                      use_raw=use_raw,
                                      init_ton_zero=init_ton_zero,
                                      scale_cell = self.config["scale_cell"],
                                      separate_us_scale=self.config["separate_us_scale"],
                                      device=self.device, 
                                      init_method = init_method,
                                      init_key = init_key,
                                      init_type = init_type,
                                      checkpoint=checkpoints[1]).float()
        
        self.tmax=tmax
        self.time_distribution = time_distribution
        if(init_type is not None):
            tprior = 'tprior'
            self.config['tprior'] = tprior
            self.config['train_ton'] = False
        self.get_prior(adata, time_distribution, tmax, tprior)
        
        self.p_z = torch.stack([torch.zeros(N,dim_z), torch.ones(N,dim_z)*self.config["std_z_prior"]]).float().to(self.device)
        
        self.use_knn = False
        self.u0 = None
        self.s0 = None
        self.t0 = None
        self.neighbor_idx, self.neighbor_idx_fw = None, None
        self.lu_scale = torch.tensor(np.log(adata.obs['library_scale_u'].to_numpy())).unsqueeze(-1).float().to(self.device)
        self.ls_scale = torch.tensor(np.log(adata.obs['library_scale_s'].to_numpy())).unsqueeze(-1).float().to(self.device)
        
        print(f"Library scale (U): Max={np.exp(self.lu_scale.detach().cpu().numpy().max()):.2f}, Min={np.exp(self.lu_scale.detach().cpu().numpy().min()):.2f}, Mean={np.exp(self.lu_scale.detach().cpu().numpy().mean()):.2f}")
        print(f"Library scale (S): Max={np.exp(self.ls_scale.detach().cpu().numpy().max()):.2f}, Min={np.exp(self.ls_scale.detach().cpu().numpy().min()):.2f}, Mean={np.exp(self.ls_scale.detach().cpu().numpy().mean()):.2f}")
        
        #Determine Count Distribution
        dispersion_u = adata.var["dispersion_u"].to_numpy()
        dispersion_s = adata.var["dispersion_s"].to_numpy()
        if(count_distribution=="auto"):
            p_nb = np.sum((dispersion_u>1) & (dispersion_s>1))/adata.n_vars
            if(p_nb > 0.5):
                count_distribution = "NB"
                self.vae_risk = self.vae_risk_nb
            else:
                count_distribution = "Poisson"
                self.vae_risk = self.vae_risk_poisson
            print(f"Mean dispersion: u={dispersion_u.mean():.2f}, s={dispersion_s.mean():.2f}")
            print(f"Over-Dispersion = {p_nb:.2f} => Using {count_distribution} to model count data.")
        elif(count_distribution=="NB"):
            self.vae_risk = self.vae_risk_nb
        else:
            self.vae_risk = self.vae_risk_poisson
        
        dispersion_u[dispersion_u<1] = 1.001
        dispersion_s[dispersion_s<1] = 1.001
        mean_u = adata.var["mean_u"].to_numpy()
        mean_s = adata.var["mean_s"].to_numpy()
        self.eta_u = torch.tensor(np.log(dispersion_u-1)-np.log(mean_u)).float().to(self.device)
        self.eta_s = torch.tensor(np.log(dispersion_s-1)-np.log(mean_s)).float().to(self.device)
        
        
        #Class attributes for training
        self.loss_train, self.loss_test = [],[]
        self.logp_list, self.kldt_list, self.kldz_list, self.kldparam_list = [],[],[],[]
        self.counter = 0 #Count the number of iterations
        self.n_drop = 0 #Count the number of consecutive iterations with little decrease in loss
        
        self.p_log_alpha = torch.tensor([[rate_prior['alpha'][0]], [rate_prior['alpha'][1]]]).to(self.device)
        self.p_log_beta = torch.tensor([[rate_prior['beta'][0]], [rate_prior['beta'][1]]]).to(self.device)
        self.p_log_gamma = torch.tensor([[rate_prior['gamma'][0]], [rate_prior['gamma'][1]]]).to(self.device)
        
        self.timer = time.time() - t_start
    
    def eval_model(self, data_in, lu_scale, ls_scale, u0=None, s0=None, t0=None, t1=None, condition=None, continuous=True):
        data_in_scale = data_in
        if(self.config["scale_gene_encoder"]):
            data_in_scale = torch.cat((data_in_scale[:,:data_in_scale.shape[1]//2]/self.decoder.scaling.exp(), data_in_scale[:,data_in_scale.shape[1]//2:]),1)
        if(self.config["scale_cell_encoder"]):
            data_in_scale = torch.cat((data_in_scale[:,:data_in_scale.shape[1]//2]/lu_scale, data_in_scale[:,data_in_scale.shape[1]//2:]/ls_scale),1)
        if(self.config["log1p"]):
            data_in_scale = torch.log1p(data_in_scale)
        mu_t, std_t, mu_z, std_z = self.encoder.forward(data_in_scale, condition)
        uhat, shat, vu, vs = self.decoder.eval_model(mu_t, mu_z, condition, u0=u0, s0=s0, t0=t0, neg_slope=0.0)

        if(t1 is not None):
            uhat_fw, shat_fw, vu_fw, vs_fw = self.decoder.forward(t1, mu_z, condition, uhat, shat, mu_t, neg_slope=self.config["neg_slope"])
        else:
            uhat_fw, shat_fw, vu_fw, vs_fw = None, None, None, None
        if(not continuous):
            poisson_u = Poisson(F.softplus(uhat, beta=100))
            poisson_s = Poisson(F.softplus(shat, beta=100))
            if(t1 is not None):
                poisson_u = Poisson(F.softplus(uhat_fw, beta=100))
                poisson_s = Poisson(F.softplus(shat_fw, beta=100))
                u_out_fw = poisson_u_fw.sample()
                s_out_fw = poisson_s_fw.sample()
            else:
                u_out_fw, s_out_fw = None, None
            u_out = poisson_u.sample()
            s_out = poisson_s.sample()
            
            return mu_t, std_t, mu_z, std_z, u_out, s_out, u_out_fw, s_out_fw, vu, vs, vu_fw, vs_fw
        return mu_t, std_t, mu_z, std_z, uhat, shat, uhat_fw, shat_fw, vu, vs, vu_fw, vs_fw
    
    def vae_risk_poisson(self, 
                         q_tx, p_t, 
                         q_zx, p_z, 
                         u, s, uhat, shat,
                         u1=None, s1=None, uhat_fw=None, shat_fw=None,
                         weight=None,
                         eps=1e-2):
        kldt = self.kl_time(q_tx[0], q_tx[1], p_t[0], p_t[1], tail=self.config["tail"])
        kldz = kl_gaussian(q_zx[0], q_zx[1], p_z[0], p_z[1])
        kld_param = (kl_gaussian(self.decoder.alpha[0].view(1,-1), self.decoder.alpha[1].exp().view(1,-1), self.p_log_alpha[0], self.p_log_alpha[1]) + \
                     kl_gaussian(self.decoder.beta[0].view(1,-1), self.decoder.beta[1].exp().view(1,-1), self.p_log_beta[0], self.p_log_beta[1]) + \
                     kl_gaussian(self.decoder.gamma[0].view(1,-1), self.decoder.gamma[1].exp().view(1,-1), self.p_log_gamma[0], self.p_log_gamma[1]) ) / u.shape[0]
        #poisson
        mask_u = (torch.isnan(uhat) | torch.isinf(uhat)).float()
        uhat = uhat * (1-mask_u)
        mask_s = (torch.isnan(shat) | torch.isinf(shat)).float()
        shat = shat * (1-mask_s)
        
        poisson_u = Poisson(F.relu(uhat)+eps)
        poisson_s = Poisson(F.relu(shat)+eps)
        
        
        logp = poisson_u.log_prob(u) + poisson_s.log_prob(s)
        #velocity continuity loss
        if(uhat_fw is not None and shat_fw is not None):
            logp = logp - self.kl_poisson(u1, uhat_fw) - self.kl_poisson(s1, shat_fw)
        
        if( weight is not None):
            logp = logp*weight
        err_rec = torch.mean(torch.sum(logp,1))
        
        return err_rec, kldt, kldz, kld_param
    
    def vae_risk_nb(self, 
                    q_tx, p_t, 
                    q_zx, p_z, 
                    u, s, uhat, shat, 
                    weight=None,
                    eps=1e-10):
        #This is the negative ELBO.
        kldt = self.kl_time(q_tx[0], q_tx[1], p_t[0], p_t[1], tail=self.config["tail"])
        kldz = kl_gaussian(q_zx[0], q_zx[1], p_z[0], p_z[1])
        kld_param = (kl_gaussian(self.decoder.alpha[0].view(1,-1), self.decoder.alpha[1].exp().view(1,-1), self.p_log_alpha[0], self.p_log_alpha[1]) + \
                     kl_gaussian(self.decoder.beta[0].view(1,-1), self.decoder.beta[1].exp().view(1,-1), self.p_log_beta[0], self.p_log_beta[1]) + \
                     kl_gaussian(self.decoder.gamma[0].view(1,-1), self.decoder.gamma[1].exp().view(1,-1), self.p_log_gamma[0], self.p_log_gamma[1])) / u.shape[0]
        
        #NB
        p_nb_u = torch.sigmoid(self.eta_u+torch.log(F.relu(uhat)+eps))
        p_nb_s = torch.sigmoid(self.eta_s+torch.log(F.relu(shat)+eps))
        nb_u = NegativeBinomial((F.relu(uhat)+eps)*(1-p_nb_u)/p_nb_u, probs=p_nb_u)
        nb_s = NegativeBinomial((F.relu(shat)+eps)*(1-p_nb_s)/p_nb_s, probs=p_nb_s)
        logp = nb_u.log_prob(u) + nb_s.log_prob(s)
        
        if( weight is not None):
            logp = logp*weight
        err_rec = torch.mean(torch.sum(logp,1))
        
        return err_rec, kldt, kldz, kld_param 
    
    def get_mean_param(self, gene_indices=None):
        mu_logalpha = self.decoder.alpha[0].detach().cpu().numpy()
        std_logalpha = np.exp(self.decoder.alpha[1].detach().cpu().numpy())
        alpha = np.exp(mu_logalpha)
        
        mu_logbeta = self.decoder.beta[0].detach().cpu().numpy()
        std_logbeta = np.exp(self.decoder.beta[1].detach().cpu().numpy())
        beta = np.exp(mu_logbeta)
        
        mu_loggamma = self.decoder.gamma[0].detach().cpu().numpy()
        std_loggamma = np.exp(self.decoder.gamma[1].detach().cpu().numpy())
        gamma = np.exp(mu_loggamma)
        
        scaling = np.exp(self.decoder.scaling.detach().cpu().numpy())
        if(gene_indices is None):
            return alpha, beta, gamma, scaling
        return alpha[gene_indices],beta[gene_indices],gamma[gene_indices],scaling[gene_indices]
    
    def save_anndata(self, adata, key, file_path, file_name=None):
        self.set_mode('eval')
        os.makedirs(file_path, exist_ok=True)
        
        adata.var[f"{key}_logmu_alpha"] = self.decoder.alpha[0].detach().cpu().numpy()
        adata.var[f"{key}_logmu_beta"] = self.decoder.beta[0].detach().cpu().numpy()
        adata.var[f"{key}_logmu_gamma"] = self.decoder.gamma[0].detach().cpu().numpy()
        adata.var[f"{key}_logstd_alpha"] = self.decoder.alpha[1].detach().cpu().exp().numpy()
        adata.var[f"{key}_logstd_beta"] = self.decoder.beta[1].detach().cpu().exp().numpy()
        adata.var[f"{key}_logstd_gamma"] = self.decoder.gamma[1].detach().cpu().exp().numpy()
        adata.var[f"{key}_ton"] = self.decoder.ton.exp().detach().cpu().numpy()
        adata.var[f"{key}_scaling"] = self.decoder.scaling.exp().detach().cpu().numpy()
        
        if(self.config["use_raw"]):
            U,S = np.array(adata.layers['unspliced'].todense()), np.array(adata.layers['spliced'].todense())
        else:
            U,S = adata.layers["Cu"], adata.layers["Cs"]
        X = np.concatenate((U,S),1)

        out, elbo = self.pred_all(X, self.cell_labels, "both", gene_idx=np.array(range(adata.n_vars)))
        Uhat, U_sum, Shat, S_sum, t, std_t, z, std_z = out
        
        adata.obs[f"{key}_time"] = t
        adata.obs[f"{key}_std_t"] = std_t
        adata.obsm[f"{key}_z"] = z
        adata.obsm[f"{key}_std_z"] = std_z
        
        adata.layers[f"{key}_uhat"] = Uhat
        adata.layers[f"{key}_shat"] = Shat
        
        rho = np.zeros(U.shape)
        with torch.no_grad():
            B = min(U.shape[0]//10, 1000)
            Nb = U.shape[0] // B
            for i in range(Nb):
                rho_batch = torch.sigmoid(self.decoder.fc_out2(self.decoder.net_rho2(torch.tensor(z[i*B:(i+1)*B]).float().to(self.device))))
                rho[i*B:(i+1)*B] = rho_batch.cpu().numpy()
            rho_batch = torch.sigmoid(self.decoder.fc_out2(self.decoder.net_rho2(torch.tensor(z[Nb*B:]).float().to(self.device))))
            rho[Nb*B:] = rho_batch.cpu().numpy()
        
        adata.layers[f"{key}_rho"] = rho
        
        adata.obs[f"{key}_t0"] = self.t0.squeeze()
        adata.layers[f"{key}_u0"] = self.u0
        adata.layers[f"{key}_s0"] = self.s0
        if(self.config["vel_continuity_loss"]):
            adata.obs[f"{key}_t1"] = self.t1.squeeze()
            adata.layers[f"{key}_u1"] = self.u1
            adata.layers[f"{key}_s1"] = self.s1

        adata.uns[f"{key}_train_idx"] = self.train_idx
        adata.uns[f"{key}_test_idx"] = self.test_idx
        adata.uns[f"{key}_run_time"] = self.timer
        
        rna_velocity_vae(adata, key, use_raw=False, use_scv_genes=False, full_vb=True)
        
        if(file_name is not None):
            adata.write_h5ad(f"{file_path}/{file_name}")
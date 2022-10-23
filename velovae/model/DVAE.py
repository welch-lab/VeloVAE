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
from velovae.plotting import plot_sig, plot_sig_, plot_time, plot_train_loss, plot_test_loss, plot_vel
from torchdiffeq import odeint_adjoint

from .VAE import VAE
from .VAE import encoder as encoder_velovae
from .VAE import decoder as decoder_velovae
from .model_util import hist_equal, scale_by_gene, scale_by_cell, get_cell_scale, get_gene_scale, init_params, init_params_raw, get_ts_global, reinit_params, convert_time, get_gene_index, optimal_transport_duality_gap
from .model_util import pred_su, ode, knnx0
from .TransitionGraph import encode_type
from .TrainingData import SCData
from .VanillaVAE import VanillaVAE, kl_gaussian, kl_uniform
from .VanillaVAE import encoder as encoder_vanilla
from .VanillaVAE import decoder as decoder_vanilla
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

    def init_weights(self):
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
        #with torch.no_grad():
        #    zero_prop = torch.sum(h==0,1)/h.shape[1]
        #    print(zero_prop.mean(), zero_prop.std())
        if(condition is not None):
            h = torch.cat((h,condition),1)
        mu_tx, std_tx = self.spt1(self.fc_mu_t(h)), self.spt2(self.fc_std_t(h))
        mu_zx, std_zx = self.fc_mu_z(h), self.spt3(self.fc_std_z(h))
        return mu_tx, std_tx, mu_zx, std_zx

class decoderBasic(nn.Module):
    def __init__(self, 
                 adata, 
                 tmax,
                 train_idx,
                 p=98, 
                 scale_gene=True,
                 scale_cell=False,
                 separate_us_scale=True,
                 device=torch.device('cpu'), 
                 init_method="steady", 
                 init_key=None, 
                 checkpoint=None):
        super(decoderBasic,self).__init__()
        G = adata.n_vars
        if(checkpoint is not None):
            self.alpha = nn.Parameter(torch.empty(G, device=device).float())
            self.beta = nn.Parameter(torch.empty(G, device=device).float())
            self.gamma = nn.Parameter(torch.empty(G, device=device).float())
            self.scaling_u = nn.Parameter(torch.empty(G, device=device).float())
            self.scaling_s = nn.Parameter(torch.empty(G, device=device).float())
            self.ton = nn.Parameter(torch.empty(G, device=device).float())
            
            self.load_state_dict(torch.load(checkpoint, map_location=device))
        else:
            U, S = np.array(adata.layers['unspliced'].todense()), np.array(adata.layers['spliced'].todense())
            
            #Library Size
            
            if(scale_cell):
                U, S, lu, ls = scale_by_cell(U, S, train_idx, separate_us_scale)
                adata.obs["library_scale_u"] = lu
                adata.obs["library_scale_s"] = ls
            else:
                lu, ls = np.ones((adata.n_obs)),np.ones((adata.n_obs))
                adata.obs["library_scale_u"] = lu
                adata.obs["library_scale_s"] = ls
            #Genewise scaling
            if(scale_gene):
                U, S, scaling_u, scaling_s = scale_by_gene(U,S,train_idx)
            else:
                scaling_u, scaling_s = np.ones((G)), np.ones((G))
            
            U = U[train_idx]
            S = S[train_idx]
            noise = np.abs(np.random.normal(size=(len(train_idx), 2*G))*1e-3)
            X = np.concatenate((U,S),1) + noise

            #Dynamical Model Parameters
            if(init_method == "random"):
                print("Random Initialization.")
                alpha, beta, gamma, scaling, toff, u0, s0, sigma_u, sigma_s, T, Rscore = init_params_(X,p,fit_scaling=True, scaling_u=scaling_u)
                
                self.alpha = nn.Parameter(torch.normal(0.0, 0.01, size=(G,), device=device).float())
                self.beta =  nn.Parameter(torch.normal(0.0, 0.01, size=(G,), device=device).float())
                self.gamma = nn.Parameter(torch.normal(0.0, 0.01, size=(G,), device=device).float())
                self.ton = nn.Parameter(torch.normal(0.0, 0.01, size=(G,), device=device).float())
                self.toff = nn.Parameter(torch.normal(0.0, 0.01, size=(G,), device=device).float()+self.ton.detach())
                #self.scaling = nn.Parameter(torch.tensor(np.log(scaling), device=device).float())
                self.sigma_u = nn.Parameter(torch.tensor(np.log(sigma_u), device=device).float())
                self.sigma_s = nn.Parameter(torch.tensor(np.log(sigma_s), device=device).float())
            elif(init_method == "tprior"):
                print("Initialization using prior time.")
                alpha, beta, gamma, scaling, toff, u0, s0, sigma_u, sigma_s, T, Rscore = init_params_(X,p,fit_scaling=True, scaling_u=scaling_u)
                t_prior = adata.obs[init_key].to_numpy()
                t_prior = t_prior[train_idx]
                std_t = (np.std(t_prior)+1e-3)*0.2
                self.t_init = np.random.uniform(t_prior-std_t, t_prior+std_t)
                self.t_init -= self.t_init.min()
                self.t_init = self.t_init
                self.t_init = self.t_init/self.t_init.max()*tmax
                toff = get_ts_global(self.t_init, U/scaling, S, 95)
                alpha, beta, gamma, ton = reinit_params(U/scaling, S, self.t_init, toff)
                
                self.alpha = nn.Parameter(torch.tensor(np.log(alpha), device=device).float())
                self.beta = nn.Parameter(torch.tensor(np.log(beta), device=device).float())
                self.gamma = nn.Parameter(torch.tensor(np.log(gamma), device=device).float())
                #self.scaling = nn.Parameter(torch.tensor(np.log(scaling), device=device).float())
                self.ton = nn.Parameter(torch.tensor(np.log(ton+1e-10), device=device).float())
                self.toff = nn.Parameter(torch.tensor(np.log(toff+1e-10), device=device).float())
                self.sigma_u = nn.Parameter(torch.tensor(np.log(sigma_u), device=device).float())
                self.sigma_s = nn.Parameter(torch.tensor(np.log(sigma_s), device=device).float())
            else:
                print("Initialization using the steady-state and dynamical models.")
                alpha, beta, gamma, scaling, toff, u0, s0, sigma_u, sigma_s, T, Rscore = init_params_(X,p,fit_scaling=True, scaling_u=scaling_u)
                if(init_key is not None):
                    t_init = adata.obs['init_key'].to_numpy()
                else:
                    T = T+np.random.rand(T.shape[0],T.shape[1]) * 1e-3
                    T_eq = np.zeros(T.shape)
                    Nbin = T.shape[0]//50+1
                    for i in range(T.shape[1]):
                        T_eq[:, i] = hist_equal(T[:, i], tmax, 0.9, Nbin)
                    self.t_init = np.quantile(T_eq,0.5,1)
                toff = get_ts_global(self.t_init, U/scaling, S, 95)
                alpha, beta, gamma,ton = reinit_params(U/scaling, S, self.t_init, toff)
            
                self.alpha = nn.Parameter(torch.tensor(np.log(alpha), device=device).float())
                self.beta = nn.Parameter(torch.tensor(np.log(beta), device=device).float())
                self.gamma = nn.Parameter(torch.tensor(np.log(gamma), device=device).float())
                #self.scaling = nn.Parameter(torch.tensor(np.log(scaling), device=device).float())
                self.ton = nn.Parameter(torch.tensor(np.log(ton+1e-10), device=device).float())
                self.toff = nn.Parameter(torch.tensor(np.log(toff+1e-10), device=device).float())
                self.sigma_u = nn.Parameter(torch.tensor(np.log(sigma_u), device=device).float())
                self.sigma_s = nn.Parameter(torch.tensor(np.log(sigma_s), device=device).float())

        #self.scaling.requires_grad = False
        self.sigma_u.requires_grad = False
        self.sigma_s.requires_grad = False
        plot_time(self.t_init, adata.obsm["X_umap"][train_idx],save="figures/tinit.png")
        self.scaling_u = nn.Parameter(torch.tensor(np.log(scaling_u), device=device).float())
        self.scaling_s = nn.Parameter(torch.tensor(np.log(scaling_s), device=device).float())
        self.scaling_u.requires_grad = False
        self.scaling_s.requires_grad = False
            
    
    def forward(self, t, neg_slope=0.0):
        Uhat, Shat = ode(t, torch.exp(self.alpha), torch.exp(self.beta), torch.exp(self.gamma), torch.exp(self.ton), torch.exp(self.toff), neg_slope=neg_slope)
        return nn.functional.relu(Uhat), nn.functional.relu(Shat)

class VanillaDVAE(VanillaVAE):
    """
    Discrete VeloVAE Model
    """
    def __init__(self, 
                 adata, 
                 tmax, 
                 device='cpu', 
                 hidden_size=(500, 250), 
                 init_method="steady",
                 init_key=None,
                 tprior=None, 
                 time_distribution="gaussian",
                 scale_gene=True,
                 scale_cell=False,
                 separate_us_scale=True,
                 checkpoints=None):
        """Discrete VeloVAE with latent time only
        
        Arguments 
        ---------
        adata : :class:`anndata.AnnData`
        tmax : float
            Time Range 
        device : {'cpu','gpu'}, optional
        hidden_size : tuple, optional
            Width of the first and second hidden layer
        init_type : str, optional
            The stem cell type. Used to estimated the initial conditions.
            This is not commonly used in practice and please consider leaving it to default.
        init_key : str, optional
            column in the AnnData object containing the capture time
        tprior : str, optional
            key in adata.obs that stores the capture time.
            Used for informative time prior
        time_distribution : str, optional
            Should be either "gaussian" or "uniform.
        checkpoints : string list
            Contains the path to saved encoder and decoder models. Should be a .pt file.
        """
        #Extract Input Data
        try:
            U,S = adata.layers['Mu'], adata.layers['Ms']
        except KeyError:
            print('Unspliced/Spliced count matrices not found in the layers! Exit the program...')
        
        #Default Training Configuration
        self.config = {
            #Model Parameters
            "tmax":tmax,
            "hidden_size":hidden_size,
            "init_method": init_method,
            "init_key": init_key,
            "tprior":tprior,
            "tail":0.01,
            "time_overlap":0.5,

            #Training Parameters
            "n_epochs":2000, 
            "batch_size":128,
            "learning_rate":2e-4, 
            "learning_rate_ode":5e-4, 
            "lambda":1e-3, 
            "kl_t":1.0, 
            "test_iter":None, 
            "save_epoch":100,
            "n_warmup":5,
            "early_stop":5,
            "early_stop_thred":1e-3*adata.n_vars,
            "train_test_split":0.7,
            "k_alt":1,
            "neg_slope":0.0,
            "train_scaling":False, 
            "train_std":False, 
            "weight_sample":False,
            "scale_gene":scale_gene,
            "scale_cell":scale_cell,
            "separate_us_scale":separate_us_scale,
            "log1p":True,
            
            #Plotting
            "sparsify":1
        }
        
        self.set_device(device)
        self.split_train_test(adata.n_obs)
        
        N, G = adata.n_obs, adata.n_vars
        try:
            self.encoder = encoder_vanilla(2*G, hidden_size[0], hidden_size[1], self.device, checkpoint=checkpoints).float()
        except IndexError:
            print('Please provide two dimensions!')
        
        self.decoder = decoderBasic(adata, 
                                   tmax, 
                                   self.train_idx,
                                   scale_gene = scale_gene,
                                   scale_cell = scale_cell,
                                   separate_us_scale = separate_us_scale,
                                   device=self.device, 
                                   init_method = init_method,
                                   init_key = init_key,
                                   checkpoint=checkpoints).float()
        
        self.tmax=tmax
        self.time_distribution = time_distribution
        self.get_prior(adata, time_distribution, tmax, tprior)
        
        self.lu_scale = torch.tensor(adata.obs['library_scale_u'].to_numpy()).unsqueeze(-1).float().to(self.device)
        self.ls_scale = torch.tensor(adata.obs['library_scale_s'].to_numpy()).unsqueeze(-1).float().to(self.device)
        

        print(f"Mean library scale: {self.lu_scale.detach().cpu().numpy().mean()}, {self.ls_scale.detach().cpu().numpy().mean()}")

        #Class attributes for training
        self.loss_train, self.loss_test = [],[]
        self.counter = 0 #Count the number of iterations
        self.n_drop = 0 #Count the number of consecutive iterations with little decrease in loss
    
    def set_mode(self,mode):
        #Set the model to either training or evaluation mode.
        
        if(mode=='train'):
            self.encoder.train()
            self.decoder.train()
        elif(mode=='eval'):
            self.encoder.eval()
            self.decoder.eval()
        else:
            print("Warning: mode not recognized. Must be 'train' or 'test'! ")
    
    def forward(self, data_in, lu_scale, ls_scale):
        scaling_u = self.decoder.scaling_u.exp() * lu_scale
        scaling_s = self.decoder.scaling_s.exp() * ls_scale
        data_in_scale = torch.cat((data_in[:,:data_in.shape[1]//2]/scaling_u, data_in[:,data_in.shape[1]//2:]/scaling_s),1)
        if(self.config["log1p"]):
            data_in_log = torch.log1p(data_in_scale)
        else:
            data_in_log = data_in_scale
        mu_t, std_t = self.encoder.forward(data_in_log)
        t = self.reparameterize(mu_t, std_t)
         
        uhat, shat = self.decoder.forward(t, neg_slope=self.config["neg_slope"]) #uhat is scaled
        uhat = uhat*scaling_u
        shat = shat*scaling_s
        
        return mu_t, std_t, t, uhat, shat
    
    def eval_model(self, data_in, lu_scale, ls_scale, continuous=True):
        scaling_u = self.decoder.scaling_u.exp() * lu_scale
        scaling_s = self.decoder.scaling_s.exp() * ls_scale
        data_in_scale = torch.cat((data_in[:,:data_in.shape[1]//2]/scaling_u, data_in[:,data_in.shape[1]//2:]/scaling_s),1)
        if(self.config["log1p"]):
            data_in_log = torch.log1p(data_in_scale)
        else:
            data_in_log = data_in_scale
        mu_t, std_t = self.encoder.forward(data_in_log)
        
        uhat, shat = self.decoder.forward(mu_t, neg_slope=self.config["neg_slope"]) #uhat is scaled
        uhat = uhat*scaling_u
        shat = shat*scaling_s
        
        if(not continuous):
            poisson_u = Poisson(F.softplus(uhat, beta=100))
            poisson_s = Poisson(F.softplus(shat, beta=100))
            u_out = poisson_u.sample()
            s_out = poisson_s.sample()
            return mu_t, std_t, mu_z, std_z, u_out, s_out
        return mu_t, std_t, uhat, shat
    
    
    def sample_poisson(self, uhat, shat):
        u_sample = torch.poisson(uhat)
        s_sample = torch.poisson(shat)
        return u_sample.cpu(), s_sample.cpu()
    
    def sample_nb(self, uhat, shat, pu, ps):
        u_nb = NegativeBinomial(uhat*(1-pu)/pu, pu.repeat(uhat.shape[0],1))
        s_nb = NegativeBinomial(shat*(1-ps)/ps, ps.repeat(shat.shape[0],1))
        return u_nb.sample(), s_nb.sample()
    
    def vae_risk_poisson(self, 
                         q_tx, p_t, 
                         u, s, uhat, shat, 
                         weight=None,
                         eps=1e-6):
        kldt = self.kl_time(q_tx[0], q_tx[1], p_t[0], p_t[1], tail=self.config["tail"])

        #poisson
        try:
            poisson_u = Poisson(F.relu(uhat)+1e-2)
            poisson_s = Poisson(F.relu(shat)+1e-2)
        except ValueError:
            uhat[torch.isnan(uhat)] = 0
            shat[torch.isnan(shat)] = 0
            poisson_u = Poisson(F.relu(uhat)+1e-2)
            poisson_s = Poisson(F.relu(shat)+1e-2)
        
        logp = poisson_u.log_prob(u) + poisson_s.log_prob(s)
        
        if( weight is not None):
            logp = logp*weight
        
        err_rec = torch.mean(torch.sum(logp,1))
        
        return (- err_rec + self.config["kl_t"]*kldt)
    
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
            
            xbatch, idx = batch[0].to(self.device), batch[3]
            u = xbatch[:,:xbatch.shape[1]//2]
            s = xbatch[:,xbatch.shape[1]//2:]
            
            lu_scale = self.lu_scale[self.train_idx[idx]]
            ls_scale = self.ls_scale[self.train_idx[idx]]
            mu_tx, std_tx, t, uhat, shat = self.forward(xbatch.float(), lu_scale, ls_scale)
            
            loss = self.vae_risk_poisson((mu_tx, std_tx), self.p_t[:,self.train_idx[idx],:],
                                          u.int(),
                                          s.int(), 
                                          uhat, shat,
                                          None)
            
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
              U_raw,
              S_raw,
              config={}, 
              plot=False, 
              gene_plot=[], 
              cluster_key = "clusters",
              figure_path = "figures", 
              embed="umap"):
        self.train_mse = []
        self.test_mse = []
        
        self.load_config(config)
        
        print("------------------------ Train a Vanilla VAE ------------------------")
        #Get data loader
        U, S = np.array(adata.layers['unspliced'].todense()), np.array(adata.layers['spliced'].todense())
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
        param_nn = list(self.encoder.parameters())
        param_ode = [self.decoder.alpha, self.decoder.beta, self.decoder.gamma, self.decoder.ton, self.decoder.toff] 
        if(self.config['train_scaling']):
            param_ode = param_ode+[self.decoder.scaling_u, self.decoder.scaling_s]

        optimizer = torch.optim.Adam(param_nn, lr=self.config["learning_rate"], weight_decay=self.config["lambda"])
        optimizer_ode = torch.optim.Adam(param_ode, lr=self.config["learning_rate_ode"])
        print("*********                      Finished.                      *********")
      
        #Main Training Process
        print("*********                    Start training                   *********")
        print(f"Total Number of Iterations Per Epoch: {len(data_loader)}, test iteration: {self.config['test_iter']}")
        
        n_epochs = self.config["n_epochs"]
        
        start = time.time()
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
                                       True, 
                                       figure_path)
                self.set_mode('train')
                elbo_test = self.loss_test[-1] if len(self.loss_test)>0 else -np.inf
                print(f"Epoch {epoch+1}: Train ELBO = {elbo_train:.3f}, Test ELBO = {elbo_test:.3f}, \t Total Time = {convert_time(time.time()-start)}")
            
            if(stop_training):
                print(f"********* Early Stop Triggered at epoch {epoch+1}. *********")
                break
        
        print(f"*********              Finished. Total Time = {convert_time(time.time()-start)}             *********")
        #Plot final results
        if(plot):
            elbo_train = self.test(train_set,
                                   Xembed[self.train_idx],
                                   "final-train", 
                                   False,
                                   gind, 
                                   gene_plot,
                                   True, 
                                   figure_path)
            elbo_test = self.test(test_set,
                                  Xembed[self.test_idx],
                                  "final-test", 
                                  True,
                                  gind, 
                                  gene_plot,
                                  True, 
                                  figure_path)
            print(f"Final: Train ELBO = {elbo_train:.3f}, Test ELBO = {elbo_test:.3f}")
            plot_train_loss(self.loss_train, range(1,len(self.loss_train)+1), save=f'{figure_path}/train_loss_velovae.png')
            if(self.config["test_iter"]>0):
                plot_test_loss(self.loss_test, [i*self.config["test_iter"] for i in range(1,len(self.loss_test)+1)], save=f'{figure_path}/test_loss_velovae.png')
        
        if(plot):
            plot_train_loss(self.train_mse, range(1,len(self.train_mse)+1), save=f'{figure_path}/train_mse_velovae.png')
            if(self.config["test_iter"]>0):
                plot_test_loss(self.test_mse, [i*self.config["test_iter"] for i in range(1,len(self.test_mse)+1)], save=f'{figure_path}/test_mse_velovae.png')
    
    def pred_all(self, data, cell_labels, mode='test', output=["uhat", "shat", "t"], gene_idx=None, continuous=True):
        N, G = data.shape[0], data.shape[1]//2
        elbo, mse = 0, 0
        if("uhat" in output):
            Uhat = None if gene_idx is None else np.zeros((N,len(gene_idx)))
        if("shat" in output):
            Shat = None if gene_idx is None else np.zeros((N,len(gene_idx)))
        if("t" in output):
            t_out = np.zeros((N))
            std_t_out = np.zeros((N))
        
        corr = 0
        with torch.no_grad():
            B = min(N//3, 5000)
            Nb = N // B
            for i in range(Nb):
                data_in = torch.tensor(data[i*B:(i+1)*B]).float().to(self.device)
                if(mode=="test"):
                    p_t = self.p_t[:,self.test_idx[i*B:(i+1)*B],:]
                    lu_scale = self.lu_scale[self.test_idx[i*B:(i+1)*B]]
                    ls_scale = self.ls_scale[self.test_idx[i*B:(i+1)*B]]
                elif(mode=="train"):
                    p_t = self.p_t[:,self.train_idx[i*B:(i+1)*B],:]
                    lu_scale = self.lu_scale[self.train_idx[i*B:(i+1)*B]]
                    ls_scale = self.ls_scale[self.train_idx[i*B:(i+1)*B]]
                else:
                    p_t = self.p_t[:,i*B:(i+1)*B,:]
                    lu_scale = self.lu_scale[i*B:(i+1)*B]
                    ls_scale = self.ls_scale[i*B:(i+1)*B]
                mu_tx, std_tx, uhat, shat = self.eval_model(data_in, lu_scale, ls_scale)
                u_raw, s_raw = data_in[:,:G], data_in[:,G:]
                #u_sum, s_sum = torch.sum(uhat, 1, keepdim=True), torch.sum(shat, 1, keepdim=True)
                
                loss = self.vae_risk_poisson((mu_tx, std_tx), p_t,
                                             u_raw.int(), s_raw.int(), 
                                             uhat, shat, 
                                             None)
                u_sample, s_sample = self.sample_poisson(uhat, shat)
                
                mse_batch = np.mean( (u_raw.cpu().numpy() - u_sample.cpu().numpy())**2 + ( s_raw.cpu().numpy() - s_sample.cpu().numpy())**2)
                elbo = elbo - (B/N)*loss
                mse = mse + (B/N) * mse_batch
                #corr = corr + (B/N) * self.corr_vel(uhat*lu_scale, shat*ls_scale, rho).detach().cpu().item()
                
                if("uhat" in output and gene_idx is not None):
                    Uhat[i*B:(i+1)*B] = uhat[:,gene_idx].cpu().numpy()
                if("shat" in output and gene_idx is not None):
                    Shat[i*B:(i+1)*B] = shat[:,gene_idx].cpu().numpy()
                if("t" in output):
                    t_out[i*B:(i+1)*B] = mu_tx.cpu().squeeze().numpy()
                    std_t_out[i*B:(i+1)*B] = std_tx.cpu().squeeze().numpy()
                
            if(N > B*Nb):
                data_in = torch.tensor(data[B*Nb:]).float().to(self.device)
                if(mode=="test"):
                    p_t = self.p_t[:,self.test_idx[B*Nb:],:]
                    lu_scale = self.lu_scale[self.test_idx[B*Nb:]]
                    ls_scale = self.ls_scale[self.test_idx[B*Nb:]]
                elif(mode=="train"):
                    p_t = self.p_t[:,self.train_idx[B*Nb:],:]
                    lu_scale = self.lu_scale[self.train_idx[B*Nb:]]
                    ls_scale = self.ls_scale[self.train_idx[B*Nb:]]
                else:
                    p_t = self.p_t[:,B*Nb:,:]
                    lu_scale = self.lu_scale[B*Nb:]
                    ls_scale = self.ls_scale[B*Nb:]
                mu_tx, std_tx, uhat, shat = self.eval_model(data_in, lu_scale, ls_scale)
                u_raw, s_raw = data_in[:,:G], data_in[:,G:]
                #u_sum, s_sum = torch.sum(uhat, 1, keepdim=True), torch.sum(shat, 1, keepdim=True)
                
                loss = self.vae_risk_poisson((mu_tx, std_tx), p_t,
                                             u_raw.int(), s_raw.int(), 
                                             uhat, shat,
                                             None)
                u_sample, s_sample = self.sample_poisson(uhat, shat)
                
                mse_batch = np.mean( (u_raw.cpu().numpy() - u_sample.cpu().numpy())**2 + ( s_raw.cpu().numpy() - s_sample.cpu().numpy())**2)
                elbo = elbo - ((N-B*Nb)/N)*loss
                mse = mse + ((N-B*Nb)/N)*mse_batch

                if("uhat" in output and gene_idx is not None):
                    Uhat[Nb*B:] = uhat[:,gene_idx].cpu().numpy()
                if("shat" in output and gene_idx is not None):
                    Shat[Nb*B:] = shat[:,gene_idx].cpu().numpy()
                if("t" in output):
                    t_out[Nb*B:] = mu_tx.cpu().squeeze().numpy()
                    std_t_out[Nb*B:] = std_tx.cpu().squeeze().numpy()
        out = []
        if("uhat" in output):
            out.append(Uhat)
        if("shat" in output):
            out.append(Shat)
        if("t" in output):
            out.append(t_out)
            out.append(std_t_out)

        return out, elbo.cpu().item()
    
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
        out, elbo = self.pred_all(dataset.data, self.cell_labels, mode, ["uhat", "shat", "t"], gind)
        Uhat, Shat, t, std_t = out[0], out[1], out[2], out[3]
        
        
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
    
    def gene_likelihood_poisson(self, 
                                U,S,
                                Uhat, Shat, 
                                b=5000):
        N,G = U.shape
        Nb = N//b
        logpu, logps = np.empty((N,G)), np.empty((N,G))
        for i in range(Nb):
            ps_u = Poisson(torch.tensor(Uhat[i*b:(i+1)*b]))
            ps_s = Poisson(torch.tensor(Shat[i*b:(i+1)*b]))
            logpu[i*b:(i+1)*b] = ps_u.log_prob(torch.tensor(U[i*b:(i+1)*b], dtype=int))
            logps[i*b:(i+1)*b] = ps_s.log_prob(torch.tensor(S[i*b:(i+1)*b], dtype=int))
        if(Nb*b<N):
            ps_u = Poisson(torch.tensor(Uhat[Nb*b:]))
            ps_s = Poisson(torch.tensor(Shat[Nb*b:]))
            logpu[Nb*b:] = ps_u.log_prob(torch.tensor(U[Nb*b:], dtype=int))
            logps[Nb*b:] = ps_s.log_prob(torch.tensor(S[Nb*b:], dtype=int))
        return np.exp((logpu+logps)).mean(0)
    
    def save_anndata(self, adata, key, file_path, file_name=None):
        self.set_mode('eval')
        os.makedirs(file_path, exist_ok=True)
        
        adata.var[f"{key}_alpha"] = np.exp(self.decoder.alpha.detach().cpu().numpy())
        adata.var[f"{key}_beta"] = np.exp(self.decoder.beta.detach().cpu().numpy())
        adata.var[f"{key}_gamma"] = np.exp(self.decoder.gamma.detach().cpu().numpy())
        adata.var[f"{key}_toff"] = np.exp(self.decoder.toff.detach().cpu().numpy())
        adata.var[f"{key}_ton"] = np.exp(self.decoder.ton.detach().cpu().numpy())
        adata.var[f"{key}_scaling_u"] = self.decoder.scaling_u.exp().detach().cpu().numpy()
        adata.var[f"{key}_scaling_s"] = self.decoder.scaling_s.exp().detach().cpu().numpy()
        
        U,S = np.array(adata.layers['unspliced'].todense()), np.array(adata.layers['spliced'].todense())
        X = np.concatenate((U,S),1)

        out, elbo = self.pred_all(X, self.cell_labels, "both", gene_idx=np.array(range(adata.n_vars)))
        Uhat, Shat, t, std_t = out
        
        adata.obs[f"{key}_time"] = t
        adata.obs[f"{key}_std_t"] = std_t
        
        adata.layers[f"{key}_uhat"] = Uhat
        adata.layers[f"{key}_shat"] = Shat
        
        adata.uns[f"{key}_train_idx"] = self.train_idx
        adata.uns[f"{key}_test_idx"] = self.test_idx
        
        scaling_u = adata.var[f"{key}_scaling_u"].to_numpy()
        scaling_s = adata.var[f"{key}_scaling_s"].to_numpy()
        lu_scale = adata.obs["library_scale_u"].to_numpy().reshape(-1,1)
        ls_scale = adata.obs["library_scale_s"].to_numpy().reshape(-1,1)
        adata.layers[f"{key}_velocity"] = adata.var[f"{key}_beta"].to_numpy() * Uhat / scaling_u / lu_scale - adata.var[f"{key}_gamma"].to_numpy() * Shat / scaling_s / ls_scale
        
        if(file_name is not None):
            adata.write_h5ad(f"{file_path}/{file_name}")



























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
            self.scaling_u = nn.Parameter(torch.empty(G, device=device).float())
            self.scaling_s = nn.Parameter(torch.empty(G, device=device).float())
            self.ton = nn.Parameter(torch.empty(G, device=device).float())
            
            self.load_state_dict(torch.load(checkpoint, map_location=device))
        else:
            self.init_weights()
            
            #Library Size
            U, S = np.array(adata.layers['unspliced'].todense()), np.array(adata.layers['spliced'].todense())
            if(scale_cell):
                U, S, lu, ls = scale_by_cell(U, S, train_idx, separate_us_scale)
                adata.obs["library_scale_u"] = lu
                adata.obs["library_scale_s"] = ls
            else:
                lu, ls = get_cell_scale(U, S, train_idx, separate_us_scale)
                #lu, ls = np.ones((adata.n_obs)), np.ones((adata.n_obs))
                adata.obs["library_scale_u"] = lu
                adata.obs["library_scale_s"] = ls
            U, S = np.array(adata.layers['unspliced'].todense()), np.array(adata.layers['spliced'].todense())
            U = U[train_idx]
            S = S[train_idx]
            #U, S = adata.layers["Mu"][train_idx], adata.layers["Ms"][train_idx]
            X = np.concatenate((U,S),1)
            if(add_noise):
                noise = np.exp(np.random.normal(size=(len(train_idx), 2*G))*1e-3)
                X = X + noise
            

            if(init_method == "random"):
                print("Random Initialization.")
                #alpha, beta, gamma, scaling, toff, u0, s0, sigma_u, sigma_s, T, Rscore = init_params(X,p,fit_scaling=True)
                scaling, scaling_s = get_gene_scale(U,S,None)
                self.alpha = nn.Parameter(torch.normal(0.0, 0.01, size=(G,), device=device).float())
                self.beta =  nn.Parameter(torch.normal(0.0, 0.01, size=(G,), device=device).float())
                self.gamma = nn.Parameter(torch.normal(0.0, 0.01, size=(G,), device=device).float())
                self.ton = torch.nn.Parameter(torch.ones(G, device=device).float()*(-10))
            elif(init_method == "tprior"):
                print("Initialization using prior time.")
                alpha, beta, gamma, scaling, toff, u0, s0, T = init_params_raw(X,p,fit_scaling=True)
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
                alpha, beta, gamma, scaling, toff, u0, s0, T = init_params_raw(X,p,fit_scaling=True)
                
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
                #self.ton = nn.Parameter((torch.ones(adata.n_vars, device=device)*(-10)).float()) if init_ton_zero else nn.Parameter(torch.tensor(np.log(ton+1e-10), device=device).float())
                self.ton = nn.Parameter((torch.ones(adata.n_vars, device=device)*(-10)).float())
                
            
            
            self.scaling_u = nn.Parameter(torch.tensor(np.log(scaling), device=device).float())
            #self.scaling_s = nn.Parameter(torch.tensor(np.log(scaling_s), device=device).float())
            self.scaling_s = nn.Parameter(torch.zeros(G,device=device).float())
            
        
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
            cell_labels = adata.obs["clusters"].to_numpy()[train_idx]
            cell_mask = cell_labels==init_type
            self.u0 = nn.Parameter(torch.tensor(np.log(U[cell_mask]/lu[cell_mask].reshape(-1,1)/scaling_u.exp().mean(0)+1e-10), device=device).float())
            self.s0 = nn.Parameter(torch.tensor(np.log(S[cell_mask]/ls[cell_mask].reshape(-1,1)/scaling_s.exp().mean(0)+1e-10), device=device).float())

        self.scaling_u.requires_grad = False
        self.scaling_s.requires_grad = False
    
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
            Uhat, Shat = pred_su(F.leaky_relu(t - self.ton.exp(), neg_slope), self.u0.exp()/self.scaling_u.exp(),  self.s0.exp()/self.scaling_s.exp(), alpha, self.beta.exp(), self.gamma.exp())
        else:
            if(condition is None):
                rho = torch.sigmoid(self.fc_out2(self.net_rho2(z)))
            else:
                rho = torch.sigmoid(self.fc_out2(self.net_rho2(torch.cat((z,condition),1))))
            alpha = self.alpha.exp()*rho
            Uhat, Shat = pred_su(F.leaky_relu(t-t0, neg_slope), u0, s0, alpha, self.beta.exp(), self.gamma.exp())
        
        return nn.functional.relu(Uhat), nn.functional.relu(Shat), rho
    
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


def logp_exp(x, lamb, eps=1e-6):
    return torch.log(1-torch.exp(-(lamb+eps)))-lamb*x

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
                 hidden_size=(500, 250, 250, 500), 
                 init_method="steady", 
                 init_key=None,
                 tprior=None, 
                 init_type=None,
                 init_ton_zero=True,
                 add_noise=True,
                 time_distribution="gaussian",
                 std_z_prior=0.01,
                 scale_cell=False,
                 separate_us_scale=True,
                 checkpoints=[None, None]):
        
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
            "knn_adaptive": False,

            #Training Parameters
            "n_epochs":1000, 
            "n_epochs_post":1000,
            "batch_size":128, 
            "learning_rate":2e-4, 
            "learning_rate_ode":5e-4, 
            "learning_rate_post":2e-4,
            "learning_rate_ode_post":2e-4,
            "lambda":1e-3, 
            "lambda_rho":1e-3,
            "kl_t":1.0, 
            "kl_z":1.0, 
            "kl_l":1.0,
            "test_iter":None, 
            "save_epoch":100, 
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
            "reg_vel":False,
            
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
        self.get_prior(adata, time_distribution, tmax, tprior)
        
        self.p_z = torch.stack([torch.zeros(N,dim_z), torch.ones(N,dim_z)*self.config["std_z_prior"]]).float().to(self.device)
        
        self.use_knn = False
        self.u0 = None
        self.s0 = None
        self.t0 = None
        self.lu_scale = torch.tensor(adata.obs['library_scale_u'].to_numpy()).unsqueeze(-1).float().to(self.device)
        self.ls_scale = torch.tensor(adata.obs['library_scale_s'].to_numpy()).unsqueeze(-1).float().to(self.device)
        

        print(f"Mean library scale: {self.lu_scale.detach().cpu().numpy().mean()}, {self.ls_scale.detach().cpu().numpy().mean()}")

        #Class attributes for training
        self.loss_train, self.loss_test = [],[]
        self.counter = 0 #Count the number of iterations
        self.n_drop = 0 #Count the number of consecutive iterations with little decrease in loss
    
    def set_mode(self,mode):
        #Set the model to either training or evaluation mode.
        
        if(mode=='train'):
            self.encoder.train()
            self.decoder.train()
        elif(mode=='eval'):
            self.encoder.eval()
            self.decoder.eval()
        else:
            print("Warning: mode not recognized. Must be 'train' or 'test'! ")
    
    def forward(self, data_in, lu_scale, ls_scale, u0=None, s0=None, t0=None, condition=None):
        data_in_scale = data_in
        if(self.config["scale_gene_encoder"]):
            data_in_scale = torch.cat((data_in_scale[:,:data_in_scale.shape[1]//2]/self.decoder.scaling_u.exp(), data_in_scale[:,data_in_scale.shape[1]//2:]),1)
        if(self.config["scale_cell_encoder"]):
            data_in_scale = torch.cat((data_in_scale[:,:data_in_scale.shape[1]//2]/lu_scale, data_in_scale[:,data_in_scale.shape[1]//2:]/ls_scale),1)
        if(self.config["log1p"]):
            data_in_scale = torch.log1p(data_in_scale)
        mu_t, std_t, mu_z, std_z = self.encoder.forward(data_in_scale, condition)
        t = self.sample(mu_t, std_t)
        z = self.sample(mu_z, std_z)

        uhat, shat, rho = self.decoder.forward(t, z, condition, u0, s0, t0, neg_slope=self.config["neg_slope"])
        uhat = uhat*self.decoder.scaling_u.exp()

        return mu_t, std_t, mu_z, std_z, t, z, uhat, shat, rho
    
    def eval_model(self, data_in, lu_scale, ls_scale, u0=None, s0=None, t0=None, condition=None, continuous=True):
        data_in_scale = data_in
        if(self.config["scale_gene_encoder"]):
            data_in_scale = torch.cat((data_in_scale[:,:data_in_scale.shape[1]//2]/self.decoder.scaling_u.exp(), data_in_scale[:,data_in_scale.shape[1]//2:]),1)
        if(self.config["scale_cell_encoder"]):
            data_in_scale = torch.cat((data_in_scale[:,:data_in_scale.shape[1]//2]/lu_scale, data_in_scale[:,data_in_scale.shape[1]//2:]/ls_scale),1)
        if(self.config["log1p"]):
            data_in_scale = torch.log1p(data_in_scale)
        mu_t, std_t, mu_z, std_z = self.encoder.forward(data_in_scale, condition)
        uhat, shat, rho = self.decoder.forward(mu_t, mu_z, condition, u0=u0, s0=s0, t0=t0, neg_slope=0.0)
        uhat = uhat*self.decoder.scaling_u.exp()

        if(not continuous):
            poisson_u = Poisson(F.softplus(uhat, beta=100))
            poisson_s = Poisson(F.softplus(shat, beta=100))
            u_out = poisson_u.sample()
            s_out = poisson_s.sample()
            return mu_t, std_t, mu_z, std_z, u_out, s_out, rho
        return mu_t, std_t, mu_z, std_z, uhat, shat, rho
    
    def update_x0(self, U, S, gind, gene_plot, path):
        start = time.time()
        self.set_mode('eval')
        out, elbo = self.pred_all(np.concatenate((U,S),1), self.cell_labels, "both", ["uhat","shat","t","z"], gene_idx=np.array(range(U.shape[1])))
        t, std_t, z = out[4], out[5], out[6]
        #Clip the time to avoid outliers
        t = np.clip(t, 0, np.quantile(t, 0.99))
        dt = (self.config["dt"][0]*(t.max()-t.min()), self.config["dt"][1]*(t.max()-t.min()))
        scaling_u = np.exp(self.decoder.scaling_u.detach().cpu().numpy())
        #scaling_s = np.exp(self.decoder.scaling_s.detach().cpu().numpy())
        u0, s0, t0 = knnx0(out[0][self.train_idx]/scaling_u, out[2][self.train_idx], t[self.train_idx], z[self.train_idx], t, z, dt, self.config["n_neighbors"], adaptive=self.config["knn_adaptive"], std_t=std_t)
        print(f"Finished. Actual Time: {convert_time(time.time()-start)}")
        self.set_mode('train')
        return u0, s0, t0.reshape(-1,1)
    
    
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
    
    def corr_vel(self, uhat, shat, rho):
        cos_sim = nn.CosineSimilarity(dim=0)
        vu = rho * self.decoder.alpha.exp() - self.decoder.beta.exp() * uhat / self.decoder.scaling_u.exp()
        vs = self.decoder.beta.exp() * uhat / self.decoder.scaling_u.exp() - self.decoder.gamma.exp() * shat
        vu = vu-vu.mean(0)
        vs = vs-vs.mean(0)
        uhat_0 = uhat - uhat.mean(0)
        shat_0 = shat - shat.mean(0)
        
        return torch.sum(cos_sim(vu,uhat_0)+cos_sim(vs,shat_0)-cos_sim(vs,uhat))
    
    def vae_risk_poisson(self, 
                         q_tx, p_t, 
                         q_zx, p_z, 
                         u, s, uhat, shat, 
                         weight=None,
                         eps=1e-6):
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
        
        if( weight is not None):
            logp = logp*weight
        
        err_rec = torch.mean(logp.sum(1))
        
        return (- err_rec + self.config["kl_t"]*kldt + self.config["kl_z"]*kldz)
    
    def vae_risk_exp(self, 
                     q_tx, p_t, 
                     q_zx, p_z, 
                     u, s, uhat, shat, 
                     weight=None,
                     eps=1e-6):
        kldt = self.kl_time(q_tx[0], q_tx[1], p_t[0], p_t[1], tail=self.config["tail"])
        kldz = kl_gaussian(q_zx[0], q_zx[1], p_z[0], p_z[1])
        
        #poisson
        mask_u = (torch.isnan(uhat) | torch.isinf(uhat)).float()
        uhat = uhat * (1-mask_u)
        mask_s = (torch.isnan(shat) | torch.isinf(shat)).float()
        shat = shat * (1-mask_s)
        logp = logp_exp(u, 1/uhat)+logp_exp(s, 1/shat)
        
        if( weight is not None):
            logp = logp*weight
        
        err_rec = torch.mean(torch.sum(logp,1))
        
        return (- err_rec + self.config["kl_t"]*kldt + self.config["kl_z"]*kldz)
        
    
    def vae_risk_nb(self, 
                    q_tx, p_t, 
                    q_zx, p_z, 
                    u, s, uhat, shat, 
                    p_nb_u, p_nb_s,
                    weight=None,
                    eps=1e-6):
        #This is the negative ELBO.
        kldt = self.kl_time(q_tx[0], q_tx[1], p_t[0], p_t[1], tail=self.config["tail"])
        kldz = kl_gaussian(q_zx[0], q_zx[1], p_z[0], p_z[1])
        
        nb_u = NegativeBinomial(F.relu(uhat)+1e-2)*(1-p_nb_u)/p_nb_u, p_nb_u.repeat(uhat.shape[0],1)
        nb_s = NegativeBinomial(F.relu(shat)+1e-2)*(1-p_nb_s)/p_nb_s, p_nb_s.repeat(shat.shape[0],1)
        logp = nb_u.log_prob(u) + nb_s.log_prob(s)
        
        if( weight is not None):
            logp = logp*weight
        err_rec = torch.mean(torch.sum(logp,1))
        
        return (- err_rec + self.config["kl_t"]*kldt + self.config["kl_z"]*kldz )
    
    def train_epoch(self, train_loader, test_set, optimizer, optimizer2=None, K=1, reg_vel=False):
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
            
            xbatch, idx = batch[0].to(self.device), batch[3]
            u = xbatch[:,:xbatch.shape[1]//2]
            s = xbatch[:,xbatch.shape[1]//2:]
            
            u0 = self.u0[self.train_idx[idx]] if self.use_knn else None
            s0 = self.s0[self.train_idx[idx]] if self.use_knn else None
            t0 = self.t0[self.train_idx[idx]] if self.use_knn else None
            lu_scale = self.lu_scale[self.train_idx[idx]]
            ls_scale = self.ls_scale[self.train_idx[idx]]
            
            condition = F.one_hot(batch[1].to(self.device), self.n_type).float() if self.enable_cvae else None
            mu_tx, std_tx, mu_zx, std_zx, t, z, uhat, shat, rho = self.forward(xbatch.float(), lu_scale, ls_scale, u0, s0, t0, condition)
            
            loss = self.vae_risk_poisson((mu_tx, std_tx), self.p_t[:,self.train_idx[idx],:],
                                          (mu_zx, std_zx), self.p_z[:,self.train_idx[idx],:],
                                          u.int(),
                                          s.int(), 
                                          uhat*lu_scale, shat*ls_scale,
                                          None)
            if(reg_vel):
                loss = loss+self.corr_vel(uhat, shat, rho)
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
              U_raw,
              S_raw,
              config={}, 
              plot=False, 
              gene_plot=[], 
              cluster_key = "clusters",
              figure_path = "figures", 
              embed="umap"):
        self.train_mse = []
        self.test_mse = []
        
        self.load_config(config)
        
        print("--------------------------- Train a VeloVAE ---------------------------")
        #Get data loader
        U, S = np.array(adata.layers['unspliced'].todense()), np.array(adata.layers['spliced'].todense())
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
            param_ode.append(self.decoder.ton)
        if(self.config['train_scaling']):
            self.decoder.scaling_u.requires_grad = True
            self.decoder.scaling_s.requires_grad = True
            param_ode = param_ode+[self.decoder.scaling_u, self.decoder.scaling_s]

        optimizer = torch.optim.Adam(param_nn, lr=self.config["learning_rate"], weight_decay=self.config["lambda"])
        optimizer_ode = torch.optim.Adam(param_ode, lr=self.config["learning_rate_ode"])
        print("*********                      Finished.                      *********")
      
        #Main Training Process
        print("*********                    Start training                   *********")
        print("*********                      Stage  1                       *********")
        print(f"Total Number of Iterations Per Epoch: {len(data_loader)}, test iteration: {self.config['test_iter']}")
        
        n_epochs = self.config["n_epochs"]
        
        start = time.time()
        for epoch in range(n_epochs):
            #Train the encoder
            if(self.config["k_alt"] is None):
                stop_training = self.train_epoch(data_loader, test_set, optimizer, reg_vel=self.config["reg_vel"])
                
                if(epoch>=self.config["n_warmup"]):
                    stop_training_ode = self.train_epoch(data_loader, test_set, optimizer_ode, reg_vel=self.config["reg_vel"])
                    if(stop_training_ode):
                        print(f"*********       Stage 1: Early Stop Triggered at epoch {epoch+1}.       *********")
                        break
            else:
                if(epoch>=self.config["n_warmup"]):
                    stop_training = self.train_epoch(data_loader, test_set, optimizer_ode, optimizer, self.config["k_alt"], reg_vel=self.config["reg_vel"])
                else:
                    stop_training = self.train_epoch(data_loader, test_set, optimizer, None, self.config["k_alt"], reg_vel=self.config["reg_vel"])
            
            if(plot and (epoch==0 or (epoch+1) % self.config["save_epoch"] == 0)):
                elbo_train = self.test(train_set,
                                       Xembed[self.train_idx],
                                       f"train{epoch+1}", 
                                       False,
                                       gind, 
                                       gene_plot,
                                       True, 
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
        u0, s0, t0 = self.update_x0(U, S, gind, gene_plot, figure_path)
        self.u0 = torch.tensor(u0, dtype=float, device=self.device)
        self.s0 = torch.tensor(s0, dtype=float, device=self.device)
        self.t0 = torch.tensor(t0, dtype=float, device=self.device)
        
        
        self.use_knn = True
        self.decoder.init_weights()
        #Plot the initial conditions
        if(plot):
            plot_time(t0.squeeze(), Xembed, save=f"{figure_path}/t0.png")
            for i in range(len(gind)):
                idx = gind[i]
                t0_plot = t0[self.train_idx].squeeze()
                u0_plot = u0[self.train_idx,idx]
                s0_plot = s0[self.train_idx,idx]
                plot_sig_(t0_plot, 
                         u0_plot, s0_plot, 
                         cell_labels=cell_labels_raw[self.train_idx],
                         title=gene_plot[i], 
                         save=f"{figure_path}/{gene_plot[i]}-x0.png")
        self.n_drop = 0
        param_ode = [self.decoder.alpha, self.decoder.beta, self.decoder.gamma, self.decoder.u0, self.decoder.s0, self.u0, self.s0, self.t0] 
        optimizer_ode = torch.optim.Adam(param_ode, lr=self.config["learning_rate_ode"])
        param_post = list(self.decoder.net_rho2.parameters())+list(self.decoder.fc_out2.parameters())
        optimizer_post = torch.optim.Adam(param_post, lr=self.config["learning_rate_post"], weight_decay=self.config["lambda_rho"])
        for epoch in range(self.config["n_epochs_post"]):
            if(self.config["k_alt"] is None):
                stop_training = self.train_epoch(data_loader, test_set, optimizer_post, reg_vel=self.config["reg_vel"])
                
                if(epoch>=self.config["n_warmup"]):
                    stop_training_ode = self.train_epoch(data_loader, test_set, optimizer_ode, reg_vel=self.config["reg_vel"])
                    if(stop_training_ode):
                        print(f"*********       Stage 2: Early Stop Triggered at epoch {epoch+n_stage1+1}.       *********")
                        break
            else:
                if(epoch>=self.config["n_warmup"]):
                    stop_training = self.train_epoch(data_loader, test_set, optimizer_post, optimizer_ode, self.config["k_alt"], reg_vel=self.config["reg_vel"])
                else:
                    stop_training = self.train_epoch(data_loader, test_set, optimizer_post, None, self.config["k_alt"], reg_vel=self.config["reg_vel"])
            
            
            
            if(plot and (epoch==0 or (epoch+n_stage1+1) % self.config["save_epoch"] == 0)):
                elbo_train = self.test(train_set,
                                       Xembed[self.train_idx],
                                       f"train{epoch+n_stage1+1}", 
                                       False,
                                       gind, 
                                       gene_plot,
                                       True, 
                                       figure_path)
                self.decoder.train()
                elbo_test = self.loss_test[-1] if len(self.loss_test)>n_test1 else -np.inf
                print(f"Epoch {epoch+n_stage1+1}: Train ELBO = {elbo_train:.3f}, Test ELBO = {elbo_test:.3f}, \t Total Time = {convert_time(time.time()-start)}")
            
            if(stop_training):
                print(f"*********       Stage 2: Early Stop Triggered at epoch {epoch+n_stage1+1}.       *********")
                break
        
        print(f"*********              Finished. Total Time = {convert_time(time.time()-start)}             *********")
        print(f"Final: Train ELBO = {elbo_train:.3f},           Test ELBO = {elbo_test:.3f}")
        print(f"       Training MSE = {self.train_mse[-1]:.3f}, Test MSE = {self.test_mse[-1]:.3f}")
        #Plot final results
        if(plot):
            elbo_train = self.test(train_set,
                                   Xembed[self.train_idx],
                                   "final-train", 
                                   False,
                                   gind, 
                                   gene_plot,
                                   True, 
                                   figure_path)
            elbo_test = self.test(test_set,
                                  Xembed[self.test_idx],
                                  "final-test", 
                                  True,
                                  gind, 
                                  gene_plot,
                                  True, 
                                  figure_path)
            plot_train_loss(self.loss_train, range(1,len(self.loss_train)+1), save=f'{figure_path}/train_loss_velovae.png')
            if(self.config["test_iter"]>0):
                plot_test_loss(self.loss_test, [i*self.config["test_iter"] for i in range(1,len(self.loss_test)+1)], save=f'{figure_path}/test_loss_velovae.png')
        
        if(plot):
            plot_train_loss(self.train_mse, range(1,len(self.train_mse)+1), save=f'{figure_path}/train_mse_velovae.png')
            if(self.config["test_iter"]>0):
                plot_test_loss(self.test_mse, [i*self.config["test_iter"] for i in range(1,len(self.test_mse)+1)], save=f'{figure_path}/test_mse_velovae.png')
    
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
        
        corr = 0
        with torch.no_grad():
            B = min(N//5, 1000)
            Nb = N // B
            for i in range(Nb):
                data_in = torch.tensor(data[i*B:(i+1)*B]).float().to(self.device)
                if(mode=="test"):
                    u0 = self.u0[self.test_idx[i*B:(i+1)*B]] if self.use_knn else None
                    s0 = self.s0[self.test_idx[i*B:(i+1)*B]] if self.use_knn else None
                    t0 = self.t0[self.test_idx[i*B:(i+1)*B]] if self.use_knn else None
                    y_onehot = F.one_hot(torch.tensor(cell_labels[self.test_idx[i*B:(i+1)*B]], dtype=int, device=self.device), self.n_type).float() if self.enable_cvae else None
                    p_t = self.p_t[:,self.test_idx[i*B:(i+1)*B],:]
                    p_z = self.p_z[:,self.test_idx[i*B:(i+1)*B],:]
                    lu_scale = self.lu_scale[self.test_idx[i*B:(i+1)*B],:]
                    ls_scale = self.ls_scale[self.test_idx[i*B:(i+1)*B],:]
                    mu_tx, std_tx, mu_zx, std_zx, uhat, shat, rho = self.eval_model(data_in, lu_scale, ls_scale, u0, s0, t0, y_onehot, continuous)
                elif(mode=="train"):
                    u0 = self.u0[self.train_idx[i*B:(i+1)*B]] if self.use_knn else None
                    s0 = self.s0[self.train_idx[i*B:(i+1)*B]] if self.use_knn else None
                    t0 = self.t0[self.train_idx[i*B:(i+1)*B]] if self.use_knn else None
                    y_onehot = F.one_hot(torch.tensor(cell_labels[self.train_idx[i*B:(i+1)*B]], dtype=int, device=self.device), self.n_type).float() if self.enable_cvae else None
                    p_t = self.p_t[:,self.train_idx[i*B:(i+1)*B],:]
                    p_z = self.p_z[:,self.train_idx[i*B:(i+1)*B],:]
                    lu_scale = self.lu_scale[self.train_idx[i*B:(i+1)*B],:]
                    ls_scale = self.ls_scale[self.train_idx[i*B:(i+1)*B],:]
                    mu_tx, std_tx, mu_zx, std_zx, uhat, shat, rho = self.eval_model(data_in, lu_scale, ls_scale, u0, s0, t0, y_onehot, continuous)
                else:
                    u0 = self.u0[i*B:(i+1)*B] if self.use_knn else None
                    s0 = self.s0[i*B:(i+1)*B] if self.use_knn else None
                    t0 = self.t0[i*B:(i+1)*B] if self.use_knn else None
                    y_onehot = F.one_hot(torch.tensor(cell_labels[i*B:(i+1)*B], dtype=int, device=self.device), self.n_type).float() if self.enable_cvae else None
                    p_t = self.p_t[:,i*B:(i+1)*B,:]
                    p_z = self.p_z[:,i*B:(i+1)*B,:]
                    lu_scale = self.lu_scale[i*B:(i+1)*B,:]
                    ls_scale = self.ls_scale[i*B:(i+1)*B,:]
                    mu_tx, std_tx, mu_zx, std_zx, uhat, shat, rho = self.eval_model(data_in, lu_scale, ls_scale, u0, s0, t0, y_onehot, continuous)

                u_raw, s_raw = data_in[:,:G], data_in[:,G:]
                u_sum, s_sum = torch.sum(uhat, 1, keepdim=True), torch.sum(shat, 1, keepdim=True)
                
                loss = self.vae_risk_poisson((mu_tx, std_tx), p_t,
                                             (mu_zx, std_zx), p_z,
                                             u_raw.int(), s_raw.int(), 
                                             uhat*lu_scale, shat*ls_scale, 
                                             None)
                u_sample, s_sample = self.sample_poisson(uhat*lu_scale, shat*ls_scale)
                
                mse_batch = np.mean( (u_raw.cpu().numpy() - u_sample.cpu().numpy())**2 + ( s_raw.cpu().numpy() - s_sample.cpu().numpy())**2)
                elbo = elbo - (B/N)*loss
                mse = mse + (B/N) * mse_batch
                #corr = corr + (B/N) * self.corr_vel(uhat*lu_scale, shat*ls_scale, rho).detach().cpu().item()
                
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
                    u0 = self.u0[self.test_idx[Nb*B:]] if self.use_knn else None
                    s0 = self.s0[self.test_idx[Nb*B:]] if self.use_knn else None
                    t0 = self.t0[self.test_idx[Nb*B:]] if self.use_knn else None
                    y_onehot = F.one_hot(torch.tensor(cell_labels[self.test_idx[Nb*B:]], dtype=int, device=self.device), self.n_type).float() if self.enable_cvae else None
                    p_t = self.p_t[:,self.test_idx[Nb*B:],:]
                    p_z = self.p_z[:,self.test_idx[Nb*B:],:]
                    lu_scale = self.lu_scale[self.test_idx[Nb*B:],:]
                    ls_scale = self.ls_scale[self.test_idx[Nb*B:],:]
                    mu_tx, std_tx, mu_zx, std_zx, uhat, shat, rho = self.eval_model(data_in, lu_scale, ls_scale, u0, s0, t0, y_onehot)
                elif(mode=="train"):
                    u0 = self.u0[self.train_idx[Nb*B:]] if self.use_knn else None
                    s0 = self.s0[self.train_idx[Nb*B:]] if self.use_knn else None
                    t0 = self.t0[self.train_idx[Nb*B:]] if self.use_knn else None
                    y_onehot = F.one_hot(torch.tensor(cell_labels[self.train_idx[Nb*B:]], dtype=int, device=self.device), self.n_type).float() if self.enable_cvae else None
                    p_t = self.p_t[:,self.train_idx[Nb*B:],:]
                    p_z = self.p_z[:,self.train_idx[Nb*B:],:]
                    lu_scale = self.lu_scale[self.train_idx[Nb*B:],:]
                    ls_scale = self.ls_scale[self.train_idx[Nb*B:],:]
                    mu_tx, std_tx, mu_zx, std_zx, uhat, shat, rho = self.eval_model(data_in, lu_scale, ls_scale, u0, s0, t0, y_onehot)
                else:
                    u0 = self.u0[Nb*B:] if self.use_knn else None
                    s0 = self.s0[Nb*B:] if self.use_knn else None
                    t0 = self.t0[Nb*B:] if self.use_knn else None
                    y_onehot = F.one_hot(torch.tensor(cell_labels[Nb*B:], dtype=int, device=self.device), self.n_type).float() if self.enable_cvae else None
                    p_t = self.p_t[:,Nb*B:,:]
                    p_z = self.p_z[:,Nb*B:,:]
                    lu_scale = self.lu_scale[Nb*B:,:]
                    ls_scale = self.ls_scale[Nb*B:,:]
                    mu_tx, std_tx, mu_zx, std_zx, uhat, shat, rho = self.eval_model(data_in, lu_scale, ls_scale, u0, s0, t0, y_onehot)
                u_raw, s_raw = data_in[:,:G], data_in[:,G:]
                u_sum, s_sum = torch.sum(uhat, 1, keepdim=True), torch.sum(shat, 1, keepdim=True)
                
                loss = self.vae_risk_poisson((mu_tx, std_tx), p_t,
                                             (mu_zx, std_zx), p_z,
                                             u_raw.int(), s_raw.int(), 
                                             uhat*lu_scale, shat*ls_scale,
                                             None)
                u_sample, s_sample = self.sample_poisson(uhat*lu_scale, shat*ls_scale)
                
                mse_batch = np.mean( (u_raw.cpu().numpy() - u_sample.cpu().numpy())**2 + ( s_raw.cpu().numpy() - s_sample.cpu().numpy())**2)
                elbo = elbo - ((N-B*Nb)/N)*loss
                mse = mse + ((N-B*Nb)/N)*mse_batch

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
            #out.append(corr)
            out.append(mse)
            
        return out, elbo.cpu().item()
    
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
        scaling = np.exp(self.decoder.scaling_u.detach().cpu().numpy())
        if(gene_indices is None):
            return alpha, beta, gamma, scaling
        return alpha[gene_indices],beta[gene_indices],gamma[gene_indices],scaling[gene_indices]
    
    def get_gene_vel(self, z, G, uhat, shat, gene_indices):
        rho = self.pred_rho(z, G)
        rho = rho[:,gene_indices]
        alpha, beta, gamma, scaling_u = self.get_mean_param(gene_indices)
        return rho*alpha - beta*uhat/scaling_u, beta*uhat/scaling_u - gamma*shat
        
        
    
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
            lu_scale = self.lu_scale.detach().cpu().numpy().squeeze()
            ls_scale = self.ls_scale.detach().cpu().numpy().squeeze()
            if(mode=="test"):
                lu_scale = lu_scale[self.test_idx]
                ls_scale = ls_scale[self.test_idx]
            else:
                lu_scale = lu_scale[self.train_idx]
                ls_scale = ls_scale[self.train_idx]
            
            z = out[6]
            vu ,vs = self.get_gene_vel(z, G, Uhat, Shat, gind)
            for i in range(len(gind)):
                idx = gind[i]
                
                plot_sig(t.squeeze(), 
                         dataset.data[:,idx], dataset.data[:,idx+G], 
                         Uhat[:,i], Shat[:,i], 
                         np.array([self.label_dic_rev[x] for x in dataset.labels]),
                         gene_plot[i], 
                         save=f"{path}/sig-{gene_plot[i]}-{testid}.png",
                         sparsify=self.config['sparsify'])
                d = max(1, len(t)//1000)
                scaling_u = np.exp(self.decoder.scaling_u[idx].detach().cpu().item())
                uhat_norm = Uhat[:,i] / scaling_u
                shat_norm = Shat[:,i]
                plot_vel(t[::d].squeeze(), uhat_norm[::d], shat_norm[::d], vu[::d,i], vs[::d,i], None, None, gene_plot[i], 
                             save=f"{path}/vel-{gene_plot[i]}-{testid}.png")
        return elbo
    
    def gene_likelihood_poisson(self, 
                                U,S,
                                Uhat, Shat, 
                                b=5000):
        N,G = U.shape
        Nb = N//b
        logpu, logps = np.empty((N,G)), np.empty((N,G))
        for i in range(Nb):
            ps_u = Poisson(torch.tensor(Uhat[i*b:(i+1)*b]))
            ps_s = Poisson(torch.tensor(Shat[i*b:(i+1)*b]))
            logpu[i*b:(i+1)*b] = ps_u.log_prob(torch.tensor(U[i*b:(i+1)*b], dtype=int))
            logps[i*b:(i+1)*b] = ps_s.log_prob(torch.tensor(S[i*b:(i+1)*b], dtype=int))
        if(Nb*b<N):
            ps_u = Poisson(torch.tensor(Uhat[Nb*b:]))
            ps_s = Poisson(torch.tensor(Shat[Nb*b:]))
            logpu[Nb*b:] = ps_u.log_prob(torch.tensor(U[Nb*b:], dtype=int))
            logps[Nb*b:] = ps_s.log_prob(torch.tensor(S[Nb*b:], dtype=int))
        return np.exp((logpu+logps)).mean(0)
    
    def save_anndata(self, adata, key, file_path, file_name=None):
        self.set_mode('eval')
        os.makedirs(file_path, exist_ok=True)
        
        adata.var[f"{key}_alpha"] = np.exp(self.decoder.alpha.detach().cpu().numpy())
        adata.var[f"{key}_beta"] = np.exp(self.decoder.beta.detach().cpu().numpy())
        adata.var[f"{key}_gamma"] = np.exp(self.decoder.gamma.detach().cpu().numpy())
        adata.var[f"{key}_ton"] = self.decoder.ton.exp().detach().cpu().numpy()
        adata.var[f"{key}_scaling_u"] = self.decoder.scaling_u.exp().detach().cpu().numpy()
        adata.var[f"{key}_scaling_s"] = self.decoder.scaling_s.exp().detach().cpu().numpy()
        
        U,S = np.array(adata.layers['unspliced'].todense()), np.array(adata.layers['spliced'].todense())
        X = np.concatenate((U,S),1)

        out, elbo = self.pred_all(X, self.cell_labels, "both", gene_idx=np.array(range(adata.n_vars)))
        Uhat, U_sum, Shat, S_sum, t, std_t, z, std_z = out
        
        adata.obs[f"{key}_time"] = t
        adata.obs[f"{key}_std_t"] = std_t
        adata.obsm[f"{key}_z"] = z
        adata.obsm[f"{key}_std_z"] = std_z
        
        adata.layers[f"{key}_uhat"] = Uhat
        adata.layers[f"{key}_shat"] = Shat
        
        
        
        #adata.var[f"{key}_likelihood"] = self.gene_likelihood_poisson(U,S,Uhat*lu_scale,Shat*ls_scale)
        
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
        
        adata.obs[f"{key}_t0"] = self.t0.squeeze().detach().cpu().numpy()
        adata.layers[f"{key}_u0"] = self.u0.detach().cpu().numpy()
        adata.layers[f"{key}_s0"] = self.s0.detach().cpu().numpy()

        adata.uns[f"{key}_train_idx"] = self.train_idx
        adata.uns[f"{key}_test_idx"] = self.test_idx
        
        scaling_u = adata.var[f"{key}_scaling_u"].to_numpy()
        scaling_s = adata.var[f"{key}_scaling_s"].to_numpy()
        #lu_scale = adata.obs["library_scale_u"].to_numpy().reshape(-1,1)
        #ls_scale = adata.obs["library_scale_s"].to_numpy().reshape(-1,1)
        adata.layers[f"{key}_velocity"] = adata.var[f"{key}_beta"].to_numpy() * Uhat / scaling_u - adata.var[f"{key}_gamma"].to_numpy() * Shat / scaling_s
        
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
                 init_ton_zero=False,
                 scale_cell=True,
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
            Uhat, Shat = pred_su(F.leaky_relu(t - self.ton.exp(), neg_slope), self.u0.exp()/self.scaling_u.exp(),  self.s0.exp(), rho*alpha, beta, gamma)
        else:
            if(condition is None):
                rho = torch.sigmoid(self.fc_out2(self.net_rho2(z)))
            else:
                rho = torch.sigmoid(self.fc_out2(self.net_rho2(torch.cat((z,condition),1))))
            Uhat, Shat = pred_su(F.leaky_relu(t-t0, neg_slope), u0/self.scaling_u.exp(), s0, rho*alpha, beta, gamma)
        return nn.functional.relu(Uhat), nn.functional.relu(Shat), rho
        
    def eval_model(self, t, z, condition=None, u0=None, s0=None, t0=None, neg_slope=0.0):
        #Evaluate the decoder. Here, we use the mean instead of randomly sample the ODE parameters.
        
        alpha = (self.alpha[0] + 0.5*(self.alpha[1]).exp().pow(2)).exp()
        beta = (self.beta[0] + 0.5*(self.beta[1]).exp().pow(2)).exp()
        gamma = (self.gamma[0] + 0.5*(self.gamma[1]).exp().pow(2)).exp()
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
            Uhat, Shat = pred_su(F.leaky_relu(t-t0, neg_slope), u0/self.scaling_u.exp(), s0, rho*alpha, beta, gamma)
        return nn.functional.relu(Uhat), nn.functional.relu(Shat), rho


class DVAEFullVB(DVAE):
    def __init__(self, 
                 adata, 
                 tmax, 
                 dim_z, 
                 dim_cond=0,
                 device='cpu', 
                 hidden_size=(500, 250, 250, 500), 
                 init_method="steady", 
                 init_key=None,
                 tprior=None, 
                 init_type=None,
                 init_ton_zero=True,
                 time_distribution="gaussian",
                 std_z_prior=0.01,
                 rate_prior={
                     'alpha':(0.0,0.5),
                     'beta':(0.0,1.0),
                     'gamma':(0.0,1.0)
                 },
                 scale_cell=True,
                 separate_us_scale=True,
                 add_noise=True,
                 checkpoints=[None, None]):
        
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
            

            #Training Parameters
            "n_epochs":1000, 
            "n_epochs_post":1000,
            "batch_size":128, 
            "learning_rate":2e-4, 
            "learning_rate_ode":5e-4, 
            "learning_rate_post":2e-4,
            "learning_rate_ode_post":2e-4, 
            "lambda":1e-3, 
            "lambda_rho":1e-3,
            "kl_t":1.0, 
            "kl_z":1.0, 
            "kl_param":1.0,
            "test_iter":None, 
            "save_epoch":100, 
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
            "reg_vel":False,
            
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
        self.get_prior(adata, time_distribution, tmax, tprior)
        
        self.p_z = torch.stack([torch.zeros(N,dim_z), torch.ones(N,dim_z)*self.config["std_z_prior"]]).float().to(self.device)
        
        self.use_knn = False
        self.u0 = None
        self.s0 = None
        self.t0 = None
        self.lu_scale = torch.tensor(adata.obs['library_scale_u'].to_numpy()).unsqueeze(-1).float().to(self.device)
        self.ls_scale = torch.tensor(adata.obs['library_scale_s'].to_numpy()).unsqueeze(-1).float().to(self.device)
        

        print(f"Mean library scale: {self.lu_scale.detach().cpu().numpy().mean()}, {self.ls_scale.detach().cpu().numpy().mean()}")

        #Class attributes for training
        self.loss_train, self.loss_test = [],[]
        self.counter = 0 #Count the number of iterations
        self.n_drop = 0 #Count the number of consecutive iterations with little decrease in loss
        
        self.p_log_alpha = torch.tensor([[rate_prior['alpha'][0]], [rate_prior['alpha'][1]]]).to(self.device)
        self.p_log_beta = torch.tensor([[rate_prior['beta'][0]], [rate_prior['beta'][1]]]).to(self.device)
        self.p_log_gamma = torch.tensor([[rate_prior['gamma'][0]], [rate_prior['gamma'][1]]]).to(self.device)

    def eval_model(self, data_in, lu_scale, ls_scale, u0=None, s0=None, t0=None, condition=None, continuous=True):
        data_in_scale = data_in
        if(self.config["scale_gene_encoder"]):
            data_in_scale = torch.cat((data_in_scale[:,:data_in_scale.shape[1]//2]/self.decoder.scaling_u.exp(), data_in_scale[:,data_in_scale.shape[1]//2:]),1)
        if(self.config["scale_cell_encoder"]):
            data_in_scale = torch.cat((data_in_scale[:,:data_in_scale.shape[1]//2]/lu_scale, data_in_scale[:,data_in_scale.shape[1]//2:]/ls_scale),1)
        if(self.config["log1p"]):
            data_in_scale = torch.log1p(data_in_scale)
        mu_t, std_t, mu_z, std_z = self.encoder.forward(data_in_scale, condition)
        uhat, shat, rho = self.decoder.eval_model(mu_t, mu_z, condition, u0=u0, s0=s0, t0=t0, neg_slope=0.0)
        uhat = uhat * self.decoder.scaling_u.exp()

        if(not continuous):
            poisson_u = Poisson(F.softplus(uhat*lu_scale, beta=100))
            poisson_s = Poisson(F.softplus(shat*ls_scale, beta=100))
            u_out = poisson_u.sample()
            s_out = poisson_s.sample()
            return mu_t, std_t, mu_z, std_z, u_out, s_out
        return mu_t, std_t, mu_z, std_z, uhat, shat, rho
    
    def vae_risk_poisson(self, 
                         q_tx, p_t, 
                         q_zx, p_z, 
                         u, s, uhat, shat,
                         weight=None,
                         eps=1e-6):
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
        
        poisson_u = Poisson(F.relu(uhat)+1e-2)
        poisson_s = Poisson(F.relu(shat)+1e-2)
        
        
        logp = poisson_u.log_prob(u) + poisson_s.log_prob(s)
        
        if( weight is not None):
            logp = logp*weight
        
        err_rec = torch.mean(torch.sum(logp,1))
        
        return (- err_rec + self.config["kl_t"]*kldt + self.config["kl_z"]*kldz + self.config["kl_param"]*kld_param)
    
    def get_mean_param(self, gene_indices=None):
        mu_logalpha = self.decoder.alpha[0].detach().cpu().numpy()
        std_logalpha = np.exp(self.decoder.alpha[1].detach().cpu().numpy())
        alpha = np.exp(mu_logalpha+0.5*std_logalpha**2)
        
        mu_logbeta = self.decoder.beta[0].detach().cpu().numpy()
        std_logbeta = np.exp(self.decoder.beta[1].detach().cpu().numpy())
        beta = np.exp(mu_logbeta+0.5*std_logbeta**2)
        
        mu_loggamma = self.decoder.gamma[0].detach().cpu().numpy()
        std_loggamma = np.exp(self.decoder.gamma[1].detach().cpu().numpy())
        gamma = np.exp(mu_loggamma+0.5*std_loggamma**2)
        
        scaling = np.exp(self.decoder.scaling_u.detach().cpu().numpy())
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
        adata.var[f"{key}_scaling"] = self.decoder.scaling_u.exp().detach().cpu().numpy()
        
        U,S = np.array(adata.layers['unspliced'].todense()), np.array(adata.layers['spliced'].todense())
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
        
        adata.obs[f"{key}_t0"] = self.t0.squeeze().detach().cpu().numpy()
        adata.layers[f"{key}_u0"] = self.u0.detach().cpu().numpy()
        adata.layers[f"{key}_s0"] = self.s0.detach().cpu().numpy()

        adata.uns[f"{key}_train_idx"] = self.train_idx
        adata.uns[f"{key}_test_idx"] = self.test_idx
        
        beta_mean = np.exp(adata.var[f"{key}_logmu_beta"].to_numpy() + 0.5*adata.var[f"{key}_logstd_beta"].to_numpy()**2)
        gamma_mean = np.exp(adata.var[f"{key}_logmu_gamma"].to_numpy() + 0.5*adata.var[f"{key}_logstd_gamma"].to_numpy()**2)
        adata.layers[f"{key}_velocity"] = beta_mean * Uhat / adata.var[f"{key}_scaling"].to_numpy() - gamma_mean * Shat 
        
        if(file_name is not None):
            adata.write_h5ad(f"{file_path}/{file_name}")














##############################################################
# Neural ODE Velocity Function
##############################################################
class VelocityFunc(nn.Module):
    def __init__(self, 
                 adata, 
                 tmax,
                 train_idx,
                 dim_latent, 
                 hidden_size_rho,
                 hidden_size_vz,
                 p=98, 
                 dim_cond=0,
                 device=torch.device('cpu'), 
                 init_method="steady", 
                 init_key=None, 
                 checkpoint=None):
        super(VelocityFunc, self).__init__()
        self.G = adata.n_vars
        self.d = dim_latent
        
        N1, N2 = hidden_size_rho[0], hidden_size_rho[1]
        self.fc1 = nn.Linear(dim_latent+dim_cond, N1).to(device)
        #self.bn1 = nn.BatchNorm1d(num_features=N1).to(device)
        #self.dpt1 = nn.Dropout(p=0.2).to(device)
        self.fc2 = nn.Linear(N1, N2).to(device)
        #self.bn2 = nn.BatchNorm1d(num_features=N2).to(device)
        #self.dpt2 = nn.Dropout(p=0.2).to(device)
        self.fc_out1 = nn.Linear(N2, self.G).to(device)
        self.net_rho = nn.Sequential(self.fc1, nn.LeakyReLU(),
                                     self.fc2, nn.LeakyReLU(),)
        """
        self.fc3 = nn.Linear(dim_latent+dim_cond, N1).to(device)
        #self.bn3 = nn.BatchNorm1d(num_features=N1).to(device)
        #self.dpt3 = nn.Dropout(p=0.2).to(device)
        self.fc4 = nn.Linear(N1, N2).to(device)
        #self.bn4 = nn.BatchNorm1d(num_features=N2).to(device)
        #self.dpt4 = nn.Dropout(p=0.2).to(device)
        self.fc_out2 = nn.Linear(N2, self.G).to(device)
        self.net_rho2 = nn.Sequential(self.fc3, nn.LeakyReLU(), self.fc4, nn.LeakyReLU())
        """
        N3 = hidden_size_vz[0]
        self.fc5 = nn.Linear(dim_latent, N3).to(device)
        self.fc6 = nn.Linear(N3, dim_latent).to(device)
        self.net_vz = nn.Sequential(self.fc5, nn.ReLU(), self.fc6, nn.ReLU())
        
        self.init_weights()
        
        U, S = np.array(adata.layers['unspliced'].todense()), np.array(adata.layers['spliced'].todense())
        #Get library size factors
        lu = U.sum(1)  
        ls = S.sum(1)
        med_lu, med_ls = np.median(lu[train_idx]), np.median(ls[train_idx])
        lu = lu/med_lu
        adata.obs["library_scale_u"] = lu
        ls = ls/med_ls
        adata.obs["library_scale_s"] = ls
        
        U = U[train_idx]
        S = S[train_idx]
        X = np.concatenate((U,S),1) + np.exp(np.random.normal(size=(len(train_idx), 2*self.G))*1e-3)
        if(init_method == "random"):
            print("Random Initialization.")
            alpha, beta, gamma, scaling, toff, u0, s0, sigma_u, sigma_s, T, Rscore = init_params(X,p,fit_scaling=True)
            self.alpha = nn.Parameter(torch.normal(0.0, 0.01, size=(U.shape[1],), device=device).float())
            self.beta =  nn.Parameter(torch.normal(0.0, 0.01, size=(U.shape[1],), device=device).float())
            self.gamma = nn.Parameter(torch.normal(0.0, 0.01, size=(U.shape[1],), device=device).float())
            self.scaling = nn.Parameter(torch.tensor(np.log(scaling), device=device).float())
        elif(init_method == "tprior"):
            print("Initialization using prior time.")
            alpha, beta, gamma, scaling, toff, u0, s0, sigma_u, sigma_s, T, Rscore = init_params(X,p,fit_scaling=True)
            t_prior = adata.obs[init_key].to_numpy()
            t_prior = t_prior[train_idx]
            std_t = (np.std(t_prior)+1e-3)*0.2
            self.t_init = np.random.uniform(t_prior-std_t, t_prior+std_t)
            self.t_init -= self.t_init.min()
            self.t_init = self.t_init
            self.t_init = self.t_init/self.t_init.max()*tmax
            toff = get_ts_global(self.t_init, U/scaling, S, 95)
            alpha, beta, gamma, ton = reinit_params(U/scaling, S, self.t_init, toff)
            
            self.alpha = nn.Parameter(torch.tensor(np.log(alpha), device=device).float())
            self.beta = nn.Parameter(torch.tensor(np.log(beta), device=device).float())
            self.gamma = nn.Parameter(torch.tensor(np.log(gamma), device=device).float())
            self.scaling = nn.Parameter(torch.tensor(np.log(scaling), device=device).float())
        else:
            print("Initialization using the steady-state and dynamical models.")
            alpha, beta, gamma, scaling, toff, u0, s0, sigma_u, sigma_s, T, Rscore = init_params(X,p,fit_scaling=True)
            if(init_key is not None):
                self.t_init = adata.obs[init_key].to_numpy()[train_idx]
            else:
                T = T+np.random.rand(T.shape[0],T.shape[1]) * 1e-3
                T_eq = np.zeros(T.shape)
                Nbin = T.shape[0]//50+1
                for i in range(T.shape[1]):
                    T_eq[:, i] = hist_equal(T[:, i], tmax, 0.9, Nbin)
                self.t_init = np.quantile(T_eq,0.5,1)
            toff = get_ts_global(self.t_init, U/scaling, S, 95)
            alpha, beta, gamma, ton = reinit_params(U/scaling, S, self.t_init, toff)
            
            self.alpha = nn.Parameter(torch.tensor(np.log(alpha), device=device).float())
            self.beta = nn.Parameter(torch.tensor(np.log(beta), device=device).float())
            self.gamma = nn.Parameter(torch.tensor(np.log(gamma), device=device).float())
            self.scaling = nn.Parameter(torch.tensor(np.log(scaling), device=device).float())
        self.ton = nn.Parameter(torch.ones(self.G, device=device).float()*(-10))
        self.u0 = nn.Parameter(torch.ones(self.G, device=device).float()*(-10))
        self.s0 = nn.Parameter(torch.ones(self.G, device=device).float()*(-10))
        self.VMAX = 100
        self.VMIN = -100
    
    def init_weights(self):
        for m in self.net_rho.modules():
            if(isinstance(m, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
            elif(isinstance(m, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        """
        for m in self.net_rho2.modules():
            if(isinstance(m, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
            elif(isinstance(m, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        """
        for m in self.net_vz.modules():
            if(isinstance(m, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
            elif(isinstance(m, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in [self.fc_out1]:
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.0)
        
    def forward(self, t, x):
        #Determine dimension
        if(x.ndim==1):
            vz = self.net_vz(x[2*self.G:])
            rho = F.sigmoid(self.fc_out1(self.net_rho(x[2*self.G:])))
            vu = self.alpha.exp()*rho - self.beta.exp()*x[:self.G]
            vs = self.beta.exp()*x[:self.G] - self.gamma.exp()*x[self.G:2*self.G]
            #print(vu.max(), vu.min())
            return torch.clip(torch.concat((vu,vs,vz)), min=self.VMIN, max=self.VMAX)
        else:
            vz = self.net_vz(x[:,2*self.G:])
            rho = F.sigmoid(self.fc_out1(self.net_rho(x[:,2*self.G:])))
            vu = self.alpha.exp()*rho - self.beta.exp()*x[:,:self.G]
            vs = self.beta.exp()*x[:,:self.G] - self.gamma.exp()*x[:,self.G:2*self.G]
            return torch.clip(torch.concat((vu,vs,vz),1), min=self.VMIN, max=self.VMAX)
    
    
    
    def decode(self, t, z, condition=None, neg_slope=0.0):
        if(condition is None):
            rho = torch.sigmoid(self.fc_out1(self.net_rho(z)))
        else:
            rho = torch.sigmoid(self.fc_out1(self.net_rho(torch.cat((z,condition),1))))
        alpha = self.alpha.exp()*rho
        Uhat, Shat = pred_su(F.leaky_relu(t - self.ton.exp(), neg_slope), self.u0.exp()/self.scaling.exp(),  self.s0.exp(), alpha, self.beta.exp(), self.gamma.exp())
        
        return nn.functional.relu(Uhat), nn.functional.relu(Shat), rho
    
    def pred_rho(self, z, condition=None, neural_ode=False):
        if(neural_ode):
            if(condition is None):
                rho = torch.sigmoid(self.fc_out1(self.net_rho(z)))
            else:
                rho = torch.sigmoid(self.fc_out1(self.net_rho(torch.cat((z,condition),1))))
        else:
            if(condition is None):
                rho = torch.sigmoid(self.fc_out1(self.net_rho(z)))
            else:
                rho = torch.sigmoid(self.fc_out1(self.net_rho(torch.cat((z,condition),1))))
        return rho

class ODEDVAE(VAE):
    """
    Discrete VeloVAE Model
    """
    def __init__(self, 
                 adata, 
                 tmax, 
                 dim_z, 
                 dim_cond=0,
                 device='cpu', 
                 hidden_size_encoder=(500, 250),
                 hidden_size_rho=(250,500),
                 hidden_size_vz=(25,),
                 init_method="steady", 
                 init_key=None,
                 tprior=None, 
                 time_distribution="gaussian",
                 std_z_prior=0.01,
                 checkpoints=[None, None]):
        
        self.config = {
            #Model Parameters
            "neural_ode":False,
            "dim_z":dim_z,
            "hidden_size_encoder":hidden_size_encoder,
            "hidden_size_rho":hidden_size_rho,
            "hidden_size_vz":hidden_size_vz,
            "tmax":tmax,
            "init_method":init_method,
            "ode_method":"euler",
            "init_key":init_key,
            "tprior":tprior,
            "std_z_prior":std_z_prior,
            "tail":0.01,
            "time_overlap":0.5,
            "n_neighbors":10,
            "dt": (0.03,0.06),

            #Training Parameters
            "n_epochs":1000, 
            "n_epochs_post":1000,
            "batch_size":128,
            "learning_rate":2e-4, 
            "learning_rate_ode":5e-4, 
            "learning_rate_post":2e-4,
            "lambda":1e-3, 
            "lambda_rho":1e-3,
            "kl_t":1.0, 
            "kl_z":1.0, 
            "kl_l":1.0,
            "test_iter":None, 
            "save_epoch":100, 
            "n_warmup":5,
            "early_stop":5,
            "early_stop_thred":adata.n_vars*1e-3,
            "train_test_split":0.7,
            "neg_slope":0.0,
            "k_alt":1, 
            "train_scaling":False, 
            "train_std":False, 
            "weight_sample":False,
            
            #Plotting
            "sparsify":1
        }
        
        self.set_device(device)
        self.split_train_test(adata.n_obs)
        
        N, G = adata.n_obs, adata.n_vars
        self.dim_z = dim_z
        self.enable_cvae = dim_cond>0
        try:
            self.encoder = encoder_velovae(2*G, dim_z, dim_cond, hidden_size_encoder[0], hidden_size_encoder[1], self.device, checkpoint=checkpoints[0]).float()
        except IndexError:
            print('Please provide two dimensions!')
        
        self.decoder = VelocityFunc(adata, 
                                    tmax,
                                    self.train_idx, 
                                    dim_z, 
                                    hidden_size_rho, 
                                    hidden_size_vz, 
                                    device=self.device, 
                                    init_method=init_method, 
                                    init_key=init_key, 
                                    checkpoint=checkpoints[1])
        
        self.tmax=tmax
        self.time_distribution = time_distribution
        self.get_prior(adata, time_distribution, tmax, tprior)
        
        self.p_z = torch.stack([torch.zeros(N,dim_z), torch.ones(N,dim_z)*self.config["std_z_prior"]]).float().to(self.device)

        self.lu_scale = torch.tensor(adata.obs['library_scale_u'].to_numpy()).unsqueeze(-1).float().to(self.device)
        self.ls_scale = torch.tensor(adata.obs['library_scale_s'].to_numpy()).unsqueeze(-1).float().to(self.device)

        print(f"Mean library scale: {self.lu_scale.detach().cpu().numpy().mean()}, {self.ls_scale.detach().cpu().numpy().mean()}")

        #Class attributes for training
        self.loss_train, self.loss_test = [],[]
        self.counter = 0 #Count the number of iterations
        self.n_drop = 0 #Count the number of consecutive iterations with little decrease in loss
    
    def set_mode(self,mode):
        #Set the model to either training or evaluation mode.
        
        if(mode=='train'):
            self.encoder.train()
            self.decoder.train()
        elif(mode=='eval'):
            self.encoder.eval()
            self.decoder.eval()
        else:
            print("Warning: mode not recognized. Must be 'train' or 'test'! ")
    
    def forward(self, data_in, lu_scale, ls_scale, condition=None, eps=1e-5):
        G = self.decoder.G
        data_in_scale = torch.cat((data_in[:,:G]/torch.exp(self.decoder.scaling), data_in[:,G:]),1)
        data_in_scale = torch.log1p(data_in_scale)# + torch.normal(mean=0.0, std=torch.ones(data_in_scale.shape) * 1e-3).to(self.device)
        mu_t, std_t, mu_z, std_z = self.encoder.forward(data_in_scale, condition)
        t = self.sample(mu_t, std_t)
        z = self.sample(mu_z, std_z)
        
        idx_sort = torch.argsort(t.squeeze())
        t = t[idx_sort].squeeze()
        t = t+torch.cumsum(torch.ones(t.shape[0], device=self.device)*eps,0)
        z = z[idx_sort]

        #t needs to be strictly increasing
        tmask = torch.concat((torch.tensor([1], dtype=bool, device=self.device), t[1:]>t[:-1]))
        t = t[tmask]
        z = z[tmask]
        
        #init_state = torch.concat((data_in[idx_sort[0],:G]/torch.exp(self.decoder.scaling), data_in[idx_sort[0],G:], z[0]))
        init_state = torch.zeros(2*G+z.shape[1],device=self.device)
        t0 = torch.tensor([-eps],device=self.device)
        t = torch.concat((t0,t))
        out = odeint_adjoint(self.decoder, init_state, t, method=self.config["ode_method"]) #T x (2G+dim_latent)
        
        return mu_t, std_t, mu_z, std_z, t, z, out[1:,:G], out[1:,G:2*G], out[1:,2*G:], idx_sort
    
    def eval_model(self, data_in, lu_scale, ls_scale, condition=None, eps=1e-5):
        """
        < Description >
        Evaluate the model on the validation dataset. 
        The major difference from forward pass is that we use the mean time,
        cell state and library size instead of random sampling.
        
        < Input Arguments >
        Same as forward

        < Output >
        Same as forward
        """
        G = self.decoder.G
        data_in_scale = torch.cat((data_in[:,:G]/torch.exp(self.decoder.scaling), data_in[:,G:]),1)
        data_in_scale = torch.log1p(data_in_scale)# + torch.normal(mean=0.0, std=torch.ones(data_in_scale.shape) * 1e-3).to(self.device)
        mu_t, std_t, mu_z, std_z = self.encoder.forward(data_in_scale, condition)

        idx_sort = torch.argsort(mu_t.squeeze())
        t = mu_t[idx_sort].squeeze()
        t = t+torch.cumsum(torch.ones(t.shape[0], device=self.device)*eps,0)#t needs to be strictly increasing
        z = mu_z[idx_sort]
        
        #init_state = torch.concat((data_in[idx_sort[0],:G]/torch.exp(self.decoder.scaling), data_in[idx_sort[0],G:], z[0]))
        init_state = torch.zeros(2*G+z.shape[1],device=self.device)
        t0 = torch.tensor([-eps],device=self.device)
        t = torch.concat((t0,t))
        out = odeint_adjoint(self.decoder, init_state, t,  method=self.config["ode_method"]) #T x (2G+dim_latent)

        return mu_t, std_t, mu_z, std_z, out[1:,:G], out[1:,G:2*G], out[1:,2*G:], idx_sort
    
    def forward_vae(self, data_in, lu_scale, ls_scale, condition=None):
        G = self.decoder.G
        data_in_scale = torch.cat((data_in[:,:G]/torch.exp(self.decoder.scaling), data_in[:,G:]),1)
        data_in_scale = torch.log1p(data_in_scale)# + torch.normal(mean=0.0, std=torch.ones(data_in_scale.shape) * 1e-3).to(self.device)
        mu_t, std_t, mu_z, std_z = self.encoder.forward(data_in_scale, condition)
        t = self.sample(mu_t, std_t)
        z = self.sample(mu_z, std_z)

        uhat, shat, rho = self.decoder.decode(t, z, condition, neg_slope=self.config["neg_slope"])
        uhat = uhat * self.decoder.scaling.exp()
        
        return mu_t, std_t, mu_z, std_z, t, z, uhat, shat, rho
    
    def eval_model_vae(self, data_in, lu_scale, ls_scale, condition=None):
        """
        < Description >
        Evaluate the model on the validation dataset. 
        The major difference from forward pass is that we use the mean time,
        cell state and library size instead of random sampling.
        
        < Input Arguments >
        Same as forward

        < Output >
        Same as forward
        """
        G = self.decoder.G
        data_in_scale = torch.cat((data_in[:,:G]/torch.exp(self.decoder.scaling), data_in[:,G:]),1)
        data_in_scale = torch.log1p(data_in_scale)# + torch.normal(mean=0.0, std=torch.ones(data_in_scale.shape) * 1e-3).to(self.device)
        mu_t, std_t, mu_z, std_z = self.encoder.forward(data_in_scale, condition)
        uhat, shat, rho = self.decoder.decode(mu_t, mu_z, condition, neg_slope=0.0)
        uhat = uhat * self.decoder.scaling.exp()

        return mu_t, std_t, mu_z, std_z, uhat, shat, rho
    
    
    def sample_poisson(self, uhat, shat):
        u_sample = torch.poisson(uhat)
        s_sample = torch.poisson(shat)
        return u_sample.cpu(), s_sample.cpu()
    
    def corr_vel(self, uhat, shat, rho):
        cos_sim = nn.CosineSimilarity(dim=0)
        vu = rho * self.decoder.alpha.exp() - self.decoder.beta.exp() * uhat / self.decoder.scaling.exp()
        vs = self.decoder.beta.exp() * uhat / self.decoder.scaling.exp() - self.decoder.gamma.exp() * shat
        vu = vu-vu.mean(0)
        vs = vs-vs.mean(0)
        uhat_0 = uhat - uhat.mean(0)
        shat_0 = shat - shat.mean(0)
        
        return torch.sum(cos_sim(vu,uhat_0)+cos_sim(vs,shat_0)-cos_sim(vs,uhat))
    
    def vae_risk_poisson(self, 
                         q_tx, p_t, 
                         q_zx, p_z, 
                         u, s, uhat, shat,
                         weight=None,
                         eps=1e-6):
        kldt = self.kl_time(q_tx[0], q_tx[1], p_t[0], p_t[1], tail=self.config["tail"])
        kldz = kl_gaussian(q_zx[0], q_zx[1], p_z[0], p_z[1])
        
        #poisson
        #try:
        poisson_u = Poisson(F.relu(uhat)+1e-2)
        poisson_s = Poisson(F.relu(shat)+1e-2)
        
        
        logp = poisson_u.log_prob(u) + poisson_s.log_prob(s)
        
        if( weight is not None):
            logp = logp*weight
        
        err_rec = torch.mean(torch.sum(logp,1))
        if(rho is not None):
            l_corr = self.corr_vel(uhat, shat, rho)
        else:
            l_corr = 0
        return (- err_rec + self.config["kl_t"]*kldt + self.config["kl_z"]*kldz + l_corr)
    
    def train_epoch(self, train_loader, test_set, optimizer, optimizer2=None, K=1, reg_vel=False):
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
            
            xbatch, idx = batch[0].to(self.device), batch[3]
            u = xbatch[:,:xbatch.shape[1]//2]
            s = xbatch[:,xbatch.shape[1]//2:]
            
            lu_scale = self.lu_scale[self.train_idx[idx]]
            ls_scale = self.ls_scale[self.train_idx[idx]]
            
            condition = F.one_hot(batch[1].to(self.device), self.n_type).float() if self.enable_cvae else None
            if(self.config["neural_ode"]):
                mu_tx, std_tx, mu_zx, std_zx, t, z, uhat, shat, z_pred, idx_sort = self.forward(xbatch.float(), lu_scale, ls_scale, condition)
                u = u[idx_sort]
                s = s[idx_sort]
                lu_scale = lu_scale[idx_sort]
                ls_scale = ls_scale[idx_sort]
                loss_z = torch.mean((torch.sum(z-z_pred)/self.config["std_z_prior"]).pow(2))
            else:
                mu_tx, std_tx, mu_zx, std_zx, t, z, uhat, shat, rho  = self.forward_vae(xbatch.float(), lu_scale, ls_scale, condition)
                loss_z = 0
            if(not reg_vel):
                rho = None
            loss = self.vae_risk_poisson((mu_tx, std_tx), self.p_t[:,self.train_idx[idx],:],
                                          (mu_zx, std_zx), self.p_z[:,self.train_idx[idx],:],
                                          u.int(),
                                          s.int(), 
                                          uhat*lu_scale, shat*ls_scale,
                                          None)
            loss = loss + loss_z
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
              U_raw,
              S_raw,
              config={}, 
              plot=False, 
              gene_plot=[], 
              cluster_key = "clusters",
              figure_path = "figures", 
              embed="umap"):
        self.train_mse = []
        self.test_mse = []
        
        self.load_config(config)
        
        print("--------------------------- Train a VeloVAE ---------------------------")
        #Get data loader
        U, S = np.array(adata.layers['unspliced'].todense()), np.array(adata.layers['spliced'].todense())
        X = np.concatenate((U,S), 1)
        
        try:
            Xembed = adata.obsm[f"X_{embed}"]
        except KeyError:
            print("Embedding not found! Set to None.")
            Xembed = None
            plot = False
        
        cell_labels_raw = adata.obs[cluster_key].to_numpy() if cluster_key in adata.obs else np.array(['Unknown' for i in range(adata.n_obs)])
        #Encode the labels
        self.cell_types_raw = np.unique(cell_labels_raw)
        self.label_dic, self.label_dic_rev = encode_type(self.cell_types_raw)

        self.n_type = len(self.cell_types_raw)
        self.cell_labels = np.array([self.label_dic[x] for x in cell_labels_raw])
        self.cell_types = np.array([self.label_dic[self.cell_types_raw[i]] for i in range(self.n_type)])
        
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
        
        start = time.time()
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
                                       True, 
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
        
        self.config["neural_ode"] = True
        self.n_drop = 0
        param_ode = [self.decoder.alpha, self.decoder.beta, self.decoder.gamma] 
        optimizer_ode = torch.optim.Adam(param_ode, lr=self.config["learning_rate_ode"])
        param_post = list(self.decoder.net_rho.parameters())+list(self.decoder.fc_out1.parameters())+list(self.decoder.net_vz.parameters())
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
                """
                if(epoch>=self.config["n_warmup"]):
                    stop_training = self.train_epoch(data_loader, test_set, optimizer_post, optimizer_ode, self.config["k_alt"])
                else:
                    stop_training = self.train_epoch(data_loader, test_set, optimizer_post, None, self.config["k_alt"])
                """
                stop_training = self.train_epoch(data_loader, test_set, optimizer_post, None, self.config["k_alt"])
            
            if(plot and (epoch==0 or (epoch+n_stage1+1) % self.config["save_epoch"] == 0)):
                elbo_train = self.test(train_set,
                                       Xembed[self.train_idx],
                                       f"train{epoch+n_stage1+1}", 
                                       False,
                                       gind, 
                                       gene_plot,
                                       True, 
                                       figure_path)
                self.decoder.train()
                elbo_test = self.loss_test[-1] if len(self.loss_test)>n_test1 else -np.inf
                print(f"Epoch {epoch+n_stage1+1}: Train ELBO = {elbo_train:.3f}, Test ELBO = {elbo_test:.3f}, \t Total Time = {convert_time(time.time()-start)}")
            
            if(stop_training):
                print(f"*********       Stage 2: Early Stop Triggered at epoch {epoch+n_stage1+1}.       *********")
                break
        
        print(f"*********              Finished. Total Time = {convert_time(time.time()-start)}             *********")
        #Plot final results
        if(plot):
            elbo_train = self.test(train_set,
                                   Xembed[self.train_idx],
                                   "final-train", 
                                   False,
                                   gind, 
                                   gene_plot,
                                   True, 
                                   figure_path)
            elbo_test = self.test(test_set,
                                  Xembed[self.test_idx],
                                  "final-test", 
                                  True,
                                  gind, 
                                  gene_plot,
                                  True, 
                                  figure_path)
            print(f"Final: Train ELBO = {elbo_train:.3f}, Test ELBO = {elbo_test:.3f}")
            plot_train_loss(self.loss_train, range(1,len(self.loss_train)+1), save=f'{figure_path}/train_loss_velovae.png')
            if(self.config["test_iter"]>0):
                plot_test_loss(self.loss_test, [i*self.config["test_iter"] for i in range(1,len(self.loss_test)+1)], save=f'{figure_path}/test_loss_velovae.png')
        
        if(plot):
            plot_train_loss(self.train_mse, range(1,len(self.train_mse)+1), save=f'{figure_path}/train_mse_velovae.png')
            if(self.config["test_iter"]>0):
                plot_test_loss(self.test_mse, [i*self.config["test_iter"] for i in range(1,len(self.test_mse)+1)], save=f'{figure_path}/test_mse_velovae.png')
    
    def pred_all(self, data, cell_labels, mode='test', output=["uhat", "shat", "t", "z", "mse"], gene_idx=None):
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
        
        with torch.no_grad():
            B = 1000
            Nb = N // B
            for i in range(Nb):
                data_in = torch.tensor(data[i*B:(i+1)*B]).float().to(self.device)
                if(mode=="test"):
                    y_onehot = F.one_hot(torch.tensor(cell_labels[self.test_idx[i*B:(i+1)*B]], dtype=int, device=self.device), self.n_type).float() if self.enable_cvae else None
                    p_t = self.p_t[:,self.test_idx[i*B:(i+1)*B],:]
                    p_z = self.p_z[:,self.test_idx[i*B:(i+1)*B],:]
                    lu_scale = self.lu_scale[self.test_idx[i*B:(i+1)*B],:]
                    ls_scale = self.ls_scale[self.test_idx[i*B:(i+1)*B],:]
                    
                elif(mode=="train"):
                    y_onehot = F.one_hot(torch.tensor(cell_labels[self.train_idx[i*B:(i+1)*B]], dtype=int, device=self.device), self.n_type).float() if self.enable_cvae else None
                    p_t = self.p_t[:,self.train_idx[i*B:(i+1)*B],:]
                    p_z = self.p_z[:,self.train_idx[i*B:(i+1)*B],:]
                    lu_scale = self.lu_scale[self.train_idx[i*B:(i+1)*B],:]
                    ls_scale = self.ls_scale[self.train_idx[i*B:(i+1)*B],:]
                else:
                    y_onehot = F.one_hot(torch.tensor(cell_labels[i*B:(i+1)*B], dtype=int, device=self.device), self.n_type).float() if self.enable_cvae else None
                    p_t = self.p_t[:,i*B:(i+1)*B,:]
                    p_z = self.p_z[:,i*B:(i+1)*B,:]
                    lu_scale = self.lu_scale[i*B:(i+1)*B,:]
                    ls_scale = self.ls_scale[i*B:(i+1)*B,:]
                
                u_raw, s_raw = data_in[:,:G], data_in[:,G:]
                if(self.config["neural_ode"]):
                    mu_tx, std_tx, mu_zx, std_zx, uhat, shat, z_pred, idx_sort = self.eval_model(data_in, lu_scale, ls_scale, y_onehot)
                    u_raw = u_raw[idx_sort]
                    s_raw = s_raw[idx_sort]
                    lu_scale = lu_scale[idx_sort]
                    ls_scale = ls_scale[idx_sort]
                else:
                    mu_tx, std_tx, mu_zx, std_zx, uhat, shat, rho = self.eval_model_vae(data_in, lu_scale, ls_scale, y_onehot)
                
                u_sum, s_sum = torch.sum(uhat*lu_scale, 1, keepdim=True), torch.sum(shat*ls_scale, 1, keepdim=True)
                
                loss = self.vae_risk_poisson((mu_tx, std_tx), p_t,
                                             (mu_zx, std_zx), p_z,
                                             u_raw.int(), s_raw.int(), 
                                             uhat*lu_scale, shat*ls_scale, 
                                             None)
                u_sample, s_sample = self.sample_poisson(uhat*lu_scale, shat*ls_scale)
                
                mse_batch = np.mean( (u_raw.cpu().numpy() - u_sample.cpu().numpy())**2 + ( s_raw.cpu().numpy() - s_sample.cpu().numpy())**2)
                elbo = elbo - (B/N)*loss
                mse = mse + (B/N) * mse_batch

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
                    y_onehot = F.one_hot(torch.tensor(cell_labels[self.test_idx[Nb*B:]], dtype=int, device=self.device), self.n_type).float() if self.enable_cvae else None
                    p_t = self.p_t[:,self.test_idx[Nb*B:],:]
                    p_z = self.p_z[:,self.test_idx[Nb*B:],:]
                    lu_scale = self.lu_scale[self.test_idx[Nb*B:],:]
                    ls_scale = self.ls_scale[self.test_idx[Nb*B:],:]
                elif(mode=="train"):
                    y_onehot = F.one_hot(torch.tensor(cell_labels[self.train_idx[Nb*B:]], dtype=int, device=self.device), self.n_type).float() if self.enable_cvae else None
                    p_t = self.p_t[:,self.train_idx[Nb*B:],:]
                    p_z = self.p_z[:,self.train_idx[Nb*B:],:]
                    lu_scale = self.lu_scale[self.train_idx[Nb*B:],:]
                    ls_scale = self.ls_scale[self.train_idx[Nb*B:],:]
                else:
                    y_onehot = F.one_hot(torch.tensor(cell_labels[Nb*B:], dtype=int, device=self.device), self.n_type).float() if self.enable_cvae else None
                    p_t = self.p_t[:,Nb*B:,:]
                    p_z = self.p_z[:,Nb*B:,:]
                    lu_scale = self.lu_scale[Nb*B:,:]
                    ls_scale = self.ls_scale[Nb*B:,:]
                u_raw, s_raw = data_in[:,:G], data_in[:,G:]
                if(self.config['neural_ode']):
                    mu_tx, std_tx, mu_zx, std_zx, uhat, shat, z_pred, idx_sort = self.eval_model(data_in, lu_scale, ls_scale, y_onehot)
                    u_raw = u_raw[idx_sort]
                    s_raw = s_raw[idx_sort]
                    lu_scale = lu_scale[idx_sort]
                    ls_scale = ls_scale[idx_sort]
                else:
                    mu_tx, std_tx, mu_zx, std_zx, uhat, shat, rho = self.eval_model_vae(data_in, lu_scale, ls_scale, y_onehot)
                
                u_sum, s_sum = torch.sum(uhat*lu_scale, 1, keepdim=True), torch.sum(shat*ls_scale, 1, keepdim=True)
                
                loss = self.vae_risk_poisson((mu_tx, std_tx), p_t,
                                             (mu_zx, std_zx), p_z,
                                             u_raw.int(), s_raw.int(), 
                                             uhat*lu_scale, shat*ls_scale,
                                             None)
                u_sample, s_sample = self.sample_poisson(uhat*lu_scale, shat*ls_scale)
                
                mse_batch = np.mean( (u_raw.cpu().numpy() - u_sample.cpu().numpy())**2 + ( s_raw.cpu().numpy() - s_sample.cpu().numpy())**2)
                elbo = elbo - ((N-B*Nb)/N)*loss
                mse = mse + ((N-B*Nb)/N)*mse_batch

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
            
        return out, elbo.cpu().item()
    
    def pred_rho(self, z, G):
        N = z.shape[0]
        rho = np.zeros((N, G))
        with torch.no_grad():
            B = min(N//5, 1000)
            Nb = N // B
            for i in range(Nb):
                rho_batch = self.decoder.pred_rho(torch.tensor(z[i*B:(i+1)*B]).float().to(self.device), neural_ode=self.config["neural_ode"])
                rho[i*B:(i+1)*B] = rho_batch.cpu().numpy()
            if(N > Nb*B):
                rho_batch = self.decoder.pred_rho(torch.tensor(z[Nb*B:]).float().to(self.device), neural_ode=self.config["neural_ode"])
                rho[Nb*B:] = rho_batch.cpu().numpy()
        return rho
    
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
        out, elbo = self.pred_all(dataset.data, self.cell_labels, mode, ["uhat", "shat", "z", "t", "mse"], gind)
        Uhat, U_sum, Shat, S_sum, t, std_t = out[0], out[1], out[2], out[3], out[4], out[5]
        
        
        if(test_mode):
            self.test_mse.append(out[-1])
        else:
            self.train_mse.append(out[-1])
        
        if(plot):
            G = dataset.G
            if(self.config["neural_ode"]):
                t_plot = torch.linspace(0, np.max(t), 101)
                j = np.argmin(t)
                #init_state = torch.tensor(out[6][j], device=self.device)
                init_state = torch.zeros(2*G+self.decoder.d, device=self.device)
                ode_out = odeint_adjoint(self.decoder, init_state, t_plot,  method=self.config["ode_method"])
            
            #Plot Time
            t_ub = np.quantile(t, 0.99)
            plot_time(np.clip(t,None,t_ub), Xembed, save=f"{path}/time-{testid}-velovae.png")
            
            #Plot u/s-t and phase portrait for each gene
            lu_scale = self.lu_scale.detach().cpu().numpy().squeeze()
            ls_scale = self.ls_scale.detach().cpu().numpy().squeeze()
            if(mode=="test"):
                lu_scale = lu_scale[self.test_idx]
                ls_scale = ls_scale[self.test_idx]
            else:
                lu_scale = lu_scale[self.train_idx]
                ls_scale = ls_scale[self.train_idx]
            #print((U_sum/dataset.data[:,:G].sum(1)).mean(), (S_sum/dataset.data[:,G:].sum(1)).mean())
            for i in range(len(gind)):
                idx = gind[i]
                
                plot_sig(t.squeeze(), 
                         dataset.data[:,idx], dataset.data[:,idx+G], 
                         Uhat[:,i]*lu_scale, Shat[:,i]*ls_scale, 
                         np.array([self.label_dic_rev[x] for x in dataset.labels]),
                         gene_plot[i], 
                         save=f"{path}/sig-{gene_plot[i]}-{testid}.png",
                         sparsify=self.config['sparsify'])
                
                if(self.config["neural_ode"]):
                    z = ode_out[:,2*G:].detach().cpu().numpy()
                    rho = self.pred_rho(z, G)
                    rho = rho[:,gind]
                    u_plot, s_plot = ode_out[:,idx].detach().cpu().numpy(), ode_out[:,idx+G].detach().cpu().numpy()
                    alpha = np.exp(self.decoder.alpha[idx].detach().cpu().item())
                    beta = np.exp(self.decoder.beta[idx].detach().cpu().item())
                    gamma = np.exp(self.decoder.gamma[idx].detach().cpu().item())
                    scaling = np.exp(self.decoder.scaling[idx].detach().cpu().item())
                    vu = rho[:,i] * alpha - beta * u_plot
                    vs = beta * u_plot - gamma * s_plot
                    plot_vel(t_plot.cpu().numpy(), u_plot, s_plot, vu, vs, None, None, gene_plot[i], 
                             save=f"{path}/vel-{gene_plot[i]}-{testid}.png")
                """
                u_sample = torch.poisson(torch.tensor(Uhat[:,i]*lu_scale)).detach().cpu().numpy()
                s_sample = torch.poisson(torch.tensor(Shat[:,i]*ls_scale)).detach().cpu().numpy()

                plot_sig(t.squeeze(), 
                         dataset.data[:,idx], dataset.data[:,idx+G], 
                         u_sample, s_sample, 
                         np.array([self.label_dic_rev[x] for x in dataset.labels]),
                         gene_plot[i], 
                         save=f"{path}/sigraw-{gene_plot[i]}-{testid}.png",
                         sparsify=self.config['sparsify'])
                """
        return elbo
    
    def gene_likelihood_poisson(self, 
                                U,S,
                                Uhat, Shat, 
                                b=5000):
        N,G = U.shape
        Nb = N//b
        logpu, logps = np.empty((N,G)), np.empty((N,G))
        for i in range(Nb):
            ps_u = Poisson(torch.tensor(Uhat[i*b:(i+1)*b]))
            ps_s = Poisson(torch.tensor(Shat[i*b:(i+1)*b]))
            logpu[i*b:(i+1)*b] = ps_u.log_prob(torch.tensor(U[i*b:(i+1)*b], dtype=int))
            logps[i*b:(i+1)*b] = ps_s.log_prob(torch.tensor(S[i*b:(i+1)*b], dtype=int))
        if(Nb*b<N):
            ps_u = Poisson(torch.tensor(Uhat[Nb*b:]))
            ps_s = Poisson(torch.tensor(Shat[Nb*b:]))
            logpu[Nb*b:] = ps_u.log_prob(torch.tensor(U[Nb*b:], dtype=int))
            logps[Nb*b:] = ps_s.log_prob(torch.tensor(S[Nb*b:], dtype=int))
        return np.exp((logpu+logps)).mean(0)
    
    def save_anndata(self, adata, key, file_path, file_name=None):
        self.set_mode('eval')
        os.makedirs(file_path, exist_ok=True)
        
        adata.var[f"{key}_alpha"] = np.exp(self.decoder.alpha.detach().cpu().numpy())
        adata.var[f"{key}_beta"] = np.exp(self.decoder.beta.detach().cpu().numpy())
        adata.var[f"{key}_gamma"] = np.exp(self.decoder.gamma.detach().cpu().numpy())
        adata.var[f"{key}_scaling"] = self.decoder.scaling.exp().detach().cpu().numpy()
        
        U,S = np.array(adata.layers['unspliced'].todense()), np.array(adata.layers['spliced'].todense())
        X = np.concatenate((U,S),1)

        out, elbo = self.pred_all(X, self.cell_labels, "both", gene_idx=np.array(range(adata.n_vars)))
        Uhat, U_sum, Shat, S_sum, t, std_t, z, std_z = out
        
        adata.obs[f"{key}_time"] = t
        adata.obs[f"{key}_std_t"] = std_t
        adata.obsm[f"{key}_z"] = z
        adata.obsm[f"{key}_std_z"] = std_z
        
        adata.layers[f"{key}_uhat"] = Uhat
        adata.layers[f"{key}_shat"] = Shat
        
        lu_scale = adata.obs["library_scale_u"].to_numpy().reshape(-1,1)
        ls_scale = adata.obs["library_scale_s"].to_numpy().reshape(-1,1)
        
        adata.var[f"{key}_likelihood"] = self.gene_likelihood_poisson(U,S,Uhat*lu_scale,Shat*ls_scale)
        
        rho = self.pred_rho(z, U.shape[1])
        
        adata.layers[f"{key}_rho"] = rho

        adata.uns[f"{key}_train_idx"] = self.train_idx
        adata.uns[f"{key}_test_idx"] = self.test_idx
        
        adata.layers[f"{key}_velocity"] = (adata.var[f"{key}_beta"].to_numpy() * Uhat / adata.var[f"{key}_scaling"].to_numpy() - adata.var[f"{key}_gamma"].to_numpy() * Shat) 
        
        if(file_name is not None):
            adata.write_h5ad(f"{file_path}/{file_name}")


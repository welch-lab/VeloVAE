import numpy as np
import scipy.stats as stats
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
import time
from velovae.plotting import plot_sig, plot_sig_, plot_time, plot_train_loss, plot_test_loss

from .model_util import hist_equal, init_params, get_ts_global, reinit_params, convert_time, get_gene_index, optimal_transport_duality_gap
from .model_util import pred_su, ode, ode_numpy, knnx0, knnx0_bin
from .TransitionGraph import encode_type
from .TrainingData import SCData
from .VanillaVAE import VanillaVAE, kl_gaussian, kl_uniform
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
        
        self.net = nn.Sequential(self.fc1, self.bn1, nn.LeakyReLU(), self.dpt1,
                                 self.fc2, self.bn2, nn.LeakyReLU(), self.dpt2,
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
                 init_ton_zero=False,
                 device=torch.device('cpu'), 
                 init_method="steady", 
                 init_key=None, 
                 init_type=None,
                 checkpoint=None,
                 **kwargs):
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
            self.scaling = nn.Parameter(torch.empty(G, device=device).float())
            self.ton = nn.Parameter(torch.empty(G, device=device).float())
            self.sigma_u = nn.Parameter(torch.empty(G, device=device).float())
            self.sigma_s = nn.Parameter(torch.empty(G, device=device).float())
            
            self.load_state_dict(torch.load(checkpoint, map_location=device))
        else:
            self.init_weights()
            
            #Dynamical Model Parameters
            U,S = adata.layers['Mu'][train_idx], adata.layers['Ms'][train_idx]
            X = np.concatenate((U,S),1)
            if(init_method == "random"):
                print("Random Initialization.")
                alpha, beta, gamma, scaling, toff, u0, s0, sigma_u, sigma_s, T, Rscore = init_params(X,p,fit_scaling=True)
                self.alpha = nn.Parameter(torch.normal(0.0, 0.01, size=(U.shape[1],), device=device).float())
                self.beta =  nn.Parameter(torch.normal(0.0, 0.01, size=(U.shape[1],), device=device).float())
                self.gamma = nn.Parameter(torch.normal(0.0, 0.01, size=(U.shape[1],), device=device).float())
                self.ton = torch.nn.Parameter(torch.ones(G, device=device).float()*(-10))
                self.scaling = nn.Parameter(torch.tensor(np.log(scaling), device=device).float())
                self.sigma_u = nn.Parameter(torch.tensor(np.log(sigma_u), device=device).float())
                self.sigma_s = nn.Parameter(torch.tensor(np.log(sigma_s), device=device).float())
                self.t_init = None
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
                self.toff_init = get_ts_global(self.t_init, U/scaling, S, 95)
                self.alpha_init, self.beta_init, self.gamma_init, self.ton_init = reinit_params(U/scaling, S, self.t_init, self.toff_init)
                
                self.alpha = nn.Parameter(torch.tensor(np.log(self.alpha_init), device=device).float())
                self.beta = nn.Parameter(torch.tensor(np.log(self.beta_init), device=device).float())
                self.gamma = nn.Parameter(torch.tensor(np.log(self.gamma_init), device=device).float())
                self.scaling = nn.Parameter(torch.tensor(np.log(scaling), device=device).float())
                self.sigma_u = nn.Parameter(torch.tensor(np.log(sigma_u), device=device).float())
                self.sigma_s = nn.Parameter(torch.tensor(np.log(sigma_s), device=device).float())
                self.ton = nn.Parameter((torch.ones(G, device=device)*(-10)).float()) if init_ton_zero else nn.Parameter(torch.tensor(np.log(self.ton_init+1e-10), device=device).float())
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
                    if("init_t_quant" in kwargs):
                        self.t_init = np.quantile(T_eq,kwargs["init_t_quant"],1)
                    else:
                        self.t_init = np.quantile(T_eq,0.5,1)
                self.toff_init = get_ts_global(self.t_init, U/scaling, S, 95)
                self.alpha_init, self.beta_init, self.gamma_init, self.ton_init = reinit_params(U/scaling, S, self.t_init, self.toff_init)
                
                self.alpha = nn.Parameter(torch.tensor(np.log(self.alpha_init), device=device).float())
                self.beta = nn.Parameter(torch.tensor(np.log(self.beta_init), device=device).float())
                self.gamma = nn.Parameter(torch.tensor(np.log(self.gamma_init), device=device).float())
                self.scaling = nn.Parameter(torch.tensor(np.log(scaling), device=device).float())
                self.ton = nn.Parameter((torch.ones(adata.n_vars, device=device)*(-10)).float()) if init_ton_zero else nn.Parameter(torch.tensor(np.log(self.ton_init+1e-10), device=device).float())
                self.sigma_u = nn.Parameter(torch.tensor(np.log(sigma_u), device=device).float())
                self.sigma_s = nn.Parameter(torch.tensor(np.log(sigma_s), device=device).float())
        
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
            self.u0 = nn.Parameter(torch.tensor(np.log(U[cell_mask].mean(0)+1e-10), device=device).float())
            self.s0 = nn.Parameter(torch.tensor(np.log(S[cell_mask].mean(0)+1e-10), device=device).float())

        self.scaling.requires_grad = False
        self.sigma_u.requires_grad = False
        self.sigma_s.requires_grad = False
    
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
            Uhat, Shat = pred_su(F.leaky_relu(t-t0, neg_slope), u0/self.scaling.exp(), s0, alpha, self.beta.exp(), self.gamma.exp())
        Uhat = Uhat * torch.exp(self.scaling)
        return nn.functional.relu(Uhat), nn.functional.relu(Shat)

class VAE(VanillaVAE):
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
                 checkpoints=[None, None],
                 **kwargs):
        """VeloVAE Model
        
        Arguments
        ---------
        
        adata : :class:`anndata.AnnData`
            AnnData object containing all relevant data and meta-data
        tmax : float
            Time range.
            This is used to restrict the cell time within certain range. In the 
            case of a Gaussian model without capture time, tmax/2 will be the mean of prior.
            If capture time is provided, then they are scaled to match a range of tmax.
            In the case of a uniform model, tmax is strictly the maximum time.
        dim_z : int
            Dimension of the latent cell state
        dim_cond : int, optional
            Dimension of additional information for the conditional VAE.
            Set to zero by default, equivalent to a VAE.
            This feature is not stable now.
        device : {'gpu','cpu'}, optional
            Training device
        hidden_size : tuple of int, optional
            Width of the hidden layers. Should be a tuple of the form
            (encoder layer 1, encoder layer 2, decoder layer 1, decoder layer 2)
        init_method : {'random', 'tprior', 'steady}, optional
            Initialization method. 
            Should choose from
            (1) random: random initialization
            (2) tprior: use the capture time to estimate rate parameters. Cell time will be 
                        randomly sampled with the capture time as the mean. The variance can
                        be controlled by changing 'time_overlap' in config.
            (3) steady: use the steady-state model to estimate gamma, alpha and assume beta = 1.
                        After this, a global cell time is estimated by taking the quantile over
                        all local times. Finally, rate parameters are reinitialized using the
                        global cell time.
        init_key : str, optional
            column in the AnnData object containing the capture time
        tprior : str, optional
            key in adata.obs that stores the capture time.
            Used for informative time prior
        init_type : str, optional
            The stem cell type. Used to estimated the initial conditions.
            This is not commonly used in practice and please consider leaving it to default.
        init_ton_zero : bool, optional
            Whether to add a non-zero switch-on time for each gene.
            It's set to True if there's no capture time.
        time_distribution : {'gaussian', 'uniform'}, optional
            Time distribution, set to Gaussian by default.
        std_z_prior : float, optional
            Standard deviation of the prior (isotropical Gaussian) of cell state.
        checkpoints : list of 2 strings, optional
            Contains the path to saved encoder and decoder models.
            Should be a .pt file.
        """
        try:
            U,S = adata.layers['Mu'], adata.layers['Ms']
        except KeyError:
            print('Unspliced/Spliced count matrices not found in the layers! Exit the program...')
            return
        
        #Training Configuration
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
            "n_bin": None,

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
            "train_ton":True,
            "train_x0":False,
            "weight_sample":False,
            
            #Plotting
            "sparsify":1
        }
        
        self.set_device(device)
        self.split_train_test(adata.n_obs)
        
        G = adata.n_vars
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
                               device=self.device, 
                               init_method = init_method,
                               init_key = init_key,
                               init_type = init_type,
                               checkpoint=checkpoints[1], 
                               **kwargs).float()
        self.tmax=tmax
        self.time_distribution = time_distribution
        self.get_prior(adata, time_distribution, tmax, tprior)
        
        self.p_z = torch.stack([torch.zeros(U.shape[0],dim_z), torch.ones(U.shape[0],dim_z)*self.config["std_z_prior"]]).float().to(self.device)
        
        self.use_knn = False
        self.u0 = None
        self.s0 = None
        self.t0 = None
        
        #Class attributes for training
        self.loss_train, self.loss_test = [],[]
        self.counter = 0 #Count the number of iterations
        self.n_drop = 0 #Count the number of consecutive iterations with little decrease in loss
    
    def forward(self, data_in, u0=None, s0=None, t0=None, condition=None):
        """Standard forward pass.
        
        Arguments
        ---------
        
        data_in : `torch.tensor` 
            input count data, (N, 2G)
        u0 : `torch.tensor`, optional
            Initial condition of u, (N, G)
            This is set to None in the first stage when cell time is 
            not fixed. It will have some value in the second stage, so the users
            shouldn't worry about feeding the parameter themselves.
        s0 : `torch.tensor`, optional
            Initial condition of s, (N,G)
        t0 : `torch.tensor`, optional
            time at the initial condition, (N,1)
        
        Returns
        -------
        
        mu_t : `torch.tensor`, optional
            time mean, (N,1)
        std_t : `torch.tensor`, optional
            time standard deviation, (N,1)
        mu_z : `torch.tensor`, optional
            cell state mean, (N, Cz)
        std_z : `torch.tensor`, optional
            cell state standard deviation, (N, Cz)
        t : `torch.tensor`, optional
            sampled cell time, (N,1)
        z : `torch.tensor`, optional
            sampled cell sate, (N,Cz)
        uhat : `torch.tensor`, optional
            predicted mean u values, (N,G)
        shat : `torch.tensor`, optional
            predicted mean s values, (N,G)
        """
        data_in_scale = torch.cat((data_in[:,:data_in.shape[1]//2]/torch.exp(self.decoder.scaling), data_in[:,data_in.shape[1]//2:]),1)
        mu_t, std_t, mu_z, std_z = self.encoder.forward(data_in_scale, condition)
        t = self.sample(mu_t, std_t)
        z = self.sample(mu_z, std_z)
         
        uhat, shat = self.decoder.forward(t, z, condition, u0, s0, t0, neg_slope=self.config["neg_slope"])
        return mu_t, std_t, mu_z, std_z, t, z, uhat, shat
    
    def eval_model(self, data_in, u0=None, s0=None, t0=None, condition=None):
        """Evaluate the model on the validation dataset. 
        The major difference from forward pass is that we use the mean time and
        cell state instead of random sampling. The input arguments are the same as 'forward'.
        """
        data_in_scale = torch.cat((data_in[:,:data_in.shape[1]//2]/torch.exp(self.decoder.scaling), data_in[:,data_in.shape[1]//2:]),1)
        mu_t, std_t, mu_z, std_z = self.encoder.forward(data_in_scale, condition)
         
        uhat, shat = self.decoder.forward(mu_t, mu_z, condition, u0=u0, s0=s0, t0=t0, neg_slope=0.0)
        return mu_t, std_t, mu_z, std_z, uhat, shat
    
    def vae_risk(self, 
                q_tx, p_t, 
                q_zx, p_z, 
                u, s, uhat, shat, 
                sigma_u, sigma_s,
                weight=None):
        """Training objective function. This is the negative ELBO.
        
        Arguments
        ---------
        
        q_tx : tuple of `torch.tensor`
            Parameters of time posterior. Mean and std are both (N, 1) tensors.
        p_t : tuple of `torch.tensor`
            Parameters of time prior.
        q_zx : tuple of `torch.tensor`
            Parameters of cell state posterior. Mean and std are both (N, Dz) tensors.
        p_z  : tuple of `torch.tensor`
            Parameters of cell state prior.
        u, s : `torch.tensor`
            Input data
        uhat, shat : torch.tensor
            Prediction by VeloVAE
        sigma_u, sigma_s : `torch.tensor`
            Standard deviation of the Gaussian noise
        weight : `torch.tensor`, optional
            Sample weight. This feature is not stable. Please consider setting it to None.
        
        Returns
        -------
        Negative ELBO : torch.tensor, scalar
        """
        kldt = self.kl_time(q_tx[0], q_tx[1], p_t[0], p_t[1], tail=self.config["tail"])
        kldz = kl_gaussian(q_zx[0], q_zx[1], p_z[0], p_z[1])
        
        #u and sigma_u has the original scale
        logp = -0.5*((u-uhat)/sigma_u).pow(2)-0.5*((s-shat)/sigma_s).pow(2)-torch.log(sigma_u)-torch.log(sigma_s*2*np.pi)
        
        if( weight is not None):
            logp = logp*weight
            
        err_rec = torch.mean(torch.sum(logp,1))
        
        return (- err_rec + self.config["kl_t"]*kldt + self.config["kl_z"]*kldz)
    
    def train_epoch(self, train_loader, test_set, optimizer, optimizer2=None, K=1):
        ##########################################################################
        #Training in each epoch.
        #Early stopping if enforced by default. 
        #Arguments
        #---------
        #1.  train_loader [torch.utils.data.DataLoader]
        #    Data loader of the input data.
        #2.  test_set [torch.utils.data.Dataset]
        #    Validation dataset
        #3.  optimizer  [optimizer from torch.optim]
        #4.  optimizer2 [optimizer from torch.optim]
        #    (Optional) A second optimizer.
        #    This is used when we optimize NN and ODE simultaneously in one epoch.
        #    By default, VeloVAE performs alternating optimization in each epoch.
        #    The argument will be set to proper value automatically.
        #5.  K [int]
        #    Alternating update period.
        #    For every K updates of optimizer, there's one update for optimizer2.
        #    If set to 0, optimizer2 will be ignored and only optimizer will be
        #    updated. Users can set it to 0 if they want to update sorely NN in one 
        #    epoch and ODE in the next epoch. 
        #Returns
        #1.  stop_training [bool]
        #    Whether to stop training based on the early stopping criterium.
        ##########################################################################
        
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
            
            if(self.use_knn and self.config["train_x0"]):
                optimizer_x0 = torch.optim.Adam([s0, u0], lr=self.config["learning_rate_ode"])
                optimizer_x0.zero_grad()
            
            condition = F.one_hot(batch[1].to(self.device), self.n_type).float() if self.enable_cvae else None
            mu_tx, std_tx, mu_zx, std_zx, t, z, uhat, shat = self.forward(xbatch, u0, s0, t0, condition)
            
            loss = self.vae_risk((mu_tx, std_tx), self.p_t[:,self.train_idx[idx],:],
                                (mu_zx, std_zx), self.p_z[:,self.train_idx[idx],:],
                                u, s, 
                                uhat, shat, 
                                torch.exp(self.decoder.sigma_u), torch.exp(self.decoder.sigma_s), 
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
            
            #Update initial conditions
            if(self.use_knn and self.config["train_x0"]):
                optimizer_x0.step()
                self.t0[self.train_idx[idx]] = t0.detach().cpu().numpy()
                self.u0[self.train_idx[idx]] = u0.detach().cpu().numpy()
                self.s0[self.train_idx[idx]] = s0.detach().cpu().numpy()
            
            self.loss_train.append(loss.detach().cpu().item())
            self.counter = self.counter + 1
        return stop_training
    
    def update_x0(self, U, S, n_bin=None):
        ##########################################################################
        #Estimate the initial conditions using KNN
        #< Input Arguments >
        #1.  U [tensor (N,G)]
        #    Input unspliced count matrix
        #2   S [tensor (N,G)]
        #    Input spliced count matrix
        #3.  n_bin [int]
        #    (Optional) Set to a positive integer if binned KNN is used.
        #< Output >
        #1.  u0 [tensor (N,G)]
        #    Initial condition for u
        #    
        #2.  s0 [tensor (N,G)]
        #    Initial condition for s
        #3. t0 [tensor (N,1)]
        #   Initial time
        ##########################################################################
        start = time.time()
        self.set_mode('eval')
        out, elbo = self.pred_all(np.concatenate((U,S),1), self.cell_labels, "both", ["t","z"])
        t, z = out[0], out[2]
        #Clip the time to avoid outliers
        t = np.clip(t, 0, np.quantile(t, 0.99))
        dt = (self.config["dt"][0]*(t.max()-t.min()), self.config["dt"][1]*(t.max()-t.min()))
        if(n_bin is None):
            print(f"Cell-wise KNN Estimation.")
            u0, s0, t0 = knnx0(U[self.train_idx], S[self.train_idx], t[self.train_idx], z[self.train_idx], t, z, dt, self.config["n_neighbors"])
        else:
            print(f"Fast KNN Estimation with {n_bin} time bins.")
            u0, s0, t0 = knnx0_bin(U[self.train_idx], S[self.train_idx], t[self.train_idx], z[self.train_idx], t, z, dt, self.config["n_neighbors"])
        
        print(f"Finished. Actual Time: {convert_time(time.time()-start)}")
        return u0, s0, t0.reshape(-1,1)
        
    def train(self, 
              adata, 
              config={}, 
              plot=False, 
              gene_plot=[], 
              cluster_key = "clusters",
              figure_path = "figures", 
              embed="umap",
              use_raw=False):
        """The high-level API for training.
        
        Arguments
        ---------
        
        adata : :class:`anndata.AnnData`
        config : dictionary, optional
            Contains all hyper-parameters.
        plot : bool, optional
            Whether to plot some sample genes during training. Used for debugging.
        gene_plot : string list, optional
            List of gene names to plot. Used only if plot==True
        cluster_key : str, optional
            Key in adata.obs storing the cell type annotation.
        figure_path : str, optional
            Path to the folder for saving plots.
        embed : str, optional
            Low dimensional embedding in adata.obsm. The actual key storing the embedding should be f'X_{embed}'
        use_raw : bool, optional
            Whether to use the raw counts as training data.
        """
        self.load_config(config)
        
        print("--------------------------- Train a VeloVAE ---------------------------")
        #Get data loader
        if(use_raw):
            U, S = np.array(adata.layers['unspliced'].todense()), np.array(adata.layers['spliced'].todense())
            X = np.concatenate((U,S), 1).astype(int)
        else:
            X = np.concatenate((adata.layers['Mu'], adata.layers['Ms']), 1).astype(float)
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
        train_set = SCData(X[self.train_idx], self.cell_labels[self.train_idx], weight=self.decoder.Rscore[self.train_idx]) if self.config['weight_sample'] else SCData(X[self.train_idx], self.cell_labels[self.train_idx])
        test_set = None
        if(len(self.test_idx)>0):
            test_set = SCData(X[self.test_idx], self.cell_labels[self.test_idx], weight=self.decoder.Rscore[self.test_idx]) if self.config['weight_sample'] else SCData(X[self.test_idx], self.cell_labels[self.test_idx])
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
            self.decoder.scaling.requires_grad = True
            param_ode = param_ode+[self.decoder.scaling]
        if(self.config['train_std']):
            self.decoder.sigma_u.requires_grad = True
            self.decoder.sigma_s.requires_grad = True
            param_ode = param_ode+[self.decoder.sigma_u, self.decoder.sigma_s]

        optimizer = torch.optim.Adam(param_nn, lr=self.config["learning_rate"], weight_decay=self.config["lambda"])
        optimizer_ode = torch.optim.Adam(param_ode, lr=self.config["learning_rate_ode"])
        print("*********                      Finished.                      *********")
        
        #Debugging Plots
        if(self.decoder.t_init is not None):
            tplot = self.decoder.t_init
            for i,idx in enumerate(gind):
                alpha = self.decoder.alpha_init[idx]
                beta = self.decoder.beta_init[idx]
                gamma = self.decoder.gamma_init[idx]
                ton = self.decoder.ton_init[idx]
                toff = self.decoder.toff_init[idx]
                scaling = self.decoder.scaling[idx].detach().cpu().exp().item()
                upred, spred = ode_numpy(tplot,alpha,beta,gamma,ton,toff,scaling)
                plot_sig_(tplot, X[self.train_idx,idx], X[self.train_idx,idx+adata.n_vars], cell_labels_raw[self.train_idx], tplot, upred, spred, title=gene_plot[i], save=f"{figure_path}/{gene_plot[i]}_init.png")
      
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
            
            if(epoch==0 or (epoch+1) % self.config["save_epoch"] == 0):
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
        self.encoder.eval()
        u0, s0, t0 = self.update_x0(X[:,:X.shape[1]//2], X[:,X.shape[1]//2:], self.config["n_bin"])
        self.u0 = u0
        self.s0 = s0
        self.t0 = t0
        
        self.use_knn = True
        self.decoder.init_weights()
        #Plot the initial conditions
        if(plot):
            plot_time(self.t0.squeeze(), Xembed, save=f"{figure_path}/t0.png")
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
            
            
            
            if(epoch==0 or (epoch+n_stage1+1) % self.config["save_epoch"] == 0):
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
        
        print(f"*********              Finished. Total Time = {convert_time(time.time()-start)}             *********")
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
        self.loss_train.append(elbo_train)
        self.loss_test.append(elbo_test)
        #Plot final results
        if(plot):
            plot_train_loss(self.loss_train, range(1,len(self.loss_train)+1), save=f'{figure_path}/train_loss_velovae.png')
            if(self.config["test_iter"]>0):
                plot_test_loss(self.loss_test, [i*self.config["test_iter"] for i in range(1,len(self.loss_test)+1)], save=f'{figure_path}/test_loss_velovae.png')
        return
    
    def pred_all(self, data, cell_labels, mode='test', output=["uhat", "shat", "t", "z"], gene_idx=None):
        N, G = data.shape[0], data.shape[1]//2
        elbo = 0
        if("uhat" in output):
            Uhat = None if gene_idx is None else np.zeros((N,len(gene_idx)))
        if("shat" in output):
            Shat = None if gene_idx is None else np.zeros((N,len(gene_idx)))
        if("t" in output):
            t_out = np.zeros((N))
            std_t_out = np.zeros((N))
        if("z" in output):
            z_out = np.zeros((N, self.dim_z))
            std_z_out = np.zeros((N, self.dim_z))
        
        with torch.no_grad():
            B = min(N//5, 5000)
            Nb = N // B
            for i in range(Nb):
                data_in = torch.tensor(data[i*B:(i+1)*B]).float().to(self.device)
                if(mode=="test"):
                    u0 = torch.tensor(self.u0[self.test_idx[i*B:(i+1)*B]], dtype=torch.float, device=self.device) if self.use_knn else None
                    s0 = torch.tensor(self.s0[self.test_idx[i*B:(i+1)*B]], dtype=torch.float, device=self.device) if self.use_knn else None
                    t0 = torch.tensor(self.t0[self.test_idx[i*B:(i+1)*B]], dtype=torch.float, device=self.device) if self.use_knn else None
                    y_onehot = F.one_hot(torch.tensor(cell_labels[self.test_idx[i*B:(i+1)*B]], dtype=int, device=self.device), self.n_type).float() if self.enable_cvae else None
                    p_t = self.p_t[:,self.test_idx[i*B:(i+1)*B],:]
                    p_z = self.p_z[:,self.test_idx[i*B:(i+1)*B],:]
                    mu_tx, std_tx, mu_zx, std_zx, uhat, shat = self.eval_model(data_in, u0, s0, t0, y_onehot)
                elif(mode=="train"):
                    u0 = torch.tensor(self.u0[self.train_idx[i*B:(i+1)*B]], dtype=torch.float, device=self.device) if self.use_knn else None
                    s0 = torch.tensor(self.s0[self.train_idx[i*B:(i+1)*B]], dtype=torch.float, device=self.device) if self.use_knn else None
                    t0 = torch.tensor(self.t0[self.train_idx[i*B:(i+1)*B]], dtype=torch.float, device=self.device) if self.use_knn else None
                    y_onehot = F.one_hot(torch.tensor(cell_labels[self.train_idx[i*B:(i+1)*B]], dtype=int, device=self.device), self.n_type).float() if self.enable_cvae else None
                    p_t = self.p_t[:,self.train_idx[i*B:(i+1)*B],:]
                    p_z = self.p_z[:,self.train_idx[i*B:(i+1)*B],:]
                    mu_tx, std_tx, mu_zx, std_zx, uhat, shat = self.eval_model(data_in, u0, s0, t0, y_onehot)
                else:
                    u0 = torch.tensor(self.u0[i*B:(i+1)*B], dtype=torch.float, device=self.device) if self.use_knn else None
                    s0 = torch.tensor(self.s0[i*B:(i+1)*B], dtype=torch.float, device=self.device) if self.use_knn else None
                    t0 = torch.tensor(self.t0[i*B:(i+1)*B], dtype=torch.float, device=self.device) if self.use_knn else None
                    y_onehot = F.one_hot(torch.tensor(cell_labels[i*B:(i+1)*B], dtype=int, device=self.device), self.n_type).float() if self.enable_cvae else None
                    p_t = self.p_t[:,i*B:(i+1)*B,:]
                    p_z = self.p_z[:,i*B:(i+1)*B,:]
                    mu_tx, std_tx, mu_zx, std_zx, uhat, shat = self.eval_model(data_in, u0, s0, t0, y_onehot)
                
                loss = self.vae_risk((mu_tx, std_tx), p_t,
                                    (mu_zx, std_zx), p_z,
                                    data_in[:,:G], data_in[:,G:], 
                                    uhat, shat, 
                                    torch.exp(self.decoder.sigma_u), torch.exp(self.decoder.sigma_s), 
                                    None)
                elbo = elbo - (B/N)*loss
                if("uhat" in output and gene_idx is not None):
                    Uhat[i*B:(i+1)*B] = uhat[:,gene_idx].cpu().numpy()
                if("shat" in output and gene_idx is not None):
                    Shat[i*B:(i+1)*B] = shat[:,gene_idx].cpu().numpy()
                if("t" in output):
                    t_out[i*B:(i+1)*B] = mu_tx.cpu().squeeze().numpy()
                    std_t_out[i*B:(i+1)*B] = std_tx.cpu().squeeze().numpy()
                if("z" in output):
                    z_out[i*B:(i+1)*B] = mu_zx.cpu().numpy()
                    std_z_out[i*B:(i+1)*B] = std_zx.cpu().numpy()
            if(N > B*Nb):
                data_in = torch.tensor(data[Nb*B:]).float().to(self.device)
                if(mode=="test"):
                    u0 = torch.tensor(self.u0[self.test_idx[Nb*B:]], dtype=torch.float, device=self.device) if self.use_knn else None
                    s0 = torch.tensor(self.s0[self.test_idx[Nb*B:]], dtype=torch.float, device=self.device) if self.use_knn else None
                    t0 = torch.tensor(self.t0[self.test_idx[Nb*B:]], dtype=torch.float, device=self.device) if self.use_knn else None
                    y_onehot = F.one_hot(torch.tensor(cell_labels[self.test_idx[Nb*B:]], dtype=int, device=self.device), self.n_type).float() if self.enable_cvae else None
                    p_t = self.p_t[:,self.test_idx[Nb*B:],:]
                    p_z = self.p_z[:,self.test_idx[Nb*B:],:]
                    mu_tx, std_tx, mu_zx, std_zx, uhat, shat = self.eval_model(data_in, u0, s0, t0, y_onehot)
                elif(mode=="train"):
                    u0 = torch.tensor(self.u0[self.train_idx[Nb*B:]], dtype=torch.float, device=self.device) if self.use_knn else None
                    s0 = torch.tensor(self.s0[self.train_idx[Nb*B:]], dtype=torch.float, device=self.device) if self.use_knn else None
                    t0 = torch.tensor(self.t0[self.train_idx[Nb*B:]], dtype=torch.float, device=self.device) if self.use_knn else None
                    y_onehot = F.one_hot(torch.tensor(cell_labels[self.train_idx[Nb*B:]], dtype=int, device=self.device), self.n_type).float() if self.enable_cvae else None
                    p_t = self.p_t[:,self.train_idx[Nb*B:],:]
                    p_z = self.p_z[:,self.train_idx[Nb*B:],:]
                    mu_tx, std_tx, mu_zx, std_zx, uhat, shat = self.eval_model(data_in, u0, s0, t0, y_onehot)
                else:
                    u0 = torch.tensor(self.u0[Nb*B:], dtype=torch.float, device=self.device) if self.use_knn else None
                    s0 = torch.tensor(self.s0[Nb*B:], dtype=torch.float, device=self.device) if self.use_knn else None
                    t0 = torch.tensor(self.t0[Nb*B:], dtype=torch.float, device=self.device) if self.use_knn else None
                    y_onehot = F.one_hot(torch.tensor(cell_labels[Nb*B:], dtype=int, device=self.device), self.n_type).float() if self.enable_cvae else None
                    p_t = self.p_t[:,Nb*B:,:]
                    p_z = self.p_z[:,Nb*B:,:]
                    mu_tx, std_tx, mu_zx, std_zx, uhat, shat = self.eval_model(data_in, u0, s0, t0, y_onehot)
                loss = self.vae_risk((mu_tx, std_tx), p_t,
                                    (mu_zx, std_zx), p_z,
                                    data_in[:,:G], data_in[:,G:], 
                                    uhat, shat, 
                                    torch.exp(self.decoder.sigma_u), torch.exp(self.decoder.sigma_s), 
                                    None)
                elbo = elbo - ((N-B*Nb)/N)*loss
                if("uhat" in output and gene_idx is not None):
                    Uhat[Nb*B:] = uhat[:,gene_idx].cpu().numpy()
                if("shat" in output and gene_idx is not None):
                    Shat[Nb*B:] = shat[:,gene_idx].cpu().numpy()
                if("t" in output):
                    t_out[Nb*B:] = mu_tx.cpu().squeeze().numpy()
                    std_t_out[Nb*B:] = std_tx.cpu().squeeze().numpy()
                if("z" in output):
                    z_out[Nb*B:] = mu_zx.cpu().numpy()
                    std_z_out[Nb*B:] = std_zx.cpu().numpy()
        out = []
        if("uhat" in output):
            out.append(Uhat)
        if("shat" in output):
            out.append(Shat)
        if("t" in output):
            out.append(t_out)
            out.append(std_t_out)
        if("z" in output):
            out.append(z_out)
            out.append(std_z_out)
        
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
        """Evaluate the model upon training/test dataset.
        
        Arguments
        ---------
        dataset : `torch.utils.data.Dataset`
            Training or validation dataset
        Xembed : `numpy array`
            Low-dimensional embedding for plotting
        testid : str or int, optional
            Used to name the figures.
        test_mode : bool, optional
            Whether dataset is training or validation dataset. This is used when retreiving certain class variable, e.g. cell-specific initial condition.
        gind : `numpy array`, optional
            Index of genes in adata.var_names. Used for plotting.
        gene_plot : `numpy array`, optional
            Gene names.
        plot : bool, optional
            Whether to generate plots.
        path : str
            Saving path.
        
        Returns
        -------
        elbo : float
        """
        self.set_mode('eval')
        mode = "test" if test_mode else "train"
        out, elbo = self.pred_all(dataset.data, self.cell_labels, mode, ["uhat", "shat", "t"], gind)
        Uhat, Shat, t, std_t = out[0], out[1], out[2], out[3]
        
        G = dataset.data.shape[1]//2

        if(plot):
            #Plot Time
            plot_time(t, Xembed, save=f"{path}/time-{testid}-velovae.png")
            
            #Plot u/s-t and phase portrait for each gene
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
    
    def update_std_noise(self, train_data):
        """Update the standard deviation of Gaussian noise.
        .. deprecated:: 1.0
        """
        G = train_data.shape[1]//2
        out, elbo = self.pred_all(train_data, mode='train', output=["uhat", "shat"], gene_idx = np.array(range(G)))
        self.decoder.sigma_u = nn.Parameter(torch.tensor(np.log((out[0]-train_data[:,:G]).std(0)), device=self.device))
        self.decoder.sigma_s = nn.Parameter(torch.tensor(np.log((out[1]-train_data[:,G:]).std(0)), device=self.device))
        return
    
    def save_anndata(self, adata, key, file_path, file_name=None):
        """Save the ODE parameters and cell time to the anndata object and write it to disk.
        
        Arguments
        ---------
        adata : :class:`anndata.AnnData`
        key : str
            Used to store all parameters of the model.
        file_path : str
            Saving path.
        file_name : str, optional
            If set to a string ending with .h5ad, the updated anndata object will be written to disk.
        """
        self.set_mode('eval')
        os.makedirs(file_path, exist_ok=True)
        
        adata.var[f"{key}_alpha"] = np.exp(self.decoder.alpha.detach().cpu().numpy())
        adata.var[f"{key}_beta"] = np.exp(self.decoder.beta.detach().cpu().numpy())
        adata.var[f"{key}_gamma"] = np.exp(self.decoder.gamma.detach().cpu().numpy())
        adata.var[f"{key}_ton"] = self.decoder.ton.exp().detach().cpu().numpy()
        adata.var[f"{key}_scaling"] = np.exp(self.decoder.scaling.detach().cpu().numpy())
        adata.var[f"{key}_sigma_u"] = np.exp(self.decoder.sigma_u.detach().cpu().numpy())
        adata.var[f"{key}_sigma_s"] = np.exp(self.decoder.sigma_s.detach().cpu().numpy())
        U,S = adata.layers['Mu'], adata.layers['Ms']
        scaling = adata.var[f"{key}_scaling"].to_numpy()
        
        out, elbo = self.pred_all(np.concatenate((adata.layers['Mu'], adata.layers['Ms']),1), self.cell_labels, "both", gene_idx=np.array(range(adata.n_vars)))
        Uhat, Shat, t, std_t, z, std_z = out[0], out[1], out[2], out[3], out[4], out[5]

        adata.obs[f"{key}_time"] = t
        adata.obs[f"{key}_std_t"] = std_t
        adata.obsm[f"{key}_z"] = z
        adata.obsm[f"{key}_std_z"] = std_z
        adata.layers[f"{key}_uhat"] = Uhat
        adata.layers[f"{key}_shat"] = Shat
        sigma_u, sigma_s = adata.var[f"{key}_sigma_u"].to_numpy(), adata.var[f"{key}_sigma_s"].to_numpy()
        adata.var[f"{key}_likelihood"] = np.mean(-0.5*((adata.layers["Mu"]-Uhat)/sigma_u)**2-0.5*((adata.layers["Ms"]-Shat)/sigma_s)**2 - np.log(sigma_u) - np.log(sigma_s) - np.log(2*np.pi), 0)
        
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

        adata.uns[f"{key}_train_idx"] = self.train_idx
        adata.uns[f"{key}_test_idx"] = self.test_idx
        
        rna_velocity_vae(adata, key, use_raw=False, use_scv_genes=False)
        
        if(file_name is not None):
            adata.write_h5ad(f"{file_path}/{file_name}")


#######################################################
# Full Variational Bayes
#   In this version, we treat decoder parameters (ODE
#   parameters) as random variables and assumes a prior
#   and approximate posterior.
#######################################################
class decoder_fullvb(nn.Module):
    def __init__(self, 
                 adata, 
                 tmax,
                 train_idx,
                 dim_z, 
                 N1=250, 
                 N2=500, 
                 p=98, 
                 init_ton_zero=False,
                 device=torch.device('cpu'), 
                 init_method ="steady", 
                 init_key=None, 
                 init_type=None,
                 checkpoint=None,
                 **kwargs):
        super(decoder_fullvb, self).__init__()
        G = adata.n_vars
        self.fc1 = nn.Linear(dim_z, N1).to(device)
        self.bn1 = nn.BatchNorm1d(num_features=N1).to(device)
        self.dpt1 = nn.Dropout(p=0.2).to(device)
        self.fc2 = nn.Linear(N1, N2).to(device)
        self.bn2 = nn.BatchNorm1d(num_features=N2).to(device)
        self.dpt2 = nn.Dropout(p=0.2).to(device)
        
        self.fc_out1 = nn.Linear(N2, G).to(device)
        
        self.net_rho = nn.Sequential(self.fc1, self.bn1, nn.LeakyReLU(), self.dpt1,
                                     self.fc2, self.bn2, nn.LeakyReLU(), self.dpt2)
        
        self.fc3 = nn.Linear(dim_z, N1).to(device)
        self.bn3 = nn.BatchNorm1d(num_features=N1).to(device)
        self.dpt3 = nn.Dropout(p=0.2).to(device)
        self.fc4 = nn.Linear(N1, N2).to(device)
        self.bn4 = nn.BatchNorm1d(num_features=N2).to(device)
        self.dpt4 = nn.Dropout(p=0.2).to(device)
        
        self.fc_out2 = nn.Linear(N2, G).to(device)
        
        self.net_rho2 = nn.Sequential(self.fc3, self.bn3, nn.LeakyReLU(), self.dpt3,
                                      self.fc4, self.bn4, nn.LeakyReLU(), self.dpt4)
        
        sigma_param = np.log(0.05)
        if(checkpoint is not None):
            self.alpha = nn.Parameter(torch.empty((2,G), device=device).float())
            self.beta = nn.Parameter(torch.empty((2,G), device=device).float())
            self.gamma = nn.Parameter(torch.empty((2,G), device=device).float())
            self.scaling = nn.Parameter(torch.empty(G, device=device).float())
            self.ton = nn.Parameter(torch.empty(G, device=device).float())
            self.sigma_u = nn.Parameter(torch.empty(G, device=device).float())
            self.sigma_s = nn.Parameter(torch.empty(G, device=device).float())
            
            self.load_state_dict(torch.load(checkpoint, map_location=device))
        else:
            self.init_weights()
            
            #Dynamical Model Parameters
            U,S = adata.layers['Mu'][train_idx], adata.layers['Ms'][train_idx]
            X = np.concatenate((U,S),1)
            if(init_method == "random"):
                print("Random Initialization.")
                alpha, beta, gamma, scaling, toff, u0, s0, sigma_u, sigma_s, T, Rscore = init_params(X,p,fit_scaling=True)
                self.alpha = nn.Parameter(torch.normal(0.0, 1.0, size=(2,U.shape[1]), device=device).float())
                self.beta =  nn.Parameter(torch.normal(0.0, 0.5, size=(2,U.shape[1]), device=device).float())
                self.gamma = nn.Parameter(torch.normal(0.0, 0.5, size=(2,U.shape[1]), device=device).float())
                self.ton = torch.nn.Parameter(torch.ones(G, device=device).float()*(-10))
                self.scaling = nn.Parameter(torch.tensor(np.log(scaling), device=device).float())
                self.sigma_u = nn.Parameter(torch.tensor(np.log(sigma_u), device=device).float())
                self.sigma_s = nn.Parameter(torch.tensor(np.log(sigma_s), device=device).float())
            elif(init_method == "tprior"):
                print("Initialization using prior time.")
                alpha, beta, gamma, scaling, toff, u0, s0, sigma_u, sigma_s, T, Rscore = init_params(X,p,fit_scaling=True)
                t_prior = adata.obs[init_key].to_numpy()
                t_prior = t_prior[train_idx]
                std_t = np.std(t_prior)*0.2
                self.t_init = np.random.uniform(t_prior-std_t, t_prior+std_t)
                self.t_init -= self.t_init.min()
                self.t_init = self.t_init
                self.t_init = self.t_init/self.t_init.max()*tmax
                self.toff_init = get_ts_global(self.t_init, U/scaling, S, 95)
                self.alpha_init, self.beta_init, self.gamma_init, self.ton_init = reinit_params(U/scaling, S, self.t_init, self.toff_init)
                
                self.alpha = nn.Parameter(torch.tensor(np.stack([np.log(self.alpha_init), sigma_param*np.ones((G))]), device=device).float())
                self.beta = nn.Parameter(torch.tensor(np.stack([np.log(self.beta_init), sigma_param*np.ones((G))]), device=device).float())
                self.gamma = nn.Parameter(torch.tensor(np.stack([np.log(self.gamma_init), sigma_param*np.ones((G))]), device=device).float())
                self.scaling = nn.Parameter(torch.tensor(np.log(scaling), device=device).float())
                self.sigma_u = nn.Parameter(torch.tensor(np.log(sigma_u), device=device).float())
                self.sigma_s = nn.Parameter(torch.tensor(np.log(sigma_s), device=device).float())
                self.ton = nn.Parameter((torch.ones(G, device=device)*(-10)).float()) if init_ton_zero else nn.Parameter(torch.tensor(np.log(self.ton_init+1e-10), device=device).float())
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
                    if("init_t_quant" in kwargs):
                        self.t_init = np.quantile(T_eq,kwargs["init_t_quant"],1)
                    else:
                        self.t_init = np.quantile(T_eq,0.5,1)
                self.toff_init = get_ts_global(self.t_init, U/scaling, S, 95)
                self.alpha_init, self.beta_init, self.gamma_init, self.ton_init = reinit_params(U/scaling, S, self.t_init, self.toff_init)
                
                self.alpha = nn.Parameter(torch.tensor(np.stack([np.log(self.alpha_init), sigma_param*np.ones((G))]), device=device).float())
                self.beta = nn.Parameter(torch.tensor(np.stack([np.log(self.beta_init), sigma_param*np.ones((G))]), device=device).float())
                self.gamma = nn.Parameter(torch.tensor(np.stack([np.log(self.gamma_init), sigma_param*np.ones((G))]), device=device).float())
                self.scaling = nn.Parameter(torch.tensor(np.log(scaling), device=device).float())
                self.ton = nn.Parameter((torch.ones(G, device=device)*(-10)).float()) if init_ton_zero else nn.Parameter(torch.tensor(np.log(self.ton_init+1e-10), device=device).float())
                self.sigma_u = nn.Parameter(torch.tensor(np.log(sigma_u), device=device).float())
                self.sigma_s = nn.Parameter(torch.tensor(np.log(sigma_s), device=device).float())
        
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
            self.u0 = nn.Parameter(torch.tensor(np.log(U[cell_mask].mean(0)+1e-10), device=device).float())
            self.s0 = nn.Parameter(torch.tensor(np.log(S[cell_mask].mean(0)+1e-10), device=device).float())

        self.scaling.requires_grad = False
        self.sigma_u.requires_grad = False
        self.sigma_s.requires_grad = False
    
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
            Uhat, Shat = pred_su(F.leaky_relu(t - self.ton.exp(), neg_slope), 0, 0, rho*alpha, beta, gamma)
        else:
            if(condition is None):
                rho = torch.sigmoid(self.fc_out2(self.net_rho2(z)))
            else:
                rho = torch.sigmoid(self.fc_out2(self.net_rho2(torch.cat((z,condition),1))))
            Uhat, Shat = pred_su(F.leaky_relu(t-t0, neg_slope), u0/self.scaling.exp(), s0, rho*alpha, beta, gamma)
        Uhat = Uhat * torch.exp(self.scaling)
        return nn.functional.relu(Uhat), nn.functional.relu(Shat)
    
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
            Uhat, Shat = pred_su(F.leaky_relu(t-t0, neg_slope), u0/self.scaling.exp(), s0, rho*alpha, beta, gamma)
        Uhat = Uhat * torch.exp(self.scaling)
        return nn.functional.relu(Uhat), nn.functional.relu(Shat)

class VAEFullVB(VAE):
    """Full Variational Bayes
    This has an extra sampling of ODE parameters. Other parts are the same.
    Hence, we set it to be a subclass of VAE.
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
                 init_type=None,
                 tprior=None, 
                 init_ton_zero=True,
                 time_distribution="gaussian",
                 std_z_prior=0.01,
                 checkpoints=[None, None],
                 **kwargs):
        """Constructor of the class
        
        Arguments
        ---------
        
        adata : :class:`anndata.AnnData`
            AnnData object containing all relevant data and meta-data
        tmax : float
            Time range.
            This is used to restrict the cell time within certain range. In the 
            case of a Gaussian model without capture time, tmax/2 will be the mean of prior.
            If capture time is provided, then they are scaled to match a range of tmax.
            In the case of a uniform model, tmax is strictly the maximum time.
        dim_z : int
            Dimension of the latent cell state
        dim_cond : int, optional
            Dimension of additional information for the conditional VAE.
            Set to zero by default, equivalent to a VAE.
            This feature is not stable now.
        device : {'gpu','cpu'}, optional
            Training device
        hidden_size : tuple of int, optional
            Width of the hidden layers. Should be a tuple of the form
            (encoder layer 1, encoder layer 2, decoder layer 1, decoder layer 2)
        init_method : {'random', 'tprior', 'steady}, optional
            Initialization method. 
            Should choose from
            (1) random: random initialization
            (2) tprior: use the capture time to estimate rate parameters. Cell time will be 
                        randomly sampled with the capture time as the mean. The variance can
                        be controlled by changing 'time_overlap' in config.
            (3) steady: use the steady-state model to estimate gamma, alpha and assume beta = 1.
                        After this, a global cell time is estimated by taking the quantile over
                        all local times. Finally, rate parameters are reinitialized using the
                        global cell time.
        init_key : str, optional
            column in the AnnData object containing the capture time
        tprior : str, optional
            key in adata.obs that stores the capture time.
            Used for informative time prior
        init_type : str, optional
            The stem cell type. Used to estimated the initial conditions.
            This is not commonly used in practice and please consider leaving it to default.
        init_ton_zero : bool, optional
            Whether to add a non-zero switch-on time for each gene.
            It's set to True if there's no capture time.
        time_distribution : {'gaussian', 'uniform'}, optional
            Time distribution, set to Gaussian by default.
        std_z_prior : float, optional
            Standard deviation of the prior (isotropical Gaussian) of cell state.
        checkpoints : list of 2 strings, optional
            Contains the path to saved encoder and decoder models.
            Should be a .pt file.
        """
        try:
            U,S = adata.layers['Mu'], adata.layers['Ms']
        except KeyError:
            print('Unspliced/Spliced count matrices not found in the layers! Exit the program...')
        
        #Training Configuration
        self.config = {
            #Model Parameters
            "dim_z":dim_z,
            "tmax":tmax,
            "hidden_size":hidden_size,
            "init_method":init_method,
            "init_key":init_key,
            "tprior":tprior,
            "std_z_prior": std_z_prior,
            "tail":0.01,
            "time_overlap":0.5,
            "n_neighbors":10,
            "dt": (0.03,0.06),
            "n_bin": None,

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
            "kl_param":1.0,
            "reg_param":1.0,
            "neg_slope":0.0,
            "test_iter":None, 
            "save_epoch":100, 
            "n_warmup":5,
            "early_stop":5,
            "early_stop_thred":adata.n_vars*1e-3,
            "train_test_split":0.7,
            "k_alt":1, 
            "train_scaling":False, 
            "train_std":False, 
            "train_ton":True,
            "train_x0":False,
            "weight_sample":False,
            "sparsify":1
        }
        
        self.set_device(device)
        self.split_train_test(adata.n_obs)
        
        G = adata.n_vars
        self.dim_z = dim_z
        self.enable_cvae = dim_cond>0
        try:
            self.encoder = encoder(2*G, dim_z, dim_cond, hidden_size[0], hidden_size[1], self.device, checkpoint=checkpoints[0]).float()
        except IndexError:
            print('Please provide two dimensions!')
        
        self.decoder = decoder_fullvb(adata, 
                                      tmax, 
                                      self.train_idx,
                                      dim_z, 
                                      N1=hidden_size[2], 
                                      N2=hidden_size[3], 
                                      init_ton_zero=init_ton_zero,
                                      device=self.device, 
                                      init_method = init_method,
                                      init_key = init_key,
                                      init_type = init_type,
                                      checkpoint=checkpoints[1],
                                      **kwargs).float()
        self.tmax=tmax
        self.time_distribution = time_distribution
        self.get_prior(adata, time_distribution, tmax, tprior)
        
        self.p_z = torch.stack([torch.zeros(U.shape[0],dim_z), torch.ones(U.shape[0],dim_z)*self.config["std_z_prior"]]).float().to(self.device)
        
        self.use_knn = False
        self.u0 = None
        self.s0 = None
        self.t0 = None
        
        #Class attributes for training
        self.loss_train, self.loss_test = [],[]
        self.counter = 0 #Count the number of iterations
        self.n_drop = 0 #Count the number of consecutive iterations with little decrease in loss
        
        #Prior of Decoder Parameters
        self.p_log_alpha = torch.tensor([[0.0], [1.0]]).to(self.device)
        self.p_log_beta = torch.tensor([[0.0], [0.5]]).to(self.device)
        self.p_log_gamma = torch.tensor([[0.0], [0.5]]).to(self.device)
    
    def eval_model(self, data_in, u0=None, s0=None, t0=None, condition=None):
        """Evaluate the model. Here, we use the mean instead of randomly sample the ODE parameters.
        """
        data_in_scale = torch.cat((data_in[:,:data_in.shape[1]//2]/torch.exp(self.decoder.scaling), data_in[:,data_in.shape[1]//2:]),1)
        mu_t, std_t, mu_z, std_z = self.encoder.forward(data_in_scale)
         
        uhat, shat = self.decoder.eval_model(mu_t, mu_z, condition=condition, u0=u0, s0=s0, t0=t0, neg_slope=0.0)
        return mu_t, std_t, mu_z, std_z, uhat, shat    
    
    def vae_risk(self, 
                q_tx, p_t, 
                q_zx, p_z, 
                u, s, uhat, shat, 
                sigma_u, sigma_s, 
                weight=None):
        """Training objective function. This is the negative ELBO of the full VB. An additional KL term of decoder parameters is added.
        
        Arguments
        ---------
        
        q_tx : tuple of `torch.tensor`
            Parameters of time posterior. Mean and std are both (N, 1) tensors.
        p_t : tuple of `torch.tensor`
            Parameters of time prior.
        q_zx : tuple of `torch.tensor`
            Parameters of cell state posterior. Mean and std are both (N, Dz) tensors.
        p_z  : tuple of `torch.tensor`
            Parameters of cell state prior.
        u, s : `torch.tensor`
            Input data
        uhat, shat : torch.tensor
            Prediction by VeloVAE
        sigma_u, sigma_s : `torch.tensor`
            Standard deviation of the Gaussian noise
        weight : `torch.tensor`, optional
            Sample weight. This feature is not stable. Please consider setting it to None.
        
        Returns
        -------
        Negative ELBO : torch.tensor, scalar
        """
        kldt = self.kl_time(q_tx[0], q_tx[1], p_t[0], p_t[1], tail=self.config["tail"])
        kldz = kl_gaussian(q_zx[0], q_zx[1], p_z[0], p_z[1])
        kld_param = (kl_gaussian(self.decoder.alpha[0].view(1,-1), self.decoder.alpha[1].exp().view(1,-1), self.p_log_alpha[0], self.p_log_alpha[1]) + \
                     kl_gaussian(self.decoder.beta[0].view(1,-1), self.decoder.beta[1].exp().view(1,-1), self.p_log_beta[0], self.p_log_beta[1]) + \
                     kl_gaussian(self.decoder.gamma[0].view(1,-1), self.decoder.gamma[1].exp().view(1,-1), self.p_log_gamma[0], self.p_log_gamma[1]) ) / u.shape[0]
        
        #u and sigma_u has the original scale
        logp = -0.5*((u-uhat)/sigma_u).pow(2)-0.5*((s-shat)/sigma_s).pow(2)-torch.log(sigma_u)-torch.log(sigma_s*2*np.pi)
        
        if( weight is not None):
            logp = logp*weight
            
        err_rec = torch.mean(torch.sum(logp,1))
        
        return (- err_rec + self.config["kl_t"]*kldt + self.config["kl_z"]*kldz + self.config["kl_param"]*kld_param)
    
    def save_anndata(self, adata, key, file_path, file_name=None):
        """Save the ODE parameters and cell time to the anndata object and write it to disk.
        
        Arguments
        ---------
        adata : :class:`anndata.AnnData`
        key : str
            Used to store all parameters of the model.
        file_path : str
            Saving path.
        file_name : str, optional
            If set to a string ending with .h5ad, the updated anndata object will be written to disk.
        """
        self.set_mode('eval')
        os.makedirs(file_path, exist_ok=True)
        
        adata.var[f"{key}_logmu_alpha"] = self.decoder.alpha[0].detach().cpu().numpy()
        adata.var[f"{key}_logmu_beta"] = self.decoder.beta[0].detach().cpu().numpy()
        adata.var[f"{key}_logmu_gamma"] = self.decoder.gamma[0].detach().cpu().numpy()
        adata.var[f"{key}_logstd_alpha"] = self.decoder.alpha[1].detach().cpu().exp().numpy()
        adata.var[f"{key}_logstd_beta"] = self.decoder.beta[1].detach().cpu().exp().numpy()
        adata.var[f"{key}_logstd_gamma"] = self.decoder.gamma[1].detach().cpu().exp().numpy()
        adata.var[f"{key}_ton"] = self.decoder.ton.exp().detach().cpu().numpy()
        adata.var[f"{key}_scaling"] = np.exp(self.decoder.scaling.detach().cpu().numpy())
        adata.var[f"{key}_sigma_u"] = np.exp(self.decoder.sigma_u.detach().cpu().numpy())
        adata.var[f"{key}_sigma_s"] = np.exp(self.decoder.sigma_s.detach().cpu().numpy())
        U,S = adata.layers['Mu'], adata.layers['Ms']
        scaling = adata.var[f"{key}_scaling"].to_numpy()
        
        out, elbo = self.pred_all(np.concatenate((adata.layers['Mu'], adata.layers['Ms']),1), self.cell_labels, "both", gene_idx=np.array(range(adata.n_vars)))
        Uhat, Shat, t, std_t, z, std_z = out[0], out[1], out[2], out[3], out[4], out[5]

        adata.obs[f"{key}_time"] = t
        adata.obs[f"{key}_std_t"] = std_t
        adata.obsm[f"{key}_z"] = z
        adata.obsm[f"{key}_std_z"] = std_z
        adata.layers[f"{key}_uhat"] = Uhat
        adata.layers[f"{key}_shat"] = Shat
        sigma_u, sigma_s = adata.var[f"{key}_sigma_u"].to_numpy(), adata.var[f"{key}_sigma_s"].to_numpy()
        adata.var[f"{key}_likelihood"] = np.mean(-0.5*((adata.layers["Mu"]-Uhat)/sigma_u)**2-0.5*((adata.layers["Ms"]-Shat)/sigma_s)**2 - np.log(sigma_u) - np.log(sigma_s) - np.log(2*np.pi), 0)
        
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

        adata.uns[f"{key}_train_idx"] = self.train_idx
        adata.uns[f"{key}_test_idx"] = self.test_idx
        
        rna_velocity_vae(adata, key, use_raw=False, use_scv_genes=False, full_vb=True)
        
        if(file_name is not None):
            adata.write_h5ad(f"{file_path}/{file_name}")

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
from velovae.plotting import plot_sig, plot_sig_, plot_time, plot_train_loss, plot_test_loss

from .VAE import VAE
from .VAE import encoder as encoder_velovae
from .VAE import decoder as decoder_velovae
from .model_util import hist_equal, init_params, get_ts_global, reinit_params, convert_time, get_gene_index, optimal_transport_duality_gap, bound_prob
from .model_util import pred_su, ode, ode_numpy, knnx0, knnx0_alt, knnx0_bin
from .TransitionGraph import encode_type
from .TrainingData import SCData
from .VanillaVAE import VanillaVAE, kl_gaussian, kl_uniform
from .velocity import rna_velocity_vae

##############################################################
# VAE
##############################################################
class encoder(encoder_velovae):
    """
    Encoder of VeloVAE
    """
    def __init__(self, Cin, dim_z, dim_cond=0, N1=500, N2=250, device=torch.device('cpu'), checkpoint=None):
        """
        < Description >
        Constructor of the class
        
        < Input Arguments >
        1.  Cin [int]
            Input feature dimension. Usually just 2 x gene count
        
        2.  dim_z [int]
            Dimension of the latent cell state
            
        3.  dim_cond [int]
            (Optional) The user can optionally add the cell type information to the encoder.
        
        4.  N1 [int]
            (Optional) Width of the first hidden layer
        
        5.  N2 [int]
            (Optional) Width of the second hidden layer
        
        6.  device [torch.device]
            Either cpu or gpu device
        
        7.  checkpoint [string]
            Existing .pt file containing trained parameters
        
        < Output >
        None. Construct an instance of the class.
        """
        super(encoder, self).__init__(Cin, dim_z, dim_cond, N1, N2, device, None)
        self.fc_mu_l = nn.Linear(N2+dim_cond,1).to(device)
        self.fc_std_l, self.spt4 = nn.Linear(N2+dim_cond,1).to(device), nn.Softplus()
        
        if(checkpoint is not None):
            self.load_state_dict(torch.load(checkpoint,map_location=device))
        else:
            self.init_weights_l()

    def init_weights_l(self):
        """
        < Description >
        Initialize neural network weights.
        """
        for m in [self.fc_mu_l, self.fc_std_l]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)
    
    def forward(self, data_in, condition=None):
        """
        < Description >
        Forward propagation.
        
        < Input Arguments>
        1.  data_in [float/double tensor]
            Input data of size (N, D) = (Batch Size, Input Data Dimension)
        
        2.  condition [float/double tensor]
            (Optional) A tensor representing condition, with a size of (N, D_cond) = (Batch Size, Condition Dimension)
            Used in the conditional VAE
        
        < Output >
        1.  mu_tx [float/double tensor]
            Posterior mean of time, with a size of (N, 1)
            
        2.  std_tx [float/double tensor]
            Posterior standard deviation of time, with a size of (N, 1)
            
        3.  mu_zx [float/double tensor]
            Posterior mean of cell state, with a size of (N, D_z)
            
        4.  std_zx [float/double tensor]
            Posterior standard deviation of cell state, with a size of (N, D_z)
        
        5.  mu_lx [float/double tensor]
            Posterior mean of log library size, with a size of (N, 1)
            
        6.  std_lx [float/double tensor]
            Posterior standard deviation of log library size, with a size of (N, 1)
        """
        h = self.net(data_in)
        if(condition is not None):
            h = torch.cat((h,condition),1)
        mu_tx, std_tx = self.spt1(self.fc_mu_t(h)), self.spt2(self.fc_std_t(h))
        mu_zx, std_zx = self.fc_mu_z(h), self.spt3(self.fc_std_z(h))
        mu_l, std_l = self.fc_mu_l(h), self.spt4(self.fc_std_l(h))
        return mu_tx, std_tx, mu_zx, std_zx, mu_l, std_l

class decoder(decoder_velovae):
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
                 checkpoint=None):
        """
        < Description >
        Constructor of the class
        
        < Input Arguments >
        1.  adata [AnnData]
            Input AnnData object
        
        2.  tmax [float]
            Time range (hyperparameter)
            
        3.  train_idx [int array]
            The indices of all training samples. We pick 70% of the data as
            training samples by default.
        
        4.  dim_z [int]
            Dimension of the latent cell state
        
        5.  dim_cond [int]
            (Optional) Dimension of additional information for the conditional VAE

        6.  N1 [int]
            (Optional) Width of the first hidden layer
        
        7.  N2 [int]
            (Optional) Width of the second hidden layer
        
        8.  p [int in (0,100)]
            (Optional) Percentile threshold of u and s for picking steady-state cells.
            Used in initialization.
        
        9.  init_ton_zero [bool]
            (Optional) Whether to add a non-zero switch-on time for each gene.

        10. device [torch device]
            Either cpu or gpu
        
        11. init_method [string]
            (Optional) Initialization method. 
                        
        12. init_key [string]
            (Optional) column in the AnnData object containing the capture time
        
        13. init_type [string]
            (Optional) The stem cell type. Used to estimated the initial conditions.

        14. checkpoint [string]
            (Optional) Path to a file containing a pretrained model. If given, initialization
            will be skipped and arguments relating to initialization will be ignored.
        
        < Output >
        None. Construct an instance of the class.
        """
        super(decoder_velovae, self).__init__()
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
                self.ton = torch.nn.Parameter(torch.ones(adata.n_vars, device=device).float()*(-10))
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
                self.ton = nn.Parameter((torch.ones(adata.n_vars, device=device)*(-10)).float()) if init_ton_zero else nn.Parameter(torch.tensor(np.log(ton+1e-10), device=device).float())
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
                self.ton = nn.Parameter((torch.ones(adata.n_vars, device=device)*(-10)).float()) if init_ton_zero else nn.Parameter(torch.tensor(np.log(ton+1e-10), device=device).float())
        
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
        
        #Initialize NB variance
        self.sigma_u = torch.empty((adata.n_vars)).to(device)
        self.sigma_s = torch.empty((adata.n_vars)).to(device)
        for i in range(adata.n_vars):
            u, s = adata.layers["unspliced"][:,i].todense(), adata.layers["spliced"][:,i].todense()
            self.sigma_u[i] = float(np.log(np.std(u[train_idx])+1e-10))
            self.sigma_s[i] = float(np.log(np.std(s[train_idx])+1e-10))
        
        self.scaling.requires_grad = False
        
    
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
                 time_distribution="gaussian",
                 std_z_prior=0.01,
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
            "train_std":True, 
            "train_ton":False,
            "train_x0":False,
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
            self.encoder = encoder_velovae(2*G, dim_z, dim_cond, hidden_size[0], hidden_size[1], self.device, checkpoint=checkpoints[0]).float()
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
                               checkpoint=checkpoints[1]).float()
        self.tmax=tmax
        self.time_distribution = time_distribution
        self.get_prior(adata, time_distribution, tmax, tprior)
        
        self.p_z = torch.stack([torch.zeros(N,dim_z), torch.ones(N,dim_z)*self.config["std_z_prior"]]).float().to(self.device)
        
        #log_counts = np.log(adata.layers["unspliced"].sum(1)+adata.layers["spliced"].sum(1))
        #mu_l, std_l = np.mean(log_counts), np.std(log_counts)
        #self.p_l = torch.stack([torch.ones(N,1)*mu_l, torch.ones(N,1)*std_l]).float().to(self.device)
        cell_sizes = adata.obs["initial_size_spliced"].to_numpy() + adata.obs["initial_size_unspliced"].to_numpy()
        self.l = torch.tensor( cell_sizes / np.median(cell_sizes) ).unsqueeze(-1).float().to(self.device)
        
        self.use_knn = False
        self.u0 = None
        self.s0 = None
        self.t0 = None
        
        #Class attributes for training
        self.loss_train, self.loss_test = [],[]
        self.counter = 0 #Count the number of iterations
        self.n_drop = 0 #Count the number of consecutive iterations with little decrease in loss
    
    def forward(self, data_in, u0=None, s0=None, t0=None, condition=None):
        """
        < Description >
        Standard forward pass.
        
        < Input Arguments >
        1.  data_in [tensor (N, 2G)]
            input count data
        2.  u0 [tensor (N, G)]
            (Optional) Initial condition of u
            This is set to None in the first stage when cell time is 
            not fixed. It will have some value in the second stage, so the users
            shouldn't worry about feeding the parameter themselves.
        
        3.  s0 [tensor (N, G)]
            (Optional) Initial condition of s
        
        4.  t0 [tensor (N, 1)]
            (Optional) time at the initial condition
        
        < Output >
        1.  mu_t [tensor (N, 1)]
            time mean
        
        2.  std_t [tensor (N, 1)]
            time standard deviation
        
        3.  mu_z [tensor (N, G)]
            cell state mean
        
        4.  std_z [tensor (N, G)]
            cell state standard deviation
        
        5.  mu_l [tensor (N, 1)]
            log library size mean
        
        6.  std_l [tensor (N,G)]
            log library size standard deviation
        
        7.  t [tensor (N, 1)]
            sampled cell time
        
        8.  z [tensor (N, G)]
            sampled cell sate
        
        9.  l [tensor (N, 1)]
            sampled log library size
        
        10.  uhat [tensor (N,G)]
            predicted mean u values
        
        11.  shat [tensor (N,G)]
            predicted mean s values
        """
        data_in_scale = torch.cat((data_in[:,:data_in.shape[1]//2]/torch.exp(self.decoder.scaling), data_in[:,data_in.shape[1]//2:]),1)
        mu_t, std_t, mu_z, std_z = self.encoder.forward(data_in_scale, condition)
        t = self.sample(mu_t, std_t)
        z = self.sample(mu_z, std_z)
        #l = torch.exp(self.sample(enc_out[4], enc_out[5])) if len(enc_out==6) else self.l
         
        uhat, shat = self.decoder.forward(t, z, condition, u0, s0, t0, neg_slope=self.config["neg_slope"])
        return mu_t, std_t, mu_z, std_z, t, z, uhat, shat
    
    def eval_model(self, data_in, u0=None, s0=None, t0=None, condition=None, continuous=True):
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
        data_in_scale = torch.cat((data_in[:,:data_in.shape[1]//2]/torch.exp(self.decoder.scaling), data_in[:,data_in.shape[1]//2:]),1)
        mu_t, std_t, mu_z, std_z = self.encoder.forward(data_in_scale, condition)
         
        uhat, shat = self.decoder.forward(mu_t, mu_z, condition, u0=u0, s0=s0, t0=t0, neg_slope=0.0)
        if(not continuous):
            #sigma_u = torch.exp(self.decoder.sigma_u)
            #sigma_s = torch.exp(self.decoder.sigma_s)
            #nb_u = NegativeBinomial(total_count=(uhat/sigma_u).pow(2), probs=(l*sigma_u.pow(2))/(uhat+l*sigma_u.pow(2)+eps))
            #nb_s = NegativeBinomial(total_count=(shat/sigma_s).pow(2), probs=(l*sigma_s.pow(2))/(shat+l*sigma_s.pow(2)+eps))
            poisson_u = Poisson(F.softplus(uhat, beta=100))
            poisson_s = Poisson(F.softplus(shat, beta=100))
            u_out = poisson_u.sample()
            s_out = poisson_s.sample()
            return mu_t, std_t, mu_z, std_z, u_out, s_out
        return mu_t, std_t, mu_z, std_z, uhat, shat
    
    def vae_risk(self, 
                 q_tx, p_t, 
                 q_zx, p_z, 
                 u, s, uhat, shat, 
                 sigma_u, sigma_s,
                 l,
                 weight=None,
                 eps=1e-6):
        """
        < Description >
        Training objective function. This is the negative ELBO.
        
        < Input Arguments >
        1.  q_tx [a tuple of tensors (mean, standard deviation)]
            Parameters of time posterior. Mean and std are both (N, 1) tensors.
        
        2.  p_t [a tuple of tensors (mean, standard deviation)]
            Parameters of time prior.
        
        3.  q_zx [a tuple of tensors (mean, standard deviation)]
            Parameters of cell state posterior. Mean and std are both (N, Dz) tensors.
        
        4.  p_z [a tuple of tensors (mean, standard deviation)]
            Parameters of cell state prior.
        
        5.  u, s [tensor (B,G)]
            Input data.
        
        6.  uhat, shat [tensor (B,G)]
            Prediction by VeloVAE
        
        7.  sigma_u, sigma_s [tensor (G)]
            Standard deviation of the Gaussian noise
        
        8.  weight [tensor (N,1)]
            (Optional) Sample weight. 
            This feature is not stable. Please consider setting it to None.
        
        < Output >
        1.  Negative ELBO [tensor scalar]
        """
        kldt = self.kl_time(q_tx[0], q_tx[1], p_t[0], p_t[1], tail=self.config["tail"])
        kldz = kl_gaussian(q_zx[0], q_zx[1], p_z[0], p_z[1])
        #kldl = kl_gaussian(q_lx[0], q_lx[1], p_l[0], p_l[1])
        
        #negative binomial log likelihood
        """
        p_u = bound_prob((l*sigma_u.pow(2))/(uhat+l*sigma_u.pow(2)))
        p_s = bound_prob((l*sigma_s.pow(2))/(shat+l*sigma_s.pow(2)))
        assert torch.all( (p_u > 0) & (p_u < 1) )
        assert torch.all( (p_s > 0) & (p_s < 1) )
        nb_u = NegativeBinomial(total_count=((uhat+eps)/sigma_u).pow(2), probs=p_u)
        nb_s = NegativeBinomial(total_count=((shat+eps)/sigma_s).pow(2), probs=p_s)
        logp = nb_u.log_prob(u) + nb_s.log_prob(s)
        """
        
        #poisson
        poisson_u = Poisson(l*F.softplus(uhat, beta=100))
        poisson_s = Poisson(l*F.softplus(shat, beta=100))
        logp = poisson_u.log_prob(u) + poisson_s.log_prob(s)
        
        if( weight is not None):
            logp = logp*weight
            
        err_rec = torch.mean(torch.sum(logp,1))
        
        return (- err_rec + self.config["kl_t"]*kldt + self.config["kl_z"]*kldz ) #+ self.config["kl_l"]*kldl)
    
    def train_epoch(self, train_loader, test_set, optimizer, optimizer2=None, K=1):
        """
        < Description >
        Training in each epoch.
        Early stopping if enforced by default. 
        
        < Input Arguments >
        1.  train_loader [torch.utils.data.DataLoader]
            Data loader of the input data.
        
        2.  test_set [torch.utils.data.Dataset]
            Validation dataset
        
        3.  optimizer  [optimizer from torch.optim]
        
        4.  optimizer2 [optimizer from torch.optim]
            (Optional) A second optimizer.
            This is used when we optimize NN and ODE simultaneously in one epoch.
            By default, VeloVAE performs alternating optimization in each epoch.
            The argument will be set to proper value automatically.
        
        5.  K [int]
            Alternating update period.
            For every K updates of optimizer, there's one update for optimizer2.
            If set to 0, optimizer2 will be ignored and only optimizer will be
            updated. Users can set it to 0 if they want to update sorely NN in one 
            epoch and ODE in the next epoch. 
        
        < Output >
        1.  stop_training [bool]
            Whether to stop training based on the early stopping criterium.
        """
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
                                 self.U_raw[idx], self.S_raw[idx], 
                                 uhat, shat, 
                                 torch.exp(self.decoder.sigma_u), torch.exp(self.decoder.sigma_s), 
                                 self.l[idx],
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
        self.U_raw = torch.tensor(U_raw, dtype=torch.int).to(self.device)
        self.S_raw = torch.tensor(S_raw, dtype=torch.int).to(self.device)
        super(DVAE, self).train(adata, 
                                config, 
                                plot, 
                                gene_plot, 
                                cluster_key,
                                figure_path, 
                                embed,
                                use_raw=False)
    
    def pred_all(self, data, cell_labels, mode='test', output=["uhat", "shat", "t", "z"], gene_idx=None, continuous=True):
        """
        < Description >
        Generate all predictions.
        
        < Input Arguments >
        1.  data [array (N, 2G)] : 
            Input mRNA count
        
        2.  mode [string]
            train or test or both
            
        3.  output [list of string]
            (Optional) variables to compute
        
        4.  gene_idx [int array/list]
            (Optional) gene index, used for reducing unnecessary memory usage
        
        < Output >
        1.  out [tuple of array]
            Depends on the input argument 'output'.
            Can return predicted u,s,time and cell state
        
        2. ELBO [float]
        """
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
        """
        if("l" in output):
            l_out = np.zeros((N))
            std_l_out = np.zeros((N))
        """
        
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
                    #p_l = self.p_l[:,self.test_idx[i*B:(i+1)*B],:]
                    l = self.l[self.test_idx[i*B:(i+1)*B],:]
                    mu_tx, std_tx, mu_zx, std_zx, uhat, shat = self.eval_model(data_in, u0, s0, t0, y_onehot, continuous)
                elif(mode=="train"):
                    u0 = torch.tensor(self.u0[self.train_idx[i*B:(i+1)*B]], dtype=torch.float, device=self.device) if self.use_knn else None
                    s0 = torch.tensor(self.s0[self.train_idx[i*B:(i+1)*B]], dtype=torch.float, device=self.device) if self.use_knn else None
                    t0 = torch.tensor(self.t0[self.train_idx[i*B:(i+1)*B]], dtype=torch.float, device=self.device) if self.use_knn else None
                    y_onehot = F.one_hot(torch.tensor(cell_labels[self.train_idx[i*B:(i+1)*B]], dtype=int, device=self.device), self.n_type).float() if self.enable_cvae else None
                    p_t = self.p_t[:,self.train_idx[i*B:(i+1)*B],:]
                    p_z = self.p_z[:,self.train_idx[i*B:(i+1)*B],:]
                    #p_l = self.p_l[:,self.train_idx[i*B:(i+1)*B],:]
                    l = self.l[self.train_idx[i*B:(i+1)*B],:]
                    mu_tx, std_tx, mu_zx, std_zx, uhat, shat = self.eval_model(data_in, u0, s0, t0, y_onehot, continuous)
                else:
                    u0 = torch.tensor(self.u0[i*B:(i+1)*B], dtype=torch.float, device=self.device) if self.use_knn else None
                    s0 = torch.tensor(self.s0[i*B:(i+1)*B], dtype=torch.float, device=self.device) if self.use_knn else None
                    t0 = torch.tensor(self.t0[i*B:(i+1)*B], dtype=torch.float, device=self.device) if self.use_knn else None
                    y_onehot = F.one_hot(torch.tensor(cell_labels[i*B:(i+1)*B], dtype=int, device=self.device), self.n_type).float() if self.enable_cvae else None
                    p_t = self.p_t[:,i*B:(i+1)*B,:]
                    p_z = self.p_z[:,i*B:(i+1)*B,:]
                    #p_l = self.p_l[:,i*B:(i+1)*B,:]
                    l = self.l[i*B:(i+1)*B,:]
                    mu_tx, std_tx, mu_zx, std_zx, uhat, shat = self.eval_model(data_in, u0, s0, t0, y_onehot, continuous)
                
                loss = self.vae_risk((mu_tx, std_tx), p_t,
                                     (mu_zx, std_zx), p_z,
                                     data_in[:,:G], data_in[:,G:], 
                                     uhat, shat, 
                                     torch.exp(self.decoder.sigma_u), torch.exp(self.decoder.sigma_s), 
                                     l,
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
                if("l" in output):
                    l_out[i*B:(i+1)*B] = mu_lx.cpu().numpy()
                    std_l_out[i*B:(i+1)*B] = std_lx.cpu().numpy()
            if(N > B*Nb):
                data_in = torch.tensor(data[Nb*B:]).float().to(self.device)
                if(mode=="test"):
                    u0 = torch.tensor(self.u0[self.test_idx[Nb*B:]], dtype=torch.float, device=self.device) if self.use_knn else None
                    s0 = torch.tensor(self.s0[self.test_idx[Nb*B:]], dtype=torch.float, device=self.device) if self.use_knn else None
                    t0 = torch.tensor(self.t0[self.test_idx[Nb*B:]], dtype=torch.float, device=self.device) if self.use_knn else None
                    y_onehot = F.one_hot(torch.tensor(cell_labels[self.test_idx[Nb*B:]], dtype=int, device=self.device), self.n_type).float() if self.enable_cvae else None
                    p_t = self.p_t[:,self.test_idx[Nb*B:],:]
                    p_z = self.p_z[:,self.test_idx[Nb*B:],:]
                    #p_l = self.p_l[:,self.test_idx[Nb*B:],:]
                    l = self.l[self.test_idx[Nb*B:],:]
                    mu_tx, std_tx, mu_zx, std_zx, uhat, shat = self.eval_model(data_in, u0, s0, t0, y_onehot)
                elif(mode=="train"):
                    u0 = torch.tensor(self.u0[self.train_idx[Nb*B:]], dtype=torch.float, device=self.device) if self.use_knn else None
                    s0 = torch.tensor(self.s0[self.train_idx[Nb*B:]], dtype=torch.float, device=self.device) if self.use_knn else None
                    t0 = torch.tensor(self.t0[self.train_idx[Nb*B:]], dtype=torch.float, device=self.device) if self.use_knn else None
                    y_onehot = F.one_hot(torch.tensor(cell_labels[self.train_idx[Nb*B:]], dtype=int, device=self.device), self.n_type).float() if self.enable_cvae else None
                    p_t = self.p_t[:,self.train_idx[Nb*B:],:]
                    p_z = self.p_z[:,self.train_idx[Nb*B:],:]
                    #p_l = self.p_l[:,self.train_idx[Nb*B:],:]
                    l = self.l[self.train_idx[Nb*B:],:]
                    mu_tx, std_tx, mu_zx, std_zx, uhat, shat = self.eval_model(data_in, u0, s0, t0, y_onehot)
                else:
                    u0 = torch.tensor(self.u0[Nb*B:], dtype=torch.float, device=self.device) if self.use_knn else None
                    s0 = torch.tensor(self.s0[Nb*B:], dtype=torch.float, device=self.device) if self.use_knn else None
                    t0 = torch.tensor(self.t0[Nb*B:], dtype=torch.float, device=self.device) if self.use_knn else None
                    y_onehot = F.one_hot(torch.tensor(cell_labels[Nb*B:], dtype=int, device=self.device), self.n_type).float() if self.enable_cvae else None
                    p_t = self.p_t[:,Nb*B:,:]
                    p_z = self.p_z[:,Nb*B:,:]
                    #p_l = self.p_l[:,Nb*B:,:]
                    l = self.l[Nb*B:,:]
                    mu_tx, std_tx, mu_zx, std_zx, uhat, shat = self.eval_model(data_in, u0, s0, t0, y_onehot)
                loss = self.vae_risk((mu_tx, std_tx), p_t,
                                     (mu_zx, std_zx), p_z,
                                     data_in[:,:G], data_in[:,G:], 
                                     uhat, shat, 
                                     torch.exp(self.decoder.sigma_u), torch.exp(self.decoder.sigma_s), 
                                     l,
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
                """
                if("l" in output):
                    l_out[Nb*B:] = mu_lx.cpu().numpy()
                    std_l_out[Nb*B:] = std_lx.cpu().numpy()
                """
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
        """
        if("l" in output):
            out.append(l_out)
            out.append(std_l_out)
        """
        return out, elbo.cpu().item()
    
    def save_anndata(self, adata, key, file_path, file_name=None):
        """
        < Description >
        Save the ODE parameters and cell time to the anndata object and write it to disk.
        
        < Input Arguments >
        1.  adata [AnnData]
        
        2.  key [string]
            Used to store all parameters of the model.
        
        3.  file_path [string]
            Saving path.
        
        4.  file_name [string]
            (Optional) If set to a string ending with .h5ad, the updated anndata
            object will be written to disk.
        
        < Output >
        None
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
        U,S = np.array(adata.layers['unspliced'].todense()), np.array(adata.layers['spliced'].todense())
        scaling = adata.var[f"{key}_scaling"].to_numpy()
        
        out, elbo = self.pred_all(np.concatenate((np.array(adata.layers['unspliced'].todense()), np.array(adata.layers['spliced'].todense())),1), self.cell_labels, "both", gene_idx=np.array(range(adata.n_vars)))
        Uhat, Shat, t, std_t, z, std_z = out[0], out[1], out[2], out[3], out[4], out[5]

        adata.obs[f"{key}_time"] = t
        adata.obs[f"{key}_std_t"] = std_t
        adata.obsm[f"{key}_z"] = z
        adata.obsm[f"{key}_std_z"] = std_z
        #adata.obs[f"{key}_l"] = l
        #adata.obs[f"{key}_std_l"] = std_l
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
        
        #u0, s0, t0 = self.update_x0(adata.layers['Mu'], adata.layers['Ms'], self.config["n_bin"])
        adata.obs[f"{key}_t0"] = self.t0.squeeze()
        adata.layers[f"{key}_u0"] = self.u0
        adata.layers[f"{key}_s0"] = self.s0

        adata.uns[f"{key}_train_idx"] = self.train_idx
        adata.uns[f"{key}_test_idx"] = self.test_idx
        
        rna_velocity_vae(adata, key, use_raw=False, use_scv_genes=False)
        
        if(file_name is not None):
            adata.write_h5ad(f"{file_path}/{file_name}")
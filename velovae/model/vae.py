"""VAE Module

This module contains the VeloVAE model implementation.
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions.negative_binomial import NegativeBinomial
from torch.distributions.poisson import Poisson
import time
from velovae.plotting import plot_sig, plot_time
from velovae.plotting import plot_train_loss, plot_test_loss

from .model_util import hist_equal, init_params, get_ts_global, reinit_params
from .model_util import convert_time, get_gene_index
from .model_util import pred_su, knnx0_index, get_x0
from .model_util import elbo_collapsed_categorical
from .model_util import assign_gene_mode, assign_gene_mode_tprior, find_dirichlet_param
from .model_util import get_cell_scale, get_dispersion

from .transition_graph import encode_type
from .training_data import SCData
from .vanilla_vae import VanillaVAE, kl_gaussian
from .velocity import rna_velocity_vae
P_MAX = 1e4
GRAD_MAX = 1e7


##############################################################
# VAE
##############################################################
class encoder(nn.Module):
    """Encoder class for the VAE model
    """
    def __init__(self,
                 Cin,
                 dim_z,
                 dim_cond=0,
                 N1=500,
                 N2=250):
        """Constructor of the encoder class

        Args:
            Cin (int): Input feature dimension. We assume inputs are batches of 1-d tensors.
            dim_z (int): Latent dimension of the cell state.
            dim_cond (int, optional): Dimension of condition features, used for a conditional VAE. Defaults to 0.
            N1 (int, optional): Width of the first hidden layer. Defaults to 500.
            N2 (int, optional): Width of the second hidden layer. Defaults to 250.
        """
        super(encoder, self).__init__()
        self.fc1 = nn.Linear(Cin, N1)
        self.bn1 = nn.BatchNorm1d(num_features=N1)
        self.dpt1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(N1, N2)
        self.bn2 = nn.BatchNorm1d(num_features=N2)
        self.dpt2 = nn.Dropout(p=0.2)

        self.net = nn.Sequential(self.fc1, self.bn1, nn.LeakyReLU(), self.dpt1,
                                 self.fc2, self.bn2, nn.LeakyReLU(), self.dpt2)

        self.fc_mu_t = nn.Linear(N2+dim_cond, 1)
        self.spt1 = nn.Softplus()
        self.fc_std_t = nn.Linear(N2+dim_cond, 1)
        self.spt2 = nn.Softplus()
        self.fc_mu_z = nn.Linear(N2+dim_cond, dim_z)
        self.fc_std_z = nn.Linear(N2+dim_cond, dim_z)
        self.spt3 = nn.Softplus()

        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in [self.fc_mu_t, self.fc_std_t, self.fc_mu_z, self.fc_std_z]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, data_in, condition=None):
        """Forward function of the encoder

        Args:
            data_in (torch.Tensor): Input cell-by-gene count data.
            condition (torch.Tensor, optional): Condition features. Defaults to None.

        Returns:
            tuple:

                - :class:`torch.Tensor`: Mean of time variational posterior.

                - :class:`torch.Tensor`: Standard deviation of time variational posterior.

                - :class:`torch.Tensor`: Mean of cell state variational posterior.

                - :class:`torch.Tensor`: Standard deviation of cell state variational posterior.
        """
        h = self.net(data_in)
        if condition is not None:
            h = torch.cat((h, condition), 1)
        mu_tx, std_tx = self.spt1(self.fc_mu_t(h)), self.spt2(self.fc_std_t(h))
        mu_zx, std_zx = self.fc_mu_z(h), self.spt3(self.fc_std_z(h))
        return mu_tx, std_tx, mu_zx, std_zx


class decoder(nn.Module):
    def __init__(self,
                 adata,
                 tmax,
                 train_idx,
                 dim_z,
                 full_vb=False,
                 discrete=False,
                 dim_cond=0,
                 N1=250,
                 N2=500,
                 p=98,
                 init_ton_zero=False,
                 filter_gene=False,
                 device=torch.device('cpu'),
                 init_method="steady",
                 init_key=None,
                 checkpoint=None,
                 **kwargs):
        """Constructor of the VeloVAE decoder module

        Args:
            adata (:class:`anndata.AnnData`):
                Input AnnData object containing spliced and unspliced count matrices
                and other cell and gene annotations.
            tmax (float):
                Time range.
            train_idx (numpy array like):
                1D array containing the indices of training samples.
            dim_z (int):
                Latent cell state dimension.
            full_vb (bool, optional):
                Whether to use the full variational Bayes feature. Defaults to False.
            discrete (bool, optional):
                Whether to directly model discrete counts. Defaults to False.
            dim_cond (int, optional):
                Dimension of any condition features. Defaults to 0.
            N1 (int, optional):
                Width of the first hidden layer. Defaults to 250.
            N2 (int, optional):
                Width of the second hidden layer. Defaults to 500.
            p (int, optional):
                Percentile value, used in picking steady-state cells
                in model initialization. Defaults to 98.
            init_ton_zero (bool, optional):
                Whether to assume zero switch-on time for each gene. Defaults to False.
            filter_gene (bool, optional):
                Whether to filter out non-velocity genes based on scVelo-style initialization.
                Defaults to False.
            device (torch.device, optional):
                Device to hold the model. Defaults to torch.device('cpu').
            init_method (str, optional):
                Initialization method, should be one of {'steady', 'tprior'}. Defaults to "steady".
            init_key (str, optional):
                Key in adata.obs storing the capture time or any prior time information. Defaults to None.
            checkpoint (str, optional):
                (ToDo) Previously stored parameter values to load. Defaults to None.
        """
        super(decoder, self).__init__()
        G = adata.n_vars
        self.tmax = tmax
        self.is_full_vb = full_vb
        self.is_discrete = discrete

        if checkpoint is None:
            # Get dispersion and library size factor for the discrete model
            if discrete:
                U, S = adata.layers['unspliced'].A.astype(float), adata.layers['spliced'].A.astype(float)
                # Dispersion
                mean_u, mean_s, dispersion_u, dispersion_s = get_dispersion(U[train_idx], S[train_idx])
                adata.var["mean_u"] = mean_u
                adata.var["mean_s"] = mean_s
                adata.var["dispersion_u"] = dispersion_u
                adata.var["dispersion_s"] = dispersion_s
                lu, ls = get_cell_scale(U, S, train_idx, True, 50)
                adata.obs["library_scale_u"] = lu
                adata.obs["library_scale_s"] = ls
                scaling_discrete = np.std(U[train_idx], 0) / (np.std(S[train_idx, 0]) + 1e-16)
            # Get the ODE parameters
            U, S = adata.layers['Mu'][train_idx], adata.layers['Ms'][train_idx]
            X = np.concatenate((U, S), 1)
            # Dynamical Model Parameters
            (alpha, beta, gamma,
             scaling,
             toff,
             u0, s0,
             sigma_u, sigma_s,
             T,
             gene_score) = init_params(X, p, fit_scaling=True)
            gene_mask = (gene_score == 1.0)
            if discrete:
                scaling = np.clip(scaling_discrete, 1e-6, None)
            if filter_gene:
                adata._inplace_subset_var(gene_mask)
                U, S = U[:, gene_mask], S[:, gene_mask]
                G = adata.n_vars
                alpha = alpha[gene_mask]
                beta = beta[gene_mask]
                gamma = gamma[gene_mask]
                scaling = scaling[gene_mask]
                toff = toff[gene_mask]
                u0 = u0[gene_mask]
                s0 = s0[gene_mask]
                sigma_u = sigma_u[gene_mask]
                sigma_s = sigma_s[gene_mask]
                T = T[:, gene_mask]
            if init_method == "random":
                print("Random Initialization.")
                self.alpha = nn.Parameter(torch.normal(0.0, 0.01, size=(G,), device=device).float())
                self.beta = nn.Parameter(torch.normal(0.0, 0.01, size=(G,), device=device).float())
                self.gamma = nn.Parameter(torch.normal(0.0, 0.01, size=(G,), device=device).float())
                self.ton = torch.nn.Parameter(torch.ones(G, device=device).float()*(-10))
                self.t_init = None
            elif init_method == "tprior":
                print("Initialization using prior time.")
                t_prior = adata.obs[init_key].to_numpy()
                t_prior = t_prior[train_idx]
                std_t = (np.std(t_prior)+1e-3)*0.05
                self.t_init = np.random.uniform(t_prior-std_t, t_prior+std_t)
                self.t_init -= self.t_init.min()
                self.t_init = self.t_init
                self.t_init = self.t_init/self.t_init.max()*tmax
                self.toff_init = get_ts_global(self.t_init, U/scaling, S, 95)
                self.alpha_init, self.beta_init, self.gamma_init, self.ton_init = reinit_params(U/scaling,
                                                                                                S,
                                                                                                self.t_init,
                                                                                                self.toff_init)

                self.alpha = nn.Parameter(torch.tensor(np.log(self.alpha_init), device=device).float())
                self.beta = nn.Parameter(torch.tensor(np.log(self.beta_init), device=device).float())
                self.gamma = nn.Parameter(torch.tensor(np.log(self.gamma_init), device=device).float())
                self.ton = (nn.Parameter((torch.ones(G, device=device)*(-10)).float())
                            if init_ton_zero else
                            nn.Parameter(torch.tensor(np.log(self.ton_init+1e-10), device=device).float()))
            else:
                print("Initialization using the steady-state and dynamical models.")
                if init_key is not None:
                    self.t_init = adata.obs[init_key].to_numpy()[train_idx]
                else:
                    T = T+np.random.rand(T.shape[0], T.shape[1]) * 1e-3
                    T_eq = np.zeros(T.shape)
                    n_bin = T.shape[0]//50+1
                    for i in range(T.shape[1]):
                        T_eq[:, i] = hist_equal(T[:, i], tmax, 0.9, n_bin)
                    if "init_t_quant" in kwargs:
                        self.t_init = np.quantile(T_eq, kwargs["init_t_quant"], 1)
                    else:
                        self.t_init = np.quantile(T_eq, 0.5, 1)
                self.toff_init = get_ts_global(self.t_init, U/scaling, S, 95)
                self.alpha_init, self.beta_init, self.gamma_init, self.ton_init = reinit_params(U/scaling,
                                                                                                S,
                                                                                                self.t_init,
                                                                                                self.toff_init)

                self.alpha = nn.Parameter(torch.tensor(np.log(self.alpha_init), device=device).float())
                self.beta = nn.Parameter(torch.tensor(np.log(self.beta_init), device=device).float())
                self.gamma = nn.Parameter(torch.tensor(np.log(self.gamma_init), device=device).float())
                self.ton = (nn.Parameter((torch.ones(adata.n_vars, device=device)*(-10)).float())
                            if init_ton_zero else
                            nn.Parameter(torch.tensor(np.log(self.ton_init+1e-10), device=device).float()))
            self.register_buffer('scaling', torch.tensor(np.log(scaling), device=device).float())
            # Add Gaussian noise in case of continuous model
            if not discrete:
                self.register_buffer('sigma_u', torch.tensor(np.log(sigma_u), device=device).float())
                self.register_buffer('sigma_s', torch.tensor(np.log(sigma_s), device=device).float())

            # Add variance in case of full vb
            if full_vb:
                sigma_param = np.log(0.05) * torch.ones(adata.n_vars, device=device)
                self.alpha = nn.Parameter(torch.stack([self.alpha, sigma_param]).float())
                self.beta = nn.Parameter(torch.stack([self.beta, sigma_param]).float())
                self.gamma = nn.Parameter(torch.stack([self.gamma, sigma_param]).float())

            self.u0 = nn.Parameter(torch.tensor(np.log(u0+1e-10), device=device).float())
            self.s0 = nn.Parameter(torch.tensor(np.log(s0+1e-10), device=device).float())

        # Gene dyncamical mode initialization
        if init_method == 'tprior':
            w = assign_gene_mode_tprior(adata, init_key, train_idx)
        else:
            dyn_mask = (T > tmax*0.01) & (np.abs(T-toff) > tmax*0.01)
            w = np.sum(((T < toff) & dyn_mask), 0) / (np.sum(dyn_mask, 0) + 1e-10)
            assign_type = kwargs['assign_type'] if 'assign_type' in kwargs else 'auto'
            thred = kwargs['ks_test_thred'] if 'ks_test_thred' in kwargs else 0.05
            n_cluster_thred = kwargs['n_cluster_thred'] if 'n_cluster_thred' in kwargs else 3
            std_prior = kwargs['std_alpha_prior'] if 'std_alpha_prior' in kwargs else 0.1
            if 'reverse_gene_mode' in kwargs:
                w = (1 - assign_gene_mode(adata, w, assign_type, thred, std_prior, n_cluster_thred)
                     if kwargs['reverse_gene_mode'] else
                     assign_gene_mode(adata, w, assign_type, thred, std_prior, n_cluster_thred))
            else:
                w = assign_gene_mode(adata, w, assign_type, thred, std_prior, n_cluster_thred)
        print(f"Initial induction: {np.sum(w >= 0.5)}, repression: {np.sum(w < 0.5)}/{G}")
        adata.var["w_init"] = w
        logit_pw = 0.5*(np.log(w+1e-10) - np.log(1-w+1e-10))
        logit_pw = np.stack([logit_pw, -logit_pw], 1)
        self.logit_pw = nn.Parameter(torch.tensor(logit_pw, device=device).float())

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

        if checkpoint is not None:
            self.alpha = nn.Parameter(torch.empty(G, device=device).float())
            self.beta = nn.Parameter(torch.empty(G, device=device).float())
            self.gamma = nn.Parameter(torch.empty(G, device=device).float())
            self.scaling = nn.Parameter(torch.empty(G, device=device).float())
            self.ton = nn.Parameter(torch.empty(G, device=device).float())
            if not discrete:
                self.sigma_u = nn.Parameter(torch.empty(G, device=device).float())
                self.sigma_s = nn.Parameter(torch.empty(G, device=device).float())

            self.load_state_dict(torch.load(checkpoint, map_location=device))
        else:
            self._init_weights()

    def _init_weights(self, net_id=None):
        if net_id == 1 or net_id is None:
            for m in self.net_rho.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            nn.init.xavier_normal_(self.fc_out1.weight)
            nn.init.constant_(self.fc_out1.bias, 0.0)
        if net_id == 2 or net_id is None:
            for m in self.net_rho2.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            nn.init.xavier_normal_(self.fc_out2.weight)
            nn.init.constant_(self.fc_out2.bias, 0.0)

    def _sample_ode_param(self, random=True):
        """
        Sample rate parameters for full vb or output fixed rate parameters.
            The function outputs fixed rate parameters when
            (1) random is set to False
            (2) full vb is not enabled

        Args:
            random (bool, optional): 
                Whether to randomly sample parameters from their posterior distributions. Defaults to True.

        Returns:
            :class:`torch.Tensor`: Sampled or fixed rate parameters
        """
        ####################################################
        # Sample rate parameters for full vb or
        # output fixed rate parameters when
        # (1) random is set to False
        # (2) full vb is not enabled
        ####################################################
        if self.is_full_vb:
            if random:
                eps = torch.normal(mean=torch.zeros((3, self.alpha.shape[1]), device=self.alpha.device),
                                   std=torch.ones((3, self.alpha.shape[1]), device=self.alpha.device))
                alpha = torch.exp(self.alpha[0] + eps[0]*(self.alpha[1].exp()))
                beta = torch.exp(self.beta[0] + eps[1]*(self.beta[1].exp()))
                gamma = torch.exp(self.gamma[0] + eps[2]*(self.gamma[1].exp()))
                # self._eps = eps
            else:
                alpha = self.alpha[0].exp()
                beta = self.beta[0].exp()
                gamma = self.gamma[0].exp()
        else:
            alpha = self.alpha.exp()
            beta = self.beta.exp()
            gamma = self.gamma.exp()
        return alpha, beta, gamma

    def _clip_rate(self, rate, max_val):
        clip_fn = nn.Hardtanh(-16, np.log(max_val))
        return clip_fn(rate)

    def _forward_basis(self, t, z, condition=None, eval_mode=False, neg_slope=0.0):
        if condition is None:
            rho = torch.sigmoid(self.fc_out1(self.net_rho(z)))
        else:
            rho = torch.sigmoid(self.fc_out1(self.net_rho(torch.cat((z, condition), 1))))
        zero_mtx = torch.zeros(rho.shape, device=rho.device, dtype=float)
        zero_vec = torch.zeros(self.u0.shape, device=rho.device, dtype=float)
        alpha, beta, gamma = self._sample_ode_param(random=not eval_mode)
        # tensor shape (n_cell, 2, n_gene)
        alpha = torch.stack([alpha*rho, zero_mtx], 1)
        u0 = torch.stack([zero_vec, self.u0.exp()])
        s0 = torch.stack([zero_vec, self.s0.exp()])
        tau = torch.stack([F.leaky_relu(t - self.ton.exp(), neg_slope) for i in range(2)], 1)
        Uhat, Shat = pred_su(tau,
                             u0,
                             s0,
                             alpha,
                             beta,
                             gamma)
        Uhat = Uhat * torch.exp(self.scaling)

        Uhat = F.relu(Uhat)
        Shat = F.relu(Shat)
        vu = alpha - beta * Uhat / torch.exp(self.scaling)
        vs = beta * Uhat / torch.exp(self.scaling) - gamma * Shat
        return Uhat, Shat, vu, vs

    def forward(self, t, z, u0=None, s0=None, t0=None, condition=None, eval_mode=False, neg_slope=0.0):
        """top-level forward function for the decoder class

        Args:
            t (torch.Tensor):
                Cell time of size (n x 1)
            z (torch.Tensor):
                Cell state of size (n x dim_z)
            u0 (torch.Tensor, optional): 
                Estimated initial condition (u) of each cell. Defaults to None.
            s0 (torch.Tensor, optional): 
                Estimated initial condition (s) of each cell. Defaults to None.
            t0 (torch.Tensor, optional): 
                Estimated time at the initial condition of each cell. Defaults to None.
            condition (torch.Tensor, optional):
                Condition features. Defaults to None.
            eval_mode (bool, optional):
                Whether this function is called in model evaluation. If set to True,
                the mean time and cell state will be used to provide a deterministic prediction. Defaults to False.
            neg_slope (float, optional):
                Leaky ReLU slope to allow negative time intervals when cell time < t0. Defaults to 0.0.

        Returns:
            :class:`torch.Tensor`:
                Reconstructed unspliced and spliced counts, along with their corresponding velocity
        """
        if u0 is None or s0 is None or t0 is None:
            return self._forward_basis(t, z, condition, eval_mode, neg_slope)
        else:
            if condition is None:
                rho = torch.sigmoid(self.fc_out2(self.net_rho2(z)))
            else:
                rho = torch.sigmoid(self.fc_out2(self.net_rho2(torch.cat((z, condition), 1))))
            alpha, beta, gamma = self._sample_ode_param(random=not eval_mode)
            alpha = alpha*rho
            Uhat, Shat = pred_su(F.leaky_relu(t-t0, neg_slope),
                                 u0/self.scaling.exp(),
                                 s0,
                                 alpha,
                                 beta,
                                 gamma)
            Uhat = Uhat * torch.exp(self.scaling)

            Uhat = F.relu(Uhat)
            Shat = F.relu(Shat)
            Vu = alpha - beta * Uhat / torch.exp(self.scaling)
            Vs = beta * Uhat / torch.exp(self.scaling) - gamma * Shat
        return Uhat, Shat, Vu, Vs


class VAE(VanillaVAE):
    """VeloVAE Model
    """
    def __init__(self,
                 adata,
                 tmax,
                 dim_z,
                 dim_cond=0,
                 device='cpu',
                 hidden_size=(500, 250, 250, 500),
                 full_vb=False,
                 discrete=False,
                 init_method="steady",
                 init_key=None,
                 tprior=None,
                 init_ton_zero=True,
                 filter_gene=False,
                 count_distribution="Poisson",
                 std_z_prior=0.01,
                 checkpoints=[None, None],
                 rate_prior={
                     'alpha': (0.0, 1.0),
                     'beta': (0.0, 0.5),
                     'gamma': (0.0, 0.5)
                 },
                 **kwargs):
        """VeloVAE Model

        Args:
            adata ((:class:`anndata.AnnData`)):
                Input AnnData object
            tmax (float):
                Maximum time, specifies the time range.
            dim_z (int):
                Latent cell state dimension.
            dim_cond (int, optional):
                Dimension of any condition features. Defaults to 0.
            device (torch.device, optional):
                Device to hold the model. Defaults to 'cpu'.
            hidden_size (tuple, optional):
                The width of the first and second hidden layers of the encoder, followed by\
                    the width of the first and second hidden layers of the decoder. Defaults to (500, 250, 250, 500).
            full_vb (bool, optional):
                Whether to use the full variational Bayes feature. Defaults to False.
            discrete (bool, optional):
                Whether to directly model discrete counts. Defaults to False.
            init_method (str, optional):
                {'steady', 'tprior'}, initialization method. Defaults to "steady".
            init_key (str, optional):
                Key in adata.obs storing the capture time or any prior time information.
                This is used in initialization. Defaults to None.
            tprior (str, optional):
                Key in adata.obs containing the informative time prior.
                This is used in model training. Defaults to None.
            init_ton_zero (bool, optional):
                Whether to assume zero switch-on time for each gene. Defaults to False.
            filter_gene (bool, optional):
                Whether to filter out non-velocity genes based on scVelo-style initialization. Defaults to False.
            count_distribution (str, optional):
                Effective only when `discrete' is True. Defaults to "Poisson".
            std_z_prior (float, optional):
                Standard deviation of the cell state prior distribution. Defaults to 0.01.
            checkpoints (list, optional):
                Contains a list of two .pt files containing pretrained or saved model parameters.
                Defaults to [None, None].
            rate_prior (dict, optional):
                Contains the prior distributions of log rate parameters.
                Defaults to { 'alpha': (0.0, 1.0), 'beta': (0.0, 0.5), 'gamma': (0.0, 0.5) }.
        """
        t_start = time.time()
        self.timer = 0
        self.is_discrete = discrete
        self.is_full_vb = full_vb
        early_stop_thred = adata.n_vars*1e-4 if self.is_discrete else adata.n_vars*1e-3

        # Training Configuration
        self.config = {
            # Model Parameters
            "dim_z": dim_z,
            "hidden_size": hidden_size,
            "tmax": tmax,
            "init_method": init_method,
            "init_key": init_key,
            "tprior": tprior,
            "std_z_prior": std_z_prior,
            "tail": 0.01,
            "std_t_scaling": 0.05,
            "n_neighbors": 10,
            "dt": (0.03, 0.06),

            # Training Parameters
            "n_epochs": 1000,
            "n_epochs_post": 500,
            "n_refine": 20,
            "batch_size": 128,
            "learning_rate": None,
            "learning_rate_ode": None,
            "learning_rate_post": None,
            "lambda": 1e-3,
            "lambda_rho": 1e-3,
            "kl_t": 1.0,
            "kl_z": 1.0,
            "kl_w": 0.01,
            "reg_v": 0.0,
            "test_iter": None,
            "save_epoch": 100,
            "n_warmup": 5,
            "early_stop": 5,
            "early_stop_thred": early_stop_thred,
            "train_test_split": 0.7,
            "neg_slope": 0.0,
            "k_alt": 0,
            "train_ton": (init_method != 'tprior'),
            "vel_continuity_loss": 0,

            # hyperparameters for full vb
            "kl_param": 1.0,

            # Normalization Configurations
            "scale_gene_encoder": True,

            # Plotting
            "sparsify": 1
        }
        # set any additional customized hyperparameter
        for key in kwargs:
            if key in self.config:
                self.config[key] = kwargs[key]

        self._set_device(device)
        self._split_train_test(adata.n_obs)

        self.dim_z = dim_z
        self.enable_cvae = dim_cond > 0

        self.decoder = decoder(adata,
                               tmax,
                               self.train_idx,
                               dim_z,
                               N1=hidden_size[2],
                               N2=hidden_size[3],
                               full_vb=full_vb,
                               discrete=discrete,
                               init_ton_zero=init_ton_zero,
                               filter_gene=filter_gene,
                               device=self.device,
                               init_method=init_method,
                               init_key=init_key,
                               checkpoint=checkpoints[1],
                               **kwargs).float()

        try:
            G = adata.n_vars
            self.encoder = encoder(2*G,
                                   dim_z,
                                   dim_cond,
                                   hidden_size[0],
                                   hidden_size[1]).float().to(self.device)
        except IndexError:
            print('Please provide two dimensions!')
        if checkpoints[0] is not None:
            self.encoder.load_state_dict(torch.load(checkpoints[0], map_location=device))

        self.tmax = tmax
        self._get_prior(adata, tmax, tprior)

        self._pick_loss_func(adata, count_distribution)

        self.p_z = torch.stack([torch.zeros(adata.shape[0], dim_z, device=self.device),
                                torch.ones(adata.shape[0], dim_z, device=self.device)*self.config["std_z_prior"]])\
                        .float()
        # Prior of Decoder Parameters
        self.p_log_alpha = torch.tensor([[rate_prior['alpha'][0]], [rate_prior['alpha'][1]]], device=self.device)
        self.p_log_beta = torch.tensor([[rate_prior['beta'][0]], [rate_prior['beta'][1]]], device=self.device)
        self.p_log_gamma = torch.tensor([[rate_prior['gamma'][0]], [rate_prior['gamma'][1]]], device=self.device)

        self.alpha_w = torch.tensor(find_dirichlet_param(0.5, 0.05), device=self.device).float()

        self.use_knn = False
        self.u0 = None
        self.s0 = None
        self.t0 = None
        self.x0_index = None
        self.u1 = None
        self.s1 = None
        self.t1 = None
        self.x1_index = None
        self.lu_scale = (
            torch.tensor(np.log(adata.obs['library_scale_u'].to_numpy()), device=self.device).unsqueeze(-1).float()
            if self.is_discrete else torch.zeros(adata.n_obs, 1, device=self.device).float()
            )
        self.ls_scale = (
            torch.tensor(np.log(adata.obs['library_scale_s'].to_numpy()), device=self.device).unsqueeze(-1).float()
            if self.is_discrete else torch.zeros(adata.n_obs, 1, device=self.device).float()
            )

        # Class attributes for training
        self.loss_train, self.loss_test = [], []
        self.counter = 0  # Count the number of iterations
        self.n_drop = 0  # Count the number of consecutive iterations with little decrease in loss
        self.train_stage = 1

        self.timer = time.time()-t_start

    def _pick_loss_func(self, adata, count_distribution):
        ##############################################################
        # Pick the corresponding loss function (mainly the generative
        # likelihood function) based on count distribution
        ##############################################################
        if self.is_discrete:
            # Determine Count Distribution
            dispersion_u = adata.var["dispersion_u"].to_numpy()
            dispersion_s = adata.var["dispersion_s"].to_numpy()
            if count_distribution == "auto":
                p_nb = np.sum((dispersion_u > 1) & (dispersion_s > 1))/adata.n_vars
                if p_nb > 0.5:
                    count_distribution = "NB"
                    self.vae_risk = self._vae_risk_nb
                else:
                    count_distribution = "Poisson"
                    self.vae_risk = self._vae_risk_poisson
                print(f"Mean dispersion: u={dispersion_u.mean():.2f}, s={dispersion_s.mean():.2f}")
                print(f"Over-Dispersion = {p_nb:.2f} => Using {count_distribution} to model count data.")
            elif count_distribution == "NB":
                self.vae_risk = self._vae_risk_nb
            else:
                self.vae_risk = self._vae_risk_poisson
            mean_u = adata.var["mean_u"].to_numpy()
            mean_s = adata.var["mean_s"].to_numpy()
            dispersion_u[dispersion_u < 1] = 1.001
            dispersion_s[dispersion_s < 1] = 1.001
            self.eta_u = torch.tensor(np.log(dispersion_u-1)-np.log(mean_u), device=self.device).float()
            self.eta_s = torch.tensor(np.log(dispersion_s-1)-np.log(mean_s), device=self.device).float()
        else:
            self.vae_risk = self._vae_risk_gaussian

    def forward(self,
                data_in,
                u0=None,
                s0=None,
                t0=None,
                t1=None,
                condition=None):
        """Standard forward pass

        Args:
            data_in (:class:`torch.Tensor`):
                Cell-by-gene tensor.
            u0 (:class:`torch.Tensor`, optional):
                Estimated initial condition (u) of each cell. Defaults to None.
            s0 (:class:`torch.Tensor`, optional):
                Estimated initial condition (s) of each cell. Defaults to None.
            t0 (:class:`torch.Tensor`, optional):
                Estimated time at the initial condition of each cell. Defaults to None.
            t1 (:class:`torch.Tensor`, optional):
                Time of a future state. Effective only when
                config['vel_continuity_loss'] is True. Defaults to None.
            condition (:class:`torch.Tensor`, optional):
                Condition features. Defaults to None.

        Returns:
            tuple:

                - :class:`torch.Tensor`: Time mean, (N,1)

                - :class:`torch.Tensor`: Time standard deviation, (N,1)

                - :class:`torch.Tensor`: Cell state mean, (N, Cz)

                - :class:`torch.Tensor`: Cell state standard deviation, (N, Cz)

                - :class:`torch.Tensor`: Sampled cell time, (N,1)

                - :class:`torch.Tensor`: Sampled cell sate, (N,Cz)

                - :class:`torch.Tensor`: Predicted mean u values, (N,G)

                - :class:`torch.Tensor`: Predicted mean s values, (N,G)

                - :class:`torch.Tensor`: Predicted mean u values of the future state, (N,G).\
                    Valid only when `vel_continuity_loss` is set to True

                - :class:`torch.Tensor`: Predicted mean s values of the future state, (N,G)\
                    Valid only when `vel_continuity_loss` is set to True

                - :class:`torch.Tensor`: Unspliced velocity

                - :class:`torch.Tensor`: Spliced velocity

                - :class:`torch.Tensor`: Unspliced velocity at the future state\
                    Valid only when `vel_continuity_loss` is set to True

                - :class:`torch.Tensor`: Spliced velocity at the future state\
                    Valid only when `vel_continuity_loss` is set to True
        """
        data_in_scale = data_in
        # optional data scaling
        if self.config["scale_gene_encoder"]:
            data_in_scale = torch.cat((data_in_scale[:, :data_in_scale.shape[1]//2]/self.decoder.scaling.exp(),
                                       data_in_scale[:, data_in_scale.shape[1]//2:]), 1)
        mu_t, std_t, mu_z, std_z = self.encoder.forward(data_in_scale, condition)
        t = self.sample(mu_t, std_t)
        z = self.sample(mu_z, std_z)

        uhat, shat, vu, vs = self.decoder.forward(t, z, u0, s0, t0, condition, neg_slope=self.config["neg_slope"])

        if t1 is not None:  # predict the future state when we enable velocity continuity loss
            uhat_fw, shat_fw, vu_fw, vs_fw = self.decoder.forward(t1,
                                                                  z,
                                                                  uhat,
                                                                  shat,
                                                                  t,
                                                                  condition,
                                                                  neg_slope=self.config["neg_slope"])
        else:
            uhat_fw, shat_fw, vu_fw, vs_fw = None, None, None, None

        return mu_t, std_t, mu_z, std_z, t, z, uhat, shat, uhat_fw, shat_fw, vu, vs, vu_fw, vs_fw

    def eval_model(self,
                   data_in,
                   u0=None,
                   s0=None,
                   t0=None,
                   t1=None,
                   condition=None):
        data_in_scale = data_in
        # optional data scaling
        if self.config["scale_gene_encoder"]:
            data_in_scale = torch.cat((data_in_scale[:, :data_in_scale.shape[1]//2]/self.decoder.scaling.exp(),
                                       data_in_scale[:, data_in_scale.shape[1]//2:]), 1)
        mu_t, std_t, mu_z, std_z = self.encoder.forward(data_in_scale, condition)

        uhat, shat, vu, vs = self.decoder.forward(mu_t,
                                                  mu_z,
                                                  u0=u0,
                                                  s0=s0,
                                                  t0=t0,
                                                  condition=condition,
                                                  eval_mode=True)
        if t1 is not None:
            uhat_fw, shat_fw, vu_fw, vs_fw = self.decoder.forward(t1,
                                                                  mu_z,
                                                                  uhat,
                                                                  shat,
                                                                  mu_t,
                                                                  condition,
                                                                  eval_mode=True)
        else:
            uhat_fw, shat_fw, vu_fw, vs_fw = None, None, None, None

        return mu_t, std_t, mu_z, std_z, uhat, shat, uhat_fw, shat_fw, vu, vs, vu_fw, vs_fw

    def _cos_sim(self,
                 u0,
                 uhat,
                 vu,
                 s0,
                 shat,
                 vs):
        ##############################################################
        # Velocity correlation loss, optionally
        # added to the total loss function
        ##############################################################
        cossim = nn.CosineSimilarity(dim=1)
        delta_x = torch.cat([uhat-u0, shat-s0], 1)
        v = torch.cat([vu, vs], 1)
        return cossim(delta_x, v).mean()

    def _cos_sim_us(self,
                    u0,
                    uhat,
                    vu,
                    s0,
                    shat,
                    vs):
        ##############################################################
        # Velocity correlation loss, optionally
        # added to the total loss function
        ##############################################################
        cossim = nn.CosineSimilarity(dim=1)
        return cossim(uhat-u0, vu).mean() + cossim(shat-s0, vs).mean()

    def _cos_sim_gene(self,
                      u0,
                      uhat,
                      vu,
                      s0,
                      shat,
                      vs):
        ##############################################################
        # Velocity correlation loss, optionally
        # added to the total loss function
        ##############################################################
        cossim = nn.CosineSimilarity(dim=0)
        delta_x = torch.stack([uhat-u0, shat-s0])
        return cossim(delta_x, torch.stack([vu, vs])).mean()

    def _loss_vel(self,
                  uhat, shat,
                  vu, vs,
                  u0, s0,
                  uhat_fw=None, shat_fw=None,
                  vu_fw=None, vs_fw=None,
                  u1=None, s1=None):
        # Add velocity regularization
        loss_v = 0
        if u0 is not None and s0 is not None and self.config["reg_v"] > 0:
            scaling = self.decoder.scaling.exp()
            loss_v = loss_v + self.config["reg_v"] * self._cos_sim_us(u0/scaling,
                                                                      uhat/scaling,
                                                                      vu,
                                                                      s0,
                                                                      shat,
                                                                      vs)
            if vu_fw is not None and vs_fw is not None:
                loss_v = loss_v + self.config["reg_v"] * self._cos_sim_us(uhat/scaling,
                                                                          u1/scaling,
                                                                          vu_fw,
                                                                          shat,
                                                                          s1,
                                                                          vs_fw)
        return loss_v

    def _compute_kl_term(self, q_tx, p_t, q_zx, p_z):
        ##############################################################
        # Compute all KL-divergence terms
        # Arguments:
        # q_tx, q_zx: `tensor`
        #   conditional distribution of time and cell state given
        #   observation (count vector)
        # p_t, p_z: `tensor`
        #   Prior distribution, usually Gaussian
        ##############################################################
        kldt = self.kl_time(q_tx[0], q_tx[1], p_t[0], p_t[1], tail=self.config["tail"])
        kldz = kl_gaussian(q_zx[0], q_zx[1], p_z[0], p_z[1])
        kld_param = 0
        # In full VB, we treat all rate parameters as random variables
        if self.is_full_vb:
            kld_param = (kl_gaussian(self.decoder.alpha[0].view(1, -1),
                                     self.decoder.alpha[1].exp().view(1, -1),
                                     self.p_log_alpha[0],
                                     self.p_log_alpha[1])
                         + kl_gaussian(self.decoder.beta[0].view(1, -1),
                                       self.decoder.beta[1].exp().view(1, -1),
                                       self.p_log_beta[0],
                                       self.p_log_beta[1])
                         + kl_gaussian(self.decoder.gamma[0].view(1, -1),
                                       self.decoder.gamma[1].exp().view(1, -1),
                                       self.p_log_gamma[0],
                                       self.p_log_gamma[1])) / q_tx[0].shape[0]
        # In stage 1, dynamical mode weights are considered random
        kldw = (
            elbo_collapsed_categorical(self.decoder.logit_pw, self.alpha_w, 2, self.decoder.scaling.shape[0])
            if self.train_stage == 1 else 0
            )

        return (self.config["kl_t"]*kldt
                + self.config["kl_z"]*kldz
                + self.config["kl_param"]*kld_param
                + self.config["kl_w"]*kldw)

    def _vae_risk_gaussian(self,
                           q_tx, p_t,
                           q_zx, p_z,
                           u, s,
                           uhat, shat,
                           vu=None, vs=None,
                           u0=None, s0=None,
                           uhat_fw=None, shat_fw=None,
                           vu_fw=None, vs_fw=None,
                           u1=None, s1=None,
                           weight=None):
        kl_term = self._compute_kl_term(q_tx, p_t, q_zx, p_z)

        sigma_u = self.decoder.sigma_u.exp()
        sigma_s = self.decoder.sigma_s.exp()

        # u and sigma_u has the original scale
        clip_fn = nn.Hardtanh(-P_MAX, P_MAX)
        if uhat.ndim == 3:  # stage 1
            logp = -0.5*((u.unsqueeze(1)-uhat)/sigma_u).pow(2)\
                   - 0.5*((s.unsqueeze(1)-shat)/sigma_s).pow(2)\
                   - torch.log(sigma_u)-torch.log(sigma_s*2*np.pi)
            logp = clip_fn(logp)
            pw = F.softmax(self.decoder.logit_pw, dim=1).T
            logp = torch.sum(pw*logp, 1)
        else:
            logp = - 0.5*((u-uhat)/sigma_u).pow(2)\
                   - 0.5*((s-shat)/sigma_s).pow(2)\
                   - torch.log(sigma_u)-torch.log(sigma_s*2*np.pi)
            logp = clip_fn(logp)

        if uhat_fw is not None and shat_fw is not None:
            logp = logp - 0.5*((u1-uhat_fw)/sigma_u).pow(2)-0.5*((s1-shat_fw)/sigma_s).pow(2)

        if weight is not None:
            logp = logp*weight

        err_rec = torch.mean(torch.sum(logp, 1))

        err_rec = err_rec + self._loss_vel(uhat, shat,
                                           vu, vs,
                                           u0, s0,
                                           uhat_fw, shat_fw,
                                           vu_fw, vs_fw,
                                           u1, s1)
        return - err_rec + kl_term

    def _kl_poisson(self, lamb_1, lamb_2):
        return lamb_1 * (torch.log(lamb_1) - torch.log(lamb_2)) + lamb_2 - lamb_1

    def _kl_nb(self, m1, m2, p):
        r1 = m1 * (1 - p) / p
        r2 = m2 * (1 - p) / p
        return (r1 - r2)*torch.log(p)

    def _vae_risk_poisson(self,
                          q_tx, p_t,
                          q_zx, p_z,
                          u, s, 
                          uhat, shat,
                          vu=None, vs=None,
                          u0=None, s0=None,
                          uhat_fw=None, shat_fw=None,
                          vu_fw=None, vs_fw=None,
                          u1=None, s1=None,
                          weight=None,
                          eps=1e-2):
        """
        Training objective function for the discrete model (Poisson).
        The arugments and return values have the same meaning as vae_risk_gaussian
        """
        kl_term = self._compute_kl_term(q_tx, p_t, q_zx, p_z)

        # poisson
        mask_u = (torch.isnan(uhat) | torch.isinf(uhat)).float()
        uhat = uhat * (1-mask_u)
        mask_s = (torch.isnan(shat) | torch.isinf(shat)).float()
        shat = shat * (1-mask_s)

        poisson_u = Poisson(F.relu(uhat)+eps)
        poisson_s = Poisson(F.relu(shat)+eps)
        if uhat.ndim == 3:  # stage 1
            logp = poisson_u.log_prob(torch.stack([u, u], 1))\
                + poisson_s.log_prob(torch.stack([s, s], 1))
            pw = F.softmax(self.decoder.logit_pw, dim=1).T
            logp = torch.sum(pw*logp, 1)
        else:
            logp = poisson_u.log_prob(u) + poisson_s.log_prob(s)

        # velocity continuity loss
        if uhat_fw is not None and shat_fw is not None:
            logp = logp - self._kl_poisson(u1, uhat_fw) - self._kl_poisson(s1, shat_fw)
        if weight is not None:
            logp = logp*weight

        err_rec = torch.mean(logp.sum(1))
        err_rec = err_rec + self._loss_vel(uhat, shat,
                                           vu, vs,
                                           u0, s0,
                                           uhat_fw, shat_fw,
                                           vu_fw, vs_fw,
                                           u1, s1)

        return - err_rec + kl_term

    def _vae_risk_nb(self,
                     q_tx, p_t,
                     q_zx, p_z,
                     u, s,
                     uhat, shat,
                     vu=None, vs=None,
                     uhat_fw=None, shat_fw=None,
                     vu_fw=None, vs_fw=None,
                     u1=None, s1=None,
                     weight=None,
                     eps=1e-2):
        """Training objective function for the discrete model (negative binomial).
        The arugments and return values have the same meaning as vae_risk_gaussian
        """
        kl_term = self._compute_kl_term(q_tx, p_t, q_zx, p_z)

        # NB
        p_nb_u = torch.sigmoid(self.eta_u+torch.log(F.relu(uhat)+eps))
        p_nb_s = torch.sigmoid(self.eta_s+torch.log(F.relu(shat)+eps))
        nb_u = NegativeBinomial((F.relu(uhat)+1e-10)*(1-p_nb_u)/p_nb_u, probs=p_nb_u)
        nb_s = NegativeBinomial((F.relu(shat)+1e-10)*(1-p_nb_s)/p_nb_s, probs=p_nb_s)
        if uhat.ndim == 3:  # stage 1
            logp = nb_u.log_prob(torch.stack([u, u], 1)) + nb_s.log_prob(torch.stack([s, s], 1))
            pw = F.softmax(self.decoder.logit_pw, dim=1).T
            logp = torch.sum(pw*logp, 1)
        else:
            logp = nb_u.log_prob(u) + nb_s.log_prob(s)
        # velocity continuity loss
        if uhat_fw is not None and shat_fw is not None:
            logp = logp - self._kl_nb(uhat_fw, u1, p_nb_u) - self._kl_nb(shat_fw, s1, p_nb_s)
        if weight is not None:
            logp = logp*weight
        err_rec = torch.mean(torch.sum(logp, 1))
        err_rec = err_rec + self._loss_vel(uhat, shat,
                                           vu, vs,
                                           u0, s0,
                                           uhat_fw, shat_fw,
                                           vu_fw, vs_fw,
                                           u1, s1)

        return - err_rec + kl_term

    def _train_epoch(self, train_loader, test_set, optimizer, optimizer2=None, K=1):
        """Training in each epoch with early stopping.

        Args:
            train_loader (:class:`torch.utils.data.DataLoader`):
                Data loader of the input data.
            test_set (:class:`torch.utils.data.Dataset`):
                Validation dataset
            optimizer (optimizer from :class:`torch.optim`):
                Optimizer for neural network parameters.
            optimizer2 (optimizer from :class:`torch.optim`, optional): Defaults to None.
                Optimizer for ODE parameters.
            K (int, optional):
                Alternating update period.
                For every K updates of optimizer, there's one update for optimizer2.
                If set to 0, `optimizer2` will be ignored and only `optimizer` will be
                updated. Users can set it to 0 if they want to update sorely NN in one
                epoch and ODE in the next epoch. Defaults to 1.

        Returns:
            bool:
                Whether to stop training based on the early stopping criterium.
        """

        B = len(train_loader)
        self.set_mode('train')
        stop_training = False

        for i, batch in enumerate(train_loader):
            if self.counter == 1 or self.counter % self.config["test_iter"] == 0:
                elbo_test = self._test(test_set,
                                       None,
                                       self.counter,
                                       True)

                if len(self.loss_test) > 0:  # update the number of epochs with dropping/converging ELBO
                    if elbo_test - self.loss_test[-1] <= self.config["early_stop_thred"]:
                        self.n_drop = self.n_drop+1
                    else:
                        self.n_drop = 0
                self.loss_test.append(elbo_test)
                self.set_mode('train')

                if self.n_drop >= self.config["early_stop"] and self.config["early_stop"] > 0:
                    stop_training = True
                    break

            optimizer.zero_grad()
            if optimizer2 is not None:
                optimizer2.zero_grad()

            xbatch, idx = batch[0].float().to(self.device), batch[3]
            u = xbatch[:, :xbatch.shape[1]//2]
            s = xbatch[:, xbatch.shape[1]//2:]

            u0 = (torch.tensor(self.u0[self.train_idx[idx]], device=self.device, requires_grad=False)
                  if self.use_knn else None)
            s0 = (torch.tensor(self.s0[self.train_idx[idx]], device=self.device, requires_grad=False)
                  if self.use_knn else None)
            t0 = (torch.tensor(self.t0[self.train_idx[idx]], device=self.device, requires_grad=False)
                  if self.use_knn else None)
            u1 = (torch.tensor(self.u1[self.train_idx[idx]], device=self.device, requires_grad=False)
                  if (self.use_knn and self.config["vel_continuity_loss"]) else None)
            s1 = (torch.tensor(self.s1[self.train_idx[idx]], device=self.device, requires_grad=False)
                  if (self.use_knn and self.config["vel_continuity_loss"]) else None)
            t1 = (torch.tensor(self.t1[self.train_idx[idx]], device=self.device, requires_grad=False)
                  if (self.use_knn and self.config["vel_continuity_loss"]) else None)
            lu_scale = self.lu_scale[self.train_idx[idx]].exp()
            ls_scale = self.ls_scale[self.train_idx[idx]].exp()

            condition = F.one_hot(batch[1].to(self.device), self.n_type).float() if self.enable_cvae else None
            (mu_tx, std_tx,
             mu_zx, std_zx,
             t, z,
             uhat, shat,
             uhat_fw, shat_fw,
             vu, vs,
             vu_fw, vs_fw) = self.forward(xbatch, u0, s0, t0, t1, condition)
            if uhat.ndim == 3:
                lu_scale = lu_scale.unsqueeze(-1)
                ls_scale = ls_scale.unsqueeze(-1)
            loss = self.vae_risk((mu_tx, std_tx), self.p_t[:, self.train_idx[idx], :],
                                 (mu_zx, std_zx), self.p_z[:, self.train_idx[idx], :],
                                 u, s,
                                 uhat*lu_scale, shat*ls_scale,
                                 vu, vs,
                                 u0, s0,
                                 uhat_fw, shat_fw,
                                 vu_fw, vs_fw,
                                 u1, s1,
                                 None)

            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_value_(self.encoder.parameters(), GRAD_MAX)
            torch.nn.utils.clip_grad_value_(self.decoder.parameters(), GRAD_MAX)
            if K == 0:
                optimizer.step()
                if optimizer2 is not None:
                    optimizer2.step()
            else:
                if optimizer2 is not None and ((i+1) % (K+1) == 0 or i == B-1):
                    optimizer2.step()
                else:
                    optimizer.step()

            self.loss_train.append(loss.detach().cpu().item())
            self.counter = self.counter + 1
        return stop_training

    def _update_x0(self, U, S):
        """Estimate the initial conditions using time-based KNN.

        Args:
            U (:class:`torch.Tensor`):
                Input unspliced count matrix
            S (:class:`torch.Tensor`):
                Input spliced count matrix
        """
        self.set_mode('eval')
        out, elbo = self.pred_all(np.concatenate((U, S), 1),
                                  self.cell_labels,
                                  "both",
                                  ["uhat", "shat", "t", "z"],
                                  np.array(range(U.shape[1])))
        t, z = out["t"], out["z"]
        # Clip the time to avoid outliers
        t = np.clip(t, 0, np.quantile(t, 0.99))
        dt = (self.config["dt"][0]*(t.max()-t.min()), self.config["dt"][1]*(t.max()-t.min()))
        u1, s1, t1 = None, None, None
        # Compute initial conditions of cells without a valid pool of neighbors
        init_mask = (t <= np.quantile(t, 0.01))
        u0_init = np.mean(U[init_mask], 0)
        s0_init = np.mean(S[init_mask], 0)
        if self.x0_index is None:
            self.x0_index = knnx0_index(t[self.train_idx],
                                        z[self.train_idx],
                                        t,
                                        z,
                                        dt,
                                        self.config["n_neighbors"],
                                        hist_eq=True)
        u0, s0, t0 = get_x0(out["uhat"][self.train_idx],
                            out["shat"][self.train_idx],
                            t[self.train_idx],
                            dt,
                            self.x0_index,
                            u0_init,
                            s0_init)
        if self.config["vel_continuity_loss"]:
            if self.x1_index is None:
                self.x1_index = knnx0_index(t[self.train_idx],
                                            z[self.train_idx],
                                            t,
                                            z,
                                            dt,
                                            self.config["n_neighbors"],
                                            forward=True,
                                            hist_eq=True)
            u1, s1, t1 = get_x0(out["uhat"][self.train_idx],
                                out["shat"][self.train_idx],
                                t[self.train_idx],
                                dt,
                                self.x1_index,
                                None,
                                None,
                                forward=True)

        # return u0, s0, t0.reshape(-1,1), u1, s1, t1
        self.u0 = u0
        self.s0 = s0
        self.t0 = t0.reshape(-1, 1)
        if self.config['vel_continuity_loss']:
            self.u1 = u1
            self.s1 = s1
            self.t1 = t1.reshape(-1, 1)

    def _set_lr(self, p):
        """Set the learning rates based data sparsity.

        Args:
            p (float): Data sparsity, should be between 0 and 1.
        """
        if self.is_discrete:
            self.config["learning_rate"] = 10**(-8.3*p-2.25)
            self.config["learning_rate_post"] = self.config["learning_rate"]
            self.config["learning_rate_ode"] = 8*self.config["learning_rate"]
        else:
            self.config["learning_rate"] = 10**(-4*p-3)
            self.config["learning_rate_post"] = self.config["learning_rate"]
            self.config["learning_rate_ode"] = 8*self.config["learning_rate"]

    def train(self,
              adata,
              config={},
              plot=False,
              gene_plot=[],
              cluster_key="clusters",
              figure_path="figures",
              embed="umap"):
        """The high-level API for training

        Args:
            adata (:class:`anndata.AnnData`):
                Input AnnData object
            config (dict, optional):
                Contains the hyper-parameters users want to modify.
                Users can change the default using this argument. Defaults to {}.
            plot (bool, optional):
                Whether to plot intermediate results. Used for debugging. Defaults to False.
            gene_plot (list, optional):
                Genes to plot. Effective only when plot is True. Defaults to [].
            cluster_key (str, optional):
                Key in adata.obs storing the cell type annotation.. Defaults to "clusters".
            figure_path (str, optional):
                Path to the folder for saving plots. Defaults to "figures".
            embed (str, optional):
                Low dimensional embedding in adata.obsm. Used for plotting.
                The actual key storing the embedding should be f'X_{embed}'. Defaults to "umap".
        """
        self.load_config(config)
        if self.config["learning_rate"] is None:
            p = (np.sum(adata.layers["unspliced"].A > 0)
                 + (np.sum(adata.layers["spliced"].A > 0)))/adata.n_obs/adata.n_vars/2
            self._set_lr(p)
            print(f'Learning Rate based on Data Sparsity: {self.config["learning_rate"]:.4f}')
        print("--------------------------- Train a VeloVAE ---------------------------")
        # Get data loader
        if self.is_discrete:
            U, S = np.array(adata.layers['unspliced'].todense()), np.array(adata.layers['spliced'].todense())
            X = np.concatenate((U, S), 1).astype(int)
        else:
            X = np.concatenate((adata.layers['Mu'], adata.layers['Ms']), 1).astype(float)
        try:
            Xembed = adata.obsm[f"X_{embed}"]
        except KeyError:
            print("Embedding not found! Set to None.")
            Xembed = np.nan*np.ones((adata.n_obs, 2))
            plot = False

        cell_labels_raw = (adata.obs[cluster_key].to_numpy() if cluster_key in adata.obs
                           else np.array(['Unknown' for i in range(adata.n_obs)]))
        # Encode the labels
        cell_types_raw = np.unique(cell_labels_raw)
        self.label_dic, self.label_dic_rev = encode_type(cell_types_raw)

        self.n_type = len(cell_types_raw)
        self.cell_labels = np.array([self.label_dic[x] for x in cell_labels_raw])
        self.cell_types = np.array([self.label_dic[cell_types_raw[i]] for i in range(self.n_type)])

        print("*********        Creating Training/Validation Datasets        *********")
        train_set = SCData(X[self.train_idx], self.cell_labels[self.train_idx])
        test_set = None
        if len(self.test_idx) > 0:
            test_set = SCData(X[self.test_idx], self.cell_labels[self.test_idx])
        data_loader = DataLoader(train_set,
                                 batch_size=self.config["batch_size"],
                                 shuffle=True,
                                 pin_memory=True)
        # Automatically set test iteration if not given
        if self.config["test_iter"] is None:
            self.config["test_iter"] = len(self.train_idx)//self.config["batch_size"]*2
        print("*********                      Finished.                      *********")

        gind, gene_plot = get_gene_index(adata.var_names, gene_plot)
        os.makedirs(figure_path, exist_ok=True)

        # define optimizer
        print("*********                 Creating optimizers                 *********")
        param_nn = list(self.encoder.parameters())\
            + list(self.decoder.net_rho.parameters())\
            + list(self.decoder.fc_out1.parameters())
        param_ode = [self.decoder.alpha,
                     self.decoder.beta,
                     self.decoder.gamma,
                     self.decoder.u0,
                     self.decoder.s0,
                     self.decoder.logit_pw]
        if self.config['train_ton']:
            param_ode.append(self.decoder.ton)

        optimizer = torch.optim.Adam(param_nn, lr=self.config["learning_rate"], weight_decay=self.config["lambda"])
        optimizer_ode = torch.optim.Adam(param_ode, lr=self.config["learning_rate_ode"])
        print("*********                      Finished.                      *********")

        # Main Training Process
        print("*********                    Start training                   *********")
        print("*********                      Stage  1                       *********")
        print(f"Total Number of Iterations Per Epoch: {len(data_loader)}, test iteration: {self.config['test_iter']}")

        n_epochs = self.config["n_epochs"]

        start = time.time()
        for epoch in range(n_epochs):
            # Train the encoder
            if epoch >= self.config["n_warmup"]:
                stop_training = self._train_epoch(data_loader,
                                                  test_set,
                                                  optimizer_ode,
                                                  optimizer,
                                                  self.config["k_alt"])
            else:
                stop_training = self._train_epoch(data_loader,
                                                  test_set,
                                                  optimizer,
                                                  None,
                                                  self.config["k_alt"])

            if plot and (epoch == 0 or (epoch+1) % self.config["save_epoch"] == 0):
                elbo_train = self._test(train_set,
                                        Xembed[self.train_idx],
                                        f"train{epoch+1}",
                                        False,
                                        gind,
                                        gene_plot,
                                        plot,
                                        figure_path)
                self.set_mode('train')
                elbo_test = self.loss_test[-1] if len(self.loss_test) > 0 else -np.inf
                print(f"Epoch {epoch+1}: Train ELBO = {elbo_train:.3f},\t"
                      f"Test ELBO = {elbo_test:.3f},\t"
                      f"Total Time = {convert_time(time.time()-start)}")

            if stop_training:
                print(f"*********       Stage 1: Early Stop Triggered at epoch {epoch+1}.       *********")
                break

        count_epoch = epoch+1
        n_test1 = len(self.loss_test)

        print("*********                      Stage  2                       *********")
        self.encoder.eval()
        self.use_knn = True
        self.train_stage = 2
        self.decoder.logit_pw.requires_grad = False
        if not self.is_discrete:
            sigma_u_prev = self.decoder.sigma_u.detach().cpu().numpy()
            sigma_s_prev = self.decoder.sigma_s.detach().cpu().numpy()
            u0_prev, s0_prev = None, None
            noise_change = np.inf
        x0_change = np.inf
        x0_change_prev = np.inf
        param_post = list(self.decoder.net_rho2.parameters())+list(self.decoder.fc_out2.parameters())
        optimizer_post = torch.optim.Adam(param_post,
                                          lr=self.config["learning_rate_post"],
                                          weight_decay=self.config["lambda_rho"])
        for r in range(self.config['n_refine']):
            print(f"*********             Velocity Refinement Round {r+1}             *********")
            self.config['early_stop_thred'] *= 0.95
            stop_training = (x0_change - x0_change_prev >= -0.01 and r > 1) or (x0_change < 0.01)
            if (not self.is_discrete) and (noise_change > 0.001) and (r < self.config['n_refine']-1):
                self._update_std_noise(train_set.data)
                stop_training = False
            if stop_training:
                print(f"Stage 2: Early Stop Triggered at round {r}.")
                break
            self._update_x0(X[:, :X.shape[1]//2], X[:, X.shape[1]//2:])
            self.n_drop = 0

            for epoch in range(self.config["n_epochs_post"]):
                if epoch >= self.config["n_warmup"]:
                    stop_training = self._train_epoch(data_loader,
                                                      test_set,
                                                      optimizer_post,
                                                      optimizer_ode,
                                                      self.config["k_alt"])
                else:
                    stop_training = self._train_epoch(data_loader,
                                                      test_set,
                                                      optimizer_post,
                                                      None,
                                                      self.config["k_alt"])

                if plot and (epoch == 0 or (epoch+count_epoch+1) % self.config["save_epoch"] == 0):
                    elbo_train = self._test(train_set,
                                            Xembed[self.train_idx],
                                            f"train{epoch+count_epoch+1}",
                                            False,
                                            gind,
                                            gene_plot,
                                            plot,
                                            figure_path)
                    self.decoder.train()
                    elbo_test = self.loss_test[-1] if len(self.loss_test) > n_test1 else -np.inf
                    print(f"Epoch {epoch+count_epoch+1}: Train ELBO = {elbo_train:.3f},\t"
                          f"Test ELBO = {elbo_test:.3f},\t"
                          f"Total Time = {convert_time(time.time()-start)}")

                if stop_training:
                    print(f"*********     "
                          f"Round {r+1}: Early Stop Triggered at epoch {epoch+count_epoch+1}."
                          f"    *********")
                    break
            count_epoch += (epoch+1)
            if not self.is_discrete:
                sigma_u = self.decoder.sigma_u.detach().cpu().numpy()
                sigma_s = self.decoder.sigma_s.detach().cpu().numpy()
                norm_delta_sigma = np.sum((sigma_u-sigma_u_prev)**2 + (sigma_s-sigma_s_prev)**2)
                norm_sigma = np.sum(sigma_u_prev**2 + sigma_s_prev**2)
                sigma_u_prev = self.decoder.sigma_u.detach().cpu().numpy()
                sigma_s_prev = self.decoder.sigma_s.detach().cpu().numpy()
                noise_change = norm_delta_sigma/norm_sigma
                print(f"Change in noise variance: {noise_change:.4f}")
            if r > 0:
                x0_change_prev = x0_change
                norm_delta_x0 = np.sqrt(((self.u0 - u0_prev)**2 + (self.s0 - s0_prev)**2).sum(1).mean())
                std_x = np.sqrt((self.u0.var(0) + self.s0.var(0)).sum())
                x0_change = norm_delta_x0/std_x
                print(f"Change in x0: {x0_change:.4f}")
            u0_prev = self.u0
            s0_prev = self.s0

        elbo_train = self._test(train_set,
                                Xembed[self.train_idx],
                                "final-train",
                                False,
                                gind,
                                gene_plot,
                                plot,
                                figure_path)
        elbo_test = self._test(test_set,
                               Xembed[self.test_idx],
                               "final-test",
                               True,
                               gind,
                               gene_plot,
                               plot,
                               figure_path)
        self.loss_train.append(elbo_train)
        self.loss_test.append(elbo_test)
        # Plot final results
        if plot:
            plot_train_loss(self.loss_train,
                            range(1, len(self.loss_train)+1),
                            save=f'{figure_path}/train_loss_velovae.png')
            if self.config["test_iter"] > 0:
                plot_test_loss(self.loss_test,
                               [i*self.config["test_iter"] for i in range(1, len(self.loss_test)+1)],
                               save=f'{figure_path}/test_loss_velovae.png')

        self.timer = self.timer + (time.time()-start)
        print(f"*********              Finished. Total Time = {convert_time(self.timer)}             *********")
        print(f"Final: Train ELBO = {elbo_train:.3f},\tTest ELBO = {elbo_test:.3f}")
        return

    def _pred_all(self,
                  data,
                  cell_labels,
                  mode='test',
                  output=["uhat", "shat", "t", "z"],
                  gene_idx=None):
        N, G = data.shape[0], data.shape[1]//2
        if gene_idx is None:
            gene_idx = np.array(range(G))
        elbo = 0
        save_uhat_fw = "uhat_fw" in output and self.use_knn and self.config["vel_continuity_loss"]
        save_shat_fw = "shat_fw" in output and self.use_knn and self.config["vel_continuity_loss"]

        if "uhat" in output:
            Uhat = np.zeros((N, len(gene_idx)))
        if save_uhat_fw:
            Uhat_fw = np.zeros((N, len(gene_idx)))
        if "shat" in output:
            Shat = np.zeros((N, len(gene_idx)))
        if save_shat_fw:
            Shat_fw = np.zeros((N, len(gene_idx)))
        if "t" in output:
            t_out = np.zeros((N))
            std_t_out = np.zeros((N))
        if "z" in output:
            z_out = np.zeros((N, self.dim_z))
            std_z_out = np.zeros((N, self.dim_z))
        if "v" in output:
            Vu = np.zeros((N, len(gene_idx)))
            Vs = np.zeros((N, len(gene_idx)))

        with torch.no_grad():
            B = min(N//5, 5000)
            if N % B == 1:
                B = B - 1
            Nb = N // B
            has_init_cond = not ((self.t0 is None) or (self.u0 is None) or (self.s0 is None))

            w_hard = F.one_hot(torch.argmax(self.decoder.logit_pw, 1), num_classes=2).T
            for i in range(Nb):
                data_in = torch.tensor(data[i*B:(i+1)*B], device=self.device).float()
                if mode == "test":
                    batch_idx = self.test_idx[i*B:(i+1)*B]
                elif mode == "train":
                    batch_idx = self.train_idx[i*B:(i+1)*B]
                else:
                    batch_idx = np.array(range(i*B, (i+1)*B))

                u0 = (
                    torch.tensor(self.u0[batch_idx], dtype=torch.float, device=self.device)
                    if has_init_cond
                    else None)
                s0 = (
                    torch.tensor(self.s0[batch_idx], dtype=torch.float, device=self.device)
                    if has_init_cond
                    else None)
                t0 = (
                    torch.tensor(self.t0[batch_idx], dtype=torch.float, device=self.device)
                    if has_init_cond
                    else None)
                u1 = (
                    torch.tensor(self.u1[batch_idx], dtype=torch.float, device=self.device)
                    if (has_init_cond and self.config['vel_continuity_loss'])
                    else None)
                s1 = (
                    torch.tensor(self.s1[batch_idx], dtype=torch.float, device=self.device)
                    if (has_init_cond and self.config['vel_continuity_loss'])
                    else None)
                t1 = (
                    torch.tensor(self.t1[batch_idx], dtype=torch.float, device=self.device)
                    if (has_init_cond and self.config['vel_continuity_loss'])
                    else None)
                lu_scale = self.lu_scale[batch_idx, :].exp()
                ls_scale = self.ls_scale[batch_idx, :].exp()
                y_onehot = (
                    F.one_hot(torch.tensor(cell_labels[batch_idx],
                                           dtype=int,
                                           device=self.device),
                              self.n_type).float()
                    if self.enable_cvae else None)
                p_t = self.p_t[:, batch_idx, :]
                p_z = self.p_z[:, batch_idx, :]
                (mu_tx, std_tx,
                 mu_zx, std_zx,
                 uhat, shat,
                 uhat_fw, shat_fw,
                 vu, vs,
                 vu_fw, vs_fw) = self.eval_model(data_in, u0, s0, t0, t1, y_onehot)

                if uhat.ndim == 3:
                    lu_scale = lu_scale.unsqueeze(-1)
                    ls_scale = ls_scale.unsqueeze(-1)
                if uhat_fw is not None and shat_fw is not None:
                    uhat_fw = uhat_fw*lu_scale
                    shat_fw = uhat_fw*ls_scale
                loss = self.vae_risk((mu_tx, std_tx), p_t,
                                     (mu_zx, std_zx), p_z,
                                     data_in[:, :G], data_in[:, G:],
                                     uhat*lu_scale, shat*ls_scale,
                                     vu, vs,
                                     u0, s0,
                                     uhat_fw, shat_fw,
                                     vu_fw, vs_fw,
                                     u1, s1,
                                     None)
                elbo = elbo - (B/N)*loss
                if "uhat" in output and gene_idx is not None:
                    if uhat.ndim == 3:
                        uhat = torch.sum(uhat*w_hard, 1)
                    Uhat[i*B:(i+1)*B] = uhat[:, gene_idx].detach().cpu().numpy()
                if save_uhat_fw:
                    Uhat_fw[i*B:(i+1)*B] = uhat_fw[:, gene_idx].detach().cpu().numpy()
                if "shat" in output and gene_idx is not None:
                    if shat.ndim == 3:
                        shat = torch.sum(shat*w_hard, 1)
                    Shat[i*B:(i+1)*B] = shat[:, gene_idx].detach().cpu().numpy()
                if save_shat_fw:
                    Shat_fw[i*B:(i+1)*B] = shat_fw[:, gene_idx].detach().cpu().numpy()
                if "t" in output:
                    t_out[i*B:(i+1)*B] = mu_tx.detach().cpu().squeeze().numpy()
                    std_t_out[i*B:(i+1)*B] = std_tx.detach().cpu().squeeze().numpy()
                if "z" in output:
                    z_out[i*B:(i+1)*B] = mu_zx.detach().cpu().numpy()
                    std_z_out[i*B:(i+1)*B] = std_zx.detach().cpu().numpy()
                if "v" in output:
                    Vu[i*B:(i+1)*B] = vu[:, gene_idx].detach().cpu().numpy()
                    Vs[i*B:(i+1)*B] = vs[:, gene_idx].detach().cpu().numpy()

            if N > B*Nb:
                data_in = torch.tensor(data[Nb*B:], device=self.device).float()
                if mode == "test":
                    batch_idx = self.test_idx[Nb*B:]
                elif mode == "train":
                    batch_idx = self.train_idx[Nb*B:]
                else:
                    batch_idx = np.array(range(Nb*B, N))

                u0 = (
                    torch.tensor(self.u0[batch_idx], dtype=torch.float, device=self.device)
                    if has_init_cond
                    else None)
                s0 = (
                    torch.tensor(self.s0[batch_idx], dtype=torch.float, device=self.device)
                    if has_init_cond
                    else None)
                t0 = (
                    torch.tensor(self.t0[batch_idx], dtype=torch.float, device=self.device)
                    if has_init_cond
                    else None)
                u1 = (
                    torch.tensor(self.u1[batch_idx], dtype=torch.float, device=self.device)
                    if (has_init_cond and self.config['vel_continuity_loss'])
                    else None)
                s1 = (
                    torch.tensor(self.s1[batch_idx], dtype=torch.float, device=self.device)
                    if (has_init_cond and self.config['vel_continuity_loss'])
                    else None)
                t1 = (
                    torch.tensor(self.t1[batch_idx], dtype=torch.float, device=self.device)
                    if (has_init_cond and self.config['vel_continuity_loss'])
                    else None)
                lu_scale = self.lu_scale[batch_idx, :].exp()
                ls_scale = self.ls_scale[batch_idx, :].exp()
                y_onehot = (
                    F.one_hot(torch.tensor(cell_labels[batch_idx],
                                           dtype=int,
                                           device=self.device),
                              self.n_type).float()
                    if self.enable_cvae else None)
                p_t = self.p_t[:, batch_idx, :]
                p_z = self.p_z[:, batch_idx, :]
                (mu_tx, std_tx,
                 mu_zx, std_zx,
                 uhat, shat,
                 uhat_fw, shat_fw,
                 vu, vs,
                 vu_fw, vs_fw) = self.eval_model(data_in, u0, s0, t0, t1, y_onehot)
                if uhat.ndim == 3:
                    lu_scale = lu_scale.unsqueeze(-1)
                    ls_scale = ls_scale.unsqueeze(-1)
                if uhat_fw is not None and shat_fw is not None:
                    uhat_fw = uhat_fw*lu_scale
                    shat_fw = uhat_fw*ls_scale
                loss = self.vae_risk((mu_tx, std_tx), p_t,
                                     (mu_zx, std_zx), p_z,
                                     data_in[:, :G], data_in[:, G:],
                                     uhat*lu_scale, shat*ls_scale,
                                     vu, vs,
                                     u0, s0,
                                     uhat_fw, shat_fw,
                                     vu_fw, vs_fw,
                                     u1, s1,
                                     None)
                elbo = elbo - ((N-B*Nb)/N)*loss
                if "uhat" in output and gene_idx is not None:
                    if uhat.ndim == 3:
                        uhat = torch.sum(uhat*w_hard, 1)
                    Uhat[Nb*B:] = uhat[:, gene_idx].detach().cpu().numpy()
                if save_uhat_fw:
                    Uhat_fw[Nb*B:] = uhat_fw[:, gene_idx].detach().cpu().numpy()
                if "shat" in output and gene_idx is not None:
                    if shat.ndim == 3:
                        shat = torch.sum(shat*w_hard, 1)
                    Shat[Nb*B:] = shat[:, gene_idx].detach().cpu().numpy()
                if save_shat_fw:
                    Shat_fw[Nb*B:] = shat_fw[:, gene_idx].detach().cpu().numpy()
                if "t" in output:
                    t_out[Nb*B:] = mu_tx.detach().cpu().squeeze().numpy()
                    std_t_out[Nb*B:] = std_tx.detach().cpu().squeeze().numpy()
                if "z" in output:
                    z_out[Nb*B:] = mu_zx.detach().cpu().numpy()
                    std_z_out[Nb*B:] = std_zx.detach().cpu().numpy()
                if "v" in output:
                    Vu[Nb*B:] = vu[:, gene_idx].detach().cpu().numpy()
                    Vs[Nb*B:] = vs[:, gene_idx].detach().cpu().numpy()
        out = {}
        if "uhat" in output:
            out["uhat"] = Uhat
        if "shat" in output:
            out["shat"] = Shat
        if "t" in output:
            out["t"] = t_out
            out["std_t"] = std_t_out
        if "z" in output:
            out["z"] = z_out
            out["std_z"] = std_z_out
        if save_uhat_fw:
            out["uhat_fw"] = Uhat_fw
        if save_shat_fw:
            out["shat_fw"] = Shat_fw
        if "v" in output:
            out["vu"] = Vu
            out["vs"] = Vs

        return out, elbo.detach().cpu().item()

    def _test(self,
              dataset,
              Xembed,
              testid=0,
              test_mode=True,
              gind=None,
              gene_plot=None,
              plot=False,
              path='figures',
              **kwargs):
        """Evaluate the model on a training/test dataset.

        Args:
            dataset (:class:`torch.utils.data.Dataset`):
                Training or validation dataset
            Xembed (:class:`numpy array`):
                Low-dimensional embedding for plotting
            testid (int, optional):
                Used to name the figures.. Defaults to 0.
            test_mode (bool, optional):
                Whether dataset is training or validation dataset.
                This is used when retreiving certain class variable,
                e.g. cell-specific initial condition.. Defaults to True.
            gind (array like, optional):
                Index of genes in adata.var_names. Used for plotting. Defaults to None.
            gene_plot (:class:`numpy array`, optional):
                Gene names for plotting. Defaults to None.
            plot (bool, optional):
                Whether to generate plots.. Defaults to False.
            path (str, optional):
                Path for saving figures. Defaults to './figures'.

        Returns:
            float:
                VAE training/validation loss
        """
        self.set_mode('eval')
        mode = "test" if test_mode else "train"
        out_type = ["uhat", "shat", "uhat_fw", "shat_fw", "t"]
        if self.train_stage == 2:
            out_type.append("v")
        out, elbo = self.pred_all(dataset.data, self.cell_labels, mode, out_type, gind)
        Uhat, Shat, t = out["uhat"], out["shat"], out["t"]

        G = dataset.data.shape[1]//2

        if plot:
            # Plot Time
            plot_time(t, Xembed, save=f"{path}/time-{testid}-velovae.png")

            # Plot u/s-t and phase portrait for each gene
            for i in range(len(gind)):
                idx = gind[i]
                plot_sig(t.squeeze(),
                         dataset.data[:, idx], dataset.data[:, idx+G],
                         Uhat[:, i], Shat[:, i],
                         np.array([self.label_dic_rev[x] for x in dataset.labels]),
                         gene_plot[i],
                         save=f"{path}/sig-{gene_plot[i]}-{testid}.png",
                         sparsify=self.config['sparsify'])
                if self.config['vel_continuity_loss'] and self.train_stage == 2:
                    plot_sig(t.squeeze(),
                             dataset.data[:, idx], dataset.data[:, idx+G],
                             out["uhat_fw"][:, i], out["shat_fw"][:, i],
                             np.array([self.label_dic_rev[x] for x in dataset.labels]),
                             gene_plot[i],
                             save=f"{path}/sig-{gene_plot[i]}-{testid}-bw.png",
                             sparsify=self.config['sparsify'])
        plt.close()
        return elbo

    def _update_std_noise(self, train_data):
        """Update the standard deviation of Gaussian noise.

        Args:
            train_data (:class:`torch.Tensor`):
                Cell-by-gene training data.
                Unspliced and spliced counts are concatenated at the gene dimension.
        """
        G = train_data.shape[1]//2
        out, elbo = self.pred_all(train_data,
                                  self.cell_labels,
                                  mode='train',
                                  output=["uhat", "shat"],
                                  gene_idx=np.array(range(G)))
        std_u = (out["uhat"]-train_data[:, :G]).std(0)
        std_s = (out["shat"]-train_data[:, G:]).std(0)
        self.decoder.register_buffer('sigma_u',
                                     torch.tensor(np.log(std_u+1e-16),
                                                  dtype=torch.float,
                                                  device=self.device))
        self.decoder.register_buffer('sigma_s',
                                     torch.tensor(np.log(std_s+1e-16),
                                                  dtype=torch.float,
                                                  device=self.device))
        return

    def save_anndata(self, adata, key, file_path, file_name=None):
        """Updates an input AnnData object with inferred latent variable
            and estimations from the model and write it to disk.

        Args:
            adata (:class:`anndata.AnnData`):
                Input AnnData object
            key (str):
                Signature used to store all parameters of the model.
                Users can save outputs from different models to the same AnnData object using different keys.
            file_path (str):
                Path to the folder for saving.
            file_name (str, optional):
                If set to a string ending with .h5ad, the updated anndata object will be written to disk.
                Defaults to None.
        """
        self.set_mode('eval')
        os.makedirs(file_path, exist_ok=True)
        if self.is_full_vb:
            adata.var[f"{key}_logmu_alpha"] = self.decoder.alpha[0].detach().cpu().numpy()
            adata.var[f"{key}_logmu_beta"] = self.decoder.beta[0].detach().cpu().numpy()
            adata.var[f"{key}_logmu_gamma"] = self.decoder.gamma[0].detach().cpu().numpy()
            adata.var[f"{key}_logstd_alpha"] = self.decoder.alpha[1].detach().cpu().exp().numpy()
            adata.var[f"{key}_logstd_beta"] = self.decoder.beta[1].detach().cpu().exp().numpy()
            adata.var[f"{key}_logstd_gamma"] = self.decoder.gamma[1].detach().cpu().exp().numpy()
        else:
            adata.var[f"{key}_alpha"] = np.exp(self.decoder.alpha.detach().cpu().numpy())
            adata.var[f"{key}_beta"] = np.exp(self.decoder.beta.detach().cpu().numpy())
            adata.var[f"{key}_gamma"] = np.exp(self.decoder.gamma.detach().cpu().numpy())

        adata.var[f"{key}_ton"] = self.decoder.ton.exp().detach().cpu().numpy()
        adata.var[f"{key}_scaling"] = np.exp(self.decoder.scaling.detach().cpu().numpy())
        if not self.is_discrete:
            adata.var[f"{key}_sigma_u"] = np.exp(self.decoder.sigma_u.detach().cpu().numpy())
            adata.var[f"{key}_sigma_s"] = np.exp(self.decoder.sigma_s.detach().cpu().numpy())
        if self.is_discrete:
            U, S = np.array(adata.layers['unspliced'].todense()), np.array(adata.layers['spliced'].todense())
        else:
            U, S = adata.layers['Mu'], adata.layers['Ms']
        adata.varm[f"{key}_mode"] = F.softmax(self.decoder.logit_pw, 1).detach().cpu().numpy()

        out, elbo = self.pred_all(np.concatenate((U, S), 1),
                                  self.cell_labels,
                                  "both",
                                  gene_idx=np.array(range(adata.n_vars)))
        Uhat, Shat, t, std_t, z, std_z = out["uhat"], out["shat"], out["t"], out["std_t"], out["z"], out["std_z"]

        adata.obs[f"{key}_time"] = t
        adata.obs[f"{key}_std_t"] = std_t
        adata.obsm[f"{key}_z"] = z
        adata.obsm[f"{key}_std_z"] = std_z
        adata.layers[f"{key}_uhat"] = Uhat
        adata.layers[f"{key}_shat"] = Shat

        rho = np.zeros(adata.shape)
        with torch.no_grad():
            B = min(adata.n_obs//10, 1000)
            Nb = U.shape[0] // B
            for i in range(Nb):
                rho_batch = torch.sigmoid(
                    self.decoder.fc_out2(self.decoder.net_rho2(torch.tensor(z[i*B:(i+1)*B],
                                                                            device=self.device).float()))
                    )
                rho[i*B:(i+1)*B] = rho_batch.detach().cpu().numpy()
            rho_batch = torch.sigmoid(
                self.decoder.fc_out2(self.decoder.net_rho2(torch.tensor(z[Nb*B:],
                                                                        device=self.device).float()))
                )
            rho[Nb*B:] = rho_batch.detach().cpu().numpy()

        adata.layers[f"{key}_rho"] = rho

        adata.obs[f"{key}_t0"] = self.t0.squeeze()
        adata.layers[f"{key}_u0"] = self.u0
        adata.layers[f"{key}_s0"] = self.s0
        if self.config["vel_continuity_loss"]:
            adata.obs[f"{key}_t1"] = self.t1.squeeze()
            adata.layers[f"{key}_u1"] = self.u1
            adata.layers[f"{key}_s1"] = self.s1

        adata.uns[f"{key}_train_idx"] = self.train_idx
        adata.uns[f"{key}_test_idx"] = self.test_idx
        adata.uns[f"{key}_run_time"] = self.timer

        rna_velocity_vae(adata,
                         key,
                         use_raw=False,
                         use_scv_genes=False,
                         full_vb=self.is_full_vb)

        if file_name is not None:
            adata.write_h5ad(f"{file_path}/{file_name}")

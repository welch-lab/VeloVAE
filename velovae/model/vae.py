import numpy as np
import scipy.stats as stats
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions.negative_binomial import NegativeBinomial
from torch.distributions.poisson import Poisson
import time
from velovae.plotting import plot_sig, plot_sig_, plot_time
from velovae.plotting import plot_train_loss, plot_test_loss

from .model_util import hist_equal, init_params, get_ts_global, reinit_params
from .model_util import convert_time, get_gene_index
from .model_util import pred_su, ode_numpy, knnx0, knnx0_bin
from .model_util import elbo_collapsed_categorical
from .model_util import assign_gene_mode, find_dirichlet_param
from .model_util import scale_by_cell, get_cell_scale, get_dispersion

from .transition_graph import encode_type
from .training_data import SCData
from .vanilla_vae import VanillaVAE, kl_gaussian
from .velocity import rna_velocity_vae

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
                 N2=250,
                 device=torch.device('cpu'),
                 checkpoint=None):
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

        self.fc_mu_t = nn.Linear(N2+dim_cond, 1).to(device)
        self.spt1 = nn.Softplus()
        self.fc_std_t = nn.Linear(N2+dim_cond, 1).to(device)
        self.spt2 = nn.Softplus()
        self.fc_mu_z = nn.Linear(N2+dim_cond, dim_z).to(device)
        self.fc_std_z = nn.Linear(N2+dim_cond, dim_z).to(device)
        self.spt3 = nn.Softplus()

        if checkpoint is not None:
            self.load_state_dict(torch.load(checkpoint, map_location=device))
        else:
            self.init_weights()

    def init_weights(self):
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
                 init_type=None,
                 checkpoint=None,
                 scale_cell=False,
                 **kwargs):
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
                if scale_cell:
                    U, S, lu, ls = scale_by_cell(U, S, train_idx, True, 50)
                    adata.obs["library_scale_u"] = lu
                    adata.obs["library_scale_s"] = ls
                else:
                    lu, ls = get_cell_scale(U, S, train_idx, True, 50)
                    adata.obs["library_scale_u"] = lu
                    adata.obs["library_scale_s"] = ls
                del U
                del S
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
            # Gene dyncamical mode initialization
            dyn_mask = (T > tmax*0.01) & (np.abs(T-toff) > tmax*0.01)
            w = np.sum(((T < toff) & dyn_mask), 0) / np.sum(dyn_mask, 0)
            assign_type = kwargs['assign_type'] if 'assign_type' in kwargs else 'auto'
            if 'reverse_gene_mode' in kwargs:
                w = (1 - assign_gene_mode(adata, w, assign_type)
                     if kwargs['reverse_gene_mode'] else
                     assign_gene_mode(adata, w, assign_type))
            else:
                w = assign_gene_mode(adata, w, assign_type)
            adata.var["w_init"] = w
            logit_pw = 0.5*(np.log(w+1e-10) - np.log(1-w+1e-10))
            logit_pw = np.stack([logit_pw, -logit_pw], 1)

            print(f"Initial induction: {np.sum(w >= 0.5)}, repression: {np.sum(w < 0.5)}/{G}")

            if init_method == "random":
                print("Random Initialization.")
                self.alpha = nn.Parameter(torch.normal(0.0, 0.01, size=(G,), device=device).float())
                self.beta = nn.Parameter(torch.normal(0.0, 0.01, size=(G,), device=device).float())
                self.gamma = nn.Parameter(torch.normal(0.0, 0.01, size=(G,), device=device).float())
                self.ton = torch.nn.Parameter(torch.ones(G, device=device).float()*(-10))
                self.toff = torch.nn.Parameter(torch.ones(G, device=device).float()*(tmax/2))
                self.scaling = nn.Parameter(torch.tensor(np.log(scaling), device=device).float())
                self.sigma_u = (nn.Parameter(torch.tensor(np.log(sigma_u), device=device).float())
                                if not discrete else None)
                self.sigma_s = (nn.Parameter(torch.tensor(np.log(sigma_s), device=device).float())
                                if not discrete else None)
                self.t_init = None
                self.logit_pw = nn.Parameter(torch.normal(mean=0, std=torch.ones(G, 2).to(device)).float())
            elif init_method == "tprior":
                print("Initialization using prior time.")
                t_prior = adata.obs[init_key].to_numpy()
                t_prior = t_prior[train_idx]
                std_t = (np.std(t_prior)+1e-3)*0.2
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
                self.scaling = nn.Parameter(torch.tensor(np.log(scaling), device=device).float())
                self.sigma_u = (nn.Parameter(torch.tensor(np.log(sigma_u), device=device).float())
                                if not discrete else None)
                self.sigma_s = (nn.Parameter(torch.tensor(np.log(sigma_s), device=device).float())
                                if not discrete else None)
                self.ton = (nn.Parameter((torch.ones(G, device=device)*(-10)).float())
                            if init_ton_zero else
                            nn.Parameter(torch.tensor(np.log(self.ton_init+1e-10), device=device).float()))
                self.toff = nn.Parameter(torch.tensor(np.log(self.toff_init+1e-10), device=device).float())
                self.logit_pw = nn.Parameter(torch.tensor(logit_pw, device=device).float())
            else:
                print("Initialization using the steady-state and dynamical models.")
                if init_key is not None:
                    self.t_init = adata.obs[init_key].to_numpy()[train_idx]
                else:
                    T = T+np.random.rand(T.shape[0], T.shape[1]) * 1e-3
                    T_eq = np.zeros(T.shape)
                    Nbin = T.shape[0]//50+1
                    for i in range(T.shape[1]):
                        T_eq[:, i] = hist_equal(T[:, i], tmax, 0.9, Nbin)
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
                self.scaling = nn.Parameter(torch.tensor(np.log(scaling), device=device).float())
                self.ton = (nn.Parameter((torch.ones(adata.n_vars, device=device)*(-10)).float())
                            if init_ton_zero else
                            nn.Parameter(torch.tensor(np.log(self.ton_init+1e-10), device=device).float()))
                self.toff = nn.Parameter(torch.tensor(np.log(self.toff_init+1e-10), device=device).float())
                self.sigma_u = (nn.Parameter(torch.tensor(np.log(sigma_u), device=device).float())
                                if not discrete else None)
                self.sigma_s = (nn.Parameter(torch.tensor(np.log(sigma_s), device=device).float())
                                if not discrete else None)
                self.logit_pw = nn.Parameter(torch.tensor(logit_pw, device=device).float())
        # Add variance in case of full vb
        if full_vb:
            sigma_param = np.log(0.05) * torch.ones(adata.n_vars, device=device)
            self.alpha = nn.Parameter(torch.stack([self.alpha, sigma_param]).float())
            self.beta = nn.Parameter(torch.stack([self.beta, sigma_param]).float())
            self.gamma = nn.Parameter(torch.stack([self.gamma, sigma_param]).float())

        if init_type is None:
            self.u0 = nn.Parameter(torch.tensor(np.log(u0+1e-10), device=device).float())
            self.s0 = nn.Parameter(torch.tensor(np.log(s0+1e-10), device=device).float())
        elif init_type == "random":
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
        else:  # use the mean count of the initial type
            print(f"Setting the root cell to {init_type}")
            cell_labels = adata.obs["clusters"].to_numpy()[train_idx]
            cell_mask = cell_labels == init_type
            self.u0 = nn.Parameter(torch.tensor(np.log(U[cell_mask].mean(0)+1e-10), device=device).float())
            self.s0 = nn.Parameter(torch.tensor(np.log(S[cell_mask].mean(0)+1e-10), device=device).float())

            tprior = np.ones((adata.n_obs))*tmax*0.5
            tprior[adata.obs["clusters"].to_numpy() == init_type] = 0
            adata.obs['tprior'] = tprior

        self.scaling.requires_grad = False
        if not self.is_discrete:
            self.sigma_u.requires_grad = False
            self.sigma_s.requires_grad = False

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
            self.sigma_u = nn.Parameter(torch.empty(G, device=device).float())
            self.sigma_s = nn.Parameter(torch.empty(G, device=device).float())

            self.load_state_dict(torch.load(checkpoint, map_location=device))
        else:
            self.init_weights()

    def init_weights(self, net_id=None):
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
        ####################################################
        # Sample rate parameters for full vb or
        # output fixed rate parameters when
        # (1) random is set to False
        # (2) full vb is not enabled
        ####################################################
        if self.is_full_vb:
            if random:
                eps = torch.normal(mean=torch.zeros((3, self.alpha.shape[1])),
                                   std=torch.ones((3, self.alpha.shape[1]))).to(self.alpha.device)
                alpha = torch.exp(self.alpha[0] + eps[0]*(self.alpha[1].exp()))
                beta = torch.exp(self.beta[0] + eps[1]*(self.beta[1].exp()))
                gamma = torch.exp(self.gamma[0] + eps[2]*(self.gamma[1].exp()))
            else:
                alpha = self.alpha[0].exp()
                beta = self.beta[0].exp()
                gamma = self.gamma[0].exp()
        else:
            alpha = self.alpha.exp()
            beta = self.beta.exp()
            gamma = self.gamma.exp()
        return alpha, beta, gamma

    def forward_basis(self, t, z, condition=None, eval_mode=False, neg_slope=0.0):
        ####################################################
        # Outputs a (n sample, n basis, n gene) tensor
        ####################################################
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
        Uhat, Shat = pred_su(tau, u0, s0, alpha, beta, gamma)
        Uhat = Uhat * torch.exp(self.scaling)

        Uhat = F.relu(Uhat)
        Shat = F.relu(Shat)
        vu = alpha - beta * Uhat / torch.exp(self.scaling)
        vs = beta * Uhat / torch.exp(self.scaling) - gamma * Shat
        return Uhat, Shat, vu, vs

    def forward(self, t, z, u0=None, s0=None, t0=None, condition=None, eval_mode=False, neg_slope=0.0):
        ####################################################
        # top-level forward function for the decoder class
        ####################################################
        if u0 is None or s0 is None or t0 is None:
            return self.forward_basis(t, z, condition, eval_mode, neg_slope)
        else:
            if condition is None:
                rho = torch.sigmoid(self.fc_out2(self.net_rho2(z)))
            else:
                rho = torch.sigmoid(self.fc_out2(self.net_rho2(torch.cat((z, condition), 1))))
            alpha, beta, gamma = self._sample_ode_param(random=not eval_mode)
            alpha = alpha*rho
            Uhat, Shat = pred_su(F.leaky_relu(t-t0, neg_slope),
                                 u0/self.scaling.exp(), s0,
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
                 init_type=None,
                 init_ton_zero=True,
                 filter_gene=False,
                 scale_cell=False,
                 time_distribution="gaussian",
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
        full_vb : bool, optional
            Enable the full variational Bayes
        discrete : bool, optional
            Enable the discrete count model
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
        count_distriution : {'auto', 'Poisson', 'NB'}, optional
            Count distribution, effective only when discrete=True
            The current version only assumes Poisson or negative binomial distributions.
            When set to 'auto', the program determines a proper one based on over dispersion
        std_z_prior : float, optional
            Standard deviation of the prior (isotropical Gaussian) of cell state.
        checkpoints : list of 2 strings, optional
            Contains the path to saved encoder and decoder models.
            Should be a .pt file.
        """
        t_start = time.time()
        self.timer = 0
        self.is_discrete = discrete
        self.is_full_vb = full_vb

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
            "time_overlap": 0.5,
            "n_neighbors": 10,
            "dt": (0.03, 0.06),
            "n_bin": None,

            # Training Parameters
            "n_epochs": 1000,
            "n_epochs_post": 1000,
            "n_refine": 5,
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
            "early_stop_thred": adata.n_vars*1e-3,
            "train_test_split": 0.7,
            "neg_slope": 0.0,
            "k_alt": 1,
            "train_scaling": False,
            "train_std": False,
            "train_ton": (init_method != 'tprior'),
            "train_x0": False,
            "weight_sample": False,
            "vel_continuity_loss": False,

            # hyperparameters for full vb
            "kl_param": 1.0,

            # Normalization Configurations
            "scale_cell": scale_cell,
            "scale_gene_encoder": True,
            "scale_cell_encoder": False,
            "log1p": False,

            # Plotting
            "sparsify": 1
        }

        self.set_device(device)
        self.split_train_test(adata.n_obs)

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
                               init_type=init_type,
                               checkpoint=checkpoints[1],
                               scale_cell=scale_cell,
                               **kwargs).float()

        try:
            G = adata.n_vars
            self.encoder = encoder(2*G,
                                   dim_z,
                                   dim_cond,
                                   hidden_size[0],
                                   hidden_size[1],
                                   self.device,
                                   checkpoint=checkpoints[0]).float()
        except IndexError:
            print('Please provide two dimensions!')

        self.tmax = tmax
        self.time_distribution = time_distribution
        if init_type is not None and init_type != 'random':
            tprior = 'tprior'
            self.config['tprior'] = tprior
            self.config['train_ton'] = False
        self.get_prior(adata, time_distribution, tmax, tprior)

        self._pick_loss_func(adata, count_distribution)

        self.p_z = torch.stack([torch.zeros(adata.shape[0], dim_z),
                                torch.ones(adata.shape[0], dim_z)*self.config["std_z_prior"]]).float().to(self.device)
        # Prior of Decoder Parameters
        self.p_log_alpha = torch.tensor([[rate_prior['alpha'][0]], [rate_prior['alpha'][1]]]).to(self.device)
        self.p_log_beta = torch.tensor([[rate_prior['beta'][0]], [rate_prior['beta'][1]]]).to(self.device)
        self.p_log_gamma = torch.tensor([[rate_prior['gamma'][0]], [rate_prior['gamma'][1]]]).to(self.device)

        self.alpha_w = torch.tensor(find_dirichlet_param(0.5, 0.05)).float().to(self.device)

        self.use_knn = False
        self.u0 = None
        self.s0 = None
        self.t0 = None
        self.lu_scale = (
            torch.tensor(np.log(adata.obs['library_scale_u'].to_numpy())).unsqueeze(-1).float().to(self.device)
            if self.is_discrete else torch.zeros(adata.n_obs, 1).float().to(self.device)
            )
        self.ls_scale = (
            torch.tensor(np.log(adata.obs['library_scale_s'].to_numpy())).unsqueeze(-1).float().to(self.device)
            if self.is_discrete else torch.zeros(adata.n_obs, 1).float().to(self.device)
            )

        # Class attributes for training
        self.loss_train, self.loss_test = [], []
        self.counter = 0  # Count the number of iterations
        self.n_drop = 0  # Count the number of consecutive iterations with little decrease in loss
        self.train_stage = 1

        self.timer = time.time()-t_start

    def _pick_loss_func(self, adata, count_distribution):
        if self.is_discrete:
            # Determine Count Distribution
            dispersion_u = adata.var["dispersion_u"].to_numpy()
            dispersion_s = adata.var["dispersion_s"].to_numpy()
            if count_distribution == "auto":
                p_nb = np.sum((dispersion_u > 1) & (dispersion_s > 1))/adata.n_vars
                if p_nb > 0.5:
                    count_distribution = "NB"
                    self.vae_risk = self.vae_risk_nb
                else:
                    count_distribution = "Poisson"
                    self.vae_risk = self.vae_risk_poisson
                print(f"Mean dispersion: u={dispersion_u.mean():.2f}, s={dispersion_s.mean():.2f}")
                print(f"Over-Dispersion = {p_nb:.2f} => Using {count_distribution} to model count data.")
            elif count_distribution == "NB":
                self.vae_risk = self.vae_risk_nb
            else:
                self.vae_risk = self.vae_risk_poisson
            mean_u = adata.var["mean_u"].to_numpy()
            mean_s = adata.var["mean_s"].to_numpy()
            dispersion_u[dispersion_u < 1] = 1.001
            dispersion_s[dispersion_s < 1] = 1.001
            self.eta_u = torch.tensor(np.log(dispersion_u-1)-np.log(mean_u)).float().to(self.device)
            self.eta_s = torch.tensor(np.log(dispersion_s-1)-np.log(mean_s)).float().to(self.device)
        else:
            self.vae_risk = self.vae_risk_gaussian

    def forward(self, data_in, lu_scale, ls_scale, u0=None, s0=None, t0=None, t1=None, condition=None):
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
        data_in_scale = data_in
        if self.config["scale_gene_encoder"]:
            data_in_scale = torch.cat((data_in_scale[:, :data_in_scale.shape[1]//2]/self.decoder.scaling.exp(),
                                       data_in_scale[:, data_in_scale.shape[1]//2:]), 1)
        if self.config["scale_cell_encoder"]:
            data_in_scale = torch.cat((data_in_scale[:, :data_in_scale.shape[1]//2]/lu_scale,
                                       data_in_scale[:, data_in_scale.shape[1]//2:]/ls_scale), 1)
        if self.config["log1p"]:
            data_in_scale = torch.log1p(data_in_scale)
        mu_t, std_t, mu_z, std_z = self.encoder.forward(data_in_scale, condition)
        t = self.sample(mu_t, std_t)
        z = self.sample(mu_z, std_z)

        uhat, shat, vu, vs = self.decoder.forward(t, z, u0, s0, t0, condition, neg_slope=self.config["neg_slope"])
        if t1 is not None:
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

    def eval_model(self, data_in, lu_scale, ls_scale, u0=None, s0=None, t0=None, t1=None, condition=None):
        """Evaluate the model on the validation dataset.
        The major difference from forward pass is that we use the mean time and
        cell state instead of random sampling. The input arguments are the same as 'forward'.
        """
        data_in_scale = data_in
        if self.config["scale_gene_encoder"]:
            data_in_scale = torch.cat((data_in_scale[:, :data_in_scale.shape[1]//2]/self.decoder.scaling.exp(),
                                       data_in_scale[:, data_in_scale.shape[1]//2:]), 1)
        if self.config["scale_cell_encoder"]:
            data_in_scale = torch.cat((data_in_scale[:, :data_in_scale.shape[1]//2]/lu_scale,
                                       data_in_scale[:, data_in_scale.shape[1]//2:]/ls_scale), 1)
        if self.config["log1p"]:
            data_in_scale = torch.log1p(data_in_scale)
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

    def loss_vel(self, x0, xhat, v):
        cossim = nn.CosineSimilarity(dim=1)
        return cossim(xhat-x0, v).mean()

    def _compute_kl_term(self, q_tx, p_t, q_zx, p_z):
        kldt = self.kl_time(q_tx[0], q_tx[1], p_t[0], p_t[1], tail=self.config["tail"])
        kldz = kl_gaussian(q_zx[0], q_zx[1], p_z[0], p_z[1])
        kld_param = 0
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
        kldw = (
            elbo_collapsed_categorical(self.decoder.logit_pw, self.alpha_w, 2, self.decoder.scaling.shape[0])
            if self.train_stage == 1 else 0
            )
        return (self.config["kl_t"]*kldt
                + self.config["kl_z"]*kldz
                + self.config["kl_param"]*kld_param
                + self.config["kl_w"]*kldw)

    def vae_risk_gaussian(self,
                          q_tx, p_t,
                          q_zx, p_z,
                          u, s, uhat, shat,
                          uhat_fw=None, shat_fw=None,
                          u1=None, s1=None,
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
        weight : `torch.tensor`, optional
            Sample weight. This feature is not stable. Please consider setting it to None.

        Returns
        -------
        Negative ELBO : torch.tensor, scalar
        """
        kl_term = self._compute_kl_term(q_tx, p_t, q_zx, p_z)

        sigma_u = self.decoder.sigma_u.exp()
        sigma_s = self.decoder.sigma_s.exp()

        # u and sigma_u has the original scale
        if uhat.ndim == 3:  # stage 1
            logp = -0.5*((u.unsqueeze(1)-uhat)/sigma_u).pow(2)\
                   - 0.5*((s.unsqueeze(1)-shat)/sigma_s).pow(2)\
                   - torch.log(sigma_u)-torch.log(sigma_s*2*np.pi)
            pw = F.softmax(self.decoder.logit_pw, dim=1).T
            logp = torch.sum(pw*logp, 1)
        else:
            logp = -0.5*((u-uhat)/sigma_u).pow(2)\
                   - 0.5*((s-shat)/sigma_s).pow(2)\
                   - torch.log(sigma_u)-torch.log(sigma_s*2*np.pi)

        if uhat_fw is not None and shat_fw is not None:
            logp = logp - 0.5*((u1-uhat_fw)/sigma_u).pow(2)-0.5*((s1-shat_fw)/sigma_s).pow(2)

        if weight is not None:
            logp = logp*weight

        err_rec = torch.mean(torch.sum(logp, 1))

        return - err_rec + kl_term

    def _kl_poisson(self, lamb_1, lamb_2):
        return lamb_1 * (torch.log(lamb_1) - torch.log(lamb_2)) + lamb_2 - lamb_1

    def vae_risk_poisson(self,
                         q_tx, p_t,
                         q_zx, p_z,
                         u, s, uhat, shat,
                         uhat_fw=None, shat_fw=None,
                         u1=None, s1=None,
                         weight=None,
                         eps=1e-2):
        kl_term = self._compute_kl_term(q_tx, p_t, q_zx, p_z)

        # poisson
        mask_u = (torch.isnan(uhat) | torch.isinf(uhat)).float()
        uhat = uhat * (1-mask_u)
        mask_s = (torch.isnan(shat) | torch.isinf(shat)).float()
        shat = shat * (1-mask_s)

        poisson_u = Poisson(F.relu(uhat)+eps)
        poisson_s = Poisson(F.relu(shat)+eps)
        if uhat.ndim == 3:  # stage 1
            logp = poisson_u.log_prob(torch.stack([u, u], 1)) + poisson_s.log_prob(torch.stack([s, s], 1))
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

        return - err_rec + kl_term

    def vae_risk_nb(self,
                    q_tx, p_t,
                    q_zx, p_z,
                    u, s, uhat, shat,
                    uhat_fw=None, shat_fw=None,
                    u1=None, s1=None,
                    weight=None,
                    eps=1e-2):
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
            logp = logp - self._kl_poisson(u1, uhat_fw) - self._kl_poisson(s1, shat_fw)
        if weight is not None:
            logp = logp*weight
        err_rec = torch.mean(torch.sum(logp, 1))

        return - err_rec + kl_term

    def train_epoch(self, train_loader, test_set, optimizer, optimizer2=None, K=1):
        ##########################################################################
        # Training in each epoch.
        # Early stopping if enforced by default.
        # Arguments
        # ---------
        # 1.  train_loader [torch.utils.data.DataLoader]
        #     Data loader of the input data.
        # 2.  test_set [torch.utils.data.Dataset]
        #     Validation dataset
        # 3.  optimizer  [optimizer from torch.optim]
        # 4.  optimizer2 [optimizer from torch.optim
        #     (Optional) A second optimizer.
        #     This is used when we optimize NN and ODE simultaneously in one epoch.
        #     By default, VeloVAE performs alternating optimization in each epoch.
        #     The argument will be set to proper value automatically.
        # 5.  K [int]
        #     Alternating update period.
        #     For every K updates of optimizer, there's one update for optimizer2.
        #     If set to 0, optimizer2 will be ignored and only optimizer will be
        #     updated. Users can set it to 0 if they want to update sorely NN in one
        #     epoch and ODE in the next epoch.
        # Returns
        # 1.  stop_training [bool]
        #     Whether to stop training based on the early stopping criterium.
        ##########################################################################

        B = len(train_loader)
        self.set_mode('train')
        stop_training = False

        for i, batch in enumerate(train_loader):
            if self.counter == 1 or self.counter % self.config["test_iter"] == 0:
                elbo_test = self.test(test_set,
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

            u0 = (torch.tensor(self.u0[self.train_idx[idx]], device=self.device, requires_grad=True)
                  if self.use_knn else None)
            s0 = (torch.tensor(self.s0[self.train_idx[idx]], device=self.device, requires_grad=True)
                  if self.use_knn else None)
            t0 = (torch.tensor(self.t0[self.train_idx[idx]], device=self.device, requires_grad=True)
                  if self.use_knn else None)
            u1 = (torch.tensor(self.u1[self.train_idx[idx]], device=self.device, requires_grad=True)
                  if (self.use_knn and self.config["vel_continuity_loss"]) else None)
            s1 = (torch.tensor(self.s1[self.train_idx[idx]], device=self.device, requires_grad=True)
                  if (self.use_knn and self.config["vel_continuity_loss"]) else None)
            t1 = (torch.tensor(self.t1[self.train_idx[idx]], device=self.device, requires_grad=True)
                  if (self.use_knn and self.config["vel_continuity_loss"]) else None)
            lu_scale = self.lu_scale[self.train_idx[idx]].exp()
            ls_scale = self.ls_scale[self.train_idx[idx]].exp()

            if self.use_knn and self.config["train_x0"]:
                optimizer_x0 = torch.optim.Adam([s0, u0], lr=self.config["learning_rate_ode"])
                optimizer_x0.zero_grad()

            condition = F.one_hot(batch[1].to(self.device), self.n_type).float() if self.enable_cvae else None
            (mu_tx, std_tx,
             mu_zx, std_zx,
             t, z,
             uhat, shat,
             uhat_fw, shat_fw,
             vu, vs,
             vu_fw, vs_fw) = self.forward(xbatch, lu_scale, ls_scale, u0, s0, t0, t1, condition)
            if uhat.ndim == 3:
                lu_scale = lu_scale.unsqueeze(-1)
                ls_scale = ls_scale.unsqueeze(-1)
            loss = self.vae_risk((mu_tx, std_tx), self.p_t[:, self.train_idx[idx], :],
                                 (mu_zx, std_zx), self.p_z[:, self.train_idx[idx], :],
                                 u, s,
                                 uhat*lu_scale, shat*ls_scale,
                                 uhat_fw, shat_fw,
                                 u1, s1,
                                 None)
            # Add velocity regularization
            if self.use_knn and self.config["reg_v"] > 0:
                loss = loss - self.config["reg_v"] * (self.loss_vel(u0, uhat, vu) + self.loss_vel(s0, shat, vs))
                if vu_fw is not None and vs_fw is not None:
                    loss = loss - self.config["reg_v"]\
                        * (self.loss_vel(uhat, uhat_fw, vu_fw) + self.loss_vel(shat, shat_fw, vs_fw))
            loss.backward()
            if K == 0:
                optimizer.step()
                if optimizer2 is not None:
                    optimizer2.step()
            else:
                if optimizer2 is not None and ((i+1) % (K+1) == 0 or i == B-1):
                    optimizer2.step()
                else:
                    optimizer.step()

            # Update initial conditions
            if self.use_knn and self.config["train_x0"]:
                optimizer_x0.step()
                self.t0[self.train_idx[idx]] = t0.detach().cpu().numpy()
                self.u0[self.train_idx[idx]] = u0.detach().cpu().numpy()
                self.s0[self.train_idx[idx]] = s0.detach().cpu().numpy()

            self.loss_train.append(loss.detach().cpu().item())
            self.counter = self.counter + 1
        return stop_training

    def update_x0(self, U, S, n_bin=None):
        ##########################################################################
        # Estimate the initial conditions using KNN
        # < Input Arguments >
        # 1.  U [tensor (N,G)]
        #     Input unspliced count matrix
        # 2   S [tensor (N,G)]
        #     Input spliced count matrix
        # 3.  n_bin [int]
        #     (Optional) Set to a positive integer if binned KNN is used.
        # < Output >
        # 1.  u0 [tensor (N,G)]
        #     Initial condition for u
        # 2.  s0 [tensor (N,G)]
        #     Initial condition for s
        # 3. t0 [tensor (N,1)]
        #    Initial time
        ##########################################################################
        start = time.time()
        self.set_mode('eval')
        out, elbo = self.pred_all(np.concatenate((U, S), 1),
                                  self.cell_labels,
                                  "both",
                                  ["uhat", "shat", "t", "z"],
                                  np.array(range(U.shape[1])))
        t, z = out[2], out[4]
        # Clip the time to avoid outliers
        t = np.clip(t, 0, np.quantile(t, 0.99))
        dt = (self.config["dt"][0]*(t.max()-t.min()), self.config["dt"][1]*(t.max()-t.min()))
        u1, s1, t1 = None, None, None
        # Compute initial conditions of cells without a valid pool of neighbors
        with torch.no_grad():
            # w = F.softmax(self.decoder.logit_pw, 1).detach().cpu().numpy()[:, 0]
            # tensor shape (1, 2, n_gene)
            # u0_rep = np.exp(self.decoder.u0.detach().cpu().numpy())
            # s0_rep = np.exp(self.decoder.s0.detach().cpu().numpy())
            init_mask = (t <= np.quantile(t, 0.001))
            u0_init = np.mean(out[0][init_mask], 0)
            s0_init = np.mean(out[1][init_mask], 0)
            # u0_init = np.zeros((len(w)))*(w > 0.5) + u0_rep*(w <= 0.5)
            # s0_init = np.zeros((len(w)))*(w > 0.5) + s0_rep*(w <= 0.5)
        if n_bin is None:
            print("Cell-wise KNN Estimation.")
            u0, s0, t0 = knnx0(out[0][self.train_idx],
                               out[1][self.train_idx],
                               t[self.train_idx],
                               z[self.train_idx],
                               t,
                               z,
                               dt,
                               self.config["n_neighbors"],
                               u0_init,
                               s0_init,)
            if self.config["vel_continuity_loss"]:
                u1, s1, t1 = knnx0(out[0][self.train_idx],
                                   out[1][self.train_idx],
                                   t[self.train_idx],
                                   z[self.train_idx],
                                   t,
                                   z,
                                   dt,
                                   self.config["n_neighbors"],
                                   u0_init,
                                   s0_init,
                                   forward=True)
                t1 = t1.reshape(-1, 1)
        else:
            print(f"Fast KNN Estimation with {n_bin} time bins.")
            u0, s0, t0 = knnx0_bin(out[0][self.train_idx],
                                   out[1][self.train_idx],
                                   t[self.train_idx],
                                   z[self.train_idx],
                                   t,
                                   z,
                                   dt,
                                   self.config["n_neighbors"])
            if self.config["vel_continuity_loss"]:
                u1, s1, t1 = knnx0_bin(out[0][self.train_idx],
                                       out[1][self.train_idx],
                                       t[self.train_idx],
                                       z[self.train_idx],
                                       t,
                                       z,
                                       dt,
                                       self.config["n_neighbors"],
                                       forward=True)
                t1 = t1.reshape(-1, 1)
        print(f"Finished. Actual Time: {convert_time(time.time()-start)}")
        # return u0, s0, t0.reshape(-1,1), u1, s1, t1
        self.u0 = u0
        self.s0 = s0
        self.t0 = t0.reshape(-1, 1)
        if self.config['vel_continuity_loss']:
            self.u1 = u1
            self.s1 = s1
            self.t1 = t1.reshape(-1, 1)

    def train(self,
              adata,
              config={},
              plot=False,
              gene_plot=[],
              cluster_key="clusters",
              figure_path="figures",
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
        if self.config["learning_rate"] is None:
            p = (np.sum(adata.layers["unspliced"].A > 0)
                 + (np.sum(adata.layers["spliced"].A > 0)))/adata.n_obs/adata.n_vars/2
            self.config["learning_rate"] = 10**(-8.3*p-2.25)
            # self.config["learning_rate"] = 10**(-4.19*p-1.96)
            self.config["learning_rate_post"] = self.config["learning_rate"]
            self.config["learning_rate_ode"] = 8*self.config["learning_rate"]

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
            Xembed = None
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
        data_loader = DataLoader(train_set, batch_size=self.config["batch_size"], shuffle=True)
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
                     self.decoder.toff,
                     self.decoder.logit_pw]
        if self.config['train_ton']:
            param_ode.append(self.decoder.ton)
        if self.config['train_scaling']:
            self.decoder.scaling.requires_grad = True
            param_ode = param_ode+[self.decoder.scaling]
        if self.config['train_std']:
            self.decoder.sigma_u.requires_grad = True
            self.decoder.sigma_s.requires_grad = True
            param_ode = param_ode+[self.decoder.sigma_u, self.decoder.sigma_s]

        optimizer = torch.optim.Adam(param_nn, lr=self.config["learning_rate"], weight_decay=self.config["lambda"])
        optimizer_ode = torch.optim.Adam(param_ode, lr=self.config["learning_rate_ode"])
        print("*********                      Finished.                      *********")

        # Debugging Plots (to be removed in the published version)
        if self.decoder.t_init is not None and plot:
            tplot = self.decoder.t_init
            for i, idx in enumerate(gind):
                alpha = self.decoder.alpha_init[idx]
                beta = self.decoder.beta_init[idx]
                gamma = self.decoder.gamma_init[idx]
                ton = self.decoder.ton_init[idx]
                toff = self.decoder.toff_init[idx]
                scaling = self.decoder.scaling[idx].detach().cpu().exp().item()
                upred, spred = ode_numpy(tplot, alpha, beta, gamma, ton, toff, scaling)
                plot_sig_(tplot,
                          X[self.train_idx, idx],
                          X[self.train_idx, idx+adata.n_vars],
                          cell_labels_raw[self.train_idx],
                          tplot,
                          upred,
                          spred,
                          title=gene_plot[i],
                          save=f"{figure_path}/{gene_plot[i]}_init.png")

        # Main Training Process
        print("*********                    Start training                   *********")
        print("*********                      Stage  1                       *********")
        print(f"Total Number of Iterations Per Epoch: {len(data_loader)}, test iteration: {self.config['test_iter']}")

        n_epochs = self.config["n_epochs"]

        start = time.time()
        for epoch in range(n_epochs):
            # Train the encoder
            if self.config["k_alt"] is None:
                stop_training = self.train_epoch(data_loader, test_set, optimizer)

                if epoch >= self.config["n_warmup"]:
                    stop_training_ode = self.train_epoch(data_loader, test_set, optimizer_ode)
                    if stop_training_ode:
                        print(f"*********       Stage 1: Early Stop Triggered at epoch {epoch+1}.       *********")
                        break
            else:
                if epoch >= self.config["n_warmup"]:
                    stop_training = self.train_epoch(data_loader,
                                                     test_set,
                                                     optimizer_ode,
                                                     optimizer,
                                                     self.config["k_alt"])
                else:
                    stop_training = self.train_epoch(data_loader,
                                                     test_set,
                                                     optimizer,
                                                     None,
                                                     self.config["k_alt"])

            if plot and (epoch == 0 or (epoch+1) % self.config["save_epoch"] == 0):
                elbo_train = self.test(train_set,
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
        if not self.is_discrete:
            sigma_u_prev = self.decoder.sigma_u.detach().cpu().numpy()
            sigma_s_prev = self.decoder.sigma_s.detach().cpu().numpy()
            noise_change = 1
        param_post = list(self.decoder.net_rho2.parameters())+list(self.decoder.fc_out2.parameters())
        optimizer_post = torch.optim.Adam(param_post,
                                          lr=self.config["learning_rate_post"],
                                          weight_decay=self.config["lambda_rho"])
        for r in range(self.config['n_refine']):
            print(f"*********             Velocity Refinement Round {r+1}              *********")
            if (not self.is_discrete) and noise_change > 0.001 and r < self.config['n_refine']-1:
                self.update_std_noise(train_set.data)
            self.update_x0(X[:, :X.shape[1]//2], X[:, X.shape[1]//2:], self.config["n_bin"])
            # self.decoder.init_weights(2)
            self.n_drop = 0
            # param_ode = [self.decoder.alpha, self.decoder.beta, self.decoder.gamma]
            # optimizer_ode = torch.optim.Adam(param_ode,
            #                                  lr=self.config["learning_rate_ode"])
            for epoch in range(self.config["n_epochs_post"]):
                if self.config["k_alt"] is None:
                    stop_training = self.train_epoch(data_loader, test_set, optimizer_post)

                    if epoch >= self.config["n_warmup"]:
                        stop_training_ode = self.train_epoch(data_loader, test_set, optimizer_ode)
                        if stop_training_ode:
                            print(f"*********       "
                                  f"Stage 2: Early Stop Triggered at epoch {epoch+count_epoch+1}."
                                  f"       *********")
                            break
                else:
                    if epoch >= self.config["n_warmup"]:
                        stop_training = self.train_epoch(data_loader,
                                                         test_set,
                                                         optimizer_post,
                                                         optimizer_ode,
                                                         self.config["k_alt"])
                    else:
                        stop_training = self.train_epoch(data_loader,
                                                         test_set,
                                                         optimizer_post,
                                                         None,
                                                         self.config["k_alt"])

                if plot and (epoch == 0 or (epoch+count_epoch+1) % self.config["save_epoch"] == 0):
                    elbo_train = self.test(train_set,
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
                    print(f"*********       "
                          f"Stage 2: Early Stop Triggered at epoch {epoch+count_epoch+1}."
                          f"       *********")
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
                print(f"Change in noise variance: {noise_change}")
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

    def pred_all(self, data, cell_labels, mode='test', output=["uhat", "shat", "t", "z"], gene_idx=None):
        N, G = data.shape[0], data.shape[1]//2
        elbo = 0
        if "uhat" in output:
            Uhat = None if gene_idx is None else np.zeros((N, len(gene_idx)))
            if self.use_knn and self.config["vel_continuity_loss"]:
                Uhat_fw = None if gene_idx is None else np.zeros((N, len(gene_idx)))
        if "shat" in output:
            Shat = None if gene_idx is None else np.zeros((N, len(gene_idx)))
            if self.use_knn and self.config["vel_continuity_loss"]:
                Shat_fw = None if gene_idx is None else np.zeros((N, len(gene_idx)))
        if "t" in output:
            t_out = np.zeros((N))
            std_t_out = np.zeros((N))
        if "z" in output:
            z_out = np.zeros((N, self.dim_z))
            std_z_out = np.zeros((N, self.dim_z))

        with torch.no_grad():
            B = min(N//5, 5000)
            if N % B == 1:
                B = B - 1
            Nb = N // B
            has_init_cond = not ((self.t0 is None) or (self.u0 is None) or (self.s0 is None))

            w_hard = F.one_hot(torch.argmax(self.decoder.logit_pw, 1), num_classes=2).T
            for i in range(Nb):
                data_in = torch.tensor(data[i*B:(i+1)*B]).float().to(self.device)
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
                 vu_fw, vs_fw) = self.eval_model(data_in, lu_scale, ls_scale, u0, s0, t0, t1, y_onehot)
                if uhat.ndim == 3:
                    lu_scale = lu_scale.unsqueeze(-1)
                    ls_scale = ls_scale.unsqueeze(-1)
                loss = self.vae_risk((mu_tx, std_tx), p_t,
                                     (mu_zx, std_zx), p_z,
                                     data_in[:, :G], data_in[:, G:],
                                     uhat*lu_scale, shat*ls_scale,
                                     uhat_fw, shat_fw,
                                     u1, s1,
                                     None)
                elbo = elbo - (B/N)*loss
                if "uhat" in output and gene_idx is not None:
                    if uhat.ndim == 3:
                        uhat = torch.sum(uhat*w_hard, 1)
                    Uhat[i*B:(i+1)*B] = uhat[:, gene_idx].cpu().numpy()
                    if self.use_knn and self.config["vel_continuity_loss"]:
                        Uhat_fw[i*B:(i+1)*B] = uhat_fw[:, gene_idx].cpu().numpy()
                if "shat" in output and gene_idx is not None:
                    if shat.ndim == 3:
                        shat = torch.sum(shat*w_hard, 1)
                    Shat[i*B:(i+1)*B] = shat[:, gene_idx].cpu().numpy()
                    if self.use_knn and self.config["vel_continuity_loss"]:
                        Shat_fw[i*B:(i+1)*B] = shat_fw[:, gene_idx].cpu().numpy()
                if "t" in output:
                    t_out[i*B:(i+1)*B] = mu_tx.cpu().squeeze().numpy()
                    std_t_out[i*B:(i+1)*B] = std_tx.cpu().squeeze().numpy()
                if "z" in output:
                    z_out[i*B:(i+1)*B] = mu_zx.cpu().numpy()
                    std_z_out[i*B:(i+1)*B] = std_zx.cpu().numpy()

            if N > B*Nb:
                data_in = torch.tensor(data[Nb*B:]).float().to(self.device)
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
                 vu_fw, vs_fw) = self.eval_model(data_in, lu_scale, ls_scale, u0, s0, t0, t1, y_onehot)
                if uhat.ndim == 3:
                    lu_scale = lu_scale.unsqueeze(-1)
                    ls_scale = ls_scale.unsqueeze(-1)
                loss = self.vae_risk((mu_tx, std_tx), p_t,
                                     (mu_zx, std_zx), p_z,
                                     data_in[:, :G], data_in[:, G:],
                                     uhat*lu_scale, shat*ls_scale,
                                     uhat_fw, shat_fw,
                                     u1, s1,
                                     None)
                elbo = elbo - ((N-B*Nb)/N)*loss
                if "uhat" in output and gene_idx is not None:
                    if uhat.ndim == 3:
                        uhat = torch.sum(uhat*w_hard, 1)
                    Uhat[Nb*B:] = uhat[:, gene_idx].cpu().numpy()
                    if self.use_knn and self.config["vel_continuity_loss"]:
                        Uhat_fw[Nb*B:] = uhat_fw[:, gene_idx].cpu().numpy()
                if "shat" in output and gene_idx is not None:
                    if shat.ndim == 3:
                        shat = torch.sum(shat*w_hard, 1)
                    Shat[Nb*B:] = shat[:, gene_idx].cpu().numpy()
                    if self.use_knn and self.config["vel_continuity_loss"]:
                        Shat_fw[Nb*B:] = shat_fw[:, gene_idx].cpu().numpy()
                if "t" in output:
                    t_out[Nb*B:] = mu_tx.cpu().squeeze().numpy()
                    std_t_out[Nb*B:] = std_tx.cpu().squeeze().numpy()
                if "z" in output:
                    z_out[Nb*B:] = mu_zx.cpu().numpy()
                    std_z_out[Nb*B:] = std_zx.cpu().numpy()
        out = []
        if "uhat" in output:
            out.append(Uhat)
        if "shat" in output:
            out.append(Shat)
        if "t" in output:
            out.append(t_out)
            out.append(std_t_out)
        if "z" in output:
            out.append(z_out)
            out.append(std_z_out)
        if self.use_knn:
            if "uhat" in output and self.config["vel_continuity_loss"]:
                out.append(Uhat_fw)
            if "shat" in output and self.config["vel_continuity_loss"]:
                out.append(Shat_fw)

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
            Whether dataset is training or validation dataset. This is used when retreiving certain class variable,
            e.g. cell-specific initial condition.
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
        Uhat, Shat, t = out[0], out[1], out[2]

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
                if len(out) > 4:
                    plot_sig(t.squeeze(),
                             dataset.data[:, idx], dataset.data[:, idx+G],
                             out[4][:, i], out[5][:, i],
                             np.array([self.label_dic_rev[x] for x in dataset.labels]),
                             gene_plot[i],
                             save=f"{path}/sig-{gene_plot[i]}-{testid}-bw.png",
                             sparsify=self.config['sparsify'])

        return elbo

    def update_std_noise(self, train_data):
        """Update the standard deviation of Gaussian noise.
        .. deprecated:: 1.0
        """
        G = train_data.shape[1]//2
        out, elbo = self.pred_all(train_data,
                                  self.cell_labels,
                                  mode='train',
                                  output=["uhat", "shat"],
                                  gene_idx=np.array(range(G)))
        self.decoder.sigma_u = nn.Parameter(torch.tensor(np.log((out[0]-train_data[:, :G]).std(0)),
                                            device=self.device))
        self.decoder.sigma_s = nn.Parameter(torch.tensor(np.log((out[1]-train_data[:, G:]).std(0)),
                                            device=self.device))
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
        Uhat, Shat, t, std_t, z, std_z = out[0], out[1], out[2], out[3], out[4], out[5]

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
                    self.decoder.fc_out2(self.decoder.net_rho2(torch.tensor(z[i*B:(i+1)*B]).float().to(self.device)))
                    )
                rho[i*B:(i+1)*B] = rho_batch.cpu().numpy()
            rho_batch = torch.sigmoid(
                self.decoder.fc_out2(self.decoder.net_rho2(torch.tensor(z[Nb*B:]).float().to(self.device)))
                )
            rho[Nb*B:] = rho_batch.cpu().numpy()

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

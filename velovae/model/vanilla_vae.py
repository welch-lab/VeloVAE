"""Vanilla VAE Module
This module implements the basic variational mixture of ODEs model with constant rate parameters.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import time
from velovae.plotting import (plot_phase,
                              plot_sig,
                              plot_time,
                              plot_train_loss,
                              plot_test_loss)

from .model_util import hist_equal, init_params, get_ts_global, reinit_params, ode
from .model_util import convert_time, get_gene_index
from .training_data import SCData
from .velocity import rna_velocity_vanillavae


############################################################
# KL Divergence
############################################################
def kl_uniform(mu_t, std_t, t_start, t_end, **kwargs):
    """
    KL Divergence for the 1D near-uniform model
    KL(q||p) where
    q = uniform(t0, t0+dt)
    p = uniform(t_start, t_end) with exponential decays on both sides
    """

    tail = kwargs["tail"] if "tail" in kwargs else 0.05
    t0 = mu_t - np.sqrt(3)*std_t
    dt = np.sqrt(12)*std_t
    C = 1/((t_end-t_start)*(1+tail))
    lamb = 2/(tail*(t_end-t_start))

    t1 = t0+dt
    dt1_til = F.relu(torch.minimum(t_start, t1) - t0)
    dt2_til = F.relu(t1 - torch.maximum(t_end, t0))

    term1 = -lamb*(dt1_til.pow(2)+dt2_til.pow(2))/(2*dt)
    term2 = lamb*((t_start-t0)*dt1_til+(t1-t_end)*dt2_til)/dt

    return torch.mean(term1 + term2 - torch.log(C*dt))


def kl_gaussian(mu1, std1, mu2, std2, **kwargs):
    # Compute the KL divergence between two Gaussian distributions with diagonal covariance
    term_1 = torch.log(std2/std1)
    if torch.any(torch.isnan(term_1)) or torch.any(torch.isinf(term_1)):
        term_1 = torch.log(std2+1e-16) - torch.log(std1+1e-16)
    term_2 = std1.pow(2)/(2*std2.pow(2))
    term_3 = (mu1-mu2).pow(2)/(2*std2.pow(2))
    return torch.mean(torch.sum(term_1+term_2-0.5+term_3, 1))


##############################################################
# Vanilla VAE
##############################################################
class encoder(nn.Module):
    """Encoder of the vanilla VAE
    """
    def __init__(self,
                 Cin,
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
                                 self.fc2, self.bn2, nn.LeakyReLU(), self.dpt2)

        self.fc_mu, self.spt1 = nn.Linear(N2, 1).to(device), nn.Softplus()
        self.fc_std, self.spt2 = nn.Linear(N2, 1).to(device), nn.Softplus()

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
        for m in [self.fc_mu, self.fc_std]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, data_in, pos_mean=True):
        z = self.net(data_in)
        mu_zx, std_zx = self.fc_mu(z), self.spt2(self.fc_std(z))
        if pos_mean:
            mu_zx = self.spt1(mu_zx)
        return mu_zx, std_zx


class decoder(nn.Module):
    """Decoder of the vanilla VAE
    """
    def __init__(self,
                 adata,
                 tmax,
                 train_idx,
                 p=98,
                 filter_gene=False,
                 device=torch.device('cpu'),
                 init_method="steady",
                 init_key=None):
        super(decoder, self).__init__()
        U, S = adata.layers['Mu'][train_idx], adata.layers['Ms'][train_idx]
        X = np.concatenate((U, S), 1)
        N, G = U.shape

        (alpha, beta, gamma,
         scaling,
         toff,
         u0, s0,
         sigma_u, sigma_s,
         T,
         gene_score) = init_params(X, p, fit_scaling=True)
        if filter_gene:
            gene_mask = (gene_score == 1.0)
            adata._inplace_subset_var(gene_mask)
            U, S = U[:, gene_mask], S[:, gene_mask]
            G = adata.n_vars
            alpha, beta, gamma, scaling = alpha[gene_mask], beta[gene_mask], gamma[gene_mask], scaling[gene_mask]
            toff = toff[gene_mask]
            u0, s0 = u0[gene_mask], s0[gene_mask]
            sigma_u, sigma_s = sigma_u[gene_mask], sigma_s[gene_mask]
            T = T[:, gene_mask]
        # Dynamical Model Parameters
        if init_method == "random":
            print("Random Initialization.")
            self.alpha = nn.Parameter(torch.normal(0.0, 0.01, size=(G,), device=device).float())
            self.beta = nn.Parameter(torch.normal(0.0, 0.01, size=(G,), device=device).float())
            self.gamma = nn.Parameter(torch.normal(0.0, 0.01, size=(G,), device=device).float())
            self.ton = nn.Parameter(torch.normal(0.0, 0.01, size=(G,), device=device).float())
            self.toff = nn.Parameter(torch.normal(0.0, 0.01, size=(G,), device=device).float()+self.ton.detach())
        elif init_method == "tprior":
            print("Initialization using prior time.")
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
            self.ton = nn.Parameter(torch.tensor(np.log(ton+1e-10), device=device).float())
            self.toff = nn.Parameter(torch.tensor(np.log(toff+1e-10), device=device).float())
        else:
            print("Initialization using the steady-state and dynamical models.")
            if init_key is not None:
                self.t_init = adata.obs['init_key'].to_numpy()
            else:
                T = T+np.random.rand(T.shape[0], T.shape[1]) * 1e-3
                T_eq = np.zeros(T.shape)
                Nbin = T.shape[0]//50+1
                for i in range(T.shape[1]):
                    T_eq[:, i] = hist_equal(T[:, i], tmax, 0.9, Nbin)
                self.t_init = np.quantile(T_eq, 0.5, 1)

            toff = get_ts_global(self.t_init, U/scaling, S, 95)
            alpha, beta, gamma, ton = reinit_params(U/scaling, S, self.t_init, toff)

            self.alpha = nn.Parameter(torch.tensor(np.log(alpha), device=device).float())
            self.beta = nn.Parameter(torch.tensor(np.log(beta), device=device).float())
            self.gamma = nn.Parameter(torch.tensor(np.log(gamma), device=device).float())
            self.ton = nn.Parameter(torch.tensor(np.log(ton+1e-10), device=device).float())
            self.toff = nn.Parameter(torch.tensor(np.log(toff+1e-10), device=device).float())

        self.register_buffer('scaling', torch.tensor(np.log(scaling), device=device).float())
        self.register_buffer('sigma_u', torch.tensor(np.log(sigma_u), device=device).float())
        self.register_buffer('sigma_s', torch.tensor(np.log(sigma_s), device=device).float())

    def forward(self, t, neg_slope=0.0):
        Uhat, Shat = ode(t,
                         torch.exp(self.alpha),
                         torch.exp(self.beta),
                         torch.exp(self.gamma),
                         torch.exp(self.ton),
                         torch.exp(self.toff),
                         neg_slope=neg_slope)
        Uhat = Uhat * torch.exp(self.scaling)
        return F.relu(Uhat), F.relu(Shat)

    def pred_su(self, t, gidx=None):
        scaling = torch.exp(self.scaling)
        if gidx is not None:
            Uhat, Shat = ode(t,
                             torch.exp(self.alpha[gidx]),
                             torch.exp(self.beta[gidx]),
                             torch.exp(self.gamma[gidx]),
                             torch.exp(self.ton[gidx]),
                             torch.exp(self.toff[gidx]),
                             neg_slope=0.0)
            return F.relu(Uhat*scaling[gidx]), F.relu(Shat)
        Uhat, Shat = ode(t,
                         torch.exp(self.alpha),
                         torch.exp(self.beta),
                         torch.exp(self.gamma),
                         torch.exp(self.ton),
                         torch.exp(self.toff),
                         neg_slope=0.0)
        return F.relu(Uhat*scaling), F.relu(Shat)

    def get_ode_param_list(self):
        return [self.alpha, self.beta, self.gamma, self.ton, self.toff]


class VanillaVAE():
    """Basic VAE Model
    """
    def __init__(self,
                 adata,
                 tmax,
                 device='cpu',
                 hidden_size=(500, 250),
                 filter_gene=False,
                 init_method="steady",
                 init_key=None,
                 tprior=None,
                 checkpoints=None):
        """VeloVAE with constant rates

        Args:
            adata (:class:`anndata.AnnData`):
                Input AnnData object.
            tmax (float):
                Time Range.
            device (str, optional):
                {'cpu','gpu'}. Defaults to 'cpu'.
            hidden_size (tuple, optional):
                Width of the first and second hidden layers of the encoder. Defaults to (500, 250).
            filter_gene (bool, optional):
                Whether to filter out non-velocity genes based on scVelo-style initialization. Defaults to False.
            init_method (str, optional):
                {'steady', 'tprior'}, initialization method. Defaults to "steady".
            init_key (_type_, optional):
                column in the AnnData object containing the capture time. Defaults to None.
            tprior (_type_, optional):
                key in adata.obs that stores the capture time.
                Used for informative time prior. Defaults to None.
            checkpoints (_type_, optional):
                Contains the path to saved encoder and decoder models (.pt files). Defaults to None.
        """
        t_start = time.time()
        self.timer = 0

        # Default Training Configuration
        self.config = {
            # Model Parameters
            "tmax": tmax,
            "hidden_size": hidden_size,
            "init_method": init_method,
            "init_key": init_key,
            "tprior": tprior,
            "tail": 0.01,
            "std_t_scaling": 0.05,

            # Training Parameters
            "n_epochs": 2000,
            "batch_size": 128,
            "learning_rate": 2e-4,
            "learning_rate_ode": 5e-4,
            "lambda": 1e-3,
            "kl_t": 1.0,
            "test_iter": None,
            "save_epoch": 100,
            "n_warmup": 5,
            "early_stop": 5,
            "early_stop_thred": 1e-3*adata.n_vars,
            "train_test_split": 0.7,
            "k_alt": 1,
            "neg_slope": 0.0,
            "weight_sample": False,

            # Plotting
            "sparsify": 1
        }

        self._set_device(device)
        self._split_train_test(adata.n_obs)

        # Create a decoder
        self.decoder = decoder(adata,
                               tmax,
                               self.train_idx,
                               device=self.device,
                               filter_gene=filter_gene,
                               init_method=init_method,
                               init_key=init_key).float()
        G = adata.n_vars
        # Create an encoder
        try:
            self.encoder = encoder(2*G, hidden_size[0], hidden_size[1], self.device, checkpoint=checkpoints).float()
        except IndexError:
            print('Please provide two dimensions!')
        self.tmax = torch.tensor(tmax, device=self.device)
        # Time prior
        self._get_prior(adata, tmax, tprior)
        # class attributes for training
        self.loss_train, self.loss_test = [], []
        self.counter = 0  # Count the number of iterations
        self.n_drop = 0  # Count the number of consecutive epochs with negative/low ELBO gain
        self.train_stage = 1
        self.timer = time.time() - t_start

    def _get_prior(self, adata, tmax, tprior=None):
        """Compute the parameters of time prior distribution

        Args:
            adata (:class:`anndata.AnnData`):
                Input AnnData object
            tmax (float):
                Time range.
            tprior (str, optional):
                Key in adata.obs storing the capture time. Defaults to None.
        """
        self.kl_time = kl_gaussian
        self.sample = self._reparameterize
        if tprior is None:
            self.p_t = torch.stack([torch.ones(adata.n_obs, 1, device=self.device)*tmax*0.5,
                                    torch.ones(adata.n_obs, 1, device=self.device)*tmax
                                    * self.config["std_t_scaling"]]).float()
        else:
            print('Using informative time prior.')
            t = adata.obs[tprior].to_numpy()
            t = t/t.max()*tmax
            t_cap = np.sort(np.unique(t))
            std_t = np.zeros((len(t)))
            std_t[t == t_cap[0]] = (t_cap[1] - t_cap[0])*(0.5+0.5*self.config["std_t_scaling"])
            for i in range(1, len(t_cap)-1):
                std_t[t == t_cap[i]] = 0.5*(t_cap[i] - t_cap[i-1])*(0.5+0.5*self.config["std_t_scaling"]) \
                    + 0.5*(t_cap[i+1] - t_cap[i])*(0.5+0.5*self.config["std_t_scaling"])
            std_t[t == t_cap[-1]] = (t_cap[-1] - t_cap[-2])*(0.5+0.5*self.config["std_t_scaling"])
            self.p_t = torch.stack([torch.tensor(t, device=self.device).view(-1, 1),
                                    torch.tensor(std_t, device=self.device).view(-1, 1)]).float()

    def _set_device(self, device):
        if 'cuda' in device:
            if torch.cuda.is_available():
                self.device = torch.device(device)
            else:
                print('Warning: GPU not detected. Using CPU as the device.')
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')

    def _reparameterize(self, mu, std):
        """Apply the reparameterization trick for Gaussian random variables."""
        eps = torch.normal(mean=torch.zeros(mu.shape, device=self.device),
                           std=torch.ones(mu.shape, device=self.device))
        return std*eps+mu

    def _reparameterize_uniform(self, mu, std):
        """Apply the reparameterization trick for uniform random variables."""

        eps = torch.rand(mu.shape, device=self.device)
        return np.sqrt(12)*std*eps + (mu - np.sqrt(3)*std)

    def forward(self, data_in):
        """Forward function

        Args:
            data_in (:class:`torch.Tensor`):
                Cell-by-gene tensor.
                Unspliced and spliced counts are concatenated at the gene dimension (dim=1).

        Returns:
            tuple:

                - :class:`torch.Tensor`: time posterior mean

                - :class:`torch.Tensor`: time posterior standard deviation

                - :class:`torch.Tensor`: sampled time values

                - :class:`torch.Tensor`: predicted unspliced counts

                - :class:`torch.Tensor`: predicted spliced counts
        """
        data_in_scale = torch.cat((data_in[:, :data_in.shape[1]//2]/torch.exp(self.decoder.scaling),
                                   data_in[:, data_in.shape[1]//2:]), 1)
        mu_t, std_t = self.encoder.forward(data_in_scale)
        t_global = self._reparameterize(mu_t, std_t)
        # uhat is scaled
        uhat, shat = self.decoder.forward(t_global, neg_slope=self.config["neg_slope"])
        return mu_t, std_t, t_global, uhat, shat

    def eval_model(self, data_in):
        """Evaluate the model on a validation/test

        Args:
            data_in (:class:`torch.Tensor`):
                Cell-by-gene tensor.
                Unspliced and spliced counts are concatenated at the gene dimension (dim=1).

        Returns:
            tuple:

                - :class:`torch.Tensor`: Time posterior mean

                - :class:`torch.Tensor`: Time posterior standard deviation

                - :class:`torch.Tensor`: Predicted unspliced counts

                - :class:`torch.Tensor`: Predicted spliced counts
        """
        data_in_scale = torch.cat((data_in[:, :data_in.shape[1]//2]/torch.exp(self.decoder.scaling),
                                   data_in[:, data_in.shape[1]//2:]), 1)
        mu_t, std_t = self.encoder.forward(data_in_scale)

        uhat, shat = self.decoder.pred_su(mu_t)  # uhat is scaled
        return mu_t, std_t, uhat, shat

    def set_mode(self, mode):
        """Set the model to either training or evaluation mode."""
        if mode == 'train':
            self.encoder.train()
            self.decoder.train()
        elif mode == 'eval':
            self.encoder.eval()
            self.decoder.eval()
        else:
            print("Warning: mode not recognized. Must be 'train' or 'test'! ")
        if self.train_stage > 1:
            self.encoder.eval()
    ############################################################
    # Training Objective
    ############################################################

    def _vae_risk(self, q_tx, p_t, u, s, uhat, shat, sigma_u, sigma_s, weight=None, b=1.0):
        # This is the negative ELBO.
        kldt = self.kl_time(q_tx[0], q_tx[1], p_t[0], p_t[1], tail=self.config["tail"])
        # u and sigma_u has the original scale
        logp = -0.5*((u-uhat)/sigma_u).pow(2) - 0.5*((s-shat)/sigma_s).pow(2) \
               - torch.log(sigma_u) - torch.log(sigma_s*2*np.pi)
        if weight is not None:
            logp = logp*weight
        err_rec = torch.mean(torch.sum(logp, 1))

        return (- err_rec + b*(kldt))

    def _train_epoch(self, train_loader, test_set, optimizer, optimizer2=None, K=1):
        """
        Training in each epoch with early stopping.

        Args:
            train_loader (`:class:`torch.utils.data.DataLoader`):
                Data loader of the input data.
            test_set (:class:`torch.utils.data.Dataset`):
                Validation dataset
            optimizer (optimizer from :class:`torch.optim`):
                Optimizer for neural network parameters.
            optimizer2 (optimizer from :class:`torch.optim`, optional): Defaults to None.
                Optimizer for ODE parameters.
            K (int, optional): For every K updates of optimizer, there's one update for optimizer2.
                If set to 0, `optimizer2` will be ignored and only `optimizer` will be
                updated. Users can set it to 0 if they want to update sorely NN in one
                epoch and ODE in the next epoch. Defaults to 1.

        Returns:
            bool: Whether to stop training based on the early stopping criterium.
        """
        B = len(train_loader)
        self.set_mode('train')
        stop_training = False

        for i, batch in enumerate(train_loader):
            if self.counter == 1 or self.counter % self.config["test_iter"] == 0:
                elbo_test = self._test(test_set, None, self.counter)
                if len(self.loss_test) > 0:
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
            mu_tx, std_tx, t_global, uhat, shat = self.forward(xbatch)

            loss = self._vae_risk((mu_tx, std_tx),
                                  self.p_t[:, self.train_idx[idx], :],
                                  u, s,
                                  uhat, shat,
                                  torch.exp(self.decoder.sigma_u), torch.exp(self.decoder.sigma_s),
                                  None,
                                  self.config["kl_t"])

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

            self.loss_train.append(loss.detach().cpu().item())
            self.counter = self.counter + 1
        return stop_training

    def load_config(self, config):
        """Update hyper-parameters.

        Args:
            config (dict): Contains all hyper-parameters users want to modify.
        """
        for key in config:
            if key in self.config:
                self.config[key] = config[key]
            else:
                self.config[key] = config[key]
                print(f"Warning: unknown hyperparameter: {key}")

    def _split_train_test(self, N):
        # Randomly select indices as training samples.

        rand_perm = np.random.permutation(N)
        n_train = int(N*self.config["train_test_split"])
        self.train_idx = rand_perm[:n_train]
        self.test_idx = rand_perm[n_train:]
        return

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

        print("------------------------- Train a Vanilla VAE -------------------------")
        # Get data loader
        U, S = adata.layers['Mu'], adata.layers['Ms']
        X = np.concatenate((U, S), 1)
        try:
            Xembed = adata.obsm[f"X_{embed}"]
        except KeyError:
            print("Embedding not found! Please run the corresponding preprocessing step!")

        cell_labels_raw = (adata.obs[cluster_key].to_numpy() if cluster_key in adata.obs
                           else np.array(['Unknown' for i in range(adata.n_obs)]))

        print("*********        Creating Training/Validation Datasets        *********")
        train_set = SCData(X[self.train_idx], cell_labels_raw[self.train_idx])
        test_set = None
        if len(self.test_idx) > 0:
            test_set = SCData(X[self.test_idx], cell_labels_raw[self.test_idx])
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
        param_nn = list(self.encoder.parameters())
        param_ode = self.decoder.get_ode_param_list()

        optimizer = torch.optim.Adam(param_nn, lr=self.config["learning_rate"], weight_decay=self.config["lambda"])
        optimizer_ode = torch.optim.Adam(param_ode, lr=self.config["learning_rate_ode"])
        print("*********                      Finished.                      *********")

        # Main Training Process
        print("*********                    Start training                   *********")
        print(f"Total Number of Iterations Per Epoch: {len(data_loader)}, test iteration: {self.config['test_iter']}")
        n_epochs, n_save = self.config["n_epochs"], self.config["save_epoch"]
        n_warmup = self.config["n_warmup"]
        start = time.time()
        for epoch in range(n_epochs):
            # Train the encoder
            if self.config["k_alt"] is None:
                stop_training = self._train_epoch(data_loader, test_set, optimizer)
                if epoch >= n_warmup:
                    stop_training_ode = self._train_epoch(data_loader, test_set, optimizer_ode)
                    if stop_training_ode:
                        print(f"********* Early Stop Triggered at epoch {epoch+1}. *********")
                        break
            else:
                if epoch >= n_warmup:
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
            if plot and (epoch == 0 or (epoch+1) % n_save == 0):
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
                print(f"Epoch {epoch+1}: Train ELBO = {elbo_train:.3f}, \
                    Test ELBO = {elbo_test:.3f}, \t Total Time = {convert_time(time.time()-start)}")

            if stop_training:
                print(f"********* Early Stop Triggered at epoch {epoch+1}. *********")
                break

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
        if plot:
            plot_train_loss(self.loss_train,
                            range(1, len(self.loss_train)+1),
                            save=f'{figure_path}/train_loss_vanilla.png')
            if self.config["test_iter"] > 0:
                plot_test_loss(self.loss_test,
                               [i*self.config["test_iter"] for i in range(1, len(self.loss_test)+1)],
                               save=f'{figure_path}/test_loss_vanilla.png')
        self.timer = self.timer + (time.time()-start)
        print(f"*********              Finished. Total Time = {convert_time(self.timer)}             *********")
        print(f"Final: Train ELBO = {elbo_train:.3f},           Test ELBO = {elbo_test:.3f}")
        return

    def pred_all(self, data, mode='test', output=["uhat", "shat", "t"], gene_idx=None):
        """Generate different types of predictions from the model for all cells.

        Args:
            data (:class:`torch.Tensor`):
                Input cell-by-gene tensor, with U and S concatenated at the gene dimension (dim=1).
            cell_labels (:class:`torch.Tensor`):
                Cell type annotations encoded in integers.
                This is effective only for conditional VAEs with cell type as the condition.
            mode (str, optional):
                {'train','test','all}. Whether to predict on the training, validation or entire dataset.
                Defaults to 'test'.
            output (list, optional):
                Types of output to generate.
                Elements choosen from {'uhat', 'shat, 't', 'z', 'uhat_fw', 'shat_fw', 'v'}.
                'uhat' and 'shat' are predicted unspliced and spliced counts for each cell.
                't' is the cell time.
                'z' is the cell state.
                'uhat_fw' and 'shat_fw' are predictions for the future state given the current cell state.
                'v' is the velocity for both unspliced and spliced counts.
                Defaults to ['uhat', 'shat', 't', 'z'].
            gene_idx (array like, optional):
                Indices of genes for subsetting.
                If given, the outputs only preserve the selected genes. Defaults to None.

        Returns:
            tuple:

                - list: contains the corresponding data specified in the `output` argument.

                - float: VAE loss
        """
        N, G = data.shape[0], data.shape[1]//2
        if "uhat" in output:
            Uhat = None if gene_idx is None else np.zeros((N, len(gene_idx)))
        if "shat" in output:
            Shat = None if gene_idx is None else np.zeros((N, len(gene_idx)))
        if "t" in output:
            t_out = np.zeros((N))
            std_t_out = np.zeros((N))
        elbo = 0
        with torch.no_grad():
            B = min(N//10, 1000)
            Nb = N // B
            for i in range(Nb):
                data_in = torch.tensor(data[i*B:(i+1)*B], device=self.device).float()
                mu_tx, std_tx, uhat, shat = self.eval_model(data_in)
                if mode == "test":
                    p_t = self.p_t[:, self.test_idx[i*B:(i+1)*B], :]
                elif mode == "train":
                    p_t = self.p_t[:, self.train_idx[i*B:(i+1)*B], :]
                else:
                    p_t = self.p_t[:, i*B:(i+1)*B, :]
                loss = self._vae_risk((mu_tx, std_tx),
                                      p_t,
                                      data_in[:, :G], data_in[:, G:],
                                      uhat, shat,
                                      torch.exp(self.decoder.sigma_u), torch.exp(self.decoder.sigma_s),
                                      None,
                                      1.0)
                elbo = elbo-loss*B
                if "uhat" in output and gene_idx is not None:
                    Uhat[i*B:(i+1)*B] = uhat[:, gene_idx].detach().cpu().numpy()
                if "shat" in output and gene_idx is not None:
                    Shat[i*B:(i+1)*B] = shat[:, gene_idx].detach().cpu().numpy()
                if "t" in output:
                    t_out[i*B:(i+1)*B] = mu_tx.detach().cpu().squeeze().numpy()
                    std_t_out[i*B:(i+1)*B] = std_tx.detach().cpu().squeeze().numpy()
            if N > B*Nb:
                data_in = torch.tensor(data[B*Nb:], device=self.device).float()
                mu_tx, std_tx, uhat, shat = self.eval_model(data_in)
                if mode == "test":
                    p_t = self.p_t[:, self.test_idx[B*Nb:], :]
                elif mode == "train":
                    p_t = self.p_t[:, self.train_idx[B*Nb:], :]
                else:
                    p_t = self.p_t[:, B*Nb:, :]
                loss = self._vae_risk((mu_tx, std_tx),
                                      p_t,
                                      data_in[:, :G],
                                      data_in[:, G:],
                                      uhat, shat,
                                      torch.exp(self.decoder.sigma_u),
                                      torch.exp(self.decoder.sigma_s),
                                      None,
                                      1.0)
                elbo = elbo-loss*(N-B*Nb)
                if "uhat" in output and gene_idx is not None:
                    Uhat[Nb*B:] = uhat[:, gene_idx].detach().cpu().numpy()
                if "shat" in output and gene_idx is not None:
                    Shat[Nb*B:] = shat[:, gene_idx].detach().cpu().numpy()
                if "t" in output:
                    t_out[Nb*B:] = mu_tx.detach().cpu().squeeze().numpy()
                    std_t_out[Nb*B:] = std_tx.detach().cpu().squeeze().numpy()
        out = []
        if "uhat" in output:
            out.append(Uhat)
        if "shat" in output:
            out.append(Shat)
        if "t" in output:
            out.append(t_out)
            out.append(std_t_out)
        return out, elbo.detach().cpu().item()/N

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
            float: VAE training/validation loss
        """
        self.set_mode('eval')
        data = dataset.data
        mode = "test" if test_mode else "train"
        out, elbo = self.pred_all(data, mode, gene_idx=gind)
        Uhat, Shat, t = out[0], out[1], out[2]

        G = data.shape[1]//2
        if plot:
            ton = np.exp(self.decoder.ton.detach().cpu().numpy())
            toff = np.exp(self.decoder.toff.detach().cpu().numpy())
            state = np.ones(toff.shape)*(t.reshape(-1, 1) > toff)+np.ones(ton.shape)*2*(t.reshape(-1, 1) < ton)
            # Plot Time
            plot_time(t, Xembed, save=f"{path}/time-{testid}-vanilla.png")
            # Plot u/s-t and phase portrait for each gene
            for i in range(len(gind)):
                idx = gind[i]
                plot_phase(data[:, idx], data[:, idx+G],
                           Uhat[:, i], Shat[:, i],
                           gene_plot[i],
                           None,
                           state[:, idx],
                           ['Induction', 'Repression', 'Off'],
                           save=f"{path}/phase-{gene_plot[i]}-{testid}-vanilla.png")
                plot_sig(t.squeeze(),
                         data[:, idx], data[:, idx+G],
                         Uhat[:, i], Shat[:, i],
                         dataset.labels,
                         gene_plot[i],
                         save=f"{path}/sig-{gene_plot[i]}-{testid}-vanilla.png",
                         sparsify=self.config["sparsify"])
        return elbo

    def save_model(self, file_path, enc_name='encoder_vanilla', dec_name='decoder_vanilla'):
        """Save the encoder parameters to a .pt file.

        Args:
            file_path (str):
                Path to the folder for saving model parameters
            enc_name (str, optional):
                Name of the .pt file containing encoder parameters
            dec_name (str, optional):
                Name of the .pt file containing decoder parameters
        """
        os.makedirs(file_path, exist_ok=True)
        torch.save(self.encoder.state_dict(), f"{file_path}/{enc_name}.pt")
        torch.save(self.decoder.state_dict(), f"{file_path}/{dec_name}.pt")

    def save_anndata(self, adata, key, file_path, file_name=None):
        """Updates an input AnnData object with inferred latent variable\
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
        os.makedirs(file_path, exist_ok=True)

        self.set_mode('eval')
        adata.var[f"{key}_alpha"] = np.exp(self.decoder.alpha.detach().cpu().numpy())
        adata.var[f"{key}_beta"] = np.exp(self.decoder.beta.detach().cpu().numpy())
        adata.var[f"{key}_gamma"] = np.exp(self.decoder.gamma.detach().cpu().numpy())
        adata.var[f"{key}_toff"] = np.exp(self.decoder.toff.detach().cpu().numpy())
        adata.var[f"{key}_ton"] = np.exp(self.decoder.ton.detach().cpu().numpy())
        adata.var[f"{key}_scaling"] = np.exp(self.decoder.scaling.detach().cpu().numpy())
        adata.var[f"{key}_sigma_u"] = np.exp(self.decoder.sigma_u.detach().cpu().numpy())
        adata.var[f"{key}_sigma_s"] = np.exp(self.decoder.sigma_s.detach().cpu().numpy())

        out, elbo = self.pred_all(np.concatenate((adata.layers['Mu'], adata.layers['Ms']), axis=1),
                                  mode="both",
                                  gene_idx=np.array(range(adata.n_vars)))
        Uhat, Shat, t, std_t = out[0], out[1], out[2], out[3]

        adata.obs[f"{key}_time"] = t
        adata.obs[f"{key}_std_t"] = std_t
        adata.layers[f"{key}_uhat"] = Uhat
        adata.layers[f"{key}_shat"] = Shat

        adata.uns[f"{key}_train_idx"] = self.train_idx
        adata.uns[f"{key}_test_idx"] = self.test_idx
        adata.uns[f"{key}_run_time"] = self.timer

        rna_velocity_vanillavae(adata, key)

        if file_name is not None:
            adata.write_h5ad(f"{file_path}/{file_name}")

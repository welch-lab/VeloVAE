import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import os
import time
from velovae.plotting import plot_sig, plot_train_loss, plot_test_loss
from .model_util import init_params, reinit_type_params
from .model_util import convert_time, get_gene_index
from .model_util import ode_br, encode_type, str2int, int2str
from .training_data import SCTimedData
from .transition_graph import TransGraph
from .velocity import rna_velocity_brode


class decoder(nn.Module):
    def __init__(self,
                 adata,
                 cluster_key,
                 tkey,
                 embed_key,
                 train_idx,
                 param_key=None,
                 device=torch.device('cpu'),
                 p=98,
                 checkpoint=None,
                 **kwargs):
        super(decoder, self).__init__()

        U, S = adata.layers['Mu'][train_idx], adata.layers['Ms'][train_idx]
        X = np.concatenate((U, S), 1)
        G = adata.n_vars

        t = adata.obs[tkey].to_numpy()[train_idx]
        cell_labels_raw = adata.obs[cluster_key].to_numpy()
        self.cell_types = np.unique(cell_labels_raw)
        self.label_dic, self.label_dic_rev = encode_type(self.cell_types)
        cell_labels_int = str2int(cell_labels_raw, self.label_dic)
        cell_labels_int = cell_labels_int[train_idx]
        cell_types_int = str2int(self.cell_types, self.label_dic)
        self.Ntype = len(cell_types_int)

        # Transition Graph
        partition_k = kwargs['partition_k'] if 'partition_k' in kwargs else 5
        partition_res = kwargs['partition_res'] if 'partition_res' in kwargs else 0.005
        tgraph = TransGraph(adata, tkey, embed_key, cluster_key, train_idx, k=partition_k, res=partition_res)

        n_par = kwargs['n_par'] if 'n_par' in kwargs else 2
        dt = kwargs['dt'] if 'dt' in kwargs else (0.01, 0.05)
        k = kwargs['k'] if 'k' in kwargs else 5
        w = tgraph.compute_transition_deterministic(adata, n_par, dt, k)

        self.w = torch.tensor(w, device=device)
        self.par = torch.argmax(self.w, 1)

        # Dynamical Model Parameters
        if checkpoint is not None:
            self.alpha = nn.Parameter(torch.empty(G, device=device).double())
            self.beta = nn.Parameter(torch.empty(G, device=device).double())
            self.gamma = nn.Parameter(torch.empty(G, device=device).double())
            self.scaling = nn.Parameter(torch.empty(G, device=device).double())
            self.sigma_u = nn.Parameter(torch.empty(G, device=device).double())
            self.sigma_s = nn.Parameter(torch.empty(G, device=device).double())

            self.load_state_dict(torch.load(checkpoint, map_location=device))
        else:
            # Dynamical Model Parameters
            U, S = adata.layers['Mu'][train_idx], adata.layers['Ms'][train_idx]
            X = np.concatenate((U, S), 1)

            print("Initialization using type-specific dynamical model.")

            if param_key is not None:
                scaling = adata.var[f"{param_key}_scaling"].to_numpy()
                sigma_u = adata.var[f"{param_key}_sigma_u"].to_numpy()
                sigma_s = adata.var[f"{param_key}_sigma_s"].to_numpy()
            else:
                (alpha, beta, gamma,
                 scaling,
                 ts,
                 u0,
                 s0,
                 sigma_u,
                 sigma_s,
                 T,
                 Rscore) = init_params(X, p, fit_scaling=True)

            t_trans, dts = np.zeros((self.Ntype)), np.random.rand(self.Ntype, G)*0.01
            for i, type_ in enumerate(cell_types_int):
                t_trans[type_] = np.quantile(t[cell_labels_int == type_], 0.01)
            ts = t_trans.reshape(-1, 1) + dts

            alpha, beta, gamma, u0, s0 = reinit_type_params(U/scaling,
                                                            S,
                                                            t,
                                                            ts,
                                                            cell_labels_int,
                                                            cell_types_int,
                                                            cell_types_int)

            self.alpha = nn.Parameter(torch.tensor(np.log(alpha), device=device).double())
            self.beta = nn.Parameter(torch.tensor(np.log(beta), device=device).double())
            self.gamma = nn.Parameter(torch.tensor(np.log(gamma), device=device).double())
            self.t_trans = nn.Parameter(torch.tensor(np.log(t_trans+1e-10), device=device).double())
            self.u0 = nn.Parameter(torch.tensor(np.log(u0), device=device).double())
            self.s0 = nn.Parameter(torch.tensor(np.log(s0), device=device).double())
            self.scaling = nn.Parameter(torch.tensor(np.log(scaling), device=device).double())
            self.sigma_u = nn.Parameter(torch.tensor(np.log(sigma_u), device=device).double())
            self.sigma_s = nn.Parameter(torch.tensor(np.log(sigma_s), device=device).double())

        self.t_trans.requires_grad = False
        self.scaling.requires_grad = False
        self.sigma_u.requires_grad = False
        self.sigma_s.requires_grad = False
        self.u0.requires_grad = False
        self.s0.requires_grad = False

    def forward(self, t, y, neg_slope=0.0):
        return ode_br(t,
                      y,
                      self.par,
                      neg_slope=neg_slope,
                      alpha=torch.exp(self.alpha),
                      beta=torch.exp(self.beta),
                      gamma=torch.exp(self.gamma),
                      t_trans=torch.exp(self.t_trans),
                      u0=torch.exp(self.u0),
                      s0=torch.exp(self.s0),
                      sigma_u=torch.exp(self.sigma_u),
                      sigma_s=torch.exp(self.sigma_s),
                      scaling=torch.exp(self.scaling))

    def pred_su(self, t, y, gidx=None):
        if gidx is None:
            return ode_br(t,
                          y,
                          self.par,
                          neg_slope=0.0,
                          alpha=torch.exp(self.alpha),
                          beta=torch.exp(self.beta),
                          gamma=torch.exp(self.gamma),
                          t_trans=torch.exp(self.t_trans),
                          u0=torch.exp(self.u0),
                          s0=torch.exp(self.s0),
                          sigma_u=torch.exp(self.sigma_u),
                          sigma_s=torch.exp(self.sigma_s),
                          scaling=torch.exp(self.scaling))
        return ode_br(t,
                      y,
                      self.par,
                      neg_slope=0.0,
                      alpha=torch.exp(self.alpha[:, gidx]),
                      beta=torch.exp(self.beta[:, gidx]),
                      gamma=torch.exp(self.gamma[:, gidx]),
                      t_trans=torch.exp(self.t_trans),
                      u0=torch.exp(self.u0[:, gidx]),
                      s0=torch.exp(self.s0[:, gidx]),
                      sigma_u=torch.exp(self.sigma_u[gidx]),
                      sigma_s=torch.exp(self.sigma_s[gidx]),
                      scaling=torch.exp(self.scaling[gidx]))


class BrODE():
    def __init__(self,
                 adata,
                 cluster_key,
                 tkey,
                 embed_key,
                 param_key=None,
                 device='cpu',
                 checkpoint=None,
                 graph_param={}):
        """High-level ODE model for RNA velocity with branching structure.

        Arguments
        ---------

        adata : :class:`anndata.AnnData`
        cluster_key : str
            Key in adata.obs storing the cell type annotation.
        tkey : str
            Key in adata.obs storing the latent cell time
        embed_key : str
            Key in adata.obsm storing the latent cell state
        param_key : str, optional
            Used to extract sigma_u, sigma_s and scaling from adata.var
        device : `torch.device`
            Either cpu or gpu
        checkpoint : string, optional
            Path to a file containing a pretrained model. \
            If given, initialization will be skipped and arguments relating to initialization will be ignored.
        graph_param : dictionary, optional
            Hyper-parameters for the transition graph computation.
            Keys should contain:
            (1) partition_k: num_neighbors in graph partition (a KNN graph is computed by scanpy)
            (2) partition_res: resolution of Louvain clustering in graph partition
            (3) n_par: number of parents to keep in graph pruning
            (4) dt: tuple (r1,r2), proportion of time range to consider as the parent time window
                Let t_range be the time range. Then for any cell with time t, only cells in the
                time window (t-r2*t_range, t-r1*t_range)
            (5) k: KNN in parent counting.
                This is different from partition_k. When we pick the time window, KNN
                is computed to choose the most likely parents from the cells in the window.
        """
        t_start = time.time()
        self.timer = 0
        try:
            cell_labels_raw = adata.obs[cluster_key].to_numpy()
            self.cell_types_raw = np.unique(cell_labels_raw)
        except KeyError:
            print('Cluster key not found!')
            return

        # Training Configuration
        self.config = {
            # Training Parameters
            "n_epochs": 500,
            "learning_rate": 2e-4,
            "neg_slope": 0.0,
            "test_iter": None,
            "save_epoch": 100,
            "n_update_noise": 25,
            "batch_size": 128,
            "early_stop": 5,
            "early_stop_thred": adata.n_vars*1e-3,
            "train_test_split": 0.7,
            "train_scaling": False,
            "train_std": False,
            "weight_sample": False,
            "sparsify": 1
        }

        self.set_device(device)
        self.split_train_test(adata.n_obs)

        self.decoder = decoder(adata,
                               cluster_key,
                               tkey,
                               embed_key,
                               self.train_idx,
                               param_key,
                               device=self.device,
                               checkpoint=checkpoint,
                               **graph_param)

        # class attributes for training
        self.loss_train, self.loss_test = [], []
        self.counter = 0  # Count the number of iterations
        self.n_drop = 0  # Count the number of consecutive epochs with negative/low ELBO gain

        self.timer = time.time() - t_start

    def set_device(self, device):
        if 'cuda' in device:
            if torch.cuda.is_available():
                self.device = torch.device(device)
            else:
                print('Warning: GPU not detected. Using CPU as the device.')
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')

    def split_train_test(self, N):
        rand_perm = np.random.permutation(N)
        n_train = int(N*self.config["train_test_split"])
        self.train_idx = rand_perm[:n_train]
        self.test_idx = rand_perm[n_train:]

        return

    def forward(self, t, y):
        """Evaluate the model in training.

        Arguments
        ---------

        t : `torch.tensor`
            Cell time, (N,1)
        y : `torch.tensor`
            Cell type encoded in integers, (N,1)

        Returns
        -------
        uhat, shat : `torch.tensor`
            Predicted u and s values, (N,G)
        """
        uhat, shat = self.decoder.forward(t, y, neg_slope=self.config['neg_slope'])

        return uhat, shat

    def eval_model(self, t, y, gidx=None):
        """Evaluate the model in validation/test.

        Arguments
        ---------

        t : `torch.tensor`
            Cell time, (N,1)
        y : `torch.tensor`
            Cell type encoded in integers, (N,1)
        gidx : `numpy array`, optional
            A subset of genes to compute

        Returns
        -------
        uhat, shat : `torch.tensor`
            Predicted u and s values, (N,G)
        """
        uhat, shat = self.decoder.pred_su(t, y, gidx)

        return uhat, shat

    def set_mode(self, mode):
        if mode == 'train':
            self.decoder.train()
        elif mode == 'eval':
            self.decoder.eval()
        else:
            print("Warning: mode not recognized. Must be 'train' or 'test'! ")

    ############################################################
    # Training Objective
    ############################################################

    def ode_risk(self,
                 u,
                 s,
                 uhat,
                 shat,
                 sigma_u, sigma_s,
                 weight=None):
        # 1. u,s,uhat,shat: raw and predicted counts
        # 2. sigma_u, sigma_s : standard deviation of the Gaussian likelihood (decoder)
        # 3. weight: sample weight

        neg_log_gaussian = 0.5*((uhat-u)/sigma_u).pow(2) \
            + 0.5*((shat-s)/sigma_s).pow(2) \
            + torch.log(sigma_u)+torch.log(sigma_s*2*np.pi)

        if weight is not None:
            neg_log_gaussian = neg_log_gaussian*weight.view(-1, 1)

        return torch.mean(torch.sum(neg_log_gaussian, 1))

    def train_epoch(self,
                    train_loader,
                    test_set,
                    optimizer):
        self.set_mode('train')
        stop_training = False

        for i, batch in enumerate(train_loader):
            if self.counter == 1 or self.counter % self.config["test_iter"] == 0:
                ll_test = self.test(test_set, self.counter)
                if len(self.loss_test) > 0:
                    if ll_test - self.loss_test[-1] <= self.config["early_stop_thred"]:
                        self.n_drop = self.n_drop + 1
                    else:
                        self.n_drop = 0
                self.loss_test.append(ll_test)
                self.set_mode('train')
                if self.n_drop >= self.config["early_stop"] and self.config["early_stop"] > 0:
                    stop_training = True
                    break

            optimizer.zero_grad()
            xbatch, label_batch, tbatch = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
            u, s = xbatch[:, :xbatch.shape[1]//2], xbatch[:, xbatch.shape[1]//2:]

            uhat, shat = self.forward(tbatch, label_batch.squeeze())

            loss = self.ode_risk(u, s,
                                 uhat, shat,
                                 torch.exp(self.decoder.sigma_u), torch.exp(self.decoder.sigma_s))
            loss.backward()
            optimizer.step()

            self.loss_train.append(loss.detach().cpu().item())
            self.counter = self.counter + 1
        return stop_training

    def load_config(self, config):
        # We don't have to specify all the hyperparameters. Just pass the ones we want to modify.
        for key in config:
            if key in self.config:
                self.config[key] = config[key]
            else:
                self.config[key] = config[key]
                print(f"Added new hyperparameter: {key}")
        if self.config["train_scaling"]:
            self.decoder.scaling.requires_grad = True
        if self.config["train_std"]:
            self.decoder.sigma_u.requires_grad = True
            self.decoder.sigma_s.requires_grad = True

    def print_weight(self):
        w = self.decoder.w.cpu().numpy()
        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None,
                               'display.precision', 3,
                               'display.chop_threshold', 1e-3,
                               'display.width', 200):
            w_dic = {}
            cell_types = list(self.decoder.label_dic.keys())
            for i in range(self.decoder.Ntype):
                w_dic[self.decoder.label_dic_rev[i]] = w[:, i]
            w_df = pd.DataFrame(w_dic, index=pd.Index(cell_types))
            print(w_df)

    def update_std_noise(self, train_set):
        G = train_set.G
        Uhat, Shat, ll = self.pred_all(train_set.data,
                                       torch.tensor(train_set.time).double().to(self.device),
                                       train_set.labels,
                                       train_set.N,
                                       train_set.G,
                                       np.array(range(G)))
        self.decoder.sigma_u = nn.Parameter(torch.tensor(np.log((Uhat-train_set.data[:, :G]).std(0)+1e-10),
                                            device=self.device))
        self.decoder.sigma_s = nn.Parameter(torch.tensor(np.log((Shat-train_set.data[:, G:]).std(0)+1e-10),
                                            device=self.device))
        return

    def train(self,
              adata,
              tkey,
              cluster_key,
              config={},
              plot=False,
              gene_plot=[],
              figure_path="figures"):
        """Train the model.

        Arguments
        ---------
        adata : :class:`anndata.AnnData`
        tkey : str
            Key in adata.obs storing the latent cell time
        cluster_key : str
            Key in adata.obs storing the cell type annotation.
        config : dictionary, optional
            Contains training hyperparameters.
            All hyperparameters have default values, so users don't need to set every one of them.
        plot : bool, optional
            Whether to generate gene plots during training. Used mainly for debugging.
        gene_plot : `numpy array` or string list, optional
            Genes to plot during training. Effective only if 'plot' is set to True
        figure_path : str, optional
            Path to the folder to save figures
        embed : str, optional
            2D embedding name in .obsm for visualization.
        """
        self.tkey = tkey
        self.cluster_key = cluster_key
        self.load_config(config)

        if self.config["train_scaling"]:
            self.decoder.scaling.requires_grad = True
        if self.config["train_std"]:
            self.decoder.sigma_u.requires_grad = True
            self.decoder.sigma_s.requires_grad = True

        print("------------------------ Train a Branching ODE ------------------------")
        # Get data loader
        X = np.concatenate((adata.layers['Mu'], adata.layers['Ms']), 1)
        X = X.astype(float)

        cell_labels_raw = (adata.obs[cluster_key].to_numpy() if cluster_key in adata.obs else
                           np.array(['Unknown' for i in range(adata.n_obs)]))
        cell_labels = str2int(cell_labels_raw, self.decoder.label_dic)
        t = adata.obs[tkey].to_numpy()
        self.print_weight()
        print("*********        Creating Training/Validation Datasets        *********")
        train_set = SCTimedData(X[self.train_idx], cell_labels[self.train_idx], t[self.train_idx])
        test_set = None
        if len(self.test_idx) > 0:
            test_set = SCTimedData(X[self.test_idx], cell_labels[self.test_idx], t[self.test_idx])
        data_loader = torch.utils.data.DataLoader(train_set, batch_size=self.config["batch_size"], shuffle=True)
        # Automatically set test iteration if not given
        if self.config["test_iter"] is None:
            self.config["test_iter"] = len(self.train_idx)//self.config["batch_size"]*2
        print("*********                      Finished.                      *********")

        gind, gene_plot = get_gene_index(adata.var_names, gene_plot)
        os.makedirs(figure_path, exist_ok=True)

        # define optimizer
        print("*********                 Creating optimizers                 *********")
        param_ode = [self.decoder.alpha, self.decoder.beta, self.decoder.gamma,
                     self.decoder.t_trans, self.decoder.u0, self.decoder.s0]
        if self.config["train_scaling"]:
            param_ode = param_ode+[self.decoder.scaling]
        if self.config["train_std"]:
            param_ode = param_ode+[self.decoder.sigma_u, self.decoder.sigma_s]

        optimizer = torch.optim.Adam(param_ode, lr=self.config["learning_rate"])
        print("*********                      Finished.                      *********")

        # Main Training Process
        print("*********                    Start training                   *********")
        print(f"Total Number of Iterations Per Epoch: {len(data_loader)}, test iteration: {self.config['test_iter']}")
        n_epochs = self.config["n_epochs"]
        start = time.time()

        for epoch in range(n_epochs):
            stop_training = self.train_epoch(data_loader, test_set, optimizer)

            if plot and (epoch == 0 or (epoch+1) % self.config["save_epoch"] == 0):
                ll_train = self.test(train_set,
                                     f"train{epoch+1}",
                                     gind,
                                     gene_plot,
                                     True,
                                     figure_path)
                self.set_mode('train')
                ll = -np.inf if len(self.loss_test) == 0 else self.loss_test[-1]
                print(f"Epoch {epoch+1}: Train Log Likelihood = {ll_train:.3f}, \
                      Test Log Likelihood = {ll:.3f}, \t \
                      Total Time = {convert_time(time.time()-start)}")

            if (epoch+1) % self.config["n_update_noise"] == 0:
                self.update_std_noise(train_set)

            if stop_training:
                print(f"*********           Early Stop Triggered at epoch {epoch+1}.            *********")
                break

        if plot:
            plot_train_loss(self.loss_train,
                            range(1, len(self.loss_train)+1),
                            save=f'{figure_path}/train_loss_brode.png')
            if self.config["test_iter"] > 0:
                plot_test_loss(self.loss_test,
                               [i*self.config["test_iter"] for i in range(1, len(self.loss_test)+1)],
                               save=f'{figure_path}/test_loss_brode.png')

        self.timer = self.timer + (time.time()-start)
        print(f"*********              Finished. Total Time = {convert_time(self.timer)}             *********")
        return

    def pred_all(self, data, t, cell_labels, N, G, gene_idx=None):
        # data [N x 2G] : input mRNA count
        # mode : train or test or both
        # gene_idx : gene index, used for reducing unnecessary memory usage
        if gene_idx is None:
            Uhat, Shat = None, None
        else:
            Uhat, Shat = np.zeros((N, len(gene_idx))), np.zeros((N, len(gene_idx)))
        ll = 0
        with torch.no_grad():
            B = min(N//5, 5000)
            Nb = N // B
            for i in range(Nb):
                uhat, shat = self.eval_model(t[i*B:(i+1)*B], torch.tensor(cell_labels[i*B:(i+1)*B]).to(self.device))
                if gene_idx is not None:
                    Uhat[i*B:(i+1)*B] = uhat[:, gene_idx].cpu().numpy()
                    Shat[i*B:(i+1)*B] = shat[:, gene_idx].cpu().numpy()
                loss = self.ode_risk(torch.tensor(data[i*B:(i+1)*B, :G]).double().to(self.device),
                                     torch.tensor(data[i*B:(i+1)*B, G:]).double().to(self.device),
                                     uhat, shat,
                                     torch.exp(self.decoder.sigma_u), torch.exp(self.decoder.sigma_s))
                ll = ll - (B/N)*loss
            if N > B*Nb:
                uhat, shat = self.eval_model(t[B*Nb:], torch.tensor(cell_labels[B*Nb:]).to(self.device))
                if gene_idx is not None:
                    Uhat[Nb*B:] = uhat[:, gene_idx].cpu().numpy()
                    Shat[Nb*B:] = shat[:, gene_idx].cpu().numpy()
                loss = self.ode_risk(torch.tensor(data[B*Nb:, :G]).double().to(self.device),
                                     torch.tensor(data[B*Nb:, G:]).double().to(self.device),
                                     uhat, shat,
                                     torch.exp(self.decoder.sigma_u), torch.exp(self.decoder.sigma_s))
                ll = ll - ((N-B*Nb)/N)*loss
        return Uhat, Shat, ll.cpu().item()

    def test(self,
             dataset,
             testid=0,
             gind=None,
             gene_plot=[],
             plot=False,
             path='figures',
             **kwargs):

        self.set_mode('eval')
        Uhat, Shat, ll = self.pred_all(dataset.data,
                                       torch.tensor(dataset.time).double().to(self.device),
                                       dataset.labels,
                                       dataset.N,
                                       dataset.G,
                                       gind)
        cell_labels_raw = int2str(dataset.labels, self.decoder.label_dic_rev)
        if plot:
            for i in range(len(gene_plot)):
                idx = gind[i]
                plot_sig(dataset.time.squeeze(),
                         dataset.data[:, idx], dataset.data[:, idx+dataset.G],
                         Uhat[:, i], Shat[:, i],
                         cell_labels_raw,
                         gene_plot[i],
                         save=f"{path}/sig-{gene_plot[i]}-{testid}.png",
                         sparsify=self.config["sparsify"],
                         t_trans=self.decoder.t_trans.detach().cpu().exp().numpy())

        return ll

    def save_model(self, file_path, name='brode'):
        """Save the decoder parameters to a .pt file.

        Arguments
        ---------

        file_path : str
            Path to the folder for saving the model parameters
        name : str
            Name of the saved file.
        """
        os.makedirs(file_path, exist_ok=True)
        torch.save(self.decoder.state_dict(), f"{file_path}/{name}.pt")

    def save_anndata(self, adata, key, file_path, file_name=None):
        """Save the ODE parameters and cell time to the anndata object and write it to disk.

        Arguments
        ---------

        adata : :class:`anndata`
        key : str
            Key name used to store all results
        file_path : str
            Path to the folder for saving the output file
        file_name : str, optional
            Name of the output file. If set to None, the original anndata object will be overwritten,
            but nothing will be saved to disk.
        """
        self.set_mode('eval')
        os.makedirs(file_path, exist_ok=True)

        X = np.concatenate((adata.layers['Mu'], adata.layers['Ms']), 1)
        t = adata.obs[self.tkey].to_numpy()
        label_int = str2int(adata.obs[self.cluster_key].to_numpy(), self.decoder.label_dic)

        adata.obs[f"{key}_time"] = t
        adata.obs[f"{key}_label"] = label_int
        adata.varm[f"{key}_alpha"] = np.exp(self.decoder.alpha.detach().cpu().numpy()).T
        adata.varm[f"{key}_beta"] = np.exp(self.decoder.beta.detach().cpu().numpy()).T
        adata.varm[f"{key}_gamma"] = np.exp(self.decoder.gamma.detach().cpu().numpy()).T
        adata.uns[f"{key}_t_trans"] = np.exp(self.decoder.t_trans.detach().cpu().numpy())
        adata.varm[f"{key}_u0"] = np.exp(self.decoder.u0.detach().cpu().numpy()).T
        adata.varm[f"{key}_s0"] = np.exp(self.decoder.s0.detach().cpu().numpy()).T
        adata.var[f"{key}_scaling"] = np.exp(self.decoder.scaling.detach().cpu().numpy())
        adata.var[f"{key}_sigma_u"] = np.exp(self.decoder.sigma_u.detach().cpu().numpy())
        adata.var[f"{key}_sigma_s"] = np.exp(self.decoder.sigma_s.detach().cpu().numpy())
        adata.uns[f"{key}_w"] = self.decoder.w.detach().cpu().numpy()

        Uhat, Shat, ll = self.pred_all(X,
                                       torch.tensor(t.reshape(-1, 1)).to(self.device),
                                       label_int,
                                       adata.n_obs,
                                       adata.n_vars,
                                       np.array(range(adata.n_vars)))
        adata.layers[f"{key}_uhat"] = Uhat
        adata.layers[f"{key}_shat"] = Shat

        adata.uns[f"{key}_train_idx"] = self.train_idx
        adata.uns[f"{key}_test_idx"] = self.test_idx
        adata.uns[f"{key}_label_dic"] = self.decoder.label_dic
        adata.uns[f"{key}_run_time"] = self.timer

        rna_velocity_brode(adata, key)

        if file_name is not None:
            adata.write_h5ad(f"{file_path}/{file_name}")

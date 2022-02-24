import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
import time
from velovae.plotting import plot_sig, plot_sig_, plot_time, plot_train_loss, plot_test_loss

from .model_util import histEqual, initParams, getTsGlobal, reinitParams, convertTime, getGeneIndex, optimal_transport_duality_gap
from .model_util import predSU, ode, odeNumpy, knnX0, knnX0_alt, knnx0_bin
from .TrainingData import SCData
from .VanillaVAE import VanillaVAE, kl_gaussian, kl_uniform
from .velocity import rnaVelocityVAEpp

##############################################################
# VAE+
##############################################################
class encoder(nn.Module):
    def __init__(self, Cin, Cz, N1=500, N2=250, device=torch.device('cpu'), checkpoint=None):
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
        
        self.fc_mu_t, self.spt1 = nn.Linear(N2,1).to(device), nn.Softplus()
        self.fc_std_t, self.spt2 = nn.Linear(N2,1).to(device), nn.Softplus()
        self.fc_mu_z = nn.Linear(N2,Cz).to(device)
        self.fc_std_z, self.spt3 = nn.Linear(N2,Cz).to(device), nn.Softplus()
        
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
    
    def forward(self, data_in):
        h = self.net(data_in)
        mu_tx, std_tx = self.spt1(self.fc_mu_t(h)), self.spt2(self.fc_std_t(h))
        mu_zx, std_zx = self.fc_mu_z(h), self.spt3(self.fc_std_z(h))
        return mu_tx, std_tx, mu_zx, std_zx

class decoder(nn.Module):
    def __init__(self, 
                 adata, 
                 Tmax,
                 train_idx,
                 Cz, 
                 N1=250, 
                 N2=500, 
                 p=98, 
                 init_ton_zero=False,
                 device=torch.device('cpu'), 
                 init_method ="steady", 
                 init_key=None, 
                 checkpoint=None):
        super(decoder,self).__init__()
        G = adata.n_vars
        self.fc1 = nn.Linear(Cz, N1).to(device)
        self.bn1 = nn.BatchNorm1d(num_features=N1).to(device)
        self.dpt1 = nn.Dropout(p=0.2).to(device)
        self.fc2 = nn.Linear(N1, N2).to(device)
        self.bn2 = nn.BatchNorm1d(num_features=N2).to(device)
        self.dpt2 = nn.Dropout(p=0.2).to(device)
        
        self.fc_out1 = nn.Linear(N2, G).to(device)
        
        self.net_rho = nn.Sequential(self.fc1, self.bn1, nn.LeakyReLU(), self.dpt1,
                                     self.fc2, self.bn2, nn.LeakyReLU(), self.dpt2)
        
        self.fc3 = nn.Linear(Cz, N1).to(device)
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
            if(init_method == "existing" and init_key is not None):
                self.alpha = nn.Parameter(torch.tensor(np.log(adata.var[f"{init_key}_alpha"].to_numpy()), device=device).float())
                self.beta = nn.Parameter(torch.tensor(np.log(adata.var[f"{init_key}_beta"].to_numpy()), device=device).float())
                self.gamma = nn.Parameter(torch.tensor(np.log(adata.var[f"{init_key}_gamma"].to_numpy()), device=device).float())
                self.scaling = nn.Parameter(torch.tensor(np.log(adata.var[f"{init_key}_scaling"].to_numpy()), device=device).float())
                self.ton = nn.Parameter(torch.tensor(np.log(adata.var[f"{init_key}_ton"].to_numpy()), device=device).float())
                self.sigma_u = nn.Parameter(torch.tensor(np.log(adata.var[f"{init_key}_sigma_u"].to_numpy()), device=device).float())
                self.sigma_s = nn.Parameter(torch.tensor(np.log(adata.var[f"{init_key}_sigma_s"].to_numpy()), device=device).float())
            elif(init_method == "random"):
                print("Random Initialization.")
                #alpha, beta, gamma, scaling, toff, u0, s0, sigma_u, sigma_s, T, Rscore = initParams(X,p,fit_scaling=True)
                
                self.alpha = nn.Parameter(torch.normal(0.0, 1.0, size=(U.shape[1],), device=device).float())
                self.beta =  nn.Parameter(torch.normal(0.0, 1.0, size=(U.shape[1],), device=device).float())
                self.gamma = nn.Parameter(torch.normal(0.0, 1.0, size=(U.shape[1],), device=device).float())
                self.ton = torch.nn.Parameter(torch.ones(adata.n_vars, device=device).float()*(-10))
                self.scaling = nn.Parameter(torch.tensor(np.log(np.std(U,0)/np.std(S,0)), device=device).float())
                self.sigma_u = nn.Parameter(torch.tensor(np.log(np.std(U,0)), device=device).float())
                self.sigma_s = nn.Parameter(torch.tensor(np.log(np.std(S,0)), device=device).float())
            elif(init_method == "tprior"):
                print("Initialization using prior time.")
                alpha, beta, gamma, scaling, toff, u0, s0, sigma_u, sigma_s, T, Rscore = initParams(X,p,fit_scaling=True)
                t_prior = adata.obs[init_key].to_numpy()
                t_prior = t_prior[train_idx]
                std_t = np.std(t_prior)*0.2
                self.t_init = np.random.uniform(t_prior-std_t, t_prior+std_t)
                self.t_init -= self.t_init.min()
                self.t_init = self.t_init
                self.t_init = self.t_init/self.t_init.max()*Tmax
                toff = getTsGlobal(self.t_init, U/scaling, S, 95)
                alpha, beta, gamma, ton = reinitParams(U/scaling, S, self.t_init, toff)
                
                self.alpha = nn.Parameter(torch.tensor(np.log(alpha), device=device).float())
                self.beta = nn.Parameter(torch.tensor(np.log(beta), device=device).float())
                self.gamma = nn.Parameter(torch.tensor(np.log(gamma), device=device).float())
                self.scaling = nn.Parameter(torch.tensor(np.log(scaling), device=device).float())
                self.sigma_u = nn.Parameter(torch.tensor(np.log(sigma_u), device=device).float())
                self.sigma_s = nn.Parameter(torch.tensor(np.log(sigma_s), device=device).float())
                self.ton = nn.Parameter((torch.ones(adata.n_vars, device=device)*(-10)).float()) if init_ton_zero else nn.Parameter(torch.tensor(np.log(ton+1e-10), device=device).float())
            else:
                print("Initialization using the steady-state and dynamical models.")
                alpha, beta, gamma, scaling, toff, u0, s0, sigma_u, sigma_s, T, Rscore = initParams(X,p,fit_scaling=True)
                if(init_key is not None):
                    self.t_init = adata.obs[init_key].to_numpy()
                else:
                    T = T+np.random.rand(T.shape[0],T.shape[1]) * 1e-3
                    T_eq = np.zeros(T.shape)
                    Nbin = T.shape[0]//50+1
                    for i in range(T.shape[1]):
                        T_eq[:, i] = histEqual(T[:, i], Tmax, 0.9, Nbin)
                    self.t_init = np.quantile(T_eq,0.5,1)
                toff = getTsGlobal(self.t_init, U/scaling, S, 95)
                alpha, beta, gamma, ton = reinitParams(U/scaling, S, self.t_init, toff)
                
                self.alpha = nn.Parameter(torch.tensor(np.log(alpha), device=device).float())
                self.beta = nn.Parameter(torch.tensor(np.log(beta), device=device).float())
                self.gamma = nn.Parameter(torch.tensor(np.log(gamma), device=device).float())
                self.scaling = nn.Parameter(torch.tensor(np.log(scaling), device=device).float())
                self.ton = nn.Parameter((torch.ones(adata.n_vars, device=device)*(-10)).float()) if init_ton_zero else nn.Parameter(torch.tensor(np.log(ton+1e-10), device=device).float())
                self.sigma_u = nn.Parameter(torch.tensor(np.log(sigma_u), device=device).float())
                self.sigma_s = nn.Parameter(torch.tensor(np.log(sigma_s), device=device).float())
        

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
        
    def forward(self, t, z, u0=None, s0=None, t0=None, neg_slope=0.0):
        if(u0 is None or s0 is None or t0 is None):
            rho = F.sigmoid(self.fc_out1(self.net_rho(z)))
            alpha = self.alpha.exp()*rho
            #Uhat, Shat = ode(t, alpha, torch.exp(self.beta), torch.exp(self.gamma), self.ton.exp(), self.toff.exp(), neg_slope)
            Uhat, Shat = predSU(F.leaky_relu(t - self.ton.exp(), neg_slope), 0, 0, alpha, self.beta.exp(), self.gamma.exp())
        else:
            rho = F.sigmoid(self.fc_out2(self.net_rho2(z)))
            alpha = self.alpha.exp()*rho
            Uhat, Shat = predSU(F.leaky_relu(t-t0, neg_slope), u0/self.scaling.exp(), s0, alpha, self.beta.exp(), self.gamma.exp())
        Uhat = Uhat * torch.exp(self.scaling)
        return nn.functional.relu(Uhat), nn.functional.relu(Shat)

class VanillaVAEpp(VanillaVAE):
    def __init__(self, 
                 adata, 
                 Tmax, 
                 Cz, 
                 device='cpu', 
                 hidden_size=(500, 250, 250, 500), 
                 init_method="steady", 
                 init_key=None,
                 tprior=None, 
                 init_ton_zero=False,
                 time_distribution="uniform",
                 checkpoints=[None, None]):
        """
        adata: AnnData Object
        Tmax: (float/int) Time Range 
        """
        try:
            U,S = adata.layers['Mu'], adata.layers['Ms']
        except KeyError:
            print('Unspliced/Spliced count matrices not found in the layers! Exit the program...')
        
        #Training Configuration
        self.config = {
            #Model Parameters
            "tmax":Tmax,
            "Cz":Cz,
            "hidden_size":hidden_size,
            "init_method":init_method,
            "init_key":init_key,
            "tprior":tprior,
            "n_neighbors":30,
            "dt": (0.03,0.05),
            "n_bin": None,
            
            #Training Parameters
            "n_epochs":250, 
            "n_epochs_post":250,
            "learning_rate":1e-4, 
            "learning_rate_ode":1e-4, 
            "learning_rate_post":2e-5,
            "lambda":1e-3, 
            "lambda_rho":1e-3,
            "reg_t":1.0, 
            "reg_z":1.0, 
            "neg_slope":0.0,
            "test_iter":100, 
            "save_epoch":100, 
            "n_warmup":5,
            "batch_size":128, 
            "early_stop":5,
            "train_test_split":0.7,
            "K_alt":0, 
            "train_scaling":False, 
            "train_std":False, 
            "train_ton":False,
            "weight_sample":False,
            "sparsify":1
        }
        
        self.setDevice(device)
        self.splitTrainTest(adata.n_obs)
        
        G = adata.n_vars
        self.Cz = Cz
        try:
            self.encoder = encoder(2*G, Cz, hidden_size[0], hidden_size[1], self.device, checkpoint=checkpoints[0]).float()
        except IndexError:
            print('Please provide two dimensions!')
        
        self.decoder = decoder(adata, 
                               Tmax, 
                               self.train_idx,
                               Cz, 
                               N1=hidden_size[2], 
                               N2=hidden_size[3], 
                               init_ton_zero=init_ton_zero,
                               device=self.device, 
                               init_method = init_method,
                               init_key = init_key,
                               checkpoint=checkpoints[1]).float()
        self.Tmax=Tmax
        self.time_distribution = time_distribution
        self.getPrior(adata, time_distribution, Tmax, tprior)
        
        self.p_z = torch.stack([torch.zeros(U.shape[0],Cz), torch.ones(U.shape[0],Cz)*0.01]).double().to(self.device)
        
        self.use_knn = False
        self.u0 = None
        self.s0 = None
        self.t0 = None
    
    def forward(self, data_in, u0=None, s0=None, t0=None):
        data_in_scale = torch.cat((data_in[:,:data_in.shape[1]//2]/torch.exp(self.decoder.scaling), data_in[:,data_in.shape[1]//2:]),1)
        mu_t, std_t, mu_z, std_z = self.encoder.forward(data_in_scale)
        t = self.sample(mu_t, std_t)
        z = self.sample(mu_z, std_z)
         
        uhat, shat = self.decoder.forward(t, z, u0, s0, t0, neg_slope=self.config["neg_slope"])
        return mu_t, std_t, mu_z, std_z, t, z, uhat, shat
    
    def evalModel(self, data_in, u0=None, s0=None, t0=None):
        data_in_scale = torch.cat((data_in[:,:data_in.shape[1]//2]/torch.exp(self.decoder.scaling), data_in[:,data_in.shape[1]//2:]),1)
        mu_t, std_t, mu_z, std_z = self.encoder.forward(data_in_scale)
         
        uhat, shat = self.decoder.forward(mu_t, mu_z, u0=u0, s0=s0, t0=t0, neg_slope=0.0)
        return mu_t, std_t, mu_z, std_z, uhat, shat
    
    def VAERisk(self, 
                q_tx, p_t, 
                q_zx, p_z, 
                u, s, uhat, shat, 
                sigma_u, sigma_s, 
                weight=None, b=1.0, c=1.0):
        """
        This is the negative ELBO.
        q_tx: parameters of time posterior
        p_t:  parameters of time prior
        q_zx: parameters of cell state posterior
        p_z:  parameters of cell state prior
        u , s : [B x G] input data
        uhat, shat: [B x G] prediction by the ODE model
        sigma_u, sigma_s : parameter of the Gaussian distribution
        """
        
        kldt = self.kl_time(q_tx[0], q_tx[1], p_t[0], p_t[1])
        kldz = kl_gaussian(q_zx[0], q_zx[1], p_z[0], p_z[1])
        
        #u and sigma_u has the original scale
        logp = -0.5*((u-uhat)/sigma_u).pow(2)-0.5*((s-shat)/sigma_s).pow(2)-torch.log(sigma_u)-torch.log(sigma_s*2*np.pi)
        
        if( weight is not None):
            logp = logp*weight
        err_rec = torch.mean(torch.sum(logp,1))
        
        #print(kldt.detach().cpu().item(), err_rec.detach().cpu().item())
        return (- err_rec + b*kldt + c*kldz)
    
    def train_epoch(self, train_loader, test_set, optimizer, optimizer2=None, K=1, reg_t=1.0, reg_z=1.0):
        """
        Training in each epoch
        train_loader: Data loader of the input data
        test_set: validation dataset
        optimizer, optimizer2(optional): from torch.optim
        K: alternatingly update optimizer and optimizer2
        """
        iterX = iter(train_loader)
        B = len(iterX)
        train_loss, test_loss = [], []
        for i in range(B):
            self.counter = self.counter + 1
            if( self.counter % self.config["test_iter"] == 0):
                elbo_test = self.test(test_set, None, self.counter, True)
                test_loss.append(elbo_test)
                self.setMode('train')
                print(f"Iteration {self.counter}: Test ELBO = {elbo_test:.3f}")
            
            optimizer.zero_grad()
            if(optimizer2 is not None):
                optimizer2.zero_grad()
            batch = iterX.next()
            xbatch, weight, idx = batch[0].float().to(self.device), batch[2].float().to(self.device), batch[3]
            u = xbatch[:,:xbatch.shape[1]//2]
            s = xbatch[:,xbatch.shape[1]//2:]
            
            u0 = batch[4].float().to(self.device) if self.use_knn else None
            s0 = batch[5].float().to(self.device) if self.use_knn else None
            t0 = batch[6].float().to(self.device) if self.use_knn else None
            
            mu_tx, std_tx, mu_zx, std_zx, t, z, uhat, shat = self.forward(xbatch, u0, s0, t0)
            
            loss = self.VAERisk((mu_tx, std_tx), self.p_t[:,self.train_idx[idx],:],
                                (mu_zx, std_zx), self.p_z[:,self.train_idx[idx],:],
                                u, s, 
                                uhat, shat, 
                                torch.exp(self.decoder.sigma_u), torch.exp(self.decoder.sigma_s), 
                                None, reg_t, reg_z)
            
            loss.backward()
            optimizer.step()
            if( optimizer2 is not None and ((i+1) % K == 0 or i==B-1)):
                optimizer2.step()
            
            train_loss.append(loss.detach().cpu().item())
            
        return train_loss, test_loss
    
    def updateX0(self, U, S, n_bin=None):
        """
        Estimate the initial conditions using KNN
        U is unscaled
        """
        start = time.time()
        self.setMode('eval')
        out, elbo = self.predAll(np.concatenate((U,S),1), "both", ["t","z"])
        t, z = out[0], out[2]
        dt = (self.config["dt"][0]*(t.max()-t.min()), self.config["dt"][1]*(t.max()-t.min()))
        if(n_bin is None):
            print(f"Cell-wise KNN Estimation.")
            u0, s0, t0 = knnX0_alt(U[self.train_idx], S[self.train_idx], t[self.train_idx], z[self.train_idx], t, z, dt, self.config["n_neighbors"])
        else:
            print(f"Fast KNN Estimation with {n_bin} time bins.")
            u0, s0, t0 = knnx0_bin(U[self.train_idx], S[self.train_idx], t[self.train_idx], z[self.train_idx], t, z, dt, self.config["n_neighbors"])
        
        print(f"Finished. Actual Time: {convertTime(time.time()-start)}")
        return u0, s0, t0.reshape(-1,1)
        
    def train(self, 
              adata, 
              config={}, 
              plot=True, 
              gene_plot=[], 
              figure_path="figures", 
              embed="umap"):
        
        self.loadConfig(config)
        
        print("--------------------------- Train a VeloVAE ---------------------------")
        #Get data loader
        X = np.concatenate((adata.layers['Mu'], adata.layers['Ms']), 1)
        X = X.astype(float)
        try:
            Xembed = adata.obsm[f"X_{embed}"]
        except KeyError:
            print("Embedding not found! Please run the corresponding preprocessing step!")
        
        cell_labels_raw = adata.obs["clusters"].to_numpy() if "clusters" in adata.obs else np.array(['Unknown' for i in range(adata.n_obs)])
        
        print("*********        Creating Training/Validation Datasets        *********")
        train_set = SCData(X[self.train_idx], cell_labels_raw[self.train_idx], self.decoder.Rscore[self.train_idx]) if self.config['weight_sample'] else SCData(X[self.train_idx], cell_labels_raw[self.train_idx])
        test_set = None
        if(len(self.test_idx)>0):
            test_set = SCData(X[self.test_idx], cell_labels_raw[self.test_idx], self.decoder.Rscore[self.test_idx]) if self.config['weight_sample'] else SCData(X[self.test_idx], cell_labels_raw[self.test_idx])
        data_loader = torch.utils.data.DataLoader(train_set, batch_size=self.config["batch_size"], shuffle=True)
        print("*********                      Finished.                      *********")
        
        gind, gene_plot = getGeneIndex(adata.var_names, gene_plot)
        os.makedirs(figure_path, exist_ok=True)
        
        #define optimizer
        print("*********                 Creating optimizers                 *********")
        param_nn = list(self.encoder.parameters())+list(self.decoder.net_rho.parameters())+list(self.decoder.fc_out1.parameters())
        param_ode = [self.decoder.alpha, self.decoder.beta, self.decoder.gamma] 
        if(self.config['train_ton']):
            param_ode.append(self.decoder.ton)
        if(self.config['train_scaling']):
            param_ode = param_ode+[self.decoder.scaling]
        if(self.config['train_std']):
            param_ode = param_ode+[self.decoder.sigma_u, self.decoder.sigma_s]

        optimizer = torch.optim.Adam(param_nn, lr=self.config["learning_rate"], weight_decay=self.config["lambda"])
        optimizer_ode = torch.optim.Adam(param_ode, lr=self.config["learning_rate_ode"])
        print("*********                      Finished.                      *********")
      
        #Main Training Process
        print("*********                    Start training                   *********")
        print("*********                      Stage  1                       *********")
        print(f"Total Number of Iterations Per Epoch: {len(data_loader)}")
        
        n_epochs = self.config["n_epochs"]
        loss_train, loss_test = [],[]
        self.counter = 0 #Count the number of iterations
        n_drop = 0 #Count the number of consecutive iterations with little decrease in loss
        
        start = time.time()
        for epoch in range(n_epochs):
            #Train the encoder
            if(self.config["K_alt"]==0):
                loss_train_epoch, loss_test_epoch = self.train_epoch(data_loader, test_set, optimizer, reg_t=self.config["reg_t"], reg_z=self.config["reg_z"])
                if(epoch>=self.config["n_warmup"]):
                    loss_train_epoch, loss_test_epoch = self.train_epoch(data_loader, test_set, optimizer_ode, reg_t=self.config["reg_t"], reg_z=self.config["reg_z"])
            else:
                if(epoch>=self.config["n_warmup"]):
                    loss_train_epoch, loss_test_epoch = self.train_epoch(data_loader, test_set, optimizer, optimizer_ode, self.config["K_alt"], self.config["reg_t"], self.config["reg_z"])
                else:
                    loss_train_epoch, loss_test_epoch = self.train_epoch(data_loader, test_set, optimizer, None, self.config["K_alt"], self.config["reg_t"], self.config["reg_z"])
            
            for loss in loss_train_epoch:
                loss_train.append(loss)
            for loss in loss_test_epoch:
                loss_test.append(loss)
            
            if(plot and (epoch==0 or (epoch+1) % self.config["save_epoch"] == 0)):
                elbo_train = self.test(train_set,
                                       Xembed[self.train_idx],
                                       f"train{epoch+1}", 
                                       False,
                                       gind, 
                                       gene_plot,
                                       True, 
                                       figure_path)
                self.setMode('train')
            
            if(len(loss_test)>1):
                n_drop = n_drop + 1 if (loss_test[-1]-loss_test[-2]<=adata.n_vars*1e-3) else 0
                if(n_drop >= self.config["early_stop"] and self.config["early_stop"]>0):
                    print(f"*********       Stage 1: Early Stop Triggered at epoch {epoch+1}.       *********")
                    break
                    
        n_stage1 = epoch+1
        n_test1 = len(loss_test)
        
        print("*********                      Stage  2                       *********")
        self.encoder.eval()
        u0, s0, t0 = self.updateX0(adata.layers['Mu'], adata.layers['Ms'], self.config["n_bin"])
        self.u0 = u0
        self.s0 = s0
        self.t0 = t0
        train_set.u0 = u0[self.train_idx]
        train_set.s0 = s0[self.train_idx]
        train_set.t0 = t0[self.train_idx]
        if(test_set is not None):
            test_set.u0 = u0[self.test_idx]
            test_set.s0 = s0[self.test_idx]
            test_set.t0 = t0[self.test_idx]
        
        self.use_knn = True
        self.decoder.init_weights()
        #Plot the initial conditions
        if(plot):
            plot_time(self.t0.squeeze(), Xembed, f"{figure_path}/t0.png")
            for i in range(len(gind)):
                idx = gind[i]
                t0_plot = self.t0[self.train_idx].squeeze()
                u0_plot = self.u0[self.train_idx,idx]
                s0_plot = self.s0[self.train_idx,idx]
                plot_sig_(t0_plot, 
                         u0_plot, s0_plot, 
                         cell_labels=train_set.labels,
                         title=gene_plot[i], 
                         figname=f"{figure_path}/{gene_plot[i]}-x0.png")
        
        param_post = list(self.decoder.net_rho2.parameters())+list(self.decoder.fc_out2.parameters())
        optimizer_post = torch.optim.Adam(param_post, lr=self.config["learning_rate_post"], weight_decay=self.config["lambda_rho"])
        n_drop = 0
        for epoch in range(self.config["n_epochs_post"]):
            if(self.config["K_alt"]==0):
                loss_train_epoch, loss_test_epoch = self.train_epoch(data_loader, test_set, optimizer_post, reg_t=self.config["reg_t"], reg_z=self.config["reg_z"])
                if(epoch>=self.config["n_warmup"]):
                    loss_train_epoch, loss_test_epoch = self.train_epoch(data_loader, test_set, optimizer_ode, reg_t=self.config["reg_t"], reg_z=self.config["reg_z"])
            else:
                if(epoch>=self.config["n_warmup"]):
                    loss_train_epoch, loss_test_epoch = self.train_epoch(data_loader, test_set, optimizer_post, optimizer_ode, self.config["K_alt"], self.config["reg_t"], self.config["reg_z"])
                else:
                    loss_train_epoch, loss_test_epoch = self.train_epoch(data_loader, test_set, optimizer_post, None, self.config["K_alt"], self.config["reg_t"], self.config["reg_z"])
            
            for loss in loss_train_epoch:
                loss_train.append(loss)
            for loss in loss_test_epoch:
                loss_test.append(loss)
            
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
            
            if(len(loss_test)>n_test1+1):
                n_drop = n_drop + 1 if (loss_test[-1]-loss_test[-2]<=adata.n_vars*1e-4) else 0
                if(n_drop >= self.config["early_stop"] and self.config["early_stop"]>0):
                    print(f"*********       Stage 2: Early Stop Triggered at epoch {epoch+n_stage1+1}.       *********")
                    break
                
        print(f"*********              Finished. Total Time = {convertTime(time.time()-start)}             *********")
        plot_train_loss(loss_train, range(1,len(loss_train)+1),f'{figure_path}/train_loss_velovae.png')
        plot_test_loss(loss_test, [i*self.config["test_iter"] for i in range(1,len(loss_test)+1)],f'{figure_path}/test_loss_velovae.png')
        return
    
    def predAll(self, data, mode='test', output=["uhat", "shat", "t", "z"], gene_idx=None):
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
            z_out = np.zeros((N, self.Cz))
            std_z_out = np.zeros((N, self.Cz))
        
        with torch.no_grad():
            B = min(N//5, 5000)
            Nb = N // B
            for i in range(Nb):
                data_in = torch.tensor(data[i*B:(i+1)*B]).float().to(self.device)
                if(mode=="test"):
                    start = time.time()
                    u0 = torch.tensor(self.u0[self.test_idx[i*B:(i+1)*B]], dtype=float, device=self.device) if self.use_knn else None
                    s0 = torch.tensor(self.s0[self.test_idx[i*B:(i+1)*B]], dtype=float, device=self.device) if self.use_knn else None
                    t0 = torch.tensor(self.t0[self.test_idx[i*B:(i+1)*B]], dtype=float, device=self.device) if self.use_knn else None
                    p_t = self.p_t[:,self.test_idx[i*B:(i+1)*B],:]
                    p_z = self.p_z[:,self.test_idx[i*B:(i+1)*B],:]
                    mu_tx, std_tx, mu_zx, std_zx, uhat, shat = self.evalModel(data_in, u0, s0, t0)
                elif(mode=="train"):
                    u0 = torch.tensor(self.u0[self.train_idx[i*B:(i+1)*B]], dtype=float, device=self.device) if self.use_knn else None
                    s0 = torch.tensor(self.s0[self.train_idx[i*B:(i+1)*B]], dtype=float, device=self.device) if self.use_knn else None
                    t0 = torch.tensor(self.t0[self.train_idx[i*B:(i+1)*B]], dtype=float, device=self.device) if self.use_knn else None
                    p_t = self.p_t[:,self.train_idx[i*B:(i+1)*B],:]
                    p_z = self.p_z[:,self.train_idx[i*B:(i+1)*B],:]
                    mu_tx, std_tx, mu_zx, std_zx, uhat, shat = self.evalModel(data_in, u0, s0, t0)
                else:
                    u0 = torch.tensor(self.u0[i*B:(i+1)*B], dtype=float, device=self.device) if self.use_knn else None
                    s0 = torch.tensor(self.s0[i*B:(i+1)*B], dtype=float, device=self.device) if self.use_knn else None
                    t0 = torch.tensor(self.t0[i*B:(i+1)*B], dtype=float, device=self.device) if self.use_knn else None
                    p_t = self.p_t[:,i*B:(i+1)*B,:]
                    p_z = self.p_z[:,i*B:(i+1)*B,:]
                    mu_tx, std_tx, mu_zx, std_zx, uhat, shat = self.evalModel(data_in, u0, s0, t0)
                
                loss = self.VAERisk((mu_tx, std_tx), p_t,
                                    (mu_zx, std_zx), p_z,
                                    data_in[:,:G], data_in[:,G:], 
                                    uhat, shat, 
                                    torch.exp(self.decoder.sigma_u), torch.exp(self.decoder.sigma_s), 
                                    None, 1.0, 1.0)
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
                    u0 = torch.tensor(self.u0[self.test_idx[Nb*B:]], dtype=float, device=self.device) if self.use_knn else None
                    s0 = torch.tensor(self.s0[self.test_idx[Nb*B:]], dtype=float, device=self.device) if self.use_knn else None
                    t0 = torch.tensor(self.t0[self.test_idx[Nb*B:]], dtype=float, device=self.device) if self.use_knn else None
                    p_t = self.p_t[:,self.test_idx[Nb*B:],:]
                    p_z = self.p_z[:,self.test_idx[Nb*B:],:]
                    mu_tx, std_tx, mu_zx, std_zx, uhat, shat = self.evalModel(data_in, u0, s0, t0)
                elif(mode=="train"):
                    u0 = torch.tensor(self.u0[self.train_idx[Nb*B:]], dtype=float, device=self.device) if self.use_knn else None
                    s0 = torch.tensor(self.s0[self.train_idx[Nb*B:]], dtype=float, device=self.device) if self.use_knn else None
                    t0 = torch.tensor(self.t0[self.train_idx[Nb*B:]], dtype=float, device=self.device) if self.use_knn else None
                    p_t = self.p_t[:,self.train_idx[Nb*B:],:]
                    p_z = self.p_z[:,self.train_idx[Nb*B:],:]
                    mu_tx, std_tx, mu_zx, std_zx, uhat, shat = self.evalModel(data_in, u0, s0, t0)
                else:
                    u0 = torch.tensor(self.u0[Nb*B:], dtype=float, device=self.device) if self.use_knn else None
                    s0 = torch.tensor(self.s0[Nb*B:], dtype=float, device=self.device) if self.use_knn else None
                    t0 = torch.tensor(self.t0[Nb*B:], dtype=float, device=self.device) if self.use_knn else None
                    p_t = self.p_t[:,Nb*B:,:]
                    p_z = self.p_z[:,Nb*B:,:]
                    mu_tx, std_tx, mu_zx, std_zx, uhat, shat = self.evalModel(data_in, u0, s0, t0)
                loss = self.VAERisk((mu_tx, std_tx), p_t,
                                    (mu_zx, std_zx), p_z,
                                    data_in[:,:G], data_in[:,G:], 
                                    uhat, shat, 
                                    torch.exp(self.decoder.sigma_u), torch.exp(self.decoder.sigma_s), 
                                    None, 1.0, 1.0)
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
        """
        Evaluate the model upon training/test dataset.
        """
        self.setMode('eval')
        mode = "test" if test_mode else "train"
        out, elbo = self.predAll(dataset.data, mode, ["uhat", "shat", "t"], gind)
        Uhat, Shat, t, std_t = out[0], out[1], out[2], out[3]
        
        G = dataset.data.shape[1]//2

        if(plot):
            #Plot Time
            plot_time(t, Xembed, f"{path}/{testid}-velovae.png")
            
            #Plot u/s-t and phase portrait for each gene
            for i in range(len(gind)):
                idx = gind[i]
                
                plot_sig(t.squeeze(), 
                        dataset.data[:,idx], dataset.data[:,idx+G], 
                        Uhat[:,i], Shat[:,i], 
                        dataset.labels,
                        gene_plot[i], 
                        f"{path}/sig-{gene_plot[i]}-{testid}.png",
                        sparsify=self.config['sparsify'])
        
        return elbo
    
    def saveAnnData(self, adata, key, file_path, file_name=None):
        """
        Save the ODE parameters and cell time to the anndata object and write it to disk.
        """
        self.setMode('eval')
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
        
        out, elbo = self.predAll(np.concatenate((adata.layers['Mu'], adata.layers['Ms']),1), "both", gene_idx=np.array(range(adata.n_vars)))
        Uhat, Shat, t, std_t, z, std_z = out[0], out[1], out[2], out[3], out[4], out[5]

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
                rho_batch = F.sigmoid(self.decoder.fc_out2(self.decoder.net_rho2(torch.tensor(z[i*B:(i+1)*B]).float().to(self.device))))
                rho[i*B:(i+1)*B] = rho_batch.cpu().numpy()
            rho_batch = F.sigmoid(self.decoder.fc_out2(self.decoder.net_rho2(torch.tensor(z[Nb*B:]).float().to(self.device))))
            rho[Nb*B:] = rho_batch.cpu().numpy()
        
        adata.layers[f"{key}_rho"] = rho
        
        u0, s0, t0 = self.updateX0(adata.layers['Mu'], adata.layers['Ms'], self.config["n_bin"])
        adata.obs[f"{key}_t0"] = t0.squeeze()
        adata.layers[f"{key}_u0"] = u0
        adata.layers[f"{key}_s0"] = s0

        adata.uns[f"{key}_train_idx"] = self.train_idx
        adata.uns[f"{key}_test_idx"] = self.test_idx
        
        rnaVelocityVAEpp(adata, key, use_raw=False, use_scv_genes=False)
        
        if(file_name is not None):
            adata.write_h5ad(f"{file_path}/{file_name}")
    
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import time
from velovae.plotting import plotPhase, plotSig, plotTLatent, plotTrainLoss, plotTestLoss

from .model_util import histEqual, initParams, getTsGlobal, reinitParams, convertTime, ode, getGeneIndex
from .TrainingData import SCData
from .velocity import rnaVelocityVAE


############################################################
#KL Divergence
############################################################
def kl_uniform(mu_t, std_t, t_start, t_end, tail=0.05):
    """
    <Deprecated>
    KL Divergence for the 1D near-uniform model
    KL(q||p) where
    q = uniform(t0, t0+dt)
    p = uniform(t_start, t_end) with exponential decays on both sides
    """
    t0 = mu_t - np.sqrt(3)*std_t
    dt = np.sqrt(12)*std_t
    C = 1/((t_end-t_start)*(1+tail))
    lamb = 2/(tail*(t_end-t_start))
    
    t1 = t0+dt
    dt1_til = nn.functional.relu(torch.minimum(t_start, t1) - t0)
    dt2_til = nn.functional.relu(t1 - torch.maximum(t_end, t0))
    
    term1 = -lamb*(dt1_til.pow(2)+dt2_til.pow(2))/(2*dt)
    term2 = lamb*((t_start-t0)*dt1_til+(t1-t_end)*dt2_til)/dt
    
    return torch.mean(term1 + term2 - torch.log(C*dt))

def kl_gaussian(mu1, std1, mu2, std2):
    """
    Compute the KL divergence between two Gaussian distributions with diagonal covariance
    """
    return torch.mean(torch.sum(torch.log(std2/std1)+std1.pow(2)/(2*std2.pow(2))-0.5+(mu1-mu2).pow(2)/(2*std2.pow(2)),1))





##############################################################
# Vanilla VAE
##############################################################
class encoder(nn.Module):
    def __init__(self, Cin, N1=500, N2=250, device=torch.device('cpu'), checkpoint=None):
        """
        Cin: Input Data Dimension
        N1: Width of the First Hidden Layer
        N2: Width of the Second Hidden Layer
        device: (Optional) Model Device. [Default: cpu]
        checkpoint: (Optional) Pretrained Model. [Default: None]
        """
        super(encoder, self).__init__()
        self.fc1 = nn.Linear(Cin, N1).to(device)
        self.bn1 = nn.BatchNorm1d(num_features=N1).to(device)
        self.dpt1 = nn.Dropout(p=0.2).to(device)
        self.fc2 = nn.Linear(N1, N2).to(device)
        self.bn2 = nn.BatchNorm1d(num_features=N2).to(device)
        self.dpt2 = nn.Dropout(p=0.2).to(device)
        
        self.net = nn.Sequential(self.fc1, self.bn1, nn.LeakyReLU(), self.dpt1,
                                 self.fc2, self.bn2, nn.LeakyReLU(), self.dpt2)
        
        self.fc_mu, self.spt1 = nn.Linear(N2,1).to(device), nn.Softplus()
        self.fc_std, self.spt2 = nn.Linear(N2,1).to(device), nn.Softplus()
        
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
        for m in [self.fc_mu, self.fc_std]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, data_in):
        z = self.net(data_in)
        mu_zx, std_zx = self.spt1(self.fc_mu(z)), self.spt2(self.fc_std(z))
        return mu_zx, std_zx

class decoder(nn.Module):
    def __init__(self, 
                 adata, 
                 Tmax, 
                 train_idx, 
                 p=98, 
                 device=torch.device('cpu'), 
                 init_method="steady", 
                 init_key=None):
        """
        adata: AnnData Object
        Tmax: Maximum Cell Time (used in time initialization)
        p: Top Percentile Threshold to Pick Steady-State Cells
        device: (Optional) Model Device [Default: cpu]
        tkey: (Optional) Key in adata.obs that containes some prior time estimation
        checkpoint: (Optional) File path to a pretrained model
        """
        super(decoder,self).__init__()
        U,S = adata.layers['Mu'][train_idx], adata.layers['Ms'][train_idx]
        X = np.concatenate((U,S),1)
        N,G = U.shape
        #Dynamical Model Parameters
        if(init_method == "existing" and init_key is not None):
            self.alpha = nn.Parameter(torch.tensor(np.log(adata.var[f"{init_key}_alpha"].to_numpy()), device=device).float())
            self.beta = nn.Parameter(torch.tensor(np.log(adata.var[f"{init_key}_beta"].to_numpy()), device=device).float())
            self.gamma = nn.Parameter(torch.tensor(np.log(adata.var[f"{init_key}_gamma"].to_numpy()), device=device).float())
            self.scaling = nn.Parameter(torch.tensor(np.log(adata.var[f"{init_key}_scaling"].to_numpy()), device=device).float())
            self.ton = nn.Parameter(torch.tensor(np.log(adata.var[f"{init_key}_ton"].to_numpy()), device=device).float())
            self.toff = nn.Parameter(torch.tensor(np.log(adata.var[f"{init_key}_toff"].to_numpy()), device=device).float())
            self.sigma_u = nn.Parameter(torch.tensor(np.log(adata.var[f"{init_key}_sigma_u"].to_numpy()), device=device).float())
            self.sigma_s = nn.Parameter(torch.tensor(np.log(adata.var[f"{init_key}_sigma_s"].to_numpy()), device=device).float())
        elif(init_method == "random"):
            print("Random Initialization.")
            alpha, beta, gamma, scaling, toff, u0, s0, sigma_u, sigma_s, T, Rscore = initParams(X,p,fit_scaling=True)
            
            self.alpha = nn.Parameter(torch.normal(0.0, 1.0, size=(G,), device=device).float())
            self.beta =  nn.Parameter(torch.normal(0.0, 1.0, size=(G,), device=device).float())
            self.gamma = nn.Parameter(torch.normal(0.0, 1.0, size=(G,), device=device).float())
            self.ton = nn.Parameter(torch.normal(0.0, 1.0, size=(G,), device=device).float())
            self.toff = nn.Parameter(torch.normal(0.0, 1.0, size=(G,), device=device).float()+self.ton.detach())
            self.scaling = nn.Parameter(torch.tensor(np.log(scaling), device=device).float())
            self.sigma_u = nn.Parameter(torch.tensor(np.log(sigma_u), device=device).float())
            self.sigma_s = nn.Parameter(torch.tensor(np.log(sigma_s), device=device).float())
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
            self.ton = nn.Parameter(torch.tensor(np.log(ton+1e-10), device=device).float())
            self.toff = nn.Parameter(torch.tensor(np.log(toff+1e-10), device=device).float())
            self.sigma_u = nn.Parameter(torch.tensor(np.log(sigma_u), device=device).float())
            self.sigma_s = nn.Parameter(torch.tensor(np.log(sigma_s), device=device).float())
        else:
            print("Initialization using the steady-state and dynamical models.")
            alpha, beta, gamma, scaling, toff, u0, s0, sigma_u, sigma_s, T, Rscore = initParams(X,p,fit_scaling=True)
            if(init_key is not None):
                t_init = adata.obs['init_key'].to_numpy()
            else:
                T = T+np.random.rand(T.shape[0],T.shape[1]) * 1e-3
                T_eq = np.zeros(T.shape)
                Nbin = T.shape[0]//50+1
                for i in range(T.shape[1]):
                    T_eq[:, i] = histEqual(T[:, i], Tmax, 0.9, Nbin)
                self.t_init = np.quantile(T_eq,0.5,1)
            toff = getTsGlobal(self.t_init, U/scaling, S, 95)
            alpha, beta, gamma,ton = reinitParams(U/scaling, S, self.t_init, toff)
            
            self.alpha = nn.Parameter(torch.tensor(np.log(alpha), device=device).float())
            self.beta = nn.Parameter(torch.tensor(np.log(beta), device=device).float())
            self.gamma = nn.Parameter(torch.tensor(np.log(gamma), device=device).float())
            self.scaling = nn.Parameter(torch.tensor(np.log(scaling), device=device).float())
            self.ton = nn.Parameter(torch.tensor(np.log(ton+1e-10), device=device).float())
            self.toff = nn.Parameter(torch.tensor(np.log(toff+1e-10), device=device).float())
            self.sigma_u = nn.Parameter(torch.tensor(np.log(sigma_u), device=device).float())
            self.sigma_s = nn.Parameter(torch.tensor(np.log(sigma_s), device=device).float())

        self.scaling.requires_grad = False
        self.sigma_u.requires_grad = False
        self.sigma_s.requires_grad = False
    
    def forward(self, t, neg_slope=0.0):
        Uhat, Shat = ode(t, torch.exp(self.alpha), torch.exp(self.beta), torch.exp(self.gamma), torch.exp(self.ton), torch.exp(self.toff), neg_slope=neg_slope)
        Uhat = Uhat * torch.exp(self.scaling)
        return nn.functional.relu(Uhat), nn.functional.relu(Shat)
    
    def predSU(self, t, gidx=None):
        """
        Unscaled version, used for plotting
        """
        scaling = torch.exp(self.scaling)
        if(gidx is not None):
            Uhat, Shat = ode(t, torch.exp(self.alpha[gidx]), torch.exp(self.beta[gidx]), torch.exp(self.gamma[gidx]), torch.exp(self.ton[gidx]), torch.exp(self.toff[gidx]), neg_slope=0.0)
            return nn.functional.relu(Uhat*scaling[gidx]), nn.functional.relu(Shat)
        Uhat, Shat = ode(t, torch.exp(self.alpha), torch.exp(self.beta), torch.exp(self.gamma), torch.exp(self.ton), torch.exp(self.toff), neg_slope=0.0)
        return nn.functional.relu(Uhat*scaling), nn.functional.relu(Shat)

class VanillaVAE():
    def __init__(self, 
                 adata, 
                 Tmax, 
                 device='cpu', 
                 hidden_size=(500, 250), 
                 init_method="steady",
                 init_key=None,
                 tprior=None, 
                 time_distribution="uniform",
                 checkpoints=None):
        """
        adata: AnnData Object
        Tmax: (float/int) Time Range 
        device: (Optional) Model Device [Default: cpu]
        hidden_size: (Optional) Width of the first and second hidden layer [Default:(500, 250)]
        tprior: (Optional) Key in adata.obs that contains the prior time estimation
        checkpoints: (Optional) File path to the pretrained encoder and decoder models
        """
        #Extract Input Data
        try:
            U,S = adata.layers['Mu'], adata.layers['Ms']
        except KeyError:
            print('Unspliced/Spliced count matrices not found in the layers! Exit the program...')
        
        #Default Training Configuration
        self.config = {
            #Model Parameters
            "tmax":Tmax,
            "hidden_size":hidden_size,
            "init_method": init_method,
            "init_key": init_key,
            "tprior":tprior,

            #Training Parameters
            "n_epochs":500, 
            "learning_rate":1e-4, 
            "learning_rate_ode":1e-4, 
            "lambda":1e-3, 
            "reg_t":1.0, 
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
            "weight_sample":False,
            "sparsify":1
        }
        
        self.setDevice(device)
        self.splitTrainTest(adata.n_obs)
        
        G = adata.n_vars
        #Create an encoder
        try:
            self.encoder = encoder(2*G, hidden_size[0], hidden_size[1], self.device, checkpoint=checkpoints)
        except IndexError:
            print('Please provide two dimensions!')
        #Create a decoder
        self.decoder = decoder(adata, 
                               Tmax, 
                               self.train_idx, 
                               device=self.device, 
                               init_method = init_method,
                               init_key = init_key)
        self.Tmax=torch.tensor(Tmax).to(self.device)
        self.time_distribution = time_distribution
        #Time prior
        self.getPrior(adata, time_distribution, Tmax, tprior)
        
    def getPrior(self, adata, time_distribution, Tmax, tprior):
        if(time_distribution=="gaussian"):
            print("Gaussian Prior.")
            self.kl_time = kl_gaussian
            self.sample = self.reparameterize
            if(tprior is None):
                self.p_t = (torch.ones(2,adata.n_obs,1)*Tmax*0.5).double().to(self.device)
            else:
                print('Using informative time prior.')
                t = adata.obs[tprior].to_numpy()
                n_capture = len(np.unique(t))
                t = t/t.max()*Tmax
                self.p_t = torch.stack( [torch.tensor(t).view(-1,1),torch.ones(adata.n_obs,1)*Tmax/n_capture] ).double().to(self.device)
        else:
            print("Tailed Uniform Prior.")
            self.kl_time = kl_uniform
            self.sample = self.reparameterize_uniform
            if(tprior is None):
                self.p_t = torch.stack([torch.zeros(adata.n_obs,1),torch.ones(adata.n_obs,1)*Tmax]).double().to(self.device)
            else:
                print('Using informative time prior.')
                t = adata.obs[tprior].to_numpy()
                t = t/t.max()*Tmax
                t_cap = np.sort(np.unique(t))
                t_end = np.zeros((len(t)))
                for i in range(len(t_cap)-1):
                    t_end[t==t_cap[i]] = t_cap[i+1]
                t_end[t==t_cap[-1]] = t_cap[-1] + (t.max()-t.min())/len(t_cap)
                
                self.p_t = torch.stack( [torch.tensor(t).unsqueeze(-1),torch.tensor(t_end).unsqueeze(-1)] ).double().to(self.device)
    
    def setDevice(self, device, device_number=None):
        if('cuda' in device):
            if(torch.cuda.is_available()):
                self.device = torch.device(device)
            else:
                print('Warning: GPU not detected. Using CPU as the device.')
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')
    
    def reparameterize(self, mu, std):
        eps = torch.normal(mean=torch.zeros(mu.shape),std=torch.ones(mu.shape)).to(self.device)
        return std*eps+mu
    
    def reparameterize_uniform(self, mu, std):
        eps = torch.rand(mu.shape).to(self.device)
        return np.sqrt(12)*std*eps + (mu - np.sqrt(3)*std)
    
    def forward(self, data_in):
        data_in_scale = torch.cat((data_in[:,:data_in.shape[1]//2]/torch.exp(self.decoder.scaling), data_in[:,data_in.shape[1]//2:]),1)
        mu_t, std_t = self.encoder.forward(data_in_scale)
        t_global = self.reparameterize(mu_t, std_t)
         
        uhat, shat = self.decoder.forward(t_global, neg_slope=self.config["neg_slope"]) #uhat is scaled
        return mu_t, std_t, t_global, uhat, shat
    
    def evalModel(self, data_in):
        data_in_scale = torch.cat((data_in[:,:data_in.shape[1]//2]/torch.exp(self.decoder.scaling), data_in[:,data_in.shape[1]//2:]),1)
        mu_t, std_t = self.encoder.forward(data_in_scale)
        
        uhat, shat = self.decoder.predSU(mu_t) #uhat is scaled
        return mu_t, std_t, uhat, shat
        
    def setMode(self,mode):
        if(mode=='train'):
            self.encoder.train()
            self.decoder.train()
        elif(mode=='eval'):
            self.encoder.eval()
            self.decoder.eval()
        else:
            print("Warning: mode not recognized. Must be 'train' or  'test'! ")
    
    ############################################################
    #Training Objective
    ############################################################
    def VAERisk(self, q_tx, p_t, u, s, uhat, shat, sigma_u, sigma_s, weight=None, b=1.0):
        """
        This is the negative ELBO.
        t0, dt: [B x 1] encoder output, conditional uniform distribution parameters
        Tmax: parameter of the prior uniform distribution
        u , s : [B x G] input data
        uhat, shat: [B x G] prediction by the ODE model
        sigma_u, sigma_s : parameter of the Gaussian distribution
        """
        
        kldt = kl_gaussian(q_tx[0], q_tx[1], p_t[0], p_t[1])
        
        #u and sigma_u has the original scale
        logp = -0.5*((u-uhat)/sigma_u).pow(2)-0.5*((s-shat)/sigma_s).pow(2)-torch.log(sigma_u)-torch.log(sigma_s*2*np.pi)
        
        if( weight is not None):
            logp = logp*weight
        err_rec = torch.mean(torch.sum(logp,1))
        
        return (- err_rec + b*(kldt))
        
    def train_epoch(self, X_loader, test_set, optimizer, optimizer2=None, K=1, reg_t=1.0):
        """
        Training in each epoch
        X_loader: Data loader of the input data
        optimizer, optimizer2(optional): from torch.optim
        K: alternatingly update optimizer and optimizer2
        """
    
        iterX = iter(X_loader)
        B = len(iterX)
        train_loss, test_loss = [], []
        for i in range(B):
            if( self.counter==0 or self.counter % self.config["test_iter"] == 0):
                elbo_test = self.test(test_set, None, self.counter, True)
                test_loss.append(elbo_test)
                self.setMode('train')
                if((self.counter // self.config["test_iter"]) % 10 == 0):
                    print(f"Iteration {self.counter}: Test ELBO = {elbo_test:.3f}")
            
            optimizer.zero_grad()
            if(optimizer2 is not None):
                optimizer2.zero_grad()
            batch = iterX.next()
            xbatch, weight, idx = batch[0].float().to(self.device), batch[2].float().to(self.device), batch[3].to(self.device)
            u = xbatch[:,:xbatch.shape[1]//2]
            s = xbatch[:,xbatch.shape[1]//2:]
            mu_tx, std_tx, t_global, uhat, shat = self.forward(xbatch)
            
            loss = self.VAERisk((mu_tx,std_tx), 
                                self.p_t[:,self.train_idx[idx],:], 
                                u, s, 
                                uhat, shat, 
                                torch.exp(self.decoder.sigma_u), torch.exp(self.decoder.sigma_s), 
                                None,
                                reg_t)
            loss_list.append(loss.detach().cpu().item())
            with torch.autograd.detect_anomaly():
                loss.backward()
            optimizer.step()
            if( optimizer2 is not None and ((i+1) % K == 0 or i==B-1)):
                optimizer2.step()
            train_loss.append(loss.detach().cpu().item())
            self.counter = self.counter + 1
        return train_loss, test_loss
    
    def loadConfig(self, config):
        #We don't have to specify all the hyperparameters. Just pass the ones we want to modify.
        for key in config:
            if(key in self.config):
                self.config[key] = config[key]
            else:
                self.config[key] = config[key]
                print(f"Warning: unknown hyperparameter: {key}")
        if(self.config["train_scaling"]):
            self.decoder.scaling.requires_grad = True
        if(self.config["train_std"]):
            self.decoder.sigma_u.requires_grad = True
            self.decoder.sigma_s.requires_grad = True
    
    def splitTrainTest(self, N):
        rand_perm = np.random.permutation(N)
        n_train = int(N*self.config["train_test_split"])
        self.train_idx = rand_perm[:n_train]
        self.test_idx = rand_perm[n_train:]
        
        return
    
    def train(self, 
              adata, 
              config={}, 
              plot=True, 
              gene_plot=[], 
              figure_path="figures", 
              embed="umap"):
        
        self.loadConfig(config)
        
        print("------------------------- Train a Vanilla VAE -------------------------")
        #Get data loader
        U,S = adata.layers['Mu'], adata.layers['Ms']
        X = np.concatenate((U,S), 1)
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
        param_nn = list(self.encoder.parameters())
        param_ode = [self.decoder.alpha, self.decoder.beta, self.decoder.gamma, self.decoder.ton, self.decoder.toff] 
        if(self.config['train_scaling']):
            param_ode = param_ode+[self.decoder.scaling]
        if(self.config['train_std']):
            param_ode = param_ode+[self.decoder.sigma_u, self.decoder.sigma_s]

        optimizer = torch.optim.Adam(param_nn, lr=self.config["learning_rate"], weight_decay=self.config["lambda"])
        optimizer_ode = torch.optim.Adam(param_ode, lr=self.config["learning_rate_ode"])
        print("*********                      Finished.                      *********")
      
        #Main Training Process
        print("*********                    Start training                   *********")
        print(f"Total Number of Iterations Per Epoch: {len(data_loader)}")
        
        n_epochs, n_save = self.config["n_epochs"], self.config["save_epoch"]
        n_warmup = self.config["n_warmup"]
        loss_train, loss_test = [],[]
        n_drop = 0
        
        start = time.time()
        for epoch in range(n_epochs):
            #Train the encoder
            if(self.config["K_alt"]==0):
                train_loss_epoch, test_loss_epoch = self.train_epoch(data_loader, test_set, optimizer, reg_t=self.config["reg_t"])
                if(epoch>=n_warmup):
                    train_loss_epoch, test_loss_epoch = self.train_epoch(data_loader, test_set, optimizer_ode, reg_t=self.config["reg_t"])
            else:
                if(epoch>=n_warmup):
                    train_loss_epoch, test_loss_epoch = self.train_epoch(data_loader, test_set, optimizer, optimizer_ode, self.config["K_alt"],self.config["reg_t"])
                else:
                    train_loss_epoch, test_loss_epoch = self.train_epoch(data_loader, test_set, optimizer, None, self.config["K_alt"],self.config["reg_t"])
            for loss in loss_train_epoch:
                loss_train.append(loss)
            for loss in loss_test_epoch:
                loss_test.append(loss)
            
            if(plot and (epoch==0 or (epoch+1) % self.config["save_epoch"] == 0)):
                elbo_train = self.test(train_set,
                                       Xembed[self.train_idx],
                                       f"train{epoch+1}", 
                                       gind, 
                                       gene_plot,
                                       True, 
                                       figure_path)
                self.setMode('train')
                if(save):
                    print(f"Epoch {epoch+1}: Train MSE = {elbo_train:.3f}, Test MSE = {elbo_test:.3f}, \t Total Time = {convertTime(time.time()-start)}")
                if(len(loss_test)>1):
                    n_drop = n_drop + 1 if (loss_test[-1]-loss_test[-2]<=adata.n_vars*1e-3) else 0
                    if(n_drop >= self.config["early_stop"] and self.config["early_stop"]>0):
                        print(f"********* Early Stop Triggered at epoch {epoch+1}. *********")
                        break
                
                
        print("*********              Finished. Total Time = {convertTime(time.time()-start)}             *********")
        plotTrainLoss(loss_train, range(1,len(loss_train)+1),True, figure_path,'Basic')
        plotTestLoss(loss_test, [i*self.config["test_iter"] for i in range(len(loss_test))],True, figure_path,'Basic')
        return
    
    def predAll(self, data, output=["uhat", "shat", "t", "z"], gene_idx=None):
        N, G = data.shape[0], data.shape[1]//2
        if("uhat" in output):
            Uhat = None if gene_idx is None else np.zeros((N,len(gene_idx)))
        if("shat" in output):
            Shat = None if gene_idx is None else np.zeros((N,len(gene_idx)))
        if("t" in output):
            t_out = np.zeros((N))
            std_t_out = np.zeros((N))
        elbo = 0
        with torch.no_grad():
            B = min(N//10, 1000)
            Nb = N // B
            for i in range(Nb):
                data_in = torch.tensor(data[i*B:(i+1)*B]).float().to(self.device)
                mu_tx, std_tx, uhat, shat = self.evalModel(data_in)
                loss = self.VAERisk(mu_tx, 
                                    std_tx, 
                                    self.mu_t[self.test_idx][i*B:(i+1)*B], 
                                    self.std_t[self.test_idx][i*B:(i+1)*B],
                                    data_in[:,:G], data_in[:,G:], 
                                    uhat, shat, 
                                    torch.exp(self.decoder.sigma_u), torch.exp(self.decoder.sigma_s), 
                                    None,
                                    1.0)
                elbo = elbo-loss*B
                if("uhat" in output):
                    Uhat[i*B:(i+1)*B] = uhat[:,gene_idx].cpu().numpy()
                if("shat" in output):
                    Shat[i*B:(i+1)*B] = shat[:,gene_idx].cpu().numpy()
                if("t" in output):
                    t_out[i*B:(i+1)*B] = mu_tx.cpu().squeeze().numpy()
                    std_t_out[i*B:(i+1)*B] = std_tx.cpu().squeeze().numpy()
            if(N > B*Nb):
                data_in = torch.tensor(data[B*Nb:]).float().to(self.device)
                mu_tx, std_tx, uhat, shat = self.evalModel(data_in)
                loss = self.VAERisk(mu_tx, 
                                    std_tx, 
                                    self.mu_t[self.test_idx][Nb*B:], 
                                    self.std_t[self.test_idx][Nb*B:],
                                    data_in[:,:G], data_in[:,G:], 
                                    uhat, shat, 
                                    torch.exp(self.decoder.sigma_u), torch.exp(self.decoder.sigma_s), 
                                    None,
                                    1.0)
                elbo = elbo-loss*(N-B*Nb)
                if("uhat" in output):
                    Uhat[Nb*B:] = uhat[:,gene_idx].cpu().numpy()
                if("shat" in output):
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
            output.append(t)
            output.append(std_t)
        return out, elbo.cpu().item()/N
    
    def test(self,
             test_set, 
             Xembed,
             testid=0, 
             gind=None, 
             gene_plot=None,
             plot=False, 
             path='figures', 
             **kwargs):
        """
        data: ncell x ngene tensor
        """
        self.setMode('eval')
        data = test_set.data
        out, elbo = self.predAll(data, gene_idx=gind)
        Uhat, Shat, t, std_t = out[0], out[1], out[2], out[3]
        
        G = data.shape[1]//2
        if(plot):
            ton, toff = np.exp(self.decoder.ton.detach().cpu().numpy()), np.exp(self.decoder.toff.detach().cpu().numpy())
            state = np.ones(toff.shape)*(t.reshape(-1,1)>toff)+np.ones(ton.shape)*2*(t.reshape(-1,1)<ton)
            #Plot Time
            plotTLatent(t, Xembed, f"Training Epoch {testid}", plot, path, f"{testid}-vanilla")
            
            #Plot u/s-t and phase portrait for each gene
            for i in range(len(gind)):
                idx = gind[i]
                #track_idx = plotVAE.pickcell(U[:,i],S[:,i],cell_labels) if cell_labels is not None else None
                track_idx = None
                """
                plotPhase(data[:,idx], data[:,idx+G],  
                          Uhat[:,idx], Shat[:,idx], 
                          gene_plot[i], 
                          track_idx, 
                          state[:,idx], 
                          ['Induction', 'Repression', 'Off'],
                          True, path,
                          f"{gene_plot[i]}-{testid}-vanilla")
                """
                plotSig(t.squeeze(), 
                        data[:,idx], data[:,idx+G],  
                        Uhat[:,i], Shat[:,i], 
                        gene_plot[i], 
                        True, 
                        path, 
                        f"{gene_plot[i]}-{testid}-vanilla",
                        cell_labels=test_set.labels,
                        sparsify=self.config["sparsify"])
        
        return elbo
        
        
    def saveModel(self, file_path, enc_name='encoder_vanilla', dec_name='decoder_vanilla'):
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
        
        self.setMode('eval')
        adata.var[f"{key}_alpha"] = np.exp(self.decoder.alpha.detach().cpu().numpy())
        adata.var[f"{key}_beta"] = np.exp(self.decoder.beta.detach().cpu().numpy())
        adata.var[f"{key}_gamma"] = np.exp(self.decoder.gamma.detach().cpu().numpy())
        adata.var[f"{key}_toff"] = np.exp(self.decoder.toff.detach().cpu().numpy())
        adata.var[f"{key}_ton"] = np.exp(self.decoder.ton.detach().cpu().numpy())
        adata.var[f"{key}_scaling"] = np.exp(self.decoder.scaling.detach().cpu().numpy())
        adata.var[f"{key}_sigma_u"] = np.exp(self.decoder.sigma_u.detach().cpu().numpy())
        adata.var[f"{key}_sigma_s"] = np.exp(self.decoder.sigma_s.detach().cpu().numpy())
        scaling = adata.var[f"{key}_scaling"].to_numpy()
        
        out = self.predAll(np.concatenate((adata.layers['Mu'], adata.layers['Ms']),axis=1), gene_idx=np.array(range(adata.n_vars)))
        Uhat, Shat, t, std_t = out[0], out[1], out[2], out[3]
        
        adata.obs[f"{key}_time"] = t.numpy()
        adata.obs[f"{key}_std_t"] = std_t.numpy()
        adata.layers[f"{key}_uhat"] = Uhat.numpy()
        adata.layers[f"{key}_shat"] = Shat.numpy()
        
        adata.uns[f"{key}_train_idx"] = self.train_idx
        adata.uns[f"{key}_test_idx"] = self.test_idx
        
        rnaVelocityVAE(adata, key)
        
        if(file_name is not None):
            adata.write_h5ad(f"{file_path}/{file_name}")
        
        
    
import numpy as np
import sklearn
import torch
torch.manual_seed(42)
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
import time
from velovae.plotting import plotPhase, plotSig, plotSig_, plotTLatent, plotTrainLoss, plotTestLoss

from .model_util import histEqual, initParams, getTsGlobal, reinitParams, convertTime, getGeneIndex, optimal_transport_duality_gap
from .model_util import predSU, ode, odeNumpy, knnX0, knnX0_random, knn_alt
from .TrainingData import SCData
from .VanillaVAE import VanillaVAE, KLGaussian
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
    def __init__(self, adata, Tmax, Cz, N1=250, N2=500, p=98, device=torch.device('cpu'), tkey=None, param_key=None, checkpoint=None):
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
        
        if(checkpoint is not None):
            self.load_state_dict(torch.load(checkpoint, map_location=device))
        else:
            self.init_weights()
        
        #Dynamical Model Parameters
        U,S = adata.layers['Mu'], adata.layers['Ms']
        X = np.concatenate((U,S),1)
        if(param_key is not None):
            self.alpha = nn.Parameter(torch.tensor(np.log(adata.var[f"{param_key}_alpha"].to_numpy()), device=device).float())
            self.beta = nn.Parameter(torch.tensor(np.log(adata.var[f"{param_key}_beta"].to_numpy()), device=device).float())
            self.gamma = nn.Parameter(torch.tensor(np.log(adata.var[f"{param_key}_gamma"].to_numpy()), device=device).float())
            self.scaling = nn.Parameter(torch.tensor(np.log(adata.var[f"{param_key}_scaling"].to_numpy()), device=device).float())
            self.ton = nn.Parameter(torch.tensor(np.log(adata.var[f"{param_key}_ton"].to_numpy()), device=device).float())
            self.toff = nn.Parameter(torch.tensor(np.log(adata.var[f"{param_key}_toff"].to_numpy()), device=device).float())
            self.sigma_u = nn.Parameter(torch.tensor(np.log(adata.var[f"{param_key}_sigma_u"].to_numpy()), device=device).float())
            self.sigma_s = nn.Parameter(torch.tensor(np.log(adata.var[f"{param_key}_sigma_s"].to_numpy()), device=device).float())
        else:
            alpha, beta, gamma, scaling, toff, u0, s0, sigma_u, sigma_s, T, Rscore = initParams(X,p,fit_scaling=True)
            if(tkey is not None):
                self.t_init = adata.obs[f'{tkey}_time'].to_numpy()
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
            self.ton = nn.Parameter(torch.tensor(np.log(ton+1e-10), device=device).float())
            self.toff = nn.Parameter(torch.tensor(np.log(toff+1e-10), device=device).float())
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
        
        for m in [self.fc_out1]:
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.0)
        
    def forward(self, t, z, u0=None, s0=None, t0=None, neg_slope=0.0):
        rho = F.sigmoid(self.fc_out1(self.net_rho(z)))
        alpha = self.alpha.exp()*rho
        
        if(u0 is None or s0 is None or t0 is None):
            Uhat, Shat = ode(t, alpha, torch.exp(self.beta), torch.exp(self.gamma), self.ton.exp(), self.toff.exp(), neg_slope)
        else:
            Uhat, Shat = predSU(F.leaky_relu(t-t0, neg_slope), u0/self.scaling.exp(), s0, alpha, self.beta.exp(), self.gamma.exp())
        Uhat = Uhat * torch.exp(self.scaling)
        return nn.functional.relu(Uhat), nn.functional.relu(Shat)

class VanillaVAEpp(VanillaVAE):
    def __init__(self, adata, Tmax, Cz, device='cpu', hidden_size=(500, 250, 250, 500), tprior=None, tkey=None, checkpoints=[None, None], param_key=None):
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
            "tprior":tprior,
            "tkey":tkey,
            "n_neighbors":30,
            "dt": (0.03,0.05),
            
            #Training Parameters
            "num_epochs":250, 
            "num_epochs_post":250,
            "learning_rate":1e-4, 
            "learning_rate_ode":1e-4, 
            "lambda":1e-3, 
            "lambda_rho":1e-3,
            "reg_t":1.0, 
            "reg_z":1.0, 
            "neg_slope":0.0,
            "test_epoch":100, 
            "save_epoch":100, 
            "N_warmup":50,
            "batch_size":128, 
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
        self.Cz = Cz
        try:
            self.encoder = encoder(2*G, Cz, hidden_size[0], hidden_size[1], self.device, checkpoint=checkpoints[0])
        except IndexError:
            print('Please provide two dimensions!')
        self.decoder = decoder(adata, 
                               Tmax, 
                               Cz, 
                               hidden_size[2], 
                               hidden_size[3], 
                               device=self.device, 
                               tkey=tkey,
                               param_key=param_key,
                               checkpoint=checkpoints[1])
        self.Tmax=Tmax
        
        if(tprior is None):
            self.mu_t = (torch.ones(U.shape[0],1)*Tmax*0.5).double().to(self.device)
            self.std_t = (torch.ones(U.shape[0],1)*Tmax*0.5).double().to(self.device)
        else:
            print('Using informative time prior.')
            t = adata.obs[tprior].to_numpy()
            t = t/t.max()*Tmax
            self.mu_t = torch.tensor(t).view(-1,1).double().to(self.device)
            self.std_t = (torch.ones(self.mu_t.shape)*Tmax*0.25).double().to(self.device)
        self.mu_z = (torch.zeros(U.shape[0],Cz)).double().to(self.device)
        self.std_z = (torch.ones(U.shape[0],Cz)*0.01).double().to(self.device)
        
        self.use_knn = False
        self.u0 = None
        self.s0 = None
        self.t0 = None
    
    def forward(self, data_in, u0=None, s0=None, t0=None):
        data_in_scale = torch.cat((data_in[:,:data_in.shape[1]//2]/torch.exp(self.decoder.scaling), data_in[:,data_in.shape[1]//2:]),1)
        mu_t, std_t, mu_z, std_z = self.encoder.forward(data_in_scale)
        t = self.reparameterize(mu_t, std_t)
        z = self.reparameterize(mu_z, std_z)
         
        uhat, shat = self.decoder.forward(t, z, u0, s0, t0, neg_slope=self.config["neg_slope"])
        return mu_t, std_t, mu_z, std_z, t, z, uhat, shat
    
    def evalModel(self, data_in, u0=None, s0=None, t0=None):
        data_in_scale = torch.cat((data_in[:,:data_in.shape[1]//2]/torch.exp(self.decoder.scaling), data_in[:,data_in.shape[1]//2:]),1)
        mu_t, std_t, mu_z, std_z = self.encoder.forward(data_in_scale)
         
        uhat, shat = self.decoder.forward(mu_t, mu_z, u0=u0, s0=s0, t0=t0, neg_slope=0.0)
        return mu_t, std_t, mu_z, std_z, uhat, shat
    
    def VAERisk(self, 
                mu_tx, std_tx, mu_t, std_t, 
                mu_zx, std_zx, mu_z, std_z, 
                u, s, uhat, shat, 
                sigma_u, sigma_s, 
                weight=None, b=1.0, c=1.0, 
                lambda_v=None, s0=None):
        """
        This is the negative ELBO.
        t0, dt: [B x 1] encoder output, conditional uniform distribution parameters
        Tmax: parameter of the prior uniform distribution
        u , s : [B x G] input data
        uhat, shat: [B x G] prediction by the ODE model
        sigma_u, sigma_s : parameter of the Gaussian distribution
        """
        
        kldt = KLGaussian(mu_tx, std_tx, mu_t, std_t)
        kldz = KLGaussian(mu_zx, std_zx, mu_z, std_z)
        
        #u and sigma_u has the original scale
        logp = -0.5*((u-uhat)/sigma_u).pow(2)-0.5*((s-shat)/sigma_s).pow(2)-torch.log(sigma_u)-torch.log(sigma_s*2*np.pi)
        
        if( weight is not None):
            logp = logp*weight
        err_rec = torch.mean(torch.sum(logp,1))
        
        #velocity regularization
        if(lambda_v is not None):
            V = (self.decoder.beta.exp())*uhat*(self.decoder.scaling.exp()) - (self.decoder.gamma.exp())*shat
            cos_sim = nn.CosineSimilarity()
            reg_v = lambda_v * torch.sum(cos_sim(V, s-s0))
            return (- err_rec + b*kldt + c*kldz + reg_v)
        return (- err_rec + b*kldt + c*kldz)
    
    def train_epoch(self, X_loader, optimizer, optimizer2=None, K=1, reg_t=1.0, reg_z=1.0, reg_v=None):
        """
        Training in each epoch
        X_loader: Data loader of the input data
        optimizer, optimizer2(optional): from torch.optim
        K: alternatingly update optimizer and optimizer2
        """
    
        iterX = iter(X_loader)
        B = len(iterX)
        loss_list = []
        for i in range(B):
            optimizer.zero_grad()
            if(optimizer2 is not None):
                optimizer2.zero_grad()
            batch = iterX.next()
            xbatch, weight, idx = batch[0].float().to(self.device), batch[2].float().to(self.device), batch[3].to(self.device)
            u = xbatch[:,:xbatch.shape[1]//2]
            s = xbatch[:,xbatch.shape[1]//2:]
            
            u0 = self.u0[self.train_idx][idx] if self.use_knn else None
            s0 = self.s0[self.train_idx][idx] if self.use_knn else None
            t0 = self.t0[self.train_idx][idx] if self.use_knn else None
            
            mu_tx, std_tx, mu_zx, std_zx, t, z, uhat, shat = self.forward(xbatch, u0, s0, t0)
            
            loss = self.VAERisk(mu_tx, std_tx, self.mu_t[idx], self.std_t[idx],
                                mu_zx, std_zx, self.mu_z[idx], self.std_z[idx],
                                u, s, 
                                uhat, shat, 
                                torch.exp(self.decoder.sigma_u), torch.exp(self.decoder.sigma_s), 
                                None, reg_t, reg_z)
            
            loss_list.append(loss.detach().cpu().item())
            with torch.autograd.detect_anomaly():
                loss.backward()
            optimizer.step()
            if( optimizer2 is not None and ((i+1) % K == 0 or i==B-1)):
                optimizer2.step()
        return loss_list
    
    def updateX0(self, U, S):
        """
        Estimate the initial conditions using KNN
        U is unscaled
        """
        Uhat, Shat, t, std_t, z, std_z = self.predAll(torch.tensor(np.concatenate((U,S),1)).float().to(self.device), "both")
        t = t.numpy()
        dt = (self.config["dt"][0]*self.Tmax, self.config["dt"][1]*self.Tmax)
        u0, s0, t0, knn = knnX0(U[self.train_idx], S[self.train_idx], t[self.train_idx], z[self.train_idx].numpy(), t, z.numpy(), self.config["dt"], self.config["n_neighbors"])
        self.u0 = torch.tensor(u0, device=self.device).to(float)
        self.s0 = torch.tensor(s0, device=self.device).to(float)
        self.t0 = torch.tensor(t0.reshape(-1,1), device=self.device).to(float)
        self.knn = knn
        
    def train(self, 
              adata, 
              config={}, 
              plot=True, 
              gene_plot=[], 
              figure_path="figures", 
              embed="umap"):
        
        self.loadConfig(config)
        
        print("---------------------------- Train a VAE++ ----------------------------")
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
        data_loader = torch.utils.data.DataLoader(train_set, batch_size=self.config["batch_size"], shuffle=True)
        print("*********                      Finished.                      *********")
        
        gind, gene_plot = getGeneIndex(adata.var_names, gene_plot)
        os.makedirs(figure_path, exist_ok=True)
        
        #define optimizer
        print("*********                 Creating optimizers                 *********")
        param_nn = list(self.encoder.parameters())+list(self.decoder.net_rho.parameters())+list(self.decoder.fc_out1.parameters())
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
        print("*********                      Stage  1                       *********")
        print(f"Total Number of Iterations Per Epoch: {len(data_loader)}")
        
        n_epochs, n_save = self.config["num_epochs"], self.config["save_epoch"]
        loss_train, loss_test = [],[]
        
        start = time.time()
        for epoch in range(n_epochs):
            #Train the encoder
            if(self.config["K_alt"]==0):
                loss_list = self.train_epoch(data_loader, optimizer, reg_t=self.config["reg_t"], reg_z=self.config["reg_z"])
                if(epoch>=self.config["N_warmup"]):
                    loss_list = self.train_epoch(data_loader, optimizer_ode, reg_t=self.config["reg_t"], reg_z=self.config["reg_z"])
            else:
                if(epoch>=self.config["N_warmup"]):
                    loss_list = self.train_epoch(data_loader, optimizer, optimizer_ode, self.config["K_alt"], self.config["reg_t"], self.config["reg_z"])
                else:
                    loss_list = self.train_epoch(data_loader, optimizer, None, self.config["K_alt"], self.config["reg_t"], self.config["reg_z"])
            if(epoch==0 or (epoch+1) % self.config["test_epoch"] == 0):
                save = (epoch+1) % n_save==0 or epoch==0
                mse_train, t, z = self.test(train_set,
                                            Xembed[self.train_idx],
                                            f"train{epoch+1}", 
                                            False,
                                            gind, 
                                            gene_plot,
                                            save and plot, 
                                            figure_path)
                mse_test = 'N/A'
                if(test_set is not None):
                    mse_test, t, z = self.test(test_set,
                                               Xembed[self.test_idx],
                                               f"test{epoch+1}", 
                                               True,
                                               gind, 
                                               gene_plot,
                                               save and plot, 
                                               figure_path)
                print(f"Epoch {epoch+1}: Train MSE = {mse_train:.3f}, Test MSE = {mse_test:.3f}, \t Total Time = {convertTime(time.time()-start)}")
                loss_train.append(mse_train)
                loss_test.append(mse_test)
                self.setMode('train')
                
        
        print("*********                      Stage  2                       *********")
        self.decoder.init_weights()
        self.updateX0(U, S)
        self.use_knn = True
        #Plot the initial conditions
        if(plot):
            for i in range(len(gind)):
                idx = gind[i]
                t0_plot = self.t0[self.train_idx].squeeze().detach().cpu().numpy()
                u0_plot = self.u0[self.train_idx][:,idx].detach().cpu().numpy()
                s0_plot = self.s0[self.train_idx][:,idx].detach().cpu().numpy()
                plotSig_(t0_plot, 
                         u0_plot, s0_plot, 
                         cell_labels=train_set.labels,
                         title=gene_plot[i], 
                         savefig=True, 
                         path=figure_path, 
                         figname=f"{gene_plot[i]}-x0")
                t0_plot = self.t0[self.test_idx].squeeze().detach().cpu().numpy()
                u0_plot = self.u0[self.test_idx][:,idx].detach().cpu().numpy()
                s0_plot = self.s0[self.test_idx][:,idx].detach().cpu().numpy()
                plotSig_(t0_plot, 
                         u0_plot, s0_plot, 
                         cell_labels=test_set.labels,
                         title=gene_plot[i], 
                         savefig=True, 
                         path=figure_path, 
                         figname=f"{gene_plot[i]}-x0")
        
        param_post = list(self.decoder.net_rho.parameters())+list(self.decoder.fc_out1.parameters())
        optimizer_post = torch.optim.Adam(param_post, lr=self.config["learning_rate"], weight_decay=self.config["lambda_rho"])
        for epoch in range(self.config["num_epochs_post"]):
            if(self.config["K_alt"]==0):
                loss_list = self.train_epoch(data_loader, optimizer_post, reg_t=self.config["reg_t"], reg_z=self.config["reg_z"])
                loss_list = self.train_epoch(data_loader, optimizer_ode, reg_t=self.config["reg_t"], reg_z=self.config["reg_z"])
            else:
                loss_list = self.train_epoch(data_loader, optimizer_post, optimizer_ode, self.config["K_alt"], self.config["reg_t"], self.config["reg_z"])
            
            if(epoch==0 or (epoch+1) % self.config["test_epoch"] == 0):
                save = (epoch+1)%n_save==0 or epoch==0
                mse_train, t, z = self.test(train_set,
                                            Xembed[self.train_idx],
                                            f"train{epoch+1+n_epochs}", 
                                            False,
                                            gind, 
                                            gene_plot,
                                            save and plot, 
                                            figure_path)
                
                mse_test = 'N/A'
                if(test_set is not None):
                    mse_test, t, z = self.test(test_set,
                                               Xembed[self.test_idx],
                                               f"test{epoch+1+n_epochs}", 
                                               True, #test mode
                                               gind, 
                                               gene_plot,
                                               save and plot, 
                                               figure_path)
                print(f"Epoch {epoch+1+n_epochs}: Train MSE = {mse_train}, Test MSE = {mse_test}, \t Total Time = {convertTime(time.time()-start)}")
                loss_train.append(mse_train)
                loss_test.append(mse_test)
                self.setMode('train')
        print("*********                      Finished.                      *********")
        if(plot):
            plotTrainLoss(loss_train,[i for i in range(len(loss_train))],True, figure_path,'rho')
            plotTestLoss(loss_test,[1]+[i*self.config["test_epoch"] for i in range(1,len(loss_test))],True, figure_path,'rho')
        return
    
    def predAll(self, data, mode='test'):
        N, G = data.shape[0], data.shape[1]//2
        Uhat, Shat = torch.empty(N,G), torch.empty(N,G)
        t = torch.empty(N)
        std_t = torch.empty(N)
        z = torch.empty(N, self.Cz)
        std_z = torch.empty(N, self.Cz)
        
        with torch.no_grad():
            B = min(N//10, 1000)
            Nb = N // B
            for i in range(Nb):
                if(mode=="test"):
                    u0 = self.u0[self.test_idx][i*B:(i+1)*B] if self.use_knn else None
                    s0 = self.s0[self.test_idx][i*B:(i+1)*B] if self.use_knn else None
                    t0 = self.t0[self.test_idx][i*B:(i+1)*B] if self.use_knn else None
                    mu_tx, std_tx, mu_zx, std_zx, uhat, shat = self.evalModel(data[i*B:(i+1)*B], u0, s0, t0)
                elif(mode=="train"):
                    u0 = self.u0[self.train_idx][i*B:(i+1)*B] if self.use_knn else None
                    s0 = self.s0[self.train_idx][i*B:(i+1)*B] if self.use_knn else None
                    t0 = self.t0[self.train_idx][i*B:(i+1)*B] if self.use_knn else None
                    mu_tx, std_tx, mu_zx, std_zx, uhat, shat = self.evalModel(data[i*B:(i+1)*B], u0, s0, t0)
                else:
                    u0 = self.u0[i*B:(i+1)*B] if self.use_knn else None
                    s0 = self.s0[i*B:(i+1)*B] if self.use_knn else None
                    t0 = self.t0[i*B:(i+1)*B] if self.use_knn else None
                    mu_tx, std_tx, mu_zx, std_zx, uhat, shat = self.evalModel(data[i*B:(i+1)*B], u0, s0, t0)
                Uhat[i*B:(i+1)*B] = uhat.cpu()
                Shat[i*B:(i+1)*B] = shat.cpu()
                t[i*B:(i+1)*B] = mu_tx.cpu().squeeze()
                std_t[i*B:(i+1)*B] = std_tx.cpu().squeeze()
                z[i*B:(i+1)*B] = mu_zx.cpu()
                std_z[i*B:(i+1)*B] = std_zx.cpu()
            if(N > B*Nb):
                if(mode=="test"):
                    u0 = self.u0[self.test_idx][Nb*B:] if self.use_knn else None
                    s0 = self.s0[self.test_idx][Nb*B:] if self.use_knn else None
                    t0 = self.t0[self.test_idx][Nb*B:] if self.use_knn else None
                    mu_tx, std_tx, mu_zx, std_zx, uhat, shat = self.evalModel(data[Nb*B:], u0, s0, t0)
                elif(mode=="train"):
                    u0 = self.u0[self.train_idx][Nb*B:] if self.use_knn else None
                    s0 = self.s0[self.train_idx][Nb*B:] if self.use_knn else None
                    t0 = self.t0[self.train_idx][Nb*B:] if self.use_knn else None
                    mu_tx, std_tx, mu_zx, std_zx, uhat, shat = self.evalModel(data[Nb*B:], u0, s0, t0)
                else:
                    u0 = self.u0[Nb*B:] if self.use_knn else None
                    s0 = self.s0[Nb*B:] if self.use_knn else None
                    t0 = self.t0[Nb*B:] if self.use_knn else None
                    mu_tx, std_tx, mu_zx, std_zx, uhat, shat = self.evalModel(data[Nb*B:], u0, s0, t0)
                Uhat[Nb*B:] = uhat.cpu()
                Shat[Nb*B:] = shat.cpu()
                t[Nb*B:] = mu_tx.cpu().squeeze()
                std_t[Nb*B:] = std_tx.cpu().squeeze()
                z[Nb*B:] = mu_zx.cpu()
                std_z[Nb*B:] = std_zx.cpu()
        
        return Uhat, Shat, t, std_t, z, std_z
    
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
        data = torch.tensor(dataset.data).float().to(self.device)
        mode = "test" if test_mode else "train"
        Uhat, Shat, t, std_t, z, std_z = self.predAll(data, mode)
        Uhat = Uhat.numpy()
        Shat = Shat.numpy()
        t = t.numpy()
        z = z.numpy()
        
        U,S = data[:,:data.shape[1]//2].detach().cpu().numpy(), data[:,data.shape[1]//2:].detach().cpu().numpy()
        mse = np.mean((Uhat-U)**2+(Shat-S)**2)
        
        if(plot):
            #Plot Time
            plotTLatent(t, Xembed, f"Training Epoch {testid}", plot, path, f"{testid}-rho")
            
            #Plot u/s-t and phase portrait for each gene
            for i in range(len(gind)):
                idx = gind[i]
                
                plotSig(t.squeeze(), 
                        U[:,idx], S[:,idx], 
                        Uhat[:,idx], Shat[:,idx], 
                        gene_plot[i], 
                        True, 
                        path, 
                        f"{gene_plot[i]}-{testid}-rho",
                        cell_labels=dataset.labels,
                        sparsify=self.config['sparsify'])
        
        return mse, t.squeeze(), z
    
    def saveAnnData(self, adata, key, file_path, file_name=None):
        """
        Save the ODE parameters and cell time to the anndata object and write it to disk.
        """
        os.makedirs(file_path, exist_ok=True)
        
        adata.var[f"{key}_alpha"] = np.exp(self.decoder.alpha.detach().cpu().numpy())
        adata.var[f"{key}_beta"] = np.exp(self.decoder.beta.detach().cpu().numpy())
        adata.var[f"{key}_gamma"] = np.exp(self.decoder.gamma.detach().cpu().numpy())
        adata.var[f"{key}_toff"] = np.exp(self.decoder.toff.detach().cpu().numpy())
        adata.var[f"{key}_ton"] = self.decoder.ton.exp().detach().cpu().numpy()
        adata.var[f"{key}_scaling"] = np.exp(self.decoder.scaling.detach().cpu().numpy())
        adata.var[f"{key}_sigma_u"] = np.exp(self.decoder.sigma_u.detach().cpu().numpy())
        adata.var[f"{key}_sigma_s"] = np.exp(self.decoder.sigma_s.detach().cpu().numpy())
        U,S = adata.layers['Mu'], adata.layers['Ms']
        scaling = adata.var[f"{key}_scaling"].to_numpy()
        
        Uhat, Shat, t, std_t, z, std_z = self.predAll(torch.tensor(np.concatenate((adata.layers['Mu'], adata.layers['Ms']),1)).float().to(self.device), "both")
        t = t.numpy()

        adata.obs[f"{key}_time"] = t
        adata.obs[f"{key}_std_t"] = std_t.numpy()
        adata.obsm[f"{key}_z"] = z.numpy()
        adata.obsm[f"{key}_std_z"] = std_z.numpy()
        adata.layers[f"{key}_uhat"] = Uhat.numpy()
        adata.layers[f"{key}_shat"] = Shat.numpy()
        
        rho = F.sigmoid(self.decoder.fc_out1(self.decoder.net_rho(z.to(self.device))))
        
        adata.layers[f"{key}_rho"] = rho.detach().cpu().numpy()
        
        adata.obs[f"{key}_t0"] = self.t0.squeeze().detach().cpu().numpy()
        adata.layers[f"{key}_u0"] = self.u0.detach().cpu().numpy()
        adata.layers[f"{key}_s0"] = self.s0.detach().cpu().numpy()
        adata.obsm[f"{key}_knn"] = self.knn
        
        adata.uns[f"{key}_train_idx"] = self.train_idx
        adata.uns[f"{key}_test_idx"] = self.test_idx
        
        rnaVelocityVAEpp(adata, key, use_raw=False, use_scv_genes=False)
        
        if(file_name is not None):
            adata.write_h5ad(f"{file_path}/{file_name}")
    
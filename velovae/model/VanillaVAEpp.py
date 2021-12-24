import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
import time
from velovae.plotting import plotPhase, plotSig, plotSig_, plotTLatent, plotTrainLoss, plotTestLoss

from .model_util import histEqual, initParams, getTsGlobal, reinitParams, convertTime, getGeneIndex, optimal_transport_duality_gap
from .model_util import predSU, ode, odeNumpy, knnX0
from .TrainingData import SCData
from .VanillaVAE import VanillaVAE, KLGaussian
from .velocity import rnaVelocityRhoVAE

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
    def __init__(self, adata, Tmax, Cz, N1=250, N2=500, p=98, coeff_t=[0.1,0.15], n_neighbor=50, device=torch.device('cpu'), tkey=None, checkpoint=None):
        super(decoder,self).__init__()
        G = adata.n_vars
        self.fc1 = nn.Linear(Cz, N1).to(device)
        self.bn1 = nn.BatchNorm1d(num_features=N1).to(device)
        self.dpt1 = nn.Dropout(p=0.2).to(device)
        self.fc2 = nn.Linear(N1, N2).to(device)
        self.bn2 = nn.BatchNorm1d(num_features=N2).to(device)
        self.dpt2 = nn.Dropout(p=0.2).to(device)
        """
        self.fc3 = nn.Linear(Cz+1, N1).to(device)
        self.bn3 = nn.BatchNorm1d(num_features=N1).to(device)
        self.dpt3 = nn.Dropout(p=0.2).to(device)
        self.fc4 = nn.Linear(N1, N2).to(device)
        self.bn4 = nn.BatchNorm1d(num_features=N2).to(device)
        self.dpt4 = nn.Dropout(p=0.2).to(device)
        
        self.fc5 = nn.Linear(Cz+1, N1).to(device)
        self.bn5 = nn.BatchNorm1d(num_features=N1).to(device)
        self.dpt5 = nn.Dropout(p=0.2).to(device)
        self.fc6 = nn.Linear(N1, N2).to(device)
        self.bn6 = nn.BatchNorm1d(num_features=N2).to(device)
        self.dpt6 = nn.Dropout(p=0.2).to(device)
        """
        #self.L = 10
        self.fc_out1 = nn.Linear(N2, G).to(device)
        #self.fc_out2 = nn.Linear(N2, G).to(device)
        #self.fc_out3 = nn.Linear(N2, 2*G).to(device)
        
        self.net_rho = nn.Sequential(self.fc1, self.bn1, nn.LeakyReLU(), self.dpt1,
                                     self.fc2, self.bn2, nn.LeakyReLU(), self.dpt2)
        """
        self.net_t0 = nn.Sequential(self.fc3, self.bn3, nn.LeakyReLU(), self.dpt3,
                                     self.fc4, self.bn4, nn.LeakyReLU(), self.dpt4)
        self.net_x0 = nn.Sequential(self.fc5, self.bn5, nn.LeakyReLU(), self.dpt5,
                                     self.fc6, self.bn6, nn.LeakyReLU(), self.dpt6)
        """
        
        if(checkpoint is not None):
            self.load_state_dict(torch.load(checkpoint, map_location=device))
        else:
            self.init_weights()
        
        #Dynamical Model Parameters
        U,S = adata.layers['Mu'], adata.layers['Ms']
        X = np.concatenate((U,S),1)
        if(checkpoint is not None):
            self.alpha = nn.Parameter(torch.empty(adata.n_vars, device=device).float())
            self.beta = nn.Parameter(torch.empty(adata.n_vars, device=device).float())
            self.gamma = nn.Parameter(torch.empty(adata.n_vars, device=device).float())
            self.scaling = nn.Parameter(torch.empty(adata.n_vars, device=device).float())
            self.ton = nn.Parameter(torch.empty(adata.n_vars, device=device).float())
            self.toff = nn.Parameter(torch.empty(adata.n_vars, device=device).float())
            self.sigma_u = nn.Parameter(torch.empty(adata.n_vars, device=device).float())
            self.sigma_s = nn.Parameter(torch.empty(adata.n_vars, device=device).float())
            self.load_state_dict(torch.load(checkpoint,map_location=device))
        else:
            alpha, beta, gamma, scaling, toff, u0, s0, sigma_u, sigma_s, T, Rscore = initParams(X,p,fit_scaling=True)
            if(tkey is not None):
                t_init = adata.obs['{tkey}_time'].to_numpy()
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
            self.Rscore = Rscore
            
        self.scaling.requires_grad = False
        self.sigma_u.requires_grad = False
        self.sigma_s.requires_grad = False
        
        self.dt = [Tmax*coeff_t[0], Tmax*coeff_t[1]]
        self.n_neighbor = n_neighbor
        u0, s0, t0, knn = knnX0(U,S,self.t_init,adata.obsm["X_pca"],self.dt,self.n_neighbor)
        self.u0 = torch.tensor(u0, device=device).float()
        self.s0 = torch.tensor(s0, device=device).float()
        self.t0 = torch.tensor(t0.reshape(-1,1), device=device).float()
        
    
    def init_weights(self):
        for m in self.net_rho.modules():
            if(isinstance(m, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
            elif(isinstance(m, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        """
        for m in self.net_x0.modules():
            if(isinstance(m, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
            elif(isinstance(m, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        for m in self.net_t0.modules():
            if(isinstance(m, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
            elif(isinstance(m, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        """
        for m in [self.fc_out1]:
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.0)
        
    def forward(self, t, z, sample_idx, neg_slope=0.0):
        rho = F.sigmoid(self.fc_out1(self.net_rho(z)))
        """
        t0 = (self.fc_out2(self.net_t0(torch.cat((z,t),1))))
        x0 = F.softplus(self.fc_out3(self.net_x0(torch.cat((z,t),1))))
        """
        alpha = self.alpha.exp()*rho
        
        #Uhat, Shat = ode(t, alpha, torch.exp(self.beta), torch.exp(self.gamma), self.ton.exp(), self.toff.exp())
        if(neg_slope>0):
            Uhat, Shat = predSU(F.leaky_relu(t-self.t0[sample_idx], neg_slope), self.u0[sample_idx]/self.scaling.exp(), self.s0[sample_idx], alpha, self.beta.exp(), self.gamma.exp())
        else:
            Uhat, Shat = predSU(F.relu(t-self.t0[sample_idx]), self.u0[sample_idx]/self.scaling.exp(), self.s0[sample_idx], alpha, self.beta.exp(), self.gamma.exp())
        Uhat = Uhat * torch.exp(self.scaling)
        return nn.functional.relu(Uhat), nn.functional.relu(Shat)
    
    def predSU(self,t,z,sample_idx=None,gidx=None):
        """
        Unscaled version, used for plotting
        """
        scaling = torch.exp(self.scaling)
        rho = F.sigmoid(self.fc_out1(self.net_rho(z)))
        """
        t0 = (self.fc_out2(self.net_t0(torch.cat((z,t),1))))
        x0 = F.softplus(self.fc_out3(self.net_x0(torch.cat((z,t),1))))
        """
        alpha = self.alpha.exp()*rho
        
        if(gidx is None):
            if(sample_idx is None):
                Uhat, Shat = predSU(F.relu(t-self.t0), self.u0/self.scaling.exp(), self.s0, alpha, self.beta.exp(), self.gamma.exp())
            else:
                Uhat, Shat = predSU(F.relu(t-self.t0[sample_idx]), self.u0[sample_idx]/self.scaling.exp(), self.s0[sample_idx], alpha, self.beta.exp(), self.gamma.exp())
        else:
            if(sample_idx is None):
                Uhat, Shat = predSU(F.relu(t-self.t0), self.u0[:,gidx]/self.scaling[gidx].exp(), self.s0[:,gidx], alpha[:,gidx], self.beta[gidx].exp(), self.gamma[gidx].exp())
            else:
                Uhat, Shat = predSU(F.relu(t-self.t0[sample_idx]), self.u0[sample_idx,gidx]/self.scaling[gidx].exp(), self.s0[sample_idx,gidx], alpha[gidx], self.beta[gidx].exp(), self.gamma[gidx].exp())
            return nn.functional.relu(Uhat*scaling[gidx]), nn.functional.relu(Shat)
        return nn.functional.relu(Uhat*scaling), nn.functional.relu(Shat)

class VanillaVAEpp:
    def __init__(self, adata, Tmax, Cz, device='cpu', hidden_size=(500, 250, 250, 500), tprior=None, checkpoints=[None,None], coeff_t=[0.1,0.15], n_neighbor=50):
        """
        adata: AnnData Object
        Tmax: (float/int) Time Range 
        """
        try:
            U,S = adata.layers['Mu'], adata.layers['Ms']
        except KeyError:
            print('Unspliced/Spliced count matrices not found in the layers! Exit the program...')
            
        self.setDevice(device)
        
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
                               coeff_t=coeff_t, 
                               n_neighbor=n_neighbor ,
                               device=self.device, 
                               checkpoint=checkpoints[1])
        self.Tmax=torch.tensor(Tmax).to(self.device)
        
        if(tprior is None):
            self.mu_t = (torch.ones(U.shape[0],1)*Tmax*0.5).double().to(self.device)
            self.std_t = (torch.ones(U.shape[0],1)*Tmax*0.25).double().to(self.device)
        else:
            print('Using informative time prior.')
            t = adata.obs[tprior].to_numpy()
            t = t/t.max()*Tmax
            self.mu_t = torch.tensor(t).view(-1,1).double().to(self.device)
            self.std_t = (torch.ones(self.mu_t.shape)*Tmax*0.25).double().to(self.device)
        self.mu_z = (torch.zeros(U.shape[0],Cz)).double().to(self.device)
        self.std_z = (torch.ones(U.shape[0],Cz)*0.01).double().to(self.device)
        
        #Training Configuration
        self.config = {
            "num_epochs":500, "learning_rate":1e-4, "learning_rate_ode":1e-4, "lambda":1e-3, "reg_t":1.0, "reg_z":1.0, "neg_slope":0.0,\
            "test_epoch":100, "save_epoch":100, "batch_size":128, "K_alt":0, "N_knn":100, "knn":30,\
            "train_scaling":False, "train_std":False, "weight_sample":False,\
            "sparsify":2
        }
    
    def forward(self, data_in, sample_idx):
        data_in_scale = torch.cat((data_in[:,:data_in.shape[1]//2]/torch.exp(self.decoder.scaling), data_in[:,data_in.shape[1]//2:]),1)
        mu_t, std_t, mu_z, std_z = self.encoder.forward(data_in_scale)
        t = self.reparameterize(mu_t, std_t)
        z = self.reparameterize(mu_z, std_z)
         
        uhat, shat = self.decoder.forward(t, z, sample_idx, neg_slope=self.config["neg_slope"])
        return mu_t, std_t, mu_z, std_z, t, z, uhat, shat
    
    def evalModel(self, data_in, sample_idx):
        data_in_scale = torch.cat((data_in[:,:data_in.shape[1]//2]/torch.exp(self.decoder.scaling), data_in[:,data_in.shape[1]//2:]),1)
        mu_t, std_t, mu_z, std_z = self.encoder.forward(data_in_scale)
         
        uhat, shat = self.decoder.forward(mu_t, mu_z, sample_idx, neg_slope=0.0) #uhat is scaled
        return mu_t, mu_z, uhat, shat
    
    def VAERisk(self, 
                mu_tx, std_tx, mu_t, std_t, 
                mu_zx, std_zx, mu_z, std_z, 
                u, s, uhat, shat, 
                sigma_u, sigma_s, 
                weight=None, b=1.0, c=1.0):
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
        if(torch.isinf(err_rec)):
            print((u-uhat).max())
            print(sigma_u.min())
            print((s-shat).max())
            print(sigma_s.min())
        return (- err_rec + b*kldt+c*kldz)
    
    def train_epoch(self, X_loader, optimizer, optimizer2=None, K=1, reg_t=1.0, reg_z=1.0):
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
            xbatch, weight, idx = batch[0].float().to(self.device), batch[1].float().to(self.device), batch[2].to(self.device)
            u = xbatch[:,:xbatch.shape[1]//2]
            s = xbatch[:,xbatch.shape[1]//2:]
            mu_tx, std_tx, mu_zx, std_zx, t, z, uhat, shat = self.forward(xbatch, idx)
            
            loss = self.VAERisk(mu_tx, std_tx, self.mu_t[idx], self.std_t[idx],
                                    mu_zx, std_zx, self.mu_z[idx], self.std_z[idx],
                                    u, s, 
                                    uhat, shat, 
                                    torch.exp(self.decoder.sigma_u), torch.exp(self.decoder.sigma_s), 
                                    None,
                                    reg_t, reg_z)
            
            loss_list.append(loss.detach().cpu().item())
            with torch.autograd.detect_anomaly():
                loss.backward()
            optimizer.step()
            if( optimizer2 is not None and ((i+1) % K == 0 or i==B-1)):
                optimizer2.step()
        return loss_list
    
    
    def train(self, 
              adata, 
              config={}, 
              plot=True, 
              gene_plot=[], 
              figure_path="figures", 
              embed="umap"):
        
        self.loadConfig(config)
        
        print("------------------------- Train a rho-VAE -------------------------")
        #Get data loader
        U,S = adata.layers['Mu'], adata.layers['Ms']
        X = np.concatenate((U,S), 1)
        try:
            Xembed = adata.obsm[f"X_{embed}"]
        except KeyError:
            print("Embedding not found! Please run the corresponding preprocessing step!")
        
        cell_labels_raw = adata.obs["clusters"].to_numpy() if "clusters" in adata.obs else np.array(['Unknown' for i in range(adata.n_obs)])
        print("***      Creating  Dataset      ***")
        dataset = SCData(X, Rscore) if self.config['weight_sample'] else SCData(X)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True)
        print("***          Finished.          ***")
        
        gind, gene_plot = getGeneIndex(adata.var_names, gene_plot)
        if(plot):
            os.makedirs(figure_path, exist_ok=True)
            #Plot the initial estimate
            plotTLatent(self.decoder.t_init, Xembed, f"Initial Estimate", plot, figure_path, f"init-rho")
            
            Uhat_init, Shat_init = self.decoder.predSU(torch.tensor(self.decoder.t_init.reshape(-1,1), device=self.device, dtype=self.decoder.alpha.dtype),
                                                       torch.zeros(X.shape[0], self.Cz, device=self.device, dtype=self.decoder.alpha.dtype), gidx=gind)
            for i in range(len(gind)):
                track_idx = None
                
                plotSig(self.decoder.t_init, 
                        U[:,i], S[:,i], 
                        Uhat_init[:,i].detach().cpu().numpy(), Shat_init[:,i].detach().cpu().numpy(), 
                        gene_plot[i], 
                        True, 
                        figure_path, 
                        f"{gene_plot[i]}-init-rho",
                        cell_labels=cell_labels_raw)
        
        #define optimizer
        print("***     Creating optimizers     ***")
        param_nn = list(self.encoder.parameters())+list(self.decoder.net_rho.parameters())+list(self.decoder.fc_out1.parameters())
        param_ode = [self.decoder.alpha, self.decoder.beta, self.decoder.gamma, self.decoder.ton, self.decoder.toff] 
        if(self.config['train_scaling']):
            param_ode = param_ode+[self.decoder.scaling]
        if(self.config['train_std']):
            param_ode = param_ode+[self.decoder.sigma_u, self.decoder.sigma_s]

        optimizer = torch.optim.Adam(param_nn, lr=self.config["learning_rate"], weight_decay=self.config["lambda"])
        optimizer_ode = torch.optim.Adam(param_ode, lr=self.config["learning_rate_ode"])
        print("***          Finished.          ***")
      
        #Main Training Process
        print("***        Start training       ***")
        print(f"Total Number of Iterations Per Epoch: {len(data_loader)}")
        
        n_epochs, n_save = self.config["num_epochs"], self.config["save_epoch"]
        loss_train, loss_test = [],[]
        
        start = time.time()
        for epoch in range(n_epochs):
            #Train the encoder
            if(self.config["K_alt"]==0):
                loss_list = self.train_epoch(data_loader, optimizer, reg_t=self.config["reg_t"], reg_z=self.config["reg_z"])
                loss_train += loss_list
                loss_list = self.train_epoch(data_loader, optimizer_ode, reg_t=self.config["reg_t"], reg_z=self.config["reg_z"])
                loss_train += loss_list
            else:
                loss_list = self.train_epoch(data_loader, optimizer,optimizer_ode, self.config["K_alt"], self.config["reg_t"], self.config["reg_z"])
                loss_train += loss_list
            
            if(epoch==0 or (epoch+1) % self.config["test_epoch"] == 0):
                save = (epoch+1)%n_save==0 or epoch==0
                loss, t,z = self.test(torch.tensor(X).float().to(self.device),
                                       Xembed,
                                       epoch+1, 
                                       gind, 
                                       gene_plot,
                                       save, 
                                       figure_path,
                                       cell_labels_raw)
                print(f"Epoch {epoch+1}: Loss = {loss}, \t Total Time = {convertTime(time.time()-start)}")
                loss_test.append(loss)
                self.setMode('train')
            
            #Update the initial conditions
            if( (epoch+1>=self.config["N_knn"]) and ((epoch+1) % 25 == 0) ):
                u0, s0, t0, knn = knnX0(U, S, t, z, self.decoder.dt, self.decoder.n_neighbor)
                self.decoder.u0 = torch.tensor(u0, device=self.device).to(float)
                self.decoder.s0 = torch.tensor(s0, device=self.device).to(float)
                self.decoder.t0 = torch.tensor(t0.reshape(-1,1), device=self.device).to(float)
                
                
        
        print("***     Finished Training     ***")
        
        plotTrainLoss(loss_train,[i for i in range(len(loss_train))],True, figure_path,'rho')
        plotTestLoss(loss_test,[1]+[i*self.config["test_epoch"] for i in range(1,len(loss_test))],True, figure_path,'rho')
        return
    
    def predAll(self, data):
        N, G = data.shape[0], data.shape[1]//2
        Uhat, Shat = torch.empty(N,G), torch.empty(N,G)
        t = torch.empty(N)
        z = torch.empty(N, self.Cz)
        
        with torch.no_grad():
            B = self.config["batch_size"]
            Nb = N // B
            for i in range(Nb):
                sample_idx = torch.range(i*B,(i+1)*B-1,dtype=torch.long,device=self.device)
                mu_tx, mu_zx, uhat, shat = self.evalModel(data[i*B:(i+1)*B], sample_idx)
                Uhat[i*B:(i+1)*B] = uhat.cpu()
                Shat[i*B:(i+1)*B] = shat.cpu()
                t[i*B:(i+1)*B] = mu_tx.cpu().squeeze()
                z[i*B:(i+1)*B] = mu_zx.cpu()
            if(N > B*Nb):
                sample_idx = torch.range(B*Nb,N-1,dtype=torch.long,device=self.device)
                mu_tx, mu_zx, uhat, shat = self.evalModel(data[B*Nb:],sample_idx)
                Uhat[Nb*B:] = uhat.cpu()
                Shat[Nb*B:] = shat.cpu()
                t[Nb*B:] = mu_tx.cpu().squeeze()
                z[Nb*B:] = mu_zx.cpu()
        
        return Uhat, Shat, t, z
    
    def test(self,
             data, 
             Xembed,
             testid=0, 
             gind=None, 
             gene_plot=None,
             plot=False, 
             path='figures', 
             cell_labels_raw=None,
             **kwargs):
        """
        data: ncell x ngene tensor
        """
        self.setMode('eval')
        Uhat, Shat, t, z = self.predAll(data)
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
                        cell_labels=cell_labels_raw,
                        sparsify=self.config['sparsify'])
                plotSig_(self.decoder.t0.squeeze().detach().cpu().numpy(), 
                         self.decoder.u0[:,gind[i]].detach().cpu().numpy(), self.decoder.s0[:,gind[i]].detach().cpu().numpy(), 
                         cell_labels=cell_labels_raw,
                         title=gene_plot[i], 
                         savefig=True, 
                         path=path, 
                         figname=f"{gene_plot[i]}-x0-{testid}")
        
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
        data_in_scale = np.concatenate((U/scaling,S),1)
        mu_t, std_t, mu_z, std_z = self.encoder.forward(torch.tensor(data_in_scale).float().to(self.device))
        my_t = (mu_t).squeeze().detach().cpu().numpy()
        my_z = mu_z.detach().cpu().numpy()
        
        adata.obs[f"{key}_time"] = my_t
        adata.obs[f"{key}_std_t"] = std_t.squeeze().detach().cpu().numpy()
        adata.obsm[f"{key}_z"] = my_z
        adata.obsm[f"{key}_std_z"] = std_z.detach().cpu().numpy()
        
        
        rho = F.sigmoid(self.decoder.fc_out1(self.decoder.net_rho(mu_z)))
        """
        t0 = (self.decoder.fc_out2(self.decoder.net_t0(torch.cat((mu_z,mu_t),1))))
        x0 = F.softplus(self.decoder.fc_out3(self.decoder.net_x0(torch.cat((mu_z,mu_t),1))))
        
        """
        adata.layers[f"{key}_rho"] = rho.detach().cpu().numpy()
        adata.obs[f"{key}_t0"] = self.decoder.t0.squeeze().detach().cpu().numpy()
        adata.layers[f"{key}_u0"] = self.decoder.u0.detach().cpu().numpy()
        adata.layers[f"{key}_s0"] = self.decoder.s0.detach().cpu().numpy()
        
        rnaVelocityRhoVAE(adata, key, use_raw=False, use_scv_genes=False)
        
        if(file_name is not None):
            adata.write_h5ad(f"{file_path}/{file_name}")
    
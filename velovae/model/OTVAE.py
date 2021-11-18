import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
from velovae.plotting import plotPhase, plotSig, plotTLatent, plotTrainLoss, plotTestLoss

from .model_util import histEqual, initParams, getTsGlobal, reinitParams, convertTime, ode, makeDir, getGeneIndex, optimal_transport_duality_gap
from .TrainingData import SCData
from .VanillaVAE import VanillaVAE, KLGaussian

##############################################################
# OT-VAE
##############################################################
class OTVAE(VanillaVAE):
    def __init__(self, adata, Tmax, device='cpu', hidden_size=(500, 250), tprior=None):
        """
        adata: AnnData Object
        Tmax: (float/int) Time Range 
        """
        super(OTVAE, self).__init__(adata, Tmax, device, hidden_size, tprior)
        self.config['reg_ot'] = 0.05
        self.config['nbin'] = 10
    
    def computeOT(self, 
                  x, 
                  t, 
                  b,
                  lambda1=1, 
                  lambda2=50, 
                  epsilon=0.05, 
                  batch_size=5, 
                  tolerance=1e-8, 
                  tau=10000, 
                  epsilon0=1, 
                  max_iter=10000):
        bins = np.sort(np.unique(b))
        ot = np.zeros((self.config['nbin']))
        for i in range(len(bins)-1):
            x1, x2 = x[b==bins[i]], x[b==bins[i+1]]
            C = sklearn.metrics.pairwise.pairwise_distances(x1,x2,metric='sqeuclidean', n_jobs=-1)
            C = C/np.median(C)
            Pi = optimal_transport_duality_gap(C, np.ones((C.shape[0])), lambda1, lambda2, epsilon, batch_size, tolerance, tau, epsilon0, max_iter)
            ot[bins[i]] = (Pi*C).sum()
        ot[-1] = ot[-2]
        return ot
    
    def train_epoch(self, X_loader, optimizer, X_pca, optimizer2=None, K=1, reg_t=1.0):
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
            mu_tx, std_tx, t, ton, toff, uhat, shat = self.forward(xbatch,True)
            
            loss_vae = self.VAERisk(mu_tx, 
                                    std_tx, 
                                    self.mu_t[idx], 
                                    self.std_t[idx],
                                    u, s, 
                                    uhat, shat, 
                                    torch.exp(self.decoder.sigma_u), torch.exp(self.decoder.sigma_s), 
                                    None,
                                    reg_t)
            #Sample the time bins
            Nbin = self.config['nbin']
            dt = (t.squeeze().max()-t.squeeze().min())/Nbin
            tbin = (torch.range(1, Nbin).to(self.device)-0.5)*dt+t.squeeze().min()
            b_onehot = F.gumbel_softmax(-(t-tbin).pow(2), tau=0.01, hard=True)
            b = torch.argmax(b_onehot, 1)
            ot = self.computeOT(X_pca[idx.cpu().numpy()], 
                                t.squeeze().detach().cpu().numpy(),
                                b.detach().cpu().numpy())
            ot = torch.tensor(ot, device=self.device)
            loss_ot = (b_onehot*ot).sum(1).mean()
            
            loss = loss_vae + self.config['reg_ot']*loss_ot
            
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
        
        print("------------------------- Train a Vanilla VAE -------------------------")
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
            makeDir(figure_path)
            #Plot the initial estimate
            plotTLatent(self.decoder.t_init, Xembed, f"Initial Estimate", plot, figure_path, f"init-vanilla")
            
            Uhat_init, Shat_init = self.decoder.predSU(torch.tensor(self.decoder.t_init.reshape(-1,1), device=self.device, dtype=self.decoder.alpha.dtype))
            for i in range(len(gind)):
                idx = gind[i]
                track_idx = None
                state = (self.decoder.t_init>=np.exp(self.decoder.toff[idx].detach().cpu().item())) + 2*(self.decoder.t_init<np.exp(self.decoder.ton[idx].detach().cpu().item()))
                plotPhase(U[:,idx], S[:,idx], 
                          Uhat_init[:,idx].detach().cpu().numpy(), Shat_init[:,idx].detach().cpu().numpy(), 
                          gene_plot[i], 
                          track_idx, 
                          state, 
                          ['Induction', 'Repression', 'Off'],
                          True, figure_path,
                          f"{gene_plot[i]}-init")
                
                plotSig(self.decoder.t_init, 
                        U[:,idx], S[:,idx], 
                        Uhat_init[:,idx].detach().cpu().numpy(), Shat_init[:,idx].detach().cpu().numpy(), 
                        gene_plot[i], 
                        True, 
                        figure_path, 
                        f"{gene_plot[i]}-init-vanilla",
                        cell_labels=cell_labels_raw)
        
        #define optimizer
        print("***     Creating optimizers     ***")
        param_nn = list(self.encoder.parameters())
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
                loss_list = self.train_epoch(data_loader, optimizer, adata.obsm['X_pca'])
                loss_train += loss_list
                loss_list = self.train_epoch(data_loader, optimizer_ode, adata.obsm['X_pca'])
                loss_train += loss_list
            else:
                loss_list = self.train_epoch(data_loader, optimizer,optimizer_ode, adata.obsm['X_pca'], self.config["K_alt"], self.config["reg_t"])
                loss_train += loss_list
            
            if(epoch==0 or (epoch+1) % self.config["test_epoch"] == 0):
                save = (epoch+1)%n_save==0 or epoch==0
                loss, t_global, ton, toff = self.test(torch.tensor(X).float().to(self.device),
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
        
        print("***     Finished Training     ***")
        
        plotTrainLoss(loss_train,[i for i in range(len(loss_train))],True, figure_path,'vanilla')
        plotTestLoss(loss_test,[1]+[i*self.config["test_epoch"] for i in range(1,len(loss_test))],True, figure_path,'vanilla')
        return
    
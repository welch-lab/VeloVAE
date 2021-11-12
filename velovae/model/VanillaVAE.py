import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
from velovae.plotting import plotPhase, plotSig, plotTLatent, plotTrainLoss, plotTestLoss

from .model_util import histEqual, initParams, getTsGlobal, reinitParams, convertTime, ode, makeDir, getGeneIndex
from .TrainingData import SCData


############################################################
#KL Divergence
############################################################
def KLtime(t0, dt, Tmax, lamb=1):
    """
    KL Divergence for the near-uniform model, current not adopted
    """
    B = t0.shape[0]
    t1 = t0+dt
    kld = torch.tensor(0).to(float).to(t0.device)
    mask1,mask2,mask3 = t1<=Tmax, (t0<=Tmax) & (t1>Tmax), t0>Tmax
    if(torch.any(mask1)):
        kld += torch.sum(torch.log((Tmax+1)/dt[mask1]))
    if(torch.any(mask2)):
        t0_, dt_ = t0[mask2], dt[mask2]
        kld += torch.sum((Tmax-t0_)/dt_*torch.log((Tmax+1)/dt_)+(t0_+dt_-Tmax)/dt_*torch.log((Tmax+1)/(lamb*dt_))+lamb/(2*dt_)*((t0_+dt_).pow(2)-Tmax.pow(2)))
    if(torch.any(mask3)):
        t0_, dt_ = t0[mask3], dt[mask3]
        kld += torch.sum(torch.log((Tmax+1)/(lamb*dt_))+lamb/2*(2*t0_+dt_))
    
    return kld

def KLGaussian(mu1, std1, mu2, std2):
    """
    Compute the KL divergence between two Gaussian distributions
    """
    return torch.mean(torch.sum(torch.log(std2/std1)+std1.pow(2)/(2*std2.pow(2))-0.5+(mu1-mu2).pow(2)/(2*std2.pow(2)),1))



############################################################
#Training Objective
############################################################
def VAERisk(mu_tx, std_tx, mu_t, std_t, u, s, uhat, shat, sigma_u, sigma_s, weight=None, b=1.0):
    """
    This is the negative ELBO.
    t0, dt: [B x 1] encoder output, conditional uniform distribution parameters
    Tmax: parameter of the prior uniform distribution
    u , s : [B x G] input data
    uhat, shat: [B x G] prediction by the ODE model
    sigma_u, sigma_s : parameter of the Gaussian distribution
    """
    
    kldt = KLGaussian(mu_tx, std_tx, mu_t, std_t)
    
    #u and sigma_u has the original scale
    logp = -0.5*((u-uhat)/sigma_u).pow(2)-0.5*((s-shat)/sigma_s).pow(2)-torch.log(sigma_u)-torch.log(sigma_s*2*np.pi)
    
    if( weight is not None):
        logp = logp*weight
    err_rec = torch.mean(torch.sum(logp,1))
    
    return (- err_rec + b*(kldt))

##############################################################
# Vanilla VAE
##############################################################
class encoder(nn.Module):
    def __init__(self, Cin, N1=500, N2=250, device=torch.device('cpu'), **kwargs):
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
        
        self.fc_mu, self.spt1 = nn.Linear(N2,1).to(device), nn.Softplus()
        self.fc_std, self.spt2 = nn.Linear(N2,1).to(device), nn.Softplus()
        
        if('checkpoint' in kwargs):
            self.load_state_dict(torch.load(kwargs['checkpoint'],map_location=device))
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
    def __init__(self, adata, Tmax, p=98, device=torch.device('cpu'), tkey=None):
        super(decoder,self).__init__()
        #Dynamical Model Parameters
        U,S = adata.layers['Mu'], adata.layers['Ms']
        X = np.concatenate((U,S),1)
        alpha, beta, gamma, scaling, toff, u0, s0, sigma_u, sigma_s, T, Rscore = initParams(X,p,fit_scaling=True)
        if(tkey is not None):
            t_init = adata.obs['{tkey}_time'].to_numpy()
        else:
            #t_init = np.quantile(T,0.5,1)
            #t_init = np.clip(t_init, 0, np.quantile(t_init, 0.95))
            #self.t_init = t_init/t_init.max()*Tmax
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
        self.Rscore = Rscore
        
        self.scaling.requires_grad = False
        self.sigma_u.requires_grad = False
        self.sigma_s.requires_grad = False
        
        
        
    def forward(self, t, train_mode=False, scale_back=True, neg_slope=1e-4):
        Uhat, Shat = ode(t, torch.exp(self.alpha), torch.exp(self.beta), torch.exp(self.gamma), torch.exp(self.ton), torch.exp(self.toff), train_mode, neg_slope=neg_slope)
        if(scale_back):
            Uhat = Uhat * torch.exp(self.scaling)
        return nn.functional.relu(Uhat), nn.functional.relu(Shat)
    
    def predSU(self,t):
        """
        Unscaled version, used for plotting
        """
        scaling = torch.exp(self.scaling)
        Uhat, Shat = ode(t, torch.exp(self.alpha), torch.exp(self.beta), torch.exp(self.gamma), torch.exp(self.ton), torch.exp(self.toff), False)
        return nn.functional.relu(Uhat*scaling), nn.functional.relu(Shat)

class VanillaVAE():
    def __init__(self, adata, Tmax, device='cpu', hidden_size=(500, 250), tprior=None, **kwargs):
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
        try:
            self.encoder = encoder(2*G, hidden_size[0], hidden_size[1], self.device)
        except IndexError:
            print('Please provide two dimensions!')
        self.decoder = decoder(adata, Tmax, device=self.device)
        self.Tmax=torch.tensor(Tmax).to(self.device)
        
        if(tprior is None):
            self.mu_t = (torch.ones(U.shape[0],1)*Tmax*0.5).double().to(self.device)
            self.std_t = (torch.ones(U.shape[0],1)*Tmax*0.18).double().to(self.device)
        else:
            t = adata.obs[tprior].to_numpy()
            t = t/t.max()*Tmax
            self.mu_t = torch.tensor(t).view(-1,1).double().to(self.device)
            self.std_t = (torch.ones(self.mu_t.shape)*np.std(t)).double().to(self.device)
        
        #Training Configuration
        self.config = config = {
            "num_epochs":500, "learning_rate":1e-4, "learning_rate_ode":1e-4, "lambda":1e-3, "reg_t":1.0, "neg_slope":1e-4,\
            "test_epoch":100, "save_epoch":100, "batch_size":128, "K_alt":0,\
            "train_scaling":False, "train_std":False, "weight_sample":False
        }
    
    def setDevice(self, device, device_number=None):
        if(device=='gpu'):
            if(torch.cuda.is_available()):
                self.device_number = device_number if (isinstance(device_number,int)) else torch.cuda.current_device()
                self.device = torch.device('cuda:'+str(self.device_number))
            else:
                print('Warning: GPU not detected. Using CPU as the device.')
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')
    
    def reparameterize(self, mu, std):
        eps = torch.normal(mean=torch.zeros(mu.shape),std=torch.ones(mu.shape)).to(self.device)
        return std*eps+mu
    
    def reparameterize_uniform(self, t0, dt):
        eps = torch.rand(t0.shape).to(self.device)
        return eps*dt+t0
    
    def forward(self, data_in, train_mode=False):
        data_in_scale = torch.cat((data_in[:,:data_in.shape[1]//2]/torch.exp(self.decoder.scaling), data_in[:,data_in.shape[1]//2:]),1)
        mu_t, std_t = self.encoder.forward(data_in_scale)
        t_global = self.reparameterize(mu_t, std_t)
         
        uhat, shat = self.decoder.forward(t_global,train_mode,neg_slope=self.config["neg_slope"]) #uhat is scaled
        return mu_t, std_t, t_global, torch.exp(self.decoder.ton), torch.exp(self.decoder.toff), uhat, shat
        
    def setMode(self,mode):
        if(mode=='train'):
            self.encoder.train()
            self.decoder.train()
        elif(mode=='eval'):
            self.encoder.eval()
            self.decoder.eval()
        else:
            print("Warning: mode not recognized. Must be 'train' or  'test'! ")
            
    def train_epoch(self, X_loader, optimizer, optimizer2=None, K=1, reg_t=1.0):
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
            mu_tx, std_tx, t_global, ton, toff, uhat, shat = self.forward(xbatch,True)
            
            loss = VAERisk(mu_tx, 
                           std_tx, 
                           self.mu_t[idx], 
                           self.std_t[idx],
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
        return loss_list
    
    def train(self, 
              adata, 
              config={}, 
              plot=True, 
              gene_plot=[], 
              figure_path="figures", 
              embed="umap"):
        #We don't have to specify all the hyperparameters. Just pass the ones we want to modify.
        for key in config:
            if(key in self.config):
                self.config[key] = config[key]
            else:
                print(f"Warning: unknown hyperparameter: {key}")
        if(self.config["train_scaling"]):
            self.decoder.scaling.requires_grad = True
        if(self.config["train_std"]):
            self.decoder.sigma_u.requires_grad = True
            self.decoder.sigma_s.requires_grad = True
        
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
                loss_list = self.train_epoch(data_loader, optimizer)
                loss_train += loss_list
                loss_list = self.train_epoch(data_loader, optimizer_ode)
                loss_train += loss_list
            else:
                loss_list = self.train_epoch(data_loader, optimizer,optimizer_ode,self.config["K_alt"],self.config["reg_t"])
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
        with torch.no_grad():
            mu_t, std_t, t_global, ton, toff, Uhat, Shat = self.forward(data)
        U,S = data[:,:data.shape[1]//2].detach().cpu().numpy(), data[:,data.shape[1]//2:].detach().cpu().numpy()
        mse = np.mean((Uhat.detach().cpu().numpy()-U)**2+(Shat.detach().cpu().numpy()-S)**2)
        
        t = t_global.detach().cpu().numpy()
        ton, toff = ton.detach().cpu().numpy(),toff.detach().cpu().numpy()
        state = np.ones(ton.shape)*(t>toff)+np.ones(ton.shape)*2*(t<ton)
        if(plot):
            #Plot Time
            plotTLatent(t, Xembed, f"Training Epoch {testid}", plot, path, f"{testid}-vanilla")
            
            #Plot u/s-t and phase portrait for each gene
            Uhat, Shat = Uhat.detach().cpu().numpy(), Shat.detach().cpu().numpy()
            for i in range(len(gind)):
                idx = gind[i]
                #track_idx = plotVAE.pickcell(U[:,i],S[:,i],cell_labels) if cell_labels is not None else None
                track_idx = None
                
                plotPhase(U[:,idx], S[:,idx], 
                          Uhat[:,idx], Shat[:,idx], 
                          gene_plot[i], 
                          track_idx, 
                          state[:,idx], 
                          ['Induction', 'Repression', 'Off'],
                          True, path,
                          f"{gene_plot[i]}-{testid}-vanilla")
                
                plotSig(t.squeeze(), 
                        U[:,idx], S[:,idx], 
                        Uhat[:,idx], Shat[:,idx], 
                        gene_plot[i], 
                        True, 
                        path, 
                        f"{gene_plot[i]}-{testid}-vanilla",
                        cell_labels=cell_labels_raw,
                        sparsify=5)
        
        return mse, t.squeeze(), ton.squeeze(), toff.squeeze()
        
        
    def saveModel(self, file_path, enc_name='encoder_vanilla', dec_name='decoder_vanilla'):
        """
        Save the encoder parameters to a .pt file.
        Save the decoder parameters to the anndata object.
        """
        makeDir(file_path)
        torch.save(self.encoder.state_dict(), f"{file_path}/{enc_name}.pt")
        torch.save(self.decoder.state_dict(), f"{file_path}/{dec_name}.pt")
        
    def saveAnnData(self, adata, key, file_path, file_name=None):
        """
        Save the ODE parameters and cell time to the anndata object and write it to disk.
        """
        makeDir(file_path)
        
        adata.var[f"{key}_alpha"] = np.exp(self.decoder.alpha.detach().cpu().numpy())
        adata.var[f"{key}_beta"] = np.exp(self.decoder.beta.detach().cpu().numpy())
        adata.var[f"{key}_gamma"] = np.exp(self.decoder.gamma.detach().cpu().numpy())
        adata.var[f"{key}_t_"] = np.exp(self.decoder.toff.detach().cpu().numpy())
        adata.var[f"{key}_ton"] = np.exp(self.decoder.ton.detach().cpu().numpy())
        adata.var[f"{key}_scaling"] = np.exp(self.decoder.scaling.detach().cpu().numpy())
        adata.var[f"{key}_sigma_u"] = np.exp(self.decoder.sigma_u.detach().cpu().numpy())
        adata.var[f"{key}_sigma_s"] = np.exp(self.decoder.sigma_s.detach().cpu().numpy())
        U,S = adata.layers['Mu'], adata.layers['Ms']
        D = np.concatenate((U,S),axis=1)
        mu_t, std_t = self.encoder.forward(torch.tensor(D).float().to(self.device))
        my_t = (mu_t).squeeze().detach().cpu().numpy()
        
        adata.obs[f"{key}_time"] = my_t
        adata.obs[f"{key}_std_t"] = std_t.squeeze().detach().cpu().numpy()
        
        if(file_name is not None):
            adata.write_h5ad(f"{file_path}/{file_name}")
        
        
    
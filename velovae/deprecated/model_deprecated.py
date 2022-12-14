import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

############################################################
#Old Branching VAE (with a known transition graph)
############################################################
"""
Reinitialization using the estimated global cell time
"""
def transitionTimeRec(t_trans, ts, t_type, prev_type, graph, G, dt_min=0.01):
    """
    Applied to the branching ODE
    Recursive helper function for transition time initialization
    """
    if(len(graph[prev_type])==0):
        return
    for cur_type in graph[prev_type]:
        t_trans[cur_type] = np.quantile(t_type[cur_type],0.01)
        ts[cur_type] = np.clip(np.quantile(t_type[cur_type],0.02) - t_trans[cur_type], a_min=dt_min, a_max=None)
        transitionTimeRec(t_trans, ts, t_type, cur_type, graph, G)
        t_trans[cur_type] = np.clip(t_trans[cur_type]-t_trans[prev_type], a_min=dt_min, a_max=None)
    return

def transitionTime(t, cell_labels, cell_types, graph, init_type, G, dt_min=1e-4):
    """
    Applied to the branching ODE
    Initialize transition (between consecutive cell types) and switching(within the same cell type) time.
    """
    t_type = {}
    for i, type_ in enumerate(cell_types):
        t_type[type_] = t[cell_labels==type_]
    ts = np.zeros((len(cell_types),G))
    t_trans = np.zeros((len(cell_types)))
    for x in init_type:
        t_trans[x] = t_type[x].min()
        ts[x] = np.clip(np.quantile(t_type[x],0.01) - t_trans[x], a_min=dt_min, a_max=None)
        transitionTimeRec(t_trans, ts, t_type, x, graph, G, dt_min=dt_min)
    return t_trans, ts

def reinitTypeParams(U, S, t, ts, cell_labels, cell_types, init_types):
    """
    Applied under branching ODE
    Use the steady-state model and estimated cell time to initialize
    branching ODE parameters.
    """
    Ntype = len(cell_types)
    G = U.shape[1]
    alpha, beta, gamma = np.ones((Ntype,G)), np.ones((Ntype,G)), np.ones((Ntype,G))
    u0, s0 = np.zeros((len(init_types),G)), np.zeros((len(init_types),G))
    #sigma_u, sigma_s = np.zeros((Ntype,G)), np.zeros((Ntype,G))
    
    for i, type_ in enumerate(cell_types):
        mask_type = cell_labels == type_
        #Determine induction or repression
        
        t_head = np.quantile(t[mask_type],0.02)
        t_mid = (t_head+np.quantile(t[mask_type],0.98))*0.5
    
        u_head = np.mean(U[(t>=t[mask_type].min()) & (t<t_head),:],axis=0)
        u_mid = np.mean(U[(t>=t_mid*0.98) & (t<=t_mid*1.02),:],axis=0)
    
        s_head = np.mean(S[(t>=t[mask_type].min()) & (t<t_head),:],axis=0)
        s_mid = np.mean(S[(t>=t_mid*0.98) & (t<=t_mid*1.02),:],axis=0)
    
        o = u_head + s_head < u_mid + s_mid
        
        #Determine ODE parameters
        U_type, S_type = U[(cell_labels==type_)], S[(cell_labels==type_)]
        
        for g in range(G):
            u_low = np.min(U_type[:,g])
            s_low = np.min(S_type[:,g])
            u_high = np.quantile(U_type[:,g],0.93)
            s_high = np.quantile(S_type[:,g],0.93)
            mask_high =  (U_type[:,g]>u_high) | (S_type[:,g]>s_high)
            mask_low = (U_type[:,g]<u_low) | (S_type[:,g]<s_low)
            mask_q = mask_high | mask_low
            u_q = U_type[mask_q,g]
            s_q = S_type[mask_q,g]
            slope = linregMtx(u_q-U_type[:,g].min(), s_q-S_type[:,g].min())
            if(slope == 1):
                slope = 1 + 0.1*np.random.rand()
            gamma[type_, g] = np.clip(slope, 0.01, None)
        
        alpha[type_] = (np.quantile(U_type,0.93,axis=0) - np.quantile(U_type,0.07,axis=0)) * o \
                        + (np.quantile(U_type,0.93,axis=0) - np.quantile(U_type,0.07,axis=0)) * (1-o) * np.random.rand(G) * 0.001+1e-10
            
            
    for i, type_ in enumerate(init_types):
        mask_type = cell_labels == type_
        t_head = np.quantile(t[mask_type],0.03)
        u0[i] = np.mean(U[(t>=t[mask_type].min()) & (t<=t_head)],axis=0)+1e-10
        s0[i] = np.mean(S[(t>=t[mask_type].min()) & (t<=t_head)],axis=0)+1e-10
        
     
    return alpha,beta,gamma,u0,s0

def recoverTransitionTimeRec(t_trans, ts, prev_type, graph):
    """
    Recursive helper function of recovering transition time.
    """
    if(len(graph[prev_type])==0):
        return
    for cur_type in graph[prev_type]:
        t_trans[cur_type] += t_trans[prev_type]
        ts[cur_type] += t_trans[cur_type]
        recoverTransitionTimeRec(t_trans, ts, cur_type, graph)
    return

def recoverTransitionTime(t_trans, ts, graph, init_type):
    """
    Recovers the transition and switching time from the relative time.
    
    t_trans: [N type] transition time of each cell type
    ts: [N type x G] switch-time of each gene in each cell type
    graph: (dictionary) transition graph
    init_type: (list) initial cell types
    """
    t_trans_orig = deepcopy(t_trans) if isinstance(t_trans,np.ndarray) else t_trans.clone()
    ts_orig = deepcopy(ts) if isinstance(ts, np.ndarray) else ts.clone()
    for x in init_type:
        ts_orig[x] += t_trans_orig[x]
        recoverTransitionTimeRec(t_trans_orig, ts_orig, x, graph)
    return t_trans_orig, ts_orig

def odeInitialRec(U0, S0, t_trans, ts, prev_type, graph, init_type, use_numpy=False, **kwargs):
    """
    Recursive Helper Function to Compute the Initial Conditions
    1. U0, S0: stores the output, passed by reference
    2. t_trans: [N type] Transition time (starting time)
    3. ts: [N_type x 2 x G] Switching time (Notice that there are 2 phases and the time is gene-specific)
    4. prev_type: previous cell type (parent)
    5. graph: dictionary representation of the transition graph
    6. init_type: starting cell types
    """
    alpha,beta,gamma = kwargs['alpha'][prev_type], kwargs['beta'][prev_type], kwargs['gamma'][prev_type]
    u0, s0 = U0[prev_type], S0[prev_type]

    for cur_type in graph[prev_type]:
        if(use_numpy):
            u0_cur, s0_cur = predSUNumpy(np.clip(t_trans[cur_type]-ts[prev_type],0,None), u0, s0, alpha, beta, gamma)
        else:
            u0_cur, s0_cur = predSU(nn.functional.relu(t_trans[cur_type]-ts[prev_type]), u0, s0, alpha, beta, gamma)
        U0[cur_type] = u0_cur
        S0[cur_type] = s0_cur

        odeInitialRec(U0, 
                      S0, 
                      t_trans, 
                      ts, 
                      cur_type, 
                      graph, 
                      init_type,
                      use_numpy=use_numpy,
                      **kwargs)
                      
    return

def odeInitial(t_trans, 
               ts, 
               graph, 
               init_type, 
               alpha, 
               beta, 
               gamma,
               u0,
               s0,
               use_numpy=False):
    """
    Traverse the transition graph to compute the initial conditions of all cell types.
    """
    U0, S0 = {}, {}
    for i,x in enumerate(init_type):
        U0[x] = u0[i]
        S0[x] = s0[i]
        odeInitialRec(U0,
                      S0,
                      t_trans,
                      ts,
                      x,
                      graph,
                      init_type,
                      alpha=alpha,
                      beta=beta,
                      gamma=gamma,
                      use_numpy=use_numpy)
    return U0, S0

def odeBranch(t, graph, init_type, use_numpy=False, train_mode=False, neg_slope=1e-4, **kwargs):
    """
    Top-level function to compute branching ODE solution
    
    t: [B x 1] cell time
    graph: transition graph
    init_type: initial cell types
    use_numpy: (bool) whether to use the numpy version
    train_mode: (bool) affects whether to use the leaky ReLU to train the transition and switch-on time.
    """
    alpha,beta,gamma = kwargs['alpha'], kwargs['beta'], kwargs['gamma'] #[N type x G]
    t_trans, ts = kwargs['t_trans'], kwargs['ts']
    u0,s0 = kwargs['u0'], kwargs['s0']
    if(use_numpy):
        U0, S0 = np.zeros((alpha.shape[0], alpha.shape[-1])), np.zeros((alpha.shape[0], alpha.shape[-1]))
    else:
        U0, S0 = torch.empty(alpha.shape[0], alpha.shape[-1]).to(t.device), torch.empty(alpha.shape[0], alpha.shape[-1]).to(t.device)
    #Compute initial states of all cell types and genes
    U0_dic, S0_dic = odeInitial(t_trans,
                       ts,
                       graph,
                       init_type,
                       alpha,
                       beta,
                       gamma,
                       u0,
                       s0,
                       use_numpy=use_numpy)
    for i in U0_dic:
        U0[i] = U0_dic[i]
        S0[i] = S0_dic[i]
    
    #Recover the transition time
    t_trans_orig, ts_orig = recoverTransitionTime(t_trans, ts, graph, init_type) 

    #Compute ode solution
    if(use_numpy):
        tau = np.clip(t.reshape(-1,1,1)-ts_orig,0,None)
        uhat, shat = predSUNumpy(tau, U0, S0, alpha, beta, gamma)
    else:
        tau = nn.functional.leaky_relu(t.unsqueeze(-1)-ts_orig, negative_slope=neg_slope) if train_mode else nn.functional.relu(t.unsqueeze(-1)-ts_orig)
        uhat, shat = predSU(tau, U0, S0, alpha, beta, gamma)
    
    return uhat, shat


def odeBranchNumpy(t, graph, init_type, **kwargs):
    """
    (Numpy Version)
    Top-level function to compute branching ODE solution. Wraps around odeBranch
    """
    Ntype = len(graph.keys())
    cell_labels = kwargs['cell_labels']
    scaling = kwargs['scaling']
    py = np.zeros((t.shape[0], Ntype, 1))
    for i in range(Ntype):
        py[cell_labels==i,i,:] = 1
    uhat, shat = odeBranch(t, graph, init_type, True, **kwargs)
    return np.sum(py*(uhat*scaling), 1), np.sum(py*shat, 1)

def initAllPairsNumpy(alpha,
                      beta,
                      gamma,
                      t_trans,
                      ts,
                      u0,
                      s0,
                      k=10):
    """
    Notice: t_trans and ts are all the absolute values, not relative values
    """
    Ntype = alpha.shape[0]
    G = alpha.shape[1]
    
    #Compute different initial conditions
    tau0 = F.softplus(torch.tensor(t_trans.reshape(-1,1,1) - ts), beta=k).numpy()
    U0_hat, S0_hat = predSUNumpy(tau0, u0, s0, alpha, beta, gamma) #initial condition of the current type considering all possible parent types
    
    return np.clip(U0_hat, 0, None), np.clip(S0_hat, 0, None)

def ode_br_weighted_numpy(t, y, w, get_init=False, k=10, **kwargs):
    alpha,beta,gamma = kwargs['alpha'], kwargs['beta'], kwargs['gamma'] #[N type x G]
    t_trans, ts = kwargs['t_trans'], kwargs['ts']
    u0,s0 = kwargs['u0'], kwargs['s0'] #[N type x G]
    scaling=kwargs.pop("scaling", None)
    
    Ntype, G = alpha.shape
    N = len(y)
    
    U0_hat, S0_hat = initAllPairsNumpy(alpha,
                                       beta,
                                       gamma,
                                       t_trans,
                                       ts,
                                       u0,
                                       s0,
                                       k) #(type, parent type, gene)
    Uhat, Shat = np.zeros((N,G)), np.zeros((N,G))
    for i in range(Ntype):
        parent = np.argmax(w[i])
        tau = F.softplus( torch.tensor(t[y==i] - ts[i]), beta=k).numpy() #(cell, type, gene)
        Uhat_type, Shat_type = predSUNumpy(tau,
                                           U0_hat[i, parent],
                                           S0_hat[i, parent], 
                                           alpha[i],
                                           beta[i],
                                           gamma[i])
        
        Uhat[y==i] = Uhat_type
        Shat[y==i] = Shat_type
    if(scaling is not None):
        Uhat = Uhat * scaling
    if(get_init):
        return Uhat, Shat, U0_hat, S0_hat
    return Uhat, Shat




def computeMixWeight(mu_t, sigma_t,
                     cell_labels,
                     alpha,
                     beta,
                     gamma,
                     t_trans,
                     t_end,
                     ts,
                     u0,
                     s0,
                     sigma_u,
                     sigma_s,
                     eps_t,
                     k=1):
    
    U0_hat, S0_hat = initAllPairs(alpha,
                                  beta,
                                  gamma,
                                  t_trans,
                                  ts,
                                  u0,
                                  s0,
                                  False)
    
    Ntype = alpha.shape[0]
    var = torch.mean(sigma_u.pow(2)+sigma_s.pow(2))
    
    tscore = torch.empty(Ntype, Ntype).to(alpha.device)
    
    mu_t_type = [mu_t[cell_labels==i] for i in range(Ntype)]
    std_t_type = [sigma_t[cell_labels==i] for i in range(Ntype)]
    for i in range(Ntype):#child
        for j in range(Ntype):#parent
            mask1, mask2 = (mu_t_type[j]<t_trans[i]-3*eps_t).float(), (mu_t_type[j]>=t_trans[i]+3*eps_t).float()
            tscore[i, j] = torch.mean( ((mu_t_type[j]-t_trans[i]).pow(2) + (std_t_type[j] - eps_t).pow(2))*(mask1+mask2*k) )
    
    xscore = torch.mean(((U0_hat-u0.unsqueeze(1))).pow(2)+((S0_hat-s0.unsqueeze(1))).pow(2),-1) + torch.eye(alpha.shape[0]).to(alpha.device)*var*0.1
    
    #tmask = t_trans.view(-1,1)<t_trans
    #xscore[tmask] = var*1e3
    mu_tscore, mu_xscore = tscore.mean(), xscore.mean()
    logit_w = - tscore/mu_tscore - xscore/mu_xscore
    
    return logit_w, tscore, xscore

def initAllPairs(alpha,
                 beta,
                 gamma,
                 t_trans,
                 ts,
                 u0,
                 s0,
                 neg_slope=0.0):
    """
    Notice: t_trans and ts are all the absolute values, not relative values
    """
    Ntype = alpha.shape[0]
    G = alpha.shape[1]
    
    #Compute different initial conditions
    tau0 = F.leaky_relu(t_trans.view(-1,1,1) - ts, neg_slope)
    U0_hat, S0_hat = predSU(tau0, u0, s0, alpha, beta, gamma) #initial condition of the current type considering all possible parent types
    
    return F.relu(U0_hat), F.relu(S0_hat)



def ode_br_weighted(t, y_onehot, neg_slope=0, **kwargs):
    """
    Compute the ODE solution given every possible parent cell type
    """
    alpha,beta,gamma = kwargs['alpha'], kwargs['beta'], kwargs['gamma'] #[N type x G]
    t_trans, ts = kwargs['t_trans'], kwargs['ts']
    u0,s0 = kwargs['u0'], kwargs['s0'] #[N type x G]
    sigma_u = kwargs['sigma_u']
    sigma_s = kwargs['sigma_s']
    scaling=kwargs["scaling"]
    
    Ntype, G = alpha.shape
    N = y_onehot.shape[0]
    
    U0_hat, S0_hat = initAllPairs(alpha,
                                  beta,
                                  gamma,
                                  t_trans,
                                  ts,
                                  u0,
                                  s0,
                                  neg_slope) #(type, parent type, gene)
    
    tau = F.leaky_relu( t.view(N,1,1,1) - ts.view(Ntype,1,G), neg_slope) #(cell, type, parent type, gene)
    Uhat, Shat = predSU(tau,
                        U0_hat,
                        S0_hat,
                        alpha.view(Ntype, 1, G),
                        beta.view(Ntype, 1, G),
                        gamma.view(Ntype, 1, G))
    
    return ((Uhat*y_onehot.view(N,Ntype,1,1)).sum(1))*scaling, (Shat*y_onehot.view(N,Ntype,1,1)).sum(1)

def getPredictionBranching(adata, key, graph, init_types):
    """
    Given a key, the function finds the paraemeters from anndata and predicts U/S.
    """
    U, S = adata.layers["Mu"], adata.layers["Ms"]
    Ntype = len(graph.keys())
    #VAE
    alpha = adata.varm[f"{key}_alpha"].T
    beta = adata.varm[f"{key}_beta"].T
    gamma = adata.varm[f"{key}_gamma"].T
    t_trans = adata.uns[f"{key}_t_trans"]
    ts = adata.varm[f"{key}_t_"].T
    t = adata.obs[f"{key}_time"].to_numpy()
    scaling = adata.var[f"{key}_scaling"].to_numpy()
    u0 = adata.varm[f"{key}_u0"].T
    s0 = adata.varm[f"{key}_s0"].T
    labels = adata.obs[f"{key}_label"].to_numpy()
    sigma_u, sigma_s = adata.var[f"{key}_sigma_u"].to_numpy(), adata.var[f"{key}_sigma_s"].to_numpy()

    Uhat, Shat = odeBranchNumpy(t.reshape(len(t),1),
                               graph,
                               init_types,
                               alpha=alpha,
                               beta=beta,
                               gamma=gamma,
                               t_trans=t_trans,
                               ts=ts,
                               scaling=scaling,
                               u0=u0,
                               s0=s0,
                               cell_labels=labels,
                               train_mode=False)
    
    

    logp = -(U-Uhat)**2/(2*sigma_u**2)-(S-Shat)**2/(2*sigma_s**2) - np.log(sigma_u) - np.log(sigma_s) - np.log(2*np.pi)
    logp = np.nanmean(logp.sum(1))
    
    
    return Uhat, Shat, logp

def getPredictionBranchingDemo(adata, key, graph, init_types, genes=None, N=100):
    Ntype = len(graph.keys())
    #VAE
    alpha = adata.varm[f"{key}_alpha"].T
    beta = adata.varm[f"{key}_beta"].T
    gamma = adata.varm[f"{key}_gamma"].T
    t_trans = adata.uns[f"{key}_t_trans"]
    ts = adata.varm[f"{key}_t_"].T
    t = adata.obs[f"{key}_time"].to_numpy()
    scaling = adata.var[f"{key}_scaling"].to_numpy()
    u0 = adata.varm[f"{key}_u0"].T
    s0 = adata.varm[f"{key}_s0"].T
    
    t = adata.obs[f"{key}_time"].to_numpy()
    y = adata.obs[f"{key}_label"].to_numpy()
    
    t_demo = np.zeros((Ntype*N))
    y_demo = np.zeros((Ntype*N))
    t_trans_orig, ts_orig = recoverTransitionTime(t_trans, ts, graph, init_types)

    for i in range(Ntype):
        tmin = t_trans_orig[i]
        if(len(self.transgraph.graph[i])>0):
            tmax = np.max([t_trans_orig[j] for j in graph[i]])
        else:
            tmax = t[y==i].max()
        t_demo[i*N:(i+1)*N] = np.linspace(tmin, tmax, N)
        y_demo[i*N:(i+1)*N] = i
    if(genes is None):
        Uhat_demo, Shat_demo = odeBranchNumpy(t_demo.reshape(len(t),1),
                                               graph,
                                               init_types,
                                               alpha=alpha,
                                               beta=beta,
                                               gamma=gamma,
                                               t_trans=t_trans,
                                               ts=ts,
                                               scaling=scaling,
                                               u0=u0,
                                               s0=s0,
                                               cell_labels=y_demo,
                                               train_mode=False)
    else:
        gene_indices = np.array([np.where(adata.var_names==x)[0][0] for x in genes])
        Uhat_demo, Shat_demo = odeBranchNumpy(t_demo.reshape(len(t),1),
                                               graph,
                                               init_types,
                                               alpha=alpha[:, gene_indices],
                                               beta=beta[:, gene_indices],
                                               gamma=gamma[:, gene_indices],
                                               t_trans=t_trans,
                                               ts=ts[:, gene_indices],
                                               scaling=scaling[:, gene_indices],
                                               u0=u0[:, gene_indices],
                                               s0=s0[:, gene_indices],
                                               cell_labels=y_demo,
                                               train_mode=False)
    return t_demo, y_demo, Uhat_demo, Shat_demo

def rnaVelocityBranch(adata, key, use_raw=False, use_scv_genes=False):
    """
    Compute the velocity based on
    ds/dt = beta_y * u - gamma_y * s, where y is the cell type
    """
    alpha = adata.varm[f"{key}_alpha"].T
    beta = adata.varm[f"{key}_beta"].T
    gamma = adata.varm[f"{key}_gamma"].T
    t_trans = adata.uns[f"{key}_t_trans"]
    ts = adata.varm[f"{key}_ts"].T
    t = adata.obs[f"{key}_time"].to_numpy()
    scaling = adata.var[f"{key}_scaling"].to_numpy()
    u0 = adata.varm[f"{key}_u0"].T
    s0 = adata.varm[f"{key}_s0"].T
    w = adata.uns[f"{key}_w"]
    cell_labels = adata.obs[f"{key}_label"].to_numpy()
    N,G = adata.n_obs, adata.n_vars
    Ntype = alpha.shape[0]
    
    if(use_raw):
        U, S = adata.layers['Mu'], adata.layers['Ms']
    else:
        y_onehot = np.zeros((N, Ntype))
        w_ = np.zeros((Ntype, Ntype))
        for i in range(Ntype):
            y_onehot[cell_labels==i, i] = 1
            w_[i, np.argmax(w[i])] = 1
        w_onehot = np.sum(y_onehot.reshape((N, Ntype, 1))*w_, 1)
        U, S =  odeWeightedNumpy(t.reshape(len(t),1),
                                 y_onehot,
                                 w_onehot,
                                 alpha=alpha,
                                 beta=beta,
                                 gamma=gamma,
                                 t_trans=t_trans,
                                 ts=ts,
                                 scaling=scaling,
                                 u0=u0,
                                 s0=s0)
        U = U/scaling
        """
        u0_hat, s0_hat = initAllPairsNumpy(alpha,
                                           beta,
                                           gamma,
                                           t_trans,
                                           ts,
                                           u0,
                                           s0)
        u0_hat = np.sum( u0_hat * w_.reshape((Ntype,Ntype,1)), 1 )
        s0_hat = np.sum( s0_hat * w_.reshape((Ntype,Ntype,1)), 1 )
        """
    V = np.zeros((N,G))
    for i in range(Ntype):
        cell_mask = cell_labels==i
        tmask = (t[cell_mask].reshape(-1,1) >= ts[i])
        V[cell_mask] = (beta[i]*U[cell_mask] - gamma[i]*S[cell_mask] )*tmask
    
    adata.layers[f"{key}_velocity"] = V
    if(use_scv_genes):
        gene_mask = np.isnan(adata.var['fit_scaling'].to_numpy())
        V[:, gene_mask] = np.nan
    
    return V, U, S






############################################################
#Discrete Vanilla VAE
############################################################

class decoder_basic(nn.Module):
    def __init__(self, 
                 adata, 
                 tmax,
                 train_idx,
                 dim_z, 
                 dim_cond=0,
                 N1=250, 
                 N2=500, 
                 p=98, 
                 use_raw=True,
                 init_ton_zero=False,
                 scale_cell=False,
                 separate_us_scale=True,
                 add_noise=False,
                 device=torch.device('cpu'), 
                 init_method="steady", 
                 init_key=None, 
                 init_type=None,
                 checkpoint=None):
        super(decoder,self).__init__()
        
        if(checkpoint is not None):
            self.alpha = nn.Parameter(torch.empty(G, device=device).float())
            self.beta = nn.Parameter(torch.empty(G, device=device).float())
            self.gamma = nn.Parameter(torch.empty(G, device=device).float())
            self.scaling_u = nn.Parameter(torch.empty(G, device=device).float())
            self.scaling_s = nn.Parameter(torch.empty(G, device=device).float())
            self.ton = nn.Parameter(torch.empty(G, device=device).float())
            self.toff = nn.Parameter(torch.empty(G, device=device).float())
            self.u0 = nn.Parameter(torch.empty(G, device=device).float())
            self.s0 = nn.Parameter(torch.empty(G, device=device).float())
            self.load_state_dict(torch.load(checkpoint, map_location=device))
        else:
            self.init_weights()
            
            #Library Size
            if(use_raw):
                U, S = adata.layers['unspliced'].A.astype(float), adata.layers['spliced'].A.astype(float)
            else:
                U, S = adata.layers["Cu"], adata.layers["Cs"]
            
            #Dispersion
            mean_u, mean_s, dispersion_u, dispersion_s = get_dispersion(U[train_idx], S[train_idx])
            adata.var["mean_u"] = mean_u
            adata.var["mean_s"] = mean_s
            adata.var["dispersion_u"] = dispersion_u
            adata.var["dispersion_s"] = dispersion_s
            
            if(scale_cell):
                U, S, lu, ls = scale_by_cell(U, S, train_idx, separate_us_scale, 50)
                adata.obs["library_scale_u"] = lu
                adata.obs["library_scale_s"] = ls
            else:
                lu, ls = get_cell_scale(U, S, train_idx, separate_us_scale, 50)
                adata.obs["library_scale_u"] = lu
                adata.obs["library_scale_s"] = ls
            
            U = U[train_idx]
            S = S[train_idx]
            
            X = np.concatenate((U,S),1)
            if(add_noise):
                noise = np.exp(np.random.normal(size=(len(train_idx), 2*G))*1e-3)
                X = X + noise
            

            if(init_method == "random"):
                print("Random Initialization.")
                #alpha, beta, gamma, scaling, toff, u0, s0, sigma_u, sigma_s, T, Rscore = init_params(X,p,fit_scaling=True)
                scaling, scaling_s = get_gene_scale(U,S,None)
                self.alpha = nn.Parameter(torch.normal(0.0, 0.01, size=(G,), device=device).float())
                self.beta =  nn.Parameter(torch.normal(0.0, 0.01, size=(G,), device=device).float())
                self.gamma = nn.Parameter(torch.normal(0.0, 0.01, size=(G,), device=device).float())
                self.ton = torch.nn.Parameter(torch.ones(G, device=device).float()*(-10))
                self.toff = nn.Parameter(torch.normal(0.0, 0.01, size=(G,), device=device).float()+self.ton.detach())
            elif(init_method == "tprior"):
                print("Initialization using prior time.")
                alpha, beta, gamma, scaling, toff, u0, s0, T = init_params_raw(X,p,fit_scaling=True)
                t_prior = adata.obs[init_key].to_numpy()
                t_prior = t_prior[train_idx]
                std_t = (np.std(t_prior)+1e-3)*0.2
                self.t_init = np.random.uniform(t_prior-std_t, t_prior+std_t)
                self.t_init -= self.t_init.min()
                self.t_init = self.t_init
                self.t_init = self.t_init/self.t_init.max()*tmax
                toff = get_ts_global(self.t_init, U, S, 95)
                alpha, beta, gamma, ton = reinit_params(U, S, self.t_init, toff)
                
                self.alpha = nn.Parameter(torch.tensor(np.log(alpha), device=device).float())
                self.beta = nn.Parameter(torch.tensor(np.log(beta), device=device).float())
                self.gamma = nn.Parameter(torch.tensor(np.log(gamma), device=device).float())
                self.ton = nn.Parameter((torch.ones(adata.n_vars, device=device)*(-10)).float()) if init_ton_zero else nn.Parameter(torch.tensor(np.log(ton+1e-10), device=device).float())
                self.toff = nn.Parameter(torch.tensor(np.log(toff+1e-10), device=device).float())            
            else:
                print("Initialization using the steady-state and dynamical models.")
                alpha, beta, gamma, scaling, toff, u0, s0, T = init_params_raw(X,p,fit_scaling=True)
                
                if(init_key is not None):
                    self.t_init = adata.obs[init_key].to_numpy()[train_idx]
                else:
                    T = T+np.random.rand(T.shape[0],T.shape[1]) * 1e-3
                    T_eq = np.zeros(T.shape)
                    Nbin = T.shape[0]//50+1
                    for i in range(T.shape[1]):
                        T_eq[:, i] = hist_equal(T[:, i], tmax, 0.9, Nbin)
                    self.t_init = np.quantile(T_eq,0.5,1)
                toff = get_ts_global(self.t_init, U, S, 95)
                alpha, beta, gamma, ton = reinit_params(U, S, self.t_init, toff)
                
                self.alpha = nn.Parameter(torch.tensor(np.log(alpha), device=device).float())
                self.beta = nn.Parameter(torch.tensor(np.log(beta), device=device).float())
                self.gamma = nn.Parameter(torch.tensor(np.log(gamma), device=device).float())
                self.ton = nn.Parameter((torch.ones(adata.n_vars, device=device)*(-10)).float()) if init_ton_zero else nn.Parameter(torch.tensor(np.log(ton+1e-10), device=device).float())
                self.toff = nn.Parameter(torch.tensor(np.log(toff+1e-10), device=device).float())
                
            
            
            self.scaling_u = nn.Parameter(torch.tensor(np.log(scaling), device=device).float())
            #self.scaling_s = nn.Parameter(torch.tensor(np.log(scaling_s), device=device).float())
            self.scaling_s = nn.Parameter(torch.zeros(G,device=device).float())
            
        
        if(init_type is None):  
            self.u0 = nn.Parameter(torch.ones(G, device=device).float()*(-10))
            self.s0 = nn.Parameter(torch.ones(G, device=device).float()*(-10))
        elif(init_type == "random"):
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
        else: #use the mean count of the initial type
            cell_labels = adata.obs["clusters"].to_numpy()[train_idx]
            cell_mask = cell_labels==init_type
            self.u0 = nn.Parameter(torch.tensor(np.log(U[cell_mask]/lu[cell_mask].reshape(-1,1)/scaling_u.exp().mean(0)+1e-10), device=device).float())
            self.s0 = nn.Parameter(torch.tensor(np.log(S[cell_mask]/ls[cell_mask].reshape(-1,1)/scaling_s.exp().mean(0)+1e-10), device=device).float())

        self.scaling_u.requires_grad = False
        self.scaling_s.requires_grad = False
    
    def forward(self, t, neg_slope=0.0):
        Uhat, Shat = ode(t, torch.exp(self.alpha), torch.exp(self.beta), torch.exp(self.gamma), torch.exp(self.ton), torch.exp(self.toff), neg_slope=neg_slope)
        return nn.functional.relu(Uhat), nn.functional.relu(Shat)

class VanillaDVAE(VanillaVAE):
    """
    Discrete VeloVAE Model
    """
    def __init__(self, 
                 adata, 
                 tmax, 
                 device='cpu', 
                 hidden_size=(500, 250), 
                 init_method="steady",
                 init_key=None,
                 tprior=None, 
                 time_distribution="gaussian",
                 scale_gene=True,
                 scale_cell=False,
                 separate_us_scale=True,
                 checkpoints=None):
        """Discrete VeloVAE with latent time only
        
        Arguments 
        ---------
        adata : :class:`anndata.AnnData`
        tmax : float
            Time Range 
        device : {'cpu','gpu'}, optional
        hidden_size : tuple, optional
            Width of the first and second hidden layer
        init_type : str, optional
            The stem cell type. Used to estimated the initial conditions.
            This is not commonly used in practice and please consider leaving it to default.
        init_key : str, optional
            column in the AnnData object containing the capture time
        tprior : str, optional
            key in adata.obs that stores the capture time.
            Used for informative time prior
        time_distribution : str, optional
            Should be either "gaussian" or "uniform.
        checkpoints : string list
            Contains the path to saved encoder and decoder models. Should be a .pt file.
        """
        #Extract Input Data
        try:
            U,S = adata.layers['Mu'], adata.layers['Ms']
        except KeyError:
            print('Unspliced/Spliced count matrices not found in the layers! Exit the program...')
        
        #Default Training Configuration
        self.config = {
            #Model Parameters
            "tmax":tmax,
            "hidden_size":hidden_size,
            "init_method": init_method,
            "init_key": init_key,
            "tprior":tprior,
            "tail":0.01,
            "time_overlap":0.5,

            #Training Parameters
            "n_epochs":2000, 
            "batch_size":128,
            "learning_rate":2e-4, 
            "learning_rate_ode":5e-4, 
            "lambda":1e-3, 
            "kl_t":1.0, 
            "test_iter":None, 
            "save_epoch":100,
            "n_warmup":5,
            "early_stop":5,
            "early_stop_thred":1e-3*adata.n_vars,
            "train_test_split":0.7,
            "k_alt":1,
            "neg_slope":0.0,
            "train_scaling":False, 
            "train_std":False, 
            "weight_sample":False,
            "scale_gene":scale_gene,
            "scale_cell":scale_cell,
            "separate_us_scale":separate_us_scale,
            "log1p":True,
            
            #Plotting
            "sparsify":1
        }
        
        self.set_device(device)
        self.split_train_test(adata.n_obs)
        
        N, G = adata.n_obs, adata.n_vars
        try:
            self.encoder = encoder_vanilla(2*G, hidden_size[0], hidden_size[1], self.device, checkpoint=checkpoints).float()
        except IndexError:
            print('Please provide two dimensions!')
        
        self.decoder = decoderBasic(adata, 
                                   tmax, 
                                   self.train_idx,
                                   scale_gene = scale_gene,
                                   scale_cell = scale_cell,
                                   separate_us_scale = separate_us_scale,
                                   device=self.device, 
                                   init_method = init_method,
                                   init_key = init_key,
                                   checkpoint=checkpoints).float()
        
        self.tmax=tmax
        self.time_distribution = time_distribution
        self.get_prior(adata, time_distribution, tmax, tprior)
        
        self.lu_scale = torch.tensor(adata.obs['library_scale_u'].to_numpy()).unsqueeze(-1).float().to(self.device)
        self.ls_scale = torch.tensor(adata.obs['library_scale_s'].to_numpy()).unsqueeze(-1).float().to(self.device)
        

        print(f"Mean library scale: {self.lu_scale.detach().cpu().numpy().mean()}, {self.ls_scale.detach().cpu().numpy().mean()}")

        #Class attributes for training
        self.loss_train, self.loss_test = [],[]
        self.counter = 0 #Count the number of iterations
        self.n_drop = 0 #Count the number of consecutive iterations with little decrease in loss
    
    def set_mode(self,mode):
        #Set the model to either training or evaluation mode.
        
        if(mode=='train'):
            self.encoder.train()
            self.decoder.train()
        elif(mode=='eval'):
            self.encoder.eval()
            self.decoder.eval()
        else:
            print("Warning: mode not recognized. Must be 'train' or 'test'! ")
    
    def forward(self, data_in, lu_scale, ls_scale):
        scaling_u = self.decoder.scaling_u.exp() * lu_scale
        scaling_s = self.decoder.scaling_s.exp() * ls_scale
        data_in_scale = torch.cat((data_in[:,:data_in.shape[1]//2]/scaling_u, data_in[:,data_in.shape[1]//2:]/scaling_s),1)
        if(self.config["log1p"]):
            data_in_log = torch.log1p(data_in_scale)
        else:
            data_in_log = data_in_scale
        mu_t, std_t = self.encoder.forward(data_in_log)
        t = self.reparameterize(mu_t, std_t)
         
        uhat, shat = self.decoder.forward(t, neg_slope=self.config["neg_slope"]) #uhat is scaled
        uhat = uhat*scaling_u
        shat = shat*scaling_s
        
        return mu_t, std_t, t, uhat, shat
    
    def eval_model(self, data_in, lu_scale, ls_scale, continuous=True):
        scaling_u = self.decoder.scaling_u.exp() * lu_scale
        scaling_s = self.decoder.scaling_s.exp() * ls_scale
        data_in_scale = torch.cat((data_in[:,:data_in.shape[1]//2]/scaling_u, data_in[:,data_in.shape[1]//2:]/scaling_s),1)
        if(self.config["log1p"]):
            data_in_log = torch.log1p(data_in_scale)
        else:
            data_in_log = data_in_scale
        mu_t, std_t = self.encoder.forward(data_in_log)
        
        uhat, shat = self.decoder.forward(mu_t, neg_slope=self.config["neg_slope"]) #uhat is scaled
        uhat = uhat*scaling_u
        shat = shat*scaling_s
        
        if(not continuous):
            poisson_u = Poisson(F.softplus(uhat, beta=100))
            poisson_s = Poisson(F.softplus(shat, beta=100))
            u_out = poisson_u.sample()
            s_out = poisson_s.sample()
            return mu_t, std_t, mu_z, std_z, u_out, s_out
        return mu_t, std_t, uhat, shat
    
    
    def sample_poisson(self, uhat, shat):
        u_sample = torch.poisson(uhat)
        s_sample = torch.poisson(shat)
        return u_sample.cpu(), s_sample.cpu()
    
    def sample_nb(self, uhat, shat, pu, ps):
        u_nb = NegativeBinomial(uhat*(1-pu)/pu, pu.repeat(uhat.shape[0],1))
        s_nb = NegativeBinomial(shat*(1-ps)/ps, ps.repeat(shat.shape[0],1))
        return u_nb.sample(), s_nb.sample()
    
    def vae_risk_poisson(self, 
                         q_tx, p_t, 
                         u, s, uhat, shat, 
                         weight=None,
                         eps=1e-6):
        kldt = self.kl_time(q_tx[0], q_tx[1], p_t[0], p_t[1], tail=self.config["tail"])

        #poisson
        try:
            poisson_u = Poisson(F.relu(uhat)+1e-2)
            poisson_s = Poisson(F.relu(shat)+1e-2)
        except ValueError:
            uhat[torch.isnan(uhat)] = 0
            shat[torch.isnan(shat)] = 0
            poisson_u = Poisson(F.relu(uhat)+1e-2)
            poisson_s = Poisson(F.relu(shat)+1e-2)
        
        weight_u = (u==0).float()*0.05 + (u>0).float()
        weight_s = (s==0).float()*0.05 + (s>0).float()
        logp = poisson_u.log_prob(u) * weight_u + poisson_s.log_prob(s) * weight_s
        
        if( weight is not None):
            logp = logp*weight
        
        err_rec = torch.mean(torch.sum(logp,1))
        
        return (- err_rec + self.config["kl_t"]*kldt)
    
    def train_epoch(self, train_loader, test_set, optimizer, optimizer2=None, K=1):
        B = len(train_loader)
        self.set_mode('train')
        stop_training = False
            
        for i, batch in enumerate(train_loader):
            if(self.counter == 1 or self.counter % self.config["test_iter"] == 0):
                elbo_test = self.test(test_set, 
                                      None, 
                                      self.counter, 
                                      True)
                
                if(len(self.loss_test)>0): #update the number of epochs with dropping ELBO
                    if(elbo_test - self.loss_test[-1] <= self.config["early_stop_thred"]):
                        self.n_drop = self.n_drop+1
                    else:
                        self.n_drop = 0
                self.loss_test.append(elbo_test)
                self.set_mode('train')
                
                if(self.n_drop>=self.config["early_stop"] and self.config["early_stop"]>0):
                    stop_training = True
                    break
            
            optimizer.zero_grad()
            if(optimizer2 is not None):
                optimizer2.zero_grad()
            
            xbatch, idx = batch[0].to(self.device), batch[3]
            u = xbatch[:,:xbatch.shape[1]//2]
            s = xbatch[:,xbatch.shape[1]//2:]
            
            
            lu_scale = self.lu_scale[self.train_idx[idx]]
            ls_scale = self.ls_scale[self.train_idx[idx]]
            mu_tx, std_tx, t, uhat, shat = self.forward(xbatch.float(), lu_scale, ls_scale)
            
            loss = self.vae_risk_poisson((mu_tx, std_tx), self.p_t[:,self.train_idx[idx],:],
                                          u.int(),
                                          s.int(), 
                                          uhat, shat,
                                          None)
            
            loss.backward()
            if(K==0):
                optimizer.step()
                if( optimizer2 is not None ):
                    optimizer2.step()
            else:
                if( optimizer2 is not None and ((i+1) % (K+1) == 0 or i==B-1)):
                    optimizer2.step()
                else:
                    optimizer.step()
            
            self.loss_train.append(loss.detach().cpu().item())
            self.counter = self.counter + 1
        return stop_training
    
    def train(self, 
              adata, 
              U_raw,
              S_raw,
              config={}, 
              plot=False, 
              gene_plot=[], 
              cluster_key = "clusters",
              figure_path = "figures", 
              embed="umap"):
        self.train_mse = []
        self.test_mse = []
        
        self.load_config(config)
        
        print("------------------------ Train a Vanilla VAE ------------------------")
        #Get data loader
        U, S = np.array(adata.layers['unspliced'].todense()), np.array(adata.layers['spliced'].todense())
        X = np.concatenate((U,S), 1)
        
        try:
            Xembed = adata.obsm[f"X_{embed}"]
        except KeyError:
            print("Embedding not found! Set to None.")
            Xembed = None
            plot = False
        
        cell_labels_raw = adata.obs[cluster_key].to_numpy() if cluster_key in adata.obs else np.array(['Unknown' for i in range(adata.n_obs)])
        #Encode the labels
        cell_types_raw = np.unique(cell_labels_raw)
        self.label_dic, self.label_dic_rev = encode_type(cell_types_raw)

        self.n_type = len(cell_types_raw)
        self.cell_labels = np.array([self.label_dic[x] for x in cell_labels_raw])
        self.cell_types = np.array([self.label_dic[cell_types_raw[i]] for i in range(self.n_type)])
        
        print("*********        Creating Training/Validation Datasets        *********")
        train_set = SCData(X[self.train_idx], self.cell_labels[self.train_idx], self.decoder.Rscore[self.train_idx]) if self.config['weight_sample'] else SCData(X[self.train_idx], self.cell_labels[self.train_idx])
        test_set = None
        if(len(self.test_idx)>0):
            test_set = SCData(X[self.test_idx], self.cell_labels[self.test_idx], self.decoder.Rscore[self.test_idx]) if self.config['weight_sample'] else SCData(X[self.test_idx], self.cell_labels[self.test_idx])
        data_loader = torch.utils.data.DataLoader(train_set, batch_size=self.config["batch_size"], shuffle=True)
        #Automatically set test iteration if not given
        if(self.config["test_iter"] is None):
            self.config["test_iter"] = len(self.train_idx)//self.config["batch_size"]*2
        print("*********                      Finished.                      *********")
        
        gind, gene_plot = get_gene_index(adata.var_names, gene_plot)
        os.makedirs(figure_path, exist_ok=True)
        
        #define optimizer
        print("*********                 Creating optimizers                 *********")
        param_nn = list(self.encoder.parameters())
        param_ode = [self.decoder.alpha, self.decoder.beta, self.decoder.gamma, self.decoder.ton, self.decoder.toff] 
        if(self.config['train_scaling']):
            param_ode = param_ode+[self.decoder.scaling_u, self.decoder.scaling_s]

        optimizer = torch.optim.Adam(param_nn, lr=self.config["learning_rate"], weight_decay=self.config["lambda"])
        optimizer_ode = torch.optim.Adam(param_ode, lr=self.config["learning_rate_ode"])
        print("*********                      Finished.                      *********")
      
        #Main Training Process
        print("*********                    Start training                   *********")
        print(f"Total Number of Iterations Per Epoch: {len(data_loader)}, test iteration: {self.config['test_iter']}")
        
        n_epochs = self.config["n_epochs"]
        
        start = time.time()
        for epoch in range(n_epochs):
            #Train the encoder
            if(self.config["k_alt"] is None):
                stop_training = self.train_epoch(data_loader, test_set, optimizer)
                
                if(epoch>=self.config["n_warmup"]):
                    stop_training_ode = self.train_epoch(data_loader, test_set, optimizer_ode)
                    if(stop_training_ode):
                        print(f"*********       Stage 1: Early Stop Triggered at epoch {epoch+1}.       *********")
                        break
            else:
                if(epoch>=self.config["n_warmup"]):
                    stop_training = self.train_epoch(data_loader, test_set, optimizer_ode, optimizer, self.config["k_alt"])
                else:
                    stop_training = self.train_epoch(data_loader, test_set, optimizer, None, self.config["k_alt"])
            
            if(plot and (epoch==0 or (epoch+1) % self.config["save_epoch"] == 0)):
                elbo_train = self.test(train_set,
                                       Xembed[self.train_idx],
                                       f"train{epoch+1}", 
                                       False,
                                       gind, 
                                       gene_plot,
                                       True, 
                                       figure_path)
                self.set_mode('train')
                elbo_test = self.loss_test[-1] if len(self.loss_test)>0 else -np.inf
                print(f"Epoch {epoch+1}: Train ELBO = {elbo_train:.3f}, Test ELBO = {elbo_test:.3f}, \t Total Time = {convert_time(time.time()-start)}")
            
            if(stop_training):
                print(f"********* Early Stop Triggered at epoch {epoch+1}. *********")
                break
        
        print(f"*********              Finished. Total Time = {convert_time(time.time()-start)}             *********")
        #Plot final results
        if(plot):
            elbo_train = self.test(train_set,
                                   Xembed[self.train_idx],
                                   "final-train", 
                                   False,
                                   gind, 
                                   gene_plot,
                                   True, 
                                   figure_path)
            elbo_test = self.test(test_set,
                                  Xembed[self.test_idx],
                                  "final-test", 
                                  True,
                                  gind, 
                                  gene_plot,
                                  True, 
                                  figure_path)
            print(f"Final: Train ELBO = {elbo_train:.3f}, Test ELBO = {elbo_test:.3f}")
            plot_train_loss(self.loss_train, range(1,len(self.loss_train)+1), save=f'{figure_path}/train_loss_velovae.png')
            if(self.config["test_iter"]>0):
                plot_test_loss(self.loss_test, [i*self.config["test_iter"] for i in range(1,len(self.loss_test)+1)], save=f'{figure_path}/test_loss_velovae.png')
        
        if(plot):
            plot_train_loss(self.train_mse, range(1,len(self.train_mse)+1), save=f'{figure_path}/train_mse_velovae.png')
            if(self.config["test_iter"]>0):
                plot_test_loss(self.test_mse, [i*self.config["test_iter"] for i in range(1,len(self.test_mse)+1)], save=f'{figure_path}/test_mse_velovae.png')
    
    def pred_all(self, data, cell_labels, mode='test', output=["uhat", "shat", "t"], gene_idx=None, continuous=True):
        N, G = data.shape[0], data.shape[1]//2
        elbo, mse = 0, 0
        if("uhat" in output):
            Uhat = None if gene_idx is None else np.zeros((N,len(gene_idx)))
        if("shat" in output):
            Shat = None if gene_idx is None else np.zeros((N,len(gene_idx)))
        if("t" in output):
            t_out = np.zeros((N))
            std_t_out = np.zeros((N))
        
        corr = 0
        with torch.no_grad():
            B = min(N//3, 5000)
            Nb = N // B
            for i in range(Nb):
                data_in = torch.tensor(data[i*B:(i+1)*B]).float().to(self.device)
                if(mode=="test"):
                    p_t = self.p_t[:,self.test_idx[i*B:(i+1)*B],:]
                    lu_scale = self.lu_scale[self.test_idx[i*B:(i+1)*B]]
                    ls_scale = self.ls_scale[self.test_idx[i*B:(i+1)*B]]
                elif(mode=="train"):
                    p_t = self.p_t[:,self.train_idx[i*B:(i+1)*B],:]
                    lu_scale = self.lu_scale[self.train_idx[i*B:(i+1)*B]]
                    ls_scale = self.ls_scale[self.train_idx[i*B:(i+1)*B]]
                else:
                    p_t = self.p_t[:,i*B:(i+1)*B,:]
                    lu_scale = self.lu_scale[i*B:(i+1)*B]
                    ls_scale = self.ls_scale[i*B:(i+1)*B]
                mu_tx, std_tx, uhat, shat = self.eval_model(data_in, lu_scale, ls_scale)
                u_raw, s_raw = data_in[:,:G], data_in[:,G:]
                #u_sum, s_sum = torch.sum(uhat, 1, keepdim=True), torch.sum(shat, 1, keepdim=True)
                
                loss = self.vae_risk_poisson((mu_tx, std_tx), p_t,
                                             u_raw.int(), s_raw.int(), 
                                             uhat, shat, 
                                             None)
                u_sample, s_sample = self.sample_poisson(uhat, shat)
                
                mse_batch = np.mean( (u_raw.cpu().numpy() - u_sample.cpu().numpy())**2 + ( s_raw.cpu().numpy() - s_sample.cpu().numpy())**2)
                elbo = elbo - (B/N)*loss
                mse = mse + (B/N) * mse_batch
                #corr = corr + (B/N) * self.corr_vel(uhat*lu_scale, shat*ls_scale, rho).detach().cpu().item()
                
                if("uhat" in output and gene_idx is not None):
                    Uhat[i*B:(i+1)*B] = uhat[:,gene_idx].cpu().numpy()
                if("shat" in output and gene_idx is not None):
                    Shat[i*B:(i+1)*B] = shat[:,gene_idx].cpu().numpy()
                if("t" in output):
                    t_out[i*B:(i+1)*B] = mu_tx.cpu().squeeze().numpy()
                    std_t_out[i*B:(i+1)*B] = std_tx.cpu().squeeze().numpy()
                
            if(N > B*Nb):
                data_in = torch.tensor(data[B*Nb:]).float().to(self.device)
                if(mode=="test"):
                    p_t = self.p_t[:,self.test_idx[B*Nb:],:]
                    lu_scale = self.lu_scale[self.test_idx[B*Nb:]]
                    ls_scale = self.ls_scale[self.test_idx[B*Nb:]]
                elif(mode=="train"):
                    p_t = self.p_t[:,self.train_idx[B*Nb:],:]
                    lu_scale = self.lu_scale[self.train_idx[B*Nb:]]
                    ls_scale = self.ls_scale[self.train_idx[B*Nb:]]
                else:
                    p_t = self.p_t[:,B*Nb:,:]
                    lu_scale = self.lu_scale[B*Nb:]
                    ls_scale = self.ls_scale[B*Nb:]
                mu_tx, std_tx, uhat, shat = self.eval_model(data_in, lu_scale, ls_scale)
                u_raw, s_raw = data_in[:,:G], data_in[:,G:]
                #u_sum, s_sum = torch.sum(uhat, 1, keepdim=True), torch.sum(shat, 1, keepdim=True)
                
                loss = self.vae_risk_poisson((mu_tx, std_tx), p_t,
                                             u_raw.int(), s_raw.int(), 
                                             uhat, shat,
                                             None)
                u_sample, s_sample = self.sample_poisson(uhat, shat)
                
                mse_batch = np.mean( (u_raw.cpu().numpy() - u_sample.cpu().numpy())**2 + ( s_raw.cpu().numpy() - s_sample.cpu().numpy())**2)
                elbo = elbo - ((N-B*Nb)/N)*loss
                mse = mse + ((N-B*Nb)/N)*mse_batch

                if("uhat" in output and gene_idx is not None):
                    Uhat[Nb*B:] = uhat[:,gene_idx].cpu().numpy()
                if("shat" in output and gene_idx is not None):
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
            out.append(t_out)
            out.append(std_t_out)

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
        self.set_mode('eval')
        mode = "test" if test_mode else "train"
        out, elbo = self.pred_all(dataset.data, self.cell_labels, mode, ["uhat", "shat", "t"], gind)
        Uhat, Shat, t, std_t = out[0], out[1], out[2], out[3]
        
        
        if(test_mode):
            self.test_mse.append(out[-1])
        else:
            self.train_mse.append(out[-1])
        
        if(plot):
            #Plot Time
            t_ub = np.quantile(t, 0.99)
            plot_time(np.clip(t,None,t_ub), Xembed, save=f"{path}/time-{testid}-velovae.png")
            
            #Plot u/s-t and phase portrait for each gene
            G = dataset.G
            
            for i in range(len(gind)):
                idx = gind[i]
                
                plot_sig(t.squeeze(), 
                         dataset.data[:,idx], dataset.data[:,idx+G], 
                         Uhat[:,i], Shat[:,i], 
                         np.array([self.label_dic_rev[x] for x in dataset.labels]),
                         gene_plot[i], 
                         save=f"{path}/sig-{gene_plot[i]}-{testid}.png",
                         sparsify=self.config['sparsify'])
                
        return elbo
    
    def gene_likelihood_poisson(self, 
                                U,S,
                                Uhat, Shat, 
                                b=5000):
        N,G = U.shape
        Nb = N//b
        logpu, logps = np.empty((N,G)), np.empty((N,G))
        for i in range(Nb):
            ps_u = Poisson(torch.tensor(Uhat[i*b:(i+1)*b]))
            ps_s = Poisson(torch.tensor(Shat[i*b:(i+1)*b]))
            logpu[i*b:(i+1)*b] = ps_u.log_prob(torch.tensor(U[i*b:(i+1)*b], dtype=int))
            logps[i*b:(i+1)*b] = ps_s.log_prob(torch.tensor(S[i*b:(i+1)*b], dtype=int))
        if(Nb*b<N):
            ps_u = Poisson(torch.tensor(Uhat[Nb*b:]))
            ps_s = Poisson(torch.tensor(Shat[Nb*b:]))
            logpu[Nb*b:] = ps_u.log_prob(torch.tensor(U[Nb*b:], dtype=int))
            logps[Nb*b:] = ps_s.log_prob(torch.tensor(S[Nb*b:], dtype=int))
        return np.exp((logpu+logps)).mean(0)
    
    def save_anndata(self, adata, key, file_path, file_name=None):
        self.set_mode('eval')
        os.makedirs(file_path, exist_ok=True)
        
        adata.var[f"{key}_alpha"] = np.exp(self.decoder.alpha.detach().cpu().numpy())
        adata.var[f"{key}_beta"] = np.exp(self.decoder.beta.detach().cpu().numpy())
        adata.var[f"{key}_gamma"] = np.exp(self.decoder.gamma.detach().cpu().numpy())
        adata.var[f"{key}_toff"] = np.exp(self.decoder.toff.detach().cpu().numpy())
        adata.var[f"{key}_ton"] = np.exp(self.decoder.ton.detach().cpu().numpy())
        adata.var[f"{key}_scaling_u"] = self.decoder.scaling_u.exp().detach().cpu().numpy()
        adata.var[f"{key}_scaling_s"] = self.decoder.scaling_s.exp().detach().cpu().numpy()
        
        U,S = np.array(adata.layers['unspliced'].todense()), np.array(adata.layers['spliced'].todense())
        X = np.concatenate((U,S),1)

        out, elbo = self.pred_all(X, self.cell_labels, "both", gene_idx=np.array(range(adata.n_vars)))
        Uhat, Shat, t, std_t = out
        
        adata.obs[f"{key}_time"] = t
        adata.obs[f"{key}_std_t"] = std_t
        
        adata.layers[f"{key}_uhat"] = Uhat
        adata.layers[f"{key}_shat"] = Shat
        
        adata.uns[f"{key}_train_idx"] = self.train_idx
        adata.uns[f"{key}_test_idx"] = self.test_idx
        
        scaling_u = adata.var[f"{key}_scaling_u"].to_numpy()
        scaling_s = adata.var[f"{key}_scaling_s"].to_numpy()
        lu_scale = adata.obs["library_scale_u"].to_numpy().reshape(-1,1)
        ls_scale = adata.obs["library_scale_s"].to_numpy().reshape(-1,1)
        adata.layers[f"{key}_velocity"] = adata.var[f"{key}_beta"].to_numpy() * Uhat / scaling_u / lu_scale - adata.var[f"{key}_gamma"].to_numpy() * Shat / scaling_s / ls_scale
        
        if(file_name is not None):
            adata.write_h5ad(f"{file_path}/{file_name}")
















############################################################
#model_util
############################################################
def knnx0_alt(U, S, t, z, t_query, z_query, dt, k):
    N, Nq = len(t), len(t_query)
    u0 = np.zeros((Nq, U.shape[1]))
    s0 = np.zeros((Nq, S.shape[1]))
    t0 = np.ones((Nq))*(t.min() - dt[0])
    
    order_idx = np.argsort(t)
    _t = t[order_idx]
    _z = z[order_idx]
    _U = U[order_idx]
    _S = S[order_idx]
    
    order_query = np.argsort(t_query)
    _t_query = t_query[order_query]
    _z_query = z_query[order_query]
    
    knn = np.ones((Nq,k))*np.nan
    D = np.ones((Nq,k))*np.nan
    ptr = 0
    left, right = 0, 0 #pointer in the query sequence
    i = 0
    while(left<Nq and i<N): #i as initial point x0
        #Update left, right
        if(_t[i]+dt[0]>=_t_query[-1]):
            break;
        for l in range(left, Nq):
            if(_t_query[l]>=_t[i]+dt[0]):
                left = l
                break
        for l in range(right, Nq):
            if(_t_query[l]>=_t[i]+dt[1]):
                right = l
                break
        
        #Update KNN
        for j in range(left, right): #j is the set of cell with i in the range [tj-dt,tj-dt/2]
            dist = np.linalg.norm(_z[i]-_z_query[j])
            pos_nan = np.where(np.isnan(knn[j]))[0]
            if(len(pos_nan)>0): #there hasn't been k nearest neighbors for j yet
                knn[j,pos_nan[0]] = i
                D[j,pos_nan[0]] = dist
            else:
                idx_largest = np.argmax(D[j])
                if(dist<D[j,idx_largest]):
                    D[j,idx_largest] = dist
                    knn[j,idx_largest] = i
        i += 1
    #Calculate initial time and conditions
    for i in range(Nq):
        if(np.all(np.isnan(knn[i]))):
            continue
        pos = np.where(~np.isnan(knn[i]))[0]
        u0[order_query[i]] = _U[knn[i,pos].astype(int)].mean(0)
        s0[order_query[i]] = _S[knn[i,pos].astype(int)].mean(0)
        t0[order_query[i]] = _t[knn[i,pos].astype(int)].mean()
    
    return u0,s0,t0

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





# @ Lénaïc Chizat 2015 - optimal transport
def fdiv(l, x, p, dx):
    return l * np.sum(dx * (x * (np.log(x / p)) - x + p))


def fdivstar(l, u, p, dx):
    return l * np.sum((p * dx) * (np.exp(u / l) - 1))


def primal(C, K, R, dx, dy, p, q, a, b, epsilon, lambda1, lambda2):
    I = len(p)
    J = len(q)
    F1 = lambda x, y: fdiv(lambda1, x, p, y)
    F2 = lambda x, y: fdiv(lambda2, x, q, y)
    with np.errstate(divide='ignore'):
        return F1(np.dot(R, dy), dx) + F2(np.dot(R.T, dx), dy) \
               + (epsilon * np.sum(R * np.nan_to_num(np.log(R)) - R + K) \
                  + np.sum(R * C)) / (I * J)


def dual(C, K, R, dx, dy, p, q, a, b, epsilon, lambda1, lambda2):
    I = len(p)
    J = len(q)
    F1c = lambda u, v: fdivstar(lambda1, u, p, v)
    F2c = lambda u, v: fdivstar(lambda2, u, q, v)
    return - F1c(- epsilon * np.log(a), dx) - F2c(- epsilon * np.log(b), dy) \
           - epsilon * np.sum(R - K) / (I * J)


# end @ Lénaïc Chizat

def optimal_transport_duality_gap(C, G, lambda1, lambda2, epsilon, batch_size, tolerance, tau,
                                  epsilon0, max_iter, **ignored):
    """
    Compute the optimal transport with stabilized numerics, with the guarantee that the duality gap is at most `tolerance`
    Code is from `the work by Schiebinger et al. <https://www.cell.com/cell/fulltext/S0092-8674(19)30039-X?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS009286741930039X%3Fshowall%3Dtrue>`
    
    Arguments
    ----------
    C : 2-D ndarray
        The cost matrix. C[i][j] is the cost to transport cell i to cell j
    G : 1-D array_like
        Growth value for input cells.
    lambda1 : float, optional
        Regularization parameter for the marginal constraint on p
    lambda2 : float, optional
        Regularization parameter for the marginal constraint on q
    epsilon : float, optional
        Entropy regularization parameter.
    batch_size : int, optional
        Number of iterations to perform between each duality gap check
    tolerance : float, optional
        Upper bound on the duality gap that the resulting transport map must guarantee.
    tau : float, optional
        Threshold at which to perform numerical stabilization
    epsilon0 : float, optional
        Starting value for exponentially-decreasing epsilon
    max_iter : int, optional
        Maximum number of iterations. Print a warning and return if it is reached, even without convergence.
    
    Returns
    -------
    transport_map : 2-D ndarray
        The entropy-regularized unbalanced transport map
    """
    C = np.asarray(C, dtype=np.float64)
    epsilon_scalings = 5
    scale_factor = np.exp(- np.log(epsilon) / epsilon_scalings)

    I, J = C.shape
    dx, dy = np.ones(I) / I, np.ones(J) / J

    p = G
    q = np.ones(C.shape[1]) * np.average(G)

    u, v = np.zeros(I), np.zeros(J)
    a, b = np.ones(I), np.ones(J)

    epsilon_i = epsilon0 * scale_factor
    current_iter = 0

    for e in range(epsilon_scalings + 1):
        duality_gap = np.inf
        u = u + epsilon_i * np.log(a)
        v = v + epsilon_i * np.log(b)  # absorb
        epsilon_i = epsilon_i / scale_factor
        _K = np.exp(-C / epsilon_i)
        alpha1 = lambda1 / (lambda1 + epsilon_i)
        alpha2 = lambda2 / (lambda2 + epsilon_i)
        K = np.exp((np.array([u]).T - C + np.array([v])) / epsilon_i)
        a, b = np.ones(I), np.ones(J)
        old_a, old_b = a, b
        threshold = tolerance if e == epsilon_scalings else 1e-6

        while duality_gap > threshold:
            for i in range(batch_size if e == epsilon_scalings else 5):
                current_iter += 1
                old_a, old_b = a, b
                a = (p / (K.dot(np.multiply(b, dy)))) ** alpha1 * np.exp(-u / (lambda1 + epsilon_i))
                b = (q / (K.T.dot(np.multiply(a, dx)))) ** alpha2 * np.exp(-v / (lambda2 + epsilon_i))

                # stabilization
                if (max(max(abs(a)), max(abs(b))) > tau):
                    u = u + epsilon_i * np.log(a)
                    v = v + epsilon_i * np.log(b)  # absorb
                    K = np.exp((np.array([u]).T - C + np.array([v])) / epsilon_i)
                    a, b = np.ones(I), np.ones(J)

                if current_iter >= max_iter:
                    print("Reached max_iter with duality gap still above threshold. Returning")
                    return (K.T * a).T * b

            # The real dual variables. a and b are only the stabilized variables
            _a = a * np.exp(u / epsilon_i)
            _b = b * np.exp(v / epsilon_i)

            # Skip duality gap computation for the first epsilon scalings, use dual variables evolution instead
            if e == epsilon_scalings:
                R = (K.T * a).T * b
                pri = primal(C, _K, R, dx, dy, p, q, _a, _b, epsilon_i, lambda1, lambda2)
                dua = dual(C, _K, R, dx, dy, p, q, _a, _b, epsilon_i, lambda1, lambda2)
                duality_gap = (pri - dua) / abs(pri)
            else:
                duality_gap = max(
                    np.linalg.norm(_a - old_a * np.exp(u / epsilon_i)) / (1 + np.linalg.norm(_a)),
                    np.linalg.norm(_b - old_b * np.exp(v / epsilon_i)) / (1 + np.linalg.norm(_b)))

    if np.isnan(duality_gap):
        raise RuntimeError("Overflow encountered in duality gap computation, please report this incident")
    return R / C.shape[1]

"""
Pytorch Version
"""
def fdiv_ts(l, x, p, dx):
    return l * torch.sum(dx * (x * (torch.log(x / p)) - x + p))


def fdivstar_ts(l, u, p, dx):
    return l * torch.sum((p * dx) * (torch.exp(u / l) - 1))


def primal_ts(C, K, R, dx, dy, p, q, a, b, epsilon, lambda1, lambda2):
    I = len(p)
    J = len(q)
    F1 = lambda x, y: fdiv_ts(lambda1, x, p, y)
    F2 = lambda x, y: fdiv_ts(lambda2, x, q, y)
    with np.errstate(divide='ignore'):
        return F1(torch.sum(R*dy, 1), dx) + F2(torch.sum(R.T*dx, 1), dy) \
               + (epsilon * torch.sum(R * torch.nan_to_num(torch.log(R)) - R + K) \
                  + torch.sum(R * C)) / (I * J)


def dual_ts(C, K, R, dx, dy, p, q, a, b, epsilon, lambda1, lambda2):
    I = len(p)
    J = len(q)
    F1c = lambda u, v: fdivstar_ts(lambda1, u, p, v)
    F2c = lambda u, v: fdivstar_ts(lambda2, u, q, v)
    return - F1c(- epsilon * torch.log(a), dx) - F2c(- epsilon * torch.log(b), dy) \
           - epsilon * torch.sum(R - K) / (I * J)


# end @ Lénaïc Chizat

def optimal_transport_duality_gap_ts(C, G, lambda1, lambda2, epsilon, batch_size, tolerance, tau,
                                  epsilon0, max_iter, **ignored):
    # C = C.double()
    epsilon_scalings = 5
    scale_factor = np.exp(- np.log(epsilon) / epsilon_scalings)

    I, J = C.shape
    dx, dy = torch.ones(I, device=C.device, dtype=C.dtype) / I, torch.ones(J, device=C.device, dtype=C.dtype) / J

    p = G
    q = torch.ones(C.shape[1], device=C.device, dtype=C.dtype) * (G.mean())

    u, v = torch.zeros(I, device=C.device, dtype=C.dtype), torch.zeros(J, device=C.device, dtype=C.dtype)
    a, b = torch.ones(I, device=C.device, dtype=C.dtype), torch.ones(J, device=C.device, dtype=C.dtype)

    epsilon_i = epsilon0 * scale_factor
    current_iter = 0

    for e in range(epsilon_scalings + 1):
        duality_gap = np.inf
        u = u + epsilon_i * torch.log(a)
        v = v + epsilon_i * torch.log(b)  # absorb
        epsilon_i = epsilon_i / scale_factor
        _K = torch.exp(-C / epsilon_i)
        alpha1 = lambda1 / (lambda1 + epsilon_i)
        alpha2 = lambda2 / (lambda2 + epsilon_i)
        K = torch.exp((u.view(-1,1) - C + v.view(1,-1)) / epsilon_i)
        a, b = torch.ones(I, device=C.device, dtype=C.dtype), torch.ones(J, device=C.device, dtype=C.dtype)
        old_a, old_b = a, b
        threshold = tolerance if e == epsilon_scalings else 1e-6

        while duality_gap > threshold:
            for i in range(batch_size if e == epsilon_scalings else 5):
                current_iter += 1
                old_a, old_b = a, b
                a = (p / ( torch.sum(K * (b*dy), 1))).pow(alpha1) * torch.exp(-u / (lambda1 + epsilon_i))
                b = (q / ( torch.sum(K.T*(a*dx), 1))).pow(alpha2) * torch.exp(-v / (lambda2 + epsilon_i))

                # stabilization
                if (max(torch.abs(a).max(), torch.abs(b).max()) > tau):
                    u = u + epsilon_i * torch.log(a)
                    v = v + epsilon_i * torch.log(b)  # absorb
                    K = torch.exp((u.view(-1,1) - C + v.view(1,-1)) / epsilon_i)
                    a, b = torch.ones(I, device=C.device, dtype=C.dtype), torch.ones(J, device=C.device, dtype=C.dtype)

                if current_iter >= max_iter:
                    print(f"Reached max_iter with duality gap still above threshold ({duality_gap:.5f}). Returning")
                    return (K.T * a).T * b

            # The real dual variables. a and b are only the stabilized variables
            _a = a * torch.exp(u / epsilon_i)
            _b = b * torch.exp(v / epsilon_i)

            # Skip duality gap computation for the first epsilon scalings, use dual variables evolution instead
            if e == epsilon_scalings:
                R = (K.T * a).T * b
                pri = primal_ts(C, _K, R, dx, dy, p, q, _a, _b, epsilon_i, lambda1, lambda2)
                dua = dual_ts(C, _K, R, dx, dy, p, q, _a, _b, epsilon_i, lambda1, lambda2)
                duality_gap = (pri - dua) / abs(pri)
            else:
                duality_gap = max(
                    torch.norm(_a - old_a * torch.exp(u / epsilon_i)) / (1 + torch.norm(_a)),
                    torch.norm(_b - old_b * torch.exp(v / epsilon_i)) / (1 + torch.norm(_b)))

    if torch.isnan(duality_gap):
        raise RuntimeError("Overflow encountered in duality gap computation, please report this incident")
    return R / C.shape[1]

def transition_prob_util(x_embed, t, cell_labels, nbin=20, epsilon = 0.05, batch_size = 5, lambda1 = 1, lambda2 = 50, max_iter = 2000, q = 0.01):
    cell_types = np.unique(cell_labels)
    Ntype = len(cell_types)
    dt = (np.quantile(t,0.999)-t.min())/(nbin) #time resolution
    
    P = np.zeros((Ntype, Ntype))
    t_trans = []
    for i, x in enumerate(cell_types): #child type
        mask = cell_labels==x
        t0 = np.quantile(t[mask], q) #estimated transition time
        t_trans.append(t0)
        
        mask1 = (t>=t0-dt) & (t<t0) 
        mask2 = (t>=t0) & (t<t0+dt) & mask
        
        if(np.any(mask1) and np.any(mask2)):
            x1, x2 = x_embed[mask1], x_embed[mask2]
            C = pairwise_distances(x1, x2, metric='sqeuclidean', n_jobs=-1)
            C = C/np.median(C)
            g = np.power(np.sum(mask2)/np.sum(mask1), 1/dt)
            G = np.ones((C.shape[0]))*g
            print(g)
            
            Pi = optimal_transport_duality_gap(C,G,lambda1, lambda2, epsilon, 5, 0.01, 10000, 1, max_iter)
            Pi[np.isnan(Pi)] = 0
            Pi[np.isinf(Pi)] = 0
            
            #Sum the weights of each cell type
            cell_labels_1 = cell_labels[mask1]
            cell_labels_2 = cell_labels[mask2]
            
            for j, y in enumerate(cell_types): #parent
                if(np.any(cell_labels_1==y) and np.any(cell_labels_2==x)):
                    P[i,j] = np.sum(np.array(Pi[cell_labels_1==y]))
                    
        sum_p = P[i].sum()
        sum_p = sum_p + (sum_p==0)
        P[i] = P[i]/sum_p
    
    return P, cell_types, t_trans


#######################################################
# Full Variational Bayes
#   This has been absorbed into the VAE class
#######################################################
class decoder_fullvb(nn.Module):
    def __init__(self,
                 adata,
                 tmax,
                 train_idx,
                 dim_z,
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
                 **kwargs):
        super(decoder_fullvb, self).__init__()
        G = adata.n_vars

        sigma_param = np.log(0.05)

        if checkpoint is None:
            # Dynamical Model Parameters
            U, S = adata.layers['Mu'][train_idx], adata.layers['Ms'][train_idx]
            X = np.concatenate((U, S), 1)
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
                self.alpha = nn.Parameter(torch.normal(0.0, 1.0, size=(2, G), device=device).float())
                self.beta = nn.Parameter(torch.normal(0.0, 0.5, size=(2, G), device=device).float())
                self.gamma = nn.Parameter(torch.normal(0.0, 0.5, size=(2, G), device=device).float())
                self.ton = torch.nn.Parameter(torch.ones(len(alpha), device=device).float()*(-10))
                self.toff = torch.nn.Parameter(torch.ones(G, device=device).float()*(tmax/2))
                self.scaling = nn.Parameter(torch.tensor(np.log(scaling), device=device).float())
                self.sigma_u = nn.Parameter(torch.tensor(np.log(sigma_u), device=device).float())
                self.sigma_s = nn.Parameter(torch.tensor(np.log(sigma_s), device=device).float())
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

                self.alpha = nn.Parameter(torch.tensor(np.stack([np.log(self.alpha_init), sigma_param*np.ones((G))]),
                                                       device=device).float())
                self.beta = nn.Parameter(torch.tensor(np.stack([np.log(self.beta_init), sigma_param*np.ones((G))]),
                                                      device=device).float())
                self.gamma = nn.Parameter(torch.tensor(np.stack([np.log(self.gamma_init), sigma_param*np.ones((G))]),
                                                       device=device).float())
                self.scaling = nn.Parameter(torch.tensor(np.log(scaling), device=device).float())
                self.sigma_u = nn.Parameter(torch.tensor(np.log(sigma_u), device=device).float())
                self.sigma_s = nn.Parameter(torch.tensor(np.log(sigma_s), device=device).float())
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

                self.alpha = nn.Parameter(torch.tensor(np.stack([np.log(self.alpha_init), sigma_param*np.ones((G))]),
                                                       device=device).float())
                self.beta = nn.Parameter(torch.tensor(np.stack([np.log(self.beta_init), sigma_param*np.ones((G))]),
                                                      device=device).float())
                self.gamma = nn.Parameter(torch.tensor(np.stack([np.log(self.gamma_init), sigma_param*np.ones((G))]),
                                                       device=device).float())
                self.scaling = nn.Parameter(torch.tensor(np.log(scaling), device=device).float())
                self.ton = (nn.Parameter((torch.ones(adata.n_vars, device=device)*(-10)).float())
                            if init_ton_zero else
                            nn.Parameter(torch.tensor(np.log(self.ton_init+1e-10), device=device).float()))
                self.toff = nn.Parameter(torch.tensor(np.log(self.toff_init+1e-10), device=device).float())
                self.sigma_u = nn.Parameter(torch.tensor(np.log(sigma_u), device=device).float())
                self.sigma_s = nn.Parameter(torch.tensor(np.log(sigma_s), device=device).float())
                self.logit_pw = nn.Parameter(torch.tensor(logit_pw, device=device).float())

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
            self.alpha = nn.Parameter(torch.empty((2, G), device=device).float())
            self.beta = nn.Parameter(torch.empty((2, G), device=device).float())
            self.gamma = nn.Parameter(torch.empty((2, G), device=device).float())
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

    def reparameterize_param(self):
        eps = torch.normal(mean=torch.zeros((3, self.alpha.shape[1])),
                           std=torch.ones((3, self.alpha.shape[1]))).to(self.alpha.device)
        alpha = torch.exp(self.alpha[0] + eps[0]*(self.alpha[1].exp()))
        beta = torch.exp(self.beta[0] + eps[1]*(self.beta[1].exp()))
        gamma = torch.exp(self.gamma[0] + eps[2]*(self.gamma[1].exp()))
        return alpha, beta, gamma

    def forward_basis(self, t, z, *rates, condition=None, neg_slope=0.0):
        # Outputs a (n sample, n basis, n gene) tensor
        alpha_1, beta, gamma = rates[0], rates[1], rates[2]
        if condition is None:
            rho = torch.sigmoid(self.fc_out1(self.net_rho(z)))
        else:
            rho = torch.sigmoid(self.fc_out1(self.net_rho(torch.cat((z, condition), 1))))
        zero_mtx = torch.zeros(rho.shape, device=rho.device, dtype=float)
        zero_vec = torch.zeros(self.u0.shape, device=rho.device, dtype=float)
        alpha = torch.stack([alpha_1*rho, zero_mtx], 1)
        u0 = torch.stack([zero_vec, self.u0.exp()])
        s0 = torch.stack([zero_vec, self.s0.exp()])
        tau = torch.stack([F.leaky_relu(t - self.ton.exp(), neg_slope) for i in range(2)], 1)
        Uhat, Shat = pred_su(tau, u0, s0, alpha, beta, gamma)
        Uhat = Uhat * torch.exp(self.scaling)

        Uhat = F.relu(Uhat)
        Shat = F.relu(Shat)
        vu = alpha - self.beta.exp() * Uhat / torch.exp(self.scaling)
        vs = self.beta.exp() * Uhat / torch.exp(self.scaling) - self.gamma.exp() * Shat
        return Uhat, Shat, vu, vs

    def forward(self, t, z, u0=None, s0=None, t0=None, condition=None, neg_slope=0.0):
        alpha, beta, gamma = self.reparameterize_param()
        if u0 is None or s0 is None or t0 is None:
            return self.forward_basis(t, z, alpha, beta, gamma, condition, neg_slope)
        else:
            if condition is None:
                rho = torch.sigmoid(self.fc_out2(self.net_rho2(z)))
            else:
                rho = torch.sigmoid(self.fc_out2(self.net_rho2(torch.cat((z, condition), 1))))
            Uhat, Shat = pred_su(F.leaky_relu(t-t0, neg_slope), u0/self.scaling.exp(), s0, rho*alpha, beta, gamma)
            Uhat = Uhat * torch.exp(self.scaling)

            Uhat = F.relu(Uhat)
            Shat = F.relu(Shat)
            Vu = rho * alpha - beta * Uhat / torch.exp(self.scaling)
            Vs = beta * Uhat / torch.exp(self.scaling) - gamma * Shat
        return Uhat, Shat, Vu, Vs

    def eval_model(self, t, z, u0=None, s0=None, t0=None, condition=None):
        alpha = self.alpha[0].exp()
        beta = self.beta[0].exp()
        gamma = self.gamma[0].exp()
        if u0 is None or s0 is None or t0 is None:
            return self.forward_basis(t, z, alpha, beta, gamma, condition)
        else:
            if condition is None:
                rho = torch.sigmoid(self.fc_out2(self.net_rho2(z)))
            else:
                rho = torch.sigmoid(self.fc_out2(self.net_rho2(torch.cat((z, condition), 1))))

            Uhat, Shat = pred_su(F.relu(t-t0), u0/self.scaling.exp(), s0, rho*alpha, beta, gamma)
            Uhat = Uhat * torch.exp(self.scaling)

            Uhat = F.relu(Uhat)
            Shat = F.relu(Shat)
            Vu = rho*alpha - beta * (Uhat / torch.exp(self.scaling))
            Vs = beta * (Uhat / torch.exp(self.scaling)) - gamma * Shat
        return Uhat, Shat, Vu, Vs


class VAEFullVB(VAE):
    """Full Variational Bayes
    This has an extra sampling of ODE parameters. Other parts are the same.
    Hence, we set it to be a subclass of VAE.
    """
    def __init__(self,
                 adata,
                 tmax,
                 dim_z,
                 dim_cond=0,
                 device='cpu',
                 hidden_size=(500, 250, 250, 500),
                 init_method="steady",
                 init_key=None,
                 init_type=None,
                 tprior=None,
                 init_ton_zero=True,
                 time_distribution="gaussian",
                 std_z_prior=0.01,
                 rate_prior={
                     'alpha': (0.0, 1.0),
                     'beta': (0.0, 0.5),
                     'gamma': (0.0, 0.5)
                 },
                 checkpoints=[None, None],
                 **kwargs):
        """Constructor of the class

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
        std_z_prior : float, optional
            Standard deviation of the prior (isotropical Gaussian) of cell state.
        checkpoints : list of 2 strings, optional
            Contains the path to saved encoder and decoder models.
            Should be a .pt file.
        """
        t_start = time.time()
        self.timer = 0

        # Training Configuration
        self.config = {
            # Model Parameters
            "dim_z": dim_z,
            "tmax": tmax,
            "hidden_size": hidden_size,
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
            "n_refine": 2,
            "batch_size": 128,
            "learning_rate": None,
            "learning_rate_ode": None,
            "learning_rate_post": None,
            "lambda": 1e-3,
            "lambda_rho": 1e-3,
            "kl_t": 1.0,
            "kl_z": 1.0,
            "kl_w": 0.01,
            "kl_param": 1.0,
            "reg_v": 0.0,
            "reg_param": 1.0,
            "neg_slope": 0.0,
            "test_iter": None,
            "save_epoch": 100,
            "n_warmup": 5,
            "early_stop": 5,
            "early_stop_thred": adata.n_vars*1e-3,
            "train_test_split": 0.7,
            "k_alt": 1,
            "train_scaling": False,
            "train_std": False,
            "train_ton": (init_method != 'tprior'),
            "train_x0": False,
            "weight_sample": False,
            "vel_continuity_loss": False,

            "sparsify": 1
        }

        self.set_device(device)
        self.split_train_test(adata.n_obs)

        self.dim_z = dim_z
        self.enable_cvae = dim_cond > 0

        self.decoder = decoder_fullvb(adata,
                                      tmax,
                                      self.train_idx,
                                      dim_z,
                                      N1=hidden_size[2],
                                      N2=hidden_size[3],
                                      init_ton_zero=init_ton_zero,
                                      device=self.device,
                                      init_method=init_method,
                                      init_key=init_key,
                                      init_type=init_type,
                                      checkpoint=checkpoints[1],
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

        self.p_z = torch.stack([torch.zeros(adata.shape[0], dim_z),
                                torch.ones(adata.shape[0], dim_z)*self.config["std_z_prior"]]).float().to(self.device)
        self.alpha_w = torch.tensor([5.0, 5.0]).float().to(self.device)

        self.use_knn = False
        self.u0 = None
        self.s0 = None
        self.t0 = None

        # Class attributes for training
        self.loss_train, self.loss_test = [], []
        self.counter = 0  # Count the number of iterations
        self.n_drop = 0  # Count the number of consecutive iterations with little decrease in loss
        self.train_stage = 1

        # Prior of Decoder Parameters
        self.p_log_alpha = torch.tensor([[rate_prior['alpha'][0]], [rate_prior['alpha'][1]]]).to(self.device)
        self.p_log_beta = torch.tensor([[rate_prior['beta'][0]], [rate_prior['beta'][1]]]).to(self.device)
        self.p_log_gamma = torch.tensor([[rate_prior['gamma'][0]], [rate_prior['gamma'][1]]]).to(self.device)

        self.timer = time.time() - t_start

    def vae_risk(self,
                 q_tx, p_t,
                 q_zx, p_z,
                 u, s, uhat, shat,
                 sigma_u, sigma_s,
                 uhat_fw, shat_fw,
                 u1, s1,
                 weight=None):
        """Training objective function. This is the negative ELBO of the full VB.
        An additional KL term of decoder parameters is added.

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
        sigma_u, sigma_s : `torch.tensor`
            Standard deviation of the Gaussian noise
        weight : `torch.tensor`, optional
            Sample weight. This feature is not stable. Please consider setting it to None.

        Returns
        -------
        Negative ELBO : torch.tensor, scalar
        """
        kldt = self.kl_time(q_tx[0], q_tx[1], p_t[0], p_t[1], tail=self.config["tail"])
        kldz = kl_gaussian(q_zx[0], q_zx[1], p_z[0], p_z[1])
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
                                   self.p_log_gamma[1])) / u.shape[0]

        # u and sigma_u has the original scale

        if uhat.ndim == 3:
            logp = -0.5*((u.unsqueeze(1)-uhat)/sigma_u).pow(2)\
                - 0.5*((s.unsqueeze(1)-shat)/sigma_s).pow(2)\
                - torch.log(sigma_u)-torch.log(sigma_s*2*np.pi)
            pw = F.softmax(self.decoder.logit_pw, dim=1).T
            logp = torch.sum(pw*logp, 1)
            # kldw = entropy(self.decoder.logit_pw)
            kldw = elbo_collapsed_categorical(self.decoder.logit_pw, self.alpha_w, 2, u.shape[1])
        else:
            logp = -0.5*((u-uhat)/sigma_u).pow(2)\
                - 0.5*((s-shat)/sigma_s).pow(2)\
                - torch.log(sigma_u)-torch.log(sigma_s*2*np.pi)
            kldw = 0

        if uhat_fw is not None and shat_fw is not None:
            logp = logp - 0.5*((u1-uhat_fw)/sigma_u).pow(2)-0.5*((s1-shat_fw)/sigma_s).pow(2)

        if weight is not None:
            logp = logp*weight

        err_rec = torch.mean(torch.sum(logp, 1))

        return (- err_rec + self.config["kl_t"]*kldt
                + self.config["kl_z"]*kldz
                + self.config["kl_param"]*kld_param
                + self.config["kl_w"]*kldw)

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

        adata.var[f"{key}_logmu_alpha"] = self.decoder.alpha[0].detach().cpu().numpy()
        adata.var[f"{key}_logmu_beta"] = self.decoder.beta[0].detach().cpu().numpy()
        adata.var[f"{key}_logmu_gamma"] = self.decoder.gamma[0].detach().cpu().numpy()
        adata.var[f"{key}_logstd_alpha"] = self.decoder.alpha[1].detach().cpu().exp().numpy()
        adata.var[f"{key}_logstd_beta"] = self.decoder.beta[1].detach().cpu().exp().numpy()
        adata.var[f"{key}_logstd_gamma"] = self.decoder.gamma[1].detach().cpu().exp().numpy()
        adata.var[f"{key}_ton"] = self.decoder.ton.exp().detach().cpu().numpy()
        adata.var[f"{key}_scaling"] = np.exp(self.decoder.scaling.detach().cpu().numpy())
        adata.var[f"{key}_sigma_u"] = np.exp(self.decoder.sigma_u.detach().cpu().numpy())
        adata.var[f"{key}_sigma_s"] = np.exp(self.decoder.sigma_s.detach().cpu().numpy())
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
        sigma_u, sigma_s = adata.var[f"{key}_sigma_u"].to_numpy(), adata.var[f"{key}_sigma_s"].to_numpy()
        adata.var[f"{key}_likelihood"] = np.mean(-0.5*((adata.layers["Mu"]-Uhat)/sigma_u)**2
                                                 - 0.5*((adata.layers["Ms"]-Shat)/sigma_s)**2
                                                 - np.log(sigma_u) - np.log(sigma_s) - np.log(2*np.pi), 0)

        rho = np.zeros(U.shape)
        with torch.no_grad():
            B = min(U.shape[0]//10, 1000)
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

        rna_velocity_vae(adata, key, use_raw=False, use_scv_genes=False, full_vb=True)

        if file_name is not None:
            adata.write_h5ad(f"{file_path}/{file_name}")



class CycleEncoder(nn.Module):
    """Encoder of the Cycle VAE
    """
    def __init__(self, Cin, N1=500, N2=250, device=torch.device('cpu'), checkpoint=None):
        super(CycleEncoder, self).__init__()
        self.fc1 = nn.Linear(Cin, N1).to(device)
        self.bn1 = nn.BatchNorm1d(num_features=N1).to(device)
        self.dpt1 = nn.Dropout(p=0.2).to(device)
        self.fc2 = nn.Linear(N1, N2).to(device)
        self.bn2 = nn.BatchNorm1d(num_features=N2).to(device)
        self.dpt2 = nn.Dropout(p=0.2).to(device)

        self.net = nn.Sequential(self.fc1, self.bn1, nn.LeakyReLU(), self.dpt1,
                                 self.fc2, self.bn2, nn.LeakyReLU(), self.dpt2)

        self.fc_mu = nn.Linear(N2, 1).to(device)
        self.fc_std, self.spt = nn.Linear(N2, 1).to(device), nn.Softplus()

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

    def forward(self, data_in):
        z = self.net(data_in)
        mu_zx, std_zx = self.fc_mu(z), self.spt(self.fc_std(z))
        mu_zx = (torch.tanh(mu_zx)+1)*np.pi
        return mu_zx, std_zx


class CycleDecoder(nn.Module):
    def __init__(self,
                 adata,
                 tmax,
                 train_idx,
                 p=98,
                 filter_gene=False,
                 device=torch.device('cpu'),
                 init_method="steady",
                 init_key=None):
        super(CycleDecoder, self).__init__()
        U, S = adata.layers['Mu'][train_idx], adata.layers['Ms'][train_idx]
        X = np.concatenate((U, S), 1)
        N, G = U.shape
        self.tmax = tmax

        (alpha, beta, gamma,
         scaling,
         toff,
         u0, s0,
         sigma_u, sigma_s,
         T,
         gene_score) = init_params(X, p, fit_scaling=True)
        adata.var['velocity_genes'] = (gene_score == 1.0)
        if filter_gene:
            gene_mask = (gene_score == 1.0)
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
        # Dynamical Model Parameters
        if init_method == "random":
            print("Random Initialization.")
            self.alpha = nn.Parameter(torch.normal(0.0, 0.01, size=(G,), device=device).float())
            self.beta = nn.Parameter(torch.normal(0.0, 0.01, size=(G,), device=device).float())
            self.gamma = nn.Parameter(torch.normal(0.0, 0.01, size=(G,), device=device).float())
            self.theta_on = nn.Parameter(torch.rand(G, device=device).float())
            self.theta_off = nn.Parameter(torch.rand(G, device=device).float()+self.theta_on.detach())
            self.scaling = nn.Parameter(torch.tensor(np.log(scaling), device=device).float())
            self.sigma_u = nn.Parameter(torch.tensor(np.log(sigma_u), device=device).float())
            self.sigma_s = nn.Parameter(torch.tensor(np.log(sigma_s), device=device).float())
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
            self.scaling = nn.Parameter(torch.tensor(np.log(scaling), device=device).float())
            self.ton = nn.Parameter(torch.tensor(np.log(ton+1e-10), device=device).float())
            self.toff = nn.Parameter(torch.tensor(np.log(toff+1e-10), device=device).float())
            self.sigma_u = nn.Parameter(torch.tensor(np.log(sigma_u), device=device).float())
            self.sigma_s = nn.Parameter(torch.tensor(np.log(sigma_s), device=device).float())
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
            self.scaling = nn.Parameter(torch.tensor(np.log(scaling), device=device).float())
            self.ton = nn.Parameter(torch.tensor(np.log(ton+1e-10), device=device).float())
            self.toff = nn.Parameter(torch.tensor(np.log(toff+1e-10), device=device).float())
            self.sigma_u = nn.Parameter(torch.tensor(np.log(sigma_u), device=device).float())
            self.sigma_s = nn.Parameter(torch.tensor(np.log(sigma_s), device=device).float())

        self.scaling.requires_grad = False
        self.sigma_u.requires_grad = False
        self.sigma_s.requires_grad = False

    def angle2time(self, theta):
        return self.tmax/2*(1+torch.sign(theta-np.pi)*(1+torch.cos(theta))/2)

    def forward(self, theta, neg_slope=0.0):
        t = self.angle2time(theta)
        Uhat, Shat = ode(t,
                         torch.exp(self.alpha),
                         torch.exp(self.beta),
                         torch.exp(self.gamma),
                         self.ton.exp(),
                         self.toff.exp(),
                         neg_slope=neg_slope)
        Uhat = Uhat * torch.exp(self.scaling)
        return F.relu(Uhat), F.relu(Shat)

    def pred_su(self, theta, gidx=None):
        scaling = torch.exp(self.scaling)
        t = self.angle2time(theta)
        if gidx is not None:
            Uhat, Shat = ode(t,
                             torch.exp(self.alpha[gidx]),
                             torch.exp(self.beta[gidx]),
                             torch.exp(self.gamma[gidx]),
                             self.ton[gidx].exp(),
                             self.toff[gidx].exp(),
                             neg_slope=0.0)
            return F.relu(Uhat*scaling[gidx]), F.relu(Shat)
        Uhat, Shat = ode(t,
                         torch.exp(self.alpha),
                         torch.exp(self.beta),
                         torch.exp(self.gamma),
                         self.ton.exp(),
                         self.toff.exp(),
                         neg_slope=0.0)
        return F.relu(Uhat*scaling), F.relu(Shat), t

    def get_ode_param_list(self):
        return [self.alpha, self.beta, self.gamma, self.toff]


class CycleVAE(VanillaVAE):
    def __init__(self,
                 adata,
                 tmax,
                 device='cpu',
                 hidden_size=(500, 250),
                 filter_gene=False,
                 init_method="steady",
                 init_key=None,
                 tprior=None,
                 angle_distribution="gaussian",
                 checkpoints=None):
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
            "kl_theta": 1.0,
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
        self.decoder = CycleDecoder(adata,
                                    tmax,
                                    self.train_idx,
                                    device=self.device,
                                    filter_gene=filter_gene,
                                    init_method=init_method,
                                    init_key=init_key).float()
        G = adata.n_vars
        # Create an encoder
        try:
            self.encoder = CycleEncoder(2*G,
                                        hidden_size[0],
                                        hidden_size[1],
                                        self.device,
                                        checkpoint=checkpoints).float()
        except IndexError:
            print('Please provide two dimensions!')

        self.tmax = torch.tensor(tmax, device=self.device)
        self.angle_distribution = angle_distribution
        # Angle prior
        self.p_theta = torch.stack([torch.ones(adata.shape[0], 1, device=self.device)*(np.pi),
                                    torch.ones(adata.shape[0], 1, device=self.device)*(np.pi*2)]).float()
        if angle_distribution == 'uniform':
            self.kl_time = kl_uniform
        else:
            self.kl_time = kl_gaussian

        # class attributes for training
        self.loss_train, self.loss_test = [], []
        self.counter = 0  # Count the number of iterations
        self.n_drop = 0  # Count the number of consecutive epochs with negative/low ELBO gain

        self.timer = time.time() - t_start

    def forward(self, data_in):
        data_in_scale = torch.cat((data_in[:, :data_in.shape[1]//2]/torch.exp(self.decoder.scaling),
                                   data_in[:, data_in.shape[1]//2:]), 1)
        mu_theta, std_theta = self.encoder.forward(data_in_scale)
        theta = self._reparameterize(mu_theta, std_theta)

        uhat, shat = self.decoder.forward(theta, neg_slope=self.config["neg_slope"])  # uhat is scaled
        return mu_theta, std_theta, uhat, shat

    def eval_model(self, data_in):
        data_in_scale = torch.cat((data_in[:, :data_in.shape[1]//2]/torch.exp(self.decoder.scaling),
                                   data_in[:, data_in.shape[1]//2:]), 1)
        mu_theta, std_theta = self.encoder.forward(data_in_scale)
        uhat, shat, t = self.decoder.pred_su(mu_theta)  # uhat is scaled
        return mu_theta, std_theta, uhat, shat, t

    def _train_epoch(self, train_loader, test_set, optimizer, optimizer2=None, K=1):
        B = len(train_loader)
        self.set_mode('train')
        stop_training = False

        for i, batch in enumerate(train_loader):
            if self.counter == 1 or self.counter % self.config["test_iter"] == 0:
                elbo_test = self.test(test_set, None, self.counter)
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
            mu_thetax, std_thetax, uhat, shat = self.forward(xbatch)

            loss = self._vae_risk((mu_thetax, std_thetax),
                                  self.p_theta[:, self.train_idx[idx], :],
                                  u, s,
                                  uhat, shat,
                                  torch.exp(self.decoder.sigma_u), torch.exp(self.decoder.sigma_s),
                                  None,
                                  self.config["kl_theta"])

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

    def pred_all(self, data, mode='test', output=["uhat", "shat", "theta"], gene_idx=None):
        N, G = data.shape[0], data.shape[1]//2
        if "uhat" in output:
            Uhat = None if gene_idx is None else np.zeros((N, len(gene_idx)))
        if "shat" in output:
            Shat = None if gene_idx is None else np.zeros((N, len(gene_idx)))
        if "theta" in output:
            theta_out = np.zeros((N))
            std_theta_out = np.zeros((N))
            t_out = np.zeros((N))
        elbo = 0
        with torch.no_grad():
            B = min(N//10, 1000)
            Nb = N // B
            for i in range(Nb):
                data_in = torch.tensor(data[i*B:(i+1)*B]).float().to(self.device)
                mu_thetax, std_thetax, uhat, shat, t = self.eval_model(data_in)
                if mode == "test":
                    p_theta = self.p_theta[:, self.test_idx[i*B:(i+1)*B], :]
                elif mode == "train":
                    p_theta = self.p_theta[:, self.train_idx[i*B:(i+1)*B], :]
                else:
                    p_theta = self.p_theta[:, i*B:(i+1)*B, :]
                loss = self._vae_risk((mu_thetax, std_thetax),
                                      p_theta,
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
                if "theta" in output:
                    theta_out[i*B:(i+1)*B] = mu_thetax.detach().cpu().squeeze().numpy()
                    std_theta_out[i*B:(i+1)*B] = std_thetax.detach().cpu().squeeze().numpy()
                    t_out[i*B:(i+1)*B] = t.detach().cpu().squeeze().numpy()
            if N > B*Nb:
                data_in = torch.tensor(data[B*Nb:]).float().to(self.device)
                mu_thetax, std_thetax, uhat, shat, t = self.eval_model(data_in)
                if mode == "test":
                    p_theta = self.p_theta[:, self.test_idx[B*Nb:], :]
                elif mode == "train":
                    p_theta = self.p_theta[:, self.train_idx[B*Nb:], :]
                else:
                    p_theta = self.p_theta[:, B*Nb:, :]
                loss = self._vae_risk((mu_thetax, std_thetax),
                                      p_theta,
                                      data_in[:, :G], data_in[:, G:],
                                      uhat, shat,
                                      torch.exp(self.decoder.sigma_u), torch.exp(self.decoder.sigma_s),
                                      None,
                                      1.0)
                elbo = elbo-loss*(N-B*Nb)
                if "uhat" in output and gene_idx is not None:
                    Uhat[Nb*B:] = uhat[:, gene_idx].detach().cpu().numpy()
                if "shat" in output and gene_idx is not None:
                    Shat[Nb*B:] = shat[:, gene_idx].detach().cpu().numpy()
                if "theta" in output:
                    theta_out[Nb*B:] = mu_thetax.detach().cpu().squeeze().numpy()
                    std_theta_out[Nb*B:] = std_thetax.detach().cpu().squeeze().numpy()
                    t_out[Nb*B:] = t.detach().cpu().squeeze().numpy()
        out = []
        if "uhat" in output:
            out.append(Uhat)
        if "shat" in output:
            out.append(Shat)
        if "theta" in output:
            out.append(theta_out)
            out.append(std_theta_out)
            out.append(t_out)
        return out, elbo.detach().cpu().item()/N

    def test(self,
             test_set,
             Xembed,
             testid=0,
             test_mode=True,
             gind=None,
             gene_plot=None,
             plot=False,
             path='figures',
             **kwargs):
        """
        :noindex:
        Evaluate the model upon training/test dataset.

        Arguments
        ---------
        test_set : `torch.utils.data.Dataset`
            Training or validation dataset
        Xembed : `numpy array`
            Low-dimensional embedding for plotting
        testid : string or int, optional
            Used to name the figures
        gind : `numpy array`
            Index of genes in adata.var_names. Used for plotting.
        gene_plot : `numpy array`, optional
            Gene names.
        plot : bool, optional
            Whether to generate plots.
        path : str, optional
            Saving path.

        Returns
        -------
        elbo : float
        """
        self.set_mode('eval')
        data = test_set.data
        mode = "test" if test_mode else "train"
        out, elbo = self.pred_all(data, mode, gene_idx=gind)
        Uhat, Shat, t = out[0], out[1], out[2]

        G = data.shape[1]//2
        if plot:
            # Plot Time
            plot_time(t, Xembed, save=f"{path}/time-{testid}-vanilla.png")

            # Plot u/s-t and phase portrait for each gene
            for i in range(len(gind)):
                idx = gind[i]
                plot_sig(t.squeeze(),
                         data[:, idx], data[:, idx+G],
                         Uhat[:, i], Shat[:, i],
                         test_set.labels,
                         gene_plot[i],
                         save=f"{path}/sig-{gene_plot[i]}-{testid}-vanilla.png",
                         sparsify=self.config["sparsify"])

        return elbo

    def save_anndata(self, adata, key, file_path, file_name=None):
        """
        :noindex:
        Save the ODE parameters and cell time to the anndata object and write it to disk.

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

        out, elbo = self.pred_all(np.concatenate((adata.layers['Mu'],
                                                  adata.layers['Ms']), axis=1),
                                  mode="both",
                                  gene_idx=np.array(range(adata.n_vars)))
        Uhat, Shat, theta, std_theta, t = out[0], out[1], out[2], out[3], out[4]

        adata.obs[f"{key}_phase"] = theta
        adata.obs[f"{key}_std_phase"] = std_theta
        adata.obs[f"{key}_time"] = t
        adata.layers[f"{key}_uhat"] = Uhat
        adata.layers[f"{key}_shat"] = Shat

        adata.uns[f"{key}_train_idx"] = self.train_idx
        adata.uns[f"{key}_test_idx"] = self.test_idx
        adata.uns[f"{key}_run_time"] = self.timer
        adata.uns['tmax'] = self.tmax.detach().cpu().item()

        rna_velocity_vanillavae(adata, key)

        if file_name is not None:
            adata.write_h5ad(f"{file_path}/{file_name}")

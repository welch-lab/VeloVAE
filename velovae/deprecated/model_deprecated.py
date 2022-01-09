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
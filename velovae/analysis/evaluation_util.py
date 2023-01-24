import numpy as np
from scipy.stats import spearmanr, poisson
from scipy.special import loggamma
from sklearn.metrics.pairwise import pairwise_distances
from ..model.model_util import pred_su_numpy, ode_numpy, ode_br_numpy, scv_pred, scv_pred_single, optimal_transport_duality_gap

def get_mse(U,S,Uhat,Shat):
    return np.nanmean((U-Uhat)**2+(S-Shat)**2)

def get_mae(U,S,Uhat,Shat):
    return np.nanmean(np.abs(U-Uhat)+np.abs(S-Shat))
    
def time_corr(t1, t2):
    return spearmanr(t1,t2)

def poisson_log_likelihood(mu, obs):
    return -mu+obs*np.log(mu)-loggamma(obs)

def cell_state(adata, method, key, gene_indices=None, **kwargs):
    cell_state = None
    if(gene_indices is None):
        gene_indices = np.array(np.range(adata.n_vars))
    if(method=='scVelo'):
        t = adata.layers[f"{key}_t"][:,gene_indices]
        toff = adata.var[f"{key}_t_"].to_numpy()[gene_indices]
        cell_state = (t > toff)
    elif(method=='Vanilla VAE'):
        t = adata.obs[f"{key}_time"].to_numpy()
        toff = adata.var[f"{key}_toff"].to_numpy()
        ton = adata.var[f"{key}_ton"].to_numpy()
        toff = toff[gene_indices]
        ton = ton[gene_indices]
        cell_state = (t.reshape(-1,1) > toff) + (t.reshape(-1,1) < ton)*2
    elif(method in ['VeloVAE', 'FullVB', 'Discrete VeloVAE', 'Discrete FullVB']):
        rho = adata.layers[f"{key}_rho"][:,gene_indices]
        t = adata.obs[f"{key}_time"].to_numpy()
        mask_induction = rho > 0.01
        
        mask_repression = np.empty(mask_induction.shape, dtype=bool)
        mask_off = np.empty(mask_induction.shape, dtype=bool)
        for i in range(len(gene_indices)):
            ton = np.quantile(t[mask_induction[:,i]], 1e-3)
            mask_repression[:,i] = (rho[:,i] <= 0.1) & (t>=ton)
            mask_off[:,i] = (rho[:,i] <= 0.1) & (t<ton)
        
        cell_state = mask_repression * 1 + mask_off * 2
    elif(method == 'VeloVI'):
        t = adata.layers["fit_t"][:,gene_indices]
        toff = adata.var[f"{key}_t_"].to_numpy()[gene_indices]
        cell_state = (t > toff)
    else:
        cell_state = np.array([3 for i in range(adata.n_obs)])
    return cell_state

#scVelo
def get_err_scv(adata, key='fit'):
    Uhat, Shat = scv_pred(adata, key)
    mse = get_mse(adata.layers['Mu'], adata.layers['Ms'], Uhat, Shat)
    mae = get_mae(adata.layers['Mu'], adata.layers['Ms'], Uhat, Shat)
    logp = np.sum(np.log(adata.var[f"{key}_likelihood"]))
    try:
        run_time = adata.uns[f'{key}_run_time']
    except KeyError:
        run_time = np.nan
    return mse, None, mae, None, logp, None, run_time

def get_pred_scv_demo(adata, key='fit', genes=None, N=100):
    if(genes is None):
        genes = adata.var_names
    alpha, beta, gamma = adata.var[f"{key}_alpha"].to_numpy(),adata.var[f"{key}_beta"].to_numpy(),adata.var[f"{key}_gamma"].to_numpy()
    toff = adata.var[f"{key}_t_"].to_numpy()
    T = adata.layers[f"{key}_t"]
    scaling = adata.var[f"{key}_scaling"].to_numpy()
    Uhat, Shat = np.zeros((2*N,len(genes))), np.zeros((2*N,len(genes)))
    T_demo = np.zeros((2*N, len(genes)))
    for i, gene in enumerate(genes):
        idx = np.where(adata.var_names==gene)[0][0]
        t_demo = np.concatenate((np.linspace(0,toff[idx],N), np.linspace(toff[idx], max(T[:,idx].max(), toff[i]+T[:,idx].max()*0.01),N)))
        T_demo[:,i] = t_demo
        uhat, shat = scv_pred_single(t_demo,alpha[idx],beta[idx],gamma[idx],toff[idx],scaling=scaling[idx], uinit=0, sinit=0)
        Uhat[:,i] = uhat
        Shat[:,i] = shat
    return T_demo, Uhat, Shat



#Vanilla VAE
def get_pred_vanilla(adata, key):
    #Vanilla VAE
    alpha = adata.var[f"{key}_alpha"].to_numpy()
    beta = adata.var[f"{key}_beta"].to_numpy()
    gamma = adata.var[f"{key}_gamma"].to_numpy()
    toff = adata.var[f"{key}_toff"].to_numpy()
    ton = adata.var[f"{key}_ton"].to_numpy()
    t = adata.obs[f"{key}_time"].to_numpy()
    scaling = adata.var[f"{key}_scaling"].to_numpy()
    
    
    if( (f"{key}_uhat" not in adata.layers) or (f"{key}_shat" not in adata.layers)):
        Uhat, Shat = ode_numpy(t.reshape(-1,1), alpha, beta, gamma, ton, toff, scaling)
    else:
        Uhat, Shat = adata.layers[f"{key}_uhat"], adata.layers[f"{key}_shat"]
    
    return Uhat, Shat

def get_err_vanilla(adata, key, gene_mask=None):
    U, S = adata.layers["Mu"], adata.layers["Ms"]
    Uhat, Shat = get_pred_vanilla(adata, key)
    train_idx, test_idx = adata.uns[f"{key}_train_idx"], adata.uns[f"{key}_test_idx"]
    
    if(gene_mask is None):
        gene_mask = np.ones((adata.n_vars)).astype(bool)
    
    sigma_u, sigma_s = adata.var[f"{key}_sigma_u"].to_numpy()[gene_mask], adata.var[f"{key}_sigma_s"].to_numpy()[gene_mask]
    dist_u_train = np.abs(U[train_idx][:,gene_mask]-Uhat[train_idx][:,gene_mask])
    dist_s_train = np.abs(S[train_idx][:,gene_mask]-Shat[train_idx][:,gene_mask])
    dist_u_test = np.abs(U[test_idx][:,gene_mask]-Uhat[test_idx][:,gene_mask])
    dist_s_test = np.abs(S[test_idx][:,gene_mask]-Shat[test_idx][:,gene_mask])
    
    logp_train = -dist_u_train**2/(2*sigma_u**2) - dist_s_train**2/(2*sigma_s**2) - np.log(sigma_u) - np.log(sigma_s) - np.log(2*np.pi)
    logp_test = -dist_u_test**2/(2*sigma_u**2) - dist_s_test**2/(2*sigma_s**2) - np.log(sigma_u) - np.log(sigma_s) - np.log(2*np.pi)
    mse_train = np.nanmean(dist_u_train**2+dist_s_train**2)
    mse_test = np.nanmean(dist_u_test**2+dist_s_test**2)
    mae_train = np.nanmean(dist_u_train+dist_s_train)
    mae_test = np.nanmean(dist_u_test+dist_s_test)
    
    logp_train = np.nanmean(np.sum(logp_train,1))
    logp_test = np.nanmean(np.sum(logp_test,1))
    
    try:
        run_time = adata.uns[f'{key}_run_time']
    except KeyError:
        run_time = np.nan
    return mse_train, mse_test, mae_train, mae_test, logp_train, logp_test, run_time
    
    

def get_pred_vanilla_demo(adata, key, genes=None, N=100):
    alpha = adata.var[f"{key}_alpha"].to_numpy()
    beta = adata.var[f"{key}_beta"].to_numpy()
    gamma = adata.var[f"{key}_gamma"].to_numpy()
    toff = adata.var[f"{key}_toff"].to_numpy()
    ton = adata.var[f"{key}_ton"].to_numpy()
    scaling = adata.var[f"{key}_scaling"].to_numpy()
    t = adata.obs[f"{key}_time"].to_numpy() 
    
    t_demo = np.linspace(0, t.max(), N)
    if(genes is None):
        Uhat_demo, Shat_demo = ode_numpy(t_demo.reshape(-1,1), alpha, beta, gamma, ton, toff, scaling)
    else:
        gene_indices = np.array([np.where(adata.var_names==x)[0][0] for x in genes])
        Uhat_demo, Shat_demo = ode_numpy(t_demo.reshape(-1,1), alpha[gene_indices], beta[gene_indices], gamma[gene_indices], ton[gene_indices], toff[gene_indices], scaling[gene_indices])
    
    return t_demo, Uhat_demo, Shat_demo



#VeloVAE
def get_pred_velovae(adata, key, scv_key=None, full_vb=False, discrete=False):
    if( (f"{key}_uhat" not in adata.layers) or (f"{key}_shat" not in adata.layers)):
        rho = adata.layers[f"{key}_rho"]
        alpha = adata.var[f"{key}_alpha"].to_numpy() if not full_vb else np.exp(adata.var[f"{key}_logmu_alpha"].to_numpy())
        beta = adata.var[f"{key}_beta"].to_numpy() if not full_vb else np.exp(adata.var[f"{key}_logmu_beta"].to_numpy())
        gamma = adata.var[f"{key}_gamma"].to_numpy() if not full_vb else np.exp(adata.var[f"{key}_logmu_gamma"].to_numpy())
        t = adata.obs[f"{key}_time"].to_numpy()
        scaling = adata.var[f"{key}_scaling"].to_numpy()
        
        u0, s0 = adata.layers[f"{key}_u0"], adata.layers[f"{key}_s0"]
        t0 = adata.obs[f"{key}_t0"].to_numpy()
        
        Uhat, Shat = pred_su_numpy((t-t0).reshape(-1,1), u0, s0, rho*alpha, beta, gamma)
        Uhat = Uhat*scaling
    else:
        Uhat, Shat = adata.layers[f"{key}_uhat"], adata.layers[f"{key}_shat"]
    
    return Uhat, Shat


def get_err_velovae(adata, key, gene_mask=None, full_vb=False, discrete=False, n_sample=30, seed=2022):
    Uhat, Shat = get_pred_velovae(adata, key, full_vb, discrete)
    train_idx, test_idx = adata.uns[f"{key}_train_idx"], adata.uns[f"{key}_test_idx"]
    if(gene_mask is None):
        gene_mask = np.ones((adata.n_vars)).astype(bool)
    
    if(discrete):
        U, S = adata.layers["unspliced"].A, adata.layers["spliced"].A
        lu, ls = adata.obs["library_scale_u"].to_numpy(), adata.obs["library_scale_s"].to_numpy()
        Uhat = Uhat*(lu.reshape(-1,1))
        Shat = Shat*(ls.reshape(-1,1))
        Uhat = np.clip(Uhat, a_min=1e-2, a_max=None)
        Shat = np.clip(Shat, a_min=1e-2, a_max=None)
        
        logp_train = np.log(poisson.pmf(U[train_idx], Uhat[train_idx])+1e-10)+np.log(poisson.pmf(S[train_idx], Shat[train_idx])+1e-10)
        logp_test = np.log(poisson.pmf(U[test_idx], Uhat[test_idx])+1e-10)+np.log(poisson.pmf(S[test_idx], Shat[test_idx])+1e-10)
        
        #Sample multiple times
        mse_train, mae_train, mse_test, mae_test = 0,0,0,0
        np.random.seed(seed)
        
        for i in range(n_sample):
            U_sample = poisson.rvs(Uhat)
            S_sample = poisson.rvs(Shat)
            dist_u_train = np.abs(U[train_idx][:,gene_mask]-U_sample[train_idx][:,gene_mask])
            dist_s_train = np.abs(S[train_idx][:,gene_mask]-S_sample[train_idx][:,gene_mask])
            dist_u_test = np.abs(U[test_idx][:,gene_mask]-U_sample[test_idx][:,gene_mask])
            dist_s_test = np.abs(S[test_idx][:,gene_mask]-S_sample[test_idx][:,gene_mask])
            mse_train += np.nanmean(dist_u_train**2+dist_s_train**2)
            mse_test += np.nanmean(dist_u_test**2+dist_s_test**2)
            mae_train += np.nanmean(dist_u_train+dist_s_train)
            mae_test += np.nanmean(dist_u_test+dist_s_test)
        mse_train /= n_sample
        mse_test /= n_sample
        mae_train /= n_sample
        mae_test /= n_sample
    else:
        U, S = adata.layers["Mu"], adata.layers["Ms"]
        sigma_u, sigma_s = adata.var[f"{key}_sigma_u"].to_numpy()[gene_mask], adata.var[f"{key}_sigma_s"].to_numpy()[gene_mask]
    
        dist_u_train = np.abs(U[train_idx][:,gene_mask]-Uhat[train_idx][:,gene_mask])
        dist_s_train = np.abs(S[train_idx][:,gene_mask]-Shat[train_idx][:,gene_mask])
        dist_u_test = np.abs(U[test_idx][:,gene_mask]-Uhat[test_idx][:,gene_mask])
        dist_s_test = np.abs(S[test_idx][:,gene_mask]-Shat[test_idx][:,gene_mask])
        
        logp_train = -dist_u_train**2/(2*sigma_u**2) - dist_s_train**2/(2*sigma_s**2) - np.log(sigma_u) - np.log(sigma_s) - np.log(2*np.pi)
        logp_test = -dist_u_test**2/(2*sigma_u**2) - dist_s_test**2/(2*sigma_s**2) - np.log(sigma_u) - np.log(sigma_s) - np.log(2*np.pi)
    
        mse_train = np.nanmean(dist_u_train**2+dist_s_train**2)
        mse_test = np.nanmean(dist_u_test**2+dist_s_test**2)
        mae_train = np.nanmean(dist_u_train+dist_s_train)
        mae_test = np.nanmean(dist_u_test+dist_s_test)
    
    logp_train = np.nanmean(np.sum(logp_train,1))
    logp_test = np.nanmean(np.sum(logp_test,1))
    
    try:
        run_time = adata.uns[f'{key}_run_time']
    except KeyError:
        run_time = np.nan
    
    return mse_train, mse_test, mae_train, mae_test, logp_train, logp_test, run_time

def get_pred_velovae_demo(adata, key, genes=None, full_vb=False, discrete=False):
    if( (f"{key}_uhat" not in adata.layers) or (f"{key}_shat" not in adata.layers)):
        alpha = adata.var[f"{key}_alpha"].to_numpy() if not full_vb else np.exp(adata.var[f"{key}_logmu_alpha"].to_numpy())
        beta = adata.var[f"{key}_beta"].to_numpy() if not full_vb else np.exp(adata.var[f"{key}_logmu_beta"].to_numpy())
        gamma = adata.var[f"{key}_gamma"].to_numpy() if not full_vb else np.exp(adata.var[f"{key}_logmu_gamma"].to_numpy())
        t = adata.obs[f"{key}_time"].to_numpy()
        scaling = adata.var[f"{key}_scaling"].to_numpy()
        u0, s0 = adata.layers[f"{key}_u0"], adata.layers[f"{key}_s0"]
        t0 = adata.obs[f"{key}_t0"].to_numpy()
        if(genes is None):
            rho = adata.layers[f"{key}_rho"]
            Uhat, Shat = pred_su_numpy((t-t0).reshape(-1,1), u0, s0, alpha, beta, gamma)
            Uhat = Uhat*scaling
        else:
            gene_indices = np.array([np.where(adata.var_names==x)[0][0] for x in genes])
            rho = adata.layers[f"{key}_rho"][:,gene_indices]
            Uhat, Shat = pred_su_numpy((t-t0).reshape(-1,1), u0[:,gene_indices], s0[:,gene_indices], rho*alpha[gene_indices], beta[gene_indices], gamma[gene_indices])
            Uhat = Uhat*scaling[gene_indices]
    else:
        if(genes is None):
            Uhat, Shat = adata.layers[f"{key}_uhat"], adata.layers[f"{key}_shat"]
        else:
            gene_indices = np.array([np.where(adata.var_names==x)[0][0] for x in genes])
            Uhat, Shat = adata.layers[f"{key}_uhat"][:,gene_indices], adata.layers[f"{key}_shat"][:,gene_indices]
    if(discrete):
        lu = adata.obs["library_scale_u"].to_numpy().reshape(-1,1)
        ls = adata.obs["library_scale_s"].to_numpy().reshape(-1,1)
        Uhat = Uhat * lu
        Shat = Shat * ls
    return Uhat, Shat



#Branching ODE
def get_pred_brode(adata, key):
    if( (f"{key}_uhat" not in adata.layers) or (f"{key}_shat" not in adata.layers)):
        alpha = adata.varm[f"{key}_alpha"]
        beta = adata.varm[f"{key}_beta"]
        gamma = adata.varm[f"{key}_gamma"]
        u0, s0 = adata.varm[f"{key}_u0"], adata.varm[f"{key}_s0"]
        t_trans = adata.uns[f"{key}_t_trans"]
        #ts = adata.varm[f"{key}_ts"]
        scaling = adata.var[f"{key}_scaling"].to_numpy()
        par = np.argmax(adata.uns[f"{key}_w"], 1)
        
        t = adata.obs[f"{key}_time"].to_numpy()
        y = adata.obs[f"{key}_label"]
        
        Uhat, Shat = ode_br_numpy(t.reshape(-1,1),
                                  y,
                                  par,
                                  alpha=alpha,
                                  beta=beta,
                                  gamma=gamma,
                                  t_trans=t_trans,
                                  #ts=ts,
                                  scaling=scaling)
        Uhat = Uhat*scaling
    else:
        Uhat, Shat = adata.layers[f"{key}_uhat"], adata.layers[f"{key}_shat"]
    
    return Uhat, Shat

def get_err_brode(adata, key, gene_mask=None):
    U, S = adata.layers["Mu"], adata.layers["Ms"]
    Uhat, Shat = get_pred_brode(adata, key)
    train_idx, test_idx = adata.uns[f"{key}_train_idx"], adata.uns[f"{key}_test_idx"]
    
    if(gene_mask is None):
        gene_mask = np.ones((adata.n_vars)).astype(bool)
    sigma_u, sigma_s = adata.var[f"{key}_sigma_u"].to_numpy()[gene_mask], adata.var[f"{key}_sigma_s"].to_numpy()[gene_mask]
    dist_u_train = np.abs(U[train_idx][:,gene_mask]-Uhat[train_idx][:,gene_mask])
    dist_s_train = np.abs(S[train_idx][:,gene_mask]-Shat[train_idx][:,gene_mask])
    dist_u_test = np.abs(U[test_idx][:,gene_mask]-Uhat[test_idx][:,gene_mask])
    dist_s_test = np.abs(S[test_idx][:,gene_mask]-Shat[test_idx][:,gene_mask])
    
    logp_train = -dist_u_train**2/(2*sigma_u**2) - dist_s_train**2/(2*sigma_s**2) - np.log(sigma_u) - np.log(sigma_s) - np.log(2*np.pi)
    logp_test = -dist_u_test**2/(2*sigma_u**2) - dist_s_test**2/(2*sigma_s**2) - np.log(sigma_u) - np.log(sigma_s) - np.log(2*np.pi)
    mse_train = np.nanmean(dist_u_train**2+dist_s_train**2)
    mae_train = np.nanmean(dist_u_train+dist_s_train)
    mse_test = np.nanmean(dist_u_test**2+dist_s_test**2)
    mae_test = np.nanmean(dist_u_test+dist_s_test)
    
    logp_train = np.nanmean(np.sum(logp_train,1))
    logp_test = np.nanmean(np.sum(logp_test,1))
    
    try:
        run_time = adata.uns[f'{key}_run_time']
    except KeyError:
        run_time = np.nan
    
    return mse_train, mse_test, mae_train, mae_test, logp_train, logp_test, run_time

def get_pred_brode_demo(adata, key, genes=None, N=100):
    t_trans = adata.uns[f"{key}_t_trans"]
    t = adata.obs[f"{key}_time"].to_numpy()
    y = adata.obs[f"{key}_label"].to_numpy() #integer
    par = np.argmax(adata.uns[f"{key}_w"], 1)
    n_type = len(par)
    t_demo = np.zeros((N*n_type))
    y_demo = np.zeros((N*n_type)).astype(int)
    for i in range(n_type):
        y_demo[i*N:(i+1)*N] = i
        t_demo[i*N:(i+1)*N] = np.linspace(t_trans[i], np.quantile(t[y==i],0.95), N)
    if(genes is None):
        alpha = adata.varm[f"{key}_alpha"].T
        beta = adata.varm[f"{key}_beta"].T
        gamma = adata.varm[f"{key}_gamma"].T
        u0, s0 = adata.varm[f"{key}_u0"].T, adata.varm[f"{key}_s0"].T
        #ts = adata.varm[f"{key}_ts"].T
        scaling = adata.var[f"{key}_scaling"].to_numpy()
    else:
        gene_indices = np.array([np.where(adata.var_names==x)[0][0] for x in genes])
        alpha = adata.varm[f"{key}_alpha"][gene_indices].T
        beta = adata.varm[f"{key}_beta"][gene_indices].T
        gamma = adata.varm[f"{key}_gamma"][gene_indices].T
        u0, s0 = adata.varm[f"{key}_u0"][gene_indices].T, adata.varm[f"{key}_s0"][gene_indices].T
        #ts = adata.varm[f"{key}_ts"][gene_indices].T
        scaling = adata.var[f"{key}_scaling"][gene_indices].to_numpy()
    
    Uhat_demo, Shat_demo = ode_br_numpy(t_demo.reshape(-1,1),
                                        y_demo,
                                        par,
                                        alpha=alpha,
                                        beta=beta,
                                        gamma=gamma,
                                        t_trans=t_trans,
                                        #ts=ts,
                                        u0=u0,
                                        s0=s0,
                                        scaling=scaling)
    
    return t_demo, y_demo, Uhat_demo, Shat_demo

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



#UniTVelo
def get_pred_utv(adata, B=5000):
    t = adata.layers['fit_t']
    o = adata.var['fit_offset'].values
    a0 = adata.var['fit_a'].values
    t0 = adata.var['fit_t'].values
    h0 = adata.var['fit_h'].values
    i = adata.var['fit_intercept'].values
    gamma = adata.var['fit_gamma'].values
    beta = adata.var['fit_beta'].values
    scaling = adata.var['scaling'].values
    
    Nb = adata.n_obs//B
    uhat, shat = np.empty(adata.shape), np.empty(adata.shape)
    for i in range(Nb):
        shat[i*B:(i+1)*B] = h0*np.exp(-a0*(t[i*B:(i+1)*B] - t0)**2) + o
        uhat[i*B:(i+1)*B] = (adata.layers['velocity'][i*B:(i+1)*B]+gamma*shat[i*B:(i+1)*B])/beta + i
    if(Nb*B < adata.n_obs):
        shat[Nb*B:] = h0*np.exp(-a0*(t[Nb*B:] - t0)**2) + o
        uhat[Nb*B:] = (adata.layers['velocity'][Nb*B:]+gamma*shat[Nb*B:])/beta + i
    
    return uhat, shat


def get_err_utv(adata, key, gene_mask=None, B=5000):
    Uhat, Shat = get_pred_utv(adata, B)
    U, S = adata.layers['Mu'], adata.layers['Ms']
    
    if(gene_mask is None):
        gene_mask = np.where(~np.isnan(adata.layers['fit_t'][0]))[0]
    
    dist_u = np.abs(U[:,gene_mask]-Uhat[:,gene_mask])
    dist_s = np.abs(S[:,gene_mask]-Shat[:,gene_mask])
    
    #MLE of variance
    var_s = np.var((S[:,gene_mask]-Shat[:,gene_mask]),0)
    var_u = np.var((U[:,gene_mask]-Uhat[:,gene_mask]),0)
    
    mse = np.nanmean(dist_u**2+dist_s**2)
    mae = np.nanmean(dist_u+dist_s)
    logp = -dist_u**2/(2*var_u)-dist_s**2/(2*var_s) - 0.5*np.log(var_u) - 0.5*np.log(var_s) - np.log(2*np.pi)
    
    try:
        run_time = adata.uns[f'{key}_run_time']
    except KeyError:
        run_time = np.nan
    
    return mse, None, mae, None, logp.sum(1).mean(0), None, run_time

def get_pred_utv_demo(adata, genes=None, N=100):
    t = adata.layers['fit_t']

    if(genes is None):
        gene_indices = np.array(np.range(adata.n_vars))
    else:
        gene_indices = np.array([np.where(adata.var_names==x)[0][0] for x in genes])
    
    t_demo = np.linspace(t[:,gene_indices].min(0),t[:,gene_indices].max(0),N)
    o = adata.var['fit_offset'].values[gene_indices]
    a0 = adata.var['fit_a'].values[gene_indices]
    t0 = adata.var['fit_t'].values[gene_indices]
    h0 = adata.var['fit_h'].values[gene_indices]
    i = adata.var['fit_intercept'].values[gene_indices]
    gamma = adata.var['fit_gamma'].values[gene_indices]
    beta = adata.var['fit_beta'].values[gene_indices]
    scaling = adata.var['scaling'].values[gene_indices]
    
    shat = h0*np.exp(-a0*(t_demo - t0)**2) + o
    vs = shat * (-2*a0*(t_demo - t0))
    uhat = (vs + gamma*shat)/beta + i
    
    return t_demo, uhat, shat




##########################################################################
# Evaluation utility functions from DeepVelo
# Reference:
#Gao, M., Qiao, C. & Huang, Y. UniTVelo: temporally unified RNA velocity
#reinforces single-cell trajectory inference. Nat Commun 13, 6586 (2022). 
#https://doi.org/10.1038/s41467-022-34188-7
##########################################################################
import hnswlib
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
def get_neighbor_idx(
        adata,
        topC=30,
        topG=20,
    ):
        Ux_sz = adata.layers["Mu"]
        Sx_sz = adata.layers["Ms"]
        N_cell, N_gene = Sx_sz.shape

        n_pcas = 30
        pca_ = PCA(
            n_components=n_pcas,
            svd_solver="randomized",
        )
        Sx_sz_pca = pca_.fit_transform(Sx_sz)
        if N_cell < 3000:
            ori_dist = pairwise_distances(Sx_sz_pca, Sx_sz_pca)
            nn_t_idx = np.argsort(ori_dist, axis=1)[:, 1:topC]
        else:
            p = hnswlib.Index(space="l2", dim=n_pcas)
            p.init_index(max_elements=N_cell, ef_construction=200, M=30)
            p.add_items(Sx_sz_pca)
            p.set_ef(max(topC, topG) + 10)
            nn_t_idx = p.knn_query(Sx_sz_pca, k=topC)[0][:, 1:].astype(int)
       
        return nn_t_idx

def _loss_dv(
    output: torch.Tensor,
    current_state: torch.Tensor,
    idx: torch.LongTensor,
    candidate_states: torch.Tensor,
    n_spliced: int = None,
    l: int = 2,
    *args,
    **kwargs,
):
    batch_size, genes = current_state.shape
    if n_spliced is not None:
        genes = n_spliced
    num_neighbors = idx.shape[1]
    loss = []
    for i in range(num_neighbors):
        ith_neighbors = idx[:, i]
        candidate_state = candidate_states[ith_neighbors, :]  # (batch_size, genes)
        delta = (candidate_state - current_state).detach()  # (batch_size, genes)
        cos_sim = F.cosine_similarity(
            delta[:, :genes], output[:, :genes], dim=1
        )  # (batch_size,)

        # t+1 direction
        candidates = cos_sim.detach() > 0
        if(l==1):
            squared_difference = torch.mean(torch.abs(output - delta), dim=1)
        else:
            squared_difference = torch.mean(torch.pow(output - delta, 2), dim=1)
        squared_difference = squared_difference[candidates]
        loss.append(torch.sum(squared_difference) / len(candidates))

        # t-1 direction, (-output) - delta
        candidates = cos_sim.detach() < 0
        if(l==1):
            squared_difference = torch.mean(torch.pow(output + delta, 1), dim=1)
        else:
            squared_difference = torch.mean(torch.pow(output + delta, 2), dim=1)
        squared_difference = squared_difference[candidates]
        loss.append(torch.sum(squared_difference) / len(candidates))
        # TODO: check why the memory usage is high for this one
    loss = torch.stack(loss).mean()
    return loss.detach().cpu().item()

def get_err_dv(adata, key, gene_mask):
    if(gene_mask is None):
        gene_mask = np.array(range(adata.n_vars))
    nn_t_idx = get_neighbor_idx(adata)
    mse_u = _loss_dv(torch.tensor(adata.layers['velocity_unspliced'][:,gene_mask]), 
                                 torch.tensor(adata.layers['Mu'][:,gene_mask]), 
                                 torch.tensor(nn_t_idx, dtype=torch.long), 
                                 torch.tensor(adata.layers['Mu'][:,gene_mask]))
    mse_s = _loss_dv(torch.tensor(adata.layers['velocity'][:,gene_mask]), 
                                 torch.tensor(adata.layers['Ms'][:,gene_mask]), 
                                 torch.tensor(nn_t_idx, dtype=torch.long), 
                                 torch.tensor(adata.layers['Ms'][:,gene_mask]))
    mae_u = _loss_dv(torch.tensor(adata.layers['velocity_unspliced'][:,gene_mask]), 
                                 torch.tensor(adata.layers['Mu'][:,gene_mask]), 
                                 torch.tensor(nn_t_idx, dtype=torch.long), 
                                 torch.tensor(adata.layers['Mu'][:,gene_mask]),
                                 l=1)
    mae_s = _loss_dv(torch.tensor(adata.layers['velocity'][:,gene_mask]), 
                                 torch.tensor(adata.layers['Ms'][:,gene_mask]), 
                                 torch.tensor(nn_t_idx, dtype=torch.long), 
                                 torch.tensor(adata.layers['Ms'][:,gene_mask]),
                                 l=1)
    try:
        run_time = adata.uns[f'{key}_run_time']
    except KeyError:
        run_time = np.nan
    return mse_u+mse_s, None, mae_u+mae_s, None, None, None, run_time





##########################################################################
# Evaluation utility functions from PyroVelocity
# Reference:
#Qin, Q., Bingham, E., La Manno, G., Langenau, D. M., & Pinello, L. (2022).
#Pyro-Velocity: Probabilistic RNA Velocity inference from single-cell data. 
#bioRxiv.
##########################################################################
def get_err_pv(adata, key, gene_mask, discrete=True):
    train_idx, test_idx = adata.uns[f"{key}_train_idx"], adata.uns[f"{key}_test_idx"]
    if(discrete):
        U, S = adata.layers['unspliced'].A, adata.layers['spliced'].A
        Mu_u, Mu_s = adata.layers[f'{key}_ut'], adata.layers[f'{key}_st']
        Uhat, Shat = adata.layers[f'{key}_u'], adata.layers[f'{key}_s']
    else:
        U, S = adata.layers['Mu'], adata.layers['Ms']
        Uhat, Shat = adata.layers[f'{key}_u'], adata.layers[f'{key}_s']
        #MLE of variance
        var_s_train, var_s_test = np.var((S[train_idx]-Shat[train_idx]),0),np.var((S[test_idx]-Shat[test_idx]),0)
        var_u_train, var_u_test = np.var((U[train_idx]-Uhat[train_idx]),0),np.var((U[test_idx]-Uhat[test_idx]),0)
    
    if(gene_mask is None):
        gene_mask = np.ones((adata.n_vars)).astype(bool)
    
    
    dist_u_train = np.abs(U[train_idx][:,gene_mask]-Uhat[train_idx][:,gene_mask])
    dist_s_train = np.abs(S[train_idx][:,gene_mask]-Shat[train_idx][:,gene_mask])
    dist_u_test = np.abs(U[test_idx][:,gene_mask]-Uhat[test_idx][:,gene_mask])
    dist_s_test = np.abs(S[test_idx][:,gene_mask]-Shat[test_idx][:,gene_mask])
    
    if(discrete):
        logp_train = np.log(poisson.pmf(U[train_idx], Mu_u[train_idx])+1e-10).sum(1).mean()\
                    +np.log(poisson.pmf(S[train_idx], Mu_s[train_idx])+1e-10).sum(1).mean()
        logp_test = np.log(poisson.pmf(U[test_idx], Mu_u[test_idx])+1e-10).sum(1).mean()\
                    +np.log(poisson.pmf(S[test_idx], Mu_s[test_idx])+1e-10).sum(1).mean()
    else:
        logp_train = -dist_u_train**2/(2*var_u_train)-dist_s_train**2/(2*var_s_train) - 0.5*np.log(var_u_train) - 0.5*np.log(var_s_train) - np.log(2*np.pi)
        logp_test = -dist_u_test**2/(2*var_u_test)-dist_s_test**2/(2*var_s_test) - 0.5*np.log(var_u_test) - 0.5*np.log(var_s_test) - np.log(2*np.pi)
    
    mse_train = np.nanmean(dist_u_train**2+dist_s_train**2)
    mse_test = np.nanmean(dist_u_test**2+dist_s_test**2)
    mae_train = np.nanmean(dist_u_train+dist_s_train)
    mae_test = np.nanmean(dist_u_test+dist_s_test)
    
    try:
        run_time = adata.uns[f'{key}_run_time']
    except KeyError:
        run_time = np.nan
    
    return mse_train, mse_test, mae_train, mae_test, logp_train, logp_test, run_time
    


##########################################################################
# Evaluation utility functions from VeloVI
# Reference:
#Gayoso, Adam, et al. "Deep generative modeling of transcriptional dynamics 
#for RNA velocity analysis in single cells." bioRxiv (2022).
##########################################################################
def get_err_velovi(adata, key, gene_mask):
    if(gene_mask is None):
        gene_mask = np.ones((adata.n_vars)).astype(bool)
    U, S = adata.layers['Mu'][:,gene_mask], adata.layers['Ms'][:,gene_mask]
    Uhat, Shat = adata.layers[f'{key}_uhat'][:,gene_mask], adata.layers[f'{key}_shat'][:,gene_mask]
    
    try:
        run_time = adata.uns[f'{key}_run_time']
    except KeyError:
        run_time = np.nan
    
    try:
        train_idx = adata.uns[f'{key}_train_idx']
        test_idx = adata.uns[f'{key}_test_idx']
        dist_u_train = np.abs(U[train_idx]-Uhat[train_idx])
        dist_s_train = np.abs(S[train_idx]-Shat[train_idx])
        dist_u_test = np.abs(U[test_idx]-Uhat[test_idx])
        dist_s_test = np.abs(S[test_idx]-Shat[test_idx])
        
        mse_train = np.mean(dist_u_train**2+dist_s_train**2)
        mse_test = np.mean(dist_u_test**2+dist_s_test**2)
        mae_train = np.mean(dist_u_train+dist_s_train)
        mae_test = np.mean(dist_u_test+dist_s_test)
        logp_train = np.mean(adata.obs[f'{key}_likelihood'].to_numpy()[train_idx])
        logp_test = np.mean(adata.obs[f'{key}_likelihood'].to_numpy()[test_idx])
    except KeyError:
        dist_u = np.abs(U-Uhat)
        dist_s = np.abs(S-Shat)
        mse = np.mean(dist_u**2+dist_s**2)
        mae = np.mean(dist_u+dist_s)
        logp = np.mean(adata.obs[f'{key}_likelihood'].to_numpy())
    
        return mse, None, mae, None, logp, None, run_time
    
    return mse_train, mse_test, mae_train, mae_test, logp_train, logp_test, run_time









##########################################################################
# Evaluation Metrics from UniTVelo
# Reference:
#Gao, M., Qiao, C. & Huang, Y. UniTVelo: temporally unified RNA velocity
#reinforces single-cell trajectory inference. Nat Commun 13, 6586 (2022). 
#https://doi.org/10.1038/s41467-022-34188-7
##########################################################################

from sklearn.metrics.pairwise import cosine_similarity

def summary_scores(all_scores):
    #Summarize group scores.
    #Args:
    #    all_scores (dict{str,list}): 
    #        {group name: score list of individual cells}.
    #
    #Returns:
    #    dict{str,float}: 
    #        Group-wise aggregation scores.
    #    float: 
    #        score aggregated on all samples
        
    sep_scores = {k:np.mean(s) for k, s in all_scores.items() if s}
    overal_agg = np.mean([s for k, s in sep_scores.items() if s])
    return sep_scores, overal_agg

def keep_type(adata, nodes, target, k_cluster):
    #Select cells of targeted type
    #Args:
    #    adata (Anndata): 
    #        Anndata object.
    #    nodes (list): 
    #        Indexes for cells
    #    target (str): 
    #        Cluster name.
    #    k_cluster (str): 
    #        Cluster key in adata.obs dataframe
    #Returns:
    #    list: 
    #        Selected cells.
    
    return nodes[adata.obs[k_cluster][nodes].values == target]

def cross_boundary_correctness(
    adata, 
    k_cluster, 
    k_velocity, 
    cluster_edges, 
    return_raw=False, 
    x_emb="X_umap",
    gene_mask=None
):
    #Cross-Boundary Direction Correctness Score (A->B)
    #Args:
    #    adata (Anndata): 
    #        Anndata object.
    #    k_cluster (str): 
    #        key to the cluster column in adata.obs DataFrame.
    #    k_velocity (str): 
    #        key to the velocity matrix in adata.obsm.
    #    cluster_edges (list of tuples("A", "B")): 
    #        pairs of clusters has transition direction A->B
    #    return_raw (bool): 
    #        return aggregated or raw scores.
    #    x_emb (str): 
    #        key to x embedding for visualization.
    #Returns:
    #    dict: 
    #        all_scores indexed by cluster_edges or mean scores indexed by cluster_edges
    #    float: 
    #        averaged score over all cells.
    
    scores = {}
    all_scores = {}
    x_emb_name = x_emb
    if(x_emb in adata.obsm):
        x_emb = adata.obsm[x_emb]
        if x_emb_name == "X_umap":
            v_emb = adata.obsm['{}_umap'.format(k_velocity)]
        else:
            v_emb = adata.obsm[[key for key in adata.obsm if key.startswith(k_velocity)][0]]
    else:
        x_emb = adata.layers[x_emb]
        v_emb = adata.layers[k_velocity]
        if(gene_mask is None):
            gene_mask = ~np.isnan(v_emb[0])
        x_emb = x_emb[:,gene_mask]
        v_emb = v_emb[:,gene_mask]
        
    for u, v in cluster_edges:
        sel = adata.obs[k_cluster] == u
        nbs = adata.uns['neighbors']['indices'][sel] # [n * 30]
        
        boundary_nodes = map(lambda nodes:keep_type(adata, nodes, v, k_cluster), nbs)
        x_points = x_emb[sel]
        x_velocities = v_emb[sel]
        
        type_score = []
        for x_pos, x_vel, nodes in zip(x_points, x_velocities, boundary_nodes):
            if len(nodes) == 0: continue

            position_dif = x_emb[nodes] - x_pos
            dir_scores = cosine_similarity(position_dif, x_vel.reshape(1,-1)).flatten()
            type_score.append(np.nanmean(dir_scores))
        if(len(type_score)==0):
            print(f'Warning: cell type transition pair ({u},{v}) does not exist in the KNN graph. Ignored.')
        else:
            scores[(u, v)] = np.nanmean(type_score)
            all_scores[(u, v)] = type_score
        
    if return_raw:
        return all_scores 
    
    return scores, np.mean([sc for sc in scores.values()])

def inner_cluster_coh(adata, k_cluster, k_velocity, return_raw=False):
    #In-cluster Coherence Score.
    #
    #Args:
    #    adata (Anndata): 
    #        Anndata object.
    #    k_cluster (str): 
    #        key to the cluster column in adata.obs DataFrame.
    #    k_velocity (str): 
    #        key to the velocity matrix in adata.obsm.
    #    return_raw (bool): 
    #        return aggregated or raw scores.
    #    
    #Returns:
    #    dict: 
    #        all_scores indexed by cluster_edges mean scores indexed by cluster_edges
    #    float: 
    #        averaged score over all cells.
    #    
    clusters = np.unique(adata.obs[k_cluster])
    scores = {}
    all_scores = {}

    for cat in clusters:
        sel = adata.obs[k_cluster] == cat
        nbs = adata.uns['neighbors']['indices'][sel]
        same_cat_nodes = map(lambda nodes:keep_type(adata, nodes, cat, k_cluster), nbs)

        velocities = adata.layers[k_velocity]
        nan_mask = ~np.isnan(velocities[0])
        velocities = velocities[:,nan_mask]
        
        cat_vels = velocities[sel]
        cat_score = [cosine_similarity(cat_vels[[ith]], velocities[nodes]).mean() 
                     for ith, nodes in enumerate(same_cat_nodes) 
                     if len(nodes) > 0]
        all_scores[cat] = cat_score
        scores[cat] = np.mean(cat_score)
    
    if return_raw:
        return all_scores
    
    return scores, np.mean([sc for sc in scores.values()])

##########################################################################
# End of Reference
##########################################################################

# New performance metric based on the idea of CBDir
def branch_fit_score(adata, k_cluster, k_velocity, branches, x_emb="X_umap"):
    cell_labels = adata.obs[k_cluster].to_numpy()
    x_emb_name = x_emb
    if(x_emb in adata.obsm):
        x_emb = adata.obsm[x_emb]
        if x_emb_name == "X_umap":
            v_emb = adata.obsm['{}_umap'.format(k_velocity)]
        else:
            v_emb = adata.obsm[[key for key in adata.obsm if key.startswith(k_velocity)][0]]
    else:
        x_emb = adata.layers[x_emb]
        v_emb = adata.layers[k_velocity]
    
    all_boundary_nodes = {}
    for u in branches:
        sel = adata.obs[k_cluster] == u
        nbs = adata.uns['neighbors']['indices'][sel] # [n * 30]
        for v in branches[u]:
            boundary_nodes = map(lambda nodes:keep_type(adata, nodes, v, k_cluster), nbs) #cells of type v that are neighbors of type u
            indices = np.concatenate([x for x in boundary_nodes]).astype(int)
            all_boundary_nodes[v] = indices
    
    for u in branches:
        sel = adata.obs[k_cluster] == u
        nbs = adata.uns['neighbors']['indices'][sel] # [n * 30]
        branch_score = []
        for v in branches[u]:       
            boundary_nodes = map(lambda nodes:keep_type(adata, nodes, v, k_cluster), nbs)
            x_points = x_emb[sel]
            x_velocities = v_emb[sel]
            
            type_score = []
            weights = []
            for x_pos, x_vel, nodes in zip(x_points, x_velocities, boundary_nodes):
                if len(nodes) == 0: continue
                #check if the cell has the most transition to type v
                weights.append(np.sum(cell_labels[nodes]==v))
                
                # cosine similarity with desired neighbor
                position_dif = x_emb[nodes] - x_pos
                dir_scores = cosine_similarity(position_dif, x_vel.reshape(1,-1)).flatten()
                theta_1 = np.nanmean(np.arccos(dir_scores))/np.pi
                
                # cosine similarity with other branches
                dir_scores = np.array([])
                for w in branches[u]:
                    if(w==v):
                        continue
                    position_dif = x_emb[all_boundary_nodes[w]] - x_pos
                    dir_scores = np.concatenate((dir_scores, cosine_similarity(position_dif, x_vel.reshape(1,-1)).flatten()))
                theta_2 = np.nanmean(np.arccos(dir_scores))/np.pi
                type_score.append(theta_2-theta_1)
            branch_score.append(np.sum(np.array(type_score)*np.array(weights))/np.sum(weights))
    return branch_score, np.mean(branch_score)
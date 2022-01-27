import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import  matplotlib.colors as clr
import umap
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from loess import loess_1d

#######################################################################################
#Default colors and markers for plotting
#######################################################################################
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'lime', 'grey', \
   'olive', 'cyan', 'pink', 'gold', 'steelblue', 'salmon', 'teal', \
   'magenta', 'rosybrown', 'darkorange', 'yellow', 'greenyellow', 'darkseagreen', 'yellowgreen', 'palegreen', \
   'hotpink', 'navajowhite', 'aqua', 'navy', 'saddlebrown', 'black', 'maroon']

colors_state=['r','b','k']

colormaps = [clr.LinearSegmentedColormap.from_list('my '+colors[i], ['white',colors[i]], 1024) for i in range(len(colors))]

markers = ["o","x","s","v","+","d","1","*","^","p","h","8","1","2","|"]



def pickcell(u,s,cell_labels):
    """
    Picks some cells from each cell type. Used in plotting the phase trajectory.
    """
    cell_type = np.unique(cell_labels)
    quantiles = [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
    track_idx = []
    for i in range(len(cell_type)):
        for j in range(len(quantiles)-1):
            thred1 = np.quantile(s, quantiles[j])
            thred2 = np.quantile(s, quantiles[j+1])
            ans = np.where((s>=thred1)&(s<thred2)&(cell_labels==cell_type[i]))[0]
            if(len(ans)>0):
                track_idx.append(ans[0])
            thred1 = np.quantile(u, quantiles[j])
            thred2 = np.quantile(u, quantiles[j+1])
            ans = np.where((s>=thred1)&(s<thred2)&(cell_labels==cell_type[i]))[0]
            if(len(ans)>0):
                track_idx.append(ans[0])
    return np.array(track_idx)

def plotSig_(t, 
            u, s, 
            cell_labels,
            tpred=None,
            upred=None, spred=None, 
            type_specific=False,
            title='gene', 
            savefig=False, 
            path=None, 
            figname="gene", 
            **kwargs):
    
    fig, ax = plt.subplots(2,1,figsize=(15,12))
    D = kwargs['sparsify'] if('sparsify' in kwargs) else 1
    cell_types = np.unique(cell_labels)

    for i, type_ in enumerate(cell_types):
        mask_type = cell_labels==type_
        ax[0].plot(t[mask_type][::D], u[mask_type][::D],'.',color=colors[i], alpha=0.25, label=type_)
        ax[1].plot(t[mask_type][::D], s[mask_type][::D],'.',color=colors[i], alpha=0.25, label=type_)
    
    if((tpred is not None) and (upred is not None) and (spred is not None)):
        if(type_specific):
            for i, type_ in enumerate(cell_types):
                mask_type = cell_labels==type_
                order = np.argsort(tpred[mask_type])
                ax[0].plot(tpred[mask_type][order], upred[mask_type][order], '-', color=colors[i], label=type_, linewidth=1.5)
                ax[1].plot(tpred[mask_type][order], spred[mask_type][order], '-', color=colors[i], label=type_, linewidth=1.5)
        else:
            order = np.argsort(tpred)
            ax[0].plot(tpred[order], upred[order], 'k-', linewidth=1.5)
            ax[1].plot(tpred[order], spred[order], 'k-', linewidth=1.5)
    
    if('ts' in kwargs and 't_trans' in kwargs):
        ts = kwargs['ts']
        t_trans = kwargs['t_trans']
        for i, type_ in enumerate(cell_types):
            ax[0].plot([t_trans[i],t_trans[i]], [0, u.max()], '-x', color=colors[i])
            ax[0].plot([ts[i],ts[i]], [0, u.max()], '--x', color=colors[i])
            ax[1].plot([t_trans[i],t_trans[i]], [0, s.max()], '-x', color=colors[i])
            ax[1].plot([ts[i],ts[i]], [0, s.max()], '--x', color=colors[i])
    
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("U", fontsize=18)
    
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("S", fontsize=18)
    handles, labels = ax[1].get_legend_handles_labels()
    
    ax[0].set_title('Unspliced, VAE')
    ax[1].set_title('Spliced, VAE')
    
    lgd=fig.legend(handles, labels, fontsize=15, markerscale=5, bbox_to_anchor=(1.0,1.0), loc='upper left')
    fig.suptitle(title)
    if(savefig):
        try:
            if(path is None):
                fig.savefig(f"figures/sig_{figname}.png",bbox_extra_artists=(lgd,), bbox_inches='tight')
            else:
                fig.savefig(f"{path}/sig_{figname}.png",bbox_extra_artists=(lgd,), bbox_inches='tight')
        except FileNotFoundError:
            print("Saving failed. File path doesn't exist!")
    plt.close(fig)
    return

def plotSig(t, 
            u, s, 
            upred, spred, 
            title, 
            savefig=False, 
            path=None, 
            figname="gene", 
            cell_labels=None,
            **kwargs):
    """
    Plots u/s vs. time (Single Plot)
    """
    D = kwargs['sparsify'] if('sparsify' in kwargs) else 1
    tscv = kwargs['tscv'] if 'tscv' in kwargs else t
    tdemo = kwargs["tdemo"] if "tdemo" in kwargs else t
    if('cell_labels' is None):
        fig, ax = plt.subplots(2,1,figsize=(15,12))
        #order = np.argsort(t)
        ax[0].plot(t[::D], u[::D],'b.',label="raw")
        ax[1].plot(t[::D], s[::D],'b.',label="raw")
        ax[0].plot(tdemo, upred, '.', color='lawngreen', label='Prediction', linewidth=2.0)
        ax[1].plot(tdemo, spred, '.', color='lawngreen', label="Prediction", linewidth=2.0)

        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("U", fontsize=18)
        
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("S", fontsize=18)
        
        #fig.subplots_adjust(right=0.7)
        handles, labels = ax[1].get_legend_handles_labels()
    else:
        fig, ax = plt.subplots(2,2,figsize=(24,12))
        labels_pred = kwargs['labels_pred'] if 'labels_pred' in kwargs else []
        labels_demo = kwargs['labels_demo'] if 'labels_demo' in kwargs else labels_pred
        cell_types = np.unique(cell_labels)
        
        for i, type_ in enumerate(cell_types):
            mask_type = cell_labels==type_
            ax[0,0].plot(tscv[mask_type][::D], u[mask_type][::D],'.',color=colors[i], alpha=0.25, label=type_)
            ax[0,1].plot(tscv[mask_type][::D], s[mask_type][::D],'.',color=colors[i], alpha=0.25, label=type_)
            if(len(labels_pred) > 0):
                mask_mytype = labels_pred==type_
                ax[1,0].plot(t[mask_mytype][::D], u[mask_mytype][::D],'.',color=colors[i], alpha=0.25, label=type_)
                ax[1,1].plot(t[mask_mytype][::D], s[mask_mytype][::D],'.',color=colors[i], alpha=0.25, label=type_)
            else:
                ax[1,0].plot(t[mask_type][::D], u[mask_type][::D],'.',color=colors[i], alpha=0.25, label=type_)
                ax[1,1].plot(t[mask_type][::D], s[mask_type][::D],'.',color=colors[i], alpha=0.25, label=type_)

        
        if(len(labels_pred) > 0):
            for i, type_ in enumerate(cell_types):
                mask_mytype = labels_demo==type_
                order = np.argsort(tdemo[mask_mytype])
                ax[1,0].plot(tdemo[mask_mytype][order], upred[mask_mytype][order], '.', color=colors[i], label=type_+" ode", linewidth=1.5)
                ax[1,1].plot(tdemo[mask_mytype][order], spred[mask_mytype][order], '.', color=colors[i], label=type_+" ode", linewidth=1.5)
        else:
            order = np.argsort(tdemo)
            ax[1,0].plot(tdemo[order], upred[order], 'k.', linewidth=1.5)
            ax[1,1].plot(tdemo[order], spred[order], 'k.', linewidth=1.5)

        if('ts' in kwargs and 't_trans' in kwargs):
            ts = kwargs['ts']
            t_trans = kwargs['t_trans']
            for i, type_ in enumerate(cell_types):
                for j in range(2):
                    ax[j,0].plot([t_trans[i],t_trans[i]], [0, u.max()], '-x', color=colors[i])
                    ax[j,0].plot([ts[i],ts[i]], [0, u.max()], '--x', color=colors[i])
                    #ax[j,0].plot([ts[i,1],ts[i,1]], [0, u.max()], '--x', color=colors[i])
                    ax[j,1].plot([t_trans[i],t_trans[i]], [0, s.max()], '-x', color=colors[i])
                    ax[j,1].plot([ts[i],ts[i]], [0, s.max()], '--x', color=colors[i])
                    #ax[j,1].plot([ts[i,1],ts[i,1]], [0, s.max()], '--x', color=colors[i])
        for j in range(2): 
            ax[j,0].set_xlabel("Time")
            ax[j,0].set_ylabel("U", fontsize=18)
            
            ax[j,1].set_xlabel("Time")
            ax[j,1].set_ylabel("S", fontsize=18)
            handles, labels = ax[1,0].get_legend_handles_labels()
           
        
        if('subtitles' in kwargs):
            ax[0,0].set_title(f'Unspliced, {subtitle[0]}')
            ax[0,1].set_title(f'Spliced, {subtitle[0]}')
            ax[1,0].set_title(f'Unspliced, {subtitle[1]}')
            ax[1,1].set_title(f'Spliced, {subtitle[1]}')
        else:
            ax[0,0].set_title('Unspliced, True Label')
            ax[0,1].set_title('Spliced, True Label')
            ax[1,0].set_title('Unspliced, VAE')
            ax[1,1].set_title('Spliced, VAE')
    
    lgd=fig.legend(handles, labels, fontsize=15, markerscale=5, bbox_to_anchor=(1.0,1.0), loc='upper left')
    fig.suptitle(title)
    if(savefig):
        try:
            if(path is None):
                fig.savefig(f"figures/sig_{figname}.png",bbox_extra_artists=(lgd,), bbox_inches='tight')
            else:
                fig.savefig(f"{path}/sig_{figname}.png",bbox_extra_artists=(lgd,), bbox_inches='tight')
        except FileNotFoundError:
            print("Saving failed. File path doesn't exist!")
    plt.close(fig)
    


def plotPhase(u, s, 
              upred, spred, 
              title, 
              track_idx=None, 
              labels=None, # array/list of integer
              types=None,  # array/list of string
              savefig=False, 
              path='figures',  
              figname="gene"):
    fig, ax = plt.subplots(figsize=(6,6))
    if(labels is None or types is None):
        ax.scatter(s,u,c="b",alpha=0.5)
    else:
        for i, type_ in enumerate(types):
            ax.scatter(s[labels==i],u[labels==i],c=colors[i],alpha=0.3,label=type_)
    ax.plot(spred,upred,'k.',label="ode")
    #Plot the correspondence
    if(track_idx is None):
        rng = np.random.default_rng()
        perm = rng.permutation(len(s))
        Nsample = 50
        s_comb = np.stack([s[perm[:Nsample]],spred[perm[:Nsample]]]).ravel('F')
        u_comb = np.stack([u[perm[:Nsample]],upred[perm[:Nsample]]]).ravel('F')
    else:
        s_comb = np.stack([s[track_idx],spred[track_idx]]).ravel('F')
        u_comb = np.stack([u[track_idx],upred[track_idx]]).ravel('F')
        
    for i in range(0, len(s_comb), 2):
        ax.plot(s_comb[i:i+2], u_comb[i:i+2], 'k-', linewidth=0.8)
    ax.set_xlabel("S", fontsize=18)
    ax.set_ylabel("U", fontsize=18)
    ax.set_title(title)
    ax.legend()
    #plt.show()
    
    if(savefig):
        try:
            fig.savefig(f"{path}/phase_{figname}.png")
        except FileNotFoundError:
            print("Saving failed. File path doesn't exist!")
    plt.close(fig)

def plotCluster(X_umap, p_type, cell_labels=None, show_colormap=False, path='figures', figname='cells'):
    """
    Plot the predicted cell types from the encoder
    """
    
    cell_types = np.unique(cell_labels) if cell_labels is not None else np.unique(pred_labels)
    fig, ax = plt.subplots(1,2,figsize=(18,8))
    x = X_umap[:,0]
    y = X_umap[:,1]
    x_range = x.max()-x.min()
    y_range = y.max()-y.min()
    if(cell_labels is not None):
        for i, typei in enumerate(cell_types):
            mask = cell_labels==typei
            xbar, ybar = np.mean(x[mask]), np.mean(y[mask])
            ax[0].plot(x[mask], y[mask], '.', color=colors[i])
            ax[0].text(xbar - x_range*0.05, ybar - y_range*0.05, typei, fontsize=20, color='k')
    ax[0].set_title('True Labels')
    #Color cells according to the mode
    pred_labels = np.argmax(p_type,1)
    if(show_colormap):
        p_labels = np.array([p_type[i,pred_labels[i]] for i in range(pred_labels.shape[0])])
        for j, typej in enumerate(cell_types):
            mask = pred_labels==typej if isinstance(typej, int) else pred_labels==j
            if(not np.any(mask)):
                continue
            xbar, ybar = np.mean(x[mask]), np.mean(y[mask])
            ax[1].scatter(x[mask], y[mask],  c=p_labels[mask], cmap=colormaps[j], linewidth=0.02)
            if(j==0):
                norm1 = matplotlib.colors.Normalize(vmin=0.5, vmax=1.0)
                sm1 = matplotlib.cm.ScalarMappable(norm=norm1, cmap=colormaps[j])
                cbar1 = fig.colorbar(sm1,ax=ax)
                cbar1.ax.get_yaxis().labelpad = 15
                cbar1.ax.set_ylabel('Cell Type Probability',rotation=270,fontsize=12)
            ax[1].text(xbar - x_range*0.05, ybar - y_range*0.05, f'{typej}', fontsize=20, color='k')
    else:
        for j, typej in enumerate(cell_types):
            mask = pred_labels==typej if isinstance(typej, int) else pred_labels==j
            xbar, ybar = np.mean(x[mask]), np.mean(y[mask])
            ax[1].plot(x[mask], y[mask], '.', color=colors[j])
            ax[1].text(xbar - x_range*0.05, ybar - y_range*0.05, typej, fontsize=20, color='k')
    ax[1].set_title('Predicted Labels')
    ax[0].set_xlabel('Umap 1') 
    ax[0].set_ylabel('Umap 2') 
    ax[1].set_xlabel('Umap 1') 
    ax[1].set_ylabel('Umap 2') 
    try:
        fig.savefig(f'{path}/type_{figname}.png')
    except FileNotFoundError:
        print("Saving failed. File path doesn't exist!")
    plt.close(fig)


def plotTrainLoss(loss, iters, savefig=False, path='figures', figname="gene"):
    fig, ax = plt.subplots()
    ax.plot(iters, loss, '.-')
    ax.set_title("Training Loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    #plt.show()
    if(savefig):
        try:
            fig.savefig(f"{path}/train_loss_{figname}.png")
        except FileNotFoundError:
            print("Saving failed. File path doesn't exist!")
    plt.close(fig)

def plotTestLoss(loss, epoch, savefig=False, path='figures', figname="gene"):
    fig, ax = plt.subplots()
    ax.plot(epoch, loss, '.-')
    ax.set_title("Testing Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    #plt.show()
    if(savefig):
        try:
            fig.savefig(f"{path}/test_loss_{figname}.png")
        except FileNotFoundError:
            print("Saving failed. File path doesn't exist!")
    plt.close(fig)

def plotTestAcc(acc, epoch, savefig=False, path='figures', figname="gene"):
    fig, ax = plt.subplots()
    ax.plot(epoch, acc, '.-')
    ax.set_title("Test Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    #plt.show()
    if(savefig):
        try:
            if(path is None):
                fig.savefig(f"figures/test_acc_{figname}.png")
            else:
                fig.savefig(f"{path}/test_acc_{figname}.png")
        except FileNotFoundError:
            print("Saving failed. File path doesn't exist!")
    plt.close(fig)

def plotTLatent(t_latent, X_umap, title, savefig=False, path=None, figname="gene"):
    fig, ax = plt.subplots()
    ax.scatter(X_umap[:,0], X_umap[:,1], c=t_latent,cmap='plasma')
    norm = matplotlib.colors.Normalize(vmin=np.min(t_latent), vmax=np.max(t_latent))
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap='plasma')
    cbar = plt.colorbar(sm)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Latent Time',rotation=270)
    ax.set_title(title)
    
    if(savefig):
        try:
            fig.savefig(f"{path}/tlatent_{figname}.png")
        except FileNotFoundError:
            print("Saving failed. File path doesn't exist!")
    plt.close(fig)

#########################################################################
# Post Analysis
#########################################################################
def plotPhaseAxis(ax,
                  u,
                  s,
                  marker='.',
                  a=1.0,
                  D=1,
                  labels=None,
                  legends=None,
                  title=None):
    try:
        if(labels is None):
            ax.plot(s[::D],u[::D], marker, color='k')
        elif(legends is None):
            types = np.unique(labels)
            for l in types:
                mask = labels==l
                if(np.any(mask)):
                    ax.plot(s[mask][::D],u[mask][::D],marker,color=colors[l],alpha=a)
        else:
            for l in range(len(legends)): #l: label index, labels are cell types
                mask = labels==l
                if(np.any(mask)):
                    ax.plot(s[mask][::D],u[mask][::D],marker,color=colors[l],alpha=a,label=legends[l])
    except TypeError:
        return ax
    
    if(title is not None):
        ax.set_title(title, fontsize=12)
    
    return ax
    
def plotPhaseGrid(Nr, 
                  Nc, 
                  gene_list,
                  U, 
                  S, 
                  Labels, 
                  Legends,
                  Uhat={}, 
                  Shat={}, 
                  Labels_demo={},
                  W=6,
                  H=4,
                  alpha=0.3,
                  sparsify=1,
                  savefig=False,  
                  path='figures', 
                  figname="genes",
                  **kwargs):
    """
    Plot the phase portrait of a list of genes in an [Nr x Nc] grid.
    Cells are colored according to their dynamical state or cell type.
    U, S: [N_cell x N_gene]
    Uhat, Shat: Dictionary with arrays of size [N_cell x N_gene] as values
    """
    D = sparsify
    methods = list(Uhat.keys())
    
    M = min(1, len(methods))
    
    #Detect whether multiple figures are needed
    Nfig = len(gene_list) // (Nr*Nc)
    if(Nfig*Nr*Nc < Nfig):
        Nfig += 1
    
    for l in range(Nfig):
        fig_phase, ax_phase = plt.subplots(Nr, M*Nc, figsize=(W*M*Nc+1.0,H*Nr))
        if(Nr==1 and M*Nc==1): #Single Gene, Single Method
            labels = Labels[methods[0]] if Labels[methods[0]].ndim==1 else Labels[methods[0]][:,l]
            plotPhaseAxis(ax_phase, U[:,l], S[:,l], '.', alpha, D, labels, Legends[methods[0]], f"{gene_list[l]} ({methods[0]})")
            try:
                plotPhaseAxis(ax_phase, Uhat[methods[0]][:,l], Shat[methods[0]][:,l], '-', 1.0, 1, Labels_demo[methods[0]], f"{gene_list[l]} ({methods[0]})")
            except (KeyError, TypeError):
                print("[** Warning **]: Skip plotting the prediction because of key value error or invalid data type.")
                pass
            lgd = ax_phase.legend(fontsize=10, markerscale=3.0, bbox_to_anchor=(-0.15,1.0), loc='upper right')
        elif(Nr==1): #Single Gene, Multiple Method
            for j in range(min(Nc, len(gene_list)-l*Nc)):
                for k, method in enumerate(methods): 
                    labels = Labels[method] if Labels[method].ndim==1 else Labels[method][:,l*Nc+j]
                    plotPhaseAxis(ax_phase[M*j+k], U[:,l*Nc+j], S[:,l*Nc+j], '.', alpha, D, labels, Legends[method], f"{gene_list[l*Nc+j]} ({method})")
                    try:
                        plotPhaseAxis(ax_phase[M*j+k], Uhat[method][:,l*Nc+j], Shat[method][:,l*Nc+j], '-', 1.0, 1, Labels_demo[method], Legends[method], f"{gene_list[l*Nc+j]} ({method})")
                    except (KeyError, TypeError):
                        print("[** Warning **]: Skip plotting the prediction because of key value error or invalid data type.")
                        pass
            lgd = ax_phase[0].legend(fontsize=10, markerscale=3.0, bbox_to_anchor=(-0.15,1.0), loc='upper right')
        elif(M*Nc==1): #Multiple Gene, Single Method
            for i in range(min(Nr, len(gene_list)-l*Nr)):
                labels = Labels[methods[0]] if Labels[methods[0]].ndim==1 else Labels[methods[0]][:,l*Nr+i]
                plotPhaseAxis(ax_phase[i], U[:,l*Nr+i], S[:,l*Nr+i], '.',  alpha, D, labels, Legends[methods[0]], f"{gene_list[l*Nr+i]} ({method})")
                try:
                    plotPhaseAxis(ax_phase[i], Uhat[methods[0]][:,l*Nr+i], Shat[methods[0]][:,l*Nr+i], '-', 1.0, 1, Labels_demo[methods[0]], Legends[methods[0]], f"{gene_list[l*Nr+i]} ({method})")
                except (KeyError, TypeError):
                    print("[** Warning **]: Skip plotting the prediction because of key value error or invalid data type.")
                    pass
            lgd = ax_phase[0].legend(fontsize=10, markerscale=3.0, bbox_to_anchor=(-0.15,1.0), loc='upper right')
        else:
            for i in range(Nr):
                for j in range(Nc): #i, j: row and column gene index
                    idx = l*Nc*Nr+i*Nc+j
                    if(idx >= len(gene_list)):
                        break
                    u, s = U[:,idx], S[:,idx]
                    for k, method in enumerate(methods): 
                        labels = Labels[method] if Labels[method].ndim==1 else Labels[method][:,idx]
                        plotPhaseAxis(ax_phase[i,M*j+k], U[:,idx], S[:,idx], '.', alpha, D, labels, Legends[method], f"{gene_list[idx]} ({method})")
                        try:
                            plotPhaseAxis(ax_phase[i,M*j+k], Uhat[method][:,idx], Shat[method][:,idx], '-', 1.0, 1, Labels_demo[method], Legends[method], f"{gene_list[idx]} ({method})")
                        except (KeyError, TypeError):
                            print("[** Warning **]: Skip plotting the prediction because of key value error or invalid data type.")
                            pass
            lgd = ax_phase[0,0].legend(fontsize=10, markerscale=3.0, bbox_to_anchor=(-0.15,1.0), loc='upper right')
        
        fig_phase.subplots_adjust(hspace = 0.3, wspace=0.3)
        if(savefig):
            try:
                fig_phase.savefig(f'{path}/phase_{figname}_{l+1}.png',bbox_extra_artists=(lgd,), bbox_inches='tight')
            except FileNotFoundError:
                print("Saving failed. File path doesn't exist!")
        plt.close(fig_phase)

def plotSigAxis(ax,
                t,
                x,
                labels=None,
                legends=None,
                marker='.',
                a=1.0,
                D=1,
                show_legend=False,
                title=None):
    lines = []
    if(labels is None or legends is None):
        lines.append( ax.plot(t[::D], x[::D], marker, markersize=5, color='k', alpha=a)[0] )
    else:
        for i in range(len(legends)):
            mask = labels==i
            if(np.any(mask)):
                if(show_legend):
                    line = ax.plot(t[mask][::D], x[mask][::D], marker, markersize=5, color=colors[i], alpha=a, label=legends[i])[0]
                    lines.append(line)
                else:
                    ax.plot(t[mask][::D], x[mask][::D], marker, markersize=5, color=colors[i], alpha=a)
                
    if(title is not None):
        ax.set_title(title, fontsize=36)
    return lines

def plotSigPredAxis(ax,
                t,
                x,
                labels=None,
                legends=None,
                marker='.',
                a=1.0,
                D=1,
                show_legend=False,
                title=None):
    if(labels is None or legends is None):
        ax.plot(t[::D], x[::D], marker, linewidth=5, color='k', alpha=a)
    else:
        for i in range(len(legends)):
            mask = labels==i
            if(np.any(mask)):
                if(show_legend):
                    ax.plot(t[mask][::D], x[mask][::D], marker, linewidth=5, color='k', alpha=a, label=legends[i])
                else:
                    ax.plot(t[mask][::D], x[mask][::D], marker, linewidth=5, color='k', alpha=a)
    if(title is not None):
        ax.set_title(title, fontsize=36)
    return ax

def plotSigloessAxis(ax,
                   t,
                   x,
                   labels,
                   legends,
                   frac=0.5,
                   a=1.0,
                   D=1,
                   show_legend=False,
                   title=None,):
    xt = np.stack([t,x])
    Ngrid = max(len(t)//200, 50)
    for i in range(len(legends)):
        mask = labels==i
        if(np.any(mask)):
            t_lb, t_ub = np.quantile(t[mask], 0.05), np.quantile(t[mask], 0.95)
            mask2 = (t<=t_ub) & (t >= t_lb) & mask
            tout, xout, wout = loess_1d.loess_1d(t[mask2], x[mask2], xnew=None, degree=1, frac=frac, npoints=None, rotate=False, sigy=None)
            torder = np.argsort(tout)
            if(show_legend):
                ax.plot(tout[torder][::D], xout[torder][::D], 'k-', linewidth=5, alpha=a, label=legends[i])
            else:
                ax.plot(tout[torder][::D], xout[torder][::D], 'k-', linewidth=5, alpha=a)
    if(title is not None):
        ax.set_title(title, fontsize=36)
    return ax

def sampleQuiverPlot(t, dt):
    tmax, tmin = t.max()+1e-3, t.min()
    Nbin = max(1, int( (tmax-tmin)/dt ))
    indices = []
    for i in range(Nbin):
        I = np.where((t>=tmin+i*dt) & (t<=tmin+(i+1)*dt))[0]
        if(len(I)>0):
            indices.append(I[0])
    return np.array(indices).astype(int)

def plotVelAxis(ax,
                t,
                x,
                v,
                labels=None,
                legends=None,
                dt=0.1,
                a=1.0,
                show_legend=False,
                title=None,):
    
    if(labels is None or legends is None):
        dt_sample = (t.max()-t.min())/50
        torder = np.argsort(t)
        indices = sampleQuiverPlot(t[torder], dt_sample)
        ax.quiver(t[torder][indices], 
                  x[torder][indices], 
                  dt*np.ones((len(t))), 
                  dt*v[torder][indices], 
                  angles='xy', 
                  scale=None, 
                  scale_units='inches', 
                  headwidth=5.0, 
                  headlength=8.0, 
                  color='k')
    else:
        for i in range(len(legends)):
            mask = labels==i
            t_type = t[mask]
            if(np.any(mask)):
                t_lb, t_ub = np.quantile(t_type, 0.02), np.quantile(t_type, 0.98)
                mask2 = (t_type<t_ub) & (t_type>=t_lb)
                t_type = t_type[mask2]
                dt_sample = (t_type.max()-t_type.min())/20
                torder = np.argsort(t_type)
                indices = sampleQuiverPlot(t_type[torder], dt_sample)
                v_type = v[mask][torder][indices]
                v_type = np.clip(v_type, np.quantile(v_type,0.02), np.quantile(v_type,0.98))
                ax.quiver(t_type[torder][indices], 
                          x[mask][mask2][torder][indices], 
                          dt*np.ones((len(indices))), 
                          dt*v_type, 
                          label=legends[i],
                          angles='xy', 
                          scale=None, 
                          scale_units='inches', 
                          headwidth=5.0, 
                          headlength=8.0, 
                          color=colors[i])
    return ax

def plotSigGrid(Nr, 
                Nc, 
                gene_list,
                T,
                U, 
                S, 
                Labels,
                Legends,
                That={},
                Uhat={}, 
                Shat={}, 
                V={},
                Labels_demo={},
                W=6,
                H=5,
                frac=0.5,
                alpha=1.0,
                sparsify=1,
                savefig=False,  
                path='figures', 
                figname="genes"):
    """
    Plot u/s of a list of genes vs. time in an [Nr x Nc] grid.
    Cells are colored according to their dynamical state or cell type.
    T: [N_method x N_cell] 
    T_local: [N_cell x N_gene] 
    U, S: [N_cell x N_gene]
    Uhat, Shat: Dictionary with method names as keys and values of size [N_method x N_cell x N_gene]
    Labels: Dictionary with array of labels as values
    """
    methods = list(Uhat.keys())
    M = max(1, len(methods))
    
    
    #Detect whether multiple figures are needed
    Nfig = len(gene_list) // (Nr*Nc)
    if(Nfig * Nr * Nc < len(gene_list)):
        Nfig += 1
    
    #Plotting
    for l in range(Nfig):
        fig_sig, ax_sig = plt.subplots(3*Nr,M*Nc,figsize=(W*M*Nc+1.0, 3*H*Nr))
        if(M*Nc==1):
            for i in range(min(Nr,len(gene_list)-l*Nr)):
                idx = l*Nr+i
                t = T[methods[0]][:,idx] if T[methods[0]].ndim==2 else T[methods[0]]
                that = That[methods[0]][:,idx] if methods[0].ndim==2 else That[methods[0]]
                line_u = plotSigAxis(ax_sig[3*i], t, U[:,idx], Labels[methods[0]], Legends[methods[0]], '.', alpha, sparsify, True, f"{gene_list[idx]} ({methods[0]})")
                line_s = plotSigAxis(ax_sig[3*i+1], t, S[:,idx], Labels[methods[0]], Legends[methods[0]], '.', alpha, sparsify)
                if(len(line_u)>0):
                    lines = line_u
                try:
                    if methods[0]=='VeloVAE':
                        K = max(len(that)//5000, 1)
                        plotSigloessAxis(ax_sig[3*i], that[::K], Uhat[methods[0]][:,idx][::K], Labels_demo[methods[0]][::K], Legends[methods[0]], frac=frac)
                        plotSigloessAxis(ax_sig[3*i+1], that[::K], Shat[methods[0]][:,idx][::K], Labels_demo[methods[0]][::K], Legends[methods[0]], frac=frac)
                        plotVelAxis(ax_sig[3*i+2], t[::K], S[:,idx][::K], V[methods[0]][:,idx][::K], Labels[methods[0]][::K], Legends[methods[0]])
                    else:
                        plotSigPredAxis(ax_sig[3*i], that, Uhat[methods[0]][:,idx], Labels_demo[methods[0]], Legends[methods[0]], '-', 1.0, 1)
                        plotSigPredAxis(ax_sig[3*i+1], that, Shat[methods[0]][:,idx], Labels_demo[methods[0]], Legends[methods[0]], '-', 1.0, 1)
                        plotVelAxis(ax_sig[3*i+2], t, S[:,idx], V[methods[0]][:,idx], Labels[methods[0]], Legends[methods[0]])
                except (KeyError, TypeError):
                    print("[** Warning **]: Skip plotting the prediction because of key value error or invalid data type.")
                    pass
                
                ax_sig[3*i].axis("off")
                ax_sig[3*i+1].axis("off")
                tmin, tmax = ax_sig[3*i].get_xlim()
                umin, umax = ax_sig[3*i].get_ylim()
                ax_sig[3*i].text(tmin - 0.02*(tmax-tmin), (umax+umin)*0.5, "U", fontsize=36)
                tmin, tmax = ax_sig[3*i+1].get_xlim()
                smin, smax = ax_sig[3*i+1].get_ylim()
                ax_sig[3*i+1].text(tmin - 0.02*(tmax-tmin), (smax+smin)*0.5,"S", fontsize=36)
                tmin, tmax = ax_sig[3*i+2].get_xlim()
                smin, smax = ax_sig[3*i+2].get_ylim()
                ax_sig[3*i+2].text(tmin - 0.02*(tmax-tmin), (smax+smin)*0.5,"V", fontsize=36)
            lgd = fig_sig.legend(lines, Legends[methods[0]], fontsize=18, markerscale=5.0, ncol=8, bbox_to_anchor=(0.5, 1.0), loc='center')
        else:
            legends = []
            for i in range(Nr):
                for j in range(Nc): #i, j: row and column gene index
                    idx = l*Nr*Nc+i*Nc+j #which gene
                    if(idx >= len(gene_list)):
                        break
                    for k, method in enumerate(methods): #k: method index
                        #Pick time according to the method
                        if(T[method].ndim==2):
                            t = T[method][:,idx] 
                            that = That[method][:,idx]
                        else:
                            t = T[method]
                            that = That[method]
                        
                        line_u = plotSigAxis(ax_sig[3*i, M*j+k], t, U[:,idx], Labels[method], Legends[method], '.', alpha, sparsify, True, f"{gene_list[idx]} ({method})")
                        line_s = plotSigAxis(ax_sig[3*i+1, M*j+k], t, S[:,idx], Labels[method], Legends[method], '.', alpha, sparsify)
                        if(len(line_u)>0):
                            lines = line_u
                        if(len(Legends[method])>len(legends)):
                            legends = Legends[method]
                        try:
                            if method=='VeloVAE':
                                K = max(len(that)//5000, 1)
                                plotSigloessAxis(ax_sig[3*i, M*j+k], that[::K], Uhat[method][:,idx][::K], Labels_demo[method][::K], Legends[method], frac=frac)
                                plotSigloessAxis(ax_sig[3*i+1, M*j+k], that[::K], Shat[method][:,idx][::K], Labels_demo[method][::K], Legends[method], frac=frac)
                                plotVelAxis(ax_sig[3*i+2, M*j+k], t[::K], S[:,idx][::K], V[method][:,idx][::K], Labels[method][::K], Legends[method])
                            else:
                                plotSigPredAxis(ax_sig[3*i, M*j+k], that, Uhat[method][:,idx], Labels_demo[method], Legends[method], '-', 1.0, 1)
                                plotSigPredAxis(ax_sig[3*i+1, M*j+k], that, Shat[method][:,idx], Labels_demo[method], Legends[method], '-', 1.0, 1)
                                plotVelAxis(ax_sig[3*i+2, M*j+k], t, S[:,idx], V[method][:,idx], Labels[method], Legends[method])
                        except (KeyError, TypeError):
                            print("[** Warning **]: Skip plotting the prediction because of key value error or invalid data type.")
                            pass
                        #ax_sig[3*i,  M*j+k].axis("off")
                        #ax_sig[3*i+1, M*j+k].axis("off")
                        #ax_sig[3*i+2, M*j+k].axis("off")
                        ax_sig[3*i,  M*j+k].set_xticks([])
                        ax_sig[3*i+1,  M*j+k].set_xticks([])
                        ax_sig[3*i+2,  M*j+k].set_xticks([])
                        ax_sig[3*i,  M*j+k].set_yticks([])
                        ax_sig[3*i+1,  M*j+k].set_yticks([])
                        ax_sig[3*i+2,  M*j+k].set_yticks([])
                        
                        tmin, tmax = ax_sig[3*i,  M*j+k].get_xlim()
                        umin, umax = ax_sig[3*i,  M*j+k].get_ylim()
                        ax_sig[3*i,  M*j+k].text(tmin - 0.08*(tmax-tmin), (umax+umin)*0.5, "U", fontsize=30)
                        #ax_sig[3*i,  M*j+k].text((tmin+tmax)*0.4, umin - (umax-umin)*0.1, "Time", fontsize=36)
                        tmin, tmax = ax_sig[3*i+1,  M*j+k].get_xlim()
                        smin, smax = ax_sig[3*i+1,  M*j+k].get_ylim()
                        ax_sig[3*i+1, M*j+k].text(tmin - 0.08*(tmax-tmin), (smax+smin)*0.5,"S", fontsize=30)
                        #ax_sig[3*i+1,  M*j+k].text((tmin+tmax)*0.4, smin - (smax-smin)*0.1, "Time", fontsize=36)
                        tmin, tmax = ax_sig[3*i+2,  M*j+k].get_xlim()
                        smin, smax = ax_sig[3*i+2,  M*j+k].get_ylim()
                        ax_sig[3*i+2, M*j+k].text(tmin - 0.08*(tmax-tmin), (smax+smin)*0.5,"S", fontsize=30)
                        #ax_sig[3*i+2,  M*j+k].text((tmin+tmax)*0.4, smin - (smax-smin)*0.1, "Time", fontsize=36)
                        
                        ax_sig[3*i,  M*j+k].set_xlabel("Time", fontsize=30)
                        ax_sig[3*i+1,  M*j+k].set_xlabel("Time", fontsize=30)
                        ax_sig[3*i+2,  M*j+k].set_xlabel("Time", fontsize=30)
            lgd = fig_sig.legend(lines, legends, fontsize=19*Nc, markerscale=5*Nc, ncol=min(2*M*Nc, 4), bbox_to_anchor=(0.0, 1.0, 1.0, 0.01), loc='center')
        fig_sig.subplots_adjust(hspace = 0.3, wspace=0.12)
        if(savefig):
            try:
                fig_sig.savefig(f'{path}/sig_{figname}_{l+1}.png',bbox_extra_artists=(lgd,), dpi=300, bbox_inches='tight') 
                print(f'Saved to {path}/sig_{figname}_{l+1}.png')
            except FileNotFoundError:
                print("Saving failed. File path doesn't exist!")
        plt.close(fig_sig)

def plotClusterGrid(X_umap,
                    Py, 
                    cell_types,
                    show_colormap=False,
                    savefig=True,  
                    path='figures', 
                    figname="cluster",
                    **kwargs):
    methods = list(Py.keys())
    M = len(methods)
    
    x = X_umap[:,0]
    y = X_umap[:,1]
    x_range = x.max()-x.min()
    y_range = y.max()-y.min()
    
    fig, ax = plt.subplots(1, M, figsize=(4*M, 3))
    for i, method in enumerate(methods):
        p_type = Py[method]
        pred_labels = np.argmax(p_type,1)
        if(show_colormap):
            p_labels = np.max(p_type, 1)
            for j, typej in enumerate(cell_types):
                mask = pred_labels==typej if isinstance(typej, int) else pred_labels==j
                if(not np.any(mask)):
                    continue
                xbar, ybar = np.mean(x[mask]), np.mean(y[mask])
                try:
                    ax[i].scatter(x[mask], y[mask],  c=p_labels[mask], cmap=colormaps[j], linewidth=0.02)
                    if(i==M-1 and j==0):
                        norm1 = matplotlib.colors.Normalize(vmin=0.5, vmax=1.0)
                        sm1 = matplotlib.cm.ScalarMappable(norm=norm1, cmap=colormaps[j])
                        cbar1 = fig.colorbar(sm1,ax=ax[i])
                        cbar1.ax.get_yaxis().labelpad = 15
                        cbar1.ax.set_ylabel('Cell Type Probability',rotation=270,fontsize=12)
                    ax[i].text(xbar - x_range*0.05, ybar - y_range*0.05, f'{typej}', fontsize=12, color='k')
                except TypeError:
                    ax.scatter(x[mask], y[mask],  c=p_labels[mask], cmap=colormaps[j], linewidth=0.02)
                    if(i==M-1 and j==0):
                        norm1 = matplotlib.colors.Normalize(vmin=0.5, vmax=1.0)
                        sm1 = matplotlib.cm.ScalarMappable(norm=norm1, cmap=colormaps[j])
                        cbar1 = fig.colorbar(sm1,ax=ax)
                        cbar1.ax.get_yaxis().labelpad = 15
                        cbar1.ax.set_ylabel('Cell Type Probability',rotation=270,fontsize=12)
                    ax.text(xbar - x_range*0.05, ybar - y_range*0.05, f'{typej}', fontsize=12, color='k')
        else:
            for j, typej in enumerate(cell_types):
                mask = pred_labels==typej if isinstance(typej, int) else pred_labels==j
                if(not np.any(mask)):
                    continue
                xbar, ybar = np.mean(x[mask]), np.mean(y[mask])
                try:
                    ax[i].plot(x[mask], y[mask], '.', color=colors[j])
                    ax[i].text(xbar - x_range*0.05, ybar - y_range*0.05, typej, fontsize=12, color='k')
                except TypeError:
                    ax.plot(x[mask], y[mask], '.', color=colors[j])
                    ax.text(xbar - x_range*0.05, ybar - y_range*0.05, typej, fontsize=12, color='k')
        if(M>1):
            ax[i].set_title(f'{method} Labels')
            ax[i].set_xlabel('Umap 1') 
            ax[i].set_ylabel('Umap 2') 
        else:
            ax.set_title(f'{method} Labels')
            ax.set_xlabel('Umap 1') 
            ax.set_ylabel('Umap 2') 
    fig.subplots_adjust(hspace = 0.25, wspace=0.1)
    if(savefig):
        try:
            fig.savefig(f'{path}/{figname}.png')
            print(f'Saved to {path}/{figname}.png')
        except FileNotFoundError:
            print("Saving failed. File path doesn't exist!")
    plt.close(fig)
    return

def plotTimeGrid(T,
                 X_emb,
                 capture_time=None,
                 std_t=None,
                 savefig=False,  
                 figname="time",
                 path='figures'):
    """
    Plot the latent time of different methods.
    T: dictionary
    X_emb: [N_cell x 2]
    to, ts: [N_gene]
    """
    if(capture_time is not None):
        methods = ["Capture Time"] + list(T.keys())
    else:
        methods = list(T.keys())
    M = len(methods)
    if(std_t is not None):
        fig_time, ax = plt.subplots(2, M, figsize=(6*M+2,8))
        for i, method in enumerate(methods):
            t = capture_time if method=="Capture Time" else T[method]
            t = t - t.min()
            t = t/t.max()
            ax[0, i].scatter(X_emb[:,0], X_emb[:,1], s=2.0, c=t, cmap='plasma', edgecolors='none')
            ax[0, i].set_title(f'{method}',fontsize=24)
            ax[0, i].axis('off')

            #Plot the Time Variance in a Colormap
            var_t = std_t[method]**2
            
            if(np.any(var_t>0)):
                ax[1, i].scatter(X_emb[:,0], X_emb[:,1], s=2.0, c=var_t, cmap='Reds', edgecolors='none')
                norm1 = matplotlib.colors.Normalize(vmin=np.min(var_t), vmax=np.max(var_t))
                sm1 = matplotlib.cm.ScalarMappable(norm=norm1, cmap='Reds')
                cbar1 = fig_time.colorbar(sm1,ax=ax[1, i])
                cbar1.ax.get_yaxis().labelpad = 15
                cbar1.ax.set_ylabel('Time Variance',rotation=270,fontsize=12)
                ax[1, i].axis('off')
    else:
        fig_time, ax = plt.subplots(1, M, figsize=(8*M,4))
        for i, method in enumerate(methods):
            t = capture_time if method=="Capture Time" else T[method]
            t = t - t.min()
            t = t/t.max()
            ax[i].scatter(X_emb[:,0], X_emb[:,1], s=2.0, c=t, cmap='plasma', edgecolors='none')
            ax[i].set_title(f'{method}',fontsize=24)
            ax[i].axis('off')
    norm0 = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sm0 = matplotlib.cm.ScalarMappable(norm=norm0, cmap='plasma')
    cbar0 = fig_time.colorbar(sm0,ax=ax, location="right")
    cbar0.ax.get_yaxis().labelpad = 20
    cbar0.ax.set_ylabel('Latent Time',rotation=270,fontsize=24)
    if(savefig):
        try:
            fig_time.savefig(f'{path}/{figname}.png')
            print(f'Saved to {path}/{figname}.png')
        except FileNotFoundError:
            print("Saving failed. File path doesn't exist!")
    plt.close(fig_time)




def plotVelocity(X_umap, vx, vy, scale=1.0, path='figures', figname='genes'):
    umap1, umap2 = X_umap[:,0], X_umap[:,1]
    fig, ax = plt.subplots(figsize=(12,8))
    v = np.sqrt(vx**2+vy**2)
    vmax, vmin = np.quantile(v,0.95), np.quantile(v,0.05)
    v = np.clip(v, vmin, vmax)
    ax.plot(umap1, umap2, '.', alpha=0.5)
    ax.quiver(umap1, umap2, vx, vy, v, angles='xy', scale=scale)
    try:
        fig.savefig(f"{path}/vel_{figname}.png")
    except FileNotFoundError:
        print("Saving failed. File path doesn't exist!")
    plt.close(fig)

def plotVelocityStream(X_umap, vx, vy, cell_labels, figsize=(7,5), path='figures', figname='genes'):
    #Compute velocity on a grid
    N = 50
    umap1, umap2 = X_umap[:,0], X_umap[:,1]
    Y, X = np.mgrid[umap2.min():umap2.max():50j, umap1.min():umap1.max():50j]
    grid = np.stack([X.flatten(), Y.flatten()]).T
    knn = NearestNeighbors(n_neighbors=30)
    knn = knn.fit(X_umap)
    dist, ind = knn.kneighbors(grid)
    sigma = np.max(dist, 1)
    A = np.exp(-(dist/sigma.reshape(-1,1))**2)
    A = A/A.sum(1).reshape(-1,1)
    
    dist_thred = (umap1.max()-umap1.min())/N + (umap2.max()-umap2.min())/N
    dist_med = np.quantile(dist,0.5,1)
    
    vx_grid = np.array([A[i].dot(vx[ind[i]]) for i in range(A.shape[0])])
    vy_grid = np.array([A[i].dot(vy[ind[i]]) for i in range(A.shape[0])])
    
    vx_grid[dist_med > dist_thred] = 0
    vy_grid[dist_med > dist_thred] = 0
    
    fig, ax = plt.subplots(figsize=figsize)
    #Plot cells by label
    for i, type_ in enumerate(np.unique(cell_labels)):
        mask = cell_labels==type_
        ax.plot(umap1[mask], umap2[mask], '.', color=colors[i], alpha=0.3, label=type_)
    ax.streamplot(grid[:,0].reshape(N,N), grid[:,1].reshape(N,N), vx_grid.reshape(N,N), vy_grid.reshape(N,N), density=2, color='k')
    ax.set_title('Velocity Stream Plot')
    ax.set_xlabel('Umap 1')
    ax.set_ylabel('Umap 2')
    try:
        fig.savefig(f"{path}/velstream_{figname}.png")
    except FileNotFoundError:
        print("Saving failed. File path doesn't exist!")
    plt.close(fig)

def plotUmapTransition(graph, X_umap, cell_labels, label_dic_rev, path='figures', figname='umaptrans'):
    """
    Plot the Umap coordinates and connect the cluster centers with a line.
    Transition probability is encoded in the opacity of the line
    T: transition matrix [n_type x n_type]
    X_umap: umap coordinates [ncell x 2]
    """
    fig, ax = plt.subplots(figsize=(8,6))
    Xmean = {}
    for i in graph:
        mask = cell_labels==i
        xbar, ybar = np.mean(X_umap[mask,0]),np.mean(X_umap[mask,1])
        Xmean[i] = (xbar,ybar)
    
    for i in graph.keys():
        mask = cell_labels==i
        ax.plot(X_umap[mask,0],X_umap[mask,1],'.',color=colors[i],alpha=0.1)
    
    for i in graph.keys():
        mask = cell_labels==i
        if(Xmean[i][0]>0):
            ax.text(Xmean[i][0]*1.05,Xmean[i][1],label_dic_rev[i],fontsize=14)
        else:
            ax.text(Xmean[i][0]*0.95,Xmean[i][1],label_dic_rev[i],fontsize=14)
        for j in graph[i]:
            ax.arrow(Xmean[i][0],Xmean[i][1],Xmean[j][0]-Xmean[i][0],Xmean[j][1]-Xmean[i][1],width=0.15,head_width=0.6,length_includes_head=True,color='k')
    ax.set_xlabel('Umap Dim 1')
    ax.set_ylabel('Umap Dim 2')
    try:
        fig.savefig(f'{path}/{figname}.png')
    except FileNotFoundError:
        print("Saving failed. File path doesn't exist!")
    plt.close(fig)


def plotLatentEmbedding(X,  n_cluster, labels, label_dic_rev, savefig=True, path="figures", figname="gene"):
    #Y = TSNE().fit_transform(X)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
    Y = reducer.fit_transform(X)
    fig, ax = plt.subplots()
    
    for i in range(n_cluster):
        ax.plot(Y[labels==i,0], Y[labels==i,1], '.', c=colors[i], label=label_dic_rev[i])
    
    ax.set_xlabel('Umap 1')
    ax.set_ylabel('Umap 2')
    
    lgd = ax.legend(fontsize=10, markerscale=3.0,  bbox_to_anchor=(-0.15,1.0), loc='upper right')
    if(savefig):
        try:
            fig.savefig(f"{path}/latent_{figname}.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
        except FileNotFoundError:
            print("Saving failed. File path doesn't exist!")
    plt.close(fig)

def plotTs(ts,savefig=True, path="figures", figname="gene"):
    fig, ax = plt.subplots()
    ax.plot(ts, np.ones(ts.shape), 'k+')
    ub = np.quantile(ts,0.99)
    print(np.sum(ts>ub))
    ax.hist(ts,bins=np.linspace(0,ub,100),range=(ts.min(),ub))
    ax.set_title("Distribution of Switching Time")
    plt.show()
    if(savefig):
        try:
            fig.savefig(f"{path}/ts_{figname}.png")
        except FileNotFoundError:
            print("Saving failed. File path doesn't exist!")
    plt.close(fig)


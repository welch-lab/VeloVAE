import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import igraph as ig
import pynndescent
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm


#######################################################################################
# Default colors and markers for plotting
#######################################################################################
TAB10 = list(plt.get_cmap("tab10").colors)
TAB20 = list(plt.get_cmap("tab20").colors)
TAB20B = list(plt.get_cmap("tab20b").colors)
TAB20C = list(plt.get_cmap("tab20c").colors)
RAINBOW = [plt.cm.rainbow(i) for i in range(256)]

markers = ["o", "x", "s", "v", "+", "d", "1", "*", "^", "p", "h", "8", "1", "2", "|"]
# change dpi via the function set_dpi()
DPI = 'figure'


def set_dpi(dpi):
    global DPI
    DPI = dpi


def get_colors(n, color_map=None):
    """Get colors for plotting cell clusters.

    Args:
        n (int):
            Number of cell clusters
        color_map (str, optional):
            User-defined colormap. 
            If not set, the colors will be chosen as
            the colors for tabular data in matplotlib.
            Defaults to None.

    Returns:
        list[tuple]: list of color parameters
    """
    if color_map is None:  # default color
        if n <= 10:
            return TAB10[:n]
        elif n <= 20:
            return TAB20[:n]
        elif n <= 40:
            TAB40 = TAB20B+TAB20C
            return TAB40[:n]
        else:
            print("Warning: Number of colors exceeds the maximum (40)! Use a continuous colormap (256) instead.")
            return RAINBOW[:n]
    else:
        color_map_obj = list(plt.get_cmap(color_map).colors)
        k = len(color_map_obj)//n
        colors = ([color_map_obj(i) for i in range(0, len(color_map_obj), k)]
                  if k > 0 else
                  [color_map_obj(i) for i in range(len(color_map_obj))])
    return colors


def save_fig(fig, save, bbox_extra_artists=None):
    global DPI
    if save is not None:
        try:
            idx = save.rfind('.')
            fig.savefig(save,
                        dpi=DPI,
                        bbox_extra_artists=bbox_extra_artists,
                        format=save[idx+1:],
                        bbox_inches='tight')
        except FileNotFoundError:
            print("Saving failed. File path doesn't exist!")
        # plt.close(fig)


############################################################
# Functions used in debugging.
############################################################
def plot_sig(t,
             u, s,
             upred, spred,
             cell_labels=None,
             title="Gene",
             save=None,
             **kwargs):
    """Generate a 2x2 u/s-t plot for a single gene

    Args:
        t (:class:`numpy.ndarray`):
            Cell time, (N, )
        u (:class:`numpy.ndarray`):
            Unspliced counts of a single gene, (N, )
        s (:class:`numpy.ndarray`):
            Spliced counts of a single gene, (N, )
        upred (:class:`numpy.ndarray`):
            Predicted unspliced counts of a single gene, (N, )
        spred (:class:`numpy.ndarray`):
            Predicted spliced counts of a single gene, (N, )
        cell_labels (:class:`numpy.ndarray`, optional):
            Cell type annotations. Defaults to None.
        title (str, optional):
            Figure title. Defaults to "Gene".
        save (str, optional):
            Figure name for saving (including path). Defaults to None.
    """
    D = kwargs['sparsify'] if 'sparsify' in kwargs else 1
    tscv = kwargs['tscv'] if 'tscv' in kwargs else t
    tdemo = kwargs["tdemo"] if "tdemo" in kwargs else t
    if cell_labels is None:
        fig, ax = plt.subplots(2, 1, figsize=(15, 12), facecolor='white')
        ax[0].plot(t[::D], u[::D], 'b.', label="raw")
        ax[1].plot(t[::D], s[::D], 'b.', label="raw")
        ax[0].plot(tdemo, upred, '.', color='lawngreen', label='Prediction', linewidth=2.0)
        ax[1].plot(tdemo, spred, '.', color='lawngreen', label="Prediction", linewidth=2.0)

        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("U", fontsize=18)

        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("S", fontsize=18)

        handles, labels = ax[1].get_legend_handles_labels()
    else:
        fig, ax = plt.subplots(2, 2, figsize=(24, 12), facecolor='white')
        labels_pred = kwargs['labels_pred'] if 'labels_pred' in kwargs else []
        labels_demo = kwargs['labels_demo'] if 'labels_demo' in kwargs else None
        cell_types = np.unique(cell_labels)
        colors = get_colors(len(cell_types), None)

        # Plot the input data in the true labels
        for i, type_ in enumerate(cell_types):
            mask_type = cell_labels == type_
            ax[0, 0].scatter(tscv[mask_type][::D],
                             u[mask_type][::D],
                             s=8.0,
                             color=colors[i % len(colors)],
                             alpha=0.7,
                             label=type_,
                             edgecolors='none')
            ax[0, 1].scatter(tscv[mask_type][::D],
                             s[mask_type][::D],
                             s=8.0,
                             color=colors[i % len(colors)],
                             alpha=0.7,
                             label=type_, edgecolors='none')
            if len(labels_pred) > 0:
                mask_mytype = labels_pred == type_
                ax[1, 0].scatter(t[mask_mytype][::D],
                                 u[mask_mytype][::D],
                                 s=8.0,
                                 color=colors[i % len(colors)],
                                 alpha=0.7,
                                 label=type_,
                                 edgecolors='none')
                ax[1, 1].scatter(t[mask_mytype][::D],
                                 s[mask_mytype][::D],
                                 s=8.0,
                                 color=colors[i % len(colors)],
                                 alpha=0.7,
                                 label=type_,
                                 edgecolors='none')
            else:
                ax[1, 0].scatter(t[mask_type][::D],
                                 u[mask_type][::D],
                                 s=8.0,
                                 color=colors[i % len(colors)],
                                 alpha=0.7,
                                 label=type_,
                                 edgecolors='none')
                ax[1, 1].scatter(t[mask_type][::D],
                                 s[mask_type][::D],
                                 s=8.0,
                                 color=colors[i % len(colors)],
                                 alpha=0.7,
                                 label=type_,
                                 edgecolors='none')

        if labels_demo is not None:
            for i, type_ in enumerate(cell_types):
                mask_mytype = labels_demo == type_
                order = np.argsort(tdemo[mask_mytype])
                ax[1, 0].plot(tdemo[mask_mytype][order],
                              upred[mask_mytype][order],
                              color=colors[i % len(colors)],
                              linewidth=2.0)
                ax[1, 1].plot(tdemo[mask_mytype][order],
                              spred[mask_mytype][order],
                              color=colors[i % len(colors)],
                              linewidth=2.0)
        else:
            order = np.argsort(tdemo)
            ax[1, 0].plot(tdemo[order], upred[order], 'k.', linewidth=2.0)
            ax[1, 1].plot(tdemo[order], spred[order], 'k.', linewidth=2.0)

        if 't_trans' in kwargs:
            t_trans = kwargs['t_trans']
            for i, type_ in enumerate(cell_types):
                ax[1, 0].plot([t_trans[i], t_trans[i]], [0, u.max()], '-x', color=colors[i % len(colors)])
                ax[1, 1].plot([t_trans[i], t_trans[i]], [0, s.max()], '-x', color=colors[i % len(colors)])
        for j in range(2):
            ax[j, 0].set_xlabel("Time")
            ax[j, 0].set_ylabel("U", fontsize=18)

            ax[j, 1].set_xlabel("Time")
            ax[j, 1].set_ylabel("S", fontsize=18)
            handles, labels = ax[1, 0].get_legend_handles_labels()

        if 'subtitles' in kwargs:
            ax[0, 0].set_title(f"Unspliced, {kwargs['subtitles'][0]}")
            ax[0, 1].set_title(f"Spliced, {kwargs['subtitles'][0]}")
            ax[1, 0].set_title(f"Unspliced, {kwargs['subtitles'][1]}")
            ax[1, 1].set_title(f"Spliced, {kwargs['subtitles'][1]}")
        else:
            ax[0, 0].set_title('Unspliced, True Label')
            ax[0, 1].set_title('Spliced, True Label')
            ax[1, 0].set_title('Unspliced, VAE')
            ax[1, 1].set_title('Spliced, VAE')
    lgd = fig.legend(handles,
                     labels,
                     fontsize=15,
                     markerscale=5,
                     ncol=4,
                     bbox_to_anchor=(0.0, 1.0, 1.0, 0.25),
                     loc='center')
    fig.suptitle(title, fontsize=28)
    plt.tight_layout()

    save_fig(fig, save, (lgd,))
    return


def plot_phase(u, s,
               upred, spred,
               title,
               track_idx=None,
               labels=None,
               types=None,
               save=None):
    """Plot the phase portrait of a gene

    Args:
        u (:class:`numpy array`):
            Unspliced counts of a single gene, (N, )
        s (:class:`numpy array`):
            Spliced counts of a single gene, (N, )
        upred (:class:`numpy array`):
            Predicted unspliced counts of a single gene, (N, )
        spred (:class:`numpy array`):
            Predicted spliced counts of a single gene, (N, )
        title (str):
            Figure title.
        track_idx (:class:`numpy array`, optional):
            Cell indices to plot correspondence between data points and predicted phase portrait.
            Defaults to None.
        labels (_type_, optional):
            Cell state annotation (off, induction or repression). Defaults to None.
        types (:class:`numpy.ndarray`, optional):
            Unique cell types
        save (str, optional):
            Figure name for saving (including path). Defaults to None.
    """
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
    if labels is None or types is None:
        ax.scatter(s, u, c="b", alpha=0.5)
    else:
        colors = get_colors(len(types), None)
        for i, type_ in enumerate(types):
            ax.scatter(s[labels == i], u[labels == i], color=colors[i % len(colors)], alpha=0.3, label=type_)
    ax.plot(spred, upred, 'k.', label="ode")
    # Plot the correspondence
    if track_idx is None:
        perm = np.random.permutation(len(s))
        Nsample = 50
        s_comb = np.stack([s[perm[:Nsample]], spred[perm[:Nsample]]]).ravel('F')
        u_comb = np.stack([u[perm[:Nsample]], upred[perm[:Nsample]]]).ravel('F')
    else:
        s_comb = np.stack([s[track_idx], spred[track_idx]]).ravel('F')
        u_comb = np.stack([u[track_idx], upred[track_idx]]).ravel('F')

    for i in range(0, len(s_comb), 2):
        ax.plot(s_comb[i:i+2], u_comb[i:i+2], 'k-', linewidth=0.8)
    ax.set_xlabel("S", fontsize=18)
    ax.set_ylabel("U", fontsize=18)

    handles, labels = ax.get_legend_handles_labels()
    lgd = fig.legend(handles,
                     labels,
                     fontsize=15,
                     markerscale=5,
                     ncol=4,
                     bbox_to_anchor=(0.0, 1.0, 1.0, 0.25),
                     loc='center')
    fig.suptitle(title)

    save_fig(fig, save, (lgd,))


def plot_cluster(X_embed, cell_labels, color_map=None, embed='umap', show_labels=True, save=None):
    """Plot the predicted cell types from the encoder

    Args:
        X_embed (:class:`numpy.ndarray`):
            2D embedding for visualization, (N,2)
        cell_labels (:class:`numpy.ndarray`):
             Cell type annotation, (N,)
        color_map (str, optional):
            User-defined colormap for cell clusters. Defaults to None.
        embed (str, optional):
            Embedding name. Used for labeling axes. Defaults to 'umap'.
        show_labels (bool, optional):
            Whether to add cell cluster names to the plot. Defaults to True.
        save (str, optional):
            Figure name for saving (including path). Defaults to None.
    """
    cell_types = np.unique(cell_labels)
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
    x = X_embed[:, 0]
    y = X_embed[:, 1]
    x_range = x.max()-x.min()
    y_range = y.max()-y.min()
    colors = get_colors(len(cell_types), color_map)

    n_char_max = np.max([len(x) for x in cell_types])
    for i, typei in enumerate(cell_types):
        mask = cell_labels == typei
        xbar, ybar = np.mean(x[mask]), np.mean(y[mask])
        ax.plot(x[mask], y[mask], '.', color=colors[i % len(colors)])
        n_char = len(typei)
        if show_labels:
            txt = ax.text(xbar - x_range*4e-3*n_char, ybar - y_range*4e-3, typei, fontsize=200//n_char_max, color='k')
            txt.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='black'))

    ax.set_xlabel(f'{embed} 1')
    ax.set_ylabel(f'{embed} 2')

    save_fig(fig, save)


def plot_train_loss(loss, iters, save=None):
    """Plots the training loss values versus iteration numbers.

    Args:
        loss (array like):
            Loss values.
        iters (array like):
            Iteration numbers.
        save (str, optional):
            Figure name for saving (including path). Defaults to None.
    """
    fig, ax = plt.subplots(facecolor='white')
    ax.plot(iters, loss, '.-')
    ax.set_title("Training Loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")

    save_fig(fig, save)


def plot_test_loss(loss, iters, save=None):
    """Plots the validation loss values versus iteration numbers.

    Args:
        loss (array like):
            Loss values.
        iters (array like):
            Iteration numbers.
        save (str, optional):
            Figure name for saving (including path). Defaults to None.
    """
    fig, ax = plt.subplots(facecolor='white')
    ax.plot(iters, loss, '.-')
    ax.set_title("Testing Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    save_fig(fig, save)


def cellwise_vel(adata,
                 key,
                 gidx,
                 plot_indices,
                 dt=0.2,
                 plot_raw=False,
                 u0=None,
                 s0=None,
                 t0=None,
                 save=None):
    """Plots u and s vs. time and velocity arrows for a subset of cells.
    Used for debugging.

    Args:
        adata (:class:`anndata.AnnData`):
            AnnData Object.
        key (str):
            Key for extracting inferred latent variables and parameters.
        gidx (int):
            Index of the gene to plot.
        plot_indices (:class:`numpy.ndarray`):
            Indices of cells for velocity quiver plot.
        dt (float, optional):
            Time interval to compute displacement of u and s. Defaults to 0.2.
        plot_raw (bool, optional):
            Whether to plot raw count data. Defaults to False.
        u0 (:class:`numpy.ndarray`, optional):
            Unspliced initial conditions. Defaults to None.
        s0 (:class:`numpy.ndarray`, optional):
            Spliced initial conditions. Defaults to None.
        t0 (:class:`numpy.ndarray`, optional):
            Time at initial conditions. Defaults to None.
        save (str, optional):
            Figure name for saving (including path). Defaults to None.
    """
    fig, ax = plt.subplots(1, 2, figsize=(18, 6), facecolor='white')

    u = np.array(adata.layers["unspliced"][:, gidx].todense()).squeeze()
    s = np.array(adata.layers["spliced"][:, gidx].todense()).squeeze()
    t = adata.obs[f'{key}_time'].to_numpy()
    if u0 is None:
        u0 = adata.layers[f'{key}_u0'][:, gidx]
    if s0 is None:
        s0 = adata.layers[f'{key}_s0'][:, gidx]
    if t0 is None:
        t0 = adata.obs[f'{key}_t0'].to_numpy()

    uhat = adata.layers[f'{key}_uhat'][:, gidx]
    shat = adata.layers[f'{key}_shat'][:, gidx]
    scaling = adata.var[f'{key}_scaling'].to_numpy()[gidx]
    rho = adata.layers[f'{key}_rho'][:, gidx]
    try:
        alpha = adata.var[f'{key}_alpha'].to_numpy()[gidx]
        beta = adata.var[f'{key}_beta'].to_numpy()[gidx]
    except KeyError:
        alpha = np.exp(adata.var[f'{key}_logmu_alpha'].to_numpy()[gidx])
        beta = np.exp(adata.var[f'{key}_logmu_beta'].to_numpy()[gidx])
    vu = rho * alpha - beta * uhat / scaling
    v = adata.layers[f'{key}_velocity'][:, gidx]
    ax[0].plot(t, uhat/scaling, '.', color='grey', alpha=0.1)
    ax[1].plot(t, shat, '.', color='grey', alpha=0.1)
    if plot_raw:
        ax[0].plot(t[plot_indices], u[plot_indices], 'o', color='b', label="Raw Count")
        ax[1].plot(t[plot_indices], s[plot_indices], 'o', color='b')
    if dt > 0:
        ax[0].quiver(t[plot_indices],
                     uhat[plot_indices]/scaling,
                     dt*np.ones((len(plot_indices),)),
                     vu[plot_indices]*dt,
                     angles='xy')
        ax[1].quiver(t[plot_indices],
                     shat[plot_indices],
                     dt*np.ones((len(plot_indices),)),
                     v[plot_indices]*dt,
                     angles='xy')
    for i, k in enumerate(plot_indices):
        if i == 0:
            ax[0].plot([t0[k], t[k]], [u0[k]/scaling, uhat[k]/scaling], 'r-o', label='Prediction')
        else:
            ax[0].plot([t0[k], t[k]], [u0[k]/scaling, uhat[k]/scaling], 'r-o')
        ax[1].plot([t0[k], t[k]], [s0[k], shat[k]], 'r-o')
        if plot_raw:
            ax[0].plot(t[k]*np.ones((2,)), [min(u[k], uhat[k]/scaling), max(u[k], uhat[k]/scaling)], 'b--')
            ax[1].plot(t[k]*np.ones((2,)), [min(s[k], shat[k]), max(s[k], shat[k])], 'b--')

    ax[0].set_ylabel("U", fontsize=16)
    ax[1].set_ylabel("S", fontsize=16)
    fig.suptitle(adata.var_names[gidx], fontsize=30)
    fig.legend(loc=1, fontsize=18)
    plt.tight_layout()

    save_fig(fig, save)


def cellwise_vel_embedding(adata, key, type_name=None, idx=None, embed='umap', save=None):
    """Plots velocity of a subset of cells on an embedding space.
    Used for debugging.

    Args:
        adata (:class:`anndata.AnnData`):
            AnnData object.
        key (str):
            Key for retreiving parameters and data for plotting.
        type_name (str, optional):
            Specific cell type to plot. Defaults to None.
        idx (:class:`numpy.ndarray`, optional):
            Indices of cells for plotting. Defaults to None.
            When set to None, cells will be randomly sampled.
        embed (str, optional):
            Embedding velocity is computed upon. Defaults to 'umap'.
        save (str, optional):
            Figure name for saving (including path). Defaults to None.

    Returns:
        :class:`numpy.ndarray`:
            Indices of cells for plotting.
    """
    if f'{key}_velocity_graph' not in adata.uns:
        print("Please run 'velocity_graph' and 'velocity_embedding' first!")
        return

    A = np.array(adata.uns[f"{key}_velocity_graph"].todense())
    A_neg = np.array(adata.uns[f"{key}_velocity_graph_neg"].todense())
    v_embed = adata.obsm[f'{key}_velocity_{embed}']
    X_embed = adata.obsm[f'X_{embed}']
    x_embed_1, x_embed_2 = X_embed[:, 0], X_embed[:, 1]

    if idx is None:
        if type_name is None:
            idx = np.random.choice(X_embed.shape[0])
        else:
            cell_labels = adata.obs["clusters"].to_numpy()
            idx = np.random.choice(np.where(cell_labels == type_name)[0])

    neighbors = np.where((A[idx] > 0) | (A_neg[idx] < 0))[0]
    t = adata.obs[f'{key}_time'].to_numpy()

    fig, ax = plt.subplots(1, 2, figsize=(24, 9), facecolor='white')
    ax[0].plot(x_embed_1, x_embed_2, '.', color='grey', alpha=0.25)
    tmask = t[neighbors] > t[idx]
    ax[0].plot(x_embed_1[neighbors[~tmask]], x_embed_2[neighbors[~tmask]], 'c.', label="Earlier Neighbors")
    ax[0].plot(x_embed_1[neighbors[tmask]], x_embed_2[neighbors[tmask]], 'b.', label="Later Neighbors")
    ax[0].plot(x_embed_1[[idx]], x_embed_2[[idx]], 'ro', label="Target Cell")
    ax[0].legend(loc=1)

    corr = A[idx, neighbors]+A_neg[idx, neighbors]
    _plot_heatmap(ax[1], corr, X_embed[neighbors], 'Cosine Similarity', markersize=80)
    ax[1].quiver(x_embed_1[[idx]], x_embed_2[[idx]], [v_embed[idx, 0]], [v_embed[idx, 1]], angles='xy')
    ax[1].plot(x_embed_1[[idx]], x_embed_2[[idx]], 'ks', markersize=10, label="Target Cell")

    save_fig(fig, save)
    return idx


#########################################################################
# Velocity quiver plot on the phase portrait
# Reference:
# Shengyu Li#, Pengzhi Zhang#, Weiqing Chen, Lingqun Ye, 
# Kristopher W. Brannan, Nhat-Tu Le, Jun-ichi Abe, John P. Cooke, 
# Guangyu Wang. A relay velocity model infers cell-dependent RNA velocity. 
# Nature Biotechnology (2023) https://doi.org/10.1038/s41587-023-01728-5
#########################################################################
def pick_grid_points(x, grid_size=(30, 30), percentile=25):
    def gaussian_kernel(X, mu=0, sigma=1):
        return np.exp(-(X - mu)**2 / (2*sigma**2)) / np.sqrt(2*np.pi*sigma**2)
    grs = []
    for dim_i in range(x.shape[1]):
        m, M = np.min(x[:, dim_i]), np.max(x[:, dim_i])
        m = m - 0.025 * np.abs(M - m)
        M = M + 0.025 * np.abs(M - m)
        gr = np.linspace(m, M, grid_size[dim_i])
        grs.append(gr)
    meshes_tuple = np.meshgrid(*grs)
    gridpoints_coordinates = np.vstack([i.flat for i in meshes_tuple]).T
    gridpoints_coordinates = gridpoints_coordinates + norm.rvs(loc=0, scale=0.15, size=gridpoints_coordinates.shape)
    
    np.random.seed(42)  # set random seed
    
    nn = NearestNeighbors()

    neighbors_1 = min((x.shape[0]-1), 20)
    nn.fit(x)
    dist, ixs = nn.kneighbors(gridpoints_coordinates, neighbors_1)

    ix_choice = ixs[:, 0].flat[:]
    ix_choice = np.unique(ix_choice)

    nn = NearestNeighbors()

    neighbors_2 = min((x.shape[0]-1), 20)
    nn.fit(x)
    dist, ixs = nn.kneighbors(x[ix_choice], neighbors_2)

    density_extimate = gaussian_kernel(dist, mu=0, sigma=0.5).sum(1)
    bool_density = density_extimate > np.percentile(density_extimate, percentile)
    ix_choice = ix_choice[bool_density]
    return ix_choice


def plot_phase_vel(adata,
                   gene,
                   key,
                   dt=0.05,
                   grid_size=(30, 30),
                   percentile=25,
                   save=None):
    """Plots RNA velocity stream on a phase portrait.

    Args:
        adata (:class:`anndata.AnnData`):
            AnnData object
        gene (str):
            Gene name.
        key (str):
            Key for retreiving data.
        dt (float, optional):
            Time interval used to compute u and s displacement.
            Defaults to 0.05.
        grid_size (tuple[int], optional):
            Number of rows and columns for grid points
            on which velocity will be computed based on KNN interpolation.
            Defaults to (30, 30).
        percentile (int, optional):
            Hyperparameter for grid point picking. Defaults to 25.
        save (str, optional):
            Figure name for saving (including path). Defaults to None.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    gidx = np.where(adata.var_names == gene)[0][0]
    scaling = adata.var[f'{key}_scaling'].iloc[gidx]
    t = adata.obs[f'{key}_time'].to_numpy()
    vu = adata.layers[f'{key}_velocity_u'][:, gidx]
    vs = adata.layers[f'{key}_velocity'][:, gidx]
    u = adata.layers['Mu'][:, gidx]/scaling
    s = adata.layers['Ms'][:, gidx]
    x = np.stack([s, u]).T
    _plot_heatmap(ax, t, x, 'time')
    grid_points = pick_grid_points(x, grid_size, percentile)
    ax.quiver(s[grid_points],
              u[grid_points],
              dt*vs[grid_points],
              dt*vu[grid_points],
              angles='xy',
              scale=None,
              scale_units='inches',
              headwidth=5.0,
              headlength=8.0,
              color='k')
    save_fig(fig, save)


def plot_velocity(X_embed, vx, vy, save=None):
    """2D quiver plot of velocity

    Args:
        X_embed (:class:`numpy.ndarray`):
            2D coordinates for visualization, (N, 2)
        vx (:class:`numpy.ndarray`):
            Velocity in the x direction.
        vy (:class:`numpy.ndarray`):
            Velocity in the y direction.
        save (str, optional):
            Figure name for saving (including path). Defaults to None.
    """
    umap1, umap2 = X_embed[:, 0], X_embed[:, 1]
    fig, ax = plt.subplots(figsize=(12, 8))
    v = np.sqrt(vx**2+vy**2)
    vmax, vmin = np.quantile(v, 0.95), np.quantile(v, 0.05)
    v = np.clip(v, vmin, vmax)
    ax.plot(umap1, umap2, '.', alpha=0.5)
    ax.quiver(umap1, umap2, vx, vy, v, angles='xy')

    save_fig(fig, save)

#########################################################################
# Evaluation Plots
#########################################################################


def plot_legend(adata, cluster_key='clusters', ncol=1, save='figures/legend.png'):
    """Plots figure legend containing all cell types.

    Args:
        adata (:class:`anndata.AnnData`):
            AnnData object.
        cluster_key (str, optional):
            Key for cell type annotations. Defaults to 'clusters'.
        ncol (int, optional):
            Number of columns of the legend. Defaults to 1.
        save (str, optional):
            Figure name for saving (including path). Defaults to 'figures/legend.png'.
    """
    cell_labels = adata.obs[cluster_key].to_numpy()
    cell_labels = np.array([str(x) for x in cell_labels])
    cell_types = np.unique(cell_labels)
    colors = get_colors(len(cell_types))
    lines = []

    fig, ax = plt.subplots()
    for i, x in enumerate(cell_types):
        line = ax.plot([], 'o', color=colors[i], label=x)
        lines.append(line)
    ax.axis("off")
    plt.tight_layout()
    lgd = ax.legend(markerscale=2.0, ncol=ncol, fontsize=20, loc='center', frameon=False)
    save_fig(fig, save, (lgd,))


def _plot_heatmap(ax,
                  vals,
                  X_embed,
                  colorbar_name,
                  colorbar_ticklabels=None,
                  markersize=5,
                  cmap='plasma',
                  show_color_bar=True,
                  axis_off=True):
    """General heatmap plotting helper function.
    """
    ax.scatter(X_embed[:, 0],
               X_embed[:, 1],
               s=markersize,
               c=vals,
               cmap=cmap,
               edgecolors='none')
    vmin = np.quantile(vals, 0.01)
    vmax = np.quantile(vals, 0.99)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(sm, ax=ax)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel(colorbar_name, rotation=270, fontsize=24)
    if colorbar_ticklabels is not None:
        if len(colorbar_ticklabels) == 2:
            cbar.ax.get_yaxis().labelpad = 5
        cbar.set_ticks(np.linspace(vmin, vmax, len(colorbar_ticklabels)))
        cbar.ax.set_yticklabels(colorbar_ticklabels, fontsize=15)
    if axis_off:
        ax.axis("off")

    return ax


def histeq(x, perc=0.95, Nbin=101):
    x_ub = np.quantile(x, perc)
    x_lb = x.min()
    delta_x = (x_ub - x_lb)/(Nbin-1)
    bins = [x_lb+i*delta_x for i in range(Nbin)]+[x.max()]
    pdf_x, edges = np.histogram(x, bins, density=True)
    px, edges = np.histogram(x, bins, density=False)

    # Perform histogram equalization
    cdf = np.concatenate(([0], np.cumsum(px)))
    cdf = cdf/cdf[-1]
    x_out = np.zeros((len(x)))
    for i in range(Nbin):
        mask = (x >= bins[i]) & (x < bins[i+1])
        x_out[mask] = (cdf[i] + (x[mask]-bins[i])*pdf_x[i])*np.abs(x.max())
    return x_out


def plot_heatmap(vals,
                 X_embed,
                 colorbar_name="Latent Time",
                 colorbar_ticks=None,
                 save=None):
    """Plots a quantity as a heatmap.

    Args:
        vals (:class:`numpy.ndarray`):
            Values to be plotted as a heatmap, (N,).
        X_embed (:class:`numpy.ndarray`):
            2D coordinates for visualization, (N,2).
        colorbar_name (str, optional):
            Name shown next to the colorbar. Defaults to "Latent Time".
        colorbar_ticks (str, optional):
            Name shown on the colorbar axis. Defaults to None.
        save (str, optional):
            Figure name for saving (including path). Defaults to None.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax = _plot_heatmap(ax, vals, X_embed, colorbar_name, colorbar_ticks, axis_off=True)
    save_fig(fig, save)


def plot_time(t_latent,
              X_embed,
              cmap='plasma',
              legend_label='Latent Time',
              save=None):
    """Plots mean cell time as a heatmap.

    Args:
        t_latent (`numpy.ndarray`):
            Mean latent time, (N,)
        X_embed (`numpy.ndarray`):
            2D coordinates for visualization, (N,2)
        cmap (str, optional):
            Colormap name. Defaults to 'plasma'.
        legend_label (str, optional):
            Text added next to the color bar. Defaults to 'Latent Time'.
        save (str, optional):
            Figure name for saving (including path). Defaults to None.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    _plot_heatmap(ax, t_latent, X_embed, legend_label, ['early', 'late'], cmap=cmap, axis_off=True)
    save_fig(fig, save)


def plot_time_var(std_t,
                  X_embed,
                  t=None,
                  hist_eq=True,
                  cmap='viridis',
                  save=None):
    """Plots cell time coefficient of variation as a heatmap.

    Args:
        std_t (:class:`numpy.ndarray`):
            Standard deviation of latent time, (N,)
        X_embed (:class:`numpy.ndarray`):
            2D coordinates for visualization, (N,2)
        t (:class:`numpy.ndarray`, optional):
            Mean latent time. Defaults to None.
        hist_eq (bool, optional):
            Whether to perform histogram equalization. Defaults to True.
        cmap (str, optional):
            Colormap name. Defaults to 'viridis'.
        save (str, optional):
            Figure name for saving (including path). Defaults to None.
    """
    t_norm = np.ones((std_t.shape)) if t is None else np.abs(t) + 1e-10
    diff_entropy = np.log(std_t/t_norm)+0.5*(1+np.log(2*np.pi))
    if hist_eq:
        diff_entropy = histeq(diff_entropy, Nbin=len(diff_entropy)//50)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax = _plot_heatmap(ax, diff_entropy, X_embed, "Time Variance", ['low', 'high'], cmap=cmap, axis_off=True)
    save_fig(fig, save)


def plot_state_var(std_z,
                   X_embed,
                   z=None,
                   hist_eq=True,
                   cmap='viridis',
                   save=None):
    """Plots cell state variance (in the form of coefficient of variation) as a heatmap.

    Args:
        std_z (:class:`numpy.ndarray`):
            Standard deviation of cell state, assuming diagonal covariance, (N, dim z)
        X_embed (:class:`numpy.ndarray`):
            2D coordinates for visualization, (N,2)
        z (:class:`numpy.ndarray`, optional):
            Mean cell state, (N, dim z). Defaults to None.
        hist_eq (bool, optional):
            Whether to perform histogram equalization. Defaults to True.
        cmap (str, optional):
            Colormap name. Defaults to 'viridis'.
        save (str, optional):
            Figure name for saving (including path). Defaults to None.
    """
    z_norm = np.ones((std_z.shape)) if z is None else np.linalg.norm(z, axis=1).reshape(-1, 1) + 1e-10
    diff_entropy = np.sum(np.log(std_z/z_norm), 1) + 0.5*std_z.shape[1]*(1+np.log(2*np.pi))
    if hist_eq:
        diff_entropy = histeq(diff_entropy, Nbin=len(diff_entropy)//50)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax = _plot_heatmap(ax, diff_entropy, X_embed, "State Uncertainty", ['low', 'high'], cmap=cmap, axis_off=True)
    save_fig(fig, save)


def plot_phase_axis(ax,
                    u,
                    s,
                    marker='.',
                    a=1.0,
                    D=1,
                    labels=None,
                    legends=None,
                    title=None,
                    show_legend=False,
                    label_fontsize=30,
                    color_map=None):
    """Plot phase in a subplot of a figure."""
    try:
        if labels is None:
            ax.plot(s[::D], u[::D], marker, color='k')
        elif legends is None:
            types = np.unique(labels)
            colors = get_colors(len(types), color_map)
            for type_int in types:
                mask = labels == type_int
                if np.any(mask):
                    ax.plot(s[mask][::D],
                            u[mask][::D],
                            marker,
                            color=colors[type_int % len(colors)],
                            alpha=a)
        else:
            colors = get_colors(len(legends), color_map)
            for type_int in range(len(legends)):  # type_int: label index, labels are cell types
                mask = labels == type_int
                if np.any(mask):
                    if show_legend:
                        ax.plot(s[mask][::D],
                                u[mask][::D],
                                marker,
                                color=colors[type_int % len(colors)],
                                alpha=a,
                                label=legends[type_int])
                    else:
                        ax.plot(s[mask][::D],
                                u[mask][::D],
                                marker,
                                color=colors[type_int % len(colors)],
                                alpha=a)
                elif show_legend:
                    ax.plot([np.nan],
                            [np.nan],
                            marker,
                            color=colors[type_int % len(colors)],
                            alpha=a,
                            label=legends[type_int])
    except TypeError:
        return ax

    if title is not None:
        ax.set_title(title, fontsize=30)

    return ax


def plot_phase_grid(Nr,
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
                    H=3,
                    alpha=0.2,
                    downsample=1,
                    legend_fontsize=None,
                    color_map=None,
                    path='figures',
                    figname=None,
                    format='png',
                    **kwargs):
    """Plot the phase portrait of a list of genes in an [Nr x Nc] grid.
    Cells are colored according to their dynamical state or cell type.
    If Nr*Nc < number of genes, the last few genes will be ignored.
    If Nr*Nc > number of genes, the last few subplots will be blank.

    Args:
        Nr (int):
            Number of rows of the grid plot.
        Nc (int):
            Number of columns of the grid plot.
        gene_list (list[str]):
            Genes to plot.
        U (:class:`numpy.ndarray`):
            Unspliced count matrices. The gene dimension should equal len(gene_list).
        S (:class:`numpy.ndarray`):
            Spliced count matrices. The gene dimension should equal len(gene_list).
        Labels (dict):
            Keys are method names and values are (N) cell annotations
            For the regular ODE, this can be induction/repression annotation.
            Otherwise, it's usually just the cell type annotation.
        Legends (dict):
            Keys are method names and values are the legend names to show.
            If the labels are phase labels, then the legends are usually
            {'off', induction', 'repression'}.
            If the labels are cell type annotations, the legends will be the unique
            cell type names.
        Uhat (dict, optional):
            Predicted unspliced counts.
            Keys are method names and values are arrays of size (N_pred, N_gene).
            Notice that N_pred is not necessarily the number of cells.
            This could happen if we want to save computational cost and evaluate
            the ODE just at a user-defined number of time points. Defaults to {}.
        Shat (dict, optional):
            Predicted spliced counts, similar to Uhat. Defaults to {}.
        Labels_demo (dict, optional):
            Keys are method names and values are arrays of size (N_pred).
            This is the annotation for the predictions.. Defaults to {}.
        W (int, optional):
            Width of a subplot. Defaults to 6.
        H (int, optional):
            Height of a subplot. Defaults to 3.
        alpha (float, optional):
            Transparency of the data points. Defaults to 0.2.
        downsample (int, optional):
            Down-sampling factor to display the data points.. Defaults to 1.
        legend_fontsize (int/float, optional):
            Defaults to None.
        color_map (str, optional):
            User-defined colormap for cell labels. Defaults to None.
        path (str, optional):
            Path to the saved figure. Defaults to 'figures'.
        figname (_type_, optional):
            Name of the saved figure, without format at the end. Defaults to None.
        format (str, optional):
            Figure format. Defaults to 'png'.

    """
    D = downsample
    methods = list(Uhat.keys())

    M = max(1, len(methods))

    # Detect whether multiple figures are needed
    Nfig = len(gene_list) // (Nr*Nc)
    if Nfig*Nr*Nc < Nfig:
        Nfig += 1

    label_fontsize = W*H
    for i_fig in range(Nfig):
        fig_phase, ax_phase = plt.subplots(Nr, M*Nc, figsize=(W * M * Nc + 1.0, H * Nr), facecolor='white')
        if Nr == 1 and M * Nc == 1:  # Single Gene, Single Method
            labels = Labels[methods[0]]
            if labels is not None:
                if labels.ndim == 2:
                    labels = labels[:, i_fig]
            title = f"{gene_list[i_fig]} (VeloVAE)" if methods[0] == "FullVB" else f"{gene_list[i_fig]} ({methods[0]})"
            ax_phase = plot_phase_axis(ax_phase,
                                       U[:, i_fig],
                                       S[:, i_fig],
                                       '.',
                                       alpha,
                                       D,
                                       labels,
                                       Legends[methods[0]],
                                       title,
                                       show_legend=True,
                                       color_map=color_map)
            try:
                ax_phase = plot_phase_axis(ax_phase,
                                           Uhat[methods[0]][:, i_fig],
                                           Shat[methods[0]][:, i_fig],
                                           '.',
                                           1.0,
                                           1,
                                           Labels_demo[methods[0]],
                                           Legends[methods[0]],
                                           title,
                                           show_legend=False,
                                           color_map=color_map)
            except (KeyError, TypeError):
                print("[** Warning **]: Skip plotting the prediction because of key value error or invalid data type.")
                pass
            ax_phase.set_xlabel("S", fontsize=label_fontsize)
            ax_phase.set_ylabel("U", fontsize=label_fontsize)
        elif Nr == 1:  # Single Gene, Multiple Method
            for j in range(min(Nc, len(gene_list) - i_fig * Nc)):  # gene
                for k, method in enumerate(methods):  # method
                    labels = Labels[method]
                    if labels is not None:
                        if labels.ndim == 2:
                            labels = labels[:, i_fig*Nc+j]
                    title = (f"{gene_list[i_fig*Nc+j]} (VeloVAE)"
                             if methods[0] == "FullVB" else
                             f"{gene_list[i_fig*Nc+j]} ({method})")
                    ax_phase[M*j+k] = plot_phase_axis(ax_phase[M*j+k],
                                                      U[:, i_fig*Nc+j],
                                                      S[:, i_fig*Nc+j],
                                                      '.',
                                                      alpha,
                                                      D,
                                                      labels,
                                                      Legends[method],
                                                      title,
                                                      show_legend=True,
                                                      color_map=color_map)
                    try:
                        ax_phase[M*j+k] = plot_phase_axis(ax_phase[M*j+k],
                                                          Uhat[method][:, i_fig*Nc+j],
                                                          Shat[method][:, i_fig*Nc+j],
                                                          '.',
                                                          1.0,
                                                          1,
                                                          Labels_demo[method],
                                                          Legends[method],
                                                          title,
                                                          show_legend=False,
                                                          color_map=color_map)
                    except (KeyError, TypeError):
                        print("[** Warning **]: "
                              "Skip plotting the prediction because of key value error or invalid data type.")
                        pass
                    ax_phase[M*j+k].set_xlabel("S", fontsize=label_fontsize)
                    ax_phase[M*j+k].set_ylabel("U", fontsize=label_fontsize)
        elif M * Nc == 1:  # Multiple Gene, Single Method
            for i in range(min(Nr, len(gene_list) - i_fig * Nr)):
                labels = Labels[methods[0]]
                if labels is not None:
                    if labels.ndim == 2:
                        labels = labels[:, i_fig * Nr + i]
                title = (f"{gene_list[i_fig*Nr+i]} (VeloVAE)"
                         if methods[0] == "FullVB" else
                         f"{gene_list[i_fig*Nr+i]} ({methods[0]})")
                ax_phase[i] = plot_phase_axis(ax_phase[i],
                                              U[:, i_fig*Nr+i],
                                              S[:, i_fig*Nr+i],
                                              '.',
                                              alpha,
                                              D,
                                              labels,
                                              Legends[methods[0]],
                                              title,
                                              show_legend=True,
                                              color_map=color_map)
                try:
                    ax_phase[i] = plot_phase_axis(ax_phase[i],
                                                  Uhat[methods[0]][:, i_fig*Nr+i],
                                                  Shat[methods[0]][:, i_fig*Nr+i],
                                                  '.',
                                                  1.0,
                                                  1,
                                                  Labels_demo[methods[0]],
                                                  Legends[methods[0]],
                                                  title,
                                                  show_legend=False,
                                                  color_map=color_map)
                except (KeyError, TypeError):
                    print("[** Warning **]: "
                          "Skip plotting the prediction because of key value error or invalid data type.")
                    pass
                ax_phase[i].set_xlabel("S", fontsize=label_fontsize)
                ax_phase[i].set_ylabel("U", fontsize=label_fontsize)
        else:
            for i in range(Nr):
                for j in range(Nc):  # i, j: row and column gene index
                    idx = i_fig * Nc * Nr + i * Nc + j
                    if idx >= len(gene_list):
                        break
                    for k, method in enumerate(methods):
                        labels = Labels[method]
                        if labels is not None:
                            if labels.ndim == 2:
                                labels = labels[:, idx]
                        title = (f"{gene_list[idx]} (VeloVAE)"
                                 if methods[0] == "FullVB" else
                                 f"{gene_list[idx]} ({method})")
                        ax_phase[i, M * j + k] = plot_phase_axis(ax_phase[i, M * j + k],
                                                                 U[:, idx],
                                                                 S[:, idx],
                                                                 '.',
                                                                 alpha,
                                                                 D,
                                                                 labels,
                                                                 Legends[method],
                                                                 title,
                                                                 show_legend=True,
                                                                 color_map=color_map)
                        try:
                            ax_phase[i, M * j + k] = plot_phase_axis(ax_phase[i, M * j + k],
                                                                     Uhat[method][:, idx],
                                                                     Shat[method][:, idx],
                                                                     '.',
                                                                     1.0,
                                                                     1,
                                                                     Labels_demo[method],
                                                                     Legends[method],
                                                                     title,
                                                                     show_legend=False,
                                                                     color_map=color_map)
                        except (KeyError, TypeError):
                            print("[** Warning **]:"
                                  "Skip plotting the prediction because of key value error or invalid data type.")
                            pass
                        ax_phase[i, M * j + k].set_xlabel("S", fontsize=label_fontsize)
                        ax_phase[i, M * j + k].set_ylabel("U", fontsize=label_fontsize)
        if Nr == 1 and M*Nc == 1:
            handles, labels = ax_phase.get_legend_handles_labels()
        elif ax_phase.ndim == 1:
            handles, labels = ax_phase[0].get_legend_handles_labels()
        else:
            handles, labels = ax_phase[0, 0].get_legend_handles_labels()
        n_label = len(Legends[methods[0]])

        l_indent = 1 - 0.02/Nr
        if legend_fontsize is None:
            legend_fontsize = min(int(10*Nr), 300*Nr/n_label)
        lgd = fig_phase.legend(handles,
                               labels,
                               fontsize=legend_fontsize,
                               markerscale=5.0,
                               bbox_to_anchor=(-0.03/Nc, l_indent),
                               loc='upper right')

        fig_phase.subplots_adjust(hspace=0.3, wspace=0.12)
        fig_phase.tight_layout()

        save = None if (path is None or figname is None) else f'{path}/{figname}_phase_{i_fig+1}.{format}'
        save_fig(fig_phase, save, (lgd,))


def sample_scatter_plot(x, down_sample, n_bins=20):
    """Sample cells for a scatter plot."""
    idx_downsample = []
    n_sample = max(1, len(x)//down_sample)
    if n_bins > n_sample:
        n_bins = n_sample
    sample_per_bin = n_sample // n_bins
    n_res = n_sample - sample_per_bin * n_bins

    edges = np.linspace(x.min(), x.max(), n_bins+1)

    for i in range(len(edges)-1):
        idx_pool = np.where((x >= edges[i]) & (x <= edges[i+1]))[0]
        if len(idx_pool) > sample_per_bin + int(i+1 <= n_res):
            idx_downsample.extend(np.random.choice(idx_pool, sample_per_bin+int(i + 1 <= n_res)))
        else:  # fewer samples in the bin than needed
            idx_downsample.extend(idx_pool)

    return np.array(idx_downsample).astype(int)


def plot_sig_axis(ax,
                  t,
                  x,
                  labels=None,
                  legends=None,
                  marker='.',
                  a=1.0,
                  D=1,
                  show_legend=False,
                  color_map=None,
                  title=None):
    """Plot a modality versus time in a subplot."""
    lines = []
    if labels is None or legends is None:
        lines.append(ax.plot(t[::D], x[::D], marker, markersize=5, color='k', alpha=a)[0])
    else:
        colors = get_colors(len(legends), color_map)
        for i in range(len(legends)):
            mask = labels == i
            if np.any(mask):
                idx_sample = sample_scatter_plot(x[mask], D)
                if show_legend:
                    line = ax.plot(t[mask][idx_sample],
                                   x[mask][idx_sample],
                                   marker,
                                   markersize=5,
                                   color=colors[i % len(colors)],
                                   alpha=a,
                                   label=legends[i])[0]
                    lines.append(line)
                else:
                    ax.plot(t[mask][idx_sample],
                            x[mask][idx_sample],
                            marker,
                            markersize=5,
                            color=colors[i % len(colors)],
                            alpha=a)
    ymin, ymax = ax.get_ylim()
    ax.set_yticks([0, ymax])
    if title is not None:
        ax.set_title(title, fontsize=30)
    return lines


def plot_sig_pred_axis(ax,
                       t,
                       x,
                       labels=None,
                       legends=None,
                       marker='.',
                       a=1.0,
                       D=1,
                       show_legend=False,
                       title=None):
    """Plot predicted modality versus time in a subplot."""
    if labels is None or legends is None:
        ax.plot(t[::D], x[::D], marker, linewidth=5, color='k', alpha=a)
    else:
        for i in range(len(legends)):
            mask = labels == i
            if np.any(mask):
                idx_ordered = np.argsort(t[mask][::D])
                if show_legend:
                    ax.plot(t[mask][::D][idx_ordered],
                            x[mask][::D][idx_ordered],
                            marker,
                            linewidth=5,
                            color='k',
                            alpha=a,
                            label=legends[i])
                else:
                    ax.plot(t[mask][::D][idx_ordered],
                            x[mask][::D][idx_ordered],
                            marker,
                            linewidth=5,
                            color='k',
                            alpha=a)
    ymin, ymax = ax.get_ylim()
    ax.set_yticks([0, ymax])
    if title is not None:
        ax.set_title(title, fontsize=30)
    return ax


def plot_sig_loess_axis(ax,
                        t,
                        x,
                        labels,
                        legends,
                        frac=0.5,
                        a=1.0,
                        D=1,
                        show_legend=False,
                        title=None,):
    """LOESS plot in a subplot."""
    from loess import loess_1d
    for i in range(len(legends)):
        mask = labels == i
        if np.any(mask):
            t_lb, t_ub = np.quantile(t[mask], 0.05), np.quantile(t[mask], 0.95)
            mask2 = (t <= t_ub) & (t >= t_lb) & mask
            if np.sum(mask2) >= 20:
                tout, xout, wout = loess_1d.loess_1d(t[mask2],
                                                     x[mask2],
                                                     xnew=None,
                                                     degree=1,
                                                     frac=frac,
                                                     npoints=None,
                                                     rotate=False,
                                                     sigy=None)
                torder = np.argsort(tout)
                if show_legend:
                    ax.plot(tout[torder][::D], xout[torder][::D], 'k-', linewidth=5, alpha=a, label=legends[i])
                else:
                    ax.plot(tout[torder][::D], xout[torder][::D], 'k-', linewidth=5, alpha=a)
    ymin, ymax = ax.get_ylim()
    ax.set_yticks([0, ymax])
    if title is not None:
        ax.set_title(title, fontsize=30)
    return ax


def sample_quiver_plot(t, dt, x=None, n_bins=3):
    """Sample cells for a velocity quiver plot."""
    tmax, tmin = t.max()+1e-3, np.quantile(t, 0.01)
    Nbin = int(np.clip((tmax-tmin)/dt, 1, len(t)//2))
    indices = []
    for i in range(Nbin):
        batch_idx = np.where((t >= tmin + i * dt) & (t <= tmin + (i+1) * dt))[0]
        if len(batch_idx) > 0:
            if x is None:
                indices.append(batch_idx[len(batch_idx)//2])
            else:
                edges = np.linspace(np.quantile(x[batch_idx], 0.1), np.quantile(x[batch_idx], 0.9), n_bins+1)
                for j in range(n_bins):
                    mask = (x[batch_idx] >= edges[j]) & (x[batch_idx] <= edges[j+1])
                    if np.any(mask):
                        indices.append(np.random.choice(batch_idx[mask]))
    return np.array(indices).astype(int)


def plot_vel_axis(ax,
                  t,
                  x,
                  v,
                  labels=None,
                  legends=None,
                  dt=0.1,
                  a=1.0,
                  show_legend=False,
                  sparsity_correction=False,
                  color_map=None,
                  title=None):
    """Velocity quiver plot on a u/s-t subplot."""
    if labels is None or legends is None:
        dt_sample = (t.max()-t.min())/50
        torder = np.argsort(t)
        try:
            indices = (sample_quiver_plot(t[torder], dt_sample, x[torder], n_bins=5)
                       if sparsity_correction else
                       sample_quiver_plot(t[torder], dt_sample, n_bins=5))
        except ValueError:
            np.random.seed(42)
            indices = np.random.choice(len(t), len(t)//30, replace=False)
        if len(indices) > 0:
            ax.quiver(t[torder][indices],
                      x[torder][indices],
                      dt*np.ones((len(indices))),
                      dt*v[torder][indices],
                      angles='xy',
                      scale=None,
                      scale_units='inches',
                      headwidth=5.0,
                      headlength=8.0,
                      color='k')
    else:
        colors = get_colors(len(legends), color_map)
        for i in range(len(legends)):
            mask = labels == i
            t_type = t[mask]
            dt_sample = (t_type.max()-t_type.min())/30
            if np.any(mask):
                torder = np.argsort(t_type)
                try:
                    indices = (sample_quiver_plot(t_type[torder], dt_sample, x[mask][torder], n_bins=4)
                               if sparsity_correction else
                               sample_quiver_plot(t_type[torder], dt_sample, n_bins=4))
                except ValueError:
                    np.random.seed(42)
                    indices = np.random.choice(len(t_type), len(t_type)//30+1, replace=False)
                if len(indices) == 0:  # edge case handling
                    continue
                v_type = v[mask][torder][indices]
                v_type = np.clip(v_type, np.quantile(v_type, 0.01), np.quantile(v_type, 0.95))
                # Actual Quiver Plot
                if show_legend:
                    ax.quiver(t_type[torder][indices],
                              x[mask][torder][indices],
                              dt*np.ones((len(indices))),
                              dt*v_type,
                              label=legends[i],
                              angles='xy',
                              scale=None,
                              scale_units='inches',
                              headwidth=5.0,
                              headlength=8.0,
                              color=colors[i % len(colors)])
                else:
                    ax.quiver(t_type[torder][indices],
                              x[mask][torder][indices],
                              dt*np.ones((len(indices))),
                              dt*v_type,
                              angles='xy',
                              scale=None,
                              scale_units='inches',
                              headwidth=5.0,
                              headlength=8.0,
                              color=colors[i % len(colors)])
    ymin, ymax = ax.get_ylim()
    ax.set_yticks([0, ymax])
    return ax


def plot_sig_grid(Nr,
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
                  H=3,
                  frac=0.0,
                  alpha=1.0,
                  down_sample=1,
                  legend_fontsize=None,
                  sparsity_correction=False,
                  plot_loess=False,
                  color_map=None,
                  path='figures',
                  figname=None,
                  format='png'):
    """Plot u/s of a list of genes vs. time in an [Nr x Nc] grid of subplots.
    Cells are colored according to their dynamical state or cell type.

    Args:
        Nr (int):
            Number of rows of the grid plot.
        Nc (int):
            Number of columns of the grid plot.
        gene_list (array like):
            Genes to plot. If the length exceeds Nr*Nc, multiple figures will
            be generated. If length is less than Nr*Nc, some subplots will be
            blank.
        T (dict):
            Keys are methods (string) and values are time arrays.
            For some methods, the value is an (N,G) array
            instead of an (N) array because of local fitting.
        U (:class:`numpy.ndarray`):
            Unspliced count data.
            Contain just the genes for plotting.
        S (:class:`numpy.ndarray`):
            Spliced count data.
            Contain just the genes for plotting.
        Labels (dict):
            Keys are methods and values are arrays of cell annotation.
            Usually the values are cell type annotations.
        Legends (dict):
            Keys are methods and values are legend names.
            Usually the legend names are unique values of cell annotation.
            In our application, these are unique cell types.
        That (dict, optional):
            Keys are methods and values are (N_eval) of cell time.
            Time used in evaluation. N_eval is generally unequal to number of cells
            in the original data and the time points don't necessarily match the original
            cell because we often need fewer time points to evaluate a parametric model.
            For scVelo, the value is an (N_eval,G) array instead of an (N_eval) array
            because of local fitting. Defaults to {}.
        Uhat (dict, optional):
            Dictionary with method names as keys and arrays of predicted u as values.
            Defaults to {}.
        Shat (dict, optional):
            Dictionary with method names as keys and arrays of predicted s as values.
            Defaults to {}.
        V (dict, optional):
            Keys are methods and values are (N,G) arrays of velocity.
            Defaults to {}.
        Labels_demo (dict, optional):
            Keys are methods and values are cell type annotations of the prediction.
            Defaults to {}.
        W (int, optional):
            Subplot width. Defaults to 6.
        H (int, optional):
            Subplot height. Defaults to 3.
        frac (float, optional):
            Hyper-parameter for the LOESS plot.
            This is the window length of the local regression.
            If it's 0, LOESS will not be plotted. Defaults to 0.0.
        alpha (float, optional):
            Transparency of the data points.. Defaults to 1.0.
        down_sample (int, optional):
            Down-sampling factor to reduce the overlapping of data points. Defaults to 1.
        legend_fontsize (int, optional):
            Defaults to None.
        sparsity_correction (bool, optional):
            Whether to sample u/s uniformly in the range to avoid
            sapling most zeros in sparse expression profiles.
            Defaults to False.
        plot_loess (bool, optional):
            Whether to plot a line fit for VeloVAE. Defaults to False.
        color_map (_type_, optional):
            User-defined colormap for different cell types. Defaults to None.
        path (str, optional):
            Saving path. Defaults to 'figures'.
        figname (str, optional):
            Name if the figure.
            Because there can be multiple figures generated in this function.
            We will append a number to figname when saving the figures.
            Figures will not be saved if set to None. Defaults to None.
        format (str, optional):
            Figure format, could be png, pdf, svg, eps and ps. Defaults to 'png'.

    """
    methods = list(Uhat.keys())
    M = max(1, len(methods))

    # Detect whether multiple figures are needed
    Nfig = len(gene_list) // (Nr*Nc)
    if Nfig * Nr * Nc < len(gene_list):
        Nfig += 1

    # Plotting
    for i_fig in range(Nfig):
        fig_sig, ax_sig = plt.subplots(3 * Nr, M * Nc, figsize=(W * M * Nc + 1.0, 3 * H * Nr), facecolor='white')
        if M * Nc == 1:
            for i in range(min(Nr, len(gene_list) - i_fig * Nr)):
                idx = i_fig*Nr+i
                t = T[methods[0]][:, idx] if T[methods[0]].ndim == 2 else T[methods[0]]
                if np.any(np.isnan(t)):
                    continue
                that = That[methods[0]][:, idx] if That[methods[0]].ndim == 2 else That[methods[0]]
                title = f"{gene_list[idx]} (VeloVAE)" if methods[0] == "FullVB" else f"{gene_list[idx]} ({methods[0]})"
                plot_sig_axis(ax_sig[3*i],
                              t,
                              U[:, idx],
                              Labels[methods[0]],
                              Legends[methods[0]],
                              '.',
                              alpha,
                              down_sample,
                              True,
                              color_map=color_map,
                              title=title)
                plot_sig_axis(ax_sig[3*i+1],
                              t,
                              S[:, idx],
                              Labels[methods[0]],
                              Legends[methods[0]],
                              '.',
                              alpha,
                              down_sample,
                              color_map=color_map)

                try:
                    if ('VeloVAE' in methods[0])\
                        or ('FullVB' in methods[0])\
                            or (methods[0] in ['DeepVelo',
                                               'Discrete PyroVelocity',
                                               'PyroVelocity',
                                               'VeloVI',
                                               'cellDancer']):
                        K = min(10, max(len(that)//5000, 1))
                        
                        if frac > 0 and frac < 1:
                            plot_sig_loess_axis(ax_sig[3 * i],
                                                that[::K],
                                                Uhat[methods[0]][:, idx][::K],
                                                Labels_demo[methods[0]][::K],
                                                Legends[methods[0]],
                                                frac=frac)
                            plot_sig_loess_axis(ax_sig[3 * i + 1],
                                                that[::K],
                                                Shat[methods[0]][:, idx][::K],
                                                Labels_demo[methods[0]][::K],
                                                Legends[methods[0]],
                                                frac=frac)
                        elif 'Discrete' in methods[0]:
                            uhat_plot = np.random.poisson(Uhat[methods[0]][:, idx])
                            shat_plot = np.random.poisson(Shat[methods[0]][:, idx])
                            plot_sig_pred_axis(ax_sig[3*i], that, uhat_plot)
                            plot_sig_pred_axis(ax_sig[3*i+1], that, shat_plot)
                        plot_vel_axis(ax_sig[3 * i + 2],
                                      t,
                                      Shat[methods[0]][:, idx],
                                      V[methods[0]][:, idx],
                                      Labels[methods[0]],
                                      Legends[methods[0]],
                                      sparsity_correction=sparsity_correction,
                                      color_map=color_map)
                    else:
                        plot_sig_pred_axis(ax_sig[3*i],
                                           that,
                                           Uhat[methods[0]][:, idx],
                                           Labels_demo[methods[0]],
                                           Legends[methods[0]],
                                           '-',
                                           1.0,
                                           1)
                        plot_sig_pred_axis(ax_sig[3*i+1],
                                           that,
                                           Shat[methods[0]][:, idx],
                                           Labels_demo[methods[0]],
                                           Legends[methods[0]],
                                           '-',
                                           1.0,
                                           1)
                        plot_vel_axis(ax_sig[3*i+2],
                                      t,
                                      S[:, idx],
                                      V[methods[0]][:, idx],
                                      Labels[methods[0]],
                                      Legends[methods[0]],
                                      sparsity_correction=sparsity_correction,
                                      color_map=color_map)
                except (KeyError, TypeError):
                    print("[** Warning **]: "
                          "Skip plotting the prediction because of key value error or invalid data type.")
                    return
                if np.all(~np.isnan(t)):
                    ax_sig[3*i].set_xlim(t.min(), np.quantile(t, 0.999)+0.1)
                    ax_sig[3*i+1].set_xlim(t.min(), np.quantile(t, 0.999)+0.1)
                    ax_sig[3*i+2].set_xlim(t.min(), np.quantile(t, 0.999)+0.1)

                ax_sig[3*i].set_ylabel("U", fontsize=30, rotation=0)
                ax_sig[3*i].yaxis.set_label_coords(-0.03, 0.5)

                ax_sig[3*i+1].set_ylabel("S", fontsize=30, rotation=0)
                ax_sig[3*i+1].yaxis.set_label_coords(-0.03, 0.5)

                ax_sig[3*i+2].set_ylabel("S", fontsize=30, rotation=0)
                ax_sig[3*i+2].yaxis.set_label_coords(-0.03, 0.5)

                ax_sig[3*i].set_xticks([])
                ax_sig[3*i+1].set_xticks([])
                ax_sig[3*i+2].set_xticks([])
                ax_sig[3*i].set_yticks([])
                ax_sig[3*i+1].set_yticks([])
                ax_sig[3*i+2].set_yticks([])

        else:
            legends = []
            for i in range(Nr):
                for j in range(Nc):  # i, j: row and column gene index
                    idx = i_fig * Nr * Nc + i * Nc + j  # which gene
                    if idx >= len(gene_list):
                        break
                    for k, method in enumerate(methods):  # k: method index
                        # Pick time according to the method
                        if T[method].ndim == 2:
                            t = T[method][:, idx]
                            that = That[method][:, idx]
                        else:
                            t = T[method]
                            that = That[method]

                        title = f"{gene_list[idx]} (VeloVAE)" if method == "FullVB" else f"{gene_list[idx]} ({method})"
                        plot_sig_axis(ax_sig[3*i, M*j+k],
                                      t,
                                      U[:, idx],
                                      Labels[method],
                                      Legends[method],
                                      '.',
                                      alpha,
                                      down_sample,
                                      True,
                                      color_map=color_map,
                                      title=title)
                        plot_sig_axis(ax_sig[3*i+1, M*j+k],
                                      t,
                                      S[:, idx],
                                      Labels[method],
                                      Legends[method],
                                      '.',
                                      alpha,
                                      down_sample,
                                      color_map=color_map)

                        if len(Legends[method]) > len(legends):
                            legends = Legends[method]
                        try:
                            if ('VeloVAE' in method)\
                                or ('FullVB' in method)\
                                    or (methods[0] in ['DeepVelo',
                                                       'Discrete PyroVelocity',
                                                       'PyroVelocity',
                                                       'VeloVI',
                                                       'cellDancer']):
                                # These methods don't have line prediction
                                K = min(10, max(len(that)//5000, 1))
                                if frac > 0 and frac < 1:
                                    plot_sig_loess_axis(ax_sig[3*i, M*j+k],
                                                        that[::K],
                                                        Uhat[method][:, idx][::K],
                                                        Labels_demo[method][::K],
                                                        Legends[method],
                                                        frac=frac)
                                    plot_sig_loess_axis(ax_sig[3*i+1, M*j+k],
                                                        that[::K],
                                                        Shat[method][:, idx][::K],
                                                        Labels_demo[method][::K],
                                                        Legends[method], frac=frac)
                                elif 'Discrete' in method:
                                    uhat_plot = np.random.poisson(Uhat[method][:, idx])
                                    shat_plot = np.random.poisson(Shat[method][:, idx])
                                    plot_sig_pred_axis(ax_sig[3*i, M*j+k], that, uhat_plot)
                                    plot_sig_pred_axis(ax_sig[3*i+1, M*j+k], that, shat_plot)
                                plot_vel_axis(ax_sig[3*i+2, M*j+k],
                                              t,
                                              Shat[method][:, idx],
                                              V[method][:, idx],
                                              Labels[method],
                                              Legends[method],
                                              sparsity_correction=sparsity_correction,
                                              color_map=color_map)
                            else:  # plot line prediction
                                plot_sig_pred_axis(ax_sig[3*i, M*j+k],
                                                   that,
                                                   Uhat[method][:, idx],
                                                   Labels_demo[method],
                                                   Legends[method],
                                                   '-',
                                                   1.0,
                                                   1)
                                plot_sig_pred_axis(ax_sig[3*i+1, M*j+k],
                                                   that,
                                                   Shat[method][:, idx],
                                                   Labels_demo[method],
                                                   Legends[method],
                                                   '-',
                                                   1.0,
                                                   1)
                                plot_vel_axis(ax_sig[3*i+2, M*j+k],
                                              t,
                                              S[:, idx],
                                              V[method][:, idx],
                                              Labels[method],
                                              Legends[method],
                                              sparsity_correction=sparsity_correction,
                                              color_map=color_map)
                        except (KeyError, TypeError):
                            print("[** Warning **]: "
                                  "Skip plotting the prediction because of key value error or invalid data type.")
                            pass
                        if np.all(~np.isnan(t)):
                            ax_sig[3*i,  M*j+k].set_xlim(t.min(), np.quantile(t, 0.999)+0.1)
                            ax_sig[3*i+1,  M*j+k].set_xlim(t.min(), np.quantile(t, 0.999)+0.1)
                            ax_sig[3*i+2,  M*j+k].set_xlim(t.min(), np.quantile(t, 0.999)+0.1)

                        ax_sig[3*i,  M*j+k].set_xticks([])
                        ax_sig[3*i+1,  M*j+k].set_xticks([])
                        ax_sig[3*i+2,  M*j+k].set_xticks([])

                        ax_sig[3*i,  M*j+k].set_ylabel("U", fontsize=30, rotation=0)
                        ax_sig[3*i,  M*j+k].yaxis.set_label_coords(-0.03, 0.5)

                        ax_sig[3*i+1,  M*j+k].set_ylabel("S", fontsize=30, rotation=0)
                        ax_sig[3*i+1,  M*j+k].yaxis.set_label_coords(-0.03, 0.5)

                        ax_sig[3*i+2,  M*j+k].set_ylabel("S", fontsize=30, rotation=0)
                        ax_sig[3*i+2,  M*j+k].yaxis.set_label_coords(-0.03, 0.5)

                        ax_sig[3*i,  M*j+k].set_xlabel("Time", fontsize=30)
                        ax_sig[3*i+1,  M*j+k].set_xlabel("Time", fontsize=30)
                        ax_sig[3*i+2,  M*j+k].set_xlabel("Time", fontsize=30)
        if ax_sig.ndim == 1:
            handles, labels = ax_sig[0].get_legend_handles_labels()
        else:
            handles, labels = ax_sig[0, 0].get_legend_handles_labels()

        fig_sig.tight_layout()

        l_indent = 1 - 0.02/Nr
        if legend_fontsize is None:
            legend_fontsize = np.min([int(30*Nr), 300*Nr/len(Legends[methods[0]]), int(10*Nc)])
        lgd = fig_sig.legend(handles,
                             labels,
                             fontsize=legend_fontsize,
                             markerscale=5.0,
                             bbox_to_anchor=(-0.03/Nc, l_indent),
                             loc='upper right')

        fig_sig.subplots_adjust(hspace=0.3, wspace=0.12)

        save = None if (path is None or figname is None) else f'{path}/{figname}_sig_{i_fig+1}.{format}'
        save_fig(fig_sig, save, (lgd,))


def plot_time_grid(T,
                   X_emb,
                   capture_time=None,
                   std_t=None,
                   down_sample=1,
                   q=0.99,
                   W=6,
                   H=3,
                   save="figures/time_grid.png"):
    """Plot the latent time of different methods.

    Args:
        T (dict):
            Keys are method names and values are (N) arrays containing time
        X_emb (:class:`numpy.ndarray`):
            2D embedding for visualization, (N,2)
        capture_time (:class:`numpy.ndarray`, optional):
            Capture time, (N,). Defaults to None.
        std_t (dict, optional):
            Keys are method names and values are (N) arrays
            containing standard deviations of cell time.
            Not applicable to some methods. Defaults to None.
        down_sample (int, optional):
            Down-sampling factor to reduce data point overlapping.. Defaults to 1.
        q (float, optional):
            Top quantile for clipping extreme values. Defaults to 0.99.
        W (int, optional):
            Subplot width. Defaults to 6.
        H (int, optional):
            Subplot height. Defaults to 3.
        save (str, optional):
            Figure name for saving (including path). Defaults to "figures/time_grid.png".
    """
    if capture_time is not None:
        methods = ["Capture Time"] + list(T.keys())
    else:
        methods = list(T.keys())
    M = len(methods)
    if std_t is not None:
        fig_time, ax = plt.subplots(2, M, figsize=(W*M+2, H), facecolor='white')
        for i, method in enumerate(methods):
            t = capture_time if method == "Capture Time" else T[method]
            t = np.clip(t, None, np.quantile(t, q))
            t = t - t.min()
            t = t/t.max()
            if M > 1:
                ax[0, i].scatter(X_emb[::down_sample, 0],
                                 X_emb[::down_sample, 1],
                                 s=2.0,
                                 c=t[::down_sample],
                                 cmap='plasma',
                                 edgecolors='none')
                title = "VeloVAE" if method == "FullVB" else method
                ax[0, i].set_title(title, fontsize=24)
                ax[0, i].axis('off')
            else:
                ax[0].scatter(X_emb[::down_sample, 0],
                              X_emb[::down_sample, 1],
                              s=2.0,
                              c=t[::down_sample],
                              cmap='plasma',
                              edgecolors='none')
                title = "VeloVAE" if method == "FullVB" else method
                ax[0].set_title(title, fontsize=24)
                ax[0].axis('off')

            # Plot the Time Variance in a Colormap
            var_t = std_t[method]**2

            if np.any(var_t > 0):
                if M > 1:
                    ax[1, i].scatter(X_emb[::down_sample, 0],
                                     X_emb[::down_sample, 1],
                                     s=2.0,
                                     c=var_t[::down_sample],
                                     cmap='Reds',
                                     edgecolors='none')
                    norm1 = matplotlib.colors.Normalize(vmin=np.min(var_t), vmax=np.max(var_t))
                    sm1 = matplotlib.cm.ScalarMappable(norm=norm1, cmap='Reds')
                    cbar1 = fig_time.colorbar(sm1, ax=ax[1, i])
                    cbar1.ax.get_yaxis().labelpad = 15
                    cbar1.ax.set_ylabel('Time Variance', rotation=270, fontsize=12)
                    ax[1, i].axis('off')
                else:
                    ax[1].scatter(X_emb[::down_sample, 0],
                                  X_emb[::down_sample, 1],
                                  s=2.0,
                                  c=var_t[::down_sample],
                                  cmap='Reds',
                                  edgecolors='none')
                    norm1 = matplotlib.colors.Normalize(vmin=np.min(var_t), vmax=np.max(var_t))
                    sm1 = matplotlib.cm.ScalarMappable(norm=norm1, cmap='Reds')
                    cbar1 = fig_time.colorbar(sm1, ax=ax[1])
                    cbar1.ax.get_yaxis().labelpad = 15
                    cbar1.ax.set_ylabel('Time Variance', rotation=270, fontsize=12)
                    ax[1].axis('off')
    else:
        fig_time, ax = plt.subplots(1, M, figsize=(8*M, 4), facecolor='white')
        for i, method in enumerate(methods):
            t = capture_time if method == "Capture Time" else T[method]
            t = np.clip(t, None, np.quantile(t, q))
            t = t - t.min()
            t = t/t.max()
            if M > 1:
                ax[i].scatter(X_emb[::down_sample, 0],
                              X_emb[::down_sample, 1],
                              s=2.0,
                              c=t[::down_sample],
                              cmap='plasma',
                              edgecolors='none')
                title = "VeloVAE" if method == "FullVB" else method
                ax[i].set_title(title, fontsize=24)
                ax[i].axis('off')
            else:
                ax.scatter(X_emb[::down_sample, 0],
                           X_emb[::down_sample, 1],
                           s=2.0,
                           c=t[::down_sample],
                           cmap='plasma',
                           edgecolors='none')
                title = "VeloVAE" if method == "FullVB" else method
                ax.set_title(title, fontsize=24)
                ax.axis('off')
    norm0 = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sm0 = matplotlib.cm.ScalarMappable(norm=norm0, cmap='plasma')
    cbar0 = fig_time.colorbar(sm0, ax=ax, location="right") if M > 1 else fig_time.colorbar(sm0, ax=ax)
    cbar0.ax.get_yaxis().labelpad = 20
    cbar0.ax.set_ylabel('Cell Time', rotation=270, fontsize=24)

    save_fig(fig_time, save)


def _adj_mtx_to_map(w):
    """Convert adjacency matrix to a mapping (adjacency list)."""
    # w[i,j] = 1 if j is the parent of i
    G = {}
    for i in range(w.shape[1]):
        G[i] = []
        for j in range(w.shape[0]):
            if w[j, i] > 0:
                G[i].append(j)
    return G


def get_depth(graph):
    """Get the depths of all nodes in a tree-like graph."""
    depth = np.zeros((len(graph.keys())))
    roots = []
    for u in graph:
        if u in graph[u]:
            roots.append(u)
    for root in roots:
        queue = [root]
        depth[root] = 0
        while len(queue) > 0:
            v = queue.pop(0)
            for u in graph[v]:
                if u == root:
                    continue
                queue.append(u)
                depth[u] = depth[v] + 1
    return depth


def _plot_branch(ax, t, x, graph, label_dic_rev, plot_depth=True, color_map=None):
    """Plot some attributes of all nodes in a tree-like graph. """
    colors = get_colors(len(t), color_map)
    if plot_depth:
        depth = get_depth(graph)
        for i in range(len(t)):
            ax.scatter(depth[i:i+1], x[i:i+1], s=80, color=colors[i], label=label_dic_rev[i])
        for parent in graph:
            for child in graph[parent]:
                ax.plot([depth[child], depth[parent]], [x[child], x[parent]], "k-", alpha=0.2, linewidth=3)
    else:
        for i in range(len(t)):
            ax.scatter(t[i:i+1], x[i:i+1], s=80, color=colors[i], label=label_dic_rev[i])
        for parent in graph:
            for child in graph[parent]:
                ax.plot([t[child], t[parent]], [x[child], x[parent]], "k-", alpha=0.2, linewidth=3)
    return ax


def plot_rate_grid(adata,
                   key,
                   gene_list,
                   Nr,
                   Nc,
                   W=6,
                   H=3,
                   legend_ncol=8,
                   plot_depth=True,
                   color_map=None,
                   path="figures",
                   figname="genes",
                   format="png"):
    """Plot cell-type-specific rate parameters inferred from branching ODE.

    Args:
        adata (:class:`anndata.AnnData`):
            AnnData object.
        key (str):
            Key used to extract the corresponding rate parameters.
            For example, f"{key}_alpha" will be used to extract the transcription rate from .varm
        gene_list (array like):
            List of genes to plot
        Nr (int):
            Number of rows of the subplot grid.
        Nc (int):
            Number of columns of the subplot grid.
        W (int, optional):
            Subplot width. Defaults to 6.
        H (int, optional):
            Subplot width. Defaults to 3.
        legend_ncol (int, optional):
            Number of columns in the legend. Defaults to 8.
        plot_depth (bool, optional):
            Whether to plot the depth in transition graph as a surrogate of time.
            Set to true by default for better visualization. Defaults to True.
        color_map (str, optional):
            Defaults to None.
        path (str, optional):
            Path to the folder for saving the figure. Defaults to "figures".
        figname (str, optional):
            Name of the saved figure. Defaults to "genes".
        format (str, optional):
            Figure format, could be png, pdf, svg, eps and ps. Defaults to 'png'. Defaults to "png".

    """
    Nfig = len(gene_list) // (Nr*Nc)
    if Nfig * Nr * Nc < len(gene_list):
        Nfig += 1
    graph = _adj_mtx_to_map(adata.uns['brode_w'])
    label_dic = adata.uns['brode_label_dic']
    label_dic_rev = {}
    for type_ in label_dic:
        label_dic_rev[label_dic[type_]] = type_

    # Plotting
    for i_fig in range(Nfig):
        fig, ax = plt.subplots(3*Nr, Nc, figsize=(W*Nc, H*3*Nr), facecolor='white')
        if Nc == 1:
            for i in range(Nc):
                idx = i_fig*Nr * Nc + i
                gidx = np.where(adata.var_names == gene_list[idx])[0][0]
                alpha = adata.varm[f"{key}_alpha"][gidx]
                beta = adata.varm[f"{key}_beta"][gidx]
                gamma = adata.varm[f"{key}_gamma"][gidx]
                t_trans = adata.uns[f"{key}_t_trans"]

                ax[3*i] = _plot_branch(ax[3*i],
                                       t_trans,
                                       alpha,
                                       graph,
                                       label_dic_rev,
                                       plot_depth,
                                       color_map=color_map)
                ax[3*i+1] = _plot_branch(ax[3*i+1],
                                         t_trans,
                                         beta,
                                         graph,
                                         label_dic_rev,
                                         plot_depth,
                                         color_map=color_map)
                ax[3*i+2] = _plot_branch(ax[3*i+2],
                                         t_trans,
                                         gamma,
                                         graph,
                                         label_dic_rev,
                                         plot_depth,
                                         color_map=color_map)

                ax[3*i].set_ylabel(r"$\alpha$", fontsize=20, rotation=0)
                ax[3*i+1].set_ylabel(r"$\beta$", fontsize=20, rotation=0)
                ax[3*i+2].set_ylabel(r"$\gamma$", fontsize=20, rotation=0)
                for k in range(3):
                    ax[3*i+k].set_xticks([])
                    ax[3*i+k].set_yticks([])
                    if plot_depth:
                        ax[3*i+k].set_xlabel("Depth", fontsize=30)
                    else:
                        ax[3*i+k].set_xlabel("Time", fontsize=30)
                    ax[3*i+k].yaxis.set_label_coords(-0.03, 0.5)
                    ax[3*i+k].set_title(gene_list[idx], fontsize=30)
            handles, labels = ax[0].get_legend_handles_labels()
        else:
            for i in range(Nr):
                for j in range(Nc):  # i, j: row and column gene index
                    idx = i_fig*Nr*Nc+i*Nc+j  # which gene
                    if idx >= len(gene_list):
                        break
                    idx = i_fig*Nr*Nc+i*Nc+j
                    gidx = np.where(adata.var_names == gene_list[idx])[0][0]
                    alpha = adata.varm[f"{key}_alpha"][gidx]
                    beta = adata.varm[f"{key}_beta"][gidx]
                    gamma = adata.varm[f"{key}_gamma"][gidx]
                    t_trans = adata.uns[f"{key}_t_trans"]

                    ax[3*i, j] = _plot_branch(ax[3*i, j],
                                              t_trans,
                                              alpha,
                                              graph,
                                              label_dic_rev,
                                              color_map=color_map)
                    ax[3*i+1, j] = _plot_branch(ax[3*i+1, j],
                                                t_trans,
                                                beta,
                                                graph,
                                                label_dic_rev,
                                                color_map=color_map)
                    ax[3*i+2, j] = _plot_branch(ax[3*i+2, j],
                                                t_trans,
                                                gamma,
                                                graph,
                                                label_dic_rev,
                                                color_map=color_map)

                    ax[3*i, j].set_ylabel(r"$\alpha$", fontsize=30, rotation=0)
                    ax[3*i+1, j].set_ylabel(r"$\beta$", fontsize=30, rotation=0)
                    ax[3*i+2, j].set_ylabel(r"$\gamma$", fontsize=30, rotation=0)
                    for k in range(3):
                        ax[3*i+k, j].set_xticks([])
                        ax[3*i+k, j].set_yticks([])
                        ax[3*i+k, j].set_xlabel("Time", fontsize=30)
                        ax[3*i+k, j].yaxis.set_label_coords(-0.03, 0.5)
                        ax[3*i+k, j].set_title(gene_list[idx], fontsize=30)
            handles, labels = ax[0, 0].get_legend_handles_labels()
        plt.tight_layout()

        l_indent = 1 - 0.02/Nr
        legend_fontsize = np.min([int(30*Nr), int(10*Nc)])
        # min(Nr*10, Nr*120/len(graph.keys()))
        lgd = fig.legend(handles,
                         labels,
                         fontsize=legend_fontsize,
                         markerscale=1,
                         bbox_to_anchor=(-0.03/Nc, l_indent),
                         loc='upper right')

        save = None if figname is None else f'{path}/{figname}_brode_rates_{i_fig+1}.{format}'
        save_fig(fig, save, (lgd,))
    return


def plot_velocity_stream(X_embed,
                         t,
                         vx,
                         vy,
                         cell_labels,
                         n_grid=50,
                         k=50,
                         k_time=10,
                         dist_thred=None,
                         eps_t=None,
                         scale=1.5,
                         color_map=None,
                         figsize=(8, 6),
                         save='figures/velstream.png'):
    """
    .. deprecated:: 0.1.0
    """
    # Compute velocity on a grid
    knn_model = pynndescent.NNDescent(X_embed, n_neighbors=2*k)
    umap1, umap2 = X_embed[:, 0], X_embed[:, 1]
    x = np.linspace(X_embed[:, 0].min(), X_embed[:, 0].max(), n_grid)
    y = np.linspace(X_embed[:, 1].min(), X_embed[:, 1].max(), n_grid)

    xgrid, ygrid = np.meshgrid(x, y)
    xgrid, ygrid = xgrid.flatten(), ygrid.flatten()
    Xgrid = np.stack([xgrid, ygrid]).T

    neighbors_grid, dist_grid = knn_model.query(Xgrid, k=k)
    neighbors_grid_, dist_grid_ = knn_model.query(Xgrid, k=k_time)
    neighbors_grid = neighbors_grid.astype(int)
    neighbors_grid_ = neighbors_grid_.astype(int)

    # Prune grid points
    if dist_thred is None:
        ind, dist = knn_model.neighbor_graph
        dist_thred = dist.mean() * scale
    mask = np.quantile(dist_grid, 0.5, 1) <= dist_thred

    # transition probability on UMAP
    def transition_prob(dist, sigma):
        P = np.exp(-(dist/sigma)**2)
        P = P/P.sum(1).reshape(-1, 1)
        return P

    P = transition_prob(dist_grid, dist_thred)
    P_ = transition_prob(dist_grid_, dist_thred)

    # Local Averaging
    tgrid = np.sum(np.stack([t[neighbors_grid_[i]] for i in range(len(xgrid))])*P_, 1)
    if eps_t is None:
        eps_t = (t.max()-t.min())/500
    P = P*np.stack([t[neighbors_grid[i]] >= tgrid[i]+eps_t for i in range(len(xgrid))])
    vx_grid = np.sum(np.stack([vx[neighbors_grid[i]] for i in range(len(xgrid))])*P, 1)
    vy_grid = np.sum(np.stack([vy[neighbors_grid[i]] for i in range(len(xgrid))])*P, 1)

    fig, ax = plt.subplots(figsize=figsize)
    # Plot cells by label
    font_shift = (x.max()-x.min())/100
    cell_types = np.unique(cell_labels)
    colors = get_colors(len(cell_types), color_map)

    for i, type_ in enumerate(cell_types):
        cell_mask = cell_labels == type_
        ax.scatter(umap1[cell_mask], umap2[cell_mask], s=5.0, color=colors[i % len(colors)], alpha=0.5)
        ax.text(umap1[cell_mask].mean() - len(type_)*font_shift,
                umap2[cell_mask].mean(),
                type_,
                fontsize=15,
                color='k')
    ax.streamplot(xgrid.reshape(n_grid, n_grid),
                  ygrid.reshape(n_grid, n_grid),
                  (vx_grid*mask).reshape(n_grid, n_grid),
                  (vy_grid*mask).reshape(n_grid, n_grid),
                  density=2.0,
                  color='k',
                  integration_direction='both')
    ax.set_title('Velocity Stream Plot')
    ax.set_xlabel('Umap 1')
    ax.set_ylabel('Umap 2')

    save_fig(fig, save)


def plot_cell_trajectory(X_embed,
                         t,
                         cell_labels,
                         n_grid=50,
                         k=30,
                         k_grid=8,
                         scale=1.5,
                         eps_t=None,
                         color_map=None,
                         save=None):
    """Plot the velocity stream based on time. This is not stable yet and we suggest not using it for now.
    .. deprecated:: 0.1.0
    """
    # Compute the time on a grid
    knn_model = pynndescent.NNDescent(X_embed, n_neighbors=k+20)
    ind, dist = knn_model.neighbor_graph
    dist_thred = dist.mean() * scale

    x = np.linspace(X_embed[:, 0].min(), X_embed[:, 0].max(), n_grid)
    y = np.linspace(X_embed[:, 1].min(), X_embed[:, 1].max(), n_grid)

    xgrid, ygrid = np.meshgrid(x, y)
    xgrid, ygrid = xgrid.flatten(), ygrid.flatten()
    Xgrid = np.stack([xgrid, ygrid]).T

    neighbors_grid, dist_grid = knn_model.query(Xgrid, k=k)
    mask = np.quantile(dist_grid, 0.5, 1) <= dist_thred

    # transition probability on UMAP
    def transition_prob(dist, sigma):
        P = np.exp(-(np.clip(dist/sigma, -100, None))**2)
        psum = P.sum(1).reshape(-1, 1)
        psum[psum == 0] = 1.0
        P = P/psum
        return P

    P = transition_prob(dist_grid, dist_thred)
    tgrid = np.sum(np.stack([t[neighbors_grid[i]] for i in range(len(xgrid))])*P, 1)
    tgrid = tgrid[mask]

    # Compute velocity based on grid time
    # filter out distant grid points
    knn_grid = pynndescent.NNDescent(Xgrid[mask], n_neighbors=k_grid, metric="l2")
    neighbor_grid, dist_grid = knn_grid.neighbor_graph

    if eps_t is None:
        eps_t = (t.max()-t.min())/len(t)*10
    delta_t = tgrid[neighbor_grid] - tgrid.reshape(-1, 1) - eps_t
    sigma_t = (t.max()-t.min())/n_grid
    P = (np.exp((np.clip(delta_t/sigma_t, -100, 100))**2))*(delta_t >= 0)
    psum = P.sum(1).reshape(-1, 1)
    psum[psum == 0] = 1.0
    P = P/psum

    delta_x = (xgrid[mask][neighbor_grid] - xgrid[mask].reshape(-1, 1))
    delta_y = (ygrid[mask][neighbor_grid] - ygrid[mask].reshape(-1, 1))
    norm = np.sqrt(delta_x**2+delta_y**2)
    norm[norm == 0] = 1.0
    vx_grid_filter = ((delta_x/norm)*P).sum(1)
    vy_grid_filter = ((delta_y/norm)*P).sum(1)
    vx_grid = np.zeros((n_grid*n_grid))
    vy_grid = np.zeros((n_grid*n_grid))
    vx_grid[mask] = vx_grid_filter
    vy_grid[mask] = vy_grid_filter

    fig, ax = plt.subplots(figsize=(15, 12))
    # Plot cells by label
    cell_types = np.unique(cell_labels)
    colors = get_colors(len(cell_types), color_map)

    for i, type_ in enumerate(cell_types):
        cell_mask = cell_labels == type_
        ax.scatter(X_embed[:, 0][cell_mask], X_embed[:, 1][cell_mask], s=5.0, c=colors[i], alpha=0.5, label=type_)

    ax.streamplot(xgrid.reshape(n_grid, n_grid),
                  ygrid.reshape(n_grid, n_grid),
                  vx_grid.reshape(n_grid, n_grid),
                  vy_grid.reshape(n_grid, n_grid),
                  density=2.0,
                  color='k',
                  integration_direction='both')

    ax.set_title('Velocity Stream Plot')
    ax.set_xlabel('Umap 1')
    ax.set_ylabel('Umap 2')
    lgd = ax.legend(fontsize=12, ncol=4, markerscale=3.0, bbox_to_anchor=(0.0, 1.0, 1.0, 0.5), loc='center')

    save_fig(fig, save, (lgd,))

    return


def plot_velocity_3d(X_embed,
                     t,
                     cell_labels,
                     plot_arrow=True,
                     n_grid=50,
                     k=30,
                     k_grid=8,
                     scale=1.5,
                     angle=(15, 45),
                     eps_t=None,
                     color_map=None,
                     save=None):
    """3D velocity quiver plot.
    Arrows follow the direction of time to nearby points.
    This is not stable yet and we suggest not using it for now.
    .. deprecated:: 0.1.0
    """
    fig = plt.figure(figsize=(30, 15))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(angle[0], angle[1])

    t_clip = np.clip(t, np.quantile(t, 0.01), np.quantile(t, 0.99))
    cell_types = np.unique(cell_labels)
    colors = get_colors(len(cell_types), color_map)
    for i, type_ in enumerate(cell_types):
        cell_mask = cell_labels == type_
        d = max(1, np.sum(cell_mask)//3000)
        ax.scatter(X_embed[:, 0][cell_mask][::d],
                   X_embed[:, 1][cell_mask][::d],
                   X_embed[:, 2][cell_mask][::d],
                   s=5.0,
                   color=colors[i],
                   label=type_,
                   alpha=0.5,
                   edgecolor='none')

    if plot_arrow:
        # Used for filtering target grid points
        knn_model = pynndescent.NNDescent(X_embed, n_neighbors=k+20)
        ind, dist = knn_model.neighbor_graph
        dist_thred = dist.mean() * scale  # filter grid points distant from data cloud

        x = np.linspace(X_embed[:, 0].min(), X_embed[:, 0].max(), n_grid)
        y = np.linspace(X_embed[:, 1].min(), X_embed[:, 1].max(), n_grid)
        z = np.linspace(X_embed[:, 2].min(), X_embed[:, 2].max(), n_grid)

        xgrid, ygrid, zgrid = np.meshgrid(x, y, z)
        xgrid, ygrid, zgrid = xgrid.flatten(), ygrid.flatten(), zgrid.flatten()
        Xgrid = np.stack([xgrid, ygrid, zgrid]).T

        neighbors_grid, dist_grid = knn_model.query(Xgrid, k=k)
        mask = np.quantile(dist_grid, 0.5, 1) <= dist_thred

        # transition probability on UMAP
        def transition_prob(dist, sigma):
            P = np.exp(-(np.clip(dist/sigma, -5, None))**2)
            psum = P.sum(1).reshape(-1, 1)
            psum[psum == 0] = 1.0
            P = P/psum
            return P

        P = transition_prob(dist_grid, dist_thred)
        tgrid = np.sum(np.stack([t[neighbors_grid[i]] for i in range(len(xgrid))])*P, 1)
        tgrid = tgrid[mask]

        # Compute velocity based on grid time
        # filter out distant grid points
        knn_grid = pynndescent.NNDescent(Xgrid[mask], n_neighbors=k_grid, metric="l2")
        neighbor_grid, dist_grid = knn_grid.neighbor_graph

        if eps_t is None:
            eps_t = (t_clip.max()-t_clip.min())/len(t)*10
        delta_t = tgrid[neighbor_grid] - tgrid.reshape(-1, 1) - eps_t

        sigma_t = (t_clip.max()-t_clip.min())/n_grid

        # Filter out backflow and distant points in 2D space
        dist_thred_ = (dist_grid.mean(1)+dist_grid.std(1)).reshape(-1, 1)  # filter grid points distant from data cloud
        P = (np.exp((np.clip(delta_t/sigma_t, -5, 5))**2))*((delta_t >= 0) & (dist_grid <= dist_thred_))
        psum = P.sum(1).reshape(-1, 1)
        psum[psum == 0] = 1.0
        P = P/psum

        delta_x = (xgrid[mask][neighbor_grid] - xgrid[mask].reshape(-1, 1))
        delta_y = (ygrid[mask][neighbor_grid] - ygrid[mask].reshape(-1, 1))
        delta_z = (zgrid[mask][neighbor_grid] - zgrid[mask].reshape(-1, 1))
        norm = np.sqrt(delta_x**2+delta_y**2+delta_z**2)
        norm[norm == 0] = 1.0
        vx_grid_filter = ((delta_x/norm)*P).sum(1)
        vy_grid_filter = ((delta_y/norm)*P).sum(1)
        vz_grid_filter = ((delta_z/norm)*P).sum(1)
        # KNN Smoothing
        vx_grid_filter = vx_grid_filter[neighbor_grid].mean(1)
        vy_grid_filter = vy_grid_filter[neighbor_grid].mean(1)
        vz_grid_filter = vz_grid_filter[neighbor_grid].mean(1)

        vx_grid = np.zeros((n_grid*n_grid*n_grid))
        vy_grid = np.zeros((n_grid*n_grid*n_grid))
        vz_grid = np.zeros((n_grid*n_grid*n_grid))
        vx_grid[mask] = vx_grid_filter
        vy_grid[mask] = vy_grid_filter
        vz_grid[mask] = vz_grid_filter

        ax.quiver(xgrid.reshape(n_grid, n_grid, n_grid),
                  ygrid.reshape(n_grid, n_grid, n_grid),
                  zgrid.reshape(n_grid, n_grid, n_grid),
                  vx_grid.reshape(n_grid, n_grid, n_grid),
                  vy_grid.reshape(n_grid, n_grid, n_grid),
                  vz_grid.reshape(n_grid, n_grid, n_grid),
                  color='k',
                  length=2*np.median(X_embed.max(1)-X_embed.min(1))/n_grid,
                  normalize=True)

    ax.set_xlabel('Embedding 1', fontsize=16)
    ax.set_ylabel('Embedding 2', fontsize=16)
    ax.set_zlabel('Embedding 3', fontsize=16)

    lgd = ax.legend(fontsize=12, ncol=4, markerscale=5.0, bbox_to_anchor=(0.0, 1.0, 1.0, -0.05), loc='center')
    plt.tight_layout()

    save_fig(fig, save, (lgd,))

    return


def plot_trajectory_3d(X_embed,
                       t,
                       cell_labels,
                       plot_arrow=False,
                       n_grid=50,
                       n_time=20,
                       k=30,
                       k_grid=8,
                       scale=1.5,
                       angle=(15, 45),
                       figsize=(12, 9),
                       eps_t=None,
                       color_map=None,
                       embed='umap',
                       save=None,
                       **kwargs):
    """3D quiver plot. x-y plane is a 2D embedding such as UMAP.
    z axis is the cell time. Arrows follow the direction of time to nearby points.

    Args:
        X_embed (:class:`numpy.ndarray`):
            2D embedding for visualization
        t (:class:`numpy.ndarray`):
            Cell time.
        cell_labels (:class:`numpy.ndarray`):
            Cell type annotations.
        plot_arrow (bool, optional):
            Whether to add a quiver plot upon the background 3D scatter plot.
            Defaults to False.
        n_grid (int, optional):
            Grid size of the x-y plane. Defaults to 50.
        n_time (int, optional):
            Grid size of the z (time) axis. Defaults to 20.
        k (int, optional):
            Number of neighbors when computing velocity of each grid point. Defaults to 30.
        k_grid (int, optional):
            Number of neighbors when averaging across the 3D grid. Defaults to 8.
        scale (float, optional):
            Parameter to control boundary detection. Defaults to 1.5.
        angle (tuple, optional):
            Angle of the 3D plot. Defaults to (15, 45).
        figsize (tuple, optional):
            Defaults to (12, 9).
        eps_t (float, optional):
            Parameter to control the relative time order of cells. Defaults to None.
        color_map (str, optional):
            Defaults to None.
        embed (str, optional):
            Name of the embedding.. Defaults to 'umap'.
        save (str, optional):
            Figure name for saving (including path). Defaults to None.

    """
    t_clip = np.clip(t, np.quantile(t, 0.01), np.quantile(t, 0.99))
    range_z = np.max(X_embed.max(0) - X_embed.min(0))
    w = range_z/(t_clip.max()-t_clip.min())
    x_3d = np.concatenate((X_embed, (t_clip - t_clip.min()).reshape(-1, 1)*w), 1)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    ax.view_init(angle[0], angle[1])
    # Plot cells by label
    cell_types = np.unique(cell_labels)
    colors = get_colors(len(cell_types), color_map)
    for i, type_ in enumerate(cell_types):
        cell_mask = cell_labels == type_
        d = max(1, np.sum(cell_mask)//3000)
        ax.scatter(x_3d[:, 0][cell_mask][::d],
                   x_3d[:, 1][cell_mask][::d],
                   x_3d[:, 2][cell_mask][::d],
                   s=5.0,
                   color=colors[i],
                   label=type_,
                   edgecolor='none')
    if plot_arrow:
        # Used for filtering target grid points
        knn_model_2d = pynndescent.NNDescent(X_embed, n_neighbors=k)

        # Compute the time on a grid
        knn_model = pynndescent.NNDescent(x_3d, n_neighbors=k+20)
        ind, dist = knn_model.neighbor_graph
        dist_thred = dist.mean() * scale

        x = np.linspace(x_3d[:, 0].min(), x_3d[:, 0].max(), n_grid)
        y = np.linspace(x_3d[:, 1].min(), x_3d[:, 1].max(), n_grid)
        z = np.linspace(x_3d[:, 2].min(), x_3d[:, 2].max(), n_time)

        xgrid, ygrid, zgrid = np.meshgrid(x, y, z)
        xgrid, ygrid, zgrid = xgrid.flatten(), ygrid.flatten(), zgrid.flatten()
        Xgrid = np.stack([xgrid, ygrid, zgrid]).T

        neighbors_grid, dist_grid = knn_model.query(Xgrid, k=k)
        mask = np.quantile(dist_grid, 0.5, 1) <= dist_thred

        # transition probability on UMAP
        def transition_prob(dist, sigma):
            P = np.exp(-(np.clip(dist/sigma, -5, None))**2)
            psum = P.sum(1).reshape(-1, 1)
            psum[psum == 0] = 1.0
            P = P/psum
            return P

        P = transition_prob(dist_grid, dist_thred)
        tgrid = np.sum(np.stack([t[neighbors_grid[i]] for i in range(len(xgrid))])*P, 1)
        tgrid = tgrid[mask]

        # Compute velocity based on grid time
        # filter out distant grid points
        knn_grid = pynndescent.NNDescent(Xgrid[mask], n_neighbors=k_grid, metric="l2")
        neighbor_grid, dist_grid = knn_grid.neighbor_graph

        if eps_t is None:
            eps_t = (t_clip.max()-t_clip.min())/len(t)*10
        delta_t = tgrid[neighbor_grid] - tgrid.reshape(-1, 1) - eps_t

        sigma_t = (t_clip.max()-t_clip.min())/n_grid
        ind_grid_2d, dist_grid_2d = knn_model_2d.query(Xgrid[mask], k=k_grid)
        dist_thred_2d = (dist_grid_2d.mean(1)+dist_grid_2d.std(1)).reshape(-1, 1)
        # Filter out backflow and distant points in 2D space
        P = (np.exp((np.clip(delta_t/sigma_t, -5, 5))**2))*((delta_t >= 0) & (dist_grid_2d <= dist_thred_2d))
        psum = P.sum(1).reshape(-1, 1)
        psum[psum == 0] = 1.0
        P = P/psum

        delta_x = (xgrid[mask][neighbor_grid] - xgrid[mask].reshape(-1, 1))
        delta_y = (ygrid[mask][neighbor_grid] - ygrid[mask].reshape(-1, 1))
        delta_z = (zgrid[mask][neighbor_grid] - zgrid[mask].reshape(-1, 1))
        norm = np.sqrt(delta_x**2+delta_y**2+delta_z**2)
        norm[norm == 0] = 1.0
        vx_grid_filter = ((delta_x/norm)*P).sum(1)
        vy_grid_filter = ((delta_y/norm)*P).sum(1)
        vz_grid_filter = ((delta_z/norm)*P).sum(1)
        # KNN Smoothing
        vx_grid_filter = vx_grid_filter[neighbor_grid].mean(1)
        vy_grid_filter = vy_grid_filter[neighbor_grid].mean(1)
        vz_grid_filter = vz_grid_filter[neighbor_grid].mean(1)

        vx_grid = np.zeros((n_grid*n_grid*n_time))
        vy_grid = np.zeros((n_grid*n_grid*n_time))
        vz_grid = np.zeros((n_grid*n_grid*n_time))
        vx_grid[mask] = vx_grid_filter
        vy_grid[mask] = vy_grid_filter
        vz_grid[mask] = vz_grid_filter

        range_x = np.mean(X_embed.max(0) - X_embed.min(0))
        ax.quiver(xgrid.reshape(n_grid, n_grid, n_time),
                  ygrid.reshape(n_grid, n_grid, n_time),
                  zgrid.reshape(n_grid, n_grid, n_time),
                  vx_grid.reshape(n_grid, n_grid, n_time),
                  vy_grid.reshape(n_grid, n_grid, n_time),
                  vz_grid.reshape(n_grid, n_grid, n_time),
                  color='k',
                  length=(0.8*range_x/n_grid + 0.8*range_x/n_time),
                  normalize=True)

    ax.set_xlabel(f'{embed} 1', fontsize=12)
    ax.set_ylabel(f'{embed} 2', fontsize=12)
    ax.set_zlabel('Time', fontsize=12)

    ncol = kwargs['ncol'] if 'ncol' in kwargs else 4
    fontsize = kwargs['legend_fontsize'] if 'legend_fontsize' in kwargs else 12
    lgd = ax.legend(fontsize=fontsize, ncol=ncol, markerscale=5.0, bbox_to_anchor=(0.0, 1.0, 1.0, -0.05), loc='center')
    fig.tight_layout()
    if 'axis_off' in kwargs:
        ax.axis('off')
    save_fig(fig, save, (lgd,))


def plot_transition_graph(adata,
                          key="brode",
                          figsize=(4, 8),
                          color_map=None,
                          save=None):
    """Plot a directed graph with cell types as nodes
    and progenitor-descendant relation as edges.

    Args:
        adata (:class:`anndata.AnnData`):
            AnnData object.
        key (str, optional):
            Key used to extract the transition probability from .uns. Defaults to "brode".
        figsize (tuple, optional):
            Defaults to (4, 8).
        color_map (str, optional):
            Defaults to None.
        save (str, optional):
            Figure name for saving (including path). Defaults to None.

    """
    fig, ax = plt.subplots(figsize=figsize)
    adj_mtx = adata.uns[f"{key}_w"]
    n_type = adj_mtx.shape[0]
    label_dic = adata.uns['brode_label_dic']
    label_dic_rev = {}
    for key in label_dic:
        label_dic_rev[label_dic[key]] = key
    chd, par = np.where(adj_mtx > 0)
    edges = [(par[i], chd[i]) for i in range(len(par))]
    for i in range(n_type):
        if adj_mtx[i, i] == 1:
            edges.remove((i, i))
    node_name = [label_dic_rev[i] for i in range(len(label_dic_rev.keys()))]

    g = ig.Graph(directed=True, edges=edges)
    g.vs["name"] = node_name

    colors = get_colors(n_type, color_map)
    layout = g.layout_reingold_tilford()
    ig.plot(g,
            layout=layout,
            vertex_color=colors,
            vertex_size=0.5,
            edge_width=2,
            target=ax)

    ax.axis("off")
    plt.tight_layout()

    # Get legends
    _fig, _ax = plt.subplots()
    handles = []
    for i in range(len(colors)):
        handles.append(_ax.plot([], [], marker='o', color=colors[i], label=node_name[i])[0])
    plt.close(_fig)
    labels = node_name
    _fig.legend(handles, labels, loc=3, framealpha=1, frameon=False)

    lgd = fig.legend(handles,
                     labels,
                     fontsize=15,
                     markerscale=2,
                     ncol=1,
                     bbox_to_anchor=(0.0, min(0.95, 0.5+0.02*n_type)),
                     loc='upper right')
    save_fig(fig, save, (lgd,))

    return


def plot_rate_hist(adata, model, key, tprior='tprior', figsize=(18, 4), save="figures/hist.png"):
    """Convert rate parameters to real interpretable units and plot the histogram

    Args:
        adata (:class:`anndata.AnnData`):
            AnnData object.
        model (str):
            Model name.
        key (str):
            Key for retreiving model predictions/inferred parameters and other data.
        tprior (str, optional):
            Key for capture time.
            This is used to convert rates to transcript/minute.
            If not provided or doesn't exist in adata, we assume the experiment lasts
            one day. Defaults to 'tprior'.
        figsize (tuple, optional):
            Defaults to (18, 4).
        save (str, optional):
            Figure name for saving (including path). Defaults to "figures/hist.png".
    """
    if 'Discrete' in model:
        U, S = adata.layers[f"{key}_uhat"], adata.layers[f"{key}_shat"]
    else:
        U, S = adata.layers["Mu"], adata.layers["Ms"]
    x_median = np.median(U.sum(1)) + np.median(S.sum(1))
    sparsity_u = [1 - np.sum(U[:, i] <= U[:, i].max()*0.01)/adata.n_obs/adata.n_vars for i in range(adata.n_vars)]
    sparsity_s = [1 - np.sum(S[:, i] <= S[:, i].max()*0.01)/adata.n_obs/adata.n_vars for i in range(adata.n_vars)]
    u_median = np.array([np.quantile(U[:, i], 0.5+0.5*sparsity_u[i]) for i in range(adata.n_vars)])
    s_median = np.array([np.quantile(S[:, i], 0.5+0.5*sparsity_s[i]) for i in range(adata.n_vars)])
    sparsity_scale = (360000 / x_median)
    t = adata.obs[f"{key}_time"].to_numpy()
    if tprior in adata.obs:
        tprior = adata.obs[tprior].to_numpy()
        t_scale = (tprior.max()-tprior.min()) / (np.quantile(t, 0.99)-np.quantile(t, 0.01))
    else:
        print('Warning: No multiple capture times detected! Assume the experiment lasts one day.')
        t_scale = 1 / (np.quantile(t, 0.99) - np.quantile(t, 0.01))
    if "FullVB" in model:
        std_alpha = np.exp(adata.var[f"{key}_logstd_alpha"].to_numpy())
        std_beta = np.exp(adata.var[f"{key}_logstd_beta"].to_numpy())
        std_gamma = np.exp(adata.var[f"{key}_logstd_gamma"].to_numpy())
        alpha = np.exp(adata.var[f"{key}_logmu_alpha"].to_numpy()+0.5*std_alpha**2)\
            / (1440*t_scale) * sparsity_scale
        beta = np.exp(adata.var[f"{key}_logmu_beta"].to_numpy()+0.5*std_beta**2) * u_median\
            / (1440*t_scale) * sparsity_scale
        gamma = np.exp(adata.var[f"{key}_logmu_gamma"].to_numpy()+0.5*std_gamma**2) * s_median\
            / (1440*t_scale) * sparsity_scale
    elif "VeloVAE" in model:
        alpha = (adata.var[f"{key}_alpha"]).to_numpy() / (1440*t_scale) * sparsity_scale
        beta = (adata.var[f"{key}_beta"]).to_numpy() * u_median / (1440*t_scale) * sparsity_scale
        gamma = (adata.var[f"{key}_gamma"]).to_numpy() * s_median / (1440*t_scale) * sparsity_scale
    ub = [min(np.quantile(alpha, 0.95), alpha.mean()*4),
          min(np.quantile(beta, 0.95), beta.mean()*4),
          min(np.quantile(gamma, 0.95), gamma.mean()*4)]

    fig, ax = plt.subplots(1, 3, figsize=figsize)
    ax[0].hist(alpha, bins=np.linspace(0, ub[0], 50), color="orange", label=r"$\alpha$")
    ax[1].hist(beta, bins=np.linspace(0, ub[1], 50), color="green", label=r"$\beta$")
    ax[2].hist(gamma, bins=np.linspace(0, ub[2], 50), color="blue", label=r"$\gamma$")
    ax[0].set_xlabel(r"$\alpha$ (transcript / min)", fontsize=20)
    ax[1].set_xlabel(r"$\beta$u (transcript / min)", fontsize=20)
    ax[2].set_xlabel(r"$\gamma$s (transcript / min)", fontsize=20)
    ax[0].set_ylabel("Number of Genes", fontsize=20)
    plt.tight_layout()
    save_fig(fig, save)

import numpy as np
import pandas as pd
import os
from ..plotting import get_colors
import re


class PerfLogger:
    """Class for saving the performance metrics
    """
    def __init__(self, save_path='perf', checkpoints=None):
        """Constructor

        Args:
            save_path (str, optional):
                Path for saving the data frames to .csv files. Defaults to 'perf'.
            checkpoints (list[str], optional):
                Existing results to load (.csv). Defaults to None.
        """
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        self.n_dataset = 0
        self.n_model = 0
        self.metrics = ["MSE Train",
                        "MSE Test",
                        "MAE Train",
                        "MAE Test",
                        "LL Train",
                        "LL Test",
                        "CBDir",
                        "CBDir (Velocity Genes)",
                        "CBDir (Embed)",
                        "CBDir (Embed, Velocity Genes)",
                        "Time Score",
                        "Vel Consistency",
                        "corr"]
        self.metrics_type = ["CBDir",
                             "CBDir (Velocity Genes)",
                             "CBDir (Embed)",
                             "CBDir (Embed, Velocity Genes)",
                             "Time Score"]
        if checkpoints is None:
            self._create_empty_df()
        else:
            self.df = pd.read_csv(checkpoints[0], header=[0], index_col=[0, 1])
            self.df_type = pd.read_csv(checkpoints[1], header=[0, 1], index_col=[0, 1])

    def _create_empty_df(self):
        row_mindex = pd.MultiIndex.from_arrays([[], []], names=["Metrics", "Model"])
        col_index = pd.Index([], name='Dataset')
        col_mindex = pd.MultiIndex.from_arrays([[], []], names=["Dataset", "Pair"])
        self.df = pd.DataFrame(index=row_mindex, columns=col_index)
        self.df_type = pd.DataFrame(index=row_mindex, columns=col_mindex)

    def insert(self, data_name, res, res_type):
        """Insert the performance evaluation results from velovae.post_analysis

        Args:
            data_name (str):
                Name of the dataset
            res (:class:`pandas.DataFrame`):
                Contains performance metrics for the entire dataset.
                Rows are the performance metrics.
                columns are model names.
            res_type (:class:`pandas.DataFrame`):
                Contains the velocity and time metrics for each pair of
                cell type transition. Rows are different performance metrics
                while columns are indexed by method and cell type transitions.
        """
        self.n_dataset += 1
        # Collapse the dataframe to 1D series with multi-index
        res_1d = pd.Series(res.values.flatten(), index=pd.MultiIndex.from_product([res.index, res.columns]))
        for x in res_1d.index:
            self.df.loc[x, data_name] = res_1d.loc[x]

        # Reshape the data in res_type to match the multi-row-index in self.df_type
        methods = np.unique(res_type.columns.get_level_values(0))
        row_index = pd.MultiIndex.from_product([res_type.index.values, methods]).values
        transitions = np.unique(res_type.columns.get_level_values(1))

        for pair in transitions:
            if (data_name, pair) not in self.df_type.columns:
                self.df_type.insert(self.df_type.shape[1],
                                    (data_name, pair),
                                    np.ones((self.df_type.shape[0]))*np.nan)
        for row in row_index:
            for pair in transitions:
                self.df_type.loc[row, (data_name, pair)] = res_type.loc[row[0], (row[1], pair)]
        # update number of models
        self.n_model = len(self.df.iloc[0].index)
        self.df.sort_index(inplace=True)
        self.df_type.sort_index(inplace=True)
        return

    def plot(self, figure_path=None, bbox_to_anchor=(1.25, 1.0)):
        """Generate bar plots showing all performance metrics

        Args:
            figure_path (str, optional):
                Path to the folder for saving figures.
                If set to None, figures will not be saved.
                Defaults to None.
            bbox_to_anchor (tuple, optional):
                Location of the legend. Defaults to (1.25, 1.0).
        """
        datasets = np.unique(self.df_type.columns.get_level_values(0))
        for metric in self.metrics:
            colors = get_colors(self.df.loc[metric, :].shape[0])
            fig_name = re.sub(r'\W+', ' ', metric.lower())
            fig_name = '_'.join(fig_name.rstrip().split())
            if np.all(np.isnan(self.df.loc[metric, :].values)):
                continue
            ax = self.df.loc[metric, :].T.plot.bar(color=colors, figsize=(12, 6), fontsize=14)
            ax.set_xlabel("")          
            ax.set_title(metric, fontsize=20)
            if isinstance(bbox_to_anchor, tuple):
                ax.legend(fontsize=16, loc=1, bbox_to_anchor=bbox_to_anchor)
            fig = ax.get_figure()
            fig.tight_layout()
            if figure_path is not None:
                fig.savefig(f'{figure_path}/perf_{fig_name}.png', bbox_inches='tight')
        for metric in self.metrics_type:
            fig_name = re.sub(r'\W+', ' ', metric.lower())
            fig_name = '_'.join(fig_name.rstrip().split())
            for dataset in datasets:
                if np.all(np.isnan(self.df_type.loc[metric, dataset].values)):
                    continue
                colors = get_colors(self.df_type.loc[metric, dataset].shape[0])
                ax = self.df_type.loc[metric, dataset].T.plot.bar(color=colors, figsize=(12, 6), fontsize=14)
                ax.set_title(metric, fontsize=20)
                if isinstance(bbox_to_anchor, tuple):
                    ax.legend(fontsize=16, loc=1, bbox_to_anchor=bbox_to_anchor)
                ax.set_xlabel("")
                fig = ax.get_figure()
                fig.tight_layout()
                if figure_path is not None:
                    fig.savefig(f'{figure_path}/perf_{fig_name}_{dataset}.png', bbox_inches='tight')
        return

    def save(self, file_name=None):
        """Save data frames to .csv files.

        Args:
            file_name (str, optional):
                Name of the csv file for saving. Does not need the path
                as the path is specified when an object is created.
                If set to None, will pick 'perf' as the default name.
                Defaults to None.
        """
        if file_name is None:
            file_name = "perf"
        self.df.to_csv(f"{self.save_path}/{file_name}.csv")
        self.df_type.to_csv(f"{self.save_path}/{file_name}_type.csv")
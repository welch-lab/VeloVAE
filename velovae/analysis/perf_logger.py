import numpy as np
import pandas as pd
import os


class PerfLogger:
    """Class for saving the performance metrics
    """
    def __init__(self, save_path='perf', checkpoints=None):
        self.save_path = save_path
        self.metrics = ["MSE Train",
                        "MSE Test",
                        "MAE Train",
                        "MAE Test",
                        "LL Train",
                        "LL Test",
                        "CBDir",
                        "CBDir (Subset)"
                        "CBDir (Embed)",
                        "CBDir (Embed, Subset)",
                        "Time Score"]
        self.metrics_type = ["CBDir",
                             "CBDir (Subset)"
                             "CBDir (Embed)",
                             "CBDir (Embed, Subset)",
                             "Time Score"]
        if checkpoints is None:
            self._create_empty_df()
        else:
            self.df = pd.read_csv(checkpoints[0], header=[0], index_col=[0, 1])
            self.df_type = pd.read_csv(checkpoints[1], header=[0, 1], index_col=[0, 1])

    def _create_empty_df(self):
        row_mindex = pd.MultiIndex.from_arrays([[], []], names=["Metrics", "Model"])
        col_index = pd.Index([], name='Dataset')
        col_mindex = pd.MultiIndex.from_arrays([[], []], names=["Metrics", "Model"])
        self.df = pd.DataFrame(index=row_mindex, columns=col_index)
        self.df_type = pd.DataFrame(index=row_mindex, columns=col_mindex)

    def insert(self, data_name, res, res_type):
        """Insert the performance evaluation results from velovae.post_analysis

        Arguments
        ---------
        data_name : str
            Name of the dataset
        res : `pandas.DataFrame`
            Contains performance metrics for the entire dataset.
            Rows are the performance metrics
            columns are model names
        res_type : `pandas.DataFrame`
            Contains the velocity and time metrics for each pair of
            cell type transition. Rows are different performance metrics
            while columns are indexed by method and cell type transitions.
        Returns
        -------
        Inserts res and res_type to self.df and self.df_type
        """
        # Collapse the dataframe to 1D series with multi-index
        res_1d = pd.Series(res.values.flatten(), index=pd.MultiIndex.from_product([res.index, res.columns]))
        for x in res_1d.index:
            self.df.loc[x, :] = pd.Series(None, self.df.columns, dtype=float)
        self.df.loc[res_1d.index, data_name] = res_1d

        # Reshape the data in res_type to match the multi-row-index in self.df_type
        methods = np.unique(res_type.columns.get_level_values(0))
        row_index = pd.MultiIndex.from_product([res_type.index, methods])
        vals = np.concatenate([res_type[x].values for x in methods], axis=0)
        transitions = np.unique(res_type.columns.get_level_values(1))
        for i, pair in enumerate(transitions):
            if (data_name, pair) not in self.df_type.columns:
                self.df_type.insert(self.df_type.shape[1],
                                    (data_name, pair),
                                    np.ones((self.df_type.shape[0]))*np.nan)
        for j, row in enumerate(row_index):
            self.df_type.loc[row, data_name] = vals[j]
        return

    def save(self, path, file_name=None):
        os.makedirs(path, exist_ok=True)
        if file_name is None:
            file_name = "perf"
        self.df.to_csv(f"{path}/{file_name}.csv")
        self.df_type.to_csv(f"{path}/{file_name}_type.csv")
import numpy as np
import pandas as pd

class DataProcessor:
    def __init__(self, file_path, top_n):
        self.file_path = file_path
        self.top_n = top_n
        self.data = None
        self.data_avg = None
        self.data_max = None
        
    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        
    def preprocess_data(self):
        # Select top n columns with most non-NaN values (excluding 'TimeStamp')
        count_non_nan = np.count_nonzero(~pd.isna(self.data), axis=0)
        count_series = pd.Series(count_non_nan, index=self.data.columns)
        top_cols = count_series.nlargest(self.top_n + 1)  # Include 'TimeStamp' in count
        self.data = self.data[top_cols.index]

    def filter_data(self):
        non_nan_row_mask = self.data.notnull().all(axis=1)
        non_nan_row_indices = self.data[non_nan_row_mask].index.tolist()
        continuous_indices = np.split(non_nan_row_indices, np.where(np.diff(non_nan_row_indices) != 1)[0]+1)
        continuous_indices = [seq for seq in continuous_indices if len(seq) > 1]
        largest_sequence = max(continuous_indices, key=len)
        self.data = self.data.loc[largest_sequence]

    def resample_data(self):
        self.data['TimeStamp'] = pd.to_datetime(self.data['TimeStamp'], format='%d/%m/%Y %H:%M')
        self.data = self.data.set_index('TimeStamp')
        self.data_avg = self.data.resample('30T').mean()
        self.data_max = self.data.resample('30T').max()
        
    def save_data(self, save_path_avg, save_path_max):
        self.data_avg.to_csv(save_path_avg, index=True)
        self.data_max.to_csv(save_path_max, index=True)


# Use it like this
num_columns = 20
processor = DataProcessor('D:\\18 Battlefield Road substation\\C2C_average_power_final.csv', num_columns)
processor.load_data()
processor.preprocess_data()
processor.filter_data()
processor.resample_data()
processor.save_data('D:\\18 Battlefield Road substation\\data_avg.csv', 'D:\\18 Battlefield Road substation\\data_max.csv')


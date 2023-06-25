import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

class load_forecast:
    def __init__(self, file_path):
        self.file_path = file_path
        self.model = linear_model.LinearRegression()

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        self.data['Timestamp'] = pd.to_datetime(self.data['Timestamp'])

    def preprocess(self):
        self.data['Air Temperature'] = self.data['Air Temperature']
        self.data['hour'] = self.data['Timestamp'].dt.hour
        self.data['minute'] = self.data['Timestamp'].dt.minute
        self.data['month'] = self.data['Timestamp'].dt.month
        self.data['dayofweek'] = self.data['Timestamp'].dt.dayofweek

    def train_and_predict(self, forecast_horizon):
        TMP = self.data['Air Temperature'].values#can we be sure these labels will always be in the data
        hod = self.data['hour'].values
        moh = self.data['minute'].values
        mon = self.data['month'].values
        dow = self.data['dayofweek'].values

        X=np.array([dow*hod,mon,mon*TMP,mon*TMP*TMP,mon*TMP*TMP*TMP,hod*TMP,hod*TMP*TMP,hod*TMP*TMP*TMP]).T
        y = self.data['crawfordCrescent_F5'].values#hardcoded target!

        # Shift the target forecast_horizon units ahead
        target = y[forecast_horizon:]  # predict value is 48 after
        trainX = X[:15984]  # Use the first 11 months samples for training
        trainTarget = target[:15984]  # Use the first 11 months samples for training - again - hardcoding
        self.model.fit(trainX, trainTarget)
        self.testX = X[15984:-forecast_horizon]
        self.testDates = self.data['Timestamp'][15984+forecast_horizon:]

        

        self.predictedY = self.model.predict(self.testX)
        # The actual values are forecast_horizon units ahead
        self.testY = target[15984:]
        self.error = self.testY - self.predictedY#this is good

    #don't couple to UI - may not be running this on a system with a display
    def plot_results(self):
        fig, ax = plt.subplots()
        x_values = list(range(len(self.testY)))
        ax.plot(x_values, self.testY, color='blue', label='Actual values')
        ax.plot(x_values, self.predictedY, color='red', label='Predicted values')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title('Actual vs Predicted Values')
        ax.legend()
        plt.show()

    #don't couple to read/write of data
    def save_results_to_csv(self,forecast_horizon):        
        result = pd.DataFrame({
            'Date': self.testDates,
            'Predicted': self.predictedY,
            'Actual': self.testY,
            'Error': self.error,
        })
        filename = f'prediction_results_with_{forecast_horizon}.csv'
        result.to_csv('filename.csv', index=False)
    
    #this is good - we can move it to a driver file
    def plot_error_cdf(self):
        # calculate histogram
        hist, bin_edges = np.histogram(self.error, bins='auto', density=True)
        cum_values = np.zeros(bin_edges.shape)
        cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))

        # plot histogram + cumulative histogram
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:blue'
        ax1.set_xlabel('Error')
        ax1.set_ylabel('PDF', color=color)
        ax1.hist(bin_edges[:-1], bin_edges, weights=hist, color=color, alpha=0.6)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('CDF', color=color)
        ax2.plot(bin_edges, cum_values, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        plt.title('PDF and CDF of Prediction Error')
        plt.show()

# Example usage:
forecast_horizon = 48
file_url = "https://raw.githubusercontent.com/Junyi1111/Junyi_Strath/main/benchmark/flex_networks.csv"
predictor = load_forecast(file_url)
predictor.load_data()
predictor.preprocess()
predictor.train_and_predict(forecast_horizon)
predictor.plot_results()
predictor.save_results_to_csv(forecast_horizon)
predictor.plot_error_cdf()
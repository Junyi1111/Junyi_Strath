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
        TMP = self.data['Air Temperature'].values
        hod = self.data['hour'].values
        moh = self.data['minute'].values
        mon = self.data['month'].values
        dow = self.data['dayofweek'].values

        X=np.array([dow*hod,mon,mon*TMP,mon*TMP*TMP,mon*TMP*TMP*TMP,hod*TMP,hod*TMP*TMP,hod*TMP*TMP*TMP]).T
        y = self.data['crawfordCrescent_F5'].values

        # Shift the target forecast_horizon units ahead
        target = y[forecast_horizon:]
        trainX = X[:15984]  # Use the first 15984 samples for training
        trainTarget = target[:15984]  # Use the first 15984 samples for training
        self.model.fit(trainX, trainTarget)
        # Note that we need to keep the last forecast_horizon units of X as well, since they are needed for future prediction
        self.testX = X[15984:-forecast_horizon]
        self.testDates = self.data['Timestamp'][15984+forecast_horizon:]

        

        # Use all but the last forecast_horizon units for prediction
        self.predictedY = self.model.predict(self.testX)
        # The actual values are forecast_horizon units ahead
        self.testY = target[15984:]
        self.error = self.testY - self.predictedY

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

    def save_results_to_csv(self,forecast_horizon):
        # Adjust the dates and actual values to match the length of predicted values
        
        result = pd.DataFrame({
            'Date': self.testDates,
            'Predicted': self.predictedY,
            'Actual': self.testY,
            'Error': self.error,
        })
        filename = f'prediction_results_with_{forecast_horizon}.csv'
        result.to_csv('filename.csv', index=False)

# Example usage:
forecast_horizon = 48
file_url = "https://raw.githubusercontent.com/Junyi1111/Junyi_Strath/main/benchmark/flex_networks.csv"
predictor = load_forecast(file_url)
predictor.load_data()
predictor.preprocess()
predictor.train_and_predict(forecast_horizon)
predictor.plot_results()
predictor.save_results_to_csv(forecast_horizon)




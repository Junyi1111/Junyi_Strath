import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

#no need to paste this in - just import it
class tao_vanilla_model:
    def __init__(self, label='Linear Regression Benchmark Model')-> None:
        self.taovanilla = linear_model.LinearRegression()

    def __str__(self) -> str:
        return self.label
    
    def load_model_from_disk(cls,fname):
        with open(fname, 'rb') as fid:
            return pickle.load(fid)

    def save_model_to_disk(self,fname):
        with open(fname, 'wb') as fid:
           pickle.dump(self,fid)
        
        return
    
    @classmethod
    def load_model_from_disk(cls, fname):
        with open(fname, 'rb') as fid:
            return pickle.load(fid)

    def save_model_to_disk(self, fname):
        with open(fname, 'wb') as fid:
            pickle.dump(self, fid)
        
    @staticmethod
    def hour_of_year(dt):
        datetime_series = pd.Series(dt)
        start_times = datetime_series.apply(lambda dt: pd.Timestamp(year=dt.year, month=1, day=1, hour=0, minute=30))
        time_difference = (datetime_series - start_times) / pd.Timedelta(minutes=30)
        trend = time_difference + 1
        return trend.astype(int)
    
    def form_vanilla_covariates(hrTrnd,ambT,hod,mod,mon,dow):
        X=np.array([hrTrnd,dow*hod,mon,mon*ambT,mon*ambT*ambT,mon*ambT*ambT*ambT,hod*ambT,hod*ambT*ambT,hod*ambT*ambT*ambT]).T

        return X
    def train_model(self, timestamp, temperature, horizon, load):
        hod=np.array(timestamp.hour)
        mod=np.array(timestamp.minute)
        mon=np.array(timestamp.month)
        dow=np.array(timestamp.dayofweek)
        hour_trend=self.hour_of_year(timestamp)
        x=tao_vanilla_model.form_vanilla_covariates(hour_trend,temperature,hod,mod,mon,dow)
        x=x[:-horizon]
        y = load[horizon:]  
        self.taovanilla.fit(x, y)
        return 
    def forecast(self,timestamp,temperature,horizon,load):
        #to do - if horizon is multiple of existing model - apply recursively
        #if not, apply recursively and interpolate
        hod=np.array(timestamp.hour)#roll these forward for increased horizons
        mod=np.array(timestamp.minute)
        mon=np.array(timestamp.month)
        dow=np.array(timestamp.dayofweek)
        hour_trend=tao_vanilla_model.hour_of_year(timestamp)
        x=tao_vanilla_model.form_vanilla_covariates(hour_trend,temperature,hod,mod,mon,dow)
        x=x[:-horizon]
        yHat=self.taovanilla.predict(x)
        y=load[horizon:]
        result=pd.DataFrame({
            'Predicted': yHat,
            'Time': timestamp[horizon:],
            'Actual': y
        })
        return result



#should be able to delete the above and replace with:
#from tao_vanilla_model import tao_vanilla_model
#everything below is great and should still work...
# Example usage:
flxnet=pd.read_csv('https://raw.githubusercontent.com/Junyi1111/Junyi_Strath/main/benchmark/flex_networks.csv')
ts1=pd.DatetimeIndex(flxnet.Timestamp)[:15984]
load1=flxnet.kinnessPark_F4[:15984]
aT1=flxnet['Air Temperature'][:15984]
horizon=48
ts2=pd.DatetimeIndex(flxnet.Timestamp)[15984:]
load2=flxnet.kinnessPark_F4[15984:]
aT2=flxnet['Air Temperature'][15984:]

model= tao_vanilla_model(label='Linear Regression Benchmark Model')
model.train_model(ts1, aT1, horizon, load1)
model.forecast(ts2, aT2, horizon, load2)


ts3=pd.DatetimeIndex(flxnet.Timestamp)[5760:5760+9600+48]
load3=flxnet.kinnessPark_F4[5760:5760+9600+48]
aT3=flxnet['Air Temperature'][5760:5760+9600+48]
result=model.forecast(ts3, aT3, horizon, load3)
error=result.Actual-result.Predicted
error = error.reset_index(drop=True)

half_error=error[:-4800]
half_error = np.array(half_error)
errors_reshaped = half_error.reshape((100,48))
errors_reshaped=errors_reshaped.T
means = np.mean(errors_reshaped, axis=1)
covariances = np.cov(errors_reshaped, rowvar=True)
error_list=[4800+48*i for i in range(100)]
observer_error = error[error_list]
observer_error = observer_error.reset_index(drop=True)
updated_error = []
for i in range(47):
    column = []
    for j in range(100):#can use the @ matrix multiplier
        updated_value = means[i+1] + covariances[0, i+1] * (covariances[0, 0]**(-1)) * (observer_error[j] - means[0])
        column.append(updated_value)
    updated_error.append(column)
updated_error = np.vstack((observer_error, updated_error))
secondhalf_error=error[4800:]
secondhalf_error = np.array(secondhalf_error)
secondhalf_error_reshaped = secondhalf_error.reshape((100,48))
secondhalf_error=secondhalf_error_reshaped.T

updated_error1 = updated_error.T.flatten()
error_original=result.Actual[4800:]-result.Predicted[4800:]
error_update=result.Actual[4800:]-(result.Predicted[4800:]+updated_error1)
mae_original = mean_absolute_error(result.Actual[4800:], result.Predicted[4800:])
mse_original = mean_squared_error(result.Actual[4800:], result.Predicted[4800:])
mae_update = mean_absolute_error(result.Actual[4800:], result.Predicted[4800:]+updated_error1)
mse_update = mean_squared_error(result.Actual[4800:], result.Predicted[4800:]+updated_error1)
def smape(y_true, y_pred):
    return np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
mape_original = smape(result.Actual[4800:], result.Predicted[4800:])
mape_update = smape(result.Actual[4800:], result.Predicted[4800:]+updated_error1)

fig, axs = plt.subplots(4, 5, figsize=(15, 12))
errors = []

print("Original Errors - MAE: ", mae_original, ", MSE: ", mse_original, ", MAPE: ", mape_original)
print("Updated Errors - MAE: ", mae_update, ", MSE: ", mse_update, ", MAPE: ", mape_update)
for i in range(20):
    row = i // 5
    col = i % 5

    actual_values = secondhalf_error[:, i]
    predicted_values = updated_error[:, i]
    error = np.abs(actual_values - predicted_values)
    errors.append(error)

    axs[row, col].plot(list(range(48)), actual_values, color='blue', label='Actual error values')
    axs[row, col].plot(list(range(48)), predicted_values, color='red', label='Predicted error values')
    axs[row, col].set_title('Actual vs Predicted Values - Day {}'.format(i))
    axs[row, col].legend()

plt.tight_layout()  
plt.show()

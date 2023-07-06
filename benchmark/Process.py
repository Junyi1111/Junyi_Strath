import numpy as np
import pandas as pd
from weather import WeatherData
from vanilla import tao_vanilla_model
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from ErrorAnalysis import ErrorAnalysis
weather_data = WeatherData('AIzaSyD1qHEwXih2-63HJC77ImISVlxx22vHlqg')
df_weather_interpolated = weather_data.get_interpolated_weather_by_postal_code('BL4 8EA', '2013-06-20', '2014-01-25')
flxnet=pd.read_csv('https://raw.githubusercontent.com/Junyi1111/Junyi_Strath/main/benchmark/flex_networks.csv')
ts1=pd.DatetimeIndex(flxnet.Timestamp)
load1=flxnet.kinnessPark_F4
aT1=flxnet['Air Temperature']
horizon=48  #one day ahead
model= tao_vanilla_model(label='Linear Regression Benchmark Model')
model.train_model(ts1, aT1, horizon, load1)
load =pd.read_csv('https://raw.githubusercontent.com/Junyi1111/Junyi_Strath/main/benchmark/actual%20power.csv')
ts3=pd.DatetimeIndex(df_weather_interpolated.time)[1:-45]
load3=load.ROUSEVELTRD_avg_power
aT3=df_weather_interpolated.temperature[1:-45]
result=model.forecast(ts3, aT3, horizon, load3)
result = result.reset_index(drop=True)
error=result.Actual-result.Predicted
analysis = ErrorAnalysis(result)
model = analysis.calculate_errors()
result_update = model['result_update']
actual_load=result.Actual[len(result)//2:]
original_forecast=result.Predicted[len(result)//2:]
result_update=result_update.to_numpy()
actual_load=actual_load.to_numpy()
original_forecast=original_forecast.to_numpy()
print(model)

result_update = result_update.reshape(-1, 48)
actual_load = actual_load.reshape(-1, 48)
original_forecast = original_forecast.reshape(-1, 48)

fig, ax = plt.subplots(2, 1, figsize=(10, 15))

for i, (data, title) in enumerate(zip([actual_load-result_update, actual_load-original_forecast], 
                                       ['Result Update', 'original_forecast'])):
    # Plot a box plot for each group of 48 data points
    sns.boxplot(data=data, ax=ax[i])
    ax[i].set_title(title)
    ax[i].set_xlabel('Group number')
    ax[i].set_ylabel('Value')

plt.tight_layout()
plt.show()


fig, ax = plt.subplots(5, 1, figsize=(10, 15))

# Plot line plots for the first 5 groups of 48 data points
for i in range(5):
    sns.lineplot(data=result_update[i, :], ax=ax[i], label='Result Update')
    sns.lineplot(data=actual_load[i, :], ax=ax[i], label='Actual Load')
    sns.lineplot(data=original_forecast[i, :], ax=ax[i], label='Original Forecast')
    ax[i].set_title(f'Comparison Plot for Day {i+1} of 48 Data Points')
    ax[i].set_xlabel('Data Point Index within the Group')
    ax[i].set_ylabel('Value')

plt.tight_layout()
plt.show()


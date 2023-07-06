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






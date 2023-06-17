import pandas as pd
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime
from typing import List
import pickle

class tao_vanilla_model:
    def __init__(self) -> None:
        self.taovanilla = linear_model.LinearRegression()
        self.resolution = 48#assume 48 HH in day 
        self.horizon = self.resolution#and its one day ahead 

    def __init__(self,res) -> None:
        self.taovanilla = linear_model.LinearRegression()
        self.resolution = res
        self.horizon = self.resolution
    
    def __str__(self) -> str:
        return 'Linear Regression Benchmark Model'

    @classmethod
    def load_model_from_disk(cls,fname):
        with open(fname, 'rb') as fid:
            return pickle.load(fid)

    def save_model_to_disk(self,fname):
        with open(fname, 'wb') as fid:
           pickle.dump(self,fid)
        
        return
    
    #don't understand this... 
    def hour_trend_of_year(timestamps: List[str], timestamp_format="%d/%m/%Y %H:%M") -> List[int]:
        def _single_hour_trend(timestamp_obj):
            days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

            if (timestamp_obj.year % 4 == 0 and timestamp_obj.year % 100 != 0) or timestamp_obj.year % 400 == 0:
                days_in_month[1] = 29

            hour_trend = timestamp_obj.hour + (timestamp_obj.day - 1) * 24

            for month in range(timestamp_obj.month - 1):
                hour_trend += days_in_month[month] * 24

            return hour_trend

        hour_trends = [_single_hour_trend(datetime.strptime(ts, timestamp_format)) for ts in timestamps]
        
        return hour_trends
    
    def form_lagged_target(load,lags):
        Y=np.array([])

        return Y

    def form_vanilla_covariates(hrTrnd,ambT,hod,mod,mon,dow):
        X=np.array([hrTrnd,dow*hod,mon,mon*ambT,mon*ambT*ambT,mon*ambT*ambT*ambT,hod*ambT,hod*ambT*ambT,hod*ambT*ambT*ambT]).T

        return X
    
    def train_model(self,timestamp,temperature,load):        
        hod=np.array(timestamp.hour)
        mod=np.array(timestamp.minute)
        mon=np.array(timestamp.month)
        dow=np.array(timestamp.dayofweek)
        new_format = "%d-%b-%Y %H:%M:%S"
        hour_trend=tao_vanilla_model.hour_trend_of_year(timestamp,new_format)

        X=tao_vanilla_model.form_vanilla_covariates(hour_trend,temperature,hod,mod,mon,dow)
        Y=tao_vanilla_model.form_lagged_target(load,self.horizon)

        (T,d)=X.shape

        X=np.concat(X[:T-self.horizon],Y[:T-self.horizon,1])
        Y=Y[1+self.horizon:,2]

        self.taovanilla.fit(X,Y)

        return
    
    def forecast(self,timestamp,temperature,load):
        return self.forecast(timestamp,temperature,load,self.horizon)

    def forecast(self,timestamp,temperature,load,horizon):
        #to do - if horizon is multiple of existing model - apply recursively
        #if not, apply recursively and interpolate
        hod=np.array(timestamp.hour)
        mod=np.array(timestamp.minute)
        mon=np.array(timestamp.month)
        dow=np.array(timestamp.dayofweek)
        new_format = "%d-%b-%Y %H:%M:%S"
        hour_trend=tao_vanilla_model.hour_trend_of_year(timestamp,new_format)

        X=tao_vanilla_model.form_vanilla_covariates(hour_trend,temperature,hod,mod,mon,dow)
        
        yHat=self.taovanilla.predict(X)
        
        return yHat

flxnet=pd.read_csv('flex_networks.csv')

taovanilla = tao_vanilla_model(48)


hour_trends=hour_trend_of_year(Roosevelt_station.Timestamp)
TMP_test = np.array(Roosevelt_station['Temperature_2m'])
hod_test=np.array(pd.DatetimeIndex(Roosevelt_station.Timestamp).hour)
moh_test=np.array(pd.DatetimeIndex(Roosevelt_station.Timestamp).minute)
mon_test=np.array(pd.DatetimeIndex(Roosevelt_station.Timestamp).month)
dow_test=np.array(pd.DatetimeIndex(Roosevelt_station.Timestamp).dayofweek)

observed=flxnet.kinnessPark_F4[0:-49]

#data = df['observed_values'].values.reshape(-1, 1)
hod=np.array(pd.DatetimeIndex(flxnet.Timestamp[0:-49]).hour)
moh=np.array(pd.DatetimeIndex(flxnet.Timestamp[0:-49]).minute)
mon=np.array(pd.DatetimeIndex(flxnet.Timestamp[0:-49]).month)
dow=np.array(pd.DatetimeIndex(flxnet.Timestamp[0:-49]).dayofweek)

X=np.array([hod,moh,mon,dow,observed]).T

target=np.array(flxnet.kinnessPark_F4[49:])

trainX=X[0:30*48,:]
trainTarget=target[0:30*48]

taovanilla.fit(trainX,trainTarget)

taovanilla.save_model_to_disk('flex_networks_stlf.pkl')

testX=X[1+(30*48):60*48,:]
testTarget=target[1+(30*48):60*48]

targetHat=taovanilla.forecast(testX)

error=testTarget-targetHat

#next thing to do is turn the error into a mean and covariance
blockErr=np.reshape(error[0:-47],(-1,48))
muErr=np.mean(blockErr,axis=0)
sigErr=np.cov(blockErr,rowvar=False)

#estimate intra-day error using conditional Gaussian form of joint error
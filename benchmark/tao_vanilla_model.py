import pandas as pd
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime
from typing import List
import pickle

#vanilla linear benchmark - proposed by T. Hong for GefCom14
#eventually will have superclass that generalises this functionality
#author: J. Lu, B.Stephen
class tao_vanilla_model:
    def __init__(self) -> None:
        self.taovanilla = linear_model.LinearRegression()
        self.resolution = 48#assume 48 HH in day - actually irrelevant for this model
        self.horizon = self.resolution#and its one day ahead - again not relevant here
        self.label = ''

    def __init__(self,res) -> None:
        self.taovanilla = linear_model.LinearRegression()
        self.resolution = res
        self.horizon = self.resolution
    
    def __str__(self) -> str:
        return self.label+'Linear Regression Benchmark Model'

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
    
    #def hour_of_year(dt): 
    #    beginning_of_year = datetime(dt.year, 1, 1, tzinfo=dt.tzinfo)
    #    return (dt - beginning_of_year).total_seconds() // 3600
    
    def hour_of_year(dt): 
        return dt.hour+dt.dayofyear*24
    
    def form_vanilla_covariates(hrTrnd,ambT,hod,mod,mon,dow):
        X=np.array([hrTrnd,dow*hod,mon,mon*ambT,mon*ambT*ambT,mon*ambT*ambT*ambT,hod*ambT,hod*ambT*ambT,hod*ambT*ambT*ambT]).T

        return X
    
    def train_model(self,timestamp,temperature,load):        
        hod=np.array(timestamp.hour)
        mod=np.array(timestamp.minute)
        mon=np.array(timestamp.month)
        dow=np.array(timestamp.dayofweek)
        hour_trend=tao_vanilla_model.hour_of_year(timestamp)

        #need to do various checks here - timestamp validity
        #lengths of covariate sequences etc

        X=tao_vanilla_model.form_vanilla_covariates(hour_trend,temperature,hod,mod,mon,dow)
        Y=load

        #this is all there is to it - for a date, time and temperature estimate
        #do a point prediction of load
        self.taovanilla.fit(X,Y)

        return
    
    #make this a base class method - retained for
    #basic probabilistic forecasting
    def evaluate_error_stats(self,tstamp,Y,Yhat):
        inc=1440//self.resolution#increment
        mul=self.resolution//24#hour multiplier
        idx=(mul*ts.hour+ts.minute//inc)
        mxIdx=max(idx)
        error=Y-Yhat
        
        self.muIntraday=np.array((self.resolution,1),dtype=float)
        self.sigIntraday=np.array((self.resolution,self.resolution),dtype=float)

        for i in range(1,mxIdx):
            self.muIntraday[i-1]=np.mean(error[idx==i],axis=0)
            self.sigIntraday[i-1][i-1]=np.var(error[idx==i],axis=0)

            for j in range(1,mxIdx):
                cov=np.cov(error[idx==i],error[idx==j])
                self.sigIntraday[i-1][j-1]=cov
                self.sigIntraday[j-1][i-1]=cov

        ##not general - assumes a 48 HH resolution...
        #blockErr=np.reshape(error[0:-self.horizon-1],(-1,self.horizon))
        #mu=np.mean(blockErr,axis=0)
        #sigma=np.cov(blockErr,rowvar=False)

        return self.muIntraday,self.sigIntraday

    def forecast(self,timestamp,temperature):
        return self.forecast(timestamp,temperature,self.horizon)

    def forecast(self,timestamp,temperature,horizon):
        #to do - if horizon is multiple of existing model - apply recursively
        #if not, apply recursively and interpolate
        hod=np.array(timestamp.hour)#roll these forward for increased horizons
        mod=np.array(timestamp.minute)
        mon=np.array(timestamp.month)
        dow=np.array(timestamp.dayofweek)
        hour_trend=tao_vanilla_model.hour_of_year(timestamp)
        
        X=tao_vanilla_model.form_vanilla_covariates(hour_trend,temperature,hod,mod,mon,dow)
        
        yHat=self.taovanilla.predict(X)
        
        return yHat

    #todo: include methods for rolling forward timestamps
    #and also increasing forecast resolution...

flxnet=pd.read_csv('flex_networks.csv')

taovanilla = tao_vanilla_model(48)

ts=pd.DatetimeIndex(flxnet.Timestamp)
ld=flxnet.kinnessPark_F4
ts2=pd.DatetimeIndex(flxnet.Timestamp)
testTarget=flxnet.kinnessPark_F2
aT=flxnet['Air Temperature']

taovanilla.train_model(ts,aT,ld)
taovanilla.save_model_to_disk('flex_networks_stlf.pkl')

taovanilla2=tao_vanilla_model.load_model_from_disk('flex_networks_stlf.pkl')

targetHat=taovanilla.forecast(ts2,aT,0)#wrong! won't recognise overloaded method though...

#next thing to do is turn the error into a mean and covariance
muErr,sigErr=taovanilla.evaluate_error_stats(ts2,testTarget,targetHat)

#estimate intra-day error using conditional Gaussian form of joint error
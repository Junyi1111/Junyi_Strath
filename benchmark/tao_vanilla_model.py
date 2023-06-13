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
    
    def train_model(self,covariates,target):
        
        
        TMP = np.array(flxnet_hour['Air Temperature'])
        hod=np.array(pd.DatetimeIndex(flxnet_hour.Timestamp).hour)
        moh=np.array(pd.DatetimeIndex(flxnet_hour.Timestamp).minute)
        mon=np.array(pd.DatetimeIndex(flxnet_hour.Timestamp).month)
        dow=np.array(pd.DatetimeIndex(flxnet_hour.Timestamp).dayofweek)
        new_format = "%d-%b-%Y %H:%M:%S"
        hour_trends=hour_trend_of_year(flxnet_hour.Timestamp,new_format)
        
        X=np.array([hour_trends,dow*hod,mon,mon*TMP,mon*TMP*TMP,mon*TMP*TMP*TMP,hod*TMP,hod*TMP*TMP,hod*TMP*TMP*TMP]).T

        target=np.array(flxnet_hour.crawfordCrescent_F5[24:])
        trainX=X[0:8736]
        trainTarget=target[0:8736]
        
        self.taovanilla.fit(trainX,trainTarget)

        return
    
    def forecast(self,covariates,horizon):
        hour_trends=hour_trend_of_year(Roosevelt_station.Timestamp)
        TMP_test = np.array(Roosevelt_station['Temperature_2m'])
        hod_test=np.array(pd.DatetimeIndex(Roosevelt_station.Timestamp).hour)
        moh_test=np.array(pd.DatetimeIndex(Roosevelt_station.Timestamp).minute)
        mon_test=np.array(pd.DatetimeIndex(Roosevelt_station.Timestamp).month)
        dow_test=np.array(pd.DatetimeIndex(Roosevelt_station.Timestamp).dayofweek)
        X_test = np.array([hour_trends, dow_test * hod_test, mon_test, mon_test * TMP_test, mon_test * TMP_test * TMP_test, mon_test * TMP_test * TMP_test * TMP_test, hod_test * TMP_test, hod_test * TMP_test * TMP_test, hod_test * TMP_test * TMP_test * TMP_test], dtype=object).T

        testX=X_test
        targetHat=self.taovanilla.predict(testX)
        
        targetHat1 = pd.DataFrame({"Timestamp": Roosevelt_station.Timestamp, "Data": targetHat})
        targetHat=targetHat[14:]

        return
    
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

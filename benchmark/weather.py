#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
import pandas as pd
import numpy as np
from scipy import interpolate

class WeatherData:
    def __init__(self, google_maps_api_key):
        self.google_maps_api_key = google_maps_api_key

    def get_lat_lng(self, postal_code):
        try:
            url = 'https://maps.googleapis.com/maps/api/geocode/json'
            params = {'address': postal_code, 'key': self.google_maps_api_key}
            response = requests.get(url, params=params)
            data = response.json()
            lat_lng = data['results'][0]['geometry']['location']
            return lat_lng['lat'], lat_lng['lng']
        except Exception as e:
            print(f"Error: {e}")
            return None, None

    def get_weather_data(self, lat, lng, start_date, end_date):
        url = "https://archive-api.open-meteo.com/v1/archive"
        parameters = {
            "latitude": lat,
            "longitude": lng,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m"
        }
        response = requests.get(url, params=parameters)
        if response.status_code == 200:
            data = response.json()
            if "hourly" in data and "time" in data["hourly"] and "temperature_2m" in data["hourly"]:
                times = pd.to_datetime(data["hourly"]["time"])
                temperatures = data["hourly"]["temperature_2m"]
                df = pd.DataFrame({'time': times, 'temperature': temperatures})
                return df
        return pd.DataFrame()

    def get_weather_by_postal_code(self, postal_code, start_date, end_date):
        lat, lng = self.get_lat_lng(postal_code)
        if lat and lng:
            return self.get_weather_data(lat, lng, start_date, end_date)
        else:
            return None

    def get_interpolated_weather_by_postal_code(self, postal_code, start_date, end_date):
        df_weather = self.get_weather_by_postal_code(postal_code, start_date, end_date)
        if df_weather is not None:
            # convert the 'time' column to a numeric format (e.g., seconds since a reference time)
            df_weather['time_numeric'] = (df_weather['time'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

            # create a cubic spline interpolation function using scipy's interpolate function
            spline = interpolate.CubicSpline(df_weather.time_numeric, df_weather.temperature)

            # generate new times at 30 minute intervals
            new_times = np.arange(df_weather.time_numeric.min(), df_weather.time_numeric.max(), 30*60)  # 30 minutes = 1800 seconds

            # use the cubic spline function to interpolate the temperature at these new times
            new_temperatures = spline(new_times)

            # create a new dataframe with the interpolated values
            df_weather_interpolated = pd.DataFrame({'time': pd.to_datetime(new_times, unit='s'), 'temperature': new_temperatures})

            return df_weather_interpolated
        else:
            return None


#class to provide general representation of radial
#network; will hold load flow outputs as well as electrical
#and geographic topology.
#author: bstephen
#date: 2nd July 2023

import pandas as pd
import random
import os
import pickle

#increase...

class load:
    def __init__(self):
        self.loadprofile=None
        self.timestamp=None

    def set_profile(self,ts,ld):
        self.timestamp=ts
        self.loadprofile=ld

class power_transformer:
    def __init__(self) -> None:
        pass

class load_bus:
    def __init__(self):
        self.fed=None
        self.feeds=None

class power_line:
    def __init__(self):
        self.begin=None
        self.end=None

class radial_power_network:
    def __init__(self):
        self.buses=list()
        self.lines=list()
        self.transformer=None
        self.loads=list()
        self.label=None

    def __str__(self) -> str:
        return self.label+'_radial_network'

    @classmethod
    def load_model_from_disk(cls,fname):
        with open(fname, 'rb') as fid:
            return pickle.load(fid)

    def save_model_to_disk(self,fname):
        with open(fname, 'wb') as fid:
           pickle.dump(self,fid)
        
        return

lfname='lcl_smart_meter.pkl'

loadDat=pd.read_csv('C:\\Users\\ajp97161\\OneDrive - University of Strathclyde\\models_RT\\lv_estimation\\lcl_clean_widefmt.csv')
cols=loadDat.columns
loads=len(cols)-1

lds=list()

for c in range(1,loads):
    l=load()
    l.set_profile(loadDat[cols[0]],loadDat[cols[c]])
    lds.append(l)

with open(lfname, 'wb') as fid:
    pickle.dump(lds,fid)

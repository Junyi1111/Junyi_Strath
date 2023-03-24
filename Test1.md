```python
import pandapower as pp #import pandapower
import numpy as np
import pandas as pd
import os
import random
import pandapower as pp
from pandapower.timeseries import DFData
from pandapower.timeseries import OutputWriter
from pandapower.timeseries.run_time_series import run_timeseries
from pandapower.control import ConstControl
net = pp.create_empty_network() #create an empty network
```


```python
bus1 = pp.create_bus(net, name="HV Busbar", vn_kv=33, type="b")
bus2 = pp.create_bus(net, name="HV Transformer Bus", vn_kv=33, type="n")
bus3 = pp.create_bus(net, name="LV Transformer Bus", vn_kv=11, type="n")
for i in range(0,20):
    pp.create_bus(net, name='LV Bus %s' % i, vn_kv=11, type='b')
net.bus
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>vn_kv</th>
      <th>type</th>
      <th>zone</th>
      <th>in_service</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>HV Busbar</td>
      <td>33.0</td>
      <td>b</td>
      <td>None</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>HV Transformer Bus</td>
      <td>33.0</td>
      <td>n</td>
      <td>None</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LV Transformer Bus</td>
      <td>11.0</td>
      <td>n</td>
      <td>None</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LV Bus 0</td>
      <td>11.0</td>
      <td>b</td>
      <td>None</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LV Bus 1</td>
      <td>11.0</td>
      <td>b</td>
      <td>None</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>LV Bus 2</td>
      <td>11.0</td>
      <td>b</td>
      <td>None</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>LV Bus 3</td>
      <td>11.0</td>
      <td>b</td>
      <td>None</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>LV Bus 4</td>
      <td>11.0</td>
      <td>b</td>
      <td>None</td>
      <td>True</td>
    </tr>
    <tr>
      <th>8</th>
      <td>LV Bus 5</td>
      <td>11.0</td>
      <td>b</td>
      <td>None</td>
      <td>True</td>
    </tr>
    <tr>
      <th>9</th>
      <td>LV Bus 6</td>
      <td>11.0</td>
      <td>b</td>
      <td>None</td>
      <td>True</td>
    </tr>
    <tr>
      <th>10</th>
      <td>LV Bus 7</td>
      <td>11.0</td>
      <td>b</td>
      <td>None</td>
      <td>True</td>
    </tr>
    <tr>
      <th>11</th>
      <td>LV Bus 8</td>
      <td>11.0</td>
      <td>b</td>
      <td>None</td>
      <td>True</td>
    </tr>
    <tr>
      <th>12</th>
      <td>LV Bus 9</td>
      <td>11.0</td>
      <td>b</td>
      <td>None</td>
      <td>True</td>
    </tr>
    <tr>
      <th>13</th>
      <td>LV Bus 10</td>
      <td>11.0</td>
      <td>b</td>
      <td>None</td>
      <td>True</td>
    </tr>
    <tr>
      <th>14</th>
      <td>LV Bus 11</td>
      <td>11.0</td>
      <td>b</td>
      <td>None</td>
      <td>True</td>
    </tr>
    <tr>
      <th>15</th>
      <td>LV Bus 12</td>
      <td>11.0</td>
      <td>b</td>
      <td>None</td>
      <td>True</td>
    </tr>
    <tr>
      <th>16</th>
      <td>LV Bus 13</td>
      <td>11.0</td>
      <td>b</td>
      <td>None</td>
      <td>True</td>
    </tr>
    <tr>
      <th>17</th>
      <td>LV Bus 14</td>
      <td>11.0</td>
      <td>b</td>
      <td>None</td>
      <td>True</td>
    </tr>
    <tr>
      <th>18</th>
      <td>LV Bus 15</td>
      <td>11.0</td>
      <td>b</td>
      <td>None</td>
      <td>True</td>
    </tr>
    <tr>
      <th>19</th>
      <td>LV Bus 16</td>
      <td>11.0</td>
      <td>b</td>
      <td>None</td>
      <td>True</td>
    </tr>
    <tr>
      <th>20</th>
      <td>LV Bus 17</td>
      <td>11.0</td>
      <td>b</td>
      <td>None</td>
      <td>True</td>
    </tr>
    <tr>
      <th>21</th>
      <td>LV Bus 18</td>
      <td>11.0</td>
      <td>b</td>
      <td>None</td>
      <td>True</td>
    </tr>
    <tr>
      <th>22</th>
      <td>LV Bus 19</td>
      <td>11.0</td>
      <td>b</td>
      <td>None</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
pp.create_ext_grid(net, bus1, vm_pu=1.02, va_degree=50) # Create an external grid connection

net.ext_grid #show external grid table
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>bus</th>
      <th>vm_pu</th>
      <th>va_degree</th>
      <th>slack_weight</th>
      <th>in_service</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>None</td>
      <td>0</td>
      <td>1.02</td>
      <td>50.0</td>
      <td>1.0</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
hv_bus = pp.get_element_index(net, "bus", "HV Transformer Bus")
lv_bus = pp.get_element_index(net, "bus", "LV Transformer Bus")
pp.create_transformer_from_parameters(net, hv_bus, lv_bus, sn_mva=20, vn_hv_kv=33, vn_lv_kv=11, vkr_percent=0.06,
                                      vk_percent=8, pfe_kw=0, i0_percent=0, tp_pos=0, shift_degree=0, name='HV-LV-Trafo')

net.trafo # show trafo table
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>std_type</th>
      <th>hv_bus</th>
      <th>lv_bus</th>
      <th>sn_mva</th>
      <th>vn_hv_kv</th>
      <th>vn_lv_kv</th>
      <th>vk_percent</th>
      <th>vkr_percent</th>
      <th>pfe_kw</th>
      <th>...</th>
      <th>tap_min</th>
      <th>tap_max</th>
      <th>tap_step_percent</th>
      <th>tap_step_degree</th>
      <th>tap_pos</th>
      <th>tap_phase_shifter</th>
      <th>parallel</th>
      <th>df</th>
      <th>in_service</th>
      <th>tp_pos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>HV-LV-Trafo</td>
      <td>None</td>
      <td>1</td>
      <td>2</td>
      <td>20.0</td>
      <td>33.0</td>
      <td>11.0</td>
      <td>8.0</td>
      <td>0.06</td>
      <td>0.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>1</td>
      <td>1.0</td>
      <td>True</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 24 columns</p>
</div>




```python
test_type = {"r_ohm_per_km": 0.128, "x_ohm_per_km": 0.37, "c_nf_per_km": 0.2, "max_i_ka": 0.257, "type": "cs"}
pp.create_std_type(net, name="test_type", data=test_type, element="line")
```


```python
length=[355.8490816,614.7266682,540.6867661,381.4029833,567.6633177,433.2660756,514.5431465,540.2336011,565.7494381,554.3740258,505.464974
,618.6131674,313.0781694,226.6652451,566.7917887,374.4456499,307.6240851,462.1583225,380.9153266,13.58117217,345.7773542,150.8857917]
```


```python
# create lines
for n in range(0,20):
        from_bus = bus3
        to_bus =  pp.get_element_index(net, "bus", name='LV Bus %s'% n)
        pp.create_line(net, from_bus, to_bus, length_km=length[n]/1000,std_type="test_type",name='LV Line %s'% n)
# show line table
net.line
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>std_type</th>
      <th>from_bus</th>
      <th>to_bus</th>
      <th>length_km</th>
      <th>r_ohm_per_km</th>
      <th>x_ohm_per_km</th>
      <th>c_nf_per_km</th>
      <th>g_us_per_km</th>
      <th>max_i_ka</th>
      <th>df</th>
      <th>parallel</th>
      <th>type</th>
      <th>in_service</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LV Line 0</td>
      <td>test_type</td>
      <td>2</td>
      <td>3</td>
      <td>0.355849</td>
      <td>0.128</td>
      <td>0.37</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.257</td>
      <td>1.0</td>
      <td>1</td>
      <td>cs</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LV Line 1</td>
      <td>test_type</td>
      <td>2</td>
      <td>4</td>
      <td>0.614727</td>
      <td>0.128</td>
      <td>0.37</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.257</td>
      <td>1.0</td>
      <td>1</td>
      <td>cs</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LV Line 2</td>
      <td>test_type</td>
      <td>2</td>
      <td>5</td>
      <td>0.540687</td>
      <td>0.128</td>
      <td>0.37</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.257</td>
      <td>1.0</td>
      <td>1</td>
      <td>cs</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LV Line 3</td>
      <td>test_type</td>
      <td>2</td>
      <td>6</td>
      <td>0.381403</td>
      <td>0.128</td>
      <td>0.37</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.257</td>
      <td>1.0</td>
      <td>1</td>
      <td>cs</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LV Line 4</td>
      <td>test_type</td>
      <td>2</td>
      <td>7</td>
      <td>0.567663</td>
      <td>0.128</td>
      <td>0.37</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.257</td>
      <td>1.0</td>
      <td>1</td>
      <td>cs</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>LV Line 5</td>
      <td>test_type</td>
      <td>2</td>
      <td>8</td>
      <td>0.433266</td>
      <td>0.128</td>
      <td>0.37</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.257</td>
      <td>1.0</td>
      <td>1</td>
      <td>cs</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>LV Line 6</td>
      <td>test_type</td>
      <td>2</td>
      <td>9</td>
      <td>0.514543</td>
      <td>0.128</td>
      <td>0.37</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.257</td>
      <td>1.0</td>
      <td>1</td>
      <td>cs</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>LV Line 7</td>
      <td>test_type</td>
      <td>2</td>
      <td>10</td>
      <td>0.540234</td>
      <td>0.128</td>
      <td>0.37</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.257</td>
      <td>1.0</td>
      <td>1</td>
      <td>cs</td>
      <td>True</td>
    </tr>
    <tr>
      <th>8</th>
      <td>LV Line 8</td>
      <td>test_type</td>
      <td>2</td>
      <td>11</td>
      <td>0.565749</td>
      <td>0.128</td>
      <td>0.37</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.257</td>
      <td>1.0</td>
      <td>1</td>
      <td>cs</td>
      <td>True</td>
    </tr>
    <tr>
      <th>9</th>
      <td>LV Line 9</td>
      <td>test_type</td>
      <td>2</td>
      <td>12</td>
      <td>0.554374</td>
      <td>0.128</td>
      <td>0.37</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.257</td>
      <td>1.0</td>
      <td>1</td>
      <td>cs</td>
      <td>True</td>
    </tr>
    <tr>
      <th>10</th>
      <td>LV Line 10</td>
      <td>test_type</td>
      <td>2</td>
      <td>13</td>
      <td>0.505465</td>
      <td>0.128</td>
      <td>0.37</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.257</td>
      <td>1.0</td>
      <td>1</td>
      <td>cs</td>
      <td>True</td>
    </tr>
    <tr>
      <th>11</th>
      <td>LV Line 11</td>
      <td>test_type</td>
      <td>2</td>
      <td>14</td>
      <td>0.618613</td>
      <td>0.128</td>
      <td>0.37</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.257</td>
      <td>1.0</td>
      <td>1</td>
      <td>cs</td>
      <td>True</td>
    </tr>
    <tr>
      <th>12</th>
      <td>LV Line 12</td>
      <td>test_type</td>
      <td>2</td>
      <td>15</td>
      <td>0.313078</td>
      <td>0.128</td>
      <td>0.37</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.257</td>
      <td>1.0</td>
      <td>1</td>
      <td>cs</td>
      <td>True</td>
    </tr>
    <tr>
      <th>13</th>
      <td>LV Line 13</td>
      <td>test_type</td>
      <td>2</td>
      <td>16</td>
      <td>0.226665</td>
      <td>0.128</td>
      <td>0.37</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.257</td>
      <td>1.0</td>
      <td>1</td>
      <td>cs</td>
      <td>True</td>
    </tr>
    <tr>
      <th>14</th>
      <td>LV Line 14</td>
      <td>test_type</td>
      <td>2</td>
      <td>17</td>
      <td>0.566792</td>
      <td>0.128</td>
      <td>0.37</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.257</td>
      <td>1.0</td>
      <td>1</td>
      <td>cs</td>
      <td>True</td>
    </tr>
    <tr>
      <th>15</th>
      <td>LV Line 15</td>
      <td>test_type</td>
      <td>2</td>
      <td>18</td>
      <td>0.374446</td>
      <td>0.128</td>
      <td>0.37</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.257</td>
      <td>1.0</td>
      <td>1</td>
      <td>cs</td>
      <td>True</td>
    </tr>
    <tr>
      <th>16</th>
      <td>LV Line 16</td>
      <td>test_type</td>
      <td>2</td>
      <td>19</td>
      <td>0.307624</td>
      <td>0.128</td>
      <td>0.37</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.257</td>
      <td>1.0</td>
      <td>1</td>
      <td>cs</td>
      <td>True</td>
    </tr>
    <tr>
      <th>17</th>
      <td>LV Line 17</td>
      <td>test_type</td>
      <td>2</td>
      <td>20</td>
      <td>0.462158</td>
      <td>0.128</td>
      <td>0.37</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.257</td>
      <td>1.0</td>
      <td>1</td>
      <td>cs</td>
      <td>True</td>
    </tr>
    <tr>
      <th>18</th>
      <td>LV Line 18</td>
      <td>test_type</td>
      <td>2</td>
      <td>21</td>
      <td>0.380915</td>
      <td>0.128</td>
      <td>0.37</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.257</td>
      <td>1.0</td>
      <td>1</td>
      <td>cs</td>
      <td>True</td>
    </tr>
    <tr>
      <th>19</th>
      <td>LV Line 19</td>
      <td>test_type</td>
      <td>2</td>
      <td>22</td>
      <td>0.013581</td>
      <td>0.128</td>
      <td>0.37</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.257</td>
      <td>1.0</td>
      <td>1</td>
      <td>cs</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
    p=[16.56666676,26.13333352,22.64999962,142.2833354,175.6166661,37.79999924,48.18333308,225.0833283,78.23333359,136.1999995,154.25,165.2666677
,54.36666616,133.1666667,138.9833349,115.3333346,90.58333334,65.36666679,200,6.216666639]
    q=[5.616666635,10.03333346,7.150000016,35.26666705,54.25000064,10.5,28.04999987,71.0333341,18.38333352,43.36666616,37.00000064,40.98333359,12.71666654,16.83333365
,34.21666654,21.36666679,19.9333334,14.38333321,36.03333346,116.3000005]
```


```python
for n in range(0,20):
    bus_idx = pp.get_element_index(net, "bus", name='LV Bus %s'% n)
    pp.create_load(net, bus_idx, p_mw=p[n]/1000, q_mvar=q[n]/1000,name='LV Load %s'% n)

# show load table
net.load
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>bus</th>
      <th>p_mw</th>
      <th>q_mvar</th>
      <th>const_z_percent</th>
      <th>const_i_percent</th>
      <th>sn_mva</th>
      <th>scaling</th>
      <th>in_service</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LV Load 0</td>
      <td>3</td>
      <td>0.016567</td>
      <td>0.005617</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>True</td>
      <td>wye</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LV Load 1</td>
      <td>4</td>
      <td>0.026133</td>
      <td>0.010033</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>True</td>
      <td>wye</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LV Load 2</td>
      <td>5</td>
      <td>0.022650</td>
      <td>0.007150</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>True</td>
      <td>wye</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LV Load 3</td>
      <td>6</td>
      <td>0.142283</td>
      <td>0.035267</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>True</td>
      <td>wye</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LV Load 4</td>
      <td>7</td>
      <td>0.175617</td>
      <td>0.054250</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>True</td>
      <td>wye</td>
    </tr>
    <tr>
      <th>5</th>
      <td>LV Load 5</td>
      <td>8</td>
      <td>0.037800</td>
      <td>0.010500</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>True</td>
      <td>wye</td>
    </tr>
    <tr>
      <th>6</th>
      <td>LV Load 6</td>
      <td>9</td>
      <td>0.048183</td>
      <td>0.028050</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>True</td>
      <td>wye</td>
    </tr>
    <tr>
      <th>7</th>
      <td>LV Load 7</td>
      <td>10</td>
      <td>0.225083</td>
      <td>0.071033</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>True</td>
      <td>wye</td>
    </tr>
    <tr>
      <th>8</th>
      <td>LV Load 8</td>
      <td>11</td>
      <td>0.078233</td>
      <td>0.018383</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>True</td>
      <td>wye</td>
    </tr>
    <tr>
      <th>9</th>
      <td>LV Load 9</td>
      <td>12</td>
      <td>0.136200</td>
      <td>0.043367</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>True</td>
      <td>wye</td>
    </tr>
    <tr>
      <th>10</th>
      <td>LV Load 10</td>
      <td>13</td>
      <td>0.154250</td>
      <td>0.037000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>True</td>
      <td>wye</td>
    </tr>
    <tr>
      <th>11</th>
      <td>LV Load 11</td>
      <td>14</td>
      <td>0.165267</td>
      <td>0.040983</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>True</td>
      <td>wye</td>
    </tr>
    <tr>
      <th>12</th>
      <td>LV Load 12</td>
      <td>15</td>
      <td>0.054367</td>
      <td>0.012717</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>True</td>
      <td>wye</td>
    </tr>
    <tr>
      <th>13</th>
      <td>LV Load 13</td>
      <td>16</td>
      <td>0.133167</td>
      <td>0.016833</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>True</td>
      <td>wye</td>
    </tr>
    <tr>
      <th>14</th>
      <td>LV Load 14</td>
      <td>17</td>
      <td>0.138983</td>
      <td>0.034217</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>True</td>
      <td>wye</td>
    </tr>
    <tr>
      <th>15</th>
      <td>LV Load 15</td>
      <td>18</td>
      <td>0.115333</td>
      <td>0.021367</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>True</td>
      <td>wye</td>
    </tr>
    <tr>
      <th>16</th>
      <td>LV Load 16</td>
      <td>19</td>
      <td>0.090583</td>
      <td>0.019933</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>True</td>
      <td>wye</td>
    </tr>
    <tr>
      <th>17</th>
      <td>LV Load 17</td>
      <td>20</td>
      <td>0.065367</td>
      <td>0.014383</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>True</td>
      <td>wye</td>
    </tr>
    <tr>
      <th>18</th>
      <td>LV Load 18</td>
      <td>21</td>
      <td>0.200000</td>
      <td>0.036033</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>True</td>
      <td>wye</td>
    </tr>
    <tr>
      <th>19</th>
      <td>LV Load 19</td>
      <td>22</td>
      <td>0.006217</td>
      <td>0.116300</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>True</td>
      <td>wye</td>
    </tr>
  </tbody>
</table>
</div>




```python
#lv_bus = pp.get_element_index(net, "bus", "LV Bus 17")

#pp.create_sgen(net,lv_bus, p_mw=0.2, q_mvar=0.5, name="static generator")

#net.sgen
```


```python
#lv_bus = pp.get_element_index(net, "bus", "HV Busbar")

#pp.create_gen(net, lv_bus, p_mw=6, max_q_mvar=3, min_q_mvar=-3, vm_pu=1.03, name="generator") 

#net.gen
```


```python
sw1 = pp.create_switch(net, bus1, bus2, et="b", type="CB", closed=True)
sw2 = pp.create_switch(net, bus2, bus3, et="b", type="CB", closed=True)
```


```python
net
```




    This pandapower network includes the following parameter tables:
       - bus (23 elements)
       - load (20 elements)
       - switch (2 elements)
       - ext_grid (1 element)
       - line (20 elements)
       - trafo (1 element)




```python
#pp.runpp(net,numba=False)
```


```python
#net.res_bus
```


```python
import pandapower.networks as nw
from pandapower.plotting import simple_plot
```


```python
simple_plot(net)
```

    No or insufficient geodata available --> Creating artificial coordinates. This may take some time
    


    
![png](output_16_1.png)
    





    <AxesSubplot:>




```python
profiles = pd.read_csv('D:\\Glasgow substation model\\1 day power.csv', header=0, decimal=',')
profiles = profiles.astype('float64')
profiles=profiles/1000
ds = DFData(profiles)
```


```python
ConstControl(net, element='load', variable='p_mw', element_index=[0],
                 data_source=ds, profile_name=["load1_p"])
ConstControl(net, element='load', variable='p_mw', element_index=[1],
                 data_source=ds, profile_name=["load2_p"])
ConstControl(net, element='load', variable='p_mw', element_index=[2],
                 data_source=ds, profile_name=["load3_p"])
ConstControl(net, element='load', variable='p_mw', element_index=[3],
                 data_source=ds, profile_name=["load4_p"])
ConstControl(net, element='load', variable='p_mw', element_index=[4],
                 data_source=ds, profile_name=["load5_p"])
ConstControl(net, element='load', variable='p_mw', element_index=[5],
                 data_source=ds, profile_name=["load6_p"])
ConstControl(net, element='load', variable='p_mw', element_index=[6],
                 data_source=ds, profile_name=["load7_p"])
ConstControl(net, element='load', variable='p_mw', element_index=[7],
                 data_source=ds, profile_name=["load8_p"])
ConstControl(net, element='load', variable='p_mw', element_index=[8],
                 data_source=ds, profile_name=["load9_p"])
ConstControl(net, element='load', variable='p_mw', element_index=[9],
                 data_source=ds, profile_name=["load10_p"])
ConstControl(net, element='load', variable='p_mw', element_index=[10],
                 data_source=ds, profile_name=["load11_p"])
ConstControl(net, element='load', variable='p_mw', element_index=[11],
                 data_source=ds, profile_name=["load12_p"])
ConstControl(net, element='load', variable='p_mw', element_index=[12],
                 data_source=ds, profile_name=["load13_p"])
ConstControl(net, element='load', variable='p_mw', element_index=[13],
                 data_source=ds, profile_name=["load14_p"])
ConstControl(net, element='load', variable='p_mw', element_index=[14],
                 data_source=ds, profile_name=["load15_p"])
ConstControl(net, element='load', variable='p_mw', element_index=[15],
                 data_source=ds, profile_name=["load16_p"])
ConstControl(net, element='load', variable='p_mw', element_index=[16],
                 data_source=ds, profile_name=["load17_p"])
ConstControl(net, element='load', variable='p_mw', element_index=[17],
                 data_source=ds, profile_name=["load18_p"])
ConstControl(net, element='load', variable='p_mw', element_index=[18],
                 data_source=ds, profile_name=["load19_p"])
ConstControl(net, element='load', variable='p_mw', element_index=[19],
                 data_source=ds, profile_name=["load20_p"])
ConstControl(net, element='load', variable='q_mvar', element_index=[0],
                 data_source=ds, profile_name=["load1_q"])
ConstControl(net, element='load', variable='q_mvar', element_index=[1],
                 data_source=ds, profile_name=["load2_q"])
ConstControl(net, element='load', variable='q_mvar', element_index=[2],
                 data_source=ds, profile_name=["load3_q"])
ConstControl(net, element='load', variable='q_mvar', element_index=[3],
                 data_source=ds, profile_name=["load4_q"])
ConstControl(net, element='load', variable='q_mvar', element_index=[4],
                 data_source=ds, profile_name=["load5_q"])
ConstControl(net, element='load', variable='q_mvar', element_index=[5],
                 data_source=ds, profile_name=["load6_q"])
ConstControl(net, element='load', variable='q_mvar', element_index=[6],
                 data_source=ds, profile_name=["load7_q"])
ConstControl(net, element='load', variable='q_mvar', element_index=[7],
                 data_source=ds, profile_name=["load8_q"])
ConstControl(net, element='load', variable='q_mvar', element_index=[8],
                 data_source=ds, profile_name=["load9_q"])
ConstControl(net, element='load', variable='q_mvar', element_index=[9],
                 data_source=ds, profile_name=["load10_q"])
ConstControl(net, element='load', variable='q_mvar', element_index=[10],
                 data_source=ds, profile_name=["load11_q"])
ConstControl(net, element='load', variable='q_mvar', element_index=[11],
                 data_source=ds, profile_name=["load12_q"])
ConstControl(net, element='load', variable='q_mvar', element_index=[12],
                 data_source=ds, profile_name=["load13_q"])
ConstControl(net, element='load', variable='q_mvar', element_index=[13],
                 data_source=ds, profile_name=["load14_q"])
ConstControl(net, element='load', variable='q_mvar', element_index=[14],
                 data_source=ds, profile_name=["load15_q"])
ConstControl(net, element='load', variable='q_mvar', element_index=[15],
                 data_source=ds, profile_name=["load16_q"])
ConstControl(net, element='load', variable='q_mvar', element_index=[16],
                 data_source=ds, profile_name=["load17_q"])
ConstControl(net, element='load', variable='q_mvar', element_index=[17],
                 data_source=ds, profile_name=["load18_q"])
ConstControl(net, element='load', variable='q_mvar', element_index=[18],
                 data_source=ds, profile_name=["load19_q"])
ConstControl(net, element='load', variable='q_mvar', element_index=[19],
                 data_source=ds, profile_name=["load20_q"])
```




    This ConstControl has the following parameters: 
    
    index:             39
    json_excludes:     ['self', '__class__']




```python
time_steps=range(0, 48)
```


```python
run_timeseries(net, time_steps)

```

    100%|██████████| 48/48 [00:01<00:00, 26.33it/s]
    


```python
print(net.res_line.loading_percent)
```

    0     0.372744
    1     0.744934
    2     0.586290
    3     3.469128
    4     3.648954
    5     0.691013
    6     1.047347
    7     5.145438
    8     1.695175
    9     2.747342
    10    2.826391
    11    3.645381
    12    1.168132
    13    2.658395
    14    2.909732
    15    2.260966
    16    1.784109
    17    1.435053
    18    4.287771
    19    2.301592
    Name: loading_percent, dtype: float64
    


```python

```

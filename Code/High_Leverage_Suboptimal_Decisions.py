
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

## Load in the survey data
filename = 'Data/datafile.pkl'
with open(filename, 'rb') as f:
    demo_vars_19,demo_vars_20,demo_vars_20_only,demo_var_list,sdata,sdata_20,sdata_2001 = pickle.load(f)
    f.close()

sdata['all'] = 'All'
sdata_2001['Size'] = sdata_2001['large'].map({0:'Small',1:'Large*'})


#%% Figure A7.2 Suboptimal Decisions of Highly Levered Firms


high_lev = sdata_20[[x for x in sdata_20 if 'q58' in x if 'explain' not in x and 'other' not in x]]


rename = {1:'Pass up value-creating projects',
          2:'Pursue very risky projects',
          3:'Issue equity even though undervalued',
          4:'Cut corners in operations',
          6:'Have not observed suboptimal decisions'}

rename2 = {'q58_{}_wave2'.format(i):rename[i] for i in list(rename.keys())}
high_lev = high_lev.rename(columns = rename2)
high_lev = high_lev.dropna(how = 'all')

high_lev_cols = [x for x in list(rename.values()) if not 'not observed' in x]

high_lev = high_lev[high_lev_cols]


for col in high_lev_cols:
    high_lev[col] = (~pd.isnull(high_lev[col]))**1
    
out = high_lev[high_lev_cols].mean().to_frame().rename(columns = {0:'Percent'}).sort_values(by = 'Percent',ascending=False)

not_observed =  sdata_20[[x for x in sdata_20 if 'q58' in x if 'explain' not in x and 'other' not in x]]

not_observed = not_observed.rename(columns = rename2)
not_observed = not_observed.dropna(how = 'all')

not_observed = ((~pd.isnull(not_observed[rename[6]]))**1)
not_observed = pd.Series(not_observed.mean()).to_frame().rename(columns = {0:'Percent'},index = {0:rename[6]})

out = pd.concat([out,not_observed],axis = 0)

print("\n\n\n")
print("Displaying data for Figure A7.2:")
print(out)
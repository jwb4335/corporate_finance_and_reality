"""
Code to produce Figure 12
John Barry
2022/01/14
"""
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import pickle
import Functions.winsor as winsor

## Load in the survey data
filename = 'Data/datafile.pkl'
with open(filename, 'rb') as f:
    demo_vars_19,demo_vars_20,demo_vars_20_only,demo_var_list,sdata,sdata_20,sdata_2001 = pickle.load(f)
    f.close()


#%% Figure 12. Effect of 2019 Forecast Errors on Revenue, Spending, and Hiring Plans for 2020

difference = sdata_20[[x for x in sdata_20 if 'difference' in x]]
af = sdata_20.loc[np.isnan(sdata_20['respondent_2019_sg_id'])==False][['id_specific_2020','revenue_growth_2019forecast',
               'revenue_growth_2019actual',
               'wage_growth_2019forecast',
               'wage_growth_2019actual',
               'capex_2019forecast',
               'capex_2019actual',
               'dom_emp_growth_2019forecast',
               'dom_emp_growth_2019actual',
               'q3_1',
               'q3_7',
               'q3_13',
               'q3_16',
               'covid_time_dum']]

af = af.rename(columns = {'dom_emp_growth_2019forecast':'emp_growth_2019forecast',
                          'dom_emp_growth_2019actual'  :'emp_growth_2019actual',
                          'q3_1':'capex_2020',
                          'q3_7':'emp_2020',
                          'q3_13':'revenue_2020',
                          'q3_16':'wage_2020'})

for var in ['revenue_2020','emp_2020','capex_2020']:
    af[var] = winsor.winsor(af[var])
    
    
af['wageemp_growth_2019forecast'] = af['emp_growth_2019forecast']
af['wageemp_growth_2019actual']   = af['emp_growth_2019actual']
af['wageemp_2020'] = af['wage_2020']


af.columns = [x.replace('_growth',"") for x in af.columns]
for var in ['revenue','wage','capex','emp','wageemp']:
    af['difference_{}'.format(var)] = af['{}_2019actual'.format(var)] - af['{}_2019forecast'.format(var)]

varlist = ['revenue','capex','emp']
i = 0
for var in varlist:
    af.loc[af['difference_{}'.format(var)]<0,'miss_{}'.format(var)] = 'Low miss'
    af.loc[af['difference_{}'.format(var)]>0,'miss_{}'.format(var)] = 'High miss'
    af.loc[af['difference_{}'.format(var)]==0,'miss_{}'.format(var)] = 'Accurate'

def get_affect_miss(af):
    varlist = ['revenue','capex','emp']
    i = 0
    for var in varlist:
        af.loc[af['difference_{}'.format(var)]<0,'miss'] = 'Low miss'
        af.loc[af['difference_{}'.format(var)]>0,'miss'] = 'High miss'
        af.loc[af['difference_{}'.format(var)]==0,'miss'] = 'Accurate'
        
        af_stats = af.loc[~pd.isnull(af['difference_{}'.format(var)])]
        stats_all = af_stats['{}_2020'.format(var)].agg(['count','mean','median']).to_frame().T.rename(index = {'{}_2020'.format(var):'All firms'})
        stats = af_stats.groupby('miss')['{}_2020'.format(var)].agg(['count','mean','median'])
        stats = pd.concat([stats_all,stats],axis=0)
        stats.columns = [x + " " + var for x in stats.columns]
        if i == 0:
            stats_out = stats
        else:
            stats_out = pd.concat([stats_out,stats],axis=1)
        i = i+1
    stats_out = stats_out.reindex(['All firms','Low miss','Accurate','High miss'])
    stats_out['accuracy_2019'] = stats_out.index
    stats_out = stats_out.reset_index(drop=True)
    stats_out = stats_out[['accuracy_2019'] + [x for x in stats_out.columns if 'count' in x] +\
                          [x for x in stats_out.columns if 'mean' in x] + [x for x in stats_out.columns if 'median' in x]]
    
    i = 0
    for stat in ['count','mean','median']:
        counts = stats_out[['accuracy_2019'] + [x for x in stats_out.columns if stat in x]]
        counts = counts.rename(columns = {'{} emp'.format(stat):'Employment',
                                          '{} capex'.format(stat):'Capx',
                                          '{} wage'.format(stat):'Wages',
                                          '{} revenue'.format(stat):'Revenue',
                                          '{} wageemp'.format(stat): 'Wages (emp miss)'})
        counts = counts[['accuracy_2019','Revenue','Capx','Employment']]
        if stat !='count':
            for var in ['Revenue','Capx','Employment']:
                counts[var] = counts[var]/100
        counts.index = [stat]*4
        if i == 0:
            out_final = counts
        else:
            out_final = pd.concat([out_final,counts],axis=0)
        i = i+1
    return out_final


sample = 'Pre-March 15, 2020'
af_stats = af.loc[af['covid_time_dum']==0]
af_stats = get_affect_miss(af_stats)
af_stats = af_stats.loc[af_stats.index == 'mean']
af_stats.index = af_stats['accuracy_2019']
af_stats = af_stats[['Revenue','Capx','Employment']].rename(columns = {'Capx':'Capital Spending'})
af_stats.columns = pd.MultiIndex.from_tuples([(sample,x) for x in af_stats.columns],names = ['Sample','Variable'])

print("\n\n\n")
print("Displaying Data for Figure 12:")
print(af_stats)
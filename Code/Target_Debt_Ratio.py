"""
Code to produce Figure 14, Table V
John Barry
2022/01/14
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import copy


## Load in the survey data
filename = 'Data/datafile.pkl'
with open(filename, 'rb') as f:
    demo_vars_19,demo_vars_20,demo_vars_20_only,demo_var_list,sdata,sdata_20,sdata_2001 = pickle.load(f)
    f.close()



#%% Figure 14. Do Firms Have Target Debt Ratios?

sdata['all'] = 'All'

sdata_2001['Size'] = sdata_2001['large'].map({0:'Small',1:'Large*'})

def target_debt_ratio(sdata,sdata_2001):
       
    tdr = sdata[['q6_debt_target_range']]
    tdr = tdr[tdr.notnull().any(1)]

    def target_debt_ratio(vec):
        x = vec[0]
        if x == 1:
            label = "Strict"
        elif x == 3:
            label = "Somewhat tight"
            x = 300
        elif x == 2:
            label = "Flexible"
            x = 200
        elif x == 4:
            label = "No target"
        
        if x == 300:
            x = 2
        elif x == 200:
            x = 3
        return x, label
    
    tdr[['q6_debt_target_range','label']] = tdr[['q6_debt_target_range']].apply(target_debt_ratio,axis=1,
         result_type = "expand")
      
    
    tdr['count'] = 1
    tdr_graph = tdr.label.groupby(tdr.q6_debt_target_range).count()
    tdr_count = tdr['count'].sum()
    tdr_graph= tdr_graph/tdr_count
    tdr_graph.index = ["Strict","Somewhat tight","Flexible", "No target"]
    
    # Now, do the 2001 survey:
   
        
    sdata_2001 = sdata_2001[[col for col in sdata_2001 if col.startswith(('Q11'))]]
    
    sdata_2001 = sdata_2001[sdata_2001.Q11 !=9]
    
    sdata_2001 = sdata_2001.dropna(how = 'all')
    
    sdata_2001[['q6_debt_target_range','label']] = sdata_2001[['Q11']].apply(target_debt_ratio,
            axis=1,result_type = "expand")
    
    # Combos: Public vs private 2001 and 2019, 2001 vs 2019, and then all.
    
    
    tdr_2001 = sdata_2001[['q6_debt_target_range','label']]
    tdr_2001['survey'] = 2001
    
    
    tdr_2019 = tdr[['q6_debt_target_range','label']]
    tdr_2019['survey'] = 2019
    
    
    
    tdr_2001['count'] = 1
    tdr_2001_graph = tdr_2001.q6_debt_target_range.groupby(tdr_2001.label).count()/(tdr_2001['count'].sum())
    
    tdr_out = pd.concat([tdr_2001_graph,tdr_graph],axis=1,sort=True)
    tdr_out.columns = ["2000", "2020"]
    tdr_out = tdr_out.T
    tdr_out = tdr_out[['Strict','Somewhat tight','Flexible','No target']]

    return tdr_out

tdr_large = target_debt_ratio(sdata.loc[sdata['Size']=='Large*'],sdata_2001[sdata_2001['Size']=='Large*']).T
tdr_small = target_debt_ratio(sdata.loc[sdata['Size']=='Small'],sdata_2001[sdata_2001['Size']=='Small']).T

tdr_2001 = pd.concat([tdr_large['2000'],tdr_small['2000']],axis=1)
tdr_2001.columns = ['Large','Small']
tdr_2001 = tdr_2001[['Small','Large']]
tdr_2001.columns = pd.MultiIndex.from_tuples([('2001 Survey',x) for x in tdr_2001.columns],names = ['Survey','Size'])

tdr_2022 = pd.concat([tdr_large['2020'],tdr_small['2020']],axis=1)
tdr_2022.columns = ['Large','Small']
tdr_2022 = tdr_2022[['Small','Large']]
tdr_2022.columns = pd.MultiIndex.from_tuples([('2022 Survey',x) for x in tdr_2022.columns],names = ['Survey','Size'])

tdr_out = pd.concat([tdr_2001,tdr_2022],axis=1)

tdr_out.index = [x.title() for x in tdr_out.index]

tdr_out = tdr_out.reindex(['Strict','Somewhat Tight','Flexible','No Target'])

print("\n\n\n")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print("Displaying data for Figure 14:")
    print(tdr_out)

#%%
sdata_2001.loc[sdata_2001['Q11'] == 9,'Q11'] = np.nan
sdata_2001 = sdata_2001.rename(columns = {'Q11':'Target Debt Range'})
sdata_2001['Survey'] = '2001'

sdata['Survey'] = '2022*'
sdata = sdata.rename(columns = {'q6_debt_target_range':'Target Debt Range'})


demo_var = 'Survey'
data = sdata_2001[['Target Debt Range',demo_var,'Size']].append(sdata[['Target Debt Range',demo_var,'Size']])
data_in = data

#%% Table V. Do Companies Have Target Debt Ratios?



def compare_surveys(data_in,demo_var):
    """
    Function to compare target debt ratios across firm type
    
    Note - does chi^2 contingency tests
    """
    
    data_in = data_in.dropna(subset = ['Target Debt Range'],how = 'all').dropna(subset = [demo_var],how = 'all')
    
    data_in['Target Debt Range_update'] = data_in['Target Debt Range'].map({1:3,3:2,2:1,4:0})
    sort = [x for x in data_in[demo_var].unique().tolist() if '*' not in x] + \
           [x for x in data_in[demo_var].unique().tolist() if '*' in x]
           

    hold = data_in.groupby([demo_var])['Target Debt Range_update'].mean().to_frame().T.\
           rename(index = {x:x.replace("_update","") for x in ['Target Debt Range_update']})         
    for col in hold.columns:
        hold[col] = hold[col].map('{:.2f}'.format).replace('nan','')
        
    from scipy.stats import chi2_contingency
    contingency = pd.crosstab(data_in[demo_var],data_in['Target Debt Range'])
    c, p, dof, expected = chi2_contingency(contingency)     
    
    if (p>0.05) & (p<=0.1):
        aster = '*'
    elif (p>0.01) & (p<=0.05):
        aster = '**'
    elif (p<=0.01):
        aster = '***'
    else:
        aster = ''
    
    hold[sort[1]] = hold[sort[1]] + aster
    
    hold = hold[sort]
    
    
    col_insert = hold[sort[0]].tolist() + hold[sort[1]].tolist()
    
    n_list = data_in.groupby(demo_var)[demo_var].count().reindex(sort).tolist()
    
    cols_to_put_in= pd.MultiIndex.from_tuples([(demo_var,x.replace("*",""),n_list[i],col_insert[i]) for i,x in enumerate(hold.columns)],
                                             names = ['Demographic Variable','Group','N','Score'])
   
    
    for col in sort:
        percents = data_in.loc[data_in[demo_var] == col]['Target Debt Range_update'].value_counts(normalize=True).\
                   sort_index(ascending=False).to_frame().rename(columns = {'Target Debt Range':col}).\
                   rename(index = {0:'No Target',1:'Flexible',2:'Somewhat Tight',3:'Strict'})
        for column in percents:
            percents[column] = (percents[column]*100).map('{:.2f}'.format).replace('nan','')
        if col == sort[0]:
            final = percents
        else:
            final = pd.concat([final,percents],axis=1)
    final.columns = cols_to_put_in
    final.index = final.index.set_names(['Target Debt Range (% in Each Group)'])
    
    return final


demo_var = 'Survey'
comp_01_22 = compare_surveys(data,demo_var)
comp_01_22.columns = pd.MultiIndex.from_tuples([('Full Sample',x[1],x[2],x[3]) for x in comp_01_22.columns],
                                               names = ['Sample','Survey','N','Score'])

comp_01_22_small = compare_surveys(data.loc[data['Size'] == 'Small'],demo_var)
comp_01_22_small.columns =  pd.MultiIndex.from_tuples([('Small Firms',x[1],x[2],x[3]) for x in comp_01_22_small.columns],
                                               names = ['Sample','Survey','N','Score'])
comp_01_22_large = compare_surveys(data.loc[data['Size'] == 'Large*'],demo_var)
comp_01_22_large.columns = pd.MultiIndex.from_tuples([('Large Firms',x[1],x[2],x[3]) for x in comp_01_22_large.columns],
                                               names = ['Sample','Survey','N','Score'])

demo_vars_out = ['Size','Public','Growth Prospects','Pay Dividends','Leverage','Cash','Financial Flexibility']


for demo_var in demo_vars_out:
    if demo_var == demo_vars_out[0]:
        tdr_demo = compare_surveys(sdata,demo_var)
    else:
        tdr_demo = pd.concat([tdr_demo,compare_surveys(sdata,demo_var)],axis=1)



panel_a = pd.concat([comp_01_22,comp_01_22_large,comp_01_22_small,],axis=1)

panel_a = panel_a.reindex(['No Target','Flexible','Somewhat Tight','Strict'])

panel_a.columns = pd.MultiIndex.from_tuples(
          [('Panel A: Target Debt Ratios, 2001 vs. 2022 Comparison',) + x for x in panel_a.columns],
          names = ['Panel'] + panel_a.columns.names)

panel_b = tdr_demo
panel_b.columns = pd.MultiIndex.from_tuples(
          [('Panel B: Target Debt Ratios, Conditional on Company Characteristics',) + x for x in panel_b.columns],
          names = ['Panel'] + panel_b.columns.names)

panel_b = panel_b.reindex(['No Target','Flexible','Somewhat Tight','Strict'])


print("\n\n\n")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print("Displaying Table V Panel A:")
    print(panel_a)
    print("\n\n\n")
    print("Displaying Table V Panel B:")
    print(panel_b)
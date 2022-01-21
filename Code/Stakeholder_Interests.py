"""
Code to produce Figure 22, Figure 23
John Barry
2022/01/14
"""
import pandas as pd
import numpy as np
import pickle


## Load in the stakeholder interests data from 2010 and 2020
filename = 'Data/stk_interests.pkl'
with open(filename, 'rb') as f:
    [stk_interest,other_stk] = pickle.load(f)
    f.close()




def stakeholder_interest_bins(stk_int,nbins=10,labels = ['0-20','21-40','41-60','61-80','81-100']):
    """
    Stakeholder interests plot
    """
    ## Convert the labels to quantile bounds
    vec = np.arange(100/nbins,100+100/nbins,100/nbins)
    label_dict = {vec[i]/(100/nbins): labels[i] for i in range(len(vec))} 

    for i in np.arange(len(vec)):
        val = vec[i]
        val_l = vec[i-1]
        if i == 0:
            stk_int.loc[stk_int['stk_interest']<=val,'bins'] =  val/(100/nbins)
        else:
            stk_int.loc[(stk_int['stk_interest']<=val) & (stk_int['stk_interest']>val_l),'bins'] = val/(100/nbins)


    ## Calculate percentages within each bin            
    stk_int = stk_int.groupby(['survey','location','bins'])['stk_interest'].count().reset_index()
    totals = stk_int.groupby(['survey','location'])['stk_interest'].sum().reset_index()
    totals = totals.rename(columns = {'stk_interest':'total'})
    stk_int = stk_int.merge(totals,on = ['survey','location'],how = 'inner')
    stk_int['pct'] = stk_int['stk_interest'].divide(stk_int['total'])
    stk_int['bin_labels'] = stk_int['bins'].map(label_dict)
    stk_int = stk_int.rename(columns = {'stk_interest':'count'})
    stk_int = stk_int[['survey','location','bins','bin_labels','pct','count','total']]
    
    ## Convert to wide
    stk_int_wide = stk_int
    stk_int_wide['loc_survey'] = stk_int_wide['location']+" "+ stk_int_wide['survey']
    stk_int_wide = stk_int_wide[['loc_survey','bin_labels','pct']].pivot(index = 'bin_labels',columns = 'loc_survey',values = 'pct')
    stk_int_wide = stk_int_wide[['North America 2010','North America 2020','Europe 2010','Europe 2020','Asia 2010','Asia 2020']]

    return stk_int_wide


def other_stakes(other_stk):
    """
    Other stakeholders plot
    """
    
    stakes_rename = {'a':'Local community',
                      'b':'Creditors',
                      'c':'Customers',
                      'd':'Employees',
                      'e':'Environment',
                      'f':'Government',
                      'g':'Suppliers',
                      'h':'Other',
                      'i':'None of the above'}
    stakeholder_cols = [x for x in other_stk.columns if x.startswith('q32_2')]
    stakes = other_stk[stakeholder_cols+['id_specific_2020']]
    del stakes['q32_2other_wave2']
    stakes.columns = [x[5] for x in stakes.columns if x not in ['id_specific_2020']]+['id_specific_2020']
    stakes = stakes.rename(columns = stakes_rename)
    

    stakes = stakes.merge(other_stk[['location','id_specific_2020']],on = 'id_specific_2020')
    for var in stakes_rename.values():
        stakes.loc[~pd.isnull(stakes[var]),var] = 1
    stakes['sum'] = stakes[stakes_rename.values()].sum(axis=1)
    stakes = stakes.loc[stakes['sum']!=0]
    cols = list(stakes_rename.values())
    totals = stakes.groupby('location')['location'].count()
    stakes = stakes.groupby('location')[cols].sum()
    stakes['N'] = totals
    for var in cols:
        stakes[var] = stakes[var].divide(stakes['N'])
    stakes = stakes.T
    stakes = stakes.sort_values(by = 'North America')
    stakes = stakes[['North America'] + [x for x in stakes.columns if x not in ['North America']]]
    stakes = stakes.loc[(stakes.index !='Other') & (stakes.index !='None of the above')]

    return stakes
    


labels = ['0-20','21-40','41-60','61-80','81-100']
nbins = 5        
stk_interest_quints_wide = stakeholder_interest_bins(stk_interest,nbins=nbins,labels=labels)

other_stks_pct = other_stakes(other_stk)


## Limit to North America:
stk_interest_quints_wide = stk_interest_quints_wide[[x for x in stk_interest_quints_wide if 'North America' in x]]

other_stks_pct = other_stks_pct[[x for x in other_stks_pct if 'North America' in x]]

other_stks_pct = other_stks_pct.sort_values(by = ['North America'], ascending=False)

## Print data for figures
print("\n\n\n")
print("Displaying data for Figure 22:")
print(stk_interest_quints_wide)

print("\n\n\n")
print("Displaying data for Figure 23:")
print(other_stks_pct)

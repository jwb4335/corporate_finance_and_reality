
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


#%%

def ext_fund_type(sdata,byvar):
    

    extfund_vars = ['q13_extfund_commonstock','q13_extfund_prefstock',
                     'q13_extfund_bond','q13_extfund_convdebt',
                     'q13_extfund_bankloan','q13_extfund_nonbankloan',
                     'q13_extfund_lineofcredit','q13_extfund_commpaper']
    extfund_sources = sdata[extfund_vars + [byvar]]
    
    extfund_sources = extfund_sources.dropna(how = 'all',subset = extfund_vars)
    
    extfund_sources = extfund_sources.dropna(how = 'all',subset = [byvar])


    for var in extfund_vars:
        extfund_sources[var] = (extfund_sources[var]>=0)**1
        
    
    index_dict = {'q13_extfund_commonstock':'Common stock',
                  'q13_extfund_prefstock': 'Preferred stock',
                  'q13_extfund_bond':'Bonds',
                  'q13_extfund_convdebt':'Convertible debt',
                  'q13_extfund_bankloan':'Bank loans',
                  'q13_extfund_nonbankloan': 'Non-bank loans',
                  'q13_extfund_lineofcredit':'Lines of credit',
                  'q13_extfund_commpaper':'Commercial paper'}

    extfund_sources = extfund_sources.rename(columns=index_dict)    

    extfund_sources_total = extfund_sources.groupby(byvar)[list(index_dict.values())].sum()
    
    sources_pct = extfund_sources_total.div(extfund_sources_total.sum(axis=1),axis=0)
    
    
    extfund_purpose =  sdata[[col for col in sdata if col.startswith(('q13_extfund_purpose'))]+[byvar]]
    
    del extfund_purpose['q13_extfund_purpose_other']
    
    index_dict2 = {'q13_extfund_purpose_commonstock':'Common stock',
                  'q13_extfund_purpose_prefstock': 'Preferred stock',
                  'q13_extfund_purpose_bond':'Bonds',
                  'q13_extfund_purpose_convdebt':'Convertible debt',
                  'q13_extfund_purpose_bankloan':'Bank loans',
                  'q13_extfund_purpose_nonbankloan': 'Non-bank loans',
                  'q13_extfund_purpose_lineofcredit':'Lines of credit',
                  'q13_extfund_purpose_commpaper':'Commercial paper'}
    
    extfund_purpose_vars = list(index_dict2.keys())
    
    
    index_dict3 = {1:'Fund a specific project',
                   2:'General funding needs',
                   3:'Working capital needs',
                   4:'Cover operating losses',
                   5:'Rebalance debt/equity ratio',
                   6:'Roll over existing security'}
    
    
    extfund_purpose = extfund_purpose.dropna(how = 'all', subset = extfund_purpose_vars)
        
    
    sort = [x for x in extfund_purpose[byvar].unique().tolist() if '*' not in x] + [x for x in extfund_purpose[byvar].unique().tolist() if '*' in x]
    sort_dict = {sort[i]:i for i in np.arange(len(sort))}
    def get_purpose(extfund_purpose,byvar,var,sort_dict):
        sort_map = {'Common stock':1,
                    'Preferred stock':2,
                    'Convertible debt':3,
                    'Bonds':4,
                    'Commercial paper':5,
                    'Bank loans':6,
                    'Non-bank loans':7,
                    'Lines of credit':8}
        sort_map2 = sort_dict
        source = sources_pct.loc[sources_pct.index == var].T.reset_index().rename(columns = {'index':'type',var:'source_pct'})
        source[byvar] = var
        out = extfund_purpose.loc[extfund_purpose[byvar] == var]
        out = out[extfund_purpose_vars].apply(pd.value_counts)
        out = out.reindex(np.arange(1,8))
        out = out.loc[out.index<7]
        out = out.rename(index = index_dict3,columns = index_dict2)
        out = out.div(out.sum(axis=0),axis=1)
        out = out.T
        out = out.reset_index().rename(columns = {'index':'type'})
        out['sort'] = out['type'].map(sort_map)
        out[byvar] = var
        out['sort2'] = out[byvar].map(sort_map2)
        out = out[['type',byvar] + [x for x in out.columns if x not in ['type',byvar,'sort']] + ['sort']]
        out = out.merge(source,on = [byvar,'type'])
        out['type'] = out['type']  
        out[byvar] = var + " ("+round(out['source_pct']*100,0).astype(int).astype(str)+"%)"
        return out

    i = 0
    for var in sort:
        hold = get_purpose(extfund_purpose,byvar,var,sort_dict)        
        if i == 0:
            extfund_purposes = hold
        else:
            extfund_purposes = pd.concat([extfund_purposes,hold],axis = 0)
        i = i + 1

    extfund_purposes = extfund_purposes.sort_values(by = ['sort','sort2']).reset_index(drop=True)
    
    extfund_purposes['Size'] =     extfund_purposes['Size'].str.replace('*','',regex=False)
    
    new_index = list(zip(extfund_purposes['type'].tolist(),extfund_purposes['Size'].tolist()))
    
    extfund_purposes.index = pd.MultiIndex.from_tuples(new_index,names = ['Source','Size'])
    
    extfund_purposes = extfund_purposes[[x for x in extfund_purposes if x not in ['Size','type','source_pct','sort','sort2']]]
    

    return extfund_purposes


extfund_purposes = ext_fund_type(sdata,'Size')

## Drop convertibles, very few observations
extfund_purposes = extfund_purposes.reindex([x for x in extfund_purposes.index if 'Convertible' not in x[0]])

print("\n\n\n")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print("Displaying data for Figure A7.3")
    print(extfund_purposes)
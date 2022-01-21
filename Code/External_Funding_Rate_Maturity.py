"""
Code to produce Figure A7.4, Figure A7.5
Rong Wang & John Barry
2022/01/14
"""
import pandas as pd
import numpy as np
import pickle

## Load in the survey data
filename = 'Data/datafile.pkl'
with open(filename, 'rb') as f:
    demo_vars_19,demo_vars_20,demo_vars_20_only,demo_var_list,sdata,sdata_20,sdata_2001 = pickle.load(f)
    f.close()
  
    
#%% Figure A7.4. Floating or Fixed Interest Rate by Source of External Funding
### Figure A7.5. Maturity of External Funding

def extfund_topic(sdata,topic,byvar):
# topic = 'type' (floating/fixed) or 'term' (maturity)
# byvar = 'Size', 'Growth Prospect', etc. 

    ## compute the percentages of each source of external funding

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
    
    
    ## a dataset for the topic we care about ('type' or 'term')
    
    extfund_topic = sdata[[col for col in sdata.columns if 'q13_' and topic in col if 'other' not in col]+[byvar]]

    index_dict1 = {'q13_extfund_'+topic+'_commonstock':'Common stock',
                   'q13_extfund_'+topic+'_prefstock': 'Preferred stock',
                   'q13_extfund_'+topic+'_bond':'Bonds',
                   'q13_extfund_'+topic+'_convdebt':'Convertible debt',
                   'q13_extfund_'+topic+'_bankloan':'Bank loans',
                   'q13_extfund_'+topic+'_nonbankloan': 'Non-bank loans',
                   'q13_extfund_'+topic+'_lineofcredit':'Lines of credit',
                   'q13_extfund_'+topic+'_commpaper':'Commercial paper'}
    
    extfund_topic = extfund_topic.dropna(how = 'all',subset = list(index_dict1.keys()))
    
    extfund_topic = extfund_topic.dropna(how = 'all',subset = [byvar])
    
    extfund_topic = extfund_topic.rename(columns=index_dict1) 
    
    if topic == 'type': 
        
        # response of "4-not applicable" is equivalent to missing
        extfund_topic = extfund_topic.replace(4,np.nan)
    
        index_dict2 = {1:'Fixed rate',
                       2:'Floating rate',
                       3:'Both',
                       4:'Not applicable'}
        
    elif topic == 'term':
        
        # response of "6-none" is equivalent to missing
        extfund_topic = extfund_topic.replace(6,np.nan)
        
        # drop responses who indicate more than 1 year commercial paper
         
        index_dict2 = {1:'1 year',
                       2:'2-3 years',
                       3:'4-5 years',
                       4:'6-10 years',
                       5:'More than 10 years',
                       6:'None'}
            
    ## compute percentages of each category of responses in the "topic" by "byvar"
    
    sort = [x for x in extfund_sources[byvar].unique().tolist()]
    sort_dict = {sort[i]:i for i in np.arange(len(sort))}
    
    def get_topic(extfund_topic,byvar,demo,sort_dict):
        
        source = sources_pct.loc[sources_pct.index == demo].T.reset_index().rename(columns = {'index':'sources',demo:'source_pct'})
        source[byvar] = demo
        
        # data for the specific demographic feature ("demo")
        out = extfund_topic.loc[extfund_topic[byvar] == demo]
        
        # counting the uniue values for each variable
        out = out[list(index_dict1.values())].apply(pd.value_counts)
        
        # reindex and rename the indexes to responses
        out = out.reindex(np.arange(1,len(out)+1))
        out = out.rename(index = index_dict2)
        
        # compute percentages of each response
        out = out.div(out.sum(axis=0),axis=1)
        out = out.T
        
        # merge wirh source data
        out = out.reset_index().rename(columns = {'index':'sources'})
        out[byvar] = demo
        out = out.merge(source,on = [byvar,'sources']) 
        out[byvar] = demo + " ("+round(out['source_pct']*100,0).astype(int).astype(str)+"%)"
        del out['source_pct']
        
        # functions to reorder columns
        def movecol(df, cols_to_move=[], ref_col='', place='After'):
    
            cols = df.columns.tolist()
            if place == 'After':
                seg1 = cols[:list(cols).index(ref_col) + 1]
                seg2 = cols_to_move
            if place == 'Before':
                seg1 = cols[:list(cols).index(ref_col)]
                seg2 = cols_to_move + [ref_col]
    
            seg1 = [i for i in seg1 if i not in seg2]
            seg3 = [i for i in cols if i not in seg1 + seg2]
    
            return(df[seg1 + seg2 + seg3])
        
        # reorder "byvar" after "sources"
        out = movecol(out, cols_to_move=[byvar],ref_col='sources',place='After')
        
        return out

    # table reporting the persentges for each responses in the "topic" by "byvar"
    i = 0
    for demo in sort:
        hold = get_topic(extfund_topic,byvar,demo,sort_dict)        
        if i == 0:
            extfund_out = hold
        else:
            extfund_out = pd.concat([extfund_out,hold],axis = 0)
        i = i + 1
    
    # sorting 
    extfund_out = extfund_out.sort_values(by = ['sources',byvar]).reset_index(drop=True)
    extfund_out[byvar] = extfund_out[byvar].str.replace('*','')
    
    sort_map = {'Common stock':1,
                'Preferred stock':2,
                'Bonds':3,
                'Convertible debt':4,
                'Bank loans':5,
                'Non-bank loans':6,
                'Lines of credit':7,
                'Commercial paper':8}
    
    extfund_out['sort'] = extfund_out['sources'].map(sort_map)
    extfund_out = extfund_out.sort_values(by=['sort','Size'], ascending=[True,False])
    del extfund_out['sort']
    
    # new indexing
    new_index = list(zip(extfund_out['sources'].tolist(),extfund_out[byvar].tolist())) 
    extfund_out.index = pd.MultiIndex.from_tuples(new_index,names = ['Source',byvar])    
    extfund_out = extfund_out[[x for x in extfund_out if x not in [byvar,'sources']]]
    
    return extfund_out


#%% call the function to compute percentages

# Figure A7.4 - floating/fixed
extfund_type = extfund_topic(sdata,'type','Size')
extfund_type = extfund_type.drop(['Common stock','Preferred stock','Convertible debt','Commercial paper'],level = 0,axis = 0)

# Figure A7.5 - maturity
extfund_term = extfund_topic(sdata,'term','Size')
extfund_term = extfund_term.drop(['Common stock','Preferred stock','Convertible debt','Commercial paper'],level = 0, axis = 0)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):

    print("\n\n\n")
    print("Displaying data for Figure A7.4:")
    print(extfund_type)
    
    print("\n\n\n")
    print("Displaying data for Figure A7.5:")
    print(extfund_term)
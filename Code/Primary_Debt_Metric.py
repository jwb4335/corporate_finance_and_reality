
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import copy

# # Set the directories
# if "barry" in os.getcwd():
#     os.chdir("C:\\Users\\barry\\Dropbox\\Graham_survey\\March_2019_Survey\\survey_code\\python_code\\Functions_v3\\Submission\\")
#     cwd = os.getcwd()
#     sys.path.append(os.path.join(cwd,))
#     table_save = 'C:\\Users\\barry\\Dropbox\\Graham_Survey\\March_2019_Survey\\graph_data\\tables_for_writeup_march21.xlsx'

## Load in the survey data
filename = 'Data/datafile.pkl'
with open(filename, 'rb') as f:
    demo_vars_19,demo_vars_20,demo_vars_20_only,demo_var_list,sdata,sdata_20,sdata_2001 = pickle.load(f)
    f.close()



#%% Figure 13. How Do Companies Measure Leverage?
sdata['all'] = 'All'
def pdm_by_type(sdata_in,demo_var):
    
             
             
    pdm_dict = {'q6_debt_assets':'Debt/Assets',
                'q6_debt_value':'Debt/Value',
                'q6_debt_equity':'Debt/Equity',
                'q6_liabilities_assets':'Liabilities/ Assets',
                'q6_debt_ebitda':'Debt/EBITDA',
                'q6_interest_coverage':'Interest Coverage',
                'q6_credit_rating':'Credit Rating'}  
    
    pdm = sdata_in[list(pdm_dict.keys())+[demo_var,'credit_rating']]
    

    pdm = pdm.dropna(how = 'all',subset = list(pdm_dict.keys()))
    pdm = pdm.dropna(subset = [demo_var],how = 'all')
    
    pdm = pdm.rename(columns = pdm_dict)
    
    pdm_cols = list(pdm_dict.values())
    
        
    def get_metric_importance_by(pdm,var,demo_var):
        # if var == 'Credit Rating':
        #     pdm = pdm.loc[pdm['credit_rating'] !='NA']
        pdm = pdm[[var,demo_var]]
        pdm_pct = pdm.groupby(demo_var)[var].value_counts(normalize=False).to_frame().rename(columns = {var:'Percent'}).reset_index()
        pdm_pct = pdm_pct.rename(columns = {var:'Rank'})
        pdm_count = pdm.groupby([demo_var])[demo_var].count().to_frame().rename(columns = {demo_var:'N'}).reset_index()
        pdm_pct = pdm_pct.merge(pdm_count,on = demo_var)
        pdm_pct['Percent'] = pdm_pct['Percent'].divide(pdm_pct['N'])
        del pdm_pct['N']
        pdm_pct = pdm_pct.pivot(index=demo_var,columns = 'Rank',values = 'Percent').reset_index()
        sort = pdm[demo_var].unique().tolist()
        sort = [x for x in sort if not '*' in x]  + [x for x in sort if '*' in x] 
        pdm_pct.index = pdm_pct[demo_var]
        pdm_pct = pdm_pct.reindex(sort)
        check = [y for y in [1,2,3] if y not in [x for x in pdm_pct.columns if x not in [demo_var]]]
        for ch in check:
            pdm_pct[ch] = 0
        pdm_pct = pdm_pct.rename(columns = {1:'Primary',2:'Secondary',3:'Tertiary'})

        pdm_pct = pdm_pct[['Primary','Secondary','Tertiary']]
        pdm_pct['Sum'] =  pdm_pct[['Primary','Secondary','Tertiary']].sum(axis=1)
        new_cols = [(var,x) for x in pdm_pct]
        pdm_pct.columns = pd.MultiIndex.from_tuples(new_cols,names = ['Debt Metric','Ranking'])
        pdm_pct = pdm_pct.T

        return pdm_pct
 

    for var in pdm_cols:
        out = get_metric_importance_by(pdm,var,demo_var)
        if var == pdm_cols[0]:
            pdm_pct = out
        else:
            pdm_pct = pd.concat([pdm_pct,out],axis=0)
            
            
        
    if len(pdm[demo_var].unique().tolist()) == 1:
        dvar = pdm[demo_var].unique().tolist()[0]
        pdm_pct = pdm_pct.pivot_table(values = dvar,index = ['Debt Metric'],columns = ['Ranking']).sort_values(by = 'Sum')[['Primary', 'Secondary', 'Tertiary']]
        pdm_pct.columns = pd.MultiIndex.from_tuples([(dvar.replace('*',''),) + (x,) for x in pdm_pct.columns],names = ['Group','Ranking'])
    
    return pdm_pct

file_name = 'March_2019_survey/graph_data/primary_debt_metric/primary_debt_metric_march21.xlsx'         

pdm_all = pdm_by_type(sdata,'all')

pdm_large = pdm_by_type(sdata.loc[sdata['Size'] == 'Large*'],'Size')

pdm_small = pdm_by_type(sdata.loc[sdata['Size'] == 'Small'],'Size')

pdm_small.columns = [x[1] for x in pdm_small]
pdm_small.index = pd.MultiIndex.from_tuples([(x,'Small') for x in pdm_small.index],names = ['Debt Metric','Size'])

pdm_large.columns = [x[1] for x in pdm_large]
pdm_large.index = pd.MultiIndex.from_tuples([(x,'Large') for x in pdm_large.index],names = ['Debt Metric','Size'])

pdm_out = pd.concat([pdm_large,pdm_small],axis=0)
pdm_out = pdm_out.join(pdm_out.xs('Large',level = 'Size').sum(axis=1).to_frame().rename(columns = {0:'sum'}))
pdm_out = pdm_out.sort_values(by = ['sum','Size'],ascending = [True,False])

print("\n\n\n")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print("Displaying data for Figure 13:")
    print(pdm_out)
    

#%% TABLE IV: What Metrics Do Companies Use to Measure Debt Usage?

             
pdm_dict = {'q6_debt_assets':'Debt/Assets',
            'q6_debt_value':'Debt/Value',
            'q6_debt_equity':'Debt/Equity',
            'q6_liabilities_assets':'Liabilities/Assets',
            'q6_debt_ebitda':'Debt/EBITDA',
            'q6_interest_coverage':'Interest Coverage',
            'q6_credit_rating':'Credit Rating'}  

pdm = sdata[list(pdm_dict.keys())+['credit_rating','id_specific']]

#    pdm.loc[pdm['credit_rating'] == 'NONE','q6_credit_rating'] = np.nan

pdm = pdm.dropna(how = 'all',subset = list(pdm_dict.keys()))

pdm = pdm.rename(columns = pdm_dict)

pdm['all'] = 'All'
pdm_cols = list(pdm_dict.values())

def get_metric_importance_by(pdm,var,demo_var):
    if var == 'Credit Rating':
        pdm = pdm.loc[pdm['credit_rating'] !='NA']
    pdm = pdm[[var,demo_var]]
    pdm_pct = pdm.groupby(demo_var)[var].value_counts(normalize=False).to_frame().rename(columns = {var:'Percent'}).reset_index()
    pdm_pct = pdm_pct.rename(columns = {var:'Rank'})
    pdm_count = pdm.groupby([demo_var])[demo_var].count().to_frame().rename(columns = {demo_var:'N'}).reset_index()
    pdm_pct = pdm_pct.merge(pdm_count,on = demo_var)
    pdm_pct['Percent'] = pdm_pct['Percent'].divide(pdm_pct['N'])
    del pdm_pct['N']
    pdm_pct = pdm_pct.pivot(index=demo_var,columns = 'Rank',values = 'Percent').reset_index()
    sort = pdm[demo_var].unique().tolist()
    sort = [x for x in sort if not '*' in x]  + [x for x in sort if '*' in x] 
    pdm_pct.index = pdm_pct[demo_var]
    pdm_pct = pdm_pct.reindex(sort)
    pdm_pct = pdm_pct.rename(columns = {1:'Primary',2:'Secondary',3:'Tertiary'})
    pdm_pct = pdm_pct[['Primary','Secondary','Tertiary']]
    pdm_pct['Sum'] =  pdm_pct[['Primary','Secondary','Tertiary']].sum(axis=1)
    new_cols = [(var,x) for x in pdm_pct]
    pdm_pct.columns = pd.MultiIndex.from_tuples(new_cols,names = ['Debt Metric','Ranking'])
    pdm_pct = pdm_pct.T

    return pdm_pct

def pdm_demo_tests(pdm,sdata,demo_var,var_sort = None):
    import statsmodels.formula.api as sm

    for col in pdm_cols:
        pdm.loc[pd.isnull(pdm[col]),col] = 0
    
    pdm.loc[pdm['credit_rating']=='NA','Credit Rating'] = np.nan    
    
    pdm = pdm.merge(sdata[['id_specific',demo_var]],on = 'id_specific')
    
    pdm = pdm.dropna(subset = [demo_var])    
    sort = [x for x in pdm[demo_var].unique().tolist() if '*' not in x] +\
               [x for x in pdm[demo_var].unique().tolist() if '*' in x]    

        
    for col in pdm_cols:
        reg_df = pdm.loc[(~pd.isnull(pdm[demo_var])) & (~pd.isnull(pdm[col]))][[demo_var,col]]    
        reg_df.loc[reg_df[demo_var]==sort[1],'dummy'] = 1
        reg_df.loc[reg_df[demo_var]==sort[0],'dummy'] = 0
        reg_df['yvar'] = reg_df[col]
        p = sm.ols(formula="yvar ~ dummy", data=reg_df).fit().pvalues.dummy

        if len(reg_df[demo_var].unique().tolist()) == 1:
            p = 1
        # contingency = pd.crosstab(pdm[demo_var],pdm[col])
        # c, p, dof, expected = chi2_contingency(contingency) 
        
       
        hold = round(pdm.groupby(demo_var)[col].mean().to_frame().T,2)
        for column in hold.columns:
            hold[column] = hold[column].map('{:.2f}'.format).replace({'nan':''})
        sort = [x for x in hold.columns if '*' not in x] + [x for x in hold.columns if '*' in x]    
        
        if (p>0.05) & (p<=0.1):
            aster = '*'
        elif (p>0.01) & (p<=0.05):
            aster = '**'
        elif (p<=0.01):
            aster = '***'
        else:
            aster = ''
        
        hold[sort[1]] = hold[sort[1]] + aster
        
        if col == pdm_cols[0]:
            out = hold
        else:
            out = pd.concat([out,hold],axis = 0)
    out = out[sort]
    
    l1 = [demo_var]*2
    l2 = [x.replace('*','') for x in sort]
    l3 = pdm.groupby(demo_var)[demo_var].count()[sort].values.tolist()
    tuples= [(l1[i],l2[i],l3[i]) for i in range(len(l1))]
    out.columns = pd.MultiIndex.from_tuples(tuples,names = ['Variable','Group','N'])
    
    if var_sort is not None:
        out = out.reindex(var_sort)
            
    return out


for var in pdm_cols:
    out = get_metric_importance_by(pdm,var,'all')
    if var == pdm_cols[0]:
        pdm_pct = out
    else:
        pdm_pct = pd.concat([pdm_pct,out],axis=0)


pdm_pct = round((pdm_pct.pivot_table(values = ['All'],index = ['Debt Metric'],columns = ['Ranking']).droplevel(0,axis=1).\
          sort_values(by = 'Sum',ascending=False)[['Primary','Secondary','Tertiary']])*100,2)

pdm_pct.columns = pd.MultiIndex.from_tuples([('Percent Primary, Secondary or Tertiary',x) for x in pdm_pct.columns],names = ['Type','Variable'])

var_sort = list(pdm_pct.index)

new_mapping = {0:0,
               3:1,
               2:2,
               1:3}

for col in pdm_cols:
    pdm.loc[pd.isnull(pdm[col]),col] = 0
    pdm[col] = pdm[col].map(new_mapping)

pdm_pct.columns = pd.MultiIndex.from_tuples([(x)+('','') for x in pdm_pct.columns],names = ['Type','Variable','Group','N'])



pdm_pct_out = copy.deepcopy(pdm_pct)
pdm_pct_out['sort'] = pdm_pct_out.sum(axis=1)
pdm_pct_out = pdm_pct_out.sort_values(by = 'sort',ascending=False)
pdm_pct_out = pdm_pct_out[[x for x in pdm_pct_out if x[0] not in ['sort']]]

for col in pdm_pct.columns:
    pdm_pct_out[col] = pdm_pct_out[col].map('{:.2f}'.format).replace({'nan':''})

demo_vars_out = ['Size','Public','Leverage','Cash','Financial Flexibility']


for demo_var in demo_vars_out:
    if demo_var == demo_vars_out[0]:
        demos = pdm_demo_tests(pdm,sdata,demo_var,var_sort)
    else:
        demos = pd.concat([demos,pdm_demo_tests(pdm,sdata,demo_var,var_sort)],axis = 1)

new_index = [('Conditional on Company Characteristics',)+x for x in demos.columns]

demos.columns = pd.MultiIndex.from_tuples(new_index,names = ['Type','Variable','Group','N'])

final = pd.concat([pdm_pct_out,demos],axis=1)

print('\n\n\n')
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print("Displaying Table IV:")
    print(final)

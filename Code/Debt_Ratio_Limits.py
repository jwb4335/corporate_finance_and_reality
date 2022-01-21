"""
Code to produce Figure 15, Table VI
John Barry
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

#%% Figure 15. Debt Ratio Ranges and Timetable to Return to Target
    

def debtratio_limit(sdata,debt_ratio = 'Debt_EBITDA'):
    if debt_ratio != 'all':
        dr_lim = sdata.loc[sdata['q6_primary_debt_ratio'] == debt_ratio]
    else:
        dr_lim = sdata
    var_dict = {'Debt_EBITDA':'q6_debt_ebitda',
                'Debt_Assets':'q6_debt_assets',
                'Interest_coverage':'q6_interest_coverage',
                'Debt_Equity':'q6_debt_equity',
                'Liabilities_Assets':'q6_liabilities_assets',
                'Debt_Value':'q6_debt_value',
                'all':'q6_debt_ebitda'}
    
    
    name_dict = {'Debt_EBITDA':'Debt/EBITDA',
                'Debt_Assets':'Debt/Assets',
                'Interest_coverage':'Interest Coverage',
                'Debt_Equity':'Debt/Equity',
                'Liabilities_Assets':'Liabilities/Assets',
                'Debt_Value':'Debt/Value',
                'all':'All Debt Ratios'}
    
    
    
    dr_lim = dr_lim[[col for col in dr_lim if col.startswith(('q8_dr','respondent_id'))]+[var_dict[debt_ratio],'q6_current_debt_ratio']]
    
    dr_lim = dr_lim.dropna(subset = [ 'q8_dr_ul_set', 'q8_dr_ul', 'q8_dr_ul_reduce',
       'q8_dr_ul_reduce_years', 'q8_dr_ll_set',
       'q8_dr_ll', 'q8_dr_ll_increase', 'q8_dr_ll_increase_years'],how = 'all')
    
    rename = {'q8_dr_ul_set':'We set an upper limit',
              'q8_dr_ul':'Upper Limit',
              'q8_dr_ul_reduce_years':'Reduce Years',
              'q8_dr_ll_set':'We set a lower limit',
              'q8_dr_ll':'Lower Limit',
              'q8_dr_ll_increase_years':'Increase Years',
              'q8_dr_ul_reduce':'We set a timetable to bring down debt ratio',
              'q8_dr_ll_increase':'We set a timetable to bring up debt ratio',
              'q6_current_debt_ratio':name_dict[debt_ratio]}
    
    dr_lim = dr_lim.rename(columns = rename)
    
    dr_lim = dr_lim[[x for x in dr_lim if not 'other' in x]]
    
    
    limit_vars = ['We set an upper limit','We set a lower limit',
                  'We set a timetable to bring down debt ratio', 'We set a timetable to bring up debt ratio']

    for var in limit_vars:
        dr_lim[var] = (dr_lim[var] == 1)**1
        
    set_limits = dr_lim[limit_vars].mean().to_frame().rename(columns = {0:'Yes'})
    set_limits['No'] = 1-set_limits['Yes']
 
    limit_mean_vars = [name_dict[debt_ratio],'Upper Limit','Lower Limit','Reduce Years','Increase Years']
    
    limit_means = dr_lim[limit_mean_vars].mean().to_frame().rename(columns = {0:'Means'})


    nobs = len(dr_lim)
    return set_limits,limit_means,nobs

var_dict = {'Debt_EBITDA':'q6_debt_ebitda',
            'Debt_Assets':'q6_debt_assets',
            'Interest_coverage':'q6_interest_coverage',
            'Debt_Equity':'q6_debt_equity',
            'Liabilities_Assets':'q6_liabilities_assets',
            'Debt_Value':'q6_debt_value',
            'all':'q6_debt_ebitda'}


name_dict = {'Debt_EBITDA':'Debt/EBITDA',
            'Debt_Assets':'Debt/Assets',
            'Interest_coverage':'Interest Coverage',
            'Debt_Equity':'Debt/Equity',
            'Liabilities_Assets':'Liabilities/Assets',
            'Debt_Value':'Debt/Value',
            'all':'All Debt Ratios'}

dr = 'Debt_EBITDA'
set_limits,limit_means,nobs = debtratio_limit(sdata,debt_ratio = dr)
set_limits = set_limits[['Yes']].rename(columns = {'Yes':name_dict[dr]})
set_limits = pd.concat([set_limits,limit_means.rename(columns = {'Means':name_dict[dr]})],axis = 0)  
set_limits = set_limits.rename(index = {name_dict[dr]:'Debt Ratio'})
set_limits.columns = ['Group Average']
set_limits.columns = pd.MultiIndex.from_tuples([(name_dict[dr],nobs,x) for x in set_limits.columns],names = ['Debt Ratio','N','Variable'])


print("\n\n\n")
print("Displaying data for Figure 15:")
print(set_limits)

#%%# Table VI. High and Low Debt Bounds and Timetables to Return to Target

def test_dr_limits(sdata,demo_var,debt_ratio = 'all'):

    if debt_ratio !='all':
        dr_lim = sdata.loc[sdata['q6_primary_debt_ratio'] == debt_ratio]
    else:
        dr_lim = sdata

    import statsmodels.formula.api as sm

    dr_lim = dr_lim.loc[~pd.isnull(dr_lim[demo_var])]
    
    dr_lim = dr_lim[[col for col in dr_lim if col.startswith(('q8_dr','respondent_id'))]+[demo_var]]
    
    dr_lim = dr_lim.dropna(subset = [ 'q8_dr_ul_set', 'q8_dr_ul_reduce', 'q8_dr_ll_set',
     'q8_dr_ll_increase'],how = 'all')
    
    rename = {'q8_dr_ul_set':'We set an upper limit on our debt ratio',
              'q8_dr_ul':'Upper Limit',
              'q8_dr_ul_reduce_years':'Reduce Years',
              'q8_dr_ll_set':'We set a lower limit on our debt ratio',
              'q8_dr_ll':'Lower Limit',
              'q8_dr_ll_increase_years':'Increase Years',
              'q8_dr_ul_reduce':'We set a timetable to bring down our debt ratio',
              'q8_dr_ll_increase':'We set a timetable to bring up our debt ratio'}
    
    limit_vars = ['q8_dr_ul_set','q8_dr_ll_set','q8_dr_ul_reduce','q8_dr_ll_increase']
    dr_lim = dr_lim[[demo_var] + limit_vars]
    
    dr_lim = dr_lim[[x for x in dr_lim if not 'other' in x]]
    
    
    for col in limit_vars:
        dr_lim.loc[dr_lim[col] == 1,"{}_dum".format(col)] = 1
        dr_lim.loc[dr_lim[col] == 2,"{}_dum".format(col)] = 0
        dr_lim[col] = dr_lim["{}_dum".format(col)]
        dr_lim = dr_lim[[x for x in dr_lim if x not in ["{}_dum".format(col)]]]

    sort = [x for x in dr_lim[demo_var].unique().tolist() if '*' not in x] +\
               [x for x in dr_lim[demo_var].unique().tolist() if '*' in x]    

    
    for var in limit_vars:
        reg_df = dr_lim.loc[(~pd.isnull(dr_lim[demo_var])) & (~pd.isnull(dr_lim[var]))][[demo_var,var]]    
        reg_df.loc[reg_df[demo_var]==sort[1],'dummy'] = 1
        reg_df.loc[reg_df[demo_var]==sort[0],'dummy'] = 0
        reg_df['yvar'] = reg_df[var]
        p = sm.ols(formula="yvar ~ dummy", data=reg_df).fit().pvalues.dummy

        if len(reg_df[demo_var].unique().tolist()) == 1:
            p = 1

        
        hold = dr_lim.groupby(demo_var)[var].mean().to_frame().T*100
        for col in hold.columns:
            hold[col] = hold[col].map('{:.2f}'.format).replace({'nan':''})
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
        if var == limit_vars[0]:
            out = hold
        else:
            out = pd.concat([out,hold],axis = 0)
    out = out[sort]
    out = out.rename(index = rename)


    l1 = [demo_var]*2
    l2 = [x.replace('*','') for x in sort]
    l3 = dr_lim.groupby(demo_var)[demo_var].count()[sort].values.tolist()
    tuples= [(l1[i],l2[i],l3[i]) for i in range(len(l1))]
    out.columns = pd.MultiIndex.from_tuples(tuples,names = ['Variable','Group','N']) 
    
    return out


demo_vars_out = ['Size','Public','Growth Prospects','Pay Dividends','Leverage','Cash','Financial Flexibility']

for demo_var in demo_vars_out:
    
    hold = test_dr_limits(sdata,demo_var)
    if demo_var == demo_vars_out[0]:
        dr_limit_demo_table = hold
    else:
        dr_limit_demo_table = pd.concat([dr_limit_demo_table,hold],axis=1)

print("\n\n\n")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print("Displaying Table VI:")
    print(dr_limit_demo_table)


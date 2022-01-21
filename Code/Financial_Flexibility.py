"""
Code to produce Figure 17, Table 8
John Barry
2022/01/14
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

## Load in the survey data
filename = 'Data/datafile.pkl'
with open(filename, 'rb') as f:
    demo_vars_19,demo_vars_20,demo_vars_20_only,demo_var_list,sdata,sdata_20,sdata_2001 = pickle.load(f)
    f.close()


#%% Figure 17. Why is Maintaining Financial Flexibility Important?

fin_flex_cols = [col for col in sdata if col.startswith(('q7a')) and 'q7a_other' not in col]


## Get aspects of financial flexibility for all firms
aff = sdata[fin_flex_cols+['id_specific','respondent_id']]
aff = aff.dropna(subset = fin_flex_cols,how = 'all')

for col in fin_flex_cols:
    aff[col] = (~pd.isnull(aff[col]))**1
    
    

aff_totals = aff[fin_flex_cols].mean().to_frame()
aff_totals.columns = ["Aspects of financial flexibility"]

index_dict = {'q7a_access_lt_debt_markets':'Access long-term debt markets',
              'q7a_access_st_funding':'Access short-term funding',
              'q7a_access_equity_mkts':'Access equity markets',
              'q7a_preserve_line_of_credit':'Preserve lines of credit',
              'q7a_maintain_cash_balance':'Maintain a large cash balance',
              'q7a_avoid_stress_crises':'Avoid financial distress during downturns',
              'q7a_pursue_investment_opps':'Pursue investment opportunities'}
aff_totals = aff_totals.rename(index=index_dict)


aff_totals = aff_totals.sort_values(by = 'Aspects of financial flexibility')



## Now by size
demo_var = 'Size'

aff = sdata[fin_flex_cols + ['id_specific','respondent_id'] + demo_vars_19]
aff = aff.dropna(subset = fin_flex_cols,how = 'all')

new_cols = [x for x in demo_vars_20 if x not in demo_vars_19]
aff_demo = aff.merge(sdata_20[['respondent_2019_sg_id']+new_cols].rename(columns = {'respondent_2019_sg_id':'respondent_id'}),
                     on = 'respondent_id',how = 'left').drop_duplicates(subset = ['respondent_id'])


fin_flex_cols = [col for col in sdata if col.startswith(('q7a')) and 'q7a_other' not in col]

for col in fin_flex_cols:
    aff_demo[col] = (~pd.isnull(aff_demo[col]))**1

    

sort = [x for x in sdata[demo_var].dropna().unique().tolist() if '*'  in x] + \
       [x for x in sdata[demo_var].dropna().unique().tolist() if '*' not    in x]


aff_size = aff_demo.groupby([demo_var])[fin_flex_cols].mean().T.sort_values(by = sort[0])
aff_size = aff_size[sort].rename(columns = {x:x.replace('*','') for x in sort})

index_dict = {'q7a_access_lt_debt_markets':'Access long-term debt markets',
              'q7a_access_st_funding':'Access short-term funding',
              'q7a_access_equity_mkts':'Access equity markets',
              'q7a_preserve_line_of_credit':'Preserve lines of credit',
              'q7a_maintain_cash_balance':'Maintain a large cash balance',
              'q7a_avoid_stress_crises':'Avoid financial distress during downturns',
              'q7a_pursue_investment_opps':'Pursue investment opportunities'}
aff_size = aff_size.rename(index=index_dict)

aff_size = aff_size.sort_values(by = 'Large',ascending=False)


aff_size.columns = pd.MultiIndex.from_tuples([('Aspects of Financial Flexibility',
                                               x) for x in aff_size.columns],
                                             names = [' ','Size'])

print("\n\n\n")
print("Displaying data for Figure 17:")
print(aff_size)



#%% Table VIII. Why Do Companies Maintain Financial Flexibility?

def test_aspects(aff_demo,demo_var,fin_flex_cols):
    pd.options.mode.chained_assignment = None  # default='warn'
    aff_test = aff_demo[[demo_var]+fin_flex_cols]
    
    aff_test = aff_test.dropna(subset = [demo_var])
    aff_test = aff_test.dropna(subset = fin_flex_cols,how = 'all')
    
    
    demo_vals = aff_test[demo_var].unique().tolist()
    
    sort = [x for x in demo_vals if '*' not in x] + [x for x in demo_vals if '*' in x]
    
    out = round(aff_test.groupby([demo_var])[fin_flex_cols].mean().T[sort]*100,2)
    
    for col in out.columns:
        out[col] = out[col].map('{:.2f}'.format).replace('nan','')
        
    
    import statsmodels.formula.api as sm
    
    for var in fin_flex_cols:
        reg_df = aff_test[[var,demo_var]]
        reg_df.loc[reg_df[demo_var]==sort[1],'dummy'] = 1
        reg_df.loc[reg_df[demo_var]==sort[0],'dummy'] = 0
        result = sm.ols(formula="{} ~ dummy".format(var), data=reg_df).fit()
        
        means = out.loc[out.index == var].squeeze().replace({'*',''}).tolist()
        if (result.pvalues.dummy<=0.1) & (result.pvalues.dummy>0.05):
                means[1] = means[1]+'*'
        elif (result.pvalues.dummy<=0.05 ) & (result.pvalues.dummy>0.01 ):
            means[1] = means[1]+'**'
        elif (result.pvalues.dummy<=0.01 ):
            means[1] = means[1]+'***'
        out.loc[out.index == var,sort[0]] = means[0]
        out.loc[out.index == var,sort[1]] = means[1]
        
    n_list = aff_test.groupby([demo_var])[demo_var].count().to_dict()
    cols = out.columns.tolist()
    tuples = [(demo_var,x.replace('*',''),n_list[x]) for x in cols]
    out.columns = pd.MultiIndex.from_tuples(tuples,names = ['Demographic Variable','Group','N'])
    return out



reindex = aff_size.index.tolist()
# reindex.reverse()
demo_vars_out = ['Size','Public','Growth Prospects','Pay Dividends','Leverage','Cash','Financial Flexibility']


for demo_var in demo_vars_out:
    if demo_var == demo_vars_out[0]:
        aff_demo_test = test_aspects(aff_demo,demo_var,fin_flex_cols)
    else:
        aff_demo_test = pd.concat([aff_demo_test,test_aspects(aff_demo,demo_var,fin_flex_cols)],axis=1)
aff_demo_test = aff_demo_test.rename(index=index_dict)

aff_demo_test = aff_demo_test.reindex(reindex)

aff_demo_test.columns = pd.MultiIndex.from_tuples(
    [x for x in aff_demo_test.columns],
    names = ['Variable','Group','N'])

print('\n\n\n')
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print("Displaying Table VIII:")
    print(aff_demo_test)
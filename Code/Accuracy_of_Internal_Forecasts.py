"""
Code to produce Table A5.1
John Barry
2022/01/14
"""
import pandas as pd
import pickle
import Functions.winsor as winsor

    
## Load in the survey data
filename = 'Data/datafile.pkl'
with open(filename, 'rb') as f:
    demo_vars_19,demo_vars_20,demo_vars_20_only,demo_var_list,sdata,sdata_20,sdata_2001 = pickle.load(f)
    f.close()


#%%


sdata_20 = sdata_20.rename(columns = {'profit_margin_measure_2019foreca':'profit_margin_measure_2019forecast',
                                      'year_end_lt_borrowing_interest_r':'lt_interest_rate_2019forecast',
                                      'yearend_lt_borrowing_interest_ra':'lt_interest_rate_2019actual',
                                      'year_end_debt_measure_2019foreca':'yearend_debt_measure_2019forecast',
                                      'year_end_cash_assets_2019forecas':'yearend_cash_assets_2019forecast'})

sdata_20['rd_2019actual'] = pd.to_numeric(sdata_20['rd_2019actual'])


impact_rename = {'a':'Revenue Growth',
                 'b':'Employment Growth',
                 'c':'Wage Growth',
                 'd':'Cash/Assets',
                 'e':'Long-Term Interest Rate',
                 'f':'Profit Margin',
                 'g':'Debt Ratio',
                 'h':'Capital Spending',
                 'i':'R&D Spending',
                 'j':'Dividends',
                 'k':'Repurchases',
                 'l':'Patents',
                 'm':'Trademarks'}

miss_rename = {'revenue_growth':'Revenue Growth',
               'wage_growth':'Wage Growth',
               'patents':'Patents',
               'trademarks':'Trademarks',
               'dividends':'Dividends',
               'repurchase':'Repurchases',
               'capex':'Capital Spending',
               'rd':'R&D Spending',
               'profit_margin_measure':'Profit Margin',
               'revenue_growth':'Revenue Growth',
               'dom_emp_growth':'Employment Growth',
               'yearend_cash_assets':'Year-end Cash/Assets',
               'yearend_debt_measure':'Year-end Debt Measure',
               'lt_interest_rate':'Long-Term Interest Rate'}

fore_act = sdata_20[['id_specific_2020']+[x for x in sdata_20 if 'actual' in x or 'forecast' in x]]

fore_act = fore_act.dropna(how = 'all',subset = [x for x in fore_act if x not in ['id_specific_2020']])

forecast_rename = {x+"_2019forecast":x+"_forecast" for x in list(miss_rename.keys())}

actual_rename   = {x+"_2019actual":x+"_actual" for x in list(miss_rename.keys())}

fore_act = fore_act.rename(columns =forecast_rename ).rename(columns = actual_rename)

cols = fore_act.columns.tolist()
cols.sort()

for var in list(miss_rename.keys()):
    forecast = var+"_forecast"
    actual   = var+"_actual"
 
    fore_act.loc[((~pd.isnull(fore_act[forecast])) & (~pd.isnull(fore_act[actual])) & (fore_act[forecast]!=fore_act[actual])),"keep_{}".format(var)] = 1


for var in list(miss_rename.keys()):
    forecast = var+"_forecast"
    actual   = var+"_actual"
    
    fore_act[var+'_error'] = fore_act[actual] - fore_act[forecast]
    fore_act[var+"_error"] = winsor.winsor(fore_act[var+"_error"])
    fore_act.loc[fore_act[var+"_error"]<0,var+"_negative_miss"] = 1
    fore_act.loc[fore_act[var+"_error"]>=0,var+"_negative_miss"] = 0
    fore_act.loc[fore_act[var+"_error"]>0,var+"_positive_miss"] = 1
    fore_act.loc[fore_act[var+"_error"]<=0,var+"_positive_miss"] = 0

errors = [x for x in fore_act if 'error' in x]


neg_misses = [x for x in fore_act if 'negative_miss' in x]
pos_misses = [x for x in fore_act if 'positive_miss' in x]

check = fore_act[errors].describe().T

miss_var = fore_act[neg_misses+pos_misses]

neg = miss_var[neg_misses].mean()
neg = neg.rename(index = {x:x.replace("_negative_miss","") for x in neg.index})
pos = miss_var[pos_misses].mean()
pos = pos.rename(index = {x:x.replace("_positive_miss","") for x in pos.index})

accuracy = pd.concat([neg,pos],axis=1).rename(columns = {0:'Negative Miss Prob',1:'Positive Miss Prob'})

accuracy['Accurate'] = 1-accuracy.sum(axis=1)

accuracy = accuracy.rename(index = miss_rename)

accuracy = accuracy.reindex(['Negative Miss Prob','Accurate','Positive Miss Prob'],axis=1)

for col in accuracy.columns:
    accuracy[col] = (accuracy[col]*100).map('{:.2f}'.format).replace({'nan':''})
    
print("\n\n\n")
print("Displaying Table A5.I:")
print(accuracy)
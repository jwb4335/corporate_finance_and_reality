
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
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

#%% FIGURE 6: Reasons that Capital Spending Outcomes Differ from Forecasts


## Need to compute capital spending forecast errors
def get_misses(sdata_20):
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
                   'q3_16']]
    
    af = af.rename(columns = {'dom_emp_growth_2019forecast':'emp_growth_2019forecast',
                              'dom_emp_growth_2019actual'  :'emp_growth_2019actual',
                              'q3_1':'capex_2020',
                              'q3_7':'emp_2020',
                              'q3_13':'revenue_2020',
                              'q3_16':'wage_2020'})
    
    
    
    af['wageemp_growth_2019forecast'] = af['emp_growth_2019forecast']
    af['wageemp_growth_2019actual']   = af['emp_growth_2019actual']
    af['wageemp_2020'] = af['wage_2020']
       
    af.columns = [x.replace('_growth',"") for x in af.columns]
    for var in ['revenue','wage','capex','emp','wageemp']:
        af['difference_{}'.format(var)] = af['{}_2019actual'.format(var)] - af['{}_2019forecast'.format(var)]
    
        af.loc[af['difference_{}'.format(var)]<0,'miss_{}'.format(var)] = 'Low Miss'
        af.loc[af['difference_{}'.format(var)]>0,'miss_{}'.format(var)] = 'High Miss*'
        #af.loc[af['difference_{}'.format(var)]==0,'miss_{}'.format(var)] = 'Accurate'

    af = af[['id_specific_2020']+[x for x in af.columns if x not in ['id_specific_2020']]]

    return af


## Calculates the reasons that capital spending missed
def get_capex_reasons(sdata_20):
        
    capex = sdata_20[[x for x in sdata_20.columns if 'q15a' in x] + ['id_specific_2020']]
    capex = capex[[x for x in capex if x not in ['q15a_pleaseexplain_wave2','q15a9_wave2','q15a20_wave2','q15a13_wave2']]]

    capex = capex.dropna(how = 'all',subset = [x for x in capex.columns if 'q15a' in x])
    capex_reasons =  [x for x in capex.columns if 'q15a' in x]
    
    for var in capex_reasons:
        capex[var] = (~pd.isnull(capex[var]))**1
        
    capex_dict = {1:"Current Profits",
        2:"Expected Future Profits",
        3:"Cash Holdings",
        4:"Borrowing Interest Rate",
        5:"Access to Borrowing",
        6:"Our Current Debt Level",
        7:"Stock Price Movements",
        8:"Demand for our Product",
        9:"Analyst Outlook",
        10:"Actions of Competitors",
        11:"Planned Acquisition or Divestiture",
        12:"Credit Rating Considerations",
        13:"Attention from Activists",
        14:"Global GDP Growth",
        15:"Domestic GDP Growth",
        16:"Consumer Spending",
        17:"Government Spending",
        18:"Commodity Prices",
        19:"Interest Rates",
        20:"Inflation",
        21:"Exchange Rates",
        22:"Economic Uncertainty",
        23:"International Trade/Tariffs",
        24:"Political Uncertainty",
        25:"Price/Availability of Capital"}
    
    capex_col_dict = {"q15a{}_wave2".format(list(capex_dict.keys())[i]):list(capex_dict.values())[i] for i in np.arange(len(capex_dict))}

    
    capex = capex.rename(columns = capex_col_dict)
    
    capex = capex[['id_specific_2020']+[x for x in capex.columns if x not in ['id_specific_2020']]]
    
    capex = capex[[x for x in capex if 'Exchange rates' not in x]]
    
    capex_reasons = [x for x in capex.columns if x not in ['id_specific_2020']]
        
    return capex,capex_reasons

af = get_misses(sdata_20)

capex,capex_reasons =  get_capex_reasons(sdata_20)

capex = capex.merge(af[['id_specific_2020']+[x for x in af if 'miss_' in x]],on='id_specific_2020')

del capex['miss_wageemp']

miss_rename = {'miss_capex':'Capital Spending',
               'miss_revenue':'Revenue',
               'miss_wage':"Wages",
               'miss_emp':'Employment'}

capex = capex.rename(columns = miss_rename)


capex_reason_err = capex.groupby(['Capital Spending'])[capex_reasons].mean().T
capex_reason_err = capex_reason_err[[x for x in capex_reason_err if not '*' in x]+[x for x in capex_reason_err if '*' in x]]
capex_reason_err = capex_reason_err.sort_values(by = 'Low Miss')
capex_reason_err = capex_reason_err.rename(columns = {"Low Miss":'Actual Below Forecast in 2019',
                                                      "High Miss*":'Actual Above Forecast in 2019'})

keep = [ 'Domestic GDP Growth','Borrowing Interest Rate','Planned Acquisition or Divestiture',
         'Price/Availability of Capital', 'Political Uncertainty', 'Actions of Competitors',
         'Economic Uncertainty','International Trade/Tariffs', 'Access to Borrowing',
         'Expected Future Profits','Cash Holdings','Current Profits', 'Demand for our Product']

capex_reason_err = capex_reason_err.T[keep].T

capex_reason_err = capex_reason_err.sort_values(by = ['Actual Below Forecast in 2019'],ascending=False)

## Display table
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print("Displaying data for Figure 6")
    print(capex_reason_err)


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


#%% Figure A7.1. How Often Do Companies Change Their Target Debt Ratio?


def changed_debt_ratio(sdata,limit_credit_rating = "No"):
        
    pdm = sdata[[col for col in sdata if col.startswith(('q6_'))] + ['credit_rating']]
    
    
    def primary_dm(vec):
        x = vec[0]
        x = x.replace(" ","")
        x = x.lower()
        if x == "debt_assets":
            primary_debt_metric = 1
            label = 'Debt/Assets'
        elif x == "debt_value":
            primary_debt_metric = 2
            label = 'Debt/Value'
        elif x == "debt_equity":
            primary_debt_metric = 3
            label = 'Debt/Equity'
        elif x == "liabilities_assets":
            primary_debt_metric = 4
            label = 'Liabilities/ Assets'
        elif x == "debt_ebitda":
            primary_debt_metric = 5
            label = 'Debt/EBITDA'         
        elif x == "interest_coverage":
            primary_debt_metric = 6
            label = 'Interest Coverage'
        elif x == "credit_rating":
            primary_debt_metric = 7
            label = 'Credit Rating' 
        else:
            primary_debt_metric = 0
            label = '' 
        return primary_debt_metric, label
    
    
    pdm[['primary_debt_metric','label']] = pdm[['q6_primary_debt_ratio']].apply(primary_dm,axis=1,
             result_type = "expand")
    
    if limit_credit_rating == "Yes":
        pdm.loc[pdm.credit_rating=="NONE",'q6_credit_rating']=np.nan
    
    pdm_graph = pdm[['q6_debt_assets',
                    'q6_debt_value',
                    'q6_debt_equity',
                    'q6_liabilities_assets',
                    'q6_debt_ebitda',
                    'q6_credit_rating',
                    'q6_interest_coverage']]
    
    pdm_graph = pdm_graph[pdm_graph.notnull().any(1)]
    
    pdm_graph.columns = ["Debt/Assets",
                         "Debt/Value",
                         "Debt/Equity",
                         "Liabilities/ Assets",
                         "Debt/EBITDA",
                         "Credit Rating",
                         "Interest Coverage"]
    
    debt_ratios = ["Debt/Assets",
                         "Debt/Value",
                         "Debt/Equity",
                         "Liabilities/ Assets",
                         "Debt/EBITDA",
                         "Credit Rating",
                         "Interest Coverage"]
    def stringnan(string):
        return string != string
    
    for dr in debt_ratios:
        pdm_graph.loc[pdm_graph[dr] == 1, "primary"] = dr
        pdm_graph.loc[((pdm_graph[dr]==2) & (stringnan(pdm_graph["primary"]) == True)),"primary"] = dr
        pdm_graph.loc[((pdm_graph[dr]==3) & (stringnan(pdm_graph["primary"]) == True)),"primary"] = dr
        
        
    cd_cols = 'q6_changes_debt_target_10yr'
    cd = pdm[[cd_cols,'label','q6_debt_target_range']]
    cd = cd.loc[cd['q6_debt_target_range']!=4].loc[~pd.isnull(cd['q6_debt_target_range'])]
    cd = cd[[x for x in cd if x not in ['q6_debt_target_range']]]
    cd.loc[cd[cd_cols] == 7,cd_cols] = np.nan
    cd = cd.loc[cd['label']!=''].dropna(subset = [cd_cols])
    cd_out = cd.groupby([cd_cols,'label']).size().unstack(fill_value = 0).divide(
             cd.groupby(['label'])[cd_cols].count())
    index_rename = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6+'}
   
    cd_out = cd_out.rename(index = index_rename)
    return cd_out


cd_final = changed_debt_ratio(sdata)

sort = ['Debt/EBITDA',
 'Debt/Assets',
 'Interest Coverage',
 'Debt/Equity',
 'Liabilities/ Assets',
 'Debt/Value']

cd_final = cd_final[sort]

print("\n\n\n")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print("Displaying data for Figure A7.1:")
    print(cd_final)


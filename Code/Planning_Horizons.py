"""
Created on Sat Jan  8 17:08:25 2022

jwb
"""



import pandas as pd
import pickle
# # Set the directories
# if "barry" in os.getcwd():
#     os.chdir("C:\\Users\\barry\\Dropbox\\Graham_survey\\March_2019_Survey\\survey_code\\python_code\\Functions_v3\\Submission\\")
#     cwd = os.getcwd()
#     sys.path.append(os.path.join(cwd,))
#     table_save = 'C:\\Users\\barry\\Dropbox\\Graham_Survey\\March_2019_Survey\\graph_data\\tables_for_writeup_march21.xlsx'

## Load in the survey data
filename = 'Data/planning_horizons_2018q3.pkl'
with open(filename, 'rb') as f:
    data_18q3 = pickle.load(f)
    f.close()

#%%
project_length = data_18q3[[x for x in data_18q3 if 'global_project' in x]].dropna(how = 'any')

project_length = project_length.join(data_18q3[['industry']])

project_length = project_length.rename(columns = {'global_projectlength_1':'plan_5yrs_ago',
                                                  'global_projectlength_2':'productive_life_5yrs_ago',
                                                  'global_projectlength_3':'plan_now',
                                                  'global_projectlength_4':'productive_life_now'})

project_length = project_length.groupby('industry').mean()

project_length = project_length.reindex(['Retail','Finance','Services','Tech','Manufacturing','Healthcare'])

planning = project_length[[x for x in project_length.columns if 'plan' in x]]
planning.columns = ['5 years ago (2013)','Today (2018)']
productive_life = project_length[[x for x in project_length.columns if 'productive' in x]]
productive_life.columns = ['5 years ago (2013)','Today (2018)']

planning_columns = [('Can reliably plan T years into the future',x) for x in planning.columns]

productive_life_columns = [('Expected life of a new project',x) for x in productive_life.columns]

planning.columns = pd.MultiIndex.from_tuples(planning_columns,names = ['question','time period'])

productive_life.columns = pd.MultiIndex.from_tuples(productive_life_columns,names = ['question','time period'])

out = pd.concat([planning,productive_life],axis=1)


## Display data
print("\n\n\n")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print("Displaying data for Figure 7")
    print(out)

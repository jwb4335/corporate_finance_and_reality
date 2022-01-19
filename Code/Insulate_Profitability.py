
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


#%% Figure A5.3. Does Your Firm Take Actions to Insulate Profitability?


prof = sdata_20[['q8b_wave2','revenue_growth_2019forecast','revenue_growth_2019actual']].dropna(how = 'any')

prof.loc[prof['revenue_growth_2019actual']<prof['revenue_growth_2019forecast'],'group'] = 'Actual Revenue < Forecasted Revenue'
prof.loc[prof['revenue_growth_2019actual']>prof['revenue_growth_2019forecast'],'group'] = 'Actual Revenue > Forecasted Revenue'
prof.loc[prof['revenue_growth_2019actual']==prof['revenue_growth_2019forecast'],'group'] = 'Actual = Forecast'

prof = prof.loc[prof['group']!='Actual = Forecast'].loc[prof['q8b_wave2']!=3]
prof['action_prof'] = prof['q8b_wave2'].map({1:0,2:1})

insulate_prof = prof.groupby(['group'])['action_prof'].mean().to_frame().rename(columns = {'action_prof':'Yes'})
insulate_prof['No'] = 1-insulate_prof['Yes']

insulate_prof = insulate_prof[['No','Yes']]

insulate_prof.columns = pd.MultiIndex.from_tuples([('Actions Taken to Insulate Profitability',x) for x in insulate_prof.columns],
                                        names = ['Question','Group'])

print("\n\n\n")
print("Displaying data for Figure A5.3:")
print(insulate_prof)
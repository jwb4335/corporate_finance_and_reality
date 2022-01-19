
import pandas as pd
import pickle
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


#%%
rename =   {'commprice':'Commodity Prices',
             'cur':'Currency Exchange Rates',
             'gdp':'GDP Growth (USA)',
             'globalgdp':'GDP Growth (Global)',
             'inflation':'Inflation',
             'ir':'Interest Rates',
             'intltrade':'International Trade/Tariffs',
             'conspend':'Consumer Spending',
             'defspend':'Defense Spending',
             'infraspend':'Infrastructure Spending'}

rename_cols = {"q15g_outcome_{}".format(i):rename[i] for i in list(rename.keys())}

macro_cols = list(rename_cols.values())


macro_vars = sdata[[x for x in sdata.columns if x.startswith('q15g')]]

macro_vars = macro_vars.rename(columns = rename_cols)

macro_vars = macro_vars[[x for x in macro_vars if x in macro_cols]]

macro_vars = macro_vars.dropna(subset = macro_cols,how = 'all')



for col in macro_cols:
    macro_vars[col] = (~pd.isnull(macro_vars[col]))**1
    
graph = macro_vars[macro_cols].mean().to_frame().rename(columns = {0:'All'}).sort_values(by = 'All')
""
graph = graph.sort_values(by = 'All',ascending=False)

print("\n\n\n")
print("Displaying data for Figure A6.1:")
print(graph)

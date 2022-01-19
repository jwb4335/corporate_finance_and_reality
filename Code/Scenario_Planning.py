
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import Functions.winsor as winsor

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

## Load in the data from 2019q2 (not part of two-wave special survey)
filename = 'Data/survey_2019q2_hist.pkl'
with open(filename, 'rb') as f:
    [sdata_2019q2]= pickle.load(f)
    f.close()

    
#%% Figure A5.1: Scenario Planning

def scenario_planning_2019q2(sdata):

        
    sp = sdata[['q6a_1','q6a_2','q6a_3','q6a_5']]
    sp = sp[sp.notnull().any(1)]
    sp = ((sp.notnull()).astype(int))
    
    sp = sp.sum().to_frame()/len(sp)
    sp.columns = ["Does your company conduct scenario analysis?"]
    index_dict = {'q6a_1':'For the entire company',
                  'q6a_2':'For some divisions or units',
                  'q6a_3':'For specific projects',
                  'q6a_5':'No'}
    
    sp = sp.rename(index=index_dict)
    
    sp = sp.reindex(['No','For specific projects','For some divisions or units','For the entire company'])

    sp2 = sdata[['q6a_followup_1','q6a_followup_2','q6a_followup_3']]
    sp2 = sp2[sp2.notnull().any(1)]
    sp2 = ((sp2.notnull()).astype(int))
    
    sp2 = sp2.sum().to_frame()/len(sp2)
    sp2.columns = ["What type of scenario planning do you use?"]
    index_dict2 = {'q6a_followup_1':'Downside/base case/upside',
                  'q6a_followup_2':'Specific events (e.g. oil price spikes)',
                  'q6a_followup_3':'Government regulation or legislation',
                  'q6a_followup_4':'Other'}
    
    sp2 = sp2.rename(index=index_dict2)
    return sp, sp2


sp, sp2 = scenario_planning_2019q2(sdata_2019q2)

print("\n\n\n")
print("Displaying data for Figure A5.1 Panel A:")
print(sp)
print("\n\n\n")
print("Displaying data for Figure A5.1 Panel B:")
print(sp2)


#%% Figure A5.2. Scenario Planning for Cases:


## Percentage of firms that scenario plan for each of the five scenarios
def scenario_planning(sdata):
         
    sp = sdata[[col for col in sdata if col.startswith(('q15a_'))]]
    sp = sp[sp.notnull().any(1)]
    sp = sp.loc[sp["q15a_scen_analysis"]<3].apply(pd.value_counts)/len(sp.loc[sp["q15a_scen_analysis"]<3])
    sp.index = ["Yes","No"]
    sp.columns = ["Do you perform scenario analysis?"]
    sp = sp.T
    
    sp_columns = [x for x in sdata if x.startswith('q15c')]
    
    rename = {'extdown':'Extreme Downside',
              'down':'Downside',
              'base':'Base Case',
              'up':'Upside',
              'extup':'Extreme Upside'}
    rename2 = {"q15c_scen_planning_{}".format(x):rename[x] for x in list(rename.keys())}
    
    sp_cases = sdata[sp_columns].dropna(how = 'all')
    
    for col in sp_columns:
        sp_cases[col] = (~pd.isnull(sp_cases[col]))**1
    
    sp_cases_out = sp_cases.mean().to_frame().T.rename(columns = rename2,index = {0:'Planning for Scenarios'})
    
    return sp_cases_out

## This gets the average forecast by scenario (only revenue)
def downside_upside(sdata,sdata_20,keep_matches = "Yes",winsor_min=0.01,winsor_max=0.99):
  
    rge_cols = [col for col in sdata if col.startswith(('q5_'))]
    rge = sdata[rge_cols+['respondent_id']]
    rge = rge.dropna(subset = rge_cols,how = 'any')
    
    rge['merge'] = rge.index
    
    du_cols = [col for col in sdata if col.startswith(('q15d_'))]
    
    for col in [x for x in du_cols if 'debt' in x]:
        sdata.loc[sdata['q6_primary_debt_ratio']!='Debt_Assets',col] = np.nan
    
    du = sdata[du_cols]
    
    revenue = du[[col for col in du if 'revenue' in col]]
    revenue = revenue[revenue.notnull().any(1)]
    
    du = pd.concat([revenue],axis=1,sort=True)
    
    du['merge'] = du.index
    
    rgedu = rge.merge(du,on='merge',how='left',suffixes=[False, False], indicator=True)
    
    rgedu.loc[rgedu._merge == 'both','match'] = 1
    rgedu.loc[rgedu._merge == 'left_only','match'] = 0
    
    del rgedu['merge'], rgedu['_merge']

    if keep_matches == "Yes":
        rgedu = rgedu.loc[rgedu.match == 1]
        
    for col in [x for x in rgedu if x not in ['match','respondent_id']]:
        rgedu[col] = winsor.winsor(rgedu[col],winsor_min,winsor_max,'trim')

    rgedu = rgedu[rgedu.notnull().any(1)]

        
    actuals = [x for x in sdata_20 if '2019actual' in x]
    
    sdata_20['rd_2019actual'] = pd.to_numeric(sdata_20['rd_2019actual'])

    actuals_2019 = sdata_20[['respondent_2019_sg_id','id_specific_2020','debt_measure_ranked_item'] + \
                            actuals]
        
    actuals_2019.loc[actuals_2019['debt_measure_ranked_item']!='Debt_Assets','yearend_debt_measure_2019actual'] = np.nan
        
    actuals_2019 = actuals_2019.dropna(subset = actuals,how = 'all')
    
    actuals_2019 = actuals_2019[[x for x in actuals_2019 if x not in ['debt_measure_ranked_item']]]
    
    for col in actuals:
        actuals_2019.loc[actuals_2019[col]>=100,col] = actuals_2019[col]-100
    
    actuals_2019 = actuals_2019.rename(columns = {'respondent_2019_sg_id':'respondent_id'})
    
    out = rgedu.merge(actuals_2019,on = 'respondent_id',how = 'left')
        
    out.loc[out['respondent_id'] == 43,'q15d_revenue_extdown'] = out['q15d_revenue_extdown']*-1
    
        
    for col in actuals:
        out[col] = winsor.winsor(out[col],0.05,0.95,'trim')
    
    rev = out[[x for x in out if 'revenue' in x and 'q5' not in x]].dropna(how = 'all').agg(['count','mean']).T
    rev.index = ['Extreme Downside','Downside','Base Case','Upside','Extreme Upside','Actual']
    rev.columns = pd.MultiIndex.from_tuples([('Revenue',x) for x in rev.columns],names = ['Variable','Stat'])
   
    final = pd.concat([rev], axis=1)  

    data = out
    return final, data


sp_cases_out = scenario_planning(sdata)


## Get the average revenue forecast for each scenario, needed for the axis labels of the figure
rev_others,_ = downside_upside(sdata,sdata_20,keep_matches = "Yes",winsor_min=0.01,winsor_max=0.99) 
means = rev_others.loc[rev_others.index!='Actual'][("Revenue","mean")].round(0).astype(int).astype(str).tolist()

xlabels = sp_cases_out.columns.tolist()
xlabels = [x[0] + '\n(Average Revenue Forecast = {}%)'.format(x[1]) for x in list(zip(xlabels,means))]


def scenario_planning_cases(sp_cases_out,xlabels,fig_size = (9.5,5.5),
                           total_width=0.8, single_width=0.8, legend=True, 
                           alph = 1,legend_loc = 'best',ncols = 2,fontsize = 12):
    
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors = [colors[0]]*len(sp_cases_out.columns)
    fig,ax = plt.subplots()

    # Number of bars per group
    n_bars = len(sp_cases_out)
    
    # The width of a single bar
    bar_width = total_width / n_bars
    
    # List containing handles for the drawn bars, used for the legend
    bars = []
    x_values = []
    # Iterate over all data
    for i, (name, values) in enumerate(sp_cases_out.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2
    
        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.barh(x + x_offset, y, height=bar_width * single_width, color=colors[i],alpha=1)
    
        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])
        x_values = x_values + [x_offset]
    
    ax.set_yticks(x_values)
    ax.set_yticklabels(xlabels,fontsize = 10)
    # ax.set_xticks(np.arange(len(xlabels)))
    # ax.set_xticklabels(xlabels,fontsize = fontsize)
    ax.set_axisbelow(True)
    ax.grid(axis='x',alpha=0.5,linestyle='--',zorder=-100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    from matplotlib.ticker import FuncFormatter
    ax.set_xticks(np.arange(0,1.01,0.2))
    plt.xticks(fontsize = fontsize)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
    # plt.gca().invert_yaxis()
    return fig


print("\n\n\n")
print("Displaying Figure A5.2:")
fig = scenario_planning_cases(sp_cases_out,xlabels)
plt.show()

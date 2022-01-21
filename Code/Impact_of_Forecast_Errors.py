"""
Code to produce Figure 8
Rong Wang & John Barry 
2022/01/14
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import Functions.winsor as winsor

    
## Load in the survey data
filename = 'Data/datafile.pkl'
with open(filename, 'rb') as f:
    demo_vars_19,demo_vars_20,demo_vars_20_only,demo_var_list,sdata,sdata_20,sdata_2001 = pickle.load(f)
    f.close()

# some data items

sdata_20.loc[sdata_20['survey_date']>=pd.Timestamp('2020-03-15'),'covid_time'] = 1
sdata_20.loc[sdata_20['covid_time']!=1,'covid_time'] = 0
sdata_20 = sdata_20.rename(columns = {'profit_margin_measure_2019foreca':'profit_margin_measure_2019forecast',
                                      'year_end_lt_borrowing_interest_r':'lt_interest_rate_2019forecast',
                                      'yearend_lt_borrowing_interest_ra':'lt_interest_rate_2019actual',
                                      'year_end_debt_measure_2019foreca':'yearend_debt_measure_2019forecast',
                                      'year_end_cash_assets_2019forecas':'yearend_cash_assets_2019forecast'})

sdata_20['rd_2019actual'] = pd.to_numeric(sdata_20['rd_2019actual'])

  
    
#%% Figure 8: Which Internal Forecasts Have the Biggest Impact?

## Percentages of positive and negative misses

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
               'dom_emp_growth':'Employment Growth',
               'yearend_cash_assets':'Cash/Assets',
               'yearend_debt_measure':'Debt Ratio',
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

out = pd.concat([neg,pos],axis=1).rename(columns = {0:'Negative Miss Prob',1:'Positive Miss Prob'})

out['Accurate'] = 1-out.sum(axis=1)


## Firm impacts

impact_rename_cols = {'q7b_impact_wave2{}'.format(i):impact_rename[i] for i in list(impact_rename.keys())}

impact_vars = list(impact_rename_cols.values())

impact_tab = sdata_20[['id_specific_2020'] + list(impact_rename_cols.keys()) + demo_vars_20]

impact_tab = impact_tab.rename(columns = impact_rename_cols)

impact_tab = impact_tab.dropna(subset = impact_vars,how = 'all')

impact_tab = impact_tab.merge(fore_act[['id_specific_2020'] + [x for x in fore_act if 'keep' in x]])

for var in list(miss_rename.keys()):
    impact_tab[miss_rename[var]] = (~pd.isnull(impact_tab[miss_rename[var]]))**1
    impact_tab.loc[impact_tab["keep_{}".format(var)]!=1,miss_rename[var]] = np.nan
    impact_tab["mean_{}".format(miss_rename[var])] = (~pd.isnull(impact_tab["keep_{}".format(var)]))**1


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
                   'q3_16',
                   'covid_time']]
    
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

af = get_misses(sdata_20)

misses = [x for x in af.columns if x.startswith('miss')]

impact_tab = impact_tab.merge(af[['id_specific_2020']+misses],on = 'id_specific_2020')

impact_graph = impact_tab[impact_vars].mean().to_frame().rename(columns = {0:'% Importance'}).sort_values(by = '% Importance')
means = impact_tab[[x for x in impact_tab if "mean_" in x]].mean().to_frame().rename(columns = {0:'% with Miss'})

means = means.rename(index ={x:x.replace("mean_","") for x in impact_tab.columns.tolist() if "mean_" in x})

impact_graph = pd.concat([impact_graph,means],axis=1).sort_values(by='% Importance')

impact_graph = impact_graph.loc[(impact_graph.index!='Patents') & (impact_graph.index!='Trademarks')]


## Impact by size 

# by size
impact_by_size = impact_tab.groupby(['Size'])[impact_vars].mean().T
impact_by_size = impact_by_size.loc[(impact_by_size.index!='Patents') & (impact_by_size.index!='Trademarks')].reindex(impact_graph.index.tolist())
bysize = impact_by_size.to_dict('list')


## Draw bar graphs
print("\n\n\n")
print("Displaying Figure 8:")
# set upfigure parameters
fig_size = (6.5,6)
total_width = 0.8
single_width = 0.9
legend=True
alph=1
legend_loc='best'
ncols = 4
xlabels = [x for x in impact_by_size.index.tolist()]

# need some additional packages    
from PIL import ImageColor # PIL: Python Image Library

# set up colors
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
colors = [colors[1], colors[0]]

# draw the background
fig,ax = plt.subplots(figsize = fig_size)

# number of bars per group
n_bars = 2

# the width of a single bar
bar_width = total_width / n_bars

# the offset in x direction of that bar
x_offset = (1 - n_bars / 2) * bar_width + bar_width / 2

# list containing handles for the drawn bars, used for the legend
bars = []

# bysize - large firms
for x, y in enumerate(list(bysize.values())[0]):
    bar = ax.barh(x + x_offset, y, height=bar_width * single_width, color=colors[1],alpha=alph)
bars.append(bar[0])

# bysize - small firms
for x, y in enumerate(list(bysize.values())[1]):
    col = tuple([x/255 for x in  ImageColor.getcolor(colors[0], "RGB")])+ (0.5,)
    bar = ax.barh(x - x_offset, y, height=bar_width * single_width, color=colors[0],alpha=alph)
bars.append(bar[0])

# axis setup
ax.set_yticks(np.arange(len(xlabels)))
ax.set_yticklabels(xlabels,fontsize = 12)
ax.set_axisbelow(True)
ax.grid(axis='x',alpha=0.5,linestyle='--',zorder=-100)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
from matplotlib.ticker import FuncFormatter
ax.set_xticks(np.arange(0,1,0.2))
ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 

fig = plt.gcf()

# draw legend if we need
leg = fig.legend([bars[0],bars[1]], ['Large','Small'],ncol = 2,loc = 'center',
            bbox_to_anchor = (0.35,0.025),fontsize = 11,title_fontsize = 12)

plt.show()


"""
Code to produce Figure 9
John Barry
2022/01/14
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

## Load in the survey data
filename = 'Data/miscalibration_data_global.pkl'
with open(filename, 'rb') as f:
    [revenue_full] = pickle.load(f)
    f.close()



#%% FIGURE 9: Revenue Calibration by Region

def get_region_industry(revenue_graph):
    pd.options.mode.chained_assignment = None  # default='warn'

    out_vars = ['revenue_p10_2019', 'revenue_p50_2019','revenue_p90_2019','rev_actual_2019']
    
    graph_data = revenue_graph.groupby(['location'])[out_vars].mean()
    graph_data['N'] = revenue_graph.groupby(['location'])['rev_actual_2019'].count()
    

    graph_data = revenue_graph.groupby(['location'])[out_vars].mean()
    
    
    graph_data['N'] = revenue_graph.groupby(['location'])['rev_actual_2019'].count()
   
    
    graph_data['upper'] = abs(graph_data['revenue_p50_2019'] - graph_data['revenue_p90_2019'])
    graph_data['lower'] = abs(graph_data['revenue_p50_2019'] - graph_data['revenue_p10_2019'])
    
    graph_data = graph_data.reindex(['USA','Europe','Asia','Latin America'])

    industry_rename = {1:'Retail',2:'Finance',6:'Services',10:'Tech',11:'Manuf.',12:'Healthcare'}
    
    ind = revenue_graph.loc[revenue_graph['location'] == 'USA']
    
    ind['Industry'] = ind['industry'].map(industry_rename)
    
    
    ind = ind.rename(columns =  {'revenue_growth_2019actual':'rev_actual_2019',
                                              'revenue_growth_2019forecast':'rev_forecast_2019',
                                              'q5_1':'revenue_p10_2019',
                                              'q5_2':'revenue_p50_2019',
                                              'q5_3':'revenue_p90_2019',
                                              'q3_13':'revenue_2020'})
        
    for var in out_vars:
        ind.loc[ind[var]>=200,var] = ind[var]/10
    
    
    ind_graph = ind.groupby(['Industry'])[out_vars].mean()
    ind_graph['N'] = ind.groupby(['Industry'])['rev_actual_2019'].count()
    
    ind_graph['upper'] = abs(ind_graph['revenue_p50_2019'] - ind_graph['revenue_p90_2019'])
    ind_graph['lower'] = abs(ind_graph['revenue_p50_2019'] - ind_graph['revenue_p10_2019'])
    ind_graph = ind_graph.reindex(['Retail','Finance','Services',  'Tech', 'Manuf.', 'Healthcare'])
    
    revenue_full.loc[revenue_full['rev_actual_2019']<=revenue_full['revenue_p10_2019'],'Bad Scenario Realized'] = 1
    revenue_full.loc[revenue_full['rev_actual_2019']>=revenue_full['revenue_p90_2019'],'Good Scenario Realized'] = 1
    revenue_full.loc[(revenue_full['rev_actual_2019']<revenue_full['revenue_p90_2019']) & (revenue_full['rev_actual_2019']>revenue_full['revenue_p10_2019']),
                     'Accurate'] = 1
    
    acc_vars = ['Bad Scenario Realized','Accurate','Good Scenario Realized']
                     
    for var in acc_vars:
        revenue_full[var] = (revenue_full[var] == 1)**1
        
    ind = revenue_full.loc[revenue_full['location'] == 'USA']
    
    ind['Industry'] = ind['industry'].map(industry_rename)

    
    acc_region = revenue_full.groupby(['location'])[acc_vars].mean()*100
    acc_region = acc_region.reindex(['USA','Europe','Asia','Latin America'])
    
    acc_ind = ind.groupby(['Industry'])[acc_vars].mean()*100
    acc_ind = acc_ind.reindex(['Retail','Finance','Services',  'Tech', 'Manuf.', 'Healthcare'])
    

    return graph_data,ind_graph, acc_region,acc_ind


graph_data_full,ind_graph_full,acc_region,acc_ind = get_region_industry(revenue_full) 


for var in acc_region.columns.tolist():
    if acc_region[var].max()>1:
        acc_region[var] = acc_region[var]/100

for var in acc_ind.columns.tolist():
    if acc_ind[var].max()>1:
        acc_ind[var] = acc_ind[var]/100


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
fig, ax = plt.subplots(figsize = (6.5,3.5))


def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True, alph = 1,legend_loc = 'best',ncols = 2,second_dict = None,hatch = None):
    
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    default_colors = prop_cycle.by_key()['color']
   

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = default_colors

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i],alpha=alph,hatch = hatch[i],edgecolor = 'white')

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])
    if second_dict is not None:
        for i, (name, values) in enumerate(second_dict.items()):
            for x, y in enumerate(values):
                x_offset = x_offset = (i - n_bars / 2) * bar_width + bar_width / 2
                ax.scatter(x+x_offset,y,label = '_nolegend_',color = colors[i],edgecolor = 'black',zorder = 100)
    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys(),loc=legend_loc,ncol = ncols)
    return bars
        


def get_subplot(ax,plot_df):
    
    bars = bar_plot(ax,plot_df.to_dict('list'),colors = colors[0:2] + ['silver'],
                    alph =1,single_width = 0.9,hatch = ['','',''])
    
    ax.set_xticks(np.arange(0,len(plot_df),1))
    ax.set_xticklabels(plot_df.index.tolist(),fontsize = 11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.grid(axis='y',alpha=0.5,linestyle='--',zorder=-100)
    ax.set_axisbelow(True)
    ax.get_legend().remove()
    return bars
bars = get_subplot(ax,acc_region)
# ax.set_ylabel("Percent",fontsize = 11)
line = {'Realization $<$ 10th %ile Forecast':[0.10]*4,
       'Realization $\in$ [10th, 90th]':[np.nan]*4,
        'Realization $>$ 90th %ile Forecast':[0.10]*4}
total_width = 0.8
single_width = 0.9
# Number of bars per group
n_bars = len(line)

# The width of a single bar
bar_width = total_width / n_bars
for i, (name, values) in enumerate(line.items()):
    # The offset in x direction of that bar
    x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

    # Draw a bar for every value of that type
    for x, y in enumerate(values):
        bar = ax.bar(x + x_offset, y, width=bar_width * single_width, 
                     color='none',edgecolor = 'black',linewidth = 1.5,alpha=1,zorder = 2)
from matplotlib.ticker import FuncFormatter
ax.set_yticks(np.arange(0,0.8,0.1))
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
ax.annotate("10% if proper\ncalibration",xytext = (0.5,float(0.25)),
            xy = (.4,0.105),zorder = 5,fontsize = 11,
            ha='center',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"))
ax.annotate("",xytext = (0.45,float(0.235)),
            xy = (-0.2,0.105),zorder = 5,fontsize = 11,
            ha='center',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"))


acc_region.columns = ['Realization $<$ 10th Percentile Forecast','Realization $\in$ [10th, 90th]',
                      'Realization $>$ 90th Percentile Forecast']
fig.legend(bars, acc_region.columns.tolist(),bbox_to_anchor = (0.54,-0.1),ncol = 1,loc='center',fontsize = 11)
plt.tight_layout()
print("\n\n\n")
print("Displaying Figure 9:")
plt.show()

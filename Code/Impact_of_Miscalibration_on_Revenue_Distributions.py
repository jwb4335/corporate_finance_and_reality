
import pandas as pd
pd.options.mode.chained_assignment = None
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

#%% Prepare data for the figures

## Get the 2019 data 
id_2019 = sdata_20.loc[np.isnan(sdata_20['respondent_2019_sg_id'])==False][['respondent_2019_sg_id','id_specific_2020']]
revenue_19 = sdata.merge(id_2019,left_on = 'respondent_id',right_on = 'respondent_2019_sg_id')\
            [['id_specific_2020','respondent_2019_sg_id','q3_13','q5_revenue_10pct','q5_revenue_50pct','q5_revenue_90pct']]

revenue_19.loc[pd.isnull(revenue_19['q5_revenue_50pct']),'q5_revenue_50pct'] = revenue_19['q3_13']

revenue_19 = revenue_19.rename(columns = {'q5_revenue_10pct':'revenue_p10_2019',
                                          'q5_revenue_50pct':'revenue_p50_2019',
                                          'q5_revenue_90pct':'revenue_p90_2019',
                                          'q3_13':'revenue_2019'})
    
    
for var in ['revenue_p10_2019','revenue_p50_2019','revenue_p90_2019']:
    revenue_19.loc[revenue_19[var]>200,var] = revenue_19[var]/10

revenue_19['hold_p10'] = copy.deepcopy(revenue_19['revenue_p10_2019'])
revenue_19['hold_p50'] = copy.deepcopy(revenue_19['revenue_p50_2019'])
revenue_19['hold_p90'] = copy.deepcopy(revenue_19['revenue_p90_2019'])

revenue_19.loc[revenue_19['hold_p10']>revenue_19['hold_p90'],'revenue_p90_2019'] = revenue_19['revenue_p90_2019']
revenue_19.loc[revenue_19['hold_p10']>revenue_19['hold_p90'],'revenue_p90_2019'] = revenue_19['revenue_p90_2019']
revenue_19.loc[revenue_19['hold_p50']>revenue_19['hold_p90'],'revenue_p50_2019'] = np.nan
revenue_19.loc[revenue_19['hold_p10']>revenue_19['hold_p50'],'revenue_p50_2019'] = np.nan

del revenue_19['hold_p10'], revenue_19['hold_p50'], revenue_19['hold_p90']

hold_2019_survey = revenue_19.dropna(how = 'any', subset = ['revenue_p10_2019','revenue_p50_2019','revenue_p90_2019'])
hold_2019_survey['keep_2019_survey'] = 1

revenue_19 = revenue_19.merge(hold_2019_survey[['id_specific_2020','keep_2019_survey']],on='id_specific_2020',how='left')

#revenue_19 = revenue_19.dropna(how = 'any', subset = ['revenue_p10_2019','revenue_p50_2019','revenue_p90_2019'])

rev_2019 = ['rev_actual_2019','rev_forecast_2019']
rev_2020 = ['revenue_p10_2020','revenue_p50_2020','revenue_p90_2020']

## Get the 2020 data

revenue_20 = sdata_20[['id_specific_2020','survey_date',
                       'revenue_growth_2019actual','revenue_growth_2019forecast','q5_1','q5_2','q5_3','q3_13','q52_corona']]

revenue_20.loc[(pd.isnull(revenue_20['q5_2'])),'q5_2'] = revenue_20['q3_13']

revenue_20.loc[(revenue_20['q5_2']<revenue_20['q5_1']) | (revenue_20['q5_2']>revenue_20['q5_3']),'q5_2'] = np.nan

check = revenue_20.loc[(revenue_20['q5_3']<revenue_20['q5_1'])]

## Rename the variables
revenue_20 = revenue_20.rename(columns = {'revenue_growth_2019actual':'rev_actual_2019',
                                          'revenue_growth_2019forecast':'rev_forecast_2019',
                                          'q5_1':'revenue_p10_2020',
                                          'q5_2':'revenue_p50_2020',
                                          'q5_3':'revenue_p90_2020',
                                          'q3_13':'revenue_2020'})
for var in ['revenue_p10_2020','revenue_p50_2020','revenue_p90_2020']:
    revenue_20.loc[revenue_20[var]>200,var] = revenue_20[var]/10


    
hold = revenue_20.dropna(subset = rev_2020,how='any')
hold['keep_2020'] = 1

hold2 = revenue_20.dropna(subset = rev_2019,how='any')
hold2['keep_2019'] = 1

hold3 = revenue_20.dropna(subset = rev_2019+rev_2020,how='any')
hold3['keep_all'] = 1

revenue_20 = revenue_20.merge(hold[['id_specific_2020','keep_2020']],on = 'id_specific_2020',how = 'left')
revenue_20 = revenue_20.merge(hold2[['id_specific_2020','keep_2019']],on = 'id_specific_2020',how = 'left')
revenue_20 = revenue_20.merge(hold3[['id_specific_2020','keep_all']],on = 'id_specific_2020',how = 'left')

revenue_20 = revenue_20.merge(revenue_19[['id_specific_2020','revenue_2019','revenue_p10_2019',
                                          'revenue_p50_2019','revenue_p90_2019','keep_2019_survey']],
                              on = 'id_specific_2020',how = 'left')

revenue_20['revenue_p50_2019'] = revenue_20['rev_forecast_2019']
    
revenue_20['rev_forecast_err_2019'] = revenue_20['rev_actual_2019'] - revenue_20['rev_forecast_2019']

## Define the COVID time variable
revenue_20.loc[revenue_20['survey_date']>=pd.Timestamp('2020-03-15'),'covid_time'] = 1
revenue_20.loc[revenue_20['covid_time']!=1,'covid_time'] = 0

## Define the COVID risk variable
revenue_20.loc[pd.isnull(revenue_20['q52_corona']),'q52_corona'] = 5
revenue_20.loc[revenue_20['q52_corona']==5,'q52_corona'] = 0
revenue_20.loc[(revenue_20['q52_corona'] == 4) | (revenue_20['q52_corona'] == 3),'covid_risk'] = 1
revenue_20.loc[(revenue_20['q52_corona'] <3),'covid_risk'] = 0

## Get the 1st and 2nd moments of normal variable (see BGH managerial miscalibration)

revenue_20['1st_moment_2019'] = revenue_20['revenue_p50_2019']
revenue_20['2nd_moment_2019'] = (revenue_20['revenue_p90_2019'] - revenue_20['revenue_p10_2019'])/2.65
revenue_20['1st_moment_2020'] = revenue_20['revenue_p50_2020']
revenue_20['2nd_moment_2020'] = (revenue_20['revenue_p90_2020'] - revenue_20['revenue_p10_2020'])/2.65



check = revenue_20.loc[revenue_20['keep_2019_survey']==1]

## Get the 25th and 75th percentiles (assuming normality)
reorder = ['id_specific_2020', 'survey_date','rev_actual_2019',
           'rev_forecast_2019','rev_forecast_err_2019','revenue_p10_2020', 'revenue_p50_2020',
           'revenue_p90_2020', 'revenue_2020','revenue_p10_2019',
           'revenue_p50_2019','revenue_p90_2019','revenue_2019','1st_moment_2019','2nd_moment_2019',
           '1st_moment_2020','2nd_moment_2020','q52_corona',
           'covid_time','covid_risk','keep_2020', 'keep_2019', 'keep_all',
           'keep_2019_survey']

revenue_20 = revenue_20.reindex(reorder,axis=1)


## Define the COVID time variable
revenue_20.loc[revenue_20['survey_date']>=pd.Timestamp('2020-03-15'),'covid_time'] = 1
revenue_20.loc[revenue_20['covid_time']!=1,'covid_time'] = 0

## Define the COVID risk variable
revenue_20.loc[pd.isnull(revenue_20['q52_corona']),'q52_corona'] = 5
revenue_20.loc[revenue_20['q52_corona']==5,'q52_corona'] = 0
revenue_20.loc[(revenue_20['q52_corona'] == 4) | (revenue_20['q52_corona'] == 3),'covid_risk'] = 1
revenue_20.loc[(revenue_20['q52_corona'] <3),'covid_risk'] = 0




sample = 'pre_march15'
if sample  == 'all':
    na_rev = revenue_20.loc[(revenue_20['keep_all'] == 1) & (revenue_20['keep_2019_survey'] == 1) & (revenue_20['covid_time']  <=1)]
else:
    na_rev = revenue_20.loc[(revenue_20['keep_all'] == 1) & (revenue_20['keep_2019_survey'] == 1) & (revenue_20['covid_time']  <=0)]

na_rev = na_rev.dropna(subset = ['revenue_p10_2019','revenue_p50_2019','revenue_p90_2019'])


def get_miss_low_acc_high(na_rev,out_var = 'accuracy_2019',year = 2019, upper = 90, lower = 10):
    upper_var = 'revenue_p{}_{}'.format(upper,year)
    lower_var = 'revenue_p{}_{}'.format(lower,year)
    na_rev.loc[na_rev['rev_actual_2019']<=na_rev[lower_var],out_var] = 'Low miss'
    na_rev.loc[na_rev['rev_actual_2019']>=na_rev[upper_var],out_var] = 'High miss'
    na_rev.loc[(na_rev['rev_actual_2019']<na_rev[upper_var]) & (na_rev['rev_actual_2019']>na_rev[lower_var]),out_var] = 'Accurate'
    return na_rev

def get_miss_acc_lowhigh(na_rev,out_var = 'accuracy_2019',year = 2019, upper = 90, lower = 10):
    upper_var = 'revenue_p{}_{}'.format(upper,year)
    lower_var = 'revenue_p{}_{}'.format(lower,year)
    na_rev.loc[na_rev['rev_actual_2019']<=na_rev[lower_var],out_var] = 'Low/high miss'
    na_rev.loc[na_rev['rev_actual_2019']>=na_rev[upper_var],out_var] = 'Low/high miss'
    na_rev.loc[(na_rev['rev_actual_2019']<na_rev[upper_var]) & (na_rev['rev_actual_2019']>na_rev[lower_var]),out_var] = 'Accurate'
    return na_rev
    

na_rev = get_miss_low_acc_high(na_rev)

    
def get_rev_dist_f_err(na_rev, stat = 'mean',
                       newindex = ['All firms','Low miss','Accurate','High miss']):

    rev_dist_2019 = ['revenue_p10_2019','revenue_p50_2019','revenue_p90_2019']
    rev_dist_2020 = ['revenue_p10_2020','revenue_p50_2020','revenue_p90_2020']
    rev_act_forecast = ['rev_actual_2019','rev_forecast_2019','rev_forecast_err_2019']
    rev_moments = ['1st_moment_2019','1st_moment_2020','2nd_moment_2019','2nd_moment_2020']
    
    ss_all = na_rev[rev_dist_2019 + rev_dist_2020 + rev_act_forecast + rev_moments].agg(stat).to_frame().T.rename(index = {0:'All firms'})   
    ss_all['n'] = na_rev[[rev_dist_2020[0]]].count().values
    
    ss_ferr =  na_rev.groupby('accuracy_2019')[rev_dist_2019 + rev_dist_2020 + rev_act_forecast + rev_moments].agg(stat)
    ss_ferr['n'] = na_rev.groupby('accuracy_2019')[rev_2020[0]].count()
    
    out = pd.concat([ss_all,ss_ferr],axis=0)
    
    out = out.reindex(newindex)
    return out


na_rev_mean = get_rev_dist_f_err(na_rev, stat = 'mean',
                       newindex = ['All firms','Low miss','Accurate','High miss'])

#%% Figure 10: Impact of Past Forecast Errors on Future Forecasted Revenue Distributions

def reorderLegend(ax=None,order=None,unique=False,ncols = 1,location='best',fontsize = 10):
    
    
    def unique_everseen(seq, key=None):
        seen = set()
        seen_add = seen.add
        return [x for x,k in zip(seq,key) if not (k in seen or seen_add(k))]
    if ax is None: ax=plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0])) # sort both labels and handles by labels
    if order is not None: # Sort according to a given list (not necessarily complete)
        keys=dict(zip(order,range(len(order))))
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t,keys=keys: keys.get(t[0],np.inf)))
    if unique:  labels, handles= zip(*unique_everseen(zip(labels,handles), key = labels)) # Keep only the first of each handle
    ax.legend(handles, labels,ncol = ncols,loc = location,fontsize = fontsize)
    return(handles, labels)
    
def plot_2019_effects(na_rev_mean):

    import matplotlib.transforms as transforms
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    linestyle = '-'
    ylims = (-10,30)

    def set_xmargin(ax, left=0.0, right=0.3):
        ax.set_xmargin(0)
        ax.autoscale_view()
        lim = ax.get_xlim()
        delta = np.diff(lim)
        left = lim[0] - delta*left
        right = lim[1] + delta*right
        ax.set_xlim(left,right)
        
    
    na_rev_mean['upper_2020'] = na_rev_mean['revenue_p90_2020'] - na_rev_mean['revenue_p50_2020']
    na_rev_mean['lower_2020'] = na_rev_mean['revenue_p50_2020'] - na_rev_mean['revenue_p10_2020']    
    na_rev_mean['upper_2019'] = na_rev_mean['revenue_p90_2019'] - na_rev_mean['revenue_p50_2019']
    na_rev_mean['lower_2019'] = na_rev_mean['revenue_p50_2019'] - na_rev_mean['revenue_p10_2019']    
    
    
    
    fig, ax = plt.subplots(figsize = (7,3.5))
    
    offset = lambda p: transforms.ScaledTranslation(p/72.,0, plt.gcf().dpi_scale_trans)
    trans = plt.gca().transData
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    eb1 = ax.errorbar(na_rev_mean.index,na_rev_mean['revenue_p50_2020'],
                yerr = [na_rev_mean['lower_2020'],na_rev_mean['upper_2020']],color=colors[1],fmt = 'o',
                markersize = 6 ,capsize = 3,transform=trans+offset(+10),label='_nolegend_')
    eb2 = ax.errorbar(na_rev_mean.index,na_rev_mean['revenue_p50_2019'],
                yerr = [na_rev_mean['lower_2019'],na_rev_mean['upper_2019']],color=colors[0],fmt = 'o',markersize = 6 ,
                capsize = 3,transform=trans+offset(-10),label="_nolegend_")
    y_l = na_rev_mean.loc[na_rev_mean.index == 'All firms']['revenue_p10_2019'].values
    y_b = na_rev_mean.loc[na_rev_mean.index == 'All firms']['revenue_p50_2019'].values
    y_u = na_rev_mean.loc[na_rev_mean.index == 'All firms']['revenue_p90_2019'].values
    arrow_props = dict(fc='k', ec='k',
                       alpha=1,arrowstyle="->")
    trans = ax.get_yaxis_transform() # x in data untis, y in axes fraction
    ax.annotate("2019 10th\npercentile",xytext = (-.13,float(y_l)-2),xy = (0.115,float(y_l)),
                xycoords=trans,arrowprops = arrow_props,zorder = -5,fontsize = 9,ha='center')
    ax.annotate("2019 90th\npercentile",xytext = (-.13,float(y_u)-2),xy = (0.115,float(y_u)),
                xycoords=trans,arrowprops = arrow_props,zorder = -5,fontsize = 9,ha='center')
    ax.annotate("2019 best\nguess",xytext = (-.13,float(y_b)-2),xy = (0.115,float(y_b)),
                xycoords=trans,arrowprops = arrow_props,zorder = -5,fontsize = 9,ha='center')
  
    
    for k, spine in ax.spines.items():  #ax.spines is a dictionary
        spine.set_zorder(10)
    ax.margins(x=0.2)
    eb1[1][0].set_marker("v")
    eb1[1][1].set_marker("^")
    eb2[1][0].set_marker("v")
    eb2[1][1].set_marker("^")
    eb2[-1][0].set_linestyle(linestyle)
    eb1[-1][0].set_linestyle(linestyle)
    eb2[-1][0].set_label("2019")
    eb1[-1][0].set_label("2020")
    plt.rcParams["axes.axisbelow"] = False
    
    ax.set_axisbelow(True)
    ax.grid(axis='y',alpha=0.5,linestyle='--',zorder=-100)
    yticks = ax.get_yticks()
    ax.set_ylim((min(yticks),max(yticks)+5))
    if ylims is not None:
        ax.set_ylim(ylims)
    yticks = ax.get_yticks()
    if 0 in yticks:
        ind = np.where(yticks==0)[0][0]
        gridlines = ax.yaxis.get_gridlines()
        gridlines[ind].set_color("black")
        gridlines[ind].set_linewidth(1)
        gridlines[ind].set_zorder(0)
        gridlines[ind].set_alpha(1)
    for tick in ax.get_xticklabels():
        tick.set_rotation(0)
    ax.legend(ncol = 3,loc = 'upper left',fontsize = 8)
    reorderLegend(ax,['2019','2020'],ncols=3,location = 'upper left',fontsize = 11)   
    plt.xticks(fontsize = 11)
    plt.yticks(fontsize = 11)
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y/100))) 
    
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(['All Firms','Realization\nBelow\n10th Percentile\nForecast','Realization $\in$\n[10th, 90th]',
                      'Realization\nAbove\n90th Percentile\nForecast'])
    fig.set_size_inches(7.5,3.5)

    plt.tight_layout()
    plt.show()
    return fig

print("\n\n\n")
print("Displaying Figure 10:")
fig = plot_2019_effects(na_rev_mean)
plt.show()

#%% Figure 11: Impact of COVID-19 Shock on Forecasted 2020 Revenue Distributions

def get_rev_dist_covid(na_rev,group_var, stat = 'mean',
                       newindex = ['All firms','< March 15','≥ March 15', 'No/low COVID risk','Med/high COVID risk']):

    rev_dist_2020 = ['revenue_p10_2020','revenue_p50_2020','revenue_p90_2020']
    rev_moments = ['1st_moment_2020','2nd_moment_2020']
    
    ss_all = na_rev[rev_dist_2020 + rev_moments].agg(stat).to_frame().T.rename(index = {0:'All firms'})   
    ss_all['n'] = na_rev[[rev_dist_2020[0]]].count().values
    
    ss_ferr =  na_rev.groupby(group_var)[rev_dist_2020 + rev_moments].agg(stat)
    ss_ferr['n'] = na_rev.groupby(group_var)[rev_2020[0]].count()
    
    out = pd.concat([ss_all,ss_ferr],axis=0)
    
    if newindex is not None:
        out = out.reindex(newindex)
    return out



stat = 'mean'
na_rev = revenue_20.loc[(revenue_20['keep_all'] == 1) & (revenue_20['keep_2019_survey'] == 1) & (revenue_20['covid_time']  <=1)]
na_rev['covid risk'] = na_rev['covid_risk'].map({0:'Low risk',1:'High risk'})
na_rev['covid time'] = na_rev['covid_time'].map({0:'< March 15',1:'≥ March 15'})

newindex = None
group_var = ['covid risk','covid time']
na_rev_mean_riskXtime = get_rev_dist_covid(na_rev,group_var,stat=stat,newindex = newindex)
group_var = 'covid risk'
newindex = na_rev[group_var].drop_duplicates().tolist()
na_rev_mean_risk = get_rev_dist_covid(na_rev,group_var,stat=stat,newindex = newindex)
group_var = 'covid time'
newindex = na_rev[group_var].drop_duplicates().tolist()
na_rev_mean_time = get_rev_dist_covid(na_rev,group_var,stat=stat,newindex = newindex)

na_rev_mean = pd.concat([na_rev_mean_risk, na_rev_mean_time,na_rev_mean_riskXtime],axis = 0)
na_rev_mean = na_rev_mean.reindex(['All firms']+[x for x in na_rev_mean.index if x not in ['All firms']])

na_rev_mean_risk_time = na_rev_mean.reindex(['All firms','< March 15','≥ March 15','Low risk','High risk'])


hold = na_rev_mean_riskXtime.reindex([x for x in na_rev_mean_riskXtime.index if x not in ['All firms']])
hold['covid risk'] = hold.index
hold['covid risk'] = [x[0] for x in hold['covid risk'].tolist()]
hold['covid time']= [x[1] for x in hold.index]

riskXtime = hold.pivot(index='covid time', columns='covid risk')
riskXtime.columns = [(x[0].replace("_2020","") +"_"+ x[1].partition(' ')[0]).lower() for x in riskXtime.columns]

all_firms = na_rev_mean.loc[na_rev_mean.index == 'All firms']

    
riskXtime['upper_high'] = riskXtime['revenue_p90_high'] - riskXtime['revenue_p50_high'] 
riskXtime['lower_high'] = riskXtime['revenue_p50_high'] - riskXtime['revenue_p10_high'] 
riskXtime['upper_low'] = riskXtime['revenue_p90_low'] - riskXtime['revenue_p50_low'] 
riskXtime['lower_low'] = riskXtime['revenue_p50_low'] - riskXtime['revenue_p10_low'] 

all_firms['upper'] = all_firms['revenue_p90_2020'] - all_firms['revenue_p50_2020'] 
all_firms['lower'] = all_firms['revenue_p50_2020'] - all_firms['revenue_p10_2020']   


def plot_covid_effects(riskXtime,all_firms):
    
    import matplotlib.transforms as transforms
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    linestyle = '-'
    color1 = 0
    color2 = 1
    fig, ax = plt.subplots(figsize = (6.5,3.5))
    offset = lambda p: transforms.ScaledTranslation(p/72.,0, plt.gcf().dpi_scale_trans)
    trans = plt.gca().transData
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    eb0 = ax.errorbar(all_firms.index,all_firms['revenue_p50_2020'], 
                yerr = [all_firms['lower'],all_firms['upper']],color='gray',fmt = 'o',
                markersize = 6 ,capsize = 3,transform=trans+offset(0),label='_nolegend_')
    eb1 = ax.errorbar(riskXtime.index,riskXtime['revenue_p50_low'],
                yerr = [riskXtime['lower_low'],riskXtime['upper_low']],color=colors[color1],fmt = 'o',
                markersize = 6 ,capsize = 3,transform=trans+offset(-10),label='_nolegend_')
    eb2 = ax.errorbar(riskXtime.index,riskXtime['revenue_p50_high'],
                yerr = [riskXtime['lower_high'],riskXtime['upper_high']],color=colors[color2],fmt = 'o',
                markersize = 6 ,capsize = 3,transform=trans+offset(10),label='_nolegend_')
    hi_risk = float(riskXtime.loc[riskXtime.index =='≥ March 15']['revenue_p10_high'])
    low_risk = float(riskXtime.loc[riskXtime.index =='≥ March 15']['revenue_p90_low'])
    ax.annotate("Low financial risk\ndue to COVID",xy = (1.94,float(low_risk)-7),zorder = -5,fontsize = 11)
    ax.annotate("High financial risk\ndue to COVID",xy = (2.12,float(hi_risk)+3),zorder = -5,fontsize = 11)
    
    ax.margins(x=0.2)
    
    eb0[1][0].set_marker("v")
    eb0[1][1].set_marker("^")
    eb1[1][0].set_marker("v")
    eb1[1][1].set_marker("^")
    eb2[1][0].set_marker("v")
    eb2[1][1].set_marker("^")
    eb2[-1][0].set_linestyle(linestyle)
    eb1[-1][0].set_linestyle(linestyle)
    eb1[-1][0].set_label("Low risk")
    eb2[-1][0].set_label("High risk")
    
    ax.set_axisbelow(True)
    ax.grid(axis='y',alpha=0.5,linestyle='--',zorder=-10)
    yticks = ax.get_yticks()
    ax.set_ylim((min(yticks),max(yticks)))
    
    yticks = ax.get_yticks()
    if 0 in yticks:
        ind = np.where(yticks==0)[0][0]
        gridlines = ax.yaxis.get_gridlines()
        gridlines[ind].set_color("black")
        gridlines[ind].set_linewidth(1)
        gridlines[ind].set_zorder(0)
        gridlines[ind].set_alpha(1)
    for tick in ax.get_xticklabels():
        tick.set_rotation(0)
    ax.legend(ncol = 3,loc = 'upper left',fontsize = 10)
    ax.set_xlim((-0.25,2.5))
    plt.xticks(fontsize = 11)
    plt.yticks(fontsize = 11)
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y/100))) 
    
    plt.tight_layout()
    plt.show()
    return fig

print('\n\n\n')
print("Displaying Figure 11:")
fig = plot_covid_effects(riskXtime,all_firms)
plt.show()
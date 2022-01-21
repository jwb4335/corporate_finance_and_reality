"""
Code to produce Figure 3, Table II, Figure 4, Figure A4.1
John Barry
2022/01/14
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

## Load in the survey data
filename = 'Data/datafile.pkl'
with open(filename, 'rb') as f:
    demo_vars_19,demo_vars_20,demo_vars_20_only,demo_var_list,sdata,sdata_20,sdata_2001 = pickle.load(f)
    f.close()


filename = 'Data/hurdle_wacc_data.pkl'
# with open(filename, 'wb') as f:
#     pickle.dump([hw_data,bond_yields],f)
#     f.close()
with open(filename, 'rb') as f:
    hw_data,bond_yields = pickle.load(f)
    f.close()

filename = 'Data/hurdle_changes_intl.pkl'
# with open(filename, 'wb') as f:
#     pickle.dump([hw_data,bond_yields],f)
#     f.close()
with open(filename, 'rb') as f:
    [hurdle_changes_intl] = pickle.load(f)
    f.close()
    
    


#%% FIGURE 3: HURDLE RATES AND THE COST OF CAPITAL

## This insanely long function creates Figure 3
def yield_hurdle_rates_graph(hw_data,bond_yields,include_medians = "No",save_to_excel = "Yes",fig_size = (8,4)):
    
    from matplotlib.ticker import FuncFormatter
    pd.options.mode.chained_assignment = None 
    ## For visual pursposes, combine the two surveys from 2017 into one point on plot
    hw_data.loc[hw_data['year'] == 2017,'qtr'] = 3
    
    ## Get mean rates need for plot
    hw_data['syear'] = hw_data['year']
    hw_data['sqtr'] = hw_data['qtr']
    
    hw_data_graph = hw_data[['hurdle','wacc','syear','sqtr']].groupby(['syear','sqtr']).mean().reset_index()
    hw_data_graph = hw_data_graph.rename(columns = {'hurdle':'hurdle_mean',
                                                    'wacc':'wacc_mean'})


    ## These are the surveys which Duke CFO survey asked about hurdle rates
    survey_hurdle = [[2007,1],[2011,1],[2012,2],[2017,3],[2019,2]]

    ## Historical surveys from Summers, etc... 
    hurdle_series = [["1985","01",17,15,"Summers"],
                   ["1990","01",17.5,16,"Poterba & Summers"],
                   ["2003","11",15.08,15,"Meier & Tarhan"],
                   ["2007","01",0,0,"Duke/CFO $\u27f6$"],
                   ["2011","01",0,0,"Duke/CFO"],
                   ["2012","04",0,0,"Duke/CFO"],
                   ["2017","07",0,0,"Duke/CFO"],
                   ["2019","04",0,0,"Duke/CFO"]]
    
    
    ## Fill in hurdle_series with Duke data
    hurdle_series = pd.DataFrame(hurdle_series,columns =['year','month','hurdle_mean','hurdle_median','label'])
    hurdle_series['qtr'] = np.floor(hurdle_series['month'].astype(float)/3) + 1
    hurdle_series['year'] = hurdle_series['year'].astype(float)
    for s in survey_hurdle:
        f = hurdle_series.loc[(hurdle_series['year'] == s[0]) & (hurdle_series['qtr'] == s[1])]['label'].tolist()[0]
        if 'Duke' in f:
            svy_mean = hw_data.loc[(hw_data['syear'] == s[0])&(hw_data['sqtr'] == s[1])]['hurdle'].mean()
            svy_median = hw_data.loc[(hw_data['syear'] == s[0])&(hw_data['sqtr'] == s[1])]['hurdle'].median()
            hurdle_series.loc[(hurdle_series['year'] == s[0]) & (hurdle_series['qtr'] == s[1]),'hurdle_mean'] = svy_mean
            hurdle_series.loc[(hurdle_series['year'] == s[0]) & (hurdle_series['qtr'] == s[1]),'hurdle_median'] = svy_median
        
    hurdle_series['year'] = hurdle_series['year'].astype(int).astype(str)
    hurdle_series = hurdle_series.rename(columns = {'label':'survey'})

    ## Now, do the same for WACC
    survey_wacc = [[2009,3],[2011,1],[2012,2],[2017,3],[2019,2]]
    wacc_series = [["2003","11",9.85,9.26,"Meier & Tarhan"],
                   ["2009","07",0,0,"Duke/CFO $\u27f6$"],
                   ["2011","01",0,0,"Duke/CFO"],
                   ["2012","04",0,0,"Duke/CFO"],
                   ["2017","07",0,0,"Duke/CFO"],
                   ["2019","04",0,0,"Duke/CFO"]]

    wacc_series = pd.DataFrame(wacc_series,columns =['year','month','wacc_mean','wacc_median','label'])
    wacc_series['qtr'] = np.floor(wacc_series['month'].astype(float)/3) + 1
    wacc_series['year'] = wacc_series['year'].astype(float)
    for s in survey_wacc:
        f = wacc_series.loc[(wacc_series['year'] == s[0]) & (wacc_series['qtr'] == s[1])]['label'].tolist()[0]
        if 'Duke' in f:
            svy_mean = hw_data.loc[(hw_data['syear'] == s[0])&(hw_data['sqtr'] == s[1])]['wacc'].mean()
            svy_median = hw_data.loc[(hw_data['syear'] == s[0])&(hw_data['sqtr'] == s[1])]['wacc'].median()
            wacc_series.loc[(wacc_series['year'] == s[0]) & (wacc_series['qtr'] == s[1]),'wacc_mean'] = svy_mean
            wacc_series.loc[(wacc_series['year'] == s[0]) & (wacc_series['qtr'] == s[1]),'wacc_median'] = svy_median

    wacc_series['year'] = wacc_series['year'].astype(int).astype(str)
    wacc_series = wacc_series.rename(columns = {'label':'survey_wacc'})

    hurdle_series = hurdle_series.merge(wacc_series,on = ['year','month','qtr'],how = 'outer')
    graph = bond_yields.merge(hurdle_series,on = ["year","month"],suffixes=[False, False],how='left')

    graph["xlab"] = graph["year"]
    graph.loc[graph["month"] !="01","xlab"] = " "
    
    
    ## Labels on plot
    labels = graph.loc[np.isnan(graph["hurdle_mean"]) ==False]
    labels["xval"] = labels.index
    labels["yval"] = labels[["hurdle_mean","hurdle_median"]].max(axis=1)
    labels["yval"] = labels["yval"]+.9
    labels.loc[labels["survey"]=="Summers","xval"] = 2

    labels.loc[labels["survey"]=="Poterba & Summers","yval"] = labels["yval"]+0.5
    labels.loc[labels["survey"]=="Poterba & Summers","xval"] = labels["xval"]+5
    labels.loc[labels["survey"]=="Summers","xval"] = labels["xval"]+5

    labels.loc[labels.index == 226,'yval'] = labels['yval']-2.75

    labels.loc[labels["xval"]==264,"yval"] = labels["yval"]

    labels.loc[labels.index>311,"survey"]=""

    labels = labels[["survey","xval","yval"]]
    labels.iloc[[-1],0] = ""
    slabels = labels["survey"].tolist()
    xvals = labels["xval"].values
    yvals = labels["yval"].values
    
    labels2 = graph.loc[np.isnan(graph["wacc_mean"]) ==False]
    labels2["xval"] = labels2.index
    labels2["yval"] = labels2[["wacc_mean","wacc_median"]].max(axis=1)
    labels2["yval"] = labels2["yval"]+.9
    labels2.loc[labels2.index == 226,'yval'] = labels2['yval']-2.25
    labels2.loc[labels2["xval"]==290,"yval"] = labels2["yval"]
    
    labels2.loc[labels2.index>300,"survey_wacc"]=""
    labels2 = labels2[["survey_wacc","xval","yval"]]
    labels2.iloc[[-1],0] = ""
    slabels2 = labels2["survey_wacc"].tolist()
    xvals2 = labels2["xval"].values
    yvals2 = labels2["yval"].values
    
    
    ## Draws an annotated brace on the axes
    def draw_brace(ax, xspan, yy, text):
        xmin, xmax = xspan
        xspan = xmax - xmin
        ax_xmin, ax_xmax = ax.get_ylim()
        xax_span = ax_xmax - ax_xmin
    
        ymin, ymax = (0,ax.get_xlim()[1])
        yspan = ymax - ymin
        resolution = int(xspan/xax_span*100)*2+1 # guaranteed uneven
        beta = 600./xax_span # the higher this is, the smaller the radius
    
        x = np.linspace(xmin, xmax, resolution)
        x_half = x[:int(resolution/2)+1]
        y_half_brace = (1/(1.+np.exp(-beta*(x_half-x_half[0])))
                        + 1/(1.+np.exp(-beta*(x_half-x_half[-1]))))
        y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
        y = yy + (.03*y - .01)*yspan # adjust vertical position
    
        ax.autoscale(False)
        ax.plot(y, x, color='black', lw=1)
    
        ax.text((xmax+xmin)/2., yy+ 0.07*yspan, text, ha='center', va='bottom')
    
    ## The plot (finally) ##
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    def splitSerToArr(ser):
        return [ser.index,ser.to_numpy()]
    
    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(graph.index,graph['tbill_10'],label="10-yr Treasury Yield",color="black")
    ax.plot(graph.index,graph['bbb_10'],linestyle = "--",label="10-yr BBB Yield",color="gray")
    ax.plot(*splitSerToArr(graph["hurdle_mean"].dropna()), linestyle = '-',marker='s',markersize = 7,label="Hurdle",color = colors[0])
    ax.plot(*splitSerToArr(graph["wacc_mean"].dropna()), linestyle = '-',marker='D',markersize = 7,color = colors[1],label = "WACC")
    fig.legend(loc='center',bbox_to_anchor = (0.5,-0.05),fontsize = 11,ncol=4)

    plt.xticks(np.arange(0, 413, 24),np.arange(1985,2021,2),rotation="vertical")
    plt.ylim((0,20))
    plt.yticks(np.arange(0,22,2))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(axis='y',color='grey',alpha=0.5,linestyle='--')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y/100))) 

    ax.set_axisbelow(True)

    for i in range(len(labels)):
        if labels.iloc[i,0] != 'Duke/CFO $⟶$':
            text_object = ax.annotate(slabels[i], xy=(xvals[i], yvals[i]), ha='center',size=10)
            text_object.set_fontsize(10)
        elif labels.iloc[i,0] == 'Duke/CFO $⟶$':
            text_objecta = ax.annotate(slabels[i], xy=(xvals[i]+14, yvals[i]-0.4), ha='center',size=10)
            text_objecta.set_fontsize(10)    
    for i in range(len(labels2)):
        if labels2.iloc[i,0] != 'Duke/CFO $⟶$':
            text_object2 = ax.annotate(slabels2[i], xy=(xvals2[i], yvals2[i]), ha='center',size=10)
            text_object2.set_fontsize(10)
        elif labels2.iloc[i,0] == 'Duke/CFO $⟶$':
            text_object2a = ax.annotate(slabels2[i], xy=(xvals2[i]+14, yvals2[i]), ha='center',size=10)
            text_object2a.set_fontsize(10)

        
    yloc = (hurdle_series.loc[hurdle_series['year'] == '2019']['wacc_mean'],
            hurdle_series.loc[hurdle_series['year'] == '2019']['hurdle_mean'])
    draw_brace(ax, yloc,416, '')
    ax.annotate("6%",ha='center',xy=(420, 12), xytext=(442, 11.75), arrowprops=dict(arrowstyle="-",lw=0))

    plt.tight_layout()
    return fig

print("\n\n\n")
print("Displaying Figure 3")
fig = yield_hurdle_rates_graph(hw_data,bond_yields,fig_size = (6.5,3.5),save_to_excel = "Yes")
plt.show()

#%% TABLE II: HURDLE RATES AND THE COST OF CAPITAL

## Get the 2019 data for Table II
hw_data_2019 = hw_data.loc[hw_data['year']==2019]
hw_data_2019['buffer'] = hw_data_2019['hurdle'] - hw_data_2019['wacc']


## Merge on demographics
hw_data_2019 = hw_data_2019.merge(sdata[['id_specific'] + demo_vars_19],left_on = 'id_specific_2019q2',right_on = 'id_specific')
    
    
    
def hurdle_demo_splits(data,demo_var):

    import statsmodels.formula.api as sm
    
    data = data.loc[~pd.isnull(data[demo_var])]
    

    out = data.groupby(demo_var)[['hurdle','wacc','buffer']].mean().T.\
          rename(index = {'hurdle':'Hurdle','wacc':'WACC','buffer':'Buffer'})
    sort = [x for x in out.columns if '*' not in x] + [x for x in out.columns if '*' in x]
    out = out[sort]
    # out = round(out,2)
    # out = out.astype(str)
    for col in out.columns:
        out[col] = out[col].map('{:.2f}'.format)
    n_titles = data.groupby(demo_var)[demo_var].count().to_frame().T[sort].values[0].tolist()

    for var in ['hurdle','wacc','buffer']:
        var2 = {'hurdle':'Hurdle','wacc':'WACC','buffer':'Buffer'}[var]
        reg_df = data[[var,demo_var]]
        reg_df.loc[reg_df[demo_var]==sort[1],'dummy'] = 1
        reg_df.loc[reg_df[demo_var]==sort[0],'dummy'] = 0
        result = sm.ols(formula="{} ~ dummy".format(var), data=reg_df).fit()
        
        means = out.loc[out.index == var2].squeeze().replace({'*',''}).tolist()
        if (result.pvalues.dummy<=0.1) & (result.pvalues.dummy>0.05):
            means[1] = means[1]+'*'
        elif (result.pvalues.dummy<=0.05 ) & (result.pvalues.dummy>0.01 ):
            means[1] = means[1]+'**'
        elif (result.pvalues.dummy<=0.01 ):
            means[1] = means[1]+'***'
        out.loc[out.index == var2,sort[0]] = means[0]
        out.loc[out.index == var2,sort[1]] = means[1]
    
            

    l1 = [demo_var]*2
    l2 = [x.replace('*','') for x in sort]
    l3 = n_titles
    tuples= [(l1[i],l2[i],l3[i]) for i in range(len(l1))]
        
    out.columns = pd.MultiIndex.from_tuples(tuples,names = ['Variable','Group','N'])
    
    return out



demo_vars_out = ['Size','Public','Growth Prospects','Pay Dividends','Leverage','Cash','Financial Flexibility']

for demo_var in demo_vars_out:
    if demo_var == demo_vars_out[0]:
        hurdle_demos_out =  hurdle_demo_splits(hw_data_2019,demo_var)
    else:
        hold =  hurdle_demo_splits(hw_data_2019,demo_var)
        hurdle_demos_out = pd.concat([hurdle_demos_out,hold],axis=1)
        
hurdle_demos_out.index = ['Hurdle Rate','WACC','Buffer (Hurdle Rate - WACC)']

        
hurdle_demos_all = hw_data_2019[['hurdle','wacc','buffer']].mean().to_frame()

hurdle_demos_all.columns = pd.MultiIndex.from_tuples([('All Firms','',len(hw_data_2019))],
                                                      names = ['Demographic Variable','Group','N'])

hurdle_demos_all.index = ['Hurdle Rate','WACC','Buffer (Hurdle Rate - WACC)']

for col in hurdle_demos_all.columns:
    hurdle_demos_all[col] = hurdle_demos_all[col].map('{:.2f}'.format)
hurdle_demos_out = pd.concat([hurdle_demos_all,hurdle_demos_out],axis=1)

print("\n\n\n")
## Display table
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print("Displaying Table II")
    print(hurdle_demos_out)
    
    
#%% FIGURE 4: REASONS COMPANIES CHANGE HURDLE RATES
whc = sdata[[col for col in sdata if col.startswith(('why_hurdlec'))]]

whc = whc[[x for x in whc if x not in ['why_hurdlechange_other','why_hurdlechange5']]]

whc = whc[whc.notnull().any(1)]

whc = ((whc.notnull()).astype(int))

whc = whc.sum().to_frame()

whc = whc/whc.sum()


index_dict = {'why_hurdlechange1':'Borrowing Costs',
               'why_hurdlechange2':'Cost of Equity or Beta',
               'why_hurdlechange3':'Market Risk Premium',
               'why_hurdlechange4':'Type/Location of Investment'}

whc = whc.rename(index=index_dict)

whc.columns = ["Percent"]

whc = whc.sort_values(by = ['Percent'],ascending = False)

print("\n\n\n")
print("Displaying data for Figure 4")
# whc.plot(kind = 'bar')
# plt.show()
print(whc)


#%% FIGURE A4.1: How Frequently Do Companies Change Hurdle Rates?


def changed_hurdle_rate(sdata_raw):

    hc_cols = [col for col in sdata_raw if col.startswith(('hurdle_change'))]
    hc = sdata_raw[hc_cols + ['location']]
    hc = hc.loc[(hc["hurdle_change"]<7) & (np.isnan(hc["hurdle_change"])==False)]
    
    index_dict = {0:'0',
                  1:'1',
                  2:'2',
                  3:'3',
                  4:'4',
                  5:'5',
                  6:'6+'}
    
    hc_out = hc.groupby(['location'])["hurdle_change"].value_counts(normalize=True).\
             to_frame().rename(columns = {"hurdle_change":"Percent"}).reset_index().\
             pivot(index = ["hurdle_change"], columns = ['location'],values = "Percent")    
             
    sort = ['US/Canada','Europe','Asia','Latin America']
    hc_out = hc_out[[x for x in sort if x in hc_out.columns.tolist()]]
    hc_out = hc_out.rename(index=index_dict)
        
    
    return hc_out

hc = changed_hurdle_rate(hurdle_changes_intl)

print("\n\n\n")
print("Displaying data for Figure A4.1:")
print(hc)
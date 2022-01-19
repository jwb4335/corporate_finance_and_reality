
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

#%% FIGURE 5: How Do Companies Estimate the Cost of Equity?

## calculate the percent importand and very important by survey
def cost_equity_methods_pct(sdata_20,sdata_2001):
    
    cost_equity_2001 = {'Q3':'estimate_re',
                     'Q3A':'Historical Average',
                     'Q3B':'CAPM',
                     'Q3C':'Multi-Factor Model',
                     'Q3D':'Investor Expectations',
                     'Q3E':'Regulatory Decisions',
                     'Q3F':'Dividend Discount Model',
                     'Q3G':'Other'}
    
    cost_equity_2020 = {'q31_wave2':'estimate_re',
                         'a':'Historical Average',
                         'b':'CAPM',
                         'c':'Multi-Factor Model',
                         'd':'Investor Expectations',
                         'e':'Unchanged Estimate (\'22 only)',
                         'f':'Dividend Discount Model',
                         'g':'Regulatory Decisions',
                         'h':'Market Return (\'22 only)',
                         'i':'Other'}

    re_cols=list(cost_equity_2001.values())
    
    sdata_2001 = sdata_2001.rename(columns = cost_equity_2001)
    
    sdata_2001 = sdata_2001[re_cols]
    
    sdata_2001[sdata_2001==9]=np.nan
    
    cost_equity_cols = [x for x in cost_equity_2001.values() if x not in ['estimate_re']]
    
    
    for col in cost_equity_cols:
        sdata_2001.loc[(sdata_2001['estimate_re'] == 2) | (pd.isnull(sdata_2001['estimate_re'])),col] = np.nan
    
    sdata_2001 = sdata_2001.dropna(subset = cost_equity_cols,how = 'all')

    
    sdata_20 = sdata_20[[x for x in sdata_20.columns if x.startswith('q31')]]
    sdata_20 = sdata_20[[x for x in sdata_20.columns if x not in ['q31_other_wave2']]]
    sdata_20.columns = ['q31_wave2'] + [x[5] for x in sdata_20.columns if x.startswith('q31_2')]
    sdata_20 = sdata_20.rename(columns = cost_equity_2020)
    
    cost_equity_cols_20 = [x for x in cost_equity_2020.values() if x not in ['estimate_re']]
    for col in cost_equity_cols_20:
        sdata_20.loc[(sdata_20['estimate_re'] == 2) | (pd.isnull(sdata_20['estimate_re'])),col] = np.nan

    
    sdata_2001 = sdata_2001[[x for x in sdata_2001.columns if x not in ['estimate_re']]]
    sdata_20 = sdata_20[[x for x in sdata_20.columns if x not in ['estimate_re']]]    


    sdata_20 = sdata_20.dropna(how = 'all')
    
    sdata_2001 = sdata_2001.dropna(how = 'all')
    
    for col in sdata_2001.columns:
        sdata_2001.loc[sdata_2001[col] >=3,"{}_dum".format(col)] = 1
        sdata_2001.loc[sdata_2001[col] < 3,"{}_dum".format(col)] = 0
        # sdata_2001.loc[pd.isnull(sdata_2001[col]),"{}_dum".format(col)] = 0
        
    for col in sdata_20.columns:
        sdata_20.loc[sdata_20[col] >=3,"{}_dum".format(col)] = 1
        sdata_20.loc[sdata_20[col] < 3,"{}_dum".format(col)] = 0
        # sdata_20.loc[pd.isnull(sdata_20[col]),"{}_dum".format(col)] = 0

    
    cost_equity_2001 = sdata_2001[[x for x in sdata_2001 if '_dum' in x]].mean().to_frame()
    cost_equity_2001 = cost_equity_2001.rename( 
                       index = {x:x.replace("_dum","") for x in cost_equity_2001.index})
    cost_equity_2001 = cost_equity_2001.loc[cost_equity_2001.index!='Other']
    
    cost_equity_2020 = sdata_20[[x for x in sdata_20 if '_dum' in x]].mean().to_frame()
    cost_equity_2020 = cost_equity_2020.rename( 
                       index = {x:x.replace("_dum","") for x in cost_equity_2020.index})
    cost_equity_2020 = cost_equity_2020.loc[cost_equity_2020.index!='Other']
    
    cost_equity = cost_equity_2001.join(cost_equity_2020,lsuffix = '_2001',rsuffix='_2020',how='outer')    
    cost_equity.columns = ['2001','2022']
    cost_equity = cost_equity.sort_values(by = '2022')
    
    var_sort = cost_equity.sort_values(by = '2022',ascending = True).index.tolist()
    # var_sort.reverse()
    cost_equity = cost_equity.reindex(var_sort)

    
    return cost_equity

## Prepare data for figure
cost_equity_small = cost_equity_methods_pct(sdata_20.loc[sdata_20['large'] == 0],sdata_2001.loc[sdata_2001['large'] == 0]) 
cost_equity_large = cost_equity_methods_pct(sdata_20.loc[sdata_20['large'] == 1],sdata_2001.loc[sdata_2001['large'] == 1]) 

cost_equity_2001 = pd.concat([cost_equity_large['2001'],cost_equity_small['2001']],axis = 1)
cost_equity_2001.columns = ['Large','Small']

cost_equity_2020 = pd.concat([cost_equity_large['2022'],cost_equity_small['2022']],axis = 1)
cost_equity_2020.columns = ['Large','Small']

cost_equity = cost_equity_methods_pct(sdata_20,sdata_2001).rename(columns = {'2001':'2001'})

large = cost_equity_large.rename(columns = {'2001':'2001 Large','2022':'2022 Large'})
small = cost_equity_small.rename(columns = {'2001':'2001 Small','2022':'2022 Small'})
data = large.to_dict('list')
second_dict = small.reindex(large.index.tolist()).to_dict('list')

large.loc[pd.isnull(large['2001 Large']),'2001 Large'] = 0
small.loc[pd.isnull(small['2001 Small']),'2001 Small'] = 0

total_width = 0.8
single_width = 0.9

alpha = 1
ncols = 4
xlabels = large.index.tolist()

def crosshatch_large_small(data,second_dict,xlabels,fig_size = (9.5,3.5),
                           total_width=0.8, single_width=1, legend=True, 
                           alph = 1,legend_loc = 'best',ncols = 2):
    
    
    from PIL import ImageColor

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors = [colors[1], colors[0]]

    fig,ax = plt.subplots(figsize = fig_size)

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
            bar = ax.barh(x + x_offset, y, height=bar_width * single_width, color=colors[i],alpha=alph)
    
        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])
    if second_dict is not None:
        for i, (name, values) in enumerate(second_dict.items()):
            for x, y in enumerate(values):
                col = tuple([x/255 for x in  ImageColor.getcolor(colors[i], "RGB")])+ (0.5,)
                x_offset = x_offset = (i - n_bars / 2) * bar_width + bar_width / 2
                bar = ax.barh(x + x_offset, y, height=bar_width * single_width, 
                       facecolor = col, edgecolor=(1,1,1,1), hatch='///')
            bars.append(bar[0])
    ax.set_yticks(np.arange(len(xlabels)))
    ax.set_yticklabels(xlabels,fontsize = 11)
    ax.set_axisbelow(True)
    ax.grid(axis='x',alpha=0.5,linestyle='--',zorder=-100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    from matplotlib.ticker import FuncFormatter
    ax.set_xticks(np.arange(0,1,0.2))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
    
    fig = plt.gcf()
    # Draw legend if we need
    fig.legend([bars[1],bars[3]], ['Large','Small'],
               ncol = 2,loc = 'center',bbox_to_anchor = (0.35,-0.025),fontsize = 11,
               title = r'$\bf{2022}$',title_fontsize = 11)

    fig.legend([bars[0],bars[2]], ['Large','Small'],
               ncol = 2,loc = 'center',bbox_to_anchor = (0.7,-0.025),fontsize = 11,
               title = r'$\bf{2001}$',title_fontsize = 11)

    return fig,ax

print("\n\n\n")
print("Displaying Figure 5:")
fig,ax = crosshatch_large_small(data,second_dict,xlabels,fig_size = (6.5,4.5),
                                total_width = 0.8,single_width = 0.9,
                                legend=True,alph=1,ncols = 4)
plt.show()

#%% TABLE III: How Do Companies Estimate the Cost of Equity?

## Standardize responses across surveys
def cost_equity_fix(sdata_20,sdata_2001):
    
    cost_equity_2001 = {'Q3':'estimate_re',
                 'Q3A':'Historical Average',
                 'Q3B':'CAPM',
                 'Q3C':'Multi-Factor Model',
                 'Q3D':'Investor Expectations',
                 'Q3E':'Regulatory Decisions',
                 'Q3F':'Dividend Discount Model',
                 'Q3G':'Other'}

    cost_equity_2020 = {'q31_wave2':'estimate_re',
                 'a':'Historical Average',
                 'b':'CAPM',
                 'c':'Multi-Factor Model',
                 'd':'Investor Expectations',
                 'e':'Unchanged Estimate (\'22 only)',
                 'f':'Dividend Discount Model',
                 'g':'Regulatory Decisions',
                 'h':'Market Return (\'22 only)',
                 'i':'Other'}
    
    cost_equity_2020_rename = {"q31_2{}_wave2".format(x):cost_equity_2020[x] for x in list(cost_equity_2020.keys()) if 'estimate_re' not in cost_equity_2020[x]}
    cost_equity_2020_rename.update({'q31_wave2':'estimate_re'})
    
    ## Get the 2001 answers
    cost_equity_cols = [x for x in list(cost_equity_2001.values())  if x not in ['estimate_re','Other']]   
    ce_01 = sdata_2001.rename(columns = cost_equity_2001)    
    ce_01[ce_01==9]=np.nan

    ce_01 = ce_01[cost_equity_cols +['estimate_re']+ ['large']]
    for col in cost_equity_cols:
        ce_01.loc[(ce_01['estimate_re'] == 2) | (pd.isnull(ce_01['estimate_re'])),col] = np.nan
        
    ce_01 = ce_01.dropna(how = 'all',subset = cost_equity_cols)
        
    ## classify the survey answers    
    for var in cost_equity_cols:
        ce_01.loc[ce_01[var] >= 3,'{}_dum'.format(var)] = 1
        ce_01.loc[ce_01[var] < 3, '{}_dum'.format(var)] = 0
                       

    
    ## Get the 2020 answers    
    cost_equity_cols = [x for x in list(cost_equity_2020.values())  if x not in ['estimate_re','Other']]   
    
    ce_20 = sdata_20.rename(columns = cost_equity_2020_rename)
    ce_20 = ce_20[cost_equity_cols +['estimate_re']+ ['large'] + demo_vars_20]
    
    for col in cost_equity_cols:
        ce_20.loc[(ce_20['estimate_re'] == 2) | (pd.isnull(ce_20['estimate_re'])),col] = np.nan              
    ce_20 = ce_20.dropna(subset = cost_equity_cols,how = 'all')
    
    ## classify the survey answers    
    for var in cost_equity_cols:
        ce_20.loc[ce_20[var] >= 3,'{}_dum'.format(var)] = 1
        ce_20.loc[ce_20[var] < 3, '{}_dum'.format(var)] = 0
                       
    for col in [x for x in cost_equity_cols if '(\'22 only)' in x]:
        ce_01[col] = np.nan
    
    ce_01 = ce_01[[x for x in ce_01 if x not in ['Other','other']]]
    ce_20 = ce_20[[x for x in ce_20 if x not in ['Other','other']]]
    
    ce_01['Survey'] = '2001'
    ce_20['Survey'] = '2022*'
    return ce_20,ce_01,cost_equity_cols


## Test by demographic variable
def test_cost_equity(data,demo_var,cost_equity_cols,var_sort):

    import statsmodels.formula.api as sm

    data = data.loc[~pd.isnull(data[demo_var])]
    sort = [x for x in data[demo_var].unique().tolist() if '*' not in x] +\
               [x for x in data[demo_var].unique().tolist() if '*' in x]    

    for var in cost_equity_cols:
        reg_df = data.loc[(~pd.isnull(data[demo_var])) & (~pd.isnull(data[var]))][[demo_var,var]]    
        reg_df.loc[reg_df[demo_var]==sort[1],'dummy'] = 1
        reg_df.loc[reg_df[demo_var]==sort[0],'dummy'] = 0
        reg_df['yvar'] = reg_df[var]
        p = sm.ols(formula="yvar ~ dummy", data=reg_df).fit().pvalues.dummy

        # contingency = pd.crosstab(data[demo_var],data[var])
        # c, p, dof, expected = chi2_contingency(contingency) 
        
        hold = round(data.groupby(demo_var)[var].mean().to_frame().T,2)
        for col in hold.columns:
            hold[col] = hold[col].map('{:.2f}'.format).replace({'nan':''})
        # sort = [x for x in hold.columns if '*' not in x] + [x for x in hold.columns if '*' in x]    
        
        if len(reg_df[demo_var].unique().tolist()) == 1:
            p = 1
        if (p>0.05) & (p<=0.1):
            aster = '*'
        elif (p>0.01) & (p<=0.05):
            aster = '**'
        elif (p<=0.01):
            aster = '***'
        else:
            aster = ''
        
        hold[sort[1]] = hold[sort[1]] + aster
        if var == cost_equity_cols[0]:
            out = hold
        else:
            out = pd.concat([out,hold],axis = 0)
    out = out.reindex(var_sort)
    out = out[sort]
    
    l1 = [demo_var]*2
    l2 = [x.replace('*','') for x in sort]
    l3 = data.groupby(demo_var)[demo_var].count()[sort].values.tolist()
    tuples= [(l1[i],l2[i],l3[i]) for i in range(len(l1))]
    out.columns = pd.MultiIndex.from_tuples(tuples,names = ['Variable','Group','N'])
    
    return out


## Panel A:

## Get the data    
ce_20,ce_01,cost_equity_cols = cost_equity_fix(sdata_20,sdata_2001)

ce_20['Size'] = ce_20['large'].map({0:'Small',1:'Large*'})

ce_01['Size'] = ce_01['large'].map({0:'Small',1:'Large*'})

data = ce_01.append(ce_20)


var_sort = ['CAPM',
            'Multi-Factor Model',
            'Historical Average',
            'Investor Expectations',
            'Dividend Discount Model',
            'Regulatory Decisions',
            "Market Return ('22 only)",
            "Unchanged Estimate ('22 only)"]

comp_01_20 = test_cost_equity(data,'Survey',cost_equity_cols,var_sort)


## Test across small firms
data_small = data.loc[data['Size'] == 'Small']
data_small = data_small.rename(columns = {"Survey":"Small Firms"})
comp_01_20_small = test_cost_equity(data_small,'Small Firms',cost_equity_cols,var_sort)

## Test across large firms
data_large = data.loc[data['Size'] == 'Large*']
data_large = data_large.rename(columns = {"Survey":"Large Firms"})
comp_01_20_large = test_cost_equity(data_large,'Large Firms',cost_equity_cols,var_sort)


## Get the percent important or very important across surveys
percentages = data.groupby(['Size','Survey'])[[x+"_dum" for x in cost_equity_cols]].mean().rename(columns = {x+"_dum":x for x in cost_equity_cols}).T
percentages = percentages.sort_values(by = percentages.columns.tolist()[1],ascending=False)
for col in percentages.columns:
    percentages[col] = percentages[col].map('{:.2f}'.format).replace('nan','')

percentages = percentages[[('Large*','2001'),
                           ('Large*','2022*'),
                           ('Small','2001'),
                           ('Small','2022*')]]
n_list = data.groupby(['Size','Survey'])['Survey'].count().reindex(
                        percentages.columns).tolist()

percentages.columns = [('Large Firms','2001'),
                           ('Large Firms','2022'),
                           ('Small Firms','2001'),
                           ('Small Firms','2022')]
percentages.columns = pd.MultiIndex.from_tuples(
                        [x + (n_list[i],) for i,x in enumerate(percentages.columns)])
percentages.columns = pd.MultiIndex.from_tuples(
                      [('Panel A: Cost of Equity, 2001 vs. 2022 Comparison','Percent Always or Almost Always',
                        x[0].replace('*',''),x[1].replace('*',''),x[2]
                       ) for x in percentages.columns],
                      names = ['Panel','Variable','Group','Survey','N'])

sort = percentages.index.tolist()

compare_out = pd.concat([comp_01_20,comp_01_20_large,comp_01_20_small],axis=1)

compare_out.columns = pd.MultiIndex.from_tuples(
                      [('Panel A: Cost of Equity, 2001 vs. 2022 Comparison','Score',) + x for x in compare_out.columns],
                       names = ['Panel','Variable','Group','Survey','N'])

panel_a = pd.concat([percentages,compare_out],axis=1)

## Panel B
var_sort = panel_a.index.tolist()


## Get the 2020 data
data_20 = data.loc[data['Survey']!='2001']

demo_vars_out = ['Size','Public','Growth Prospects','Pay Dividends','Leverage','Cash','Financial Flexibility']

for demo_var in demo_vars_out:
    
    hold = test_cost_equity(data_20,demo_var,cost_equity_cols,var_sort)
    if demo_var == demo_vars_out[0]:
        cost_equity_table = hold
    else:
        cost_equity_table = pd.concat([cost_equity_table,hold],axis=1)
        
old_cols = cost_equity_table.columns

cost_equity_table.columns = pd.MultiIndex.from_tuples(
                            [('Panel B: Cost of Equity, Conditional on Company Characteristics',)+x for x in cost_equity_table.columns],
                            names = ['Panel','Demographic Variable','Group','N'])


panel_b = cost_equity_table

panel_b = panel_b.reindex(var_sort)

panel_b = panel_b.rename(
          index = {x:x.replace("(\'22 only)","") for x in panel_b.index.tolist()})


## Display table
print("\n\n\n")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print("Displaying Table III, Panel A:")
    print(panel_a)
    print("\n\n\n")
    print("Displaying Table III, Panel B:")
    print(panel_b)

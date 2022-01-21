"""
Code to produce Figure 2, Table I
John Barry
2022/01/14
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None  # default='warn'


## Load in the survey data
filename = 'Data/datafile.pkl'
with open(filename, 'rb') as f:
    demo_vars_19,demo_vars_20,demo_vars_20_only,demo_var_list,sdata,sdata_20,sdata_2001 = pickle.load(f)
    f.close()



#%% FIGURE 2: CAPITAL BUDGETING TECHNIQUES

## This function standardizes data across 2001 and 2022 data
def cap_budg_fix(sdata_20_in,sdata_2001_in):
    
    cap_budg_dict_2001 = {'Q1A':'Net Present Value',
                 'Q1B':'IRR/Hurdle Rate',
                 'Q1C':'Hurdle Rate',
                 'Q1D':'P/E Multiples',
                 'Q1E':'Adjusted Present Value',
                 'Q1F':'Payback',
                 'Q1G':'Discounted Payback',
                 'Q1H':'Profitability Index',
                 'Q1I':'ROIC (\'22) Book Return (\'01)',
                 'Q1J':'Scenario Analysis',
                 'Q1K':'Simulation Analysis/VAR',
                 'Q1L':'Real Options',
                 'Q1M':'Other'}

    cap_budg_dict_2020 = {'a':'Net Present Value',
                          'b':'IRR/Hurdle Rate',
                          'c':'P/E Multiples',
                          'd':'Payback',
                          'e':'Profitability Index',
                          'f':'ROIC (\'22) Book Return (\'01)',
                          'g':'Scenario Analysis',
                          'h':'Simulation Analysis/VAR',
                          'i':'Real Options',
                          'j':'Adjusted Present Value',
                          'k':'Other'}

    ## Get the 2001 answers
    cap_budg_cols=list(cap_budg_dict_2001.values())    
    cap_budg_cols = [x for x in cap_budg_cols if x not in ['Other','Hurdle rate','Discounted payback']]
    sdata_2001_in = sdata_2001_in.rename(columns = cap_budg_dict_2001)    
    sdata_2001_in = sdata_2001_in[cap_budg_cols + ['large']]
    sdata_2001_in[sdata_2001_in==9]=np.nan
    sdata_2001_in = sdata_2001_in.dropna(subset = cap_budg_cols,how = 'all')
    sdata_2001_in['IRR/Hurdle Rate'] = sdata_2001_in[['IRR/Hurdle Rate']].max(axis=1)
    sdata_2001_in['Payback'] = sdata_2001_in[['Payback']].max(axis=1)
    
    ## classify the survey answers    
    for var in cap_budg_cols:
        sdata_2001_in.loc[sdata_2001_in[var] >= 3,'{}_dum'.format(var)] = 1
        sdata_2001_in.loc[sdata_2001_in[var] < 3, '{}_dum'.format(var)] = 0
        # sdata_2001_in.loc[pd.isnull(sdata_2001_in[var]), '{}_dum'.format(var)] = 0

    sdata_2001_in = sdata_2001_in[[x for x in sdata_2001_in if x not in ['Discounted Payback','Hurdle Rate','Other',
                                                                         'Discounted Payback_dum','Hurdle Rate_dum','Other_dum']]]

    ## Get the 2020 answers    
    sdata_20_in = sdata_20_in[[x for x in sdata_20_in.columns if x.startswith('q30')] + ['id_specific_2020','large']]
    del sdata_20_in['q30_2_wave2'],sdata_20_in['q30_other_wave2']
    cols_from_20 = [x for x in sdata_20_in.columns if 'q30' in x]
    
    sdata_20_in = sdata_20_in.dropna(subset = cols_from_20,how = 'all')
    
    sdata_20_in = (sdata_20_in.rename(columns = \
                               {cols_from_20[i]:cols_from_20[i][4] for i in range(len(cols_from_20))})).\
                               rename(columns = cap_budg_dict_2020)
                               
    # sdata_20 = sdata_20[cap_budg_cols +  ['id_specific_2020','large']]
    cap_budg_cols=list(cap_budg_dict_2020.values())    

    for var in cap_budg_cols:
        sdata_20_in[var] = sdata_20_in[var] - 1
        sdata_20_in.loc[sdata_20_in[var] >= 3,'{}_dum'.format(var)] = 1
        sdata_20_in.loc[sdata_20_in[var] < 3, '{}_dum'.format(var)] = 0
        # sdata_20_in.loc[pd.isnull(sdata_20_in[var]), '{}_dum'.format(var)] = 0
    
    sdata_20_in = sdata_20_in[[x for x in sdata_20_in if x not in ['Discounted Payback','Hurdle Rate','Other',
                                                                   'Discounted Payback_dum','Hurdle Rate_dum','Other_dum']]]

    
    sdata_20_in['Survey'] = '2022*'
    sdata_2001_in['Survey'] = '2001'
    
    cap_budg_cols = [x for x in cap_budg_cols if x not in ['Other']]

    return sdata_20_in,sdata_2001_in,cap_budg_cols


## This function creates the figure in the paper
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


## Get the data from the 2020 and 2001 survey waves
cb_20,cb_01,cap_budg_cols = cap_budg_fix(sdata_20,sdata_2001)
cb_20['Size'] = cb_20['large'].map({0:'Small',1:'Large*'})
cb_01['Size'] = cb_01['large'].map({0:'Small',1:'Large*'})

data = cb_01.append(cb_20)
data = data.dropna(subset = cap_budg_cols,how = 'all')

## Combine datasets together
def combine(cb_20_in,cb_01_in):
    out_01 = cb_01_in[[x for x in cb_01_in if '_dum' in x]].mean().to_frame()
    out_01 = out_01.rename(index = {x:x.replace("_dum","") for x in out_01.index.tolist()},
                           columns = {0:'2001'})
    
    out_20 = cb_20_in[[x for x in cb_20_in if '_dum' in x]].mean().to_frame()
    out_20 = out_20.rename(index = {x:x.replace("_dum","") for x in out_20.index.tolist()},
                           columns = {0:'2022*'})
    
    return pd.concat([out_20,out_01],axis=1)


cap_budg = combine(cb_20,cb_01)

## Prepare the data for the figure
cap_budg_large = combine(cb_20.loc[cb_20['Size'] == 'Large*'],cb_01.loc[cb_01['Size'] == 'Large*'])
cap_budg_large = cap_budg_large.sort_values(by = '2022*')

var_sort = cap_budg_large.index.tolist()
cap_budg_small = combine(cb_20.loc[cb_20['Size'] == 'Small'],cb_01.loc[cb_01['Size'] == 'Small'])
cap_budg_small = cap_budg_small.reindex(var_sort)


large = cap_budg_large.rename(columns = {'2001':'2001 Large','2022*':'2022 Large'}).sort_values(by = '2022 Large')
small = cap_budg_small.rename(columns = {'2001':'2001 Small','2022*':'2022 Small'})

large = large[['2001 Large','2022 Large']]
small = small[['2001 Small','2022 Small']]

first_dict = large.to_dict('list')
second_dict = small.reindex(large.index.tolist()).to_dict('list')

total_width = 0.8
single_width = 0.9

alpha = 1
ncols = 4
xlabels = [x for x in large.index.tolist()]

print("\n\n\n")
print("Displaying Figure 2")
fig,ax = crosshatch_large_small(first_dict,second_dict,xlabels,fig_size = (6.5,4.5),
                                total_width = 0.8,single_width = 0.9,
                                legend=True,alph=1,ncols = 4)
plt.show()


#%% TABLE I: CAPITAL BUDGETING TECHNIQUES
##


## Tests differences in scores by demographic group
def test_cap_budg(data,demo_var,cap_budg_cols,var_sort):
    
    import statsmodels.formula.api as sm

    data = data.loc[~pd.isnull(data[demo_var])]

    sort = [x for x in data[demo_var].unique().tolist() if '*' not in x] +\
               [x for x in data[demo_var].unique().tolist() if '*' in x]    

    for var in cap_budg_cols:
        reg_df = data.loc[(~pd.isnull(data[demo_var])) & (~pd.isnull(data[var]))][[demo_var,var]]    
        reg_df.loc[reg_df[demo_var]==sort[1],'dummy'] = 1
        reg_df.loc[reg_df[demo_var]==sort[0],'dummy'] = 0
        reg_df['yvar'] = reg_df[var]
        p = sm.ols(formula="yvar ~ dummy", data=reg_df).fit().pvalues.dummy

        
        hold = round(data.groupby(demo_var)[var].mean().to_frame().T,2)
        for col in hold.columns:
            hold[col] = hold[col].map('{:.2f}'.format)
        sort = [x for x in hold.columns if '*' not in x] + [x for x in hold.columns if '*' in x]    
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
        if var == cap_budg_cols[0]:
            out = hold
        else:
            out = pd.concat([out,hold],axis = 0)
    out = out[sort]
    
    l1 = [demo_var]*2
    l2 = [x.replace('*','') for x in sort]
    l3 = data.groupby(demo_var)[demo_var].count()[sort].values.tolist()
    tuples= [(l1[i],l2[i],l3[i]) for i in range(len(l1))]
    out.columns = pd.MultiIndex.from_tuples(tuples,names = ['Variable','Group','N'])
    
    return out

## Panel A ##
## Comparison across surveys
comp_01_20 = test_cap_budg(data,'Survey',cap_budg_cols,var_sort)

## Comparison of small firms across surveys
data_small = data.loc[data['Size'] == 'Small']
data_small['Small Firms'] = data_small['Survey']
comp_01_20_small = test_cap_budg(data_small,'Small Firms',cap_budg_cols,var_sort)

## Comparison of large firms across surveys
data_large = data.loc[data['Size'] == 'Large*']
data_large['Large Firms'] = data_large['Survey']
comp_01_20_large = test_cap_budg(data_large,'Large Firms',cap_budg_cols,var_sort)

data_20 = data.loc[data['Survey']!='2001']

data_20 = data_20[[x for x in data_20 if x not in ['Size']]]

data_20 = data_20.merge(sdata_20[demo_vars_20+['id_specific_2020']],on = 'id_specific_2020')

percentages = data.groupby(['Size','Survey'])[[x+"_dum" for x in cap_budg_cols]].mean().rename(columns = {x+"_dum":x for x in cap_budg_cols}).T



percentages = round(percentages*100,2)
for col in percentages.columns:
    percentages[col] = percentages[col].map('{:.2f}'.format)

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
                      [('Percent Always or Almost Always',
                        x[0].replace('*',''),x[1].replace('*',''),x[2]
                       ) for x in percentages.columns],
                      names = ['Variable','Group','Survey','N'])

n_list = data.groupby(['Survey'])['Survey'].count().tolist()

compare = pd.concat([comp_01_20,comp_01_20_small,comp_01_20_large],axis=1)

keys = ['2001 vs. 2022']*len(compare.columns)
old_cols = list(compare.columns)
insert_tuple = [(keys[i],)+old_cols[i] for i in range(len(keys))]

compare.columns = pd.MultiIndex.from_tuples(insert_tuple,names = ['Comparison Group','Variable','Group','N'])

panel_a = pd.concat([percentages,compare],axis=1)

## Panel B ##
demo_vars_out = ['Size','Public','Growth Prospects','Pay Dividends','Leverage','Cash','Financial Flexibility']

for demo_var in demo_vars_out:
    
    hold = test_cap_budg(data_20,demo_var,cap_budg_cols,var_sort)
    if demo_var == demo_vars_out[0]:
        panel_b = hold
    else:
        panel_b = pd.concat([panel_b,hold],axis=1)
        
keys = ['2022']*(len(panel_b.columns))
old_cols = list(panel_b.columns)
insert_tuple = [(keys[i],)+old_cols[i] for i in range(len(keys))]
panel_b.columns = pd.MultiIndex.from_tuples(insert_tuple,names = ['Comparison Group','Variable','Group','N'])


## Display table
print("\n\n\n")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print("Displaying Table I, Panel A:")
    print(panel_a)
    print("\n\n\n")
    print("Displaying Table I, Panel B:")
    print(panel_b)

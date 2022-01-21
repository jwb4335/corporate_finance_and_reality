"""
Code to produce Figure 20, Table IX Panel A, Figure A8.1, A8.2
John Barry
2022/01/14
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import copy


## Load in the survey data
filename = 'Data/datafile.pkl'
with open(filename, 'rb') as f:
    demo_vars_19,demo_vars_20,demo_vars_20_only,demo_var_list,sdata,sdata_20,sdata_2001 = pickle.load(f)
    f.close()


#%%

rename = {1:'Temporary change in earnings',
          2:'Sustainable change in earnings',
          3:'Stability of future earnings',
          4:'Having extra cash/liquid assets',
          5:'Payout policy of competitors',
          6:'Personal taxes of stockholders',
          7:'Availability of investment opportunities',
          8:'Market price of stock',
          9:'Preferences of our investors'}

rename_div = {"q54_div{}_wave2".format(i):rename[i] for i in list(rename.keys())}
rename_rep = {"q54_rep{}_wave2".format(i):rename[i] for i in list(rename.keys())}

rename_index = {0:'Very Unimportant',1:'Unimportant',2:'Middle',3:'Important',4:'Very Important'}

pay_data = sdata_20

div_drivers = pay_data[[x for x in pay_data if 'q54' in x]].dropna(how = 'all')[[x for x in pay_data if 'q54_div' in x]]
rep_drivers = pay_data[[x for x in pay_data if 'q54' in x]].dropna(how = 'all')[[x for x in pay_data if 'q54_rep' in x]]

div_drivers = div_drivers.rename(columns = rename_div) + 2

rep_drivers = rep_drivers.rename(columns = rename_rep) + 2

cols = list(rename.values())

def get_percent_important(pay):
    for col in cols:
        hold = pay[col].value_counts().divide(len(pay)).to_frame().sort_index().rename(index = rename_index).T
        
        if col == cols[0]:
            out = hold
        else:
            out = pd.concat([out,hold],axis=0)
    out = out.replace(np.nan,0)
    out['sort'] = out['Important'] + out['Very Important']
    out = out.sort_values(by = 'sort')
    out = out[['Important','Very Important']]
    return out



div_drivers = div_drivers.dropna(how = 'all')

rep_drivers = rep_drivers.dropna(how = 'all')


div_imp = get_percent_important(div_drivers)
div_imp['Very Important'] = div_imp['Very Important'] + div_imp['Important']
# div_imp = div_imp[['Very Important','Important']]
div_imp.columns = pd.MultiIndex.from_tuples([('Dividends',x) for x in div_imp.columns],names = ['Payout','Importance'])

rep_imp = get_percent_important(rep_drivers)
rep_imp['Very Important'] = rep_imp['Very Important'] + rep_imp['Important']
# rep_imp = rep_imp[['Very Important','Important']]
rep_imp.columns = pd.MultiIndex.from_tuples([('Repurchases',x) for x in rep_imp.columns],names = ['Payout','Importance'])
imp = pd.concat([rep_imp,div_imp],axis=1)


div_drivers['Payout'] = 'Div.*'
rep_drivers['Payout'] = 'Rep.'

div_drivers = pd.concat([div_drivers,sdata_20[demo_vars_20]],axis=1,join='inner').reset_index(drop=True)

rep_drivers = pd.concat([rep_drivers,sdata_20[demo_vars_20]],axis=1,join='inner').reset_index(drop=True)


data_in = div_drivers.append(rep_drivers)

cols_in = div_imp.index.tolist()
cols_in.reverse()

#%% Figure 20. Important Objectives Driving Payout Decisions

def payout_explanations(data,second_dict,xlabels,fig_size = (9.5,5.5),
                           total_width=0.8, single_width=0.8, legend=True, 
                           alph = 1,legend_loc = 'best',ncols = 2,fontsize = 12):
    
    

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
            bar = ax.barh(x + x_offset, y, height=bar_width * single_width, color=colors[i],alpha=0.5)
    
        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])
    if second_dict is not None:
        colors = prop_cycle.by_key()['color']
        colors = [colors[1], colors[0]]
        for i, (name, values) in enumerate(second_dict.items()):
            for x, y in enumerate(values):
                # col = tuple([x/255 for x in  ImageColor.getcolor(colors[i], "RGB")])+ (0.5,)
                x_offset = x_offset = (i - n_bars / 2) * bar_width + bar_width / 2
                bar = ax.barh(x + x_offset, y, height=bar_width * single_width, 
                        color='white',alpha=0.25)
                bar = ax.barh(x + x_offset, y, height=bar_width * single_width, 
                        color=colors[i],alpha=1)
            bars.append(bar[0])
    ax.set_yticks(np.arange(len(xlabels)))
    ax.set_yticklabels(xlabels,fontsize = fontsize)
    ax.set_axisbelow(True)
    ax.grid(axis='x',alpha=0.5,linestyle='--',zorder=-100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    from matplotlib.ticker import FuncFormatter
    ax.set_xticks(np.arange(0,0.81,0.2))
    plt.xticks(fontsize = fontsize)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
    
    fig = plt.gcf()
    # Draw legend
    fig.legend([bars[3],bars[1]], 
               ['Important','Very Important'],
               ncol = 1,loc = 'center',
               bbox_to_anchor = (0.35,-0.025),fontsize = fontsize,title = r'$\bf{Dividend}$ $\bf{Payers}$',title_fontsize = fontsize)
    fig.legend([bars[2],bars[0]], 
               ['Important','Very Important'],
               ncol = 1,loc = 'center',
               bbox_to_anchor = (0.55,-0.025),fontsize = fontsize,title = r'$\bf{Repurchasers}$',title_fontsize = fontsize)
    return fig,ax

div_imp = get_percent_important(div_drivers)
rep_imp = get_percent_important(rep_drivers)

div_imp['sort'] = div_imp['Important'] + div_imp['Very Important']
rep_imp['sort'] = rep_imp['Important'] + rep_imp['Very Important']

data = pd.concat([rep_imp['sort'],div_imp['sort']],axis=1)
data.columns = ['Repurchases','Dividends']
data = data.sort_values(by = ['Dividends'])
data_hold = copy.deepcopy(data)
data = data.to_dict('list')
second_dict = pd.concat([rep_imp['Important'],div_imp['Important']],axis=1)
second_dict.columns = ['Repurchases','Dividends']
second_dict = second_dict.reindex(data_hold.index)
second_dict = second_dict.to_dict('list')
xlabels = data_hold.index.tolist()

print("Displaying Figure 20:")
fig,ax = payout_explanations(data,second_dict,xlabels,fontsize = 12,fig_size = (8.5,5))
plt.show()


#%% Table IX. Panel A

def test_div_imp(data_in, demo_var,cols_in):
    
    import statsmodels.formula.api as sm

    data_test = data_in.loc[~pd.isnull(data_in[demo_var])]

    sort = [x for x in data_test[demo_var].unique().tolist() if '*' not in x] +\
               [x for x in data_test[demo_var].unique().tolist() if '*' in x]    

    
    for var in cols_in:
        reg_df = data_test.loc[(~pd.isnull(data_test[demo_var])) & (~pd.isnull(data_test[var]))][[demo_var,var]]    
        reg_df.loc[reg_df[demo_var]==sort[1],'dummy'] = 1
        reg_df.loc[reg_df[demo_var]==sort[0],'dummy'] = 0
        reg_df['yvar'] = reg_df[var]
        p = sm.ols(formula="yvar ~ dummy", data=reg_df).fit().pvalues.dummy

        
        hold = data_test.groupby(demo_var)[var].mean().to_frame().T
        for col in hold.columns:
            hold[col] = hold[col].map('{:.2f}'.format).replace({'nan':''})
        sort = [x for x in hold.columns if '*' not in x] + [x for x in hold.columns if '*' in x]    
        
        if (p>0.01) & (p<=0.05):
            aster = '*'
        elif (p>0.001) & (p<=0.01):
            aster = '**'
        elif (p<=0.001):
            aster = '***'
        else:
            aster = ''
        
        hold[sort[1]] = hold[sort[1]] + aster
        if var == cols_in[0]:
            out = hold
        else:
            out = pd.concat([out,hold],axis = 0)
    out = out[sort]
    
    l1 = [demo_var]*2
    l2 = [x.replace('*','') for x in sort]
    l3 = data_test.groupby(demo_var)[demo_var].count()[sort].values.tolist()
    tuples= [(l1[i],l2[i],l3[i]) for i in range(len(l1))]
    out.columns = pd.MultiIndex.from_tuples(tuples,names = ['Variable','Group','N'])
    
    return out

demo_var = 'Payout'
test = test_div_imp(data_in,demo_var,cols_in)

test_public = test_div_imp(data_in.loc[data_in['Public'] == 'Yes*'],demo_var,cols_in)

test_large = test_div_imp(data_in.loc[data_in['Size'] == 'Large*'],demo_var,cols_in)

test_small = test_div_imp(data_in.loc[data_in['Size'] == 'Small'],demo_var,cols_in)


div_imp = get_percent_important(div_drivers)
div_imp['Very Important'] = div_imp['Very Important'] + div_imp['Important']
# div_imp = div_imp[['Very Important','Important']]
div_imp.columns = pd.MultiIndex.from_tuples([('Dividends',x) for x in div_imp.columns],names = ['Payout','Importance'])

rep_imp = get_percent_important(rep_drivers)
rep_imp['Very Important'] = rep_imp['Very Important'] + rep_imp['Important']
# rep_imp = rep_imp[['Very Important','Important']]
rep_imp.columns = pd.MultiIndex.from_tuples([('Repurchases',x) for x in rep_imp.columns],names = ['Payout','Importance'])

div_imp_out = div_imp
div_imp_out[('Div.','Percent Important or Very Important')] = div_imp_out[('Dividends','Very Important')]

rep_imp_out = rep_imp
rep_imp_out[('Rep.','Percent Important or Very Important')] = rep_imp_out[('Repurchases','Very Important')]



## Format the table
n_list = [len(rep_drivers),len(div_drivers)]

imp_out = pd.concat(
                    [rep_imp_out[[('Rep.','Percent Important or Very Important')]],
                     div_imp_out[[('Div.','Percent Important or Very Important')]]],
                    axis=1)

imp_out.columns = pd.MultiIndex.from_tuples(
                  [('Panel A: Important Factors Driving Payout Decisions',
                    'Percent Important or Very Important',
                    'All Firms',
                    x[0],n_list[i]) for i,x in enumerate(imp_out.columns)],
                  names = ['Panel','Variable','Sample','Payout Type','N'])

for col in imp_out.columns:
    imp_out[col] = (imp_out[col]*100).map('{:.2f}'.format).replace('nan','')


test_out = copy.deepcopy(test)
test_out_public = copy.deepcopy(test_public)
test_out_large = copy.deepcopy(test_large)
test_out_small = copy.deepcopy(test_small)


test_out.columns = pd.MultiIndex.from_tuples(
               [('Panel A: Important Factors Driving Payout Decisions','Sub-Sample Comparisons','All Firms',x[1],x[2]) \
                for x in test.columns],
               names = ['Panel','Variable','Sample','Payout Type','N'])
    
test_out_public.columns = pd.MultiIndex.from_tuples(
               [('Panel A: Important Factors Driving Payout Decisions','Sub-Sample Comparisons','Public Firms',x[1],x[2]) \
                for x in test_public.columns],
               names = ['Panel','Variable','Sample','Payout Type','N'])
    
test_out_large.columns = pd.MultiIndex.from_tuples(
               [('Panel A: Important Factors Driving Payout Decisions','Sub-Sample Comparisons','Large Firms',x[1],x[2]) \
                for x in test_large.columns],
               names = ['Panel','Variable','Sample','Payout Type','N'])
   
test_out_small.columns = pd.MultiIndex.from_tuples(
               [('Panel A: Important Factors Driving Payout Decisions','Sub-Sample Comparisons','Small Firms',x[1],x[2]) \
                for x in test_small.columns],
               names = ['Panel','Variable','Sample','Payout Type','N'])
panel_a = pd.concat([imp_out,test_out,test_out_public,test_out_large,test_out_small],axis=1)

panel_a = panel_a.reindex(cols_in)


## Print results
print('\n\n\n')
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print("Displaying Table IX Panel A:")
    print(panel_a)

#%% Figure A8.1
explain = sdata_20[[x for x in sdata_20 if 'q55' in x]].dropna(how = 'all') + 2

rename = {'1':'We make payout decisions after\ninvestment plans are determined',
              '2':'Payout decisions convey\ninformation about our firm to investors',
              '3':'Paying dividends/repurchasing makes our stock less risky\n(relative to retaining earnings)',
              '4':'There are negative consequences to\nreducing dividends/repurchases',
              '5':'Rather than reducing dividends/repurchases,\nwe would raise new funds to undertake a project',
              '6':'We use our payout policy as a tool\nto attain a desired credit rating',
              '7':'We use our payout policy to make us\nlook better than our competitors'}

rename_div = {"q55_div{}_wave2".format(x):rename[x] for x in list(rename.keys())}
rename_rep = {"q55_rep{}_wave2".format(x):rename[x] for x in list(rename.keys())}

div = explain[[x for x in explain if 'div' in x]].dropna(how = 'all').rename(columns = rename_div)
rep = explain[[x for x in explain if 'rep' in x]].dropna(how = 'all').rename(columns = rename_rep)

def get_graph(data):
    rename_index = {0:'Strongly Disagree',1:'Disagree',2:'Meh',3:'Agree',4:'Strongly Agree'}
    i = 0
    for col in data.columns:
        hold = data[col].value_counts().divide(len(data)).to_frame().sort_index().rename(index = rename_index).T
        check =[x for x in rename_index.values() if x not in hold.columns]
        for val in check:
            hold[val] = 0
        hold = hold[['Agree','Strongly Agree']]
        if i == 0:
            out = hold
        else:
            out = pd.concat([out,hold],axis=0)
        i = i + 1
    out['sort'] = out.sum(axis=1)
    out = out.sort_values(by = 'sort')
    return out

div_out = get_graph(div)
rep_out = get_graph(rep)


data = div_out[['Agree','Strongly Agree']].to_dict('list')
data = pd.concat([rep_out['sort'],div_out['sort']],axis=1)
data.columns = ['Repurchases','Dividends']
data = data.sort_values(by = ['Dividends'])
data_hold = copy.deepcopy(data)
data = data.to_dict('list')
second_dict = pd.concat([rep_out['Agree'],div_out['Agree']],axis=1)
second_dict.columns = ['Repurchases','Dividends']
second_dict = second_dict.reindex(data_hold.index)
second_dict = second_dict.to_dict('list')
xlabels = data_hold.index.tolist()
def payout_explanations(data,second_dict,xlabels,fig_size = (9.5,5.5),
                           total_width=0.8, single_width=0.8, legend=True, 
                           alph = 1,legend_loc = 'best',ncols = 2,fontsize = 12):
    
    

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
            bar = ax.barh(x + x_offset, y, height=bar_width * single_width, color=colors[i],alpha=0.5)
    
        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])
    if second_dict is not None:
        colors = prop_cycle.by_key()['color']
        colors = [colors[1], colors[0]]
        for i, (name, values) in enumerate(second_dict.items()):
            for x, y in enumerate(values):
                # col = tuple([x/255 for x in  ImageColor.getcolor(colors[i], "RGB")])+ (0.5,)
                x_offset = x_offset = (i - n_bars / 2) * bar_width + bar_width / 2
                bar = ax.barh(x + x_offset, y, height=bar_width * single_width, 
                        color='white',alpha=0.25)
                bar = ax.barh(x + x_offset, y, height=bar_width * single_width, 
                        color=colors[i],alpha=1)
            bars.append(bar[0])
    ax.set_yticks(np.arange(len(xlabels)))
    ax.set_yticklabels(xlabels,fontsize = fontsize)
    ax.set_axisbelow(True)
    ax.grid(axis='x',alpha=0.5,linestyle='--',zorder=-100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    from matplotlib.ticker import FuncFormatter
    ax.set_xticks(np.arange(0,0.61,0.1))
    plt.xticks(fontsize = fontsize)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
    
    fig = plt.gcf()
    # Draw legend if we need
    fig.legend([bars[3],bars[1]], 
               ['Agree','Strongly Agree'],
               ncol = 1,loc = 'center',
               bbox_to_anchor = (0.35,-0.025),fontsize = fontsize,title = r'$\bf{Dividend}$ $\bf{Payers}$',title_fontsize = fontsize)
    fig.legend([bars[2],bars[0]], 
               ['Agree','Strongly Agree'],
               ncol = 1,loc = 'center',
               bbox_to_anchor = (0.6,-0.025),fontsize = fontsize,title = r'$\bf{Repurchasers}$',title_fontsize = fontsize)
    return fig,ax


print("\n\n\n")
print("Displaying Figure A8.1:")
fig,ax = payout_explanations(data,second_dict,xlabels,fontsize =12,fig_size = (7.5,5))
plt.show()


#%% Figure A8.2. Factors Driving Payout Decisions

rename = {'a':'Temporary change in earnings',
          'b':'Sustainable change in earnings',
          'c':'Stability of future earnings',
          'd':'Having extra cash/liquid assets',
          5:'Payout policy of competitors',
          6:'Personal taxes of stockholders',
          7:'Availability of investment opportunities',
          8:'Market price of stock',
          9:'Preferences of our investors'}

rename_div = {"q54_div{}_wave2".format(i):rename[i] for i in list(rename.keys())}
rename_rep = {"q54_rep{}_wave2".format(i):rename[i] for i in list(rename.keys())}

rename_index = {0:'Very Unimportant',1:'Unimportant',2:'Middle',3:'Important',4:'Very Important'}

# pay_data = sdata_20.loc[sdata_20['Public'] == 'Yes*']
pay_data = sdata_20

drivers = pay_data[[x for x in pay_data if ('q56' in x or 'q57' in x) and ('other' not in x)]].dropna(how = 'all')[[x for x in pay_data if  ('q56' in x or 'q57' in x) and ('other' not in x)]]

drivers = drivers+ 2

div_drivers = drivers[[x for x in drivers if 'q56' in x]].dropna(how = 'all').replace({np.nan:0})
rep_drivers = drivers[[x for x in drivers if 'q57' in x]].dropna(how = 'all').replace({np.nan:0})


rename = {'q56a':'Maintain a smooth dividend stream from year-to-year',
          'q56b':'Avoid reducing dividends per share',
          'q56c':'Attract investors that can only own stocks that pay dividends',
          'q57a':'Investors pay lower taxes on repurchases than on dividends',
          'q57b':'Increase earnings per share',
          'q57c':'Accumulate shares to improve chance of resisting takeover',
          'q57d':'Change capital structure so it is closer to desired debt ratio',
          'q57e':'Whether our stock is a good investment relative to other options',
          'q57f':'Offset the dilutionary effect of stock option plans',
          'q57g':'The float/liquidity of our stock'}
rename2 = {x+"_wave2":rename[x] for x in list(rename.keys())}

def get_percent_important(pay):
    cols = pay.columns.tolist()
    for col in cols:
        hold = pay[col].value_counts().divide(len(pay)).to_frame().sort_index().rename(index = rename_index).T
        
        if col == cols[0]:
            out = hold
        else:
            out = pd.concat([out,hold],axis=0)
    out = out.replace(np.nan,0)
    out['sort'] = out['Important'] + out['Very Important']
    out = out.sort_values(by = 'sort')
    out = out[['Important','Very Important']]
    return out

div_imp = get_percent_important(div_drivers)
# div_imp['Very Important'] = div_imp['Very Important'] + div_imp['Important']
div_imp = div_imp[['Important','Very Important']]
div_imp = div_imp.rename(index = rename2)
div_imp.columns = pd.MultiIndex.from_tuples([('Dividends',x) for x in div_imp.columns],names = ['Payout','Importance'])

rep_imp = get_percent_important(rep_drivers)
# rep_imp['Very Important'] = rep_imp['Very Important'] + rep_imp['Important']
rep_imp = rep_imp[['Important','Very Important']]
rep_imp = rep_imp.rename(index = rename2)
rep_imp.columns = pd.MultiIndex.from_tuples([('Repurchases',x) for x in rep_imp.columns],names = ['Payout','Importance'])


div_imp_plot = copy.deepcopy(div_imp)
rep_imp_plot = copy.deepcopy(rep_imp)

for col in [x for x in div_imp_plot.columns if 'Very' in x[1]]:
    div_imp_plot[col] = div_imp_plot.sum(axis=1)
for col in [x for x in rep_imp_plot.columns if 'Very' in x[1]]:
    rep_imp_plot[col] = rep_imp_plot.sum(axis=1)
    
    
div_imp_plot.index = pd.MultiIndex.from_tuples([('Dividends',x) for x in div_imp_plot.index])
rep_imp_plot.index = pd.MultiIndex.from_tuples([('Repurchases',x) for x in rep_imp_plot.index])

plot = pd.concat([rep_imp_plot.droplevel(0,1).sort_values(by = 'Very Important',ascending=True),
                  div_imp_plot.droplevel(0,1).sort_values(by = 'Very Important',ascending=True)])

data = plot.loc['Dividends'][['Very Important']].to_dict('list')
second_dict = plot.loc['Dividends'][['Important']].to_dict('list')
xlabels = plot.loc['Dividends'].index.tolist()
xlabels_div = ['Attract investors that\ncanonly own stocks\nthat pay dividends',
     'Maintain a smooth\ndividend stream\nfrom year-to-year',
     'Avoid reducing\ndividends per share']

xlabels_rep = ['Accumulate shares to improve\nchance of resisting takeover',
 'Investors pay lower taxes on\nrepurchases than on dividends',
 'Change capital structure so\nit is closer to desired debt ratio',
 'The float/liquidity of our stock',
 'Offset the dilutionary effect\nof stock option plans',
 'Increase earnings per share',
 'Whether our stock is a good\ninvestment relative to other options']

xlab_dict = {'Dividends':xlabels_div,
             'Repurchases':xlabels_rep}


title_dict = {'Dividends':'(A) Dividends',
              'Repurchases':'(B) Repurchases'}

total_width_dict =  {'Dividends':0.4,
              'Repurchases':1}

# def payout_explanations(data,second_dict,xlabels,fig_size = (9.5,5.5),
#                            total_width=0.8, single_width=0.8, legend=True, 
#                            alph = 1,legend_loc = 'best',ncols = 2,fontsize = 12):
    
    

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
colors = [colors[0], colors[1]]
alph = [0.5,1]
j = 1
single_width = 0.8
fontsize = 10
for plotvar in ['Dividends','Repurchases']:

    xlabels = xlab_dict[plotvar]
    data = plot.loc[plotvar][['Very Important']].to_dict('list')
    second_dict = plot.loc[plotvar][['Important']].to_dict('list')
    total_width = total_width_dict[plotvar]
    plt.subplot(1,2,j)
    j +=1

    ax = plt.gca()
    
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
            bar = ax.barh(x, y, height=bar_width * single_width, color=colors[1],alpha=1)
    
        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])
    if second_dict is not None:
        for i, (name, values) in enumerate(second_dict.items()):
            for x, y in enumerate(values):
                # col = tuple([x/255 for x in  ImageColor.getcolor(colors[i], "RGB")])+ (0.5,)
                x_offset = x_offset = (i - n_bars / 2) * bar_width + bar_width / 2
                bar = ax.barh(x + x_offset, y, height=bar_width * single_width, 
                        color='white',alpha=1)
                bar = ax.barh(x + x_offset, y, height=bar_width * single_width, 
                        color=colors[0],alpha=1)
            bars.append(bar[0])
            
    ax.set_yticks(np.arange(len(xlabels)))
    ax.set_yticklabels(xlabels,fontsize = fontsize)
    ax.set_axisbelow(True)
    ax.grid(axis='x',alpha=0.5,linestyle='--',zorder=-100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    from matplotlib.ticker import FuncFormatter
    ax.set_xticks(np.arange(0,0.81,0.2))
    plt.xticks(fontsize = fontsize)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
    ax.set_xlim((0.,0.81))
    plt.xticks(fontsize = fontsize)
    ax.set_title(title_dict[plotvar],{'fontname':'Times New Roman'},
                 fontsize =20,fontweight = 'bold',loc = 'left',
                 x = -0.12)

print("\n\n\n")
print("Displaying Figure A8.2:")
fig = plt.gcf()
fig.set_size_inches(9,4)
plt.subplots_adjust(wspace = 1)
fig.legend([bars[1],bars[0]], 
               ['Agree','Strongly Agree'],
               ncol = 2,loc = 'center',
               bbox_to_anchor = (0.5,0),fontsize = fontsize)
plt.show()


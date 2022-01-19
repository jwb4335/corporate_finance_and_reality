
import pandas as pd
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

#%% Figure 21. How Do Companies Prioritize Capital Allocation of Funds?

rename = {'a':'Maintain historic levels of dividends',
          'b':'Increase dividend per share',
          'c':'Repurchasing shares',
          'd':'Fund existing capital spending',
          'e':'Fund new capital spending',
          'f':'Fund R&D',
          'g':'Acquisitions',
          'h':'Pay down debt',
          'i':'Increase cash holdings'}

rename2 = {'q34{}_wave2'.format(i):rename[i] for i in list(rename.keys())}

cols = list(rename.values())

cap = sdata_20[[x for x in sdata_20 if 'q34' in x and 'other' not in x and x!='q34j_wave2']].replace(5,np.nan).dropna(how = 'all').rename(columns = rename2)

cap = cap.join(sdata_20[['id_specific_2020']+demo_vars_20+['q35a_wave2','q36a_wave2']])

rename_div_rep = {'q35a_wave2':'Pay dividends',
                  'q36a_wave2':'Repurchase shares'}

for var in ['q35a_wave2','q36a_wave2']:
    cap[var] = cap[var].map({1:'Yes',2:'No',np.nan:np.nan})

cap = cap.rename(columns = rename_div_rep)


## Fix data for figure and table - if firms say paying dividends or share repurchases are important,
## label them as dividend payers or share repurchasers
cap.loc[(cap['Maintain historic levels of dividends']>=3) & (cap['Maintain historic levels of dividends']<5),
        'Pay dividends'] = 'Yes'
cap.loc[(cap['Increase dividend per share']>=3) & (cap['Increase dividend per share']<5),
        'Pay dividends'] = 'Yes'
cap.loc[(cap['Maintain historic levels of dividends']>=3) & (cap['Maintain historic levels of dividends']<5),
        'Pay dividends'] = 'Yes'
cap.loc[(cap['Repurchasing shares']>=3) & (cap['Repurchasing shares']<5),
        'Repurchase shares'] = 'Yes'




def get_percent_important(pay):
    for col in cols:
        hold = pay[col].value_counts().divide(len(pay)).to_frame().sort_index().rename(index = rename2).T
        
        if col == cols[0]:
            out = hold
        else:
            out = pd.concat([out,hold],axis=0)
    out = out.replace(np.nan,0)
    out = out.rename(columns = {1:'Unimportant',
                                        2:'<Meh',
                                        3:'Important',
                                        4:'Very Important'})
    out['sort'] = out['Important'] + out['Very Important']
    out = out.sort_values(by = 'sort')
    out = out[['Important','Very Important']]
    return out

cap_div = get_percent_important(cap.loc[cap['Pay dividends'] == "Yes"])

cap_rep = get_percent_important(cap.loc[cap['Repurchase shares'] == "Yes"])





def payout_explanations(data,second_dict,xlabels,fig_size = (9.5,5.5),
                           total_width=0.8, single_width=0.8, legend=True, 
                           alph = 1,legend_loc = 'best',ncols = 2,fontsize = 12):
    
    

    import numpy as np
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
               ['Important','Top Priority'],
               ncol = 1,loc = 'center',
               bbox_to_anchor = (0.35,-0.025),fontsize = fontsize,title = r'$\bf{Dividend}$ $\bf{Payers}$',title_fontsize = fontsize)
    fig.legend([bars[2],bars[0]], 
               ['Important','Top Priority'],
               ncol = 1,loc = 'center',
               bbox_to_anchor = (0.55,-0.025),fontsize = fontsize,title = r'$\bf{Repurchasers}$',title_fontsize = fontsize)
    return fig,ax


cap_div = cap_div.reindex([x for x in cap_div.index if x not in 'Maintain historic levels of dividends'] + \
                              ['Maintain historic levels of dividends'])

cap_div['sort'] = cap_div['Important'] + cap_div['Very Important']
cap_rep['sort'] = cap_rep['Important'] + cap_rep['Very Important']

data = pd.concat([cap_rep['sort'],cap_div['sort']],axis=1)
data.columns = ['Repurchases','Dividends']
data = data.sort_values(by = ['Dividends'])
data = data.reindex([x for x in data.index if x not in 'Maintain historic levels of dividends'] + \
                              ['Maintain historic levels of dividends'])


    
data_hold = copy.deepcopy(data)
data_hold = data_hold.reindex([x for x in data_hold.index if x not in 'Maintain historic levels of dividends'] + \
                              ['Maintain historic levels of dividends'])
data = data.to_dict('list')
second_dict = pd.concat([cap_rep['Important'],cap_div['Important']],axis=1)
second_dict.columns = ['Repurchases','Dividends']
second_dict = second_dict.reindex(data_hold.index)
second_dict = second_dict.to_dict('list')
xlabels = data_hold.index.tolist()

print("\n\n\n")
print("Displaying Figure 21:")
fig,ax = payout_explanations(data,second_dict,xlabels,fontsize = 12,fig_size = (8.5,5))
plt.show()

#%% Table IX Panel B
## Testing function for Table IX
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

        # contingency = pd.crosstab(data_test[demo_var],data_test[var])
        # c, p, dof, expected = chi2_contingency(contingency) 
        
        hold = data_test.groupby(demo_var)[var].mean().to_frame().T
        for col in hold.columns:
            hold[col] = hold[col].map('{:.2f}'.format).replace({'nan':''})
        sort = [x for x in hold.columns if '*' not in x] + [x for x in hold.columns if '*' in x]    
        if len(reg_df[demo_var].unique().tolist()) == 1:
            p = 1
    
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

div_drivers = cap.loc[cap['Pay dividends'] == "Yes"].dropna(how = 'all',subset = cols)

rep_drivers = cap.loc[cap['Repurchase shares'] == "Yes"].dropna(how = 'all',subset = cols)


div_drivers['Payout'] = 'Dividends*'
rep_drivers['Payout'] = 'Repurchases'

div_drivers = pd.concat([div_drivers,sdata_20[demo_vars_20]],axis=1,join='inner').reset_index(drop=True)

rep_drivers = pd.concat([rep_drivers,sdata_20[demo_vars_20]],axis=1,join='inner').reset_index(drop=True)


data_in = div_drivers.append(rep_drivers)

cols_in = cap_div.index.tolist()
cols_in.reverse()


data_in = data_in.loc[:,~data_in.columns.duplicated()]


## Test importance across repurchasers and dividend payers
demo_var = 'Payout'

## All firms
test_cap = test_div_imp(data_in,demo_var,cols_in)

## Public firms
test_cap_public = test_div_imp(data_in.loc[data_in['Public'] == 'Yes*'],demo_var,cols_in)

## Large firms
test_cap_large = test_div_imp(data_in.loc[data_in['Size'] == 'Large*'],demo_var,cols_in)

## Small firms
test_cap_small = test_div_imp(data_in.loc[data_in['Size'] == 'Small'],demo_var,cols_in)

## Format the table 
test_cap_out = copy.deepcopy(test_cap)
test_cap_out_public = copy.deepcopy(test_cap_public)
test_cap_out_small = copy.deepcopy(test_cap_small)
test_cap_out_large = copy.deepcopy(test_cap_large)

cap_div_out = cap_div.rename(columns = {'sort':('Percent Agree or Strongly Agree','Dividends')})[[('Percent Agree or Strongly Agree','Dividends')]]
cap_rep_out = cap_rep.rename(columns = {'sort':('Percent Agree or Strongly Agree','Repurchases')})[[('Percent Agree or Strongly Agree','Repurchases')]]

n_list = [len(rep_drivers),len(div_drivers)]

cap_out = pd.concat([cap_rep_out,cap_div_out],axis=1)

cap_out.columns = ['Rep.','Div.']

cap_out = cap_out.sort_values(by = ['Div.'],ascending=False)

cap_out = cap_out.reindex(['Maintain historic levels of dividends'] + \
                          [x for x in cap_out.index if x not in 'Maintain historic levels of dividends']
                          )


cap_out.columns = pd.MultiIndex.from_tuples(
                  [('Panel B: How Do Companies Prioritize Capital Allocation of Funds?',
                    'Percent Important or Top Priority',
                    'All Firms',
                    x,n_list[i]) for i,x in enumerate(cap_out.columns)],
                  names = ['Panel','Variable','Sample','Payout Type','N'])

for col in cap_out.columns:
    cap_out[col] = ((cap_out[col])*100).map('{:.2f}'.format).replace({'nan':''})


test_cap_out.columns = pd.MultiIndex.from_tuples(
                [('Panel B: How Do Companies Prioritize Capital Allocation of Funds?','Sub-Sample Comparisons','All Firms',x[1],x[2]) \
                for x in test_cap_out.columns],
                names = ['Panel','Variable','Sample','Payout Type','N'])
    
test_cap_out_public.columns = pd.MultiIndex.from_tuples(
                [('Panel B: How Do Companies Prioritize Capital Allocation of Funds?','Sub-Sample Comparisons','Public Firms',x[1],x[2]) \
                for x in test_cap_out_public.columns],
                names = ['Panel','Variable','Sample','Payout Type','N'])
    
test_cap_out_large.columns = pd.MultiIndex.from_tuples(
                [('Panel B: How Do Companies Prioritize Capital Allocation of Funds?','Sub-Sample Comparisons','Large Firms',x[1],x[2]) \
                for x in test_cap_out_large.columns],
                names = ['Panel','Variable','Sample','Payout Type','N'])
   
test_cap_out_small.columns = pd.MultiIndex.from_tuples(
                [('Panel B: How Do Companies Prioritize Capital Allocation of Funds?','Sub-Sample Comparisons','Small Firms',x[1],x[2]) \
                for x in test_cap_out_small.columns],
                names = ['Panel','Variable','Sample','Payout Type','N'])
    
panel_b = pd.concat([cap_out,test_cap_out,test_cap_out_public,
                      test_cap_out_large,test_cap_out_small],axis=1)

panel_b = panel_b.reindex(cols_in)

panel_b.index = [x.replace('\n',' ') for x in panel_b.index]


new_cols = [tuple(y.replace("Dividends","Div.").\
            replace("Repurchases","Rep.") if isinstance(y,str) \
            else y for y in x) \
            for x in panel_b.columns.tolist()]

panel_b.columns = pd.MultiIndex.from_tuples(
        new_cols,
        names = list(panel_b.columns.names))

    
## Print results
print('\n\n\n')
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print("Displaying Table IX Panel B:")
    print(panel_b)

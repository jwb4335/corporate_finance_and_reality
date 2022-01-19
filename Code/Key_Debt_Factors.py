
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

## Load in the survey data
filename = 'Data/datafile.pkl'
with open(filename, 'rb') as f:
    demo_vars_19,demo_vars_20,demo_vars_20_only,demo_var_list,sdata,sdata_20,sdata_2001 = pickle.load(f)
    f.close()

sdata['all'] = 'All'
sdata_2001['Size'] = sdata_2001['large'].map({0:'Small',1:'Large*'})


#%% Figure 16. Which Factors Drive Debt Decisions?


def key_debt_factors(sdata,sdata_2001,limit_credit_rating):
               
    kdf = sdata[[col for col in sdata if col.startswith(('q7_'))]+['credit_rating']]
    kdf = kdf.rename(columns = {'credit_rating':'credit_rating_letter'})
    kdf = kdf.loc[:,~kdf.columns.duplicated()]

    del kdf['q7_other_string']
    
    kdf = kdf.dropna(subset = [col for col in kdf.columns if col.startswith(('q7_'))],how = 'all')    
    list_2019 = ['financial_flexibility','credit_rating', 'volatility_earnings',
            'insufficient_internal_funds','interest_rate_levels',
            'tax_advantage_interest_ded','debt_transaction_costs',
            'equity_valuation','debt_firms_ind','costs_bankruptcy',
            'customer_supplier_debt','amount_collateral_borrow',
            'investor_interest_tax_cost']
    
                
    index_dict = {'financial_flexibility': 'Financial Flexibility',
                  'credit_rating': 'Credit Rating',
                  'volatility_earnings':'Earnings and Cash Flow Volatility',
                  'insufficient_internal_funds': 'Insufficient Internal Funds',
                  'interest_rate_levels':'Level of Interest Rates',
                   'tax_advantage_interest_ded':'Interest Tax Savings',
                   'debt_transaction_costs':'Transaction Costs and Fees',
                   'equity_valuation':'Equity Under/Over-Valuation',
                   'debt_firms_ind':'Comparable Firm Debt Levels',
                   'costs_bankruptcy':'Bankruptcy/Distress Costs',
                   'customer_supplier_debt':'Customer/Supplier Comfort',
                   'amount_collateral_borrow':'Amount of Collateral Available (\'22 only)',
                   'investor_interest_tax_cost':'Investor Interest and Tax Costs'}

    for var in list_2019:
        kdf.loc[kdf['q7_{}'.format(var)]>= 4,'{}_dum'.format(var)] = 1
        kdf.loc[kdf['q7_{}'.format(var)]<  4,'{}_dum'.format(var)] = 0
        kdf['{}'.format(var)] =  kdf['{}_dum'.format(var)]
        kdf = kdf[[x for x in kdf if '_dum' not in x]]

    kdf = kdf[list(index_dict.keys())]


    list_1999 =  ['financial_flexibility',
    'credit_rating',
    'volatility_earnings',
    'insufficient_internal_funds',        
    'interest_rate_levels',
    'tax_advantage_interest_ded',
    'equity_valuation',
    'debt_firms_ind',
    'customer_supplier_debt',
    'costs_bankruptcy',
    'debt_transaction_costs',
    'investor_interest_tax_cost']
    
    rename_1999 = {'Q12G':'financial_flexibility',
                   'Q12D':'credit_rating',
                   'Q12H':'volatility_earnings',
                   'Q13A':'insufficient_internal_funds',
                   'Q13C':'interest_rate_levels',
                   'Q12A':'tax_advantage_interest_ded',
                   'Q13D':'equity_valuation',
                   'Q12C':'debt_firms_ind',
                   'Q12I':'customer_supplier_debt',
                   'Q12B':'costs_bankruptcy',
                   'Q12E':'debt_transaction_costs',
                   'Q12F':'investor_interest_tax_cost'}
    
    kdf_2001 = sdata_2001[['Q12G','Q12D','Q12H','Q13A','Q13C','Q12A',
                           'Q13D','Q12C','Q12I','Q12B','Q12E','Q12F']+['credit_rating']]
    kdf_2001 = kdf_2001.rename(columns = {'credit_rating':'credit_rating_letter'})

    kdf_2001 = kdf_2001.rename(columns = rename_1999)
    
    kdf_2001 = kdf_2001.loc[:,~kdf_2001.columns.duplicated()]

    for var in list_1999:
        kdf_2001.loc[kdf_2001['{}'.format(var)] == 9,'{}'.format(var)] = np.nan
        kdf_2001.loc[kdf_2001['{}'.format(var)]>= 3,'{}_dum'.format(var)] = 1
        kdf_2001.loc[kdf_2001['{}'.format(var)]<  3,'{}_dum'.format(var)] = 0
        kdf_2001[var] = kdf_2001['{}_dum'.format(var)]
        kdf_2001 = kdf_2001[[x for x in kdf_2001 if '_dum' not in x]]


    if limit_credit_rating == "Yes":
        kdf.loc[kdf['credit_rating_letter'] == "NA",'credit_rating'] =np.nan
        kdf_2001.loc[kdf_2001['credit_rating_letter'] == "NA",'credit_rating'] = np.nan

    kdf = kdf[list_2019].mean().to_frame().rename(columns = {0:'2022 Survey'})
        
    kdf_2001 = kdf_2001[list_1999].mean().to_frame().rename(columns = {0:'2001 Survey'})
    
    
    out = pd.concat([kdf,kdf_2001],axis=1).sort_values(by = '2022 Survey')
    out = out.rename(index = index_dict)

    return out



def crosshatch_large_small(data,second_dict,xlabels,fig_size = (9.5,3.5),
                           total_width=0.8, single_width=1, legend=True, 
                           alph = 1,legend_loc = 'best',ncols = 2):
    
    
    from PIL import ImageColor
    import numpy as np
    import matplotlib.pyplot as plt
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
    ax.set_yticklabels(xlabels,fontsize = 12)
    ax.set_axisbelow(True)
    ax.grid(axis='x',alpha=0.5,linestyle='--',zorder=-100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    from matplotlib.ticker import FuncFormatter
    ax.set_xticks(np.arange(0,1,0.2))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
    
    fig = plt.gcf()
    # Draw legend
    fig.legend([bars[1],bars[3]], ['Large','Small'],
               ncol = 2,loc = 'center',bbox_to_anchor = (0.35,0.025),fontsize = 11,
               title = r'$\bf{2022}$',title_fontsize = 12)

    fig.legend([bars[0],bars[2]], ['Large','Small'],
               ncol = 2,loc = 'center',bbox_to_anchor = (0.7,0.025),fontsize = 11,
               title = r'$\bf{2001}$',title_fontsize = 12)
    return fig,ax


kdf_pct_small = key_debt_factors(sdata.loc[sdata['Size']=='Small'],sdata_2001.loc[sdata_2001['Size'] == 'Small'],limit_credit_rating = "No")
small = kdf_pct_small[['2001 Survey','2022 Survey']].rename(columns = {'2001 Survey':'2001 Small','2022 Survey':'2022 Small'})

kdf_pct_large = key_debt_factors(sdata.loc[sdata['Size']=='Large*'],sdata_2001.loc[sdata_2001['Size'] == 'Large*'],limit_credit_rating = "No")
large = kdf_pct_large[['2001 Survey','2022 Survey']].rename(columns = {'2001 Survey':'2001 Large','2022 Survey':'2022 Large'}).sort_values(by = '2022 Large')

data = large.to_dict('list')
second_dict = small.reindex(large.index.tolist()).to_dict('list')


xlabels = [x for x in large.index.tolist()]


print("\n\n\n")
print("Displaying Figure 16:")
fig,ax = crosshatch_large_small(data,second_dict,xlabels,fig_size = (6.5,6),
                                total_width = 0.8,single_width = 0.9,
                                legend=True,alph=1,ncols = 4)
plt.show()


#%% Table VII. Which Factors Drive Debt Decisions?


def kdf_fix(sdata_20,sdata_2001):
    
    list_2019 = ['financial_flexibility','credit_rating', 'volatility_earnings',
               'insufficient_internal_funds','interest_rate_levels',
               'tax_advantage_interest_ded','debt_transaction_costs',
               'equity_valuation','debt_firms_ind','costs_bankruptcy',
               'customer_supplier_debt','amount_collateral_borrow',
               'investor_interest_tax_cost']
        
                    
    index_dict = {'financial_flexibility': 'Financial Flexibility',
                  'credit_rating': 'Credit Rating',
                  'volatility_earnings':'Earnings and Cash Flow Volatility',
                  'insufficient_internal_funds': 'Insufficient Internal Funds',
                  'interest_rate_levels':'Level of Interest Rates',
                   'tax_advantage_interest_ded':'Interest Tax Savings',
                   'debt_transaction_costs':'Transaction Costs and Fees',
                   'equity_valuation':'Equity Under/Over-Valuation',
                   'debt_firms_ind':'Comparable Firm Debt Levels',
                   'costs_bankruptcy':'Bankruptcy/Distress Costs',
                   'customer_supplier_debt':'Customer/Supplier Comfort',
                   'amount_collateral_borrow':'Amount of Collateral Available (\'22 only)',
                   'investor_interest_tax_cost':'Investor Interest and Tax Costs'}
    
    
    kdf_19 = sdata[[x for x in sdata if "q7_" in x and 'other' not in x] + ['id_specific','Size']]
    
    kdf_19.columns = [x.replace("q7_","") for x in kdf_19.columns]
    
    kdf_19 = kdf_19.dropna(subset = list_2019,how = 'all')
    
    for col in list_2019:
        kdf_19[col] = kdf_19[col]-1
        
    list_1999 =  ['financial_flexibility',
    'credit_rating',
    'volatility_earnings',
    'insufficient_internal_funds',        
    'interest_rate_levels',
    'tax_advantage_interest_ded',
    'equity_valuation',
    'debt_firms_ind',
    'customer_supplier_debt',
    'costs_bankruptcy',
    'debt_transaction_costs',
    'investor_interest_tax_cost']
    
    rename_1999 = {'Q12G':'financial_flexibility',
                   'Q12D':'credit_rating',
                   'Q12H':'volatility_earnings',
                   'Q13A':'insufficient_internal_funds',
                   'Q13C':'interest_rate_levels',
                   'Q12A':'tax_advantage_interest_ded',
                   'Q13D':'equity_valuation',
                   'Q12C':'debt_firms_ind',
                   'Q12I':'customer_supplier_debt',
                   'Q12B':'costs_bankruptcy',
                   'Q12E':'debt_transaction_costs',
                   'Q12F':'investor_interest_tax_cost'}
        
    kdf_00 = sdata_2001[list(rename_1999.keys())+['Size']].rename(columns = rename_1999)
    
    kdf_00 = kdf_00.replace(9,np.nan)
    
    kdf_00 = kdf_00.dropna(subset = list_1999,how = 'all').dropna(subset = ['Size'],how = 'all')

    kdf_19['Survey'] = '2022*'
    
    kdf_00['Survey'] = '2001'

    kdf_19 = kdf_19.rename(columns = index_dict)
    kdf_00 = kdf_00.rename(columns = index_dict)
    
    kdf_cols = list(index_dict.values())
    
    kdf_out = kdf_00.append(kdf_19)

    for col in kdf_cols:
        kdf_out.loc[kdf_out[col]>=3, '{}_dum'.format(col)] = 1
        kdf_out.loc[kdf_out[col]<3, '{}_dum'.format(col)]  = 0

    
        
    return kdf_out, kdf_cols

def test_kdf(data,demo_var,kdf_cols):

    import statsmodels.formula.api as sm

    data = data.loc[~pd.isnull(data[demo_var])]
    
    sort = [x for x in data[demo_var].unique().tolist() if '*' not in x] +\
               [x for x in data[demo_var].unique().tolist() if '*' in x]    

    for var in kdf_cols:
        reg_df = data.loc[(~pd.isnull(data[demo_var])) & (~pd.isnull(data[var]))][[demo_var,var]]    
        reg_df.loc[reg_df[demo_var]==sort[1],'dummy'] = 1
        reg_df.loc[reg_df[demo_var]==sort[0],'dummy'] = 0
        reg_df['yvar'] = reg_df[var]
        p = sm.ols(formula="yvar ~ dummy", data=reg_df).fit().pvalues.dummy

        
        hold = data.groupby(demo_var)[var].mean().to_frame().T
        for col in hold.columns:
            hold[col] = hold[col].map('{:.2f}'.format).replace({'nan':''})
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
        if var == kdf_cols[0]:
            out = hold
        else:
            out = pd.concat([out,hold],axis = 0)
    out = out[sort]
    
    l1 = [demo_var]*2
    l2 = [x.replace('*','') for x in sort]
    l3 = data.groupby(demo_var)[demo_var].count()[sort].values.tolist()
    tuples= [(l1[i],l2[i],l3[i]) for i in range(len(l1))]
    out.columns = pd.MultiIndex.from_tuples(tuples,names = ['Variable','Group','N'])
    
    out = out.reindex(new_index)
    return out


kdf_out,kdf_cols = kdf_fix(sdata,sdata_2001)

# kdf_out['Size'] = kdf_out['large'].map({0:'Small',1:'Large*'})

demo_var = 'Size'

data = kdf_out.dropna(subset = kdf_cols,how = 'all')

kdf_dum_cols = [x+"_dum" for x in kdf_cols]

percentages = kdf_out.groupby(['Size','Survey'])[kdf_dum_cols].mean().rename(columns = {x+"_dum":x for x in kdf_cols}).T

percentages = percentages.sort_values(by = [('Large*','2022*')],ascending=False).reindex(
              [x for x in percentages.index if 'Collateral' not in x] + ['Amount of Collateral Available (\'22 only)'])

n_list = kdf_out.groupby(['Size','Survey'])['Survey'].count().tolist()

percentages.columns = pd.MultiIndex.from_tuples([('Large Firms','2001'),
                       ('Large Firms','2022'),
                       ('Small Firms','2001'),
                       ('Small Firms','2022')],names = ['Group','Survey'])

percentages.columns = pd.MultiIndex.from_tuples(
    [('Panel A: Key Debt Factors, 2001 vs. 2022 Comparison','Percent Important or Very Important',) + x + (n_list[i],) for i,x in enumerate(percentages.columns)],
    names = ['Panel','Variable','Group','Survey','N'])

for col in percentages.columns:
    percentages[col] = (percentages[col]*100).map('{:.2f}'.format).replace({'nan':''})


new_index = percentages.index.tolist()


kdf_00_20 = test_kdf(data,'Survey',kdf_cols)
kdf_00_20.columns = pd.MultiIndex.from_tuples([('Full Sample',x[1],x[2]) for x in kdf_00_20.columns],
                                               names = ['Sample','Survey','N'])

kdf_00_20_large = test_kdf(data.loc[data['Size'] == 'Large*'],'Survey',kdf_cols)
kdf_00_20_large.columns = pd.MultiIndex.from_tuples([('Large Firms',x[1],x[2]) for x in kdf_00_20_large.columns],
                                               names = ['Sample','Survey','N'])

kdf_00_20_small = test_kdf(data.loc[data['Size'] == 'Small'],'Survey',kdf_cols)
kdf_00_20_small.columns = pd.MultiIndex.from_tuples([('Small Firms',x[1],x[2]) for x in kdf_00_20_small.columns],
                                               names = ['Sample','Survey','N'])

compare = pd.concat([kdf_00_20,kdf_00_20_large,kdf_00_20_small],axis=1)

compare.columns = pd.MultiIndex.from_tuples(
    [('Panel A: 2001 vs. 2022 Comparison','Score',) + x for x in compare.columns],
    names = ['Panel','Variable','Group','Survey','N'])

panel_a = pd.concat([percentages,compare],axis=1)

new_sort = large.index.tolist()

new_sort.reverse()

panel_a = panel_a.reindex(new_sort)



data = data.rename(columns = {x:"q7_{}".format(x) for x in kdf_cols})

data_20 = data.loc[data['Survey'] == '2022*'][[x for x in data if x not in 'Size']].\
          merge(sdata[['id_specific']+demo_vars_19].rename(columns = {'Financial Flexibility':'Financial Flexibility_DEMO'}),on = 'id_specific')

new_cols = ["q7_{}".format(x) for x in kdf_cols]


demo_vars_out = ['Size','Public','Growth Prospects','Pay Dividends','Leverage','Cash','Financial Flexibility_DEMO']


data_20 = data_20.rename(columns =  {"q7_{}".format(x):x for x in kdf_cols})


data_20 = data_20.drop_duplicates(subset = ['id_specific']).merge(data.loc[data['Survey'] == '2022*']['id_specific'],on = ['id_specific'])


for demo_var in demo_vars_out:
    
    hold = test_kdf(data_20,demo_var,kdf_cols)
    hold = hold.rename(index =  {"q7_{}".format(x):x for x in kdf_cols})
    if demo_var == demo_vars_out[0]:
        kdf_table = hold
    else:
        kdf_table = pd.concat([kdf_table,hold],axis=1)

kdf_table.columns = pd.MultiIndex.from_tuples(
    [('Panel B: Key Debt Factors, Conditional on Company Characteristics',x[0].replace('_DEMO',''),x[1],x[2]) for x in kdf_table.columns],
    names = ['Panel','Variable','Group','N'])

panel_b = kdf_table

panel_b = panel_b.reindex(new_sort)

panel_b = panel_b.rename(index = {
    x:x.replace(" (\'22 only)","") for x in panel_b.index})



print('\n\n\n')
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print("Displaying Table VII Panel A:")
    print(panel_a)
    print("\n\n\n")
    print("Displaying Table VII Panel B:")
    print(panel_b)   

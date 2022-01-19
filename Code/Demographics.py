"""
Created on Sat Jan  8 12:59:50 2022

jwb
"""


"""
Created on Sat Jan  8 12:27:19 2022

jwb
"""



import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import Functions.stacked_bar as stacked_bar

## Load in the survey data
filename = 'Data/datafile.pkl'
with open(filename, 'rb') as f:
    demo_vars_19,demo_vars_20,demo_vars_20_only,demo_var_list,sdata,sdata_20,sdata_2001 = pickle.load(f)
    f.close()


## Load in the compustat data from 2019 on revenue and employment
filename = 'Data/compustat_data_2019.pkl'
with open(filename, 'rb') as f:
    [comp_data] = pickle.load(f)
    f.close()



#%%

def demo_counts(sdata,demo_var,rename = None):
    
    out = sdata[demo_var].value_counts(normalize = True).sort_index()
    
    if rename is not None:
        out = out.rename(index = rename)
    
    out = out.to_frame()
    out.columns = [demo_var]
    return out

revenue_recode = {1:'<$25m',
                  2:'$25-99m',
                  3:'$100-499m',
                  4:'$500-999m',
                  5:'$1-5bn',
                  6:'>$5bn'}

ownership_recode = {1:'Public',2:'Private',3:'Government',4:'Nonprofit'}

foreign_sales_recode = {1:'0%',2:'1-24%',3:'25-50%',4:'≥50%'}


age_code = [[0,10],[10,50],[50,100],[100,1e10]]
firm_age_rename = {0:'≤10',1:'11-49',2:'50-99',3:'≥100'}

for i in np.arange(len(age_code)):
    sdata_20.loc[(sdata_20['firmage']>=age_code[i][0]) &\
                 (sdata_20['firmage']< age_code[i][1]),
                 'firmage_bin'] = firm_age_rename[i]
        
        
age_rename = {1:'≤39',2:'40-49',3:'50-59',4:'≥60'}





tenure_rename = {1:'<4 Years',2:'4-9 Years',3:'≥10 Years'}

education_rename = {1:'High School',2:'UG',3:'MBA',4:'PG (non-MBA)',5:'PG (non-MBA)'}

employee_rename = {1:'<100',2:'<100',3:'100-499',4:'500-999',5:'1,000-4,999',
                   6:'5,000-10,000',7:'≥10,000'}

segment_rename = {1:'1',2:'2',3:'3',4:'4',5:'≥5'}

industry_rename = {'Agri/Mining/Construction':'Agri',
 'Financial':'Finance',
 'Manufacturing':'Manuf',
 'Retail/Wholesale':'Ret/Whole',
 'Services/Comms/Media':'Services',
 'Tech/Healthcare':'Tech/Health',
 'Transport/Utils/Energy':'Trans/Energy'}


hq_rename = {1:'Northeast',
             2:'Mountain',
             3:'Midwest',
             4:'South',
             6:'Pacific',
             7:'Canada'}
demo_var = 'revenue'
rename = revenue_recode
sales = demo_counts(sdata,demo_var,rename)

demo_var = 'numemployees'
rename = employee_rename
sdata.loc[sdata['numemployees']==1,'numemployees'] = 2
employees = demo_counts(sdata,demo_var,rename)

demo_var = 'ownership'
rename = ownership_recode
ownership = demo_counts(sdata,demo_var,rename)

demo_var = 'foreignsales'
rename = foreign_sales_recode
foreign_sales = demo_counts(sdata,demo_var,rename)

demo_var = 'industry_2001_named'
rename = industry_rename
industry = demo_counts(sdata.loc[sdata[demo_var]!='Other'],demo_var,rename)


demo_var = 'credit_rating_agg'
sdata[demo_var] = sdata['credit_rating'].str.replace("+","",regex=False).str.replace("-","",regex=False)
sdata.loc[(sdata[demo_var]=='CCC') | (sdata[demo_var]=='CC') | (sdata[demo_var]== 'D'),demo_var] = 'C/D'
credit_rating = demo_counts(sdata.loc[sdata[demo_var]!='NA'],demo_var)
credit_rating = credit_rating.reindex(['C/D','B','BB','BBB','A','AA','AAA'])

demo_var = 'age'
rename = age_rename
age = demo_counts(sdata_20,demo_var,rename)

demo_var = 'tenure'
rename = tenure_rename
tenure = demo_counts(sdata_20,demo_var,rename)

demo_var = 'education'
rename = education_rename
sdata_20.loc[sdata_20['education'] == 5,'education'] = 4
education = demo_counts(sdata_20.loc[sdata_20['education']>1],demo_var,rename)

demo_var = 'operatingsegments'
sdata.loc[sdata[demo_var]>4,demo_var] = 5
sdata.loc[sdata[demo_var] == 0,demo_var] = np.nan
rename = segment_rename
segments = demo_counts(sdata,demo_var,rename)

demo_var = 'headquarters_us'
sdata.loc[sdata[demo_var] == 5,demo_var] = 4
rename = hq_rename
hq = demo_counts(sdata.loc[sdata['headquarters_us']!=12],demo_var,rename)

demo_var_orig = 'q4_current_cash'
bounds = [[0,10],[10,25],[25,50],[50,100]]
labels = ['0-10%','10-25%','25-50%','50-100%']
demo_var = 'cash_assets'
for i in np.arange(len(bounds)):
    sdata.loc[(sdata[demo_var_orig] >=bounds[i][0]) & (sdata[demo_var_orig] < bounds[i][1]),
              demo_var] = labels[i]
cash = demo_counts(sdata,demo_var)

demo_var_orig = 'q6_current_debt_assets'
bounds = [[0,10],[10,25],[25,50],[50,100]]
labels = ['0-10%','10-25%','25-50%','50-100%']
demo_var = 'debt_assets'
for i in np.arange(len(bounds)):
    sdata.loc[(sdata[demo_var_orig] >=bounds[i][0]) & (sdata[demo_var_orig] < bounds[i][1]),
              demo_var] = labels[i]
debt = demo_counts(sdata,demo_var)

demo_var = 'firmage_bin'
firm_age = demo_counts(sdata_20,demo_var)
firm_age = firm_age.reindex(['≤10','11-49', '50-99', '≥100'])
demo_var = 'Family Firm'
family_firm = demo_counts(sdata_20,demo_var)
family_firm = family_firm.rename(index = {x:x.replace('*','') for x in family_firm.index})


from matplotlib.ticker import FuncFormatter
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
colors[0] = '#4472C4'

def plot_sub(data,spnumber,plot_title):
    
    import string
    alpha_list = list(string.ascii_uppercase)
    
    alpha_num_dict = {i:alpha_list[i-1] for i in np.arange(1,len(alpha_list)+1,1)}

    
    plt.subplot(nrows_plot,ncols_plot,spnumber)
    ax = plt.gca()
    bars = stacked_bar.barh_plot(ax, data.to_dict('list'), total_width=.8, single_width=.9,alph=0.9,colors = colors,legend_loc = 'lower left',ncols = 1,second_dict = None) 
    plt.legend(bars, data.index.tolist(),bbox_to_anchor = (0.5,-0.15),ncol = 3,loc='center')
    ax.set_yticks(np.arange(len(data)))
    ax.set_yticklabels(data.index.tolist(),fontsize = 8)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim((0,0.52))
    ax.set_xticks(np.arange(0,0.75,0.25))
    ax.set_xticklabels(ax.get_xticklabels(),fontsize = 9)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
    ax.set_title(plot_title,{'fontname':'Times New Roman'},fontsize=11)
    ax.grid(axis='x',alpha=0.5,linestyle='--')
    ax.set_axisbelow(True)
    ax.set_title('({}) {}'.format(alpha_num_dict[spnumber],plot_title))
    ax.get_legend().remove()
    
    
nrows_plot = 1
ncols_plot = 3
print("\n\n\n")
print("Displaying Figure 1:")
plot_sub(sales,1,'Sales Revenue')
# plot_sub(employees,2,'(B) Employees')
plot_sub(ownership,2,'Ownership')
plot_sub(industry,3,'Industry')
fig = plt.gcf()
fig.set_size_inches((6.5,2.5))
plt.subplots_adjust(hspace = 0.1,wspace = 0.75)
plt.tight_layout()
plt.show()



nrows_plot = 5
ncols_plot = 3

plot_sub(sales,1,'Sales Revenue')
plot_sub(ownership,2,'Ownership')
plot_sub(industry,3,'Industry')
plot_sub(employees,4,'Employees')
plot_sub(cash,5,'Cash/Assets')
plot_sub(debt,6,'Debt/Assets')
plot_sub(foreign_sales,7,'Foreign Sales')
plot_sub(credit_rating,8,'Credit Rating')
plot_sub(segments,9,'Operating Segments')
plot_sub(hq,10,'Headquarters')
plot_sub(firm_age,11,'Firm Age')
plot_sub(family_firm,12,'Family Firm')
plot_sub(age,13,'CFO Age')
plot_sub(tenure,14,'CFO Tenure')
plot_sub(education,15,'CFO Education')

print("\n\n\n")
print("Displaying Figure A2.1:")
fig = plt.gcf()
fig.set_size_inches((6.5,7.5))
plt.subplots_adjust(hspace = 0.5,wspace = 0.85)
plt.tight_layout()
plt.show()

#%% TABLE A2.I: Survey Demographic Summary Statistics and Compustat Comparison


## Define revenue and employment bins
rev_min = {1:0,2:25,3:100,4:500,5:1000,6:5000,7:10000}
rev_max = {1:25,2:100,3:500,4:1000,5:5000,6:10000,7:10000000}

rev_min_max = [[0,25],[25,100],[100,500],[500,1000],[1000,5000],[5000,10000],[10000,10000000]]

emp_min = {1:0,2:2,3:100,4:500,5:1000,6:5000,7:10000}
emp_max = {1:2,2:100,3:500,4:1000,5:5000,6:10000,7:10000000}

emp_min_max = [[0,2],[2,100],[100,500],[500,1000],[1000,5000],[5000,10000],[10000,10000000]]


## Define the revenue and employment groupings for Compustat firms
for i in np.arange(len(rev_min_max)):
    lb = rev_min_max[i][0]
    ub = rev_min_max[i][1]
    rev_group = i+1
    comp_data.loc[(comp_data['salesrevenue']>=lb) & (comp_data['salesrevenue']<ub),'revenue'] = rev_group

for i in np.arange(len(emp_min_max)):
    lb = emp_min_max[i][0]
    ub = emp_min_max[i][1]
    emp_group = i+1
    comp_data.loc[(comp_data['fulltime']>=lb) & (comp_data['fulltime']<ub),'numemployees'] = emp_group

## This function calculates the quartiles of employment within each specified revenue bin
def get_emps(bounds = [-1,5],label = '$\leq 5m$'):
    s = sdata_20.loc[(sdata_20['salesrevenue'] > bounds[0]) & (sdata_20['salesrevenue'] <= bounds[1])]['fulltime'].describe()[['25%','50%','75%']].to_frame().T
    s['% of Sample'] = len(sdata_20.loc[(sdata_20['salesrevenue'] > bounds[0]) & (sdata_20['salesrevenue'] <= bounds[1])])/len(sdata_20)
    s['% of Sample'] = round(s['% of Sample']*100,1)
    s[''] = ''
    s = s[['% of Sample'] + [x for x in s if x not in ['% of Sample']]]
    
    c = comp_data.loc[(comp_data['salesrevenue'] > bounds[0]) & (comp_data['salesrevenue'] <= bounds[1])]['fulltime'].describe()[['25%','50%','75%']].to_frame().T
    c['% of Sample'] = len(comp_data.loc[(comp_data['salesrevenue'] > bounds[0]) & (comp_data['salesrevenue'] <= bounds[1])])/len(comp_data)
    c['% of Sample'] = round(c['% of Sample']*100,1)
    c = c[['% of Sample'] + [x for x in c if x not in ['% of Sample']]]
    out = pd.concat([s,c],axis=1)
    out.index = [label]
    return out

bounds_list = [[-1,5],[5,25],[25,100],[100,1000],[1000,5000],[5000,1e20]]
labels = ['≤ 5m','5-25m','25-100m','100m-1bn','1-5bn','> 5bn']


## Create the table
for i in np.arange(len(bounds_list)):
    hold = get_emps(bounds = bounds_list[i],label = labels[i])
    if i == 0:
        out = hold
    else: 
        out = pd.concat([out,hold],axis=0)

for col in [x for x in out.columns if not x.startswith('%') and not x in ['']]:
    out[col] = round(out[col],0)


top_cols = ['({})'.format(i) for i in np.arange(1,5,1)] +[''] + ['({})'.format(i) for i in np.arange(5,9,1)]
cur_cols = out.columns.tolist()
new_cols = [(top_cols[i],cur_cols[i]) for i in np.arange(len(cur_cols))]

out.columns = pd.MultiIndex.from_tuples(new_cols,names = ['',''])

## Display table
print("\n\n\n")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print("Displaying Table A2.I:")
    print(out)

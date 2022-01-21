"""
Code to produce Table A8.I
John Barry
2022/01/14
"""
import pandas as pd


reg_data = pd.read_csv("Data/SOA_regressions.csv")

reg_data.columns = [x.lower() for x in reg_data.columns]

period = '1950-1964'

def table(reg_data,period):
    out = reg_data.loc[reg_data['period'] == period][['soa','tp','r2']].describe().T[['mean','std','25%','50%','75%']]

    N = reg_data.loc[reg_data['period'] == period]['period'].count()
    out = out.rename(columns = {'mean':'Mean','std':'Std Dev'},index = {'soa':'Speed of Adjustment','tp':'Target Payout','r2':'Adjusted R-Squared'})
    
    out.columns = pd.MultiIndex.from_tuples(
                  [('{} (N = {})'.format(period,N),x)  for x in out.columns],
                  names = ['Period','Statistic'])
    return out

periods =['1950-1964', '1965-1983', '1984-2002', '2003-2020']

top = pd.concat([table(reg_data,'1950-1964'),table(reg_data,'1965-1983')],axis=1)
bottom = pd.concat([table(reg_data,'1984-2002'),table(reg_data,'2003-2020')],axis=1)

bottom.columns = pd.MultiIndex.from_tuples([(x[0],'') for x in bottom.columns])


## Print results
print('\n\n\n')
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print("Displaying Table A8.1")
    print(top)
    print(bottom)

print("""
Note: see Stata file Lintner_regressions.do and lintner_regressions.log to see analysis for Table A8.I
This code just formats and prints regression results
          """)
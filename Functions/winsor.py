

def winsor(c,low=0.01,up=0.99,trim_replace="replace"):
    import numpy as np
    import pandas as pd
    pd.options.mode.chained_assignment = None  # default='warn'
    bot = np.nanquantile(c,low)
    top =  np.nanquantile(c,up)
    c2 = c
    if trim_replace == 'trim':
       c2.loc[(c2<bot)] = np.nan
       c2.loc[(c2>top)] = np.nan
    elif trim_replace == 'replace':
       c2[(c2<bot) & (np.isnan(c2)==False)] = bot
       c2[(c2>top) & (np.isnan(c2)==False)] = top
        
    return c2

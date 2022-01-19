# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 13:26:53 2020

@author: barry
"""

from matplotlib import pyplot as plt
import seaborn as sns

def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True, alph = 1,legend_loc = 'best',ncols = 2,second_dict = None):
    
    
    
    import seaborn as sns
    import numpy as np
    default_colors = sns.color_palette('bright')
    #order = [2,3,0] + [x for x in np.arange(len(default_colors)) if x not in [2,3,0]]
    #default_colors = [default_colors[i] for i in order]

    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = default_colors

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
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i],alpha=alph)

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])
    if second_dict is not None:
        for i, (name, values) in enumerate(second_dict.items()):
            for x, y in enumerate(values):
                x_offset = x_offset = (i - n_bars / 2) * bar_width + bar_width / 2
                ax.scatter(x+x_offset,y,label = '_nolegend_',color = colors[i],edgecolor = 'black',zorder = 100)
    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys(),loc=legend_loc,ncol = ncols)
    return bars
        
def barh_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True, alph = 1,legend_loc = 'best',ncols = 2,second_dict = None):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """
    
    import seaborn as sns
    import numpy as np
    default_colors = sns.color_palette('bright')
    order = [2,3,0] + [x for x in np.arange(len(default_colors)) if x not in [2,3,0]]
    default_colors = [default_colors[i] for i in order]
    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = default_colors

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
                x_offset = x_offset = (i - n_bars / 2) * bar_width + bar_width / 2
                ax.scatter(x+x_offset,y,label = '_nolegend_',color = colors[i],edgecolor = 'black',zorder = 100)
        

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys(),loc=legend_loc,ncol = ncols)
    return bars

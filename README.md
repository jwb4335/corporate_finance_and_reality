# corporate_finance_and_reality
Codes for "Corporate Finance and Reality" by John R. Graham

This repo has all code and displays results presented in "Corporate Finance and Reality" by John R. Graham.

"Main.ipynb" displays all results. Results are displayed in order of section. Results from appendices connected to each section are also displayed. For example, Section I displays figures/tables from Section I in the text and from Appendix A2.

Results are also displayed in  pdf.

Whenever a figure was created in Python, it is displayed here. Many figures in the paper were finalized in Excel, so we only print the underlying data for those figures.

The underlying code to produce each figure or table are contained in .py files. For example, the file code/Demographics.py produces Figures 1, A2.1 and Table A2.I.

The results here were created using off-the-shelf Anaconda and should run via any relatively recent version of Anaconda and Jupyter. Nevertheless, we provide below all packages (and their versions) used to produce the results.
- Python 3.7.3
- Anaconda Navigator 2.0.3
- Jupyter Notebook 6.4.3
- pandas 1.1.3
- numpy 1.19.2
- matplotlib 3.3.2
- statsmodels 0.12.0
- PIL 8.3.1

There are two user-written functions, winsor.py and stacked_bar.py, located in the subfolder Functions, which are needed to produce the results.

Lastly, results for Table A8.I were produced using Stata Version 16. See "Code/Lintner_Regressions.txt" for details. 

The data used to produce the results in the paper are nearly all proprietary, so is thus not included. 

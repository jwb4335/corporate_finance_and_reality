
cap log close Table_A8I

log using Lintner_Regressions.txt, t replace name(Table_A8I)

version 16

/* Code to run regressions for Table A8.I of Corporate Finance and Reality
 - See caption of Table A8.I for full description of methodology
*/

* lintner's model
*I follow Brav, Graham, Harvey & Michealy (2005) Table 7 for regression; and have Mark's SAS code for reference

clear all
set more off


**edited by Joy 2018-10
**edited by John Barry 2021-12

timer on 1
///////////////////////////////////////////////////////////////////
*1. compustat data: 1950-2020, all firms
import delimited Data/lintner_data.csv,clear
keep gvkey fyear adrr indfmt fic ajex dvpsx_f epspx 



*epspx: earning per share (basic) excluding extra items
*epsfx: earning per share (dilute) excluding extra items
*dvpsx_f: dividend per share-ex-date-fiscal

destring gvkey,replace
drop if fyear==2021

*drop foreign firms
bysort gvkey: egen adrrr=max(adrr)
drop if adrrr!=.
drop if fic!="USA"

duplicates tag gvkey fyear, gen(i)
drop if indfmt=="FS" & i>0
drop if fyear==.
duplicates drop gvkey fyear, force

*******
xtset gvkey fyear
*winsor2 dvpsx_f epspx, replace by(fyear)

*adjustment factor
gen lagAdjFac=l1.ajex
gen AdjFac=ajex
gen div_pershare=dvpsx_f*(lagAdjFac/AdjFac)
gen earning_pershare=epspx*(lagAdjFac/AdjFac)

*important, lag div_share is NOT adjusted
gen l1_div_pershare=l1.dvpsx_f
gen diff_div=div_pershare-l1_div_pershare

gen DE_ratio=div_pershare/earning_pershare

*to get rid of missing values, we only keep the ones with entire series
gen p=(!missing(l1_div_pershare))
gen q=(!missing(earning_pershare))
gen o=(!missing(diff_div))


*******************************************



foreach num of numlist 1/4 {

preserve

if `num'==1 {
local firstyear 1950
local lastyear 1964
local totalyear=`lastyear'-`firstyear'-1
}
if `num'==2 {
local firstyear 1965
local lastyear 1983
local totalyear=`lastyear'-`firstyear'
}
if `num'==3 {
local firstyear 1984
local lastyear 2002
local totalyear=`lastyear'-`firstyear'
}
if `num'==4 {
local firstyear 2003
local lastyear 2020
local totalyear=`lastyear'-`firstyear'
}



keep if fyear<=`lastyear' & fyear>=`firstyear'

by gvkey: egen pp=total(p)
by gvkey: egen qq=total(q)
by gvkey: egen oo=total(o)
drop if pp<`totalyear' | qq<`totalyear' |oo<`totalyear'
** Dataset needs to be saved to run statsby
tempfile hold 
save `hold'
statsby _b e(r2_a), by(gvkey): reg diff_div l1_div_pershare earning_pershare

gen period = ""
replace period = "`firstyear'-`lastyear'"
order period

tempfile myparms_`num'
save `myparms_`num''

restore

}


foreach num of numlist 1/4{
use  `myparms_`num'',clear
rename _eq2_stat_1 R2
rename _b_l1_div_pershare beta_ldps
rename _b_earning_pershare beta_eps
rename _b_cons beta_cons
keep gvkey  beta* R2 period

gen TP=-beta_eps/beta_ldps
gen SOA=-beta_ldps

winsor2 TP SOA R2, replace cuts(1 99) trim

order period
gen index=`num'
tempfile temp_`num'
save `temp_`num''
}


use `temp_1',clear
foreach num of numlist 2/4{

append using `temp_`num''
}
export delimited using "Data/SOA_regressions.csv", replace 

log close Table_A8I
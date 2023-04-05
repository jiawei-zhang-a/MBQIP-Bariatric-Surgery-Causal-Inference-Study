import pandas as pd
import numpy as np

import os
#load
maindata = pd.DataFrame(pd.read_csv('../../dat/mbqip/all_bmi_main_data.csv'))

#create the colomn with difference between the BMI and BMI_DISCH
maindata['diff_BMI'] = maindata['BMI'] - maindata['BMI_DISCH']
maindata = maindata.drop(["BMI","BMI_DISCH"],axis=1)

#scale the maindata
standard_scaler = lambda x : (x-np.mean(x))/np.std(x)
maindata['AGE'] = maindata[['AGE']].apply(standard_scaler)
#maindata['diff_BMI'] = maindata[['diff_BMI']].apply(standard_scaler)

#extract five catogry
Sleeve = maindata.dropna(subset = ['Sleeve']) #treated as t = 0

RYGB = maindata.dropna(subset = ['RYGB'])
Band = maindata.dropna(subset = ['Band'])
BPD_DS = maindata.dropna(subset = ['BPD/DS'])
SADI_S = maindata.dropna(subset = ['SADI-S'])

os.mkdir("result")
def analysis(data,name):
    maindata_T = data.describe(include = 'all').T
    maindata_T["num_nan"] = maindata.isnull().sum(axis = 0)
    maindata_T["num_distinct"] = maindata.nunique(axis = 0)
    maindata_T.head()
    maindata_T.to_csv("result/post_clean_stats_analysis_"+name+".csv")

analysis(Sleeve,"Sleeve")
analysis(RYGB,"RYGB")
analysis(Band,"Band")
analysis(BPD_DS,"BPD_DS")
analysis(SADI_S,"SADI_S")

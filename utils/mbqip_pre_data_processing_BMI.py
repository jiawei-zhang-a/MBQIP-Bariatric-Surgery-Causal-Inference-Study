import pandas as pd
import numpy as np

import os

os.chdir("../data/mbqip")

maindata = pd.DataFrame(pd.read_csv('all_bmi_main_data.csv'))

# make csv/BMI/
os.makedirs("csv/BMI",exist_ok=True)

#anlysis before processing
maindata_T = maindata.describe(include = 'all').T
maindata_T["num_nan"] = maindata.isnull().sum(axis = 0)
maindata_T["num_distinct"] = maindata.nunique(axis = 0)
maindata_T.head()
maindata_T.to_csv("analysis.csv")

#create the colomn with difference between the BMI and BMI_DISCH
maindata['diff_BMI'] = maindata['BMI'] - maindata['BMI_DISCH']
maindata = maindata.drop(["BMI","BMI_DISCH"],axis=1)

#scale the maindata
standard_scaler = lambda x : (x-np.mean(x))/np.std(x)
maindata['AGE'] = maindata[['AGE']].apply(standard_scaler)
#maindata['diff_BMI'] = maindata[['diff_BMI']].apply(standard_scaler)

#extract five catogry
Sleeve = maindata.dropna(subset = ['Sleeve']) #treated as t = 0
Sleeve["treament"] = 0
RYGB = maindata.dropna(subset = ['RYGB'])
RYGB["treament"] = 1
Band = maindata.dropna(subset = ['Band'])
Band["treament"] = 1
BPD_DS = maindata.dropna(subset = ['BPD/DS'])
BPD_DS["treament"] = 1
SADI_S = maindata.dropna(subset = ['SADI-S'])
SADI_S["treament"] = 1


dataset_1 = pd.concat([RYGB,Sleeve])
dataset_1 = dataset_1.drop(["RYGB","Band","BPD/DS","SADI-S","Sleeve","REOP30_b'Yes'","READ30_b'Yes'","INTV30_b'Yes'","DEATH_1.0"],axis=1)
dataset_1 = dataset_1.sample(frac=1)
dataset_1.to_csv("csv/BMI/mbqip_1.csv", header = False,index = False)

dataset_2 = pd.concat([Band,Sleeve])
dataset_2 = dataset_2.drop(["RYGB","Band","BPD/DS","SADI-S","Sleeve","REOP30_b'Yes'","READ30_b'Yes'","INTV30_b'Yes'","DEATH_1.0"],axis=1)
dataset_2 = dataset_2.sample(frac=1)
dataset_2.to_csv("csv/BMI/mbqip_2.csv", header = False,index = False)

dataset_3 = pd.concat([BPD_DS,Sleeve])
dataset_3 = dataset_3.drop(["RYGB","Band","BPD/DS","SADI-S","Sleeve","REOP30_b'Yes'","READ30_b'Yes'","INTV30_b'Yes'","DEATH_1.0"],axis=1)
dataset_3 = dataset_3.sample(frac=1)
dataset_3.to_csv("csv/BMI/mbqip_3.csv", header = False,index = False)

dataset_4 = pd.concat([SADI_S,Sleeve])
dataset_4 = dataset_4.drop(["RYGB","Band","BPD/DS","SADI-S","Sleeve","REOP30_b'Yes'","READ30_b'Yes'","INTV30_b'Yes'","DEATH_1.0"],axis=1)
dataset_4 = dataset_4.sample(frac=1)
dataset_4.to_csv("csv/BMI/mbqip_4.csv",header = False,index = False)

"""
slice_size = 50000
#generate four diff_BMI Dataset
os.makedirs("../dat/mbqip/csv/diff_BMI",exist_ok=True)
os.makedirs("../dat/mbqip/csv/diff_BMI/RYGB",exist_ok=True)
os.makedirs("../dat/mbqip/csv/diff_BMI/BPD_DS",exist_ok=True)
os.makedirs("../dat/mbqip/csv/diff_BMI/SADI-S",exist_ok=True)
os.makedirs("../dat/mbqip/csv/diff_BMI/Band",exist_ok=True)

dataset_1 = pd.concat([RYGB,Sleeve])
dataset_1 = dataset_1.drop(["RYGB","Band","BPD/DS","SADI-S","Sleeve","REOP30_b'Yes'","READ30_b'Yes'","INTV30_b'Yes'","DEATH_1.0"],axis=1)
dataset_1 = dataset_1.sample(frac=1)
n=dataset_1.shape[0]
for i in range(int(n/slice_size)+1):
    dat = dataset_1.iloc[i*slice_size:min((i+1)*slice_size,n)]
    dat.to_csv("../dat/mbqip/csv/diff_BMI/RYGB/mbqip_%d.csv"%i, header = False,index = False)


dataset_2 = pd.concat([Band,Sleeve])
dataset_2 = dataset_2.drop(["RYGB","Band","BPD/DS","SADI-S","Sleeve","REOP30_b'Yes'","READ30_b'Yes'","INTV30_b'Yes'","DEATH_1.0"],axis=1)
dataset_2 = dataset_2.sample(frac=1)
n=dataset_2.shape[0]
for i in range(int(n/slice_size)+1):
    dat = dataset_2.iloc[i*slice_size:min((i+1)*slice_size,n)]
    dat.to_csv("../dat/mbqip/csv/diff_BMI/Band/mbqip_%d.csv"%i, header = False,index = False)


dataset_3 = pd.concat([BPD_DS,Sleeve])
dataset_3 = dataset_3.drop(["RYGB","Band","BPD/DS","SADI-S","Sleeve","REOP30_b'Yes'","READ30_b'Yes'","INTV30_b'Yes'","DEATH_1.0"],axis=1)
dataset_3 = dataset_3.sample(frac=1)
n=dataset_3.shape[0]
for i in range(int(n/slice_size)+1):
    dat =  dataset_3.iloc[i*slice_size:min((i+1)*slice_size,n)]
    dat.to_csv("../dat/mbqip/csv/diff_BMI/BPD_DS/mbqip_%d.csv"%i,header = False, index = False)


dataset_4 = pd.concat([SADI_S,Sleeve])
dataset_4 = dataset_4.drop(["RYGB","Band","BPD/DS","SADI-S","Sleeve","REOP30_b'Yes'","READ30_b'Yes'","INTV30_b'Yes'","DEATH_1.0"],axis=1)
dataset_4 = dataset_4.sample(frac=1)
n=dataset_4.shape[0]
for i in range(int(n/slice_size)+1):
    dat = dataset_4.iloc[i*slice_size:min((i+1)*slice_size,n)]
    dat.to_csv("../dat/mbqip/csv/diff_BMI/SADI-S/mbqip_%d.csv"%i, header = False,index = False)

"""
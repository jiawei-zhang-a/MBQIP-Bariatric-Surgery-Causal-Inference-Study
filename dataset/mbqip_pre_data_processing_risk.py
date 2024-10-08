import pandas as pd
import numpy as np

import os
#load
os.chdir("../data/mbqip")

maindata = pd.DataFrame(pd.read_csv('all_bmi_main_data.csv'))

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



# make csv/Death
os.makedirs("csv/Death",exist_ok=True)

dataset_1 = pd.concat([RYGB,Sleeve])
dataset_1 = dataset_1.drop(["RYGB","Band","BPD/DS","SADI-S","Sleeve","REOP30_b'Yes'","diff_BMI","INTV30_b'Yes'","READ30_b'Yes'"],axis=1)
dataset_1 = dataset_1.sample(frac=1)
dataset_1.to_csv("csv/Death/mbqip_1.csv", header = False,index = False)

dataset_2 = pd.concat([Band,Sleeve])
dataset_2 = dataset_2.drop(["RYGB","Band","BPD/DS","SADI-S","Sleeve","REOP30_b'Yes'","diff_BMI","INTV30_b'Yes'","READ30_b'Yes'"],axis=1)
dataset_2 = dataset_2.sample(frac=1)
dataset_2.to_csv("csv/Death/mbqip_2.csv", header = False,index = False)

dataset_3 = pd.concat([BPD_DS,Sleeve])
dataset_3 = dataset_3.drop(["RYGB","Band","BPD/DS","SADI-S","Sleeve","REOP30_b'Yes'","diff_BMI","INTV30_b'Yes'","READ30_b'Yes'"],axis=1)
dataset_3 = dataset_3.sample(frac=1)
dataset_3.to_csv("csv/Death/mbqip_3.csv", header = False,index = False)

dataset_4 = pd.concat([SADI_S,Sleeve])
dataset_4 = dataset_4.drop(["RYGB","Band","BPD/DS","SADI-S","Sleeve","REOP30_b'Yes'","diff_BMI","INTV30_b'Yes'","READ30_b'Yes'"],axis=1)
dataset_4 = dataset_4.sample(frac=1)
dataset_4.to_csv("csv/Death/mbqip_4.csv", header = False,index = False)

#generate four reoperation 
os.makedirs("csv/reoperation",exist_ok=True)

dataset_1 = pd.concat([RYGB,Sleeve])
dataset_1 = dataset_1.drop(["RYGB","Band","BPD/DS","SADI-S","Sleeve","DEATH_1.0","diff_BMI","INTV30_b'Yes'","READ30_b'Yes'"],axis=1)
dataset_1 = dataset_1.sample(frac=1)
dataset_1.to_csv("csv/reoperation/mbqip_1.csv", header = False,index = False)

dataset_2 = pd.concat([Band,Sleeve])
dataset_2 = dataset_2.drop(["RYGB","Band","BPD/DS","SADI-S","Sleeve","DEATH_1.0","diff_BMI","INTV30_b'Yes'","READ30_b'Yes'"],axis=1)
dataset_2 = dataset_2.sample(frac=1)
dataset_2.to_csv("csv/reoperation/mbqip_2.csv", header = False,index = False)

dataset_3 = pd.concat([BPD_DS,Sleeve])
dataset_3 = dataset_3.drop(["RYGB","Band","BPD/DS","SADI-S","Sleeve","DEATH_1.0","diff_BMI","INTV30_b'Yes'","READ30_b'Yes'"],axis=1)
dataset_3 = dataset_3.sample(frac=1)
dataset_3.to_csv("csv/reoperation/mbqip_3.csv", header = False,index = False)

dataset_4 = pd.concat([SADI_S,Sleeve])
dataset_4 = dataset_4.drop(["RYGB","Band","BPD/DS","SADI-S","Sleeve","DEATH_1.0","diff_BMI","INTV30_b'Yes'","READ30_b'Yes'"],axis=1)
dataset_4 = dataset_4.sample(frac=1)
dataset_4.to_csv("csv/reoperation/mbqip_4.csv",header = False, index = False)

#generate four readmission Dataset
os.makedirs("csv/readmission",exist_ok=True)

dataset_1 = pd.concat([RYGB,Sleeve])
dataset_1 = dataset_1.drop(["RYGB","Band","BPD/DS","SADI-S","Sleeve","REOP30_b'Yes'","DEATH_1.0","diff_BMI","INTV30_b'Yes'"],axis=1)
dataset_1 = dataset_1.sample(frac=1)
dataset_1.to_csv("csv/readmission/mbqip_1.csv",header = False,index = False)

dataset_2 = pd.concat([Band,Sleeve])
dataset_2 = dataset_2.drop(["RYGB","Band","BPD/DS","SADI-S","Sleeve","REOP30_b'Yes'","DEATH_1.0","diff_BMI","INTV30_b'Yes'"],axis=1)
dataset_2 = dataset_2.sample(frac=1)
dataset_2.to_csv("csv/readmission/mbqip_2.csv", header = False,index = False)

dataset_3 = pd.concat([BPD_DS,Sleeve])
dataset_3 = dataset_3.drop(["RYGB","Band","BPD/DS","SADI-S","Sleeve","REOP30_b'Yes'","DEATH_1.0","diff_BMI","INTV30_b'Yes'"],axis=1)
dataset_3 = dataset_3.sample(frac=1)
dataset_3.to_csv("csv/readmission/mbqip_3.csv", header = False,index = False)

dataset_4 = pd.concat([SADI_S,Sleeve])
dataset_4 = dataset_4.drop(["RYGB","Band","BPD/DS","SADI-S","Sleeve","REOP30_b'Yes'","DEATH_1.0","diff_BMI","INTV30_b'Yes'"],axis=1)
dataset_4 = dataset_4.sample(frac=1)
dataset_4.to_csv("csv/readmission/mbqip_4.csv", header = False,index = False)

#generate four intervention Dataset
if os.path.exists("csv/intervention") == False:
    os.mkdir("csv/intervention")
dataset_1 = pd.concat([RYGB,Sleeve])
dataset_1 = dataset_1.drop(["RYGB","Band","BPD/DS","SADI-S","Sleeve","REOP30_b'Yes'","READ30_b'Yes'","DEATH_1.0","diff_BMI"],axis=1)
dataset_1 = dataset_1.sample(frac=1)
dataset_1.to_csv("csv/intervention/mbqip_1.csv", header = False,index = False)

dataset_2 = pd.concat([Band,Sleeve])
dataset_2 = dataset_2.drop(["RYGB","Band","BPD/DS","SADI-S","Sleeve","REOP30_b'Yes'","READ30_b'Yes'","DEATH_1.0","diff_BMI"],axis=1)
dataset_2 = dataset_2.sample(frac=1)
dataset_2.to_csv("csv/intervention/mbqip_2.csv", header = False,index = False)

dataset_3 = pd.concat([BPD_DS,Sleeve])
dataset_3 = dataset_3.drop(["RYGB","Band","BPD/DS","SADI-S","Sleeve","REOP30_b'Yes'","READ30_b'Yes'","DEATH_1.0","diff_BMI"],axis=1)
dataset_3 = dataset_3.sample(frac=1)
dataset_3.to_csv("csv/intervention/mbqip_3.csv", header = False,index = False)

dataset_4 = pd.concat([SADI_S,Sleeve])
dataset_4 = dataset_4.drop(["RYGB","Band","BPD/DS","SADI-S","Sleeve","REOP30_b'Yes'","READ30_b'Yes'","DEATH_1.0","diff_BMI"],axis=1)
dataset_4 = dataset_4.sample(frac=1)
dataset_4.to_csv("csv/intervention/mbqip_4.csv",header = False,index = False)



import pandas as pd
import numpy as np
import os
#load
maindata = pd.DataFrame(pd.read_csv('../dat/mbqip/csv/Death/mbqip_1.csv'))
maindata_T = maindata.describe(include = 'all').T
maindata_T["num_nan"] = maindata.isnull().sum(axis = 0)
maindata_T["num_distinct"] = maindata.nunique(axis = 0)
maindata_T.head()
maindata_T.to_csv("../dat/mbqip/csv/Death/analysis.csv")
print(maindata.loc[0][50])
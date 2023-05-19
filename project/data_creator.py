import numpy as np
import pandas as pd
import glob 
import os

#copy relative path 
path = "archive/individual_stocks_5yr/individual_stocks_5yr"
data_paths = glob.glob(path+"/*.csv")

all_df = None
for data_path in data_paths:
    print(data_path)
    df = pd.read_csv(data_path)
    name = df['Name'][0]
    df = df[["date","close"]]
    df = df.rename(columns={"close":name})
    if all_df is None:
        all_df = df
    else:
        all_df = pd.merge(all_df,df,on="date",how="outer")
all_df.to_csv("data.csv",index=False)
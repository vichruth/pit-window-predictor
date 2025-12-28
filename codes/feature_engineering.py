import pandas as pd
import numpy as np

# load the dataset to be used for feature engineering
df= pd.read_csv('/home/vichruth/ml_projects/pit-window-predictor/f1_dataset/processed_laps_2018_2024.csv')
df['is_pit']= df['pit_stop_number']>0
print("Pit stop flagged")

# stint logic 
df['stint_marker']=df.groupby(['raceId','driverId'])['is_pit'].shift(1).fillna(False)
print("Stint calculated")
df['stint_marker']=df['stint_marker'].astype(int)
df['stint_id']=df.groupby(['raceId','driverId'])['stint_marker'].cumsum()+1
print("Stint ID ASSIGNED")

#calculate tire age and save the data
df['tire_age']=df.groupby(['raceId','driverId','stint_id']).cumcount()+1
print("Tire age calculated")
df.to_csv('../f1_dataset/f1_feature_engineered_data.csv', index=False)
print("everything completed")

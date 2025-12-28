#this code is made for visualizing the tire decay 
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
df=pd.read_csv('../f1_dataset/f1_feature_engineered_data.csv')
print("Filtering data")
demodata=df[(df['code']=='HAM')&
            (df['year']==2023)&
            (df['milliseconds']<1000000)]
print("Plotting data")
plt.figure(figsize=(16,8))
sn.scatterplot(data=demodata,x='tire_age',y='milliseconds',alpha=0.5)
sn.regplot(data=demodata,x='tire_age',y='milliseconds',scatter=False,color='red')

plt.title('Tire age vs lap time scatter plot with regression line')
plt.xlabel('Tire age (in laps)')
plt.ylabel('Lap time (in milliseconds)')
plt.savefig('tire_age vs lap_time.png')
plt.grid(True)
plt.show()

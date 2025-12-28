import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle
 
# load data and clean data
data = pd.read_csv('../f1_dataset/f1_feature_engineered_data.csv')
df = data[data['milliseconds'] < 200000].copy()
features = ['circuitId','driverId','tire_age','lap']
target='milliseconds'

X = df[features]
y = df[target]

# train test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train model
model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=8, n_jobs=-1,device='cuda',tree_method='hist')
model.fit(X_train, y_train)

# evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')    

# save model 
with open('../outputs/pit_boss_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Saved model to pit_boss_model.pkl")
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, confusion_matrix
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

ddrtree_data = pd.read_csv("../prevafecg_op_unique_tree_proj_complete_outcomes.csv")
ukb_data = pd.read_csv("../ukb_af_data.csv")

# Z1
X = ddrtree_data[['X20', 'X42', 'X41', 'X8', 'X7', 'X22', 'X50',
       'X40', 'X31', 'X29', 'X21', 'X2', 'X4', 'X17', 'X38', 'X37', 'X35',
       'X16', 'X43', 'X6', 'X30', 'X33', 'X15', 'X9', 'X18', 'X19', 'X23',
       'X25', 'X1', 'X12', 'X46', 'X34', 'X45', 'X3', 'X49', 'X39', 'X24',
       'X14', 'X36', 'X10', 'X44', 'X26', 'X28', 'X13', 'X48', 'X11', 'X27',
       'X47', 'X5', 'X51', 'X32']]
y_Z1 = ddrtree_data[['Z1']]
y_Z2 = ddrtree_data[['Z2']]

X_test = ukb_data[['X20', 'X42', 'X41', 'X8', 'X7', 'X22', 'X50',
       'X40', 'X31', 'X29', 'X21', 'X2', 'X4', 'X17', 'X38', 'X37', 'X35',
       'X16', 'X43', 'X6', 'X30', 'X33', 'X15', 'X9', 'X18', 'X19', 'X23',
       'X25', 'X1', 'X12', 'X46', 'X34', 'X45', 'X3', 'X49', 'X39', 'X24',
       'X14', 'X36', 'X10', 'X44', 'X26', 'X28', 'X13', 'X48', 'X11', 'X27',
       'X47', 'X5', 'X51', 'X32']]
       
X_train_z1, X_val_z1, y_train_z1, y_val_z1 = train_test_split(X, y_Z1, test_size=0.25, random_state=42)
print(X_train_z1.shape)
print(X_val_z1.shape)

import time
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

param_grid = {
    'max_depth': [3, 5, 7],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200, 300]
}


start_time = time.time()
xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(X_train_z1, y_train_z1)

best_params = grid_search.best_params_
print("Best Parameters for Z1:", best_params)

# Get the best model
best_model = grid_search.best_estimator_

end_time = time.time()
elapsed_minutes = (end_time - start_time) / 60

print(f"The task took {elapsed_minutes:.2f} minutes to run for Z1.")

y_pred_z1 = best_model.predict(X_val_z1)
mse = mean_squared_error(y_val_z1, y_pred_z1)
print("Mean Squared Error on Validation Set:", mse)

rmse = mean_squared_error(y_val_z1, y_pred_z1, squared=False)
print(f"Root Mean Squared Error: {rmse}")

r2 = r2_score(y_val_z1, y_pred_z1)
print(f"R-squared: {r2}")

mae = mean_absolute_error(y_val_z1, y_pred_z1)
print(f"Mean Absolute Error: {mae}")

# For Z2

X_train_z2, X_val_z2, y_train_z2, y_val_z2 = train_test_split(X, y_Z2, test_size=0.25, random_state=42)
print(X_train_z2.shape)
print(X_val_z2.shape)

start_time = time.time()
xgb = XGBRegressor(objective='reg:squarederror', random_state=42)


grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(X_train_z2, y_train_z2)

best_params = grid_search.best_params_
print("Best Parameters for Z2:", best_params)

best_model = grid_search.best_estimator_

end_time = time.time()
elapsed_minutes = (end_time - start_time) / 60

print(f"The task took {elapsed_minutes:.2f} minutes to run for Z2.")

y_pred_z2 = best_model.predict(X_val_z2)
mse = mean_squared_error(y_val_z2, y_pred_z2)
print("Mean Squared Error on Validation Set:", mse)

rmse = mean_squared_error(y_val_z2, y_pred_z2, squared=False)
print(f"Root Mean Squared Error: {rmse}")

r2 = r2_score(y_val_z2, y_pred_z2)
print(f"R-squared: {r2}")

mae = mean_absolute_error(y_val_z2, y_pred_z2)
print(f"Mean Absolute Error: {mae}")

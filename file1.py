import pandas as pd
import numpy as np

# Loading dataset
df = pd.read_excel("Copy of BlinkIT Grocery Data.xlsx")

#fixing column names....
df.columns = df.columns.str.strip().str.replace(" ", "_")

print("Columns in dataset:\n", df.columns)

#data cleaning....
df.drop_duplicates(inplace=True)

#handling missing values...
# Numerical columns to be filled with mean
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())

# Categorical columns to be filled with mode
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Feature engg...
# Outlet Age
if 'Outlet_Establishment_Year' in df.columns:
    df['Outlet_Age'] = 2026 - df['Outlet_Establishment_Year']

# Fix Item Visibility
if 'Item_Visibility' in df.columns:
    df['Item_Visibility'] = df['Item_Visibility'].replace(0, df['Item_Visibility'].mean())

#encoding(ML cannot understand text, so it converts texts into numbers)
categorical_cols = [
    'Item_Fat_Content',
    'Outlet_Size',
    'Outlet_Location_Type',
    'Outlet_Type',
    'Item_Type'
]

existing_cols = [col for col in categorical_cols if col in df.columns]

if existing_cols:
    df = pd.get_dummies(df, columns=existing_cols, drop_first=True)

#target column
target_col = 'Sales'#(this is what we are predicting using the model)
# Drop unnecessary columns if present
drop_cols = [col for col in ['Item_Identifier', 'Outlet_Identifier'] if col in df.columns]

X = df.drop([target_col] + drop_cols, axis=1)
y = df[target_col]

# train test split(divides data into training and testing data)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regg
# model training
from sklearn.linear_model import LinearRegression

model = LinearRegression()#model is learning patterns in data
model.fit(X_train, y_train)

#predictions
y_pred = model.predict(X_test)

#evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

mae = mean_absolute_error(y_test, y_pred)#avg error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))#prevents large errors

print("\nLinear Regression")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2_score(y_test, y_pred))#how well model explains data

# Random Forest
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("\nRandom Forest")
print("MAE:", rf_mae)
print("RMSE:", rf_rmse)
print("R2:", rf_r2)

# Xgboost
from xgboost import XGBRegressor

xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_r2 = r2_score(y_test, xgb_pred)

print("\nXGBoost")
print("MAE:", xgb_mae)
print("RMSE:", xgb_rmse)
print("R2:", xgb_r2)

#model comparisons
print("\nModel Comparison")
print("Linear Regression → RMSE:", rmse, "R2:", r2_score(y_test, y_pred))
print("Random Forest     → RMSE:", rf_rmse, "R2:", rf_r2)
print("XGBoost           → RMSE:", xgb_rmse, "R2:", xgb_r2)

# selecting best model based on RMSE
if rf_rmse < rmse and rf_rmse < xgb_rmse:
    best_model = rf_model
    print("\nBest Model: Random Forest")
elif xgb_rmse < rmse and xgb_rmse < rf_rmse:
    best_model = xgb_model
    print("\nBest Model: XGBoost")
else:
    best_model = model
    print("\nBest Model: Linear Regression")

# full data prediction
df['Predicted_Sales'] = best_model.predict(X)

# Saving file
df.to_excel("blinkit_with_predictions.xlsx", index=False)

print("File saved as blinkit_with_predictions.xlsx")

#EDA (Item Outlet Sales Distribution Histogram)
#shows distribution of sales
import matplotlib.pyplot as plt
plt.figure()
df['Sales'].hist()
plt.title("Item Outlet Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.show()

#EDA (Correlation Heatmap)
#which features impact sales
#only numerical columns are selected
corr = df.select_dtypes(include=['float64', 'int64']).corr()
plt.figure()
plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Heatmap of Numerical Features")
plt.show()

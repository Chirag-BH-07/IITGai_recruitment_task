import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
# Metric
def mape(y_true, y_pred):
    y_true = np.clip(y_true, 1e-6, None)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# 1. Load Data
df_p = pd.read_excel('PGCB_date_power_demand.xlsx')
df_w = pd.read_excel('weather_data.xlsx')
df_e = pd.read_csv('economic_full_1.csv')


# 2. Time Alignment
df_p['datetime'] = pd.to_datetime(df_p['datetime'])
df_w['datetime'] = pd.to_datetime(df_w['datetime'])

df_p = df_p.drop_duplicates(subset=['datetime']).sort_values('datetime')
df_w = df_w.drop_duplicates(subset=['datetime']).sort_values('datetime')

df_p = df_p.set_index('datetime')
df_w = df_w.set_index('datetime')

# enforce hourly frequency
df = df_p.resample('1H').ffill()
df_w = df_w.resample('1H').ffill()

df = df.join(df_w, how='left')


# 3. Economic Data Merge
df['year'] = df.index.year
df = df.reset_index().merge(df_e, on='year', how='left').set_index('datetime')


# 4. Anomaly Handling (Robust)
roll_med = df['demand_mw'].rolling(window=24, min_periods=1).median()
roll_mad = np.abs(df['demand_mw'] - roll_med).rolling(window=24, min_periods=1).median()

upper = roll_med + 3 * roll_mad
lower = roll_med - 3 * roll_mad

df['demand_clean'] = np.where(
    (df['demand_mw'] > upper) | (df['demand_mw'] < lower),
    roll_med,
    df['demand_mw']
)


# 5. Time Features
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['month'] = df.index.month
df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)


# 6. Lag Features
df['lag_1'] = df['demand_clean'].shift(1)
df['lag_2'] = df['demand_clean'].shift(2)
df['lag_24'] = df['demand_clean'].shift(24)
df['lag_168'] = df['demand_clean'].shift(168)  # weekly


# 7. Rolling Features
df['roll_mean_3'] = df['demand_clean'].shift(1).rolling(3).mean()
df['roll_mean_24'] = df['demand_clean'].shift(1).rolling(24).mean()
df['roll_std_24'] = df['demand_clean'].shift(1).rolling(24).std()
df['roll_mean_168'] = df['demand_clean'].shift(1).rolling(168).mean()


# 8. Weather Feature Engineering
if 'temperature' in df.columns:
    df['temp_sq'] = df['temperature'] ** 2

if 'humidity' in df.columns and 'temperature' in df.columns:
    df['temp_humidity'] = df['temperature'] * df['humidity']


# 9. Economic Feature Engineering
# (assuming GDP exists, adjust if column names differ)
if 'gdp' in df.columns:
    df['gdp_growth'] = df['gdp'].pct_change()
    df['gdp_trend'] = df['gdp'] / df['gdp'].rolling(3).mean()


# 10. Target (t+1)
df['target'] = df['demand_clean'].shift(-1)


# 11. Drop NA
df = df.dropna()


# 12. Train/Test Split
train = df[df.index.year < 2023]
test = df[df.index.year == 2023]

features = [c for c in df.columns if c not in ['demand_mw', 'demand_clean', 'target', 'year']]

X_train = train[features]
y_train = train['target']

X_test = test[features]
y_test = test['target']


# 13. Model
model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)


# 14. Evaluation
preds = model.predict(X_test)
score = mape(y_test, preds)

print(f"Test MAPE: {score:.3f}%")


# 15. Feature Importance
imp = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values(by='importance', ascending=False).head(10)

plt.figure(figsize=(10, 5))
plt.barh(imp['feature'][::-1], imp['importance'][::-1])
plt.xlabel("Importance")
plt.title("Top Features")
plt.show()
# Predictive Paradox – Power Demand Forecasting

## Overview

This project focuses on building a machine learning pipeline to predict short-term electricity demand on the national grid. The goal is to forecast the next hour’s demand (demand_mw) using historical consumption data along with weather and economic indicators.

The approach follows a standard tabular ML setup where temporal dependencies are captured through feature engineering rather than sequence models.


## Dataset Description

Three datasets were provided:

1. PGCB_date_power_demand.xlsx  
  Hourly electricity demand and generation data. This is the primary dataset and contains the target variable.

2. weather_data.xlsx  
  Hourly weather data including temperature, humidity, and other environmental variables.

3. economic_full_1.csv  
  Annual macroeconomic indicators such as GDP.


## Approach

### 1. Data Cleaning & Preparation

1. Converted timestamps to datetime format and sorted chronologically.
2. Removed duplicate entries based on timestamps.
3. Resampled data to enforce strict hourly frequency.
4. Forward-filled missing values where appropriate.

---

### 2. Anomaly Handling

The demand data contained sudden spikes and irregular values.

To handle this:
1. Used a rolling median (24-hour window) to estimate local trends.
2. Computed Median Absolute Deviation (MAD) to detect outliers.
3. Replaced extreme values (beyond ±3 MAD) with the rolling median.

This method preserves seasonality while removing noise.


### 3. Feature Engineering

Since the model is non-sequential, temporal structure was explicitly encoded.

#### Time-based Features
1. Hour of day
2. Day of week
3. Month
4. Weekend indicator

#### Lag Features
1. Previous 1 hour (lag_1)
2. Previous 2 hours (lag_2)
3. Previous 24 hours (lag_24)
4. Previous 168 hours (lag_168, weekly pattern)

#### Rolling Statistics
1. Mean over last 3 hours
2. Mean over last 24 hours
3. Standard deviation over last 24 hours
4. Weekly rolling mean (168 hours)

#### Weather Features
1. Squared temperature (to capture non-linearity)
2. Interaction between temperature and humidity

#### Economic Features
1. Year-wise merge of macroeconomic data
2. Derived:
  1. GDP growth rate
  2. Rolling GDP trend


### 4. Target Variable

The prediction target is defined as:
1. Next hour demand (t+1)

This is created by shifting the cleaned demand series backward by one step.

---

### 5. Train-Test Split

1. Training data: all years before 2023  
2. Test data: full year 2023  

This ensures strict chronological separation and avoids data leakage.


### 6. Model

Used LightGBM Regressor, a gradient boosting tree-based model.

Key parameters:
1. n_estimators = 1000
2. learning_rate = 0.03
3. num_leaves = 31
4. Subsampling applied to reduce overfitting


### 7. Evaluation Metric

Performance is evaluated using:

Mean Absolute Percentage Error (MAPE)

A small constant is used to avoid division by zero issues.


### 8. Results

The model outputs a final Test MAPE (%), which reflects prediction accuracy on unseen data (2023).

Lower MAPE indicates better performance.


### 9. Feature Importance

Feature importance is extracted from the trained LightGBM model and visualized.

This helps identify key drivers of electricity demand, such as:
1. Recent demand patterns (lags)
2. Time-of-day effects
3. Weather conditions


## Key Observations

1. Short-term demand is heavily influenced by recent historical values.
2. Weekly seasonality (lag_168) plays an important role.
3. Weather variables introduce non-linear effects, especially temperature.
4. Economic indicators contribute more to long-term shifts rather than hourly variation.

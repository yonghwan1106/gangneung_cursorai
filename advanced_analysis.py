import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import numpy as np

# 데이터 로드
data = pd.read_csv('data/total_data.csv')
data['Year'] = pd.to_datetime(data['Year'], format='%Y')
data.set_index('Year', inplace=True)

# 한글 폰트 설정 (맑은 고딕 사용)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def time_series_decomposition():
    # 1. 시계열 데이터 분해
    decomposition = seasonal_decompose(data['RiceProduction'], model='additive', period=1)
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
    decomposition.observed.plot(ax=ax1)
    ax1.set_title('관측값')
    decomposition.trend.plot(ax=ax2)
    ax2.set_title('추세')
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_title('계절성')
    decomposition.resid.plot(ax=ax4)
    ax4.set_title('잔차')
    plt.tight_layout()
    return fig

def sarima_model():
    # 2. SARIMA 모델 적용
    model = SARIMAX(data['RiceProduction'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['RiceProduction'], label='실제 데이터')
    ax.plot(data.index, results.fittedvalues, color='red', label='SARIMA 모델')
    ax.set_title('SARIMA 모델 적용 결과')
    ax.legend()
    return fig

def random_forest_model():
    # 3. 랜덤 포레스트 모델
    X = data[['temperature', 'precipitation', 'Farmhouseholds', 'PaddyField+Upland']]
    y = data['RiceProduction']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('실제 값')
    ax.set_ylabel('예측 값')
    ax.set_title(f'랜덤 포레스트 모델 (MSE: {mse:.2f})')
    return fig

def xgboost_model():
    # 4. XGBoost 모델
    X = data[['temperature', 'precipitation', 'Farmhouseholds', 'PaddyField+Upland']]
    y = data['RiceProduction']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    xgb_model = xgb.XGBRegressor(objective ='reg:squarederror', n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    y_pred = xgb_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('실제 값')
    ax.set_ylabel('예측 값')
    ax.set_title(f'XGBoost 모델 (MSE: {mse:.2f})')
    return fig

def create_advanced_plots():
    return {
        'time_series_decomposition': time_series_decomposition(),
        'sarima_model': sarima_model(),
        'random_forest': random_forest_model(),
        'xgboost': xgboost_model()
    }
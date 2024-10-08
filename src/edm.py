import pandas as pd
import matplotlib.pyplot as plt

# 데이터 로드
data = pd.read_csv('data/total_data.csv')

# 한글 폰트 설정 (맑은 고딕 사용)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def plot_temperature():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['Year'], data['temperature'], marker='o')
    ax.set_title('연간 평균기온 추이')
    ax.set_xlabel('연도')
    ax.set_ylabel('평균기온 (°C)')
    ax.grid(True)
    return fig

def plot_precipitation():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(data['Year'], data['precipitation'])
    ax.set_title('연간 강수량 변화')
    ax.set_xlabel('연도')
    ax.set_ylabel('강수량 (mm)')
    return fig

def plot_farm_data():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['Year'], data['Farmhouseholds'], marker='o', label='농가수')
    ax.plot(data['Year'], data['Farmpopulation'], marker='s', label='농가인구')
    ax.set_title('농가수 및 농가인구 변화')
    ax.set_xlabel('연도')
    ax.set_ylabel('수')
    ax.legend()
    ax.grid(True)
    return fig

def plot_land_area():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(data['Year'], data['PaddyField'], label='논')
    ax.bar(data['Year'], data['Upland'], bottom=data['PaddyField'], label='밭')
    ax.set_title('경지면적 변화')
    ax.set_xlabel('연도')
    ax.set_ylabel('면적')
    ax.legend()
    return fig

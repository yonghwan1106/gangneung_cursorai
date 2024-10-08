import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

# 데이터 로드
data = pd.read_csv('data/total_data.csv')

# 한글 폰트 설정 (맑은 고딕 사용)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def correlation_analysis():
    # 1. 주요 변수 간 상관관계 분석
    variables = ['temperature', 'precipitation', 'Farmhouseholds', 'PaddyField+Upland', 'RiceProduction', 'PotatoesProduction']
    corr = data[variables].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('주요 변수 간 상관관계')
    return fig

def scatter_plots():
    # 2. 산점도 (강수량과 쌀, 감자 생산량 간의 관계)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.scatter(data['precipitation'], data['RiceProduction'])
    ax1.set_xlabel('강수량')
    ax1.set_ylabel('쌀 생산량')
    ax1.set_title('강수량과 쌀 생산량의 관계')

    ax2.scatter(data['precipitation'], data['PotatoesProduction'])
    ax2.set_xlabel('강수량')
    ax2.set_ylabel('감자 생산량')
    ax2.set_title('강수량과 감자 생산량의 관계')

    plt.tight_layout()
    return fig

def rice_regression():
    # 3. 쌀 생산량에 대한 단순회귀분석
    x = data['precipitation']
    y = data['RiceProduction']
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y)
    ax.plot(x, intercept + slope * x, color='red', label='회귀선')
    ax.set_xlabel('강수량')
    ax.set_ylabel('쌀 생산량')
    ax.set_title(f'쌀 생산량에 대한 단순회귀분석 (R^2 = {r_value**2:.2f})')
    ax.legend()
    return fig

def potato_regression():
    # 4. 감자 생산량에 대한 단순회귀분석
    x = data['precipitation']
    y = data['PotatoesProduction']
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y)
    ax.plot(x, intercept + slope * x, color='red', label='회귀선')
    ax.set_xlabel('강수량')
    ax.set_ylabel('감자 생산량')
    ax.set_title(f'감자 생산량에 대한 단순회귀분석 (R^2 = {r_value**2:.2f})')
    ax.legend()
    return fig

# 메인 함수
def create_regression_plots():
    return {
        'correlation': correlation_analysis(),
        'scatter': scatter_plots(),
        'rice_regression': rice_regression(),
        'potato_regression': potato_regression()
    }
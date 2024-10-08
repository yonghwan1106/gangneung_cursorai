import pandas as pd
import numpy as np

def load_data():
    """
    데이터를 로드하고 전처리하는 함수
    """
    # 데이터 로드
    data = pd.read_csv('data/total_data.csv')
    
    # 날짜 형식 변환
    data['Year'] = pd.to_datetime(data['Year'], format='%Y')
    
    # 쉼표 제거 및 숫자형으로 변환
    numeric_columns = ['precipitation', 'PaddyField', 'Upland', 'PaddyField+Upland', 
                       'Farmhouseholds', 'Farmpopulation', 'fullTime', 'partTime', 
                       'fullTime+partTime', 'RiceProduction', 'PotatoesProduction']
    
    for col in numeric_columns:
        data[col] = data[col].replace(',', '', regex=True).astype(float)
    
    # PM2.5 컬럼의 소수점 처리
    data['PM2.5'] = data['PM2.5'].astype(float)
    
    # 인덱스 설정
    data.set_index('Year', inplace=True)
    
    return data

# 데이터를 한 번만 로드하고 메모리에 저장
_data = None

def get_data():
    """
    데이터를 반환하는 함수. 
    이미 로드된 데이터가 있으면 그것을 반환하고, 없으면 새로 로드합니다.
    """
    global _data
    if _data is None:
        _data = load_data()
    return _data

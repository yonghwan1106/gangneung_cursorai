import streamlit as st
from src import edm, regression_analysis, advanced_analysis

def main():
    st.title('농업 및 기상 데이터 분석')

    # 사이드바 메뉴 생성
    menu = st.sidebar.selectbox(
        "분석 메뉴",
        ("기본 데이터 분석", "회귀 분석", "고급 분석")
    )

    if menu == "기본 데이터 분석":
        st.header('기본 데이터 분석')
        st.subheader('1. 연간 평균기온 추이')
        fig_temp = edm.plot_temperature()
        st.pyplot(fig_temp)
        
        st.subheader('2. 연간 강수량 변화')
        fig_prec = edm.plot_precipitation()
        st.pyplot(fig_prec)
        
        st.subheader('3. 농가수 및 농가인구 변화')
        fig_farm = edm.plot_farm_data()
        st.pyplot(fig_farm)
        
        st.subheader('4. 경지면적 변화')
        fig_land = edm.plot_land_area()
        st.pyplot(fig_land)

    elif menu == "회귀 분석":
        st.header('회귀 분석 결과')
        regression_plots = regression_analysis.create_regression_plots()
        
        st.subheader('1. 주요 변수 간 상관관계')
        st.pyplot(regression_plots['correlation'])
        
        st.subheader('2. 강수량과 생산량 간의 관계 (산점도)')
        st.pyplot(regression_plots['scatter'])
        
        st.subheader('3. 쌀 생산량에 대한 단순회귀분석')
        st.pyplot(regression_plots['rice_regression'])
        
        st.subheader('4. 감자 생산량에 대한 단순회귀분석')
        st.pyplot(regression_plots['potato_regression'])

    elif menu == "고급 분석":
        st.header('고급 분석 결과')
        advanced_plots = advanced_analysis.create_advanced_plots()
        
        st.subheader('1. 시계열 데이터 분해')
        st.pyplot(advanced_plots['time_series_decomposition'])
        
        st.subheader('2. SARIMA 모델 적용')
        st.pyplot(advanced_plots['sarima_model'])
        
        st.subheader('3. 랜덤 포레스트 모델')
        st.pyplot(advanced_plots['random_forest'])
        
        st.subheader('4. XGBoost 모델')
        st.pyplot(advanced_plots['xgboost'])

if __name__ == "__main__":
    main()
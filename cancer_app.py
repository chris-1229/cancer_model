import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import platform

# -------------------------------------------------------------
# 0. Matplotlib 한글 깨짐 방지 설정
# -------------------------------------------------------------
system_os = platform.system()

if system_os == "Windows":
    plt.rc("font", family="Malgun Gothic")  # 윈도우 맑은 고딕
elif system_os == "Darwin":
    plt.rc("font", family="AppleGothic")    # 맥 애플 고딕
else:
    # 리눅스(배포 환경)의 경우 기본 나눔 폰트 시도 또는 대체 폰트 설정
    plt.rc("font", family="NanumGothic")

# 마이너스 기호(-) 깨짐 방지
plt.rc("axes", unicode_minus=False)
# -------------------------------------------------------------

# 1. 페이지 설정 및 모델/데이터 로드
st.set_page_config(page_title="폐 건강 군집 분석", layout="centered")
st.title("🫁 환자 데이터 직접 입력 및 군집 예측")

@st.cache_resource
def load_artifacts():
    model = joblib.load("lung_model.pkl")
    scaler = joblib.load("lung_scaler.pkl")
    df = pd.read_csv("lung.csv")
    return model, scaler, df

try:
    model, scaler, df = load_artifacts()
except FileNotFoundError as e:
    st.error(f"필수 파일이 누락되었습니다: {e.filename}. 파일 위치를 확인해주세요.")
    st.stop()

# 2. 사용자 데이터 직접 입력 표 (Data Editor)
st.subheader("📝 환자 데이터 입력")
st.markdown("아래 표의 값을 더블클릭하여 수정하세요. 행을 추가하여 여러 명을 입력할 수도 있습니다.")

# 초기 예시 데이터 구성
init_data = pd.DataFrame([[10.0, 40.0, 5.0]], columns=['흡연', '나이', '알코올'])

# 사용자가 표에서 직접 수정할 수 있는 에디터
edited_df = st.data_editor(init_data, num_rows="dynamic", use_container_width=True)

# 3. 예측하기 버튼 클릭 시 수행
if st.button("🚀 군집 예측 및 시각화 실행", type="primary"):
    if edited_df.empty or edited_df.isnull().values.any():
        st.error("빈 칸 없이 데이터를 올바르게 입력해주세요.")
    else:
        # 데이터 스케일링 및 예측
        new_patients_scaled = scaler.transform(edited_df)
        pred_clusters = model.predict(new_patients_scaled)
        
        # 결과 표에 예측된 군집 추가해서 보여주기
        result_df = edited_df.copy()
        result_df['예측 군집'] = pred_clusters
        
        st.write("---")
        st.subheader("🔮 예측 결과")
        st.dataframe(result_df, use_container_width=True)
        
        # 4. 그래프 시각화
        st.write("---")
        st.subheader("📊 군집 내 환자 위치 시각화")
        
        # 기존 전체 데이터의 군집 결과 구하기 (배경용)
        if '군집' not in df.columns:
            df_scaled = scaler.transform(df[['흡연', '나이', '알코올']])
            df['군집'] = model.predict(df_scaled)
            
        # 축 선택
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("X축 선택", ['흡연', '나이', '알코올'], index=0)
        with col2:
            y_axis = st.selectbox("Y축 선택", ['흡연', '나이', '알코올'], index=1)
            
        # 그래프 그리기
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # 배경: 기존 데이터 분포
        sns.scatterplot(
            data=df, 
            x=x_axis, 
            y=y_axis, 
            hue='군집', 
            palette='Set2', 
            alpha=0.4, 
            ax=ax
        )
        
        # 강조: 새로 입력한 사용자 데이터 (빨간색 큰 별)
        ax.scatter(
            result_df[x_axis], 
            result_df[y_axis], 
            color='red', 
            marker='*', 
            s=300, 
            label='신규 입력 환자',
            edgecolor='black',
            zorder=5
        )
        
        # 한글 제목 및 레이블 설정
        ax.set_title(f"[{x_axis} vs {y_axis}] 기존 데이터 분포 및 입력 환자 위치", fontsize=14, pad=15)
        ax.set_xlabel(x_axis, fontsize=12)
        ax.set_ylabel(y_axis, fontsize=12)
        ax.legend()
        
        st.pyplot(fig)
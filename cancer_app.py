import platform
import joblib
import pandas as pd
import streamlit as st

# -------------------------------------------------------------
# 1. 페이지 설정 및 모델/데이터 로드
# -------------------------------------------------------------
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
    st.error(
        f"필수 파일이 누락되었습니다: {e.filename}. 파일 위치를 확인해주세요."
    )
    st.stop()

# -------------------------------------------------------------
# 2. 사용자 데이터 직접 입력 표 (Data Editor)
# -------------------------------------------------------------
st.subheader("📝 환자 데이터 입력")
st.markdown(
    "아래 표의 값을 더블클릭하여 수정하세요. 행을 추가하여 여러 명을 입력할 수도 있습니다."
)

# 초기 예시 데이터 구성
init_data = pd.DataFrame([[10.0, 40.0, 5.0]], columns=["흡연", "나이", "알코올"])

# 사용자가 표에서 직접 수정할 수 있는 에디터
edited_df = st.data_editor(
    init_data, num_rows="dynamic", use_container_width=True
)

# -------------------------------------------------------------
# 3. 예측하기 버튼 클릭 시 수행
# -------------------------------------------------------------
if st.button("🚀 군집 예측 및 시각화 실행", type="primary"):
    if edited_df.empty or edited_df.isnull().values.any():
        st.error("빈 칸 없이 데이터를 올바르게 입력해주세요.")
    else:
        # 데이터 스케일링 및 예측
        new_patients_scaled = scaler.transform(edited_df)
        pred_clusters = model.predict(new_patients_scaled)

        # 결과 표에 예측된 군집 추가해서 보여주기
        result_df = edited_df.copy()
        result_df["예측 군집"] = pred_clusters

        # 💡 시각화 구분을 위해 신규 데이터에 유형 태그 달기
        result_df["데이터 유형"] = "신규 입력 환자"

        st.write("---")
        st.subheader("🔮 예측 결과")
        # '데이터 유형' 칼럼은 표에서 굳이 보여줄 필요 없으므로 숨기거나 제외하고 출력
        st.dataframe(
            result_df.drop(columns=["데이터 유형"]), use_container_width=True
        )

        # -------------------------------------------------------------
        # 4. 내장 기능을 이용한 시각화 (Matplotlib/Seaborn 대체)
        # -------------------------------------------------------------
        st.write("---")
        st.subheader("📊 군집 내 환자 위치 시각화")

        # 기존 전체 데이터의 군집 결과 구하기 (배경용)
        if "군집" not in df.columns:
            df_scaled = scaler.transform(df[['Smokes', 'Age', 'Alkhol']])
            df["군집"] = model.predict(df_scaled)

        # 기존 데이터에도 유형 태그 달기 (군집명을 문자열로 변환하여 시각화 범례 최적화)
        chart_df = df.copy()
        chart_df["데이터 유형"] = chart_df["군집"].apply(
            lambda x: f"기존 데이터 (군집 {x})"
        )

        # 기존 데이터와 신규 데이터를 하나로 합치기
        # 신규 입력 데이터의 '데이터 유형'은 '신규 입력 환자'로 유지됨
        combined_df = pd.concat([chart_df, result_df], ignore_index=True)

        # 축 선택 인터페이스
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox(
                "X축 선택", ["흡연", "나이", "알코올"], index=0
            )
        with col2:
            y_axis = st.selectbox(
                "Y축 선택", ["흡연", "나이", "알코올"], index=1
            )

        # Streamlit 내장 산점도(Scatter Chart) 그리기
        # 별도의 한글 폰트 설정 없이도 글자가 깨지지 않습니다.
        st.scatter_chart(
            data=combined_df,
            x=x_axis,
            y=y_axis,
            color="데이터 유형",  # 기존 군집들과 신규 환자가 색상으로 구분됨
            size="데이터 유형",  # 신규 입력 환자를 더 크게 띄우기 위한 트릭
            use_container_width=True,
        )

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

# 💡 [컬럼명 변경] CSV 파일에 맞게 영어 컬럼명으로 초기 데이터를 구성합니다.
# (필요시 실제 lung.csv의 대소문자와 똑같이 맞춰주세요)
init_data = pd.DataFrame(
    [[10.0, 40.0, 5.0]], columns=["Smoking", "Age", "Alcohol"]
)

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
        # 데이터 스케일링 및 예측 (영어 컬럼명 상태로 transform 진행)
        new_patients_scaled = scaler.transform(edited_df)
        pred_clusters = model.predict(new_patients_scaled)

        # 결과 표에 예측된 군집 추가해서 보여주기
        result_df = edited_df.copy()
        result_df["예측 군집"] = pred_clusters

        # 시각화 구분을 위해 신규 데이터에 유형 태그 달기
        result_df["데이터 유형"] = "신규 입력 환자"

        st.write("---")
        st.subheader("🔮 예측 결과")
        # '데이터 유형' 칼럼은 표에서 숨기고 출력
        st.dataframe(
            result_df.drop(columns=["데이터 유형"]), use_container_width=True
        )

        # -------------------------------------------------------------
        # 4. 내장 기능을 이용한 시각화 (Matplotlib/Seaborn 대체)
        # -------------------------------------------------------------
        st.write("---")
        st.subheader("📊 군집 내 환자 위치 시각화")

        # 기존 전체 데이터의 군집 결과 구하기 (배경용)
        # 💡 [에러 해결 point] '흡연, 나이, 알코올' 대신 영어 컬럼명을 명시합니다.
        if "군집" not in df.columns:
            df_scaled = scaler.transform(df[["Smoking", "Age", "Alcohol"]])
            df["군집"] = model.predict(df_scaled)

        # 기존 데이터 타입 태그 달기
        chart_df = df.copy()
        chart_df["데이터 유형"] = chart_df["군집"].apply(
            lambda x: f"기존 데이터 (군집 {x})"
        )

        # 기존 데이터와 신규 입력 데이터 합치기
        combined_df = pd.concat([chart_df, result_df], ignore_index=True)

        # 축 선택 인터페이스 (영어 컬럼명 리스트로 명시)
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox(
                "X축 선택", ["Smoking", "Age", "Alcohol"], index=0
            )
        with col2:
            y_axis = st.selectbox(
                "Y축 선택", ["Smoking", "Age", "Alcohol"], index=1
            )

        # Streamlit 내장 산점도 차트 그리기
        st.scatter_chart(
            data=combined_df,
            x=x_axis,
            y=y_axis,
            color="데이터 유형",
            size="데이터 유형",
            use_container_width=True,
        )

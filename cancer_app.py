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

# 💡 [자동 인식] lung.csv 파일에서 데이터 학습에 사용된 첫 3개 컬럼명을 자동으로 가져옵니다.
csv_columns = df.columns[:3].tolist()

# -------------------------------------------------------------
# 2. 사용자 데이터 직접 입력 표 (Data Editor)
# -------------------------------------------------------------
st.subheader("📝 환자 데이터 입력")
st.markdown(
    "아래 표의 값을 더블클릭하여 수정하세요. 행을 추가하여 여러 명을 입력할 수도 있습니다."
)

# 초기 예시 데이터 구성 (CSV 파일의 실제 컬럼명과 자동으로 일치시킵니다)
init_data = pd.DataFrame([[10.0, 40.0, 5.0]], columns=csv_columns)

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
        # 💡 에러가 났던 부분: csv에서 자동으로 추출한 컬럼명을 넣어줍니다.
        if "군집" not in df.columns:
            df_scaled = scaler.transform(df[csv_columns])
            df["군집"] = model.predict(df_scaled)

        # 기존 데이터 타입 태그 달기 (군집 번호를 시각화 범례에 예쁘게 노출)
        chart_df = df.copy()
        chart_df["데이터 유형"] = chart_df["군집"].apply(
            lambda x: f"기존 데이터 (군집 {x})"
        )

        # 기존 데이터와 신규 입력 데이터 합치기
        combined_df = pd.concat([chart_df, result_df], ignore_index=True)

        # 축 선택 인터페이스 (실제 컬럼명 리스트 활용)
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("X축 선택", csv_columns, index=0)
        with col2:
            y_axis = st.selectbox("Y축 선택", csv_columns, index=1)

        # Streamlit 내장 산점도 차트 그리기
        st.scatter_chart(
            data=combined_df,
            x=x_axis,
            y=y_axis,
            color="데이터 유형",  # 군집별 색상 및 신규 환자 색상 분리
            size="데이터 유형",  # 신규 입력 환자를 더 크게 띄우기 위한 장치
            use_container_width=True,
        )

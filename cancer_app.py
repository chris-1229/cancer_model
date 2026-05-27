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

# 스케일러가 기억하는 정확한 컬럼명
try:
    target_columns = scaler.feature_names_in_.tolist()
except AttributeError:
    target_columns = ["Smokes", "Age", "Alkhol"]

# CSV 컬럼명 강제 치환 및 숫자 변환
if len(df.columns) >= len(target_columns):
    rename_dict = {
        df.columns[i]: target_columns[i] for i in range(len(target_columns))
    }
    df = df.rename(columns=rename_dict)

for col in target_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# -------------------------------------------------------------
# 2. 사용자 데이터 직접 입력 표 (Data Editor)
# -------------------------------------------------------------
st.subheader("📝 환자 데이터 입력")
st.markdown(
    "아래 표의 값을 더블클릭하여 수정하세요. 행을 추가하여 여러 명을 입력할 수도 있습니다."
)

init_data = pd.DataFrame([[10.0, 40.0, 5.0]], columns=target_columns)
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
        input_data_scaled = edited_df[target_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        pred_clusters = model.predict(new_patients_scaled := scaler.transform(input_data_scaled))

        result_df = edited_df.copy()
        result_df["예측 군집"] = pred_clusters

        st.write("---")
        st.subheader("🔮 예측 결과")
        st.dataframe(result_df, use_container_width=True)

        # -------------------------------------------------------------
        # 4. 이미지와 똑같은 스타일의 고급 시각화 (Altair 활용 트릭)
        # -------------------------------------------------------------
        st.write("---")
        st.subheader("📍 환자 위치 시각화")

        # 기존 전체 데이터 군집 부여
        if "군집" not in df.columns:
            df_scaled = scaler.transform(df[target_columns])
            df["군집"] = model.predict(df_scaled)

        # 축 선택 인터페이스
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("X축 선택", target_columns, index=1)  # 기본 나이(Age)
        with col2:
            y_axis = st.selectbox("Y축 선택", target_columns, index=0)  # 기본 흡연(Smokes)

        # 💡 [핵심 연출] 추가 설치 없이 고급 레이어 그래프를 그리기 위해 내부 가벼운 컴포넌트(Altair) 연동 기술 사용
        import altair as alt

        # 1) 배경: 기존 환자들 (작고 반투명한 원형 분포)
        bg_chart = (
            alt.Chart(df)
            .mark_circle(size=60, opacity=0.4)
            .encode(
                x=alt.X(x_axis, title=f"{x_axis}"),
                y=alt.Y(y_axis, title=f"{y_axis}"),
                color=alt.Color(
                    "군집:N",
                    scale=alt.Scale(scheme="set2"),
                    legend=alt.Legend(title="기존 군집 분포"),
                ),
            )
        )

        # 2) 강조: 새로 입력한 환자 (크고 선명한 빨간색 'X' 마크)
        new_chart = (
            alt.Chart(result_df)
            .mark_point(
                size=350,
                color="red",
                filled=True,
                shape="cross",
                stroke="black",
                strokeWidth=1.5,
            )
            .encode(x=x_axis, y=y_axis)
        )

        # 3) 두 차트를 하나로 레이어 결합 (+)
        final_chart = (bg_chart + new_chart).properties(
            width=700, height=450
        )

        # 화면에 출력
        st.altair_chart(final_chart, use_container_width=True)

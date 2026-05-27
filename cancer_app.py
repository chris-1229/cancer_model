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

# 스케일러가 기억하는 정확한 컬럼명 3개
target_columns = ["Smokes", "Age", "Alkhol"]

# 💡 [TypeError 방지 핵심 코드]
# 만약 csv에 중복 컬럼이 있거나 이름이 꼬여있더라도, 딱 필요한 첫 3개 열만 잘라내어
# 이름을 강제로 고정시킵니다. 이렇게 하면 df[col]이 무조건 1차원(Series)이 됩니다.
df = df.iloc[:, :3]  # 앞에서부터 딱 3개의 열만 선택
df.columns = target_columns  # 이름을 강제로 부여

# 안전하게 숫자형으로 변환 (이제 무조건 1차원 데이터이므로 에러가 나지 않습니다)
for col in target_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# 알코올 데이터의 평균값 계산 (자동 입력용)
default_alcohol = df["Alkhol"].mean()

# -------------------------------------------------------------
# 2. 사용자 데이터 직접 입력 표 (나이와 흡연량만 노출)
# -------------------------------------------------------------
st.subheader("📝 환자 데이터 입력")
st.markdown(
    "아래 표에 **나이**와 **흡연량**을 입력하세요. 행을 추가해 여러 명을 한 번에 입력할 수도 있습니다."
)

user_input_columns = ["Age", "Smokes"]
init_data = pd.DataFrame([[40.0, 10.0]], columns=user_input_columns)

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
        # 사용자가 입력하지 않은 'Alkhol' 컬럼을 평균값으로 생성해 채워줍니다.
        process_df = edited_df.copy()
        process_df["Alkhol"] = default_alcohol

        # 스케일러 컬럼 순서대로 재정렬
        process_df = process_df[target_columns]

        # 데이터 스케일링 및 군집 예측 실행
        input_data_scaled = process_df.apply(pd.to_numeric, errors="coerce")
        new_patients_scaled = scaler.transform(input_data_scaled)
        pred_clusters = model.predict(new_patients_scaled)

        # 결과 화면 구성
        result_df = edited_df.copy()
        result_df["예측 군집"] = pred_clusters

        st.write("---")
        st.subheader("🔮 예측 결과")
        st.dataframe(result_df, use_container_width=True)

        # -------------------------------------------------------------
        # 4. 이미지 맞춤형 고정 시각화 (나이 vs 흡연량)
        # -------------------------------------------------------------
        st.write("---")
        st.subheader("📍 환자 위치 시각화 (나이 vs 흡연량)")

        # 기존 전체 데이터의 군집 결과 채워넣기
        if "군집" not in df.columns:
            df_scaled = scaler.transform(df[target_columns])
            df["군집"] = model.predict(df_scaled)

        # 차트용 데이터만 분리
        bg_data = df[["Age", "Smokes", "군집"]].copy()
        user_data = result_df[["Age", "Smokes"]].copy()

        import altair as alt

        # 1) 배경: 기존 환자 분포 (작고 반투명한 원)
        bg_chart = (
            alt.Chart(bg_data)
            .mark_circle(size=60, opacity=0.4)
            .encode(
                x=alt.X("Age:Q", title="나이 (Age)"),
                y=alt.Y("Smokes:Q", title="흡연량 (Smoking Amount)"),
                color=alt.Color(
                    "군집:N",
                    scale=alt.Scale(scheme="set2"),
                    legend=alt.Legend(title="기존 군집"),
                ),
            )
        )

        # 2) 강조: 신규 환자 (크고 불투명한 빨간 X 마크)
        new_chart = (
            alt.Chart(user_data)
            .mark_point(
                size=350,
                color="red",
                filled=True,
                shape="cross",
                stroke="black",
                strokeWidth=1.5,
            )
            .encode(x="Age:Q", y="Smokes:Q")
        )

        # 3) 차트 합치기
        final_chart = (bg_chart + new_chart).properties(
            width=700, height=450
        )

        # 화면에 그래프 출력
        st.altair_chart(final_chart, use_container_width=True)

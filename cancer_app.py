import joblib
import pandas as pd
import streamlit as st

# -------------------------------------------------------------
# 1. 페이지 설정 및 모델/데이터 로드
# -------------------------------------------------------------
st.set_page_config(page_title="폐 건강 군집 분석", layout="centered")

# 상단 가이드 UI 추가 (이미지 참고)
st.title("🫁 폐 건강 군집 분석 및 위험도 예측")

with st.expander("ℹ️ 군집별 상태 가이드 (클릭해서 확인)", expanded=True):
    st.markdown("""
    * **0번 군집 (매우 건강):** 흡연량과 알코올 섭취가 매우 낮으며 폐 기능이 최상인 상태입니다.
    * **1번 군집 (건강군):** 일반적인 건강 상태이나 꾸준한 관리가 권장되는 상태입니다.
    * **2번 군집 (주의군):** 중간 정도의 흡연/알코올 수치로, 생활 습관 개선이 필요합니다.
    * **3번 군집 (고위험군):** 높은 흡연량 또는 연령대로 인해 정밀 검진이 강력히 권장됩니다.
    """)

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

# 스케일러 컬럼명 가져오기
try:
    target_columns = scaler.feature_names_in_.tolist()
except AttributeError:
    target_columns = ["Smokes", "Age", "Alkhol"]

# 컬럼 순서 고정 (Smokes, Age, Alkhol 순서라고 가정)
smokes_col = target_columns[0]
age_col = target_columns[1]
alcohol_col = target_columns[2]

# 기존 CSV 데이터 정제
clean_df = pd.DataFrame()
for i, col in enumerate(target_columns):
    if i < len(df.columns):
        clean_df[col] = pd.to_numeric(df.iloc[:, i], errors="coerce").fillna(0)
    else:
        clean_df[col] = 0.0
df = clean_df

# 알코올 평균값
default_alcohol = df[alcohol_col].mean() if df[alcohol_col].mean() != 0 else 5.0

# -------------------------------------------------------------
# 2. 환자 데이터 입력 (나이와 흡연량)
# -------------------------------------------------------------
st.subheader("📝 신규 환자 데이터 입력")
user_input_columns = [age_col, smokes_col]
init_data = pd.DataFrame([[40.0, 10.0]], columns=user_input_columns)

# 사용자가 직접 수정하는 에디터
edited_df = st.data_editor(
    init_data, num_rows="dynamic", use_container_width=True, key="input_editor"
)

# -------------------------------------------------------------
# 3. 군집 예측 실행
# -------------------------------------------------------------
if st.button("🚀 군집 예측 및 결과 확인", type="primary"):
    if edited_df.empty or edited_df.isnull().values.any():
        st.error("모든 칸에 숫자를 입력해주세요.")
    else:
        # 💡 예측을 위한 데이터 재구성
        process_df = edited_df.copy()
        process_df[alcohol_col] = default_alcohol # 알코올은 평균값 주입
        process_df = process_df[target_columns]    # 모델 학습 순서로 정렬
        
        # 숫자 타입 강제 및 스케일링
        numeric_input = process_df.values.astype(float)
        input_scaled = scaler.transform(numeric_input)
        
        # 예측
        pred_clusters = model.predict(input_scaled)

        # 결과 테이블 구성
        result_df = edited_df.copy()
        result_df["예측 군집"] = pred_clusters
        
        # 군집 번호를 텍스트 상태로 변환해주는 매핑 UI 로직
        status_map = {
            0: "매우 건강 (Low Risk)",
            1: "건강군 (Normal)",
            2: "주의군 (Caution)",
            3: "고위험군 (High Risk)"
        }
        result_df["상태 진단"] = result_df["예측 군집"].map(status_map)

        st.write("---")
        st.subheader("🔮 예측 결과 분석")
        
        # 결과 표 출력 (상태 진단 포함)
        st.dataframe(result_df, use_container_width=True)

        # 강조 메시지 출력 (첫 번째 환자 기준)
        top_status = result_df["상태 진단"].iloc[0]
        if "고위험" in top_status:
            st.error(f"⚠️ 진단 결과: **{top_status}** 입니다. 즉각적인 검진이 필요할 수 있습니다.")
        elif "주의" in top_status:
            st.warning(f"💡 진단 결과: **{top_status}** 입니다. 생활 습관 개선을 권장합니다.")
        else:
            st.success(f"✅ 진단 결과: **{top_status}** 입니다. 현재 상태를 유지하세요.")

        # -------------------------------------------------------------
        # 4. 환자 위치 시각화 (나이 vs 흡연량)
        # -------------------------------------------------------------
        st.write("---")
        st.subheader("📍 환자 데이터 시각화 (나이 vs 흡연량)")

        if "군집" not in df.columns:
            df_scaled = scaler.transform(df[target_columns].values.astype(float))
            df["군집"] = model.predict(df_scaled)

        bg_data = df[[age_col, smokes_col, "군집"]].copy()
        user_data = result_df[[age_col, smokes_col]].copy()

        import altair as alt

        bg_chart = alt.Chart(bg_data).mark_circle(size=60, opacity=0.3).encode(
            x=alt.X(f"{age_col}:Q", title="나이 (Age)"),
            y=alt.Y(f"{smokes_col}:Q", title="흡연량 (Smoking)"),
            color=alt.Color("군집:N", scale=alt.Scale(scheme="set2"), legend=alt.Legend(title="군집 분포"))
        )

        new_chart = alt.Chart(user_data).mark_point(
            size=400, color="red", filled=True, shape="cross", stroke="black", strokeWidth=2
        ).encode(x=f"{age_col}:Q", y=f"{smokes_col}:Q")

        st.altair_chart((bg_chart + new_chart).properties(width=700, height=400), use_container_width=True)

import joblib
import pandas as pd
import streamlit as st

# -------------------------------------------------------------
# 1. 페이지 설정 및 모델/데이터 로드
# -------------------------------------------------------------
st.set_page_config(page_title="폐 건강 군집 분석", layout="centered")
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

# 스케일러가 기억하는 진짜 컬럼명과 순서 추출
try:
    target_columns = scaler.feature_names_in_.tolist()
except AttributeError:
    target_columns = ["Smokes", "Age", "Alkhol"]

# 💡 [원인 1번 해결] 단어 포함 여부로 실제 스케일러 안의 정확한 대소문자/이름을 찾아냅니다.
age_col = next((c for c in target_columns if "age" in c.lower()), "Age")
smokes_col = next((c for c in target_columns if "smoke" in c.lower()), "Smokes")
alcohol_col = next((c for c in target_columns if "alk" in c.lower() or "alc" in c.lower()), "Alkhol")

# 기존 CSV 파일의 열들을 스케일러가 요구하는 컬럼명과 순서대로 완벽하게 재매칭합시다.
clean_df = pd.DataFrame()
for col in target_columns:
    # 기존 파일에서 가장 잘 맞는 컬럼을 매치
    matched_col = next((c for c in df.columns if col.lower() in c.lower()), None)
    if matched_col:
        clean_df[col] = pd.to_numeric(df[matched_col], errors="coerce").fillna(0)
    else:
        # 정 안 맞으면 기본값 처리
        if col == age_col: clean_df[col] = 40.0
        elif col == smokes_col: clean_df[col] = 10.0
        else: clean_df[col] = 5.0
df = clean_df

# 알코올 데이터의 평균값 계산 (자동 입력용)
default_alcohol = df[alcohol_col].mean() if df[alcohol_col].mean() != 0 else 5.0

# -------------------------------------------------------------
# 2. 환자 데이터 입력 (나이와 흡연량)
# -------------------------------------------------------------
st.subheader("📝 신규 환자 데이터 입력")

# 사용자는 항상 직관적으로 나이와 흡연량만 입력하도록 고정
user_input_columns = [age_col, smokes_col]
init_data = pd.DataFrame([[40.0, 10.0]], columns=user_input_columns)

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
        # 사용자가 입력한 데이터에 자동으로 알코올 평균값을 결합합니다.
        process_df = edited_df.copy()
        process_df[alcohol_col] = default_alcohol

        # 💡 [핵심] 스케일러가 요구하는 정확한 순서로 컬럼 배열을 강제 정렬합니다.
        process_df = process_df[target_columns]

        # 데이터 스케일링 및 군집 예측 실행
        input_data_scaled = process_df.apply(pd.to_numeric, errors="coerce")
        new_patients_scaled = scaler.transform(input_data_scaled)
        pred_clusters = model.predict(new_patients_scaled)

        # 결과 화면 구성
        result_df = edited_df.copy()
        result_df["예측 군집"] = pred_clusters
        
        status_map = {0: "매우 건강 (Low Risk)", 1: "건강군 (Normal)", 2: "주의군 (Caution)", 3: "고위험군 (High Risk)"}
        result_df["상태 진단"] = result_df["예측 군집"].map(status_map).fillna("분석 완료")

        st.write("---")
        st.subheader("🔮 예측 결과 분석")
        st.dataframe(result_df, use_container_width=True)

        # 강조 메시지 출력
        top_status = result_df["상태 진단"].iloc[0]
        if "고위험" in top_status:
            st.error(f"⚠️ 진단 결과: **{top_status}** 입니다. 즉각적인 검진이 필요할 수 있습니다.")
        elif "주의" in top_status:
            st.warning(f"💡 진단 결과: **{top_status}** 입니다. 생활 습관 개선을 권장합니다.")
        else:
            st.success(f"✅ 진단 결과: **{top_status}** 입니다. 현재 상태를 유지하세요.")

        # -------------------------------------------------------------
        # 4. 이미지 맞춤형 고정 시각화 (나이 vs 흡연량)
        # -------------------------------------------------------------
        st.write("---")
        st.subheader("📍 환자 데이터 시각화 (나이 vs 흡연량)")

        # 기존 전체 데이터의 군집 결과 채워넣기
        if "군집" not in df.columns:
            df_scaled = scaler.transform(df[target_columns])
            df["군집"] = model.predict(df_scaled)

        bg_data = df[[age_col, smokes_col, "군집"]].copy()
        user_data = result_df[[age_col, smokes_col]].copy()

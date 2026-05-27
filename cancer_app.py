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

# 💡 [핵심 해결 지점 1] 스케일러가 기억하는 정확한 컬럼명과 순서를 자동으로 알아냅니다.
try:
    target_columns = scaler.feature_names_in_.tolist()
except AttributeError:
    # 예외 상황용 기본값
    target_columns = ["Smokes", "Age", "Alkhol"]

# 💡 [핵심 해결 지점 2] 스케일러의 진짜 순서에 맞춰 CSV 파일 구조를 강제 통일합니다.
df = df.iloc[:, :3]  # 앞 3개 열 선택
df.columns = target_columns  # 스케일러 이름으로 덮어쓰기

# 안전하게 숫자형으로 변환
for col in target_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# 💡 스케일러 이름 리스트에서 'Age'와 'Smokes'가 몇 번째 인덱스에 있는지 자동으로 찾습니다.
# (스케일러 내부 순서가 어떻게 되어있든 상관없이 에러를 막기 위함)
age_col = next((c for c in target_columns if "age" in c.lower()), "Age")
smokes_col = next((c for c in target_columns if "smoke" in c.lower()), "Smokes")
alcohol_col = next((c for c in target_columns if "alk" in c.lower() or "alc" in c.lower()), "Alkhol")

# 알코올 데이터의 평균값 계산 (자동 입력용)
default_alcohol = df[alcohol_col].mean()

# -------------------------------------------------------------
# 2. 사용자 데이터 직접 입력 표 (나이와 흡연량만 노출)
# -------------------------------------------------------------
st.subheader("📝 환자 데이터 입력")
st.markdown(
    f"아래 표에 **나이**와 **흡연량**을 입력하세요. 행을 추가해 여러 명을 한 번에 입력할 수도 있습니다."
)

# 사용자는 화면에서 'Age'와 'Smokes'만 입력하도록 유도
user_input_columns = [age_col, smokes_col]
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
        # 사용자가 입력하지 않은

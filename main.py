import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from datetime import datetime

# 1. 頁面配置
st.set_page_config(page_title="信用卡違約風險預測展示（產品化 Demo）", layout="wide")

# --- 檔案路徑設定（支援根目錄或資料夾） ---
DATA_CANDIDATES = ["UCI_Credit_Card.csv", "data/UCI_Credit_Card.csv"]
MODEL_FILES = {
    "KNN": ["k-nearest_neighbors_pipeline.joblib", "models/k-nearest_neighbors_pipeline.joblib"],
    "LogisticRegression": ["logistic_regression_pipeline.joblib", "models/logistic_regression_pipeline.joblib"],
    "XGBoost": ["xgboost_classifier_pipeline.joblib", "models/xgboost_classifier_pipeline.joblib"],
}
TARGET_COL = "default.payment.next.month"


def pick_existing_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


@st.cache_data
def load_data():
    data_path = pick_existing_path(DATA_CANDIDATES)
    if data_path is None:
        raise FileNotFoundError("找不到 UCI_Credit_Card.csv，請放在專案根目錄或 data/ 資料夾。")

    df = pd.read_csv(data_path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"找不到目標欄位：{TARGET_COL}")

    # X = 刪除欄位[ID, default.payment.next.month]
    drop_cols = [TARGET_COL]
    if "ID" in df.columns:
        drop_cols.insert(0, "ID")

    X = df.drop(columns=drop_cols)

    # y = default.payment.next.month
    y = df[TARGET_COL]

    return df, X, y


@st.cache_resource
def load_model(model_name):
    model_path = pick_existing_path(MODEL_FILES[model_name])
    if model_path is None:
        raise FileNotFoundError(f"找不到模型檔：{MODEL_FILES[model_name]}")
    return joblib.load(model_path)


@st.cache_resource
def load_all_models():
    """一次載入所有模型（demo 版本：可做機率對照表）"""
    models = {}
    for name in MODEL_FILES.keys():
        models[name] = load_model(name)
    return models


# =========================
# ✅ 核心修正：欄位對齊
# =========================
def align_features(model, sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    把 sample_df 對齊成模型期待的欄位：
    - 缺欄補 0
    - 多欄丟掉
    - 依模型期待順序排序
    """
    expected = getattr(model, "feature_names_in_", None)
    if expected is None:
        return sample_df

    expected = list(expected)
    out = sample_df.copy()

    for c in expected:
        if c not in out.columns:
            out[c] = 0

    return out[expected]


def predict_prob(model, sample_df):
    """回傳違約機率 prob（若無 predict_proba 則回傳 None）"""
    sample_df = align_features(model, sample_df)

    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(sample_df)[0][1])
    return None


def predict_label(model, sample_df):
    """回傳模型預測 label（保證走欄位對齊）"""
    sample_df = align_features(model, sample_df)
    return int(model.predict(sample_df)[0])


def risk_level(prob: float | None) -> str:
    """簡單風險分級（展示用）"""
    if prob is None:
        return "N/A"
    if prob < 0.25:
        return "低風險"
    if prob < 0.50:
        return "中風險"
    return "高風險"


def label_text(y_val: int) -> str:
    return "⚠️ 違約 (1)" if y_val == 1 else "✅ 正常 (0)"


def decision_text(pred: int) -> str:
    return "⚠️ 違約" if pred == 1 else "✅ 正常"


# --- Session State 初始化（產品化：抽樣與歷史紀錄） ---
if "sample_idx" not in st.session_state:
    st.session_state.sample_idx = None

if "history" not in st.session_state:
    st.session_state.history = []  # list[dict]

if "last_log_key" not in st.session_state:
    st.session_state.last_log_key = None


def pick_random_idx(n: int) -> int:
    return int(np.random.randint(0, n))


def log_event(idx: int, actual: int, model_name: str, prob: float | None, pred: int, thr: float, correct: bool):
    st.session_state.history.insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "idx": idx,
        "actual": actual,
        "model": model_name,
        "prob": None if prob is None else round(prob, 6),
        "threshold": round(thr, 2),
        "pred": pred,
        "correct": correct,
        "risk": risk_level(prob),
    })
    st.session_state.history = st.session_state.history[:30]


def clear_history():
    st.session_state.history = []


def set_idx(idx: int, n: int):
    idx = int(idx)
    idx = max(0, min(idx, n - 1))
    st.session_state.sample_idx = idx


# --- 左側選單 ---
st.sidebar.title("🤖 模型控制中心")

selected_model = st.sidebar.selectbox("主模型（主畫面顯示用）：", ["KNN", "LogisticRegression", "XGBoost"])
threshold = st.sidebar.slider("Decision Threshold", 0.10, 0.90, 0.25, 0.05)

st.sidebar.divider()
st.sidebar.caption("說明：threshold 越低 → 越容易判定違約（Recall ↑，但可能誤殺更多正常客戶）。")

st.sidebar.subheader("🧾 展示紀錄")
if st.sidebar.button("🧹 清空紀錄", use_container_width=True):
    clear_history()
st.sidebar.caption("（最多保留 30 筆抽樣紀錄）")

# --- 主畫面 ---
st.title("💳 信用卡違約風險預測展示（產品化 Demo）")

df_full, X, y = load_data()
models = load_all_models()

# A. 資料概覽
with st.expander("📋 數據集概覽（前 10 筆）", expanded=True):
    st.dataframe(df_full.head(10), use_container_width=True)

# B. y 分佈（表 + 圖）
with st.expander("📊 目標變數 y 分佈", expanded=True):
    y_counts = y.value_counts().rename_axis("class").reset_index(name="count")
    st.dataframe(y_counts, use_container_width=True)

    y_plot = y_counts.copy()
    y_plot["class"] = y_plot["class"].astype(str)
    st.bar_chart(y_plot.set_index("class")["count"])
    st.caption("y=0 代表正常、y=1 代表違約（類別不平衡常見，因此不建議只看 Accuracy）")

st.divider()

left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("🎯 隨機抽樣預測")

    btn_row = st.columns([1, 1, 2])

    with btn_row[0]:
        if st.button("🎲 Random", use_container_width=True):
            st.session_state.sample_idx = pick_random_idx(len(X))

    with btn_row[1]:
        if st.button("➡️ Next", use_container_width=True):
            if st.session_state.sample_idx is None:
                st.session_state.sample_idx = pick_random_idx(len(X))
            else:
                st.session_state.sample_idx = int((st.session_state.sample_idx + 1) % len(X))

    with btn_row[2]:
        st.write(f"目前 threshold：**{threshold:.2f}**（可在左側調整）")

    st.write("#### 🔎 指定樣本索引（Idx）")
    idx_input = st.number_input(
        "輸入 0 ~ (資料筆數-1) 的索引",
        min_value=0,
        max_value=len(X) - 1,
        value=int(st.session_state.sample_idx) if st.session_state.sample_idx is not None else 0,
        step=1
    )
    if st.button("✅ 以此 Idx 顯示並預測", use_container_width=True):
        set_idx(idx_input, len(X))

    if st.session_state.sample_idx is None:
        st.info("請按下 Random / Next，或輸入 Idx 後按「以此 Idx 顯示並預測」。")
        st.stop()

    idx = st.session_state.sample_idx
    sample_data = X.iloc[[idx]]
    actual = int(y.iloc[idx])

    st.write(f"**抽樣索引：** `{idx}`")
    st.dataframe(sample_data, use_container_width=True)

    st.write("### ✅ 真實情況")
    st.metric("真實標籤", label_text(actual))

    st.divider()

    # ✅ Debug：顯示主模型期待欄位
    with st.expander("🧩 Column Check（模型期待欄位）", expanded=False):
        main_m = models[selected_model]
        cols = getattr(main_m, "feature_names_in_", None)
        if cols is None:
            st.write("此模型沒有 feature_names_in_，無法顯示期待欄位（可能是舊版或不同 pipeline）。")
        else:
            st.write(list(cols))

    st.subheader("🧪 三模型違約機率對照（同一筆資料）")

    rows = []
    for name, m in models.items():
        prob = predict_prob(m, sample_data)

        if prob is None:
            pred_raw = predict_label(m, sample_data)
            rows.append({
                "Model": name,
                "Default Prob": "N/A",
                "Decision(thr)": f"{decision_text(pred_raw)}（無機率）",
                "Risk": "N/A"
            })
        else:
            pred_thr = int(prob >= threshold)
            rows.append({
                "Model": name,
                "Default Prob": f"{prob:.2%}",
                "Decision(thr)": decision_text(pred_thr),
                "Risk": risk_level(prob)
            })

    compare_df = pd.DataFrame(rows).sort_values(by="Model")
    st.dataframe(compare_df, use_container_width=True)
    st.caption("展示亮點：同一筆客戶資料，不同模型可能給出不同風險評估。")

    st.divider()

    st.subheader("⭐ 主模型結果（你左側選的那個）")
    main_model = models[selected_model]
    main_prob = predict_prob(main_model, sample_data)

    if main_prob is None:
        main_pred = predict_label(main_model, sample_data)
        used_prob = None
    else:
        main_pred = int(main_prob >= threshold)
        used_prob = main_prob

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("主模型", selected_model)
    with c2:
        st.metric("風險等級", risk_level(used_prob))
    with c3:
        st.metric("違約機率", f"{used_prob:.2%}" if used_prob is not None else "N/A")
    with c4:
        st.metric("模型判定(thr)", decision_text(main_pred))

    if used_prob is not None:
        st.write("### 🎚️ 違約機率視覺化")
        st.progress(min(max(used_prob, 0.0), 1.0))
        st.caption(f"prob={used_prob:.2%} vs threshold={threshold:.2f} → 判定：{decision_text(int(used_prob >= threshold))}")

    st.info("風控場景通常更在意 FN（把違約判成正常），可透過調整 threshold 提高 Recall（但 FP 可能上升）。")

    ok = (main_pred == actual)
    if ok:
        st.success("🎉 主模型判定與真實情況一致")
    else:
        st.error("❌ 主模型判定與真實情況不一致（邊界樣本/類別不平衡很常見）")

    dedupe_key = (idx, selected_model, round(threshold, 2))
    if st.session_state.last_log_key != dedupe_key:
        log_event(
            idx=idx,
            actual=actual,
            model_name=selected_model,
            prob=used_prob,
            pred=main_pred,
            thr=threshold,
            correct=ok
        )
        st.session_state.last_log_key = dedupe_key


with right:
    st.subheader("🧾 抽樣歷史紀錄（Log）")

    if len(st.session_state.history) == 0:
        st.write("目前沒有紀錄。按 Random/Next 或指定 Idx 後預測，會自動新增。")
    else:
        hist_df = pd.DataFrame(st.session_state.history)

        show_df = hist_df.copy()
        show_df["actual"] = show_df["actual"].map(lambda v: "1(違約)" if v == 1 else "0(正常)")
        show_df["pred"] = show_df["pred"].map(lambda v: "1(違約)" if v == 1 else "0(正常)")
        show_df["prob"] = show_df["prob"].map(lambda v: "N/A" if pd.isna(v) else f"{float(v):.2%}")
        show_df["correct"] = show_df["correct"].map(lambda v: "✅" if v else "❌")
import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from datetime import datetime

st.set_page_config(page_title="信用卡違約風險預測（Demo）", layout="wide")

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

@st.cache_data(show_spinner=False)
def load_data():
    data_path = pick_existing_path(DATA_CANDIDATES)
    if data_path is None:
        raise FileNotFoundError("找不到 UCI_Credit_Card.csv，請放在根目錄或 data/")

    df = pd.read_csv(data_path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"找不到目標欄位：{TARGET_COL}")

    drop_cols = [TARGET_COL]
    if "ID" in df.columns:
        drop_cols.insert(0, "ID")

    X = df.drop(columns=drop_cols)
    y = df[TARGET_COL].astype(int)
    return df, X, y

@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    model_path = pick_existing_path(MODEL_FILES[model_name])
    if model_path is None:
        raise FileNotFoundError(f"找不到模型檔：{MODEL_FILES[model_name]}")
    return joblib.load(model_path)

def predict_prob(model, sample_df):
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(sample_df)[0][1])
    return None

def risk_level(prob):
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

# Session state
if "sample_idx" not in st.session_state:
    st.session_state.sample_idx = None
if "history" not in st.session_state:
    st.session_state.history = []
if "last_log_key" not in st.session_state:
    st.session_state.last_log_key = None

def log_event(idx, actual, model_name, prob, pred, thr, correct):
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

# Sidebar
st.sidebar.title("🤖 模型控制中心")
selected_model = st.sidebar.selectbox("主模型：", ["KNN", "LogisticRegression", "XGBoost"])
threshold = st.sidebar.slider("Decision Threshold", 0.10, 0.90, 0.45, 0.05)
st.sidebar.caption("threshold 越高 → 越不容易判定違約（更保守）")

if st.sidebar.button("🧹 清空紀錄", use_container_width=True):
    st.session_state.history = []

st.title("💳 信用卡違約風險預測（產品化 Demo）")

with st.spinner("讀取資料中..."):
    df_full, X, y = load_data()

# Data preview（不要 expanded=True，避免一開始就渲染太多）
with st.expander("📋 數據集概覽（前 10 筆）", expanded=False):
    st.dataframe(df_full.head(10), use_container_width=True)

with st.expander("📊 y 分佈", expanded=False):
    y_counts = y.value_counts().rename_axis("class").reset_index(name="count")
    st.dataframe(y_counts, use_container_width=True)
    y_plot = y_counts.copy()
    y_plot["class"] = y_plot["class"].astype(str)
    st.bar_chart(y_plot.set_index("class")["count"])

st.divider()

left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("🎯 抽樣預測")

    cA, cB, cC = st.columns([1, 1, 2])
    with cA:
        if st.button("🎲 Random", use_container_width=True):
            st.session_state.sample_idx = int(np.random.randint(0, len(X)))
    with cB:
        if st.button("➡️ Next", use_container_width=True):
            if st.session_state.sample_idx is None:
                st.session_state.sample_idx = int(np.random.randint(0, len(X)))
            else:
                st.session_state.sample_idx = int((st.session_state.sample_idx + 1) % len(X))
    with cC:
        st.write(f"目前 threshold：**{threshold:.2f}**")

    idx_input = st.number_input(
        "指定 idx（0 ~ 資料筆數-1）",
        min_value=0,
        max_value=len(X)-1,
        value=int(st.session_state.sample_idx) if st.session_state.sample_idx is not None else 0,
        step=1
    )
    if st.button("✅ 使用此 idx", use_container_width=True):
        st.session_state.sample_idx = int(idx_input)

    if st.session_state.sample_idx is None:
        st.info("請按 Random / Next 或輸入 idx。")
        st.stop()

    idx = st.session_state.sample_idx
    sample_data = X.iloc[[idx]]
    actual = int(y.iloc[idx])

    st.write(f"**Idx：** `{idx}`")
    st.dataframe(sample_data, use_container_width=True)

    st.write("### ✅ 真實情況")
    st.metric("真實標籤", label_text(actual))

    st.divider()

    # ✅ 主模型：只載這一個
    st.subheader("⭐ 主模型結果")
    with st.spinner(f"載入模型：{selected_model} ..."):
        main_model = load_model(selected_model)

    main_prob = predict_prob(main_model, sample_data)
    if main_prob is None:
        main_pred = int(main_model.predict(sample_data)[0])
        used_prob = None
    else:
        main_pred = int(main_prob >= threshold)
        used_prob = main_prob

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("主模型", selected_model)
    d2.metric("風險等級", risk_level(used_prob))
    d3.metric("違約機率", f"{used_prob:.2%}" if used_prob is not None else "N/A")
    d4.metric("模型判定(thr)", decision_text(main_pred))

    if used_prob is not None:
        st.progress(min(max(used_prob, 0.0), 1.0))
        st.caption(f"prob={used_prob:.2%} vs threshold={threshold:.2f}")

    ok = (main_pred == actual)
    st.success("🎉 判定一致") if ok else st.error("❌ 判定不一致（很正常）")

    dedupe_key = (idx, selected_model, round(threshold, 2))
    if st.session_state.last_log_key != dedupe_key:
        log_event(idx, actual, selected_model, used_prob, main_pred, threshold, ok)
        st.session_state.last_log_key = dedupe_key

    st.divider()

    # ✅ 三模型比較：改成按按鈕才跑（避免啟動慢）
    st.subheader("🧪 三模型比較（點按鈕才計算）")
    if st.button("▶️ 計算三模型比較", use_container_width=True):
        rows = []
        for name in ["KNN", "LogisticRegression", "XGBoost"]:
            with st.spinner(f"載入模型：{name} ..."):
                m = load_model(name)
            prob = predict_prob(m, sample_data)
            if prob is None:
                pred_raw = int(m.predict(sample_data)[0])
                rows.append({"Model": name, "Default Prob": "N/A", "Decision": decision_text(pred_raw), "Risk": "N/A"})
            else:
                pred_thr = int(prob >= threshold)
                rows.append({"Model": name, "Default Prob": f"{prob:.2%}", "Decision": decision_text(pred_thr), "Risk": risk_level(prob)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

with right:
    st.subheader("🧾 抽樣歷史紀錄")
    if not st.session_state.history:
        st.write("目前沒有紀錄。")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        show_df = hist_df.copy()
        show_df["actual"] = show_df["actual"].map(lambda v: "1(違約)" if v == 1 else "0(正常)")
        show_df["pred"] = show_df["pred"].map(lambda v: "1(違約)" if v == 1 else "0(正常)")
        show_df["prob"] = show_df["prob"].map(lambda v: "N/A" if pd.isna(v) else f"{float(v):.2%}")
        show_df["correct"] = show_df["correct"].map(lambda v: "✅" if v else "❌")
        show_df = show_df[["time", "idx", "model", "threshold", "prob", "risk", "pred", "actual", "correct"]]
        show_df.columns = ["Time", "Idx", "Model", "Thr", "Prob", "Risk", "Pred", "Actual", "OK"]
        st.dataframe(show_df, use_container_width=True, height=520)

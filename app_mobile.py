# ===============================
# 厳選AIフル実装 mobile版
# ===============================

import pandas as pd
import streamlit as st

# -------------------------------
# 仮：予想生成（既存のを使ってOK）
# -------------------------------
def generate_predictions(df):
    # 既存のpredict.py使う前提
    return df


# -------------------------------
# 安全変換
# -------------------------------
def safe_float(x):
    try:
        return float(x)
    except:
        return 0.0


# -------------------------------
# ROIランク
# -------------------------------
def apply_roi_ranking(df):
    if df is None or df.empty:
        return df

    out = df.copy()

    def judge(row):
        roi = safe_float(row.get("期待値", 0))

        if roi >= 140:
            return "🔥 AI推奨"
        elif roi >= 120:
            return "💰 期待値高"
        elif roi >= 105:
            return "🟡 穴"
        else:
            return "⚪ 抑え"

    out["買い目ランク"] = out.apply(judge, axis=1)
    return out


# -------------------------------
# 厳選AI（ここが本体）
# -------------------------------
def strict_filter(df, min_roi=110, max_n=12):
    if df is None or df.empty:
        return df

    out = df.copy()

    # ROIで削る
    if "期待値" in out.columns:
        out = out[out["期待値"] >= min_roi]

    # 全部消えたら救済
    if len(out) == 0:
        out = df.copy()

    # 上位抽出
    if "期待値" in out.columns:
        out = out.sort_values("期待値", ascending=False)

    # 最低3点保証
    return out.head(max(max_n, 3))


# -------------------------------
# 金額配分
# -------------------------------
def apply_amount(df, unit=100):
    if df is None or df.empty:
        return df

    out = df.copy()

    def amount(row):
        rank = row.get("買い目ランク", "")

        if rank == "🔥 AI推奨":
            return unit * 3
        elif rank == "💰 期待値高":
            return unit * 2
        elif rank == "🟡 穴":
            return unit
        else:
            return unit

    out["購入金額"] = out.apply(amount, axis=1)
    return out


# -------------------------------
# UI
# -------------------------------
st.title("競輪AI mobile（厳選AI版）")

# ダミー入力
data = pd.DataFrame({
    "買い目": ["1-2-3", "1-3-5", "2-3-4", "4-5-6"],
    "期待値": [150, 130, 105, 90]
})

df = st.dataframe(data)

if st.button("予想実行"):
    pred = generate_predictions(data)

    # 厳選AI
    pred = strict_filter(pred, min_roi=110, max_n=10)

    # ランク
    pred = apply_roi_ranking(pred)

    # 金額
    pred = apply_amount(pred)

    st.write("厳選後")
    st.dataframe(pred)

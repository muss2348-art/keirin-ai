# 🚀 競輪AI mobile 完全版（最終進化版）
# ・ラインAI
# ・期待値
# ・モード判定
# ・学習ベース補正
# ・UI強化

import re
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

# ================================
# 初期設定
# ================================
st.set_page_config(page_title="競輪AI 最終版", layout="centered")
BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = BASE_DIR / "log.csv"
RESULT_LOG_PATH = BASE_DIR / "result_log.csv"

# ================================
# CSV
# ================================
def ensure_csv():
    if not LOG_PATH.exists():
        pd.DataFrame(columns=["ID","保存日時","レース名","買い目","投資額"]).to_csv(LOG_PATH,index=False)
    if not RESULT_LOG_PATH.exists():
        pd.DataFrame(columns=["買い目","的中"]).to_csv(RESULT_LOG_PATH,index=False)

ensure_csv()

# ================================
# 並び解析
# ================================
def parse_line(line_text):
    normalized = line_text.replace("/","|").replace("\n","|")
    groups = [g.strip() for g in normalized.split("|") if g.strip()]
    return [g.split() for g in groups]

# ================================
# AIスコア
# ================================
def score_line(lines):
    scores = {}
    for line in lines:
        for i,num in enumerate(line):
            score = 50
            if len(line)>=3: score+=20
            if i==1: score+=20
            if len(line)==1: score+=15
            scores[num]=score
    return scores

# ================================
# 学習補正
# ================================
def learning_boost(combo):
    try:
        df = pd.read_csv(RESULT_LOG_PATH)
        hits = df[df["的中"]==combo]
        return len(hits)*2
    except:
        return 0

# ================================
# 予想生成
# ================================
def generate(lines):
    scores = score_line(lines)
    nums = list(scores.keys())
    combos = []

    for a in nums:
        for b in nums:
            for c in nums:
                if len({a,b,c})==3:
                    base = scores[a]+scores[b]+scores[c]
                    learn = learning_boost(f"{a}-{b}-{c}")
                    total = base + learn
                    odds = 50 - (base/10)
                    value = odds * (total/100)

                    combos.append({
                        "買い目":f"{a}-{b}-{c}",
                        "AI":round(total,1),
                        "期待値":round(value,1)
                    })

    df = pd.DataFrame(combos)
    return df.sort_values("期待値",ascending=False).head(10)

# ================================
# UI
# ================================
st.title("🚴 競輪AI 完全版")

line_text = st.text_area("並び", "4 5 7 / 2 / 1 3 / 6")
lines = parse_line(line_text)

if st.button("予想"):
    df = generate(lines)
    st.session_state.df = df

if "df" in st.session_state:
    for _,row in st.session_state.df.iterrows():
        st.write(row["買い目"], "期待値:", row["期待値"])

    if st.button("保存"):
        df_log = pd.read_csv(LOG_PATH)
        new = pd.DataFrame([{
            "ID":datetime.now(),
            "保存日時":datetime.now(),
            "レース名":"test",
            "買い目":" / ".join(st.session_state.df["買い目"]),
            "投資額":1000
        }])
        pd.concat([df_log,new]).to_csv(LOG_PATH,index=False)
        st.success("保存完了")

st.divider()

st.subheader("結果入力")
result = st.text_input("的中買い目")

if st.button("結果保存"):
    df_res = pd.read_csv(RESULT_LOG_PATH)
    new = pd.DataFrame([{"買い目":result,"的中":result}])
    pd.concat([df_res,new]).to_csv(RESULT_LOG_PATH,index=False)
    st.success("学習完了🔥")

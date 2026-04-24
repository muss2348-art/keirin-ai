# app_mobile.py
# -*- coding: utf-8 -*-

import re
import itertools
import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

from predict import auto_detect_mode, generate_predictions

st.set_page_config(page_title="競輪AI Mobile", layout="centered")
st.title("🚴 競輪AI Mobile")

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "ja",
}

DEFAULT_COLUMNS = ["車番","選手名","競走得点","脚質","ライン","ライン順","単騎"]

# =========================
# 共通
# =========================
def normalize_text(s):
    if not s:
        return ""
    s = str(s)
    s = s.replace("　"," ")
    s = re.sub(r"\s+"," ",s)
    return s.strip()

def safe_float(v, default=0.0):
    try:
        return float(str(v).replace(",",""))
    except:
        return default

# =========================
# 状態
# =========================
def init_state(n=7):
    st.session_state["race_rows"] = [
        {"車番":i,"選手名":"","競走得点":0.0,"脚質":"","ライン":0,"ライン順":0,"単騎":0}
        for i in range(1,n+1)
    ]
    st.session_state["pred_df"] = None

if "race_rows" not in st.session_state:
    init_state(7)

def get_df():
    return pd.DataFrame(st.session_state["race_rows"])

def set_df(df):
    st.session_state["race_rows"] = df.to_dict("records")

# =========================
# HTML取得
# =========================
def get_page(url):
    r = requests.get(url, headers=HEADERS, timeout=10)
    r.encoding = r.apparent_encoding
    soup = BeautifulSoup(r.text, "html.parser")
    return normalize_text(soup.get_text())

# =========================
# ★ここが今回の修正（最重要）
# =========================
def extract_single_player_by_car(text, car, num_riders):
    s = normalize_text(text)

    # 全車番の開始位置を取得
    positions = []
    for n in range(1, num_riders + 1):
        m = re.search(rf"(?<!\d){n}\s", s)
        if m:
            positions.append((n, m.start()))

    positions = sorted(positions, key=lambda x: x[1])

    block = ""
    for i,(n,pos) in enumerate(positions):
        if n != car:
            continue

        # 最後の車番でも切れないようにする
        end = positions[i+1][1] if i+1 < len(positions) else len(s)
        block = s[pos:end]
        break

    if not block:
        return None

    # 名前
    name_match = re.search(r"[一-龥ぁ-んァ-ヶ々]{2,}", block)
    name = name_match.group(0) if name_match else ""

    # 得点
    score_match = re.search(r"([4-9]\d\.\d)", block)
    score = safe_float(score_match.group(1)) if score_match else 0

    # 脚質
    style_match = re.search(r"(逃|捲|追|両)", block)
    style = style_match.group(1) if style_match else ""

    if name and score > 0:
        return {"車番":car,"選手名":name,"競走得点":score,"脚質":style}

    return None

# =========================
# 選手取得
# =========================
def fetch_players(url, n):
    text = get_page(url)

    rows = []
    for i in range(1,n+1):
        r = extract_single_player_by_car(text,i,n)
        if r:
            rows.append(r)

    return pd.DataFrame(rows)

# =========================
# UI
# =========================
n = st.selectbox("車立て",[5,6,7,9],index=2)

if st.button("初期化"):
    init_state(n)
    st.rerun()

url = st.text_input("URL")

if st.button("選手取得"):
    try:
        df = fetch_players(url,n)
        base = get_df()

        for _,r in df.iterrows():
            base.loc[base["車番"]==r["車番"],"選手名"] = r["選手名"]
            base.loc[base["車番"]==r["車番"],"競走得点"] = r["競走得点"]
            base.loc[base["車番"]==r["車番"],"脚質"] = r["脚質"]

        set_df(base)
        st.success("取得成功")
        st.rerun()

    except Exception as e:
        st.error(e)

df = get_df()
st.dataframe(df)

# =========================
# 予想
# =========================
if st.button("予想"):
    try:
        pred = generate_predictions(
            df,
            mode=auto_detect_mode(df),
            weather="晴",
            top_n=10,
            odds_dict={},
            ticket_type="3連単"
        )
        st.dataframe(pred)
    except Exception as e:
        st.error(e)

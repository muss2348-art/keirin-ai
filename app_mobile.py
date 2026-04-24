# app_mobile.py
# -*- coding: utf-8 -*-

import re
import itertools
from datetime import datetime
import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

from predict import auto_detect_mode, generate_predictions


# =========================
# 基本設定
# =========================
st.set_page_config(page_title="競輪AI（Mobile）", layout="centered")

HEADERS = {"User-Agent": "Mozilla/5.0"}

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def safe_float(x):
    try:
        return float(x)
    except:
        return 0.0


# =========================
# 初期化
# =========================
def init_state(n=7):
    st.session_state["num"] = n
    st.session_state["rows"] = [
        {"車番": i+1, "選手名": "", "競走得点": 0, "脚質": "", "ライン": 0, "ライン順": 0, "単騎": 0}
        for i in range(n)
    ]
    st.session_state["pred"] = None
    st.session_state["odds"] = {}

if "rows" not in st.session_state:
    init_state(7)


# =========================
# 並び処理
# =========================
def parse_lineup(text):
    groups = []
    parts = re.split(r"[ /]", text)
    for p in parts:
        nums = re.findall(r"[1-9]", p)
        if nums:
            groups.append([int(x) for x in nums])
    return groups


def apply_lineup(df, text):
    groups = parse_lineup(text)

    df["ライン"] = 0
    df["ライン順"] = 0
    df["単騎"] = 0

    line_id = 1
    for g in groups:
        if len(g) == 1:
            df.loc[df["車番"] == g[0], "単騎"] = 1
        else:
            for i, car in enumerate(g):
                df.loc[df["車番"] == car, "ライン"] = line_id
                df.loc[df["車番"] == car, "ライン順"] = i + 1
            line_id += 1
    return df


# =========================
# URL取得
# =========================
def fetch_html(url):
    r = requests.get(url, headers=HEADERS, timeout=10)
    return BeautifulSoup(r.text, "html.parser").get_text()


def fetch_lineup(url):
    text = fetch_html(url)
    m = re.search(r'([1-9\-\/ ]{5,})', text)
    if m:
        return m.group(1)
    raise ValueError("並び取得失敗")


def fetch_players(url, n):
    text = fetch_html(url)

    rows = []
    for i in range(1, n+1):
        m = re.search(rf"{i}.*?([一-龥]+).*?([4-9]\d\.\d)", text)
        if m:
            rows.append({
                "車番": i,
                "選手名": m.group(1),
                "競走得点": float(m.group(2)),
                "脚質": ""
            })

    return pd.DataFrame(rows)


def fetch_odds(url):
    text = fetch_html(url)
    odds = {}

    for m in re.finditer(r'([1-9]-[1-9]-[1-9])\s+([0-9]+\.?[0-9]*)', text):
        odds[m.group(1)] = float(m.group(2))

    return odds


# =========================
# UI
# =========================
st.title("🚴競輪AI（Mobile）")

# 車立て
num = st.radio("車立て", [5,6,7,9], horizontal=True)

if num != st.session_state["num"]:
    init_state(num)
    st.rerun()

# レース種別
race_type = st.selectbox("レース種別", ["通常", "ガールズ"])

ticket_type = st.selectbox("券種", ["3連単", "2車単"])

count = st.slider("点数", 3, 30, 10)
unit = st.number_input("金額", 100, 10000, 100, step=100)

url = st.text_input("URL")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("並び"):
        try:
            lineup = fetch_lineup(url)
            df = pd.DataFrame(st.session_state["rows"])
            df = apply_lineup(df, lineup)
            st.session_state["rows"] = df.to_dict("records")
        except Exception as e:
            st.error(e)

with col2:
    if st.button("選手"):
        try:
            df = pd.DataFrame(st.session_state["rows"])
            p = fetch_players(url, num)
            df.update(p)
            st.session_state["rows"] = df.to_dict("records")
        except Exception as e:
            st.error(e)

with col3:
    if st.button("オッズ"):
        try:
            st.session_state["odds"] = fetch_odds(url)
        except Exception as e:
            st.error(e)


# 入力
df = pd.DataFrame(st.session_state["rows"])

for i in range(len(df)):
    cols = st.columns(4)
    df.at[i, "選手名"] = cols[0].text_input(f"name{i}", df.at[i, "選手名"])
    df.at[i, "競走得点"] = cols[1].number_input(f"score{i}", value=float(df.at[i, "競走得点"]))
    df.at[i, "脚質"] = cols[2].text_input(f"type{i}", df.at[i, "脚質"])
    df.at[i, "単騎"] = cols[3].selectbox(f"single{i}", [0,1], index=int(df.at[i,"単騎"]))

st.session_state["rows"] = df.to_dict("records")


# =========================
# 予想
# =========================
if st.button("予想する"):
    try:
        mode = auto_detect_mode(df)

        pred = generate_predictions(
            df,
            mode=mode,
            top_n=count,
            odds_dict=st.session_state["odds"],
            ticket_type=ticket_type,
            race_type=race_type
        )

        pred["購入金額"] = unit
        st.session_state["pred"] = pred

    except Exception as e:
        st.error(f"エラー: {e}")


# =========================
# 表示
# =========================
pred = st.session_state.get("pred")

if pred is not None:
    st.dataframe(pred)

    st.metric("合計", int(pred["購入金額"].sum()))

# app_mobile_g3.py
# -*- coding: utf-8 -*-

import re
import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

from predict import auto_detect_mode, generate_predictions

st.set_page_config(page_title="競輪AI mobile G3", layout="wide")

HEADERS = {"User-Agent": "Mozilla/5.0"}

# =========================
# 共通
# =========================
def normalize_text(s):
    if not s:
        return ""
    return str(s).replace("\n", " ").strip()

def is_valid_player_name(name):
    return bool(re.fullmatch(r"[一-龥ぁ-んァ-ヶ々]{2,8}", name))

def safe_float(v, default=0.0):
    try:
        return float(v)
    except:
        return default

# =========================
# G3専用：選手取得（完全安定版）
# =========================
def extract_players_g3(html, num_riders):
    soup = BeautifulSoup(html, "html.parser")

    cards = soup.select('[class*="RaceCard"], li, tr')

    players = []
    used_names = set()

    for c in cards:
        text = normalize_text(c.get_text(" ", strip=True))

        if len(text) < 10:
            continue

        name_match = re.search(r"[一-龥ぁ-んァ-ヶ々]{2,8}", text)
        if not name_match:
            continue

        name = name_match.group(0)

        if not is_valid_player_name(name):
            continue
        if name in used_names:
            continue

        style_match = re.search(r"(逃|捲|追|両|自)", text)
        style = style_match.group(1) if style_match else ""

        players.append({
            "車番": len(players) + 1,  # ←順番で振る
            "選手名": name,
            "競走得点": 0,
            "脚質": style,
        })

        used_names.add(name)

        if len(players) >= num_riders:
            break

    return pd.DataFrame(players)

# =========================
# URL取得
# =========================
def fetch_players(url, num_riders):
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()

    df = extract_players_g3(r.text, num_riders)

    if df.empty:
        raise ValueError("選手取得失敗")

    return df

# =========================
# UI
# =========================
st.title("🚴 mobile G3 AI")

url = st.text_input("URL")

if st.button("選手取得"):
    try:
        df = fetch_players(url, 7)
        st.session_state["df"] = df
        st.success("取得成功")
        st.dataframe(df)
    except Exception as e:
        st.error(str(e))

if "df" in st.session_state:

    df = st.session_state["df"]

    if st.button("AI予想"):
        mode = auto_detect_mode(df)

        pred = generate_predictions(
            df,
            mode=mode,
            weather="晴",
            top_n=15,
            odds_dict={},
            ticket_type="3連単",
            race_type="G3"
        )

        st.dataframe(pred)

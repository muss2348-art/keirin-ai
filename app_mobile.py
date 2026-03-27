import itertools
import re

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

st.set_page_config(page_title="競輪AI Mobile", layout="centered")

st.title("🚴 競輪AI Mobile")
st.caption("軽量＆高精度版")

# ------------------------
# 初期値
# ------------------------

default_lines_7 = "147\n25\n3\n6"

if "mobile_lines_text" not in st.session_state:
    st.session_state.mobile_lines_text = default_lines_7
if "mobile_odds_text" not in st.session_state:
    st.session_state.mobile_odds_text = ""
if "mobile_race_url_text" not in st.session_state:
    st.session_state.mobile_race_url_text = ""

# ------------------------
# URL読込
# ------------------------

def normalize_winticket_race_urls(url: str):
    m = re.search(
        r"https?://www\.winticket\.jp/keirin/([^/]+)/(racecard|odds)/([^/]+)/([^/]+)/([^/?#]+)",
        url,
    )
    if not m:
        raise ValueError("URL形式エラー")

    stadium, _, cup_id, day_no, race_no = m.groups()

    racecard_url = f"https://www.winticket.jp/keirin/{stadium}/racecard/{cup_id}/{day_no}/{race_no}"
    odds_url = f"https://www.winticket.jp/keirin/{stadium}/odds/{cup_id}/{day_no}/{race_no}"
    return racecard_url, odds_url


def fetch_html(url):
    return requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text


def extract_lines(html):
    text = BeautifulSoup(html, "lxml").get_text("\n")
    lines = text.splitlines()

    idx = next(i for i, l in enumerate(lines) if "並び予想" in l)

    result, current = [], []

    for s in lines[idx: idx + 60]:
        s = s.strip()
        if s.isdigit():
            current.append(s)
        elif s == "区切り":
            if current:
                result.append("".join(current))
                current = []

    if current:
        result.append("".join(current))

    return "\n".join(result)


def extract_odds(html):
    matches = re.findall(r"([1-9]-[1-9]-[1-9])\s+(\d+\.\d+)", html)
    return "\n".join([f"{a} {b}" for a, b in matches[:50]])

# ------------------------
# データ処理
# ------------------------

def parse_lines(text):
    return [l.strip() for l in text.splitlines() if l.strip()]


def parse_odds(text):
    d = {}
    for l in text.splitlines():
        parts = l.split()
        if len(parts) >= 2:
            d[parts[0]] = float(parts[1])
    return d


def build_df(lines, mode):
    data = []
    line_no = 1

    for line in lines:
        for i, r in enumerate(line):
            r = int(r)
            score = 90 - i * 2

            data.append({
                "車番": r,
                "評価": score,
                "ライン": line_no if len(line) > 1 else 0,
                "ライン順": i + 1,
                "人数": len(line)
            })
        line_no += 1

    return pd.DataFrame(data)

# ------------------------
# ⭐強化済み予想ロジック
# ------------------------

def predict(df, mode, odds_dict):

    top = df.sort_values("評価", ascending=False).head(6)
    riders = top["車番"].tolist()

    tickets = list(itertools.permutations(riders, 3))
    result = []

    line_strength = df.groupby("ライン")["評価"].sum().to_dict()

    for t in tickets:
        score = 0

        r1 = df[df["車番"] == t[0]].iloc[0]
        r2 = df[df["車番"] == t[1]].iloc[0]
        r3 = df[df["車番"] == t[2]].iloc[0]

        score += r1["評価"] * 1.2
        score += r2["評価"] * 0.9
        score += r3["評価"] * 0.6

        # 🔥ライン強化
        if r1["ライン"] != 0:
            score += line_strength.get(r1["ライン"], 0) * 0.05

        if r1["ライン"] == r2["ライン"]:
            score += 10
        if r2["ライン"] == r3["ライン"]:
            score += 5

        # 🔥単騎
        if r1["ライン"] == 0:
            score += 6
        if r2["ライン"] == 0:
            score += 3

        # 🔥ハサミ
        if r1["ライン"] != r2["ライン"]:
            score += 4

        ticket = f"{t[0]}-{t[1]}-{t[2]}"
        odds = odds_dict.get(ticket, None)
        ev = score * odds if odds else None

        result.append([ticket, score, odds, ev])

    df_res = pd.DataFrame(result, columns=["買い目", "AI評価", "オッズ", "期待値"])
    return df_res.sort_values("AI評価", ascending=False)

# ------------------------
# UI
# ------------------------

mode = st.radio("モード", ["通常モード", "混戦モード", "穴モード"])

url = st.text_input("WINTICKET URL")

if st.button("読込"):
    html1, html2 = normalize_winticket_race_urls(url)
    st.session_state.mobile_lines_text = extract_lines(fetch_html(html1))
    st.session_state.mobile_odds_text = extract_odds(fetch_html(html2))

st.text_area("並び", key="mobile_lines_text")

if st.button("予想"):
    lines = parse_lines(st.session_state.mobile_lines_text)
    odds = parse_odds(st.session_state.mobile_odds_text)

    df = build_df(lines, mode)
    res = predict(df, mode, odds)

    st.write("## 最終買い目")
    st.dataframe(res.head(5))

    st.write("## ランキング")
    st.dataframe(res.head(15))

import streamlit as st
import requests
from bs4 import BeautifulSoup
import itertools
import re
import csv
import os
from datetime import datetime


st.set_page_config(page_title="競輪AIモバイル", layout="centered")

LOG_FILE = "race_result_log.csv"


# ==============================
# 並び取得（Kドリームス）
# ==============================
def get_lineup(url):
    try:
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")

        for dt in soup.find_all("dt"):
            if "並び予想" in dt.get_text():
                dd = dt.find_next_sibling("dd")
                line_div = dd.find("div", class_="line_position")

                groups = []
                current = []

                for icon in line_div.find_all("span", class_="icon_p", recursive=False):
                    if "space" in icon.get("class", []):
                        if current:
                            groups.append("-".join(current))
                            current = []
                        continue

                    for s in icon.find_all("span"):
                        txt = s.get_text(strip=True)
                        if txt.isdigit():
                            current.append(txt)
                            break

                if current:
                    groups.append("-".join(current))

                return " / ".join(groups)
    except:
        return None


# ==============================
# 並び処理
# ==============================
def parse_lineup(text):
    return [g.strip().split("-") for g in text.split("/")]


def detect_size(text):
    nums = re.findall(r"\d+", text)
    return 9 if len(set(nums)) == 9 else 7


# ==============================
# モード判定
# ==============================
def judge_mode(groups):
    single = sum(1 for g in groups if len(g) == 1)
    max_len = max(len(g) for g in groups)

    if max_len >= 3 and single == 0:
        return "固め"
    elif single >= 2:
        return "混戦"
    else:
        return "標準"


# ==============================
# スコア
# ==============================
def base_score(groups):
    scores = {}
    for g in groups:
        if len(g) == 3:
            scores[g[0]] = 100
            scores[g[1]] = 90
            scores[g[2]] = 75
        elif len(g) == 2:
            scores[g[0]] = 95
            scores[g[1]] = 85
        else:
            scores[g[0]] = 85
    return scores


# ==============================
# 3連単
# ==============================
def make_3rentan(groups, mode, n):
    scores = base_score(groups)
    nums = sum(groups, [])

    tickets = []

    for a, b, c in itertools.permutations(nums, 3):
        s = scores.get(a, 60)*1.0 + scores.get(b, 60)*0.7 + scores.get(c, 60)*0.5

        if any(a in g and b in g for g in groups):
            s += 20

        if any(len(g)==1 and a==g[0] for g in groups):
            s += 15

        if mode == "固め":
            if any(a in g and b in g for g in groups):
                s += 10
        elif mode == "混戦":
            if not any(a in g and b in g for g in groups):
                s += 10

        tickets.append((f"{a}-{b}-{c}", s))

    tickets.sort(key=lambda x: x[1], reverse=True)

    safe = int(n*0.7)
    return tickets[:safe] + tickets[safe:safe+(n-safe)]


# ==============================
# 2車単
# ==============================
def make_2rentan(groups, mode, n):
    scores = base_score(groups)
    nums = sum(groups, [])

    tickets = []

    for a, b in itertools.permutations(nums, 2):
        s = scores.get(a, 60)*1.0 + scores.get(b, 60)*0.7

        if any(a in g and b in g for g in groups):
            s += 20

        if any(len(g)==1 and a==g[0] for g in groups):
            s += 15

        tickets.append((f"{a}-{b}", s))

    tickets.sort(key=lambda x: x[1], reverse=True)
    return tickets[:n]


# ==============================
# UI
# ==============================
st.title("🚴競輪AIモバイル")

url = st.text_input("KドリームスURL")

if st.button("並び取得"):
    lineup = get_lineup(url)
    if lineup:
        st.session_state["lineup"] = lineup
        st.success(lineup)

if "lineup" in st.session_state:
    text = st.text_input("並び", value=st.session_state["lineup"])

    groups = parse_lineup(text)
    size = detect_size(text)
    mode = judge_mode(groups)

    st.write("車立て:", size)
    st.write("モード:", mode)

    ticket_n = st.slider("点数", 3, 20, 10)
    bet_type = st.selectbox("券種", ["3連単", "2車単"])

    if st.button("買い目生成"):
        if bet_type == "3連単":
            result = make_3rentan(groups, mode, ticket_n)
        else:
            result = make_2rentan(groups, mode, ticket_n)

        st.session_state["result"] = result

if "result" in st.session_state:
    st.markdown("### 買い目")
    for i, (t, s) in enumerate(st.session_state["result"], 1):
        st.write(f"{i}. {t}")

    res = st.text_input("結果 (例: 1-2-3)")

    if st.button("保存"):
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now(), res, str(st.session_state["result"])])
        st.success("保存完了")

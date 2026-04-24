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


st.set_page_config(
    page_title="競輪AI Mobile",
    page_icon="🚴",
    layout="centered",
)

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
}


DEFAULT_COLUMNS = [
    "車番",
    "選手名",
    "競走得点",
    "脚質",
    "ライン",
    "ライン順",
    "単騎",
]


def normalize_text(s):
    if s is None:
        return ""
    table = str.maketrans({
        "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
        "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
        "－": "-", "ー": "-", "―": "-", "／": "/", "　": " ",
    })
    s = str(s).translate(table)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def safe_float(v, default=0.0):
    try:
        if v is None or v == "":
            return float(default)
        if isinstance(v, str):
            v = v.replace(",", "").strip()
        return float(v)
    except Exception:
        return float(default)


def safe_int(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return int(default)


def init_state(num_riders=7):
    st.session_state["num_riders"] = num_riders
    st.session_state["race_rows"] = [
        {
            "車番": i,
            "選手名": "",
            "競走得点": 0.0,
            "脚質": "",
            "ライン": 0,
            "ライン順": 0,
            "単騎": 0,
        }
        for i in range(1, num_riders + 1)
    ]
    st.session_state["pred_df"] = None
    st.session_state["odds_dict"] = {}
    st.session_state["lineup_string"] = ""
    st.session_state["message"] = ""
    st.session_state["debug"] = {}


def get_df():
    if "race_rows" not in st.session_state:
        init_state(7)

    df = pd.DataFrame(st.session_state["race_rows"])

    for c in DEFAULT_COLUMNS:
        if c not in df.columns:
            df[c] = 0

    df["車番"] = pd.to_numeric(df["車番"], errors="coerce").fillna(0).astype(int)
    df["競走得点"] = pd.to_numeric(df["競走得点"], errors="coerce").fillna(0.0)
    df["ライン"] = pd.to_numeric(df["ライン"], errors="coerce").fillna(0).astype(int)
    df["ライン順"] = pd.to_numeric(df["ライン順"], errors="coerce").fillna(0).astype(int)
    df["単騎"] = pd.to_numeric(df["単騎"], errors="coerce").fillna(0).astype(int)

    return df[DEFAULT_COLUMNS].sort_values("車番").reset_index(drop=True)


def set_df(df):
    st.session_state["race_rows"] = df[DEFAULT_COLUMNS].to_dict("records")
    st.session_state["num_riders"] = len(df)


def fetch_text(url):
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or r.encoding
    soup = BeautifulSoup(r.text, "html.parser")
    return normalize_text(soup.get_text(" ", strip=True))


# =========================
# 並び
# =========================
def parse_lineup_groups(lineup_text):
    s = normalize_text(lineup_text)
    s = s.replace("→", "/").replace("|", "/").replace("・", "/").replace(",", "/")
    groups_raw = re.split(r"\s*/\s*", s)

    groups = []
    for g in groups_raw:
        nums = re.findall(r"[1-9]", g)
        if nums:
            groups.append([int(x) for x in nums])

    flat = list(itertools.chain.from_iterable(groups))
    if not flat or len(flat) != len(set(flat)):
        return []

    return groups


def groups_to_text(groups):
    return " / ".join("-".join(str(x) for x in g) for g in groups)


def extract_lineup_from_text(text):
    s = normalize_text(text)

    for kw in ["並び予想", "予想並び", "並び"]:
        pos = s.find(kw)
        if pos != -1:
            window = s[pos:pos + 1200]
            tokens = re.findall(r"区切り|/|[1-9]", window)

            if tokens:
                groups = []
                current = []

                for t in tokens:
                    if t in ["区切り", "/"]:
                        if current:
                            groups.append(current)
                            current = []
                    else:
                        current.append(int(t))

                if current:
                    groups.append(current)

                flat = list(itertools.chain.from_iterable(groups))
                if len(flat) in [5, 6, 7, 9] and set(flat) == set(range(1, len(flat) + 1)):
                    return groups_to_text(groups)

    pattern = re.compile(r'([1-9](?:\s*-\s*[1-9])*(?:\s*/\s*[1-9](?:\s*-\s*[1-9])*){1,8})')
    for m in pattern.finditer(s):
        cand = normalize_text(m.group(1))
        groups = parse_lineup_groups(cand)
        flat = list(itertools.chain.from_iterable(groups))
        if len(flat) in [5, 6, 7, 9] and set(flat) == set(range(1, len(flat) + 1)):
            return groups_to_text(groups)

    return ""


def fetch_lineup(url):
    text = fetch_text(url)
    lineup = extract_lineup_from_text(text)
    if not lineup:
        raise ValueError("URLから並びを抽出できませんでした。")
    return lineup


def apply_lineup_to_df(df, lineup_text):
    groups = parse_lineup_groups(lineup_text)
    if not groups:
        raise ValueError("並び文字列を解釈できませんでした。例: 1-4 / 2-5 / 3")

    out = df.copy()
    flat = list(itertools.chain.from_iterable(groups))
    riders = set(out["車番"].astype(int).tolist())

    if set(flat) != riders:
        raise ValueError(f"並び {sorted(flat)} と車番 {sorted(riders)} が一致しません。")

    out["ライン"] = 0
    out["ライン順"] = 0
    out["単騎"] = 0

    line_id = 1
    for g in groups:
        if len(g) == 1:
            out.loc[out["車番"] == g[0], "単騎"] = 1
            out.loc[out["車番"] == g[0], "ライン順"] = 1
        else:
            for order, car in enumerate(g, start=1):
                out.loc[out["車番"] == car, "ライン"] = line_id
                out.loc[out["車番"] == car, "ライン順"] = order
                out.loc[out["車番"] == car, "単騎"] = 0
            line_id += 1

    return out


# =========================
# 選手取得
# =========================
PREFS = [
    "北海道", "青森", "岩手", "宮城", "秋田", "山形", "福島",
    "茨城", "栃木", "群馬", "埼玉", "千葉", "東京", "神奈川",
    "新潟", "富山", "石川", "福井", "山梨", "長野",
    "岐阜", "静岡", "愛知", "三重", "滋賀", "京都", "大阪", "兵庫",
    "奈良", "和歌山", "鳥取", "島根", "岡山", "広島", "山口",
    "徳島", "香川", "愛媛", "高知", "福岡", "佐賀", "長崎",
    "熊本", "大分", "宮崎", "鹿児島", "沖縄",
]
PREF_PATTERN = "|".join(sorted(PREFS, key=len, reverse=True))


def valid_name(name):
    name = normalize_text(name)
    if not name:
        return False
    if name in ["勝率", "本命", "対抗", "単穴", "連下", "コメント", "ギヤ", "倍率"]:
        return False
    return bool(re.search(r"[一-龥ぁ-んァ-ヶ々]", name)) and 2 <= len(name) <= 12


def extract_score(block):
    b = normalize_text(block)
    patterns = [
        r"\d{2,3}期\s+(?:本命|対抗|単穴|連下)?\s*([4-9]\d(?:\.\d{1,3})?)",
        r"(?:本命|対抗|単穴|連下)\s*([4-9]\d(?:\.\d{1,3})?)",
        r"([4-9]\d(?:\.\d{1,3})?)\s+\d+\s+\d+\s+\d+\s+(?:逃|捲|追|両|自)",
    ]
    for p in patterns:
        m = re.search(p, b)
        if m:
            v = safe_float(m.group(1), 0.0)
            if 40 <= v <= 130:
                return v

    vals = []
    for m in re.finditer(r"([4-9]\d(?:\.\d{1,3})?)", b):
        v = safe_float(m.group(1), 0.0)
        before = b[max(0, m.start() - 3):m.start()]
        after = b[m.end():m.end() + 3]
        if "期" in before or "期" in after or "歳" in before or "歳" in after:
            continue
        if 40 <= v <= 130:
            vals.append(v)

    return vals[-1] if vals else 0.0


def extract_style(block):
    m = re.search(r"(逃|捲|追|両|自)", normalize_text(block))
    return m.group(1) if m else ""


def extract_name(block):
    b = normalize_text(block)

    m = re.search(rf"([一-龥ぁ-んァ-ヶ々]{{2,12}})\s+(?:{PREF_PATTERN})", b)
    if m and valid_name(m.group(1)):
        return m.group(1)

    candidates = re.findall(r"([一-龥ぁ-んァ-ヶ々]{2,12})", b)
    for c in candidates:
        if valid_name(c) and c not in PREFS:
            return c

    return ""


def extract_player_by_car(text, car, num_riders):
    s = normalize_text(text)
    next_car = car + 1

    if next_car <= num_riders:
        patterns = [
            re.compile(rf"(?<!\d){car}\s+{car}\s+(.*?)(?=(?<!\d){next_car}\s+{next_car}\s+)"),
            re.compile(rf"(?<!\d){car}\s+(.*?)(?=(?<!\d){next_car}\s+)"),
        ]
    else:
        patterns = [
            re.compile(rf"(?<!\d){car}\s+{car}\s+(.*)$"),
            re.compile(rf"(?<!\d){car}\s+(.*)$"),
        ]

    for pat in patterns:
        m = pat.search(s)
        if not m:
            continue

        block = normalize_text(m.group(1))[:700]

        name = extract_name(block)
        score = extract_score(block)
        style = extract_style(block)

        if valid_name(name) and 40 <= score <= 130 and style in ["逃", "捲", "追", "両", "自"]:
            return {
                "車番": car,
                "選手名": name,
                "競走得点": score,
                "脚質": style,
            }

    return None


def fetch_players(url, num_riders):
    text = fetch_text(url)
    rows = []

    for car in range(1, num_riders + 1):
        hit = extract_player_by_car(text, car, num_riders)
        if hit:
            rows.append(hit)

    if not rows:
        raise ValueError("選手情報を自動取得できませんでした。")

    df = pd.DataFrame(rows).groupby("車番", as_index=False).first()
    df = df.sort_values("車番").reset_index(drop=True)

    return df


def apply_players_to_df(df, players_df):
    out = df.copy()

    for _, row in players_df.iterrows():
        car = int(row["車番"])

        name = normalize_text(row.get("選手名", ""))
        score = safe_float(row.get("競走得点", 0.0))
        style = normalize_text(row.get("脚質", ""))

        if valid_name(name):
            out.loc[out["車番"] == car, "選手名"] = name
        if 40 <= score <= 130:
            out.loc[out["車番"] == car, "競走得点"] = score
        if style in ["逃", "捲", "追", "両", "自"]:
            out.loc[out["車番"] == car, "脚質"] = style

    return out


# =========================
# オッズ
# =========================
def build_odds_url(url):
    u = normalize_text(url).rstrip("/")
    if "/racecard/" in u:
        return u.replace("/racecard/", "/odds/")
    return u


def fetch_odds(url, ticket_type):
    text = fetch_text(build_odds_url(url))
    odds = {}

    if ticket_type == "2車単":
        patterns = [
            re.compile(r"([1-9]-[1-9])\s+([0-9]+(?:\.[0-9]+)?)"),
            re.compile(r'"combination"\s*:\s*"([1-9]-[1-9])".{0,80}?"odds"\s*:\s*([0-9]+(?:\.[0-9]+)?)'),
        ]
    else:
        patterns = [
            re.compile(r"([1-9]-[1-9]-[1-9])\s+([0-9]+(?:\.[0-9]+)?)"),
            re.compile(r'"combination"\s*:\s*"([1-9]-[1-9]-[1-9])".{0,80}?"odds"\s*:\s*([0-9]+(?:\.[0-9]+)?)'),
        ]

    for pat in patterns:
        for m in pat.finditer(text):
            key = normalize_text(m.group(1)).replace(" ", "")
            val = safe_float(m.group(2), 0.0)
            if val > 0:
                odds[key] = val

    if not odds:
        raise ValueError("オッズを抽出できませんでした。")

    return odds


# =========================
# UI
# =========================
st.title("🚴 競輪AI Mobile")
st.caption("最終安定版 / 5・6・7・9車 / ガールズ対応")

if "race_rows" not in st.session_state:
    init_state(7)

with st.expander("⚙️ 設定", expanded=True):
    rider_options = [5, 6, 7, 9]
    current_num = st.session_state.get("num_riders", 7)
    num_riders = st.radio(
        "車立て",
        rider_options,
        index=rider_options.index(current_num) if current_num in rider_options else 2,
        horizontal=True,
    )

    if num_riders != st.session_state.get("num_riders", 7):
        init_state(num_riders)
        st.rerun()

    race_type = st.selectbox("レース種別", ["通常", "ガールズ"])
    ticket_type = st.selectbox("券種", ["3連単", "2車単"])
    weather = st.selectbox("天候", ["晴", "雨", "風強"])
    display_count = st.selectbox("買い目点数", list(range(3, 31)), index=7)
    unit_bet = st.number_input("1点金額", min_value=100, max_value=10000, value=100, step=100)

url = st.text_input("WINTICKET URL", value=st.session_state.get("last_url", ""))
st.session_state["last_url"] = url

c1, c2, c3 = st.columns(3)

with c1:
    if st.button("並び取得", use_container_width=True):
        try:
            lineup = fetch_lineup(url)
            df = get_df()

            groups = parse_lineup_groups(lineup)
            total = len(list(itertools.chain.from_iterable(groups)))
            if total in [5, 6, 7, 9] and total != len(df):
                init_state(total)
                df = get_df()

            df = apply_lineup_to_df(df, lineup)
            set_df(df)

            st.session_state["lineup_string"] = lineup
            st.session_state["message"] = f"並び取得成功: {lineup}"
            st.rerun()
        except Exception as e:
            st.session_state["message"] = f"並び取得失敗: {e}"
            st.rerun()

with c2:
    if st.button("選手取得", use_container_width=True):
        try:
            df = get_df()
            players = fetch_players(url, len(df))
            df = apply_players_to_df(df, players)
            set_df(df)

            st.session_state["debug"]["players"] = players.to_dict("records")
            st.session_state["message"] = f"選手取得成功: {len(players)}人"
            st.rerun()
        except Exception as e:
            st.session_state["message"] = f"選手取得失敗: {e}"
            st.rerun()

with c3:
    if st.button("オッズ取得", use_container_width=True):
        try:
            odds = fetch_odds(url, ticket_type)
            st.session_state["odds_dict"] = odds
            st.session_state["message"] = f"オッズ取得成功: {len(odds)}件"
            st.rerun()
        except Exception as e:
            st.session_state["message"] = f"オッズ取得失敗: {e}"
            st.rerun()

msg = st.session_state.get("message", "")
if msg:
    if "成功" in msg:
        st.success(msg)
    else:
        st.error(msg)

lineup_input = st.text_input(
    "並び文字列",
    value=st.session_state.get("lineup_string", ""),
    placeholder="例: 1-4 / 2-5 / 3 / 6-7",
)

if st.button("並びを反映", use_container_width=True):
    try:
        df = get_df()
        df = apply_lineup_to_df(df, lineup_input)
        set_df(df)
        st.session_state["lineup_string"] = lineup_input
        st.session_state["message"] = f"並び反映成功: {lineup_input}"
        st.rerun()
    except Exception as e:
        st.session_state["message"] = f"並び反映失敗: {e}"
        st.rerun()

st.markdown("---")
st.subheader("👥 出走表")

df = get_df()

updated_rows = []
style_options = ["", "逃", "捲", "追", "両", "自"]

for i, row in df.iterrows():
    with st.container(border=True):
        st.markdown(f"### {int(row['車番'])}番車")

        name = st.text_input(f"選手名_{i}", value=str(row["選手名"]))
        score = st.number_input(
            f"競走得点_{i}",
            min_value=0.0,
            max_value=200.0,
            value=float(row["競走得点"]),
            step=0.1,
        )

        style_now = str(row["脚質"]) if str(row["脚質"]) in style_options else ""
        style = st.selectbox(
            f"脚質_{i}",
            style_options,
            index=style_options.index(style_now),
        )

        c4, c5, c6 = st.columns(3)
        line_id = c4.number_input(f"ライン_{i}", min_value=0, max_value=9, value=int(row["ライン"]), step=1)
        line_order = c5.number_input(f"順_{i}", min_value=0, max_value=9, value=int(row["ライン順"]), step=1)
        single = c6.selectbox(f"単騎_{i}", [0, 1], index=1 if int(row["単騎"]) == 1 else 0)

        updated_rows.append({
            "車番": int(row["車番"]),
            "選手名": name,
            "競走得点": float(score),
            "脚質": style,
            "ライン": int(line_id),
            "ライン順": int(line_order),
            "単騎": int(single),
        })

if st.button("出走表を更新", use_container_width=True):
    st.session_state["race_rows"] = updated_rows
    st.session_state["message"] = "出走表を更新しました。"
    st.rerun()

current_df = pd.DataFrame(updated_rows)
st.dataframe(current_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("🎯 AI予想")

detected_mode = auto_detect_mode(current_df)

if race_type == "ガールズ":
    st.info("ガールズモード（ライン評価なし）")
else:
    st.info(f"モード自動判定: {detected_mode}")

if st.button("買い目を出す", type="primary", use_container_width=True):
    try:
        pred = generate_predictions(
            current_df,
            mode=detected_mode,
            weather=weather,
            top_n=display_count,
            odds_dict=st.session_state.get("odds_dict", {}),
            ticket_type=ticket_type,
            race_type=race_type,
        )

        if pred is None or pred.empty:
            st.session_state["message"] = "買い目が生成できませんでした。"
            st.rerun()

        pred = pred.copy()
        pred["購入金額"] = [int(unit_bet)] * len(pred)

        if "期待値" in pred.columns:
            ev = pd.to_numeric(pred["期待値"], errors="coerce").fillna(0)
            pred["期待回収額(目安)"] = (ev / 100.0 * pred["購入金額"]).round(0)

        st.session_state["pred_df"] = pred
        st.session_state["message"] = "買い目生成成功"
        st.rerun()

    except Exception as e:
        st.session_state["message"] = f"予想生成失敗: {e}"
        st.rerun()

pred_df = st.session_state.get("pred_df")

if pred_df is not None and isinstance(pred_df, pd.DataFrame) and not pred_df.empty:
    if "レース判定" in pred_df.columns:
        first = pred_df.iloc[0]
        decision = str(first.get("レース判定", ""))
        hit_label = str(first.get("的中率評価", ""))
        score = str(first.get("レース評価点", ""))
        reason = str(first.get("判定理由", ""))

        if decision == "買い":
            st.success(f"レース判定: {decision} / 的中率評価: {hit_label} / 評価点: {score}")
        elif decision == "見送り":
            st.warning(f"レース判定: {decision} / 的中率評価: {hit_label} / 評価点: {score}")
        else:
            st.info(f"レース判定: {decision} / 的中率評価: {hit_label} / 評価点: {score}")

        if reason:
            st.caption(f"判定理由: {reason}")

    for _, row in pred_df.iterrows():
        with st.container(border=True):
            st.markdown(f"### {row.get('買い目ランク', '')} {row.get('買い目', '')}")
            st.write(f"AI評価: {row.get('AI評価', '-')}")
            st.write(f"期待値: {row.get('期待値', '-')}")
            st.write(f"オッズ: {row.get('オッズ', '-')}")
            st.write(f"購入金額: {int(safe_float(row.get('購入金額', 0))):,}円")

    total = int(pd.to_numeric(pred_df["購入金額"], errors="coerce").fillna(0).sum())
    st.metric("合計購入額", f"{total:,}円")

with st.expander("デバッグ", expanded=False):
    st.write(st.session_state.get("debug", {}))
    st.write("取得オッズ件数:", len(st.session_state.get("odds_dict", {})))

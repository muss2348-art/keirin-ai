import csv
import itertools
import math
import os
import re
import unicodedata
from datetime import datetime

import requests
import streamlit as st
from bs4 import BeautifulSoup


st.set_page_config(page_title="競輪AIモバイル", layout="centered")

LOG_FILE = "race_result_log.csv"


# =========================================
# 共通整形
# =========================================
def normalize_text(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("　", " ")
    return text


def clean_lines(text):
    text = normalize_text(text)
    return [line.strip() for line in text.splitlines() if line.strip()]


# =========================================
# Kドリームス 並び取得
# =========================================
def get_kdreams_lineup_from_url(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers, timeout=10)
    res.raise_for_status()

    soup = BeautifulSoup(res.text, "html.parser")

    target_dd = None
    for dt in soup.find_all("dt"):
        if "並び予想" in dt.get_text(strip=True):
            target_dd = dt.find_next_sibling("dd")
            break

    if target_dd is None:
        return None

    line_div = target_dd.find("div", class_="line_position")
    if line_div is None:
        return None

    groups = []
    current_group = []

    for icon in line_div.find_all("span", class_="icon_p", recursive=False):
        classes = icon.get("class", [])

        if "space" in classes:
            if current_group:
                groups.append("-".join(current_group))
                current_group = []
            continue

        number = None
        for s in icon.find_all("span"):
            txt = s.get_text(strip=True)
            if txt.isdigit() and len(txt) == 1:
                number = txt
                break

        if number:
            current_group.append(number)

    if current_group:
        groups.append("-".join(current_group))

    if not groups:
        return None

    return " / ".join(groups)


# =========================================
# 並び処理
# =========================================
def parse_lineup(text):
    groups = []
    if not text:
        return groups

    text = normalize_text(text)

    for part in text.split("/"):
        members = [x.strip() for x in part.strip().split("-") if x.strip()]
        if members:
            groups.append(members)
    return groups


def detect_race_size(lineup_text):
    nums = re.findall(r"\d+", normalize_text(lineup_text or ""))
    count = len(set(nums))
    if count == 9:
        return 9
    return 7


def flatten_line_groups(line_groups):
    nums = []
    for g in line_groups:
        nums.extend(g)
    return nums


# =========================================
# モード判定
# =========================================
def judge_mode(line_groups):
    if not line_groups:
        return "標準"

    lengths = [len(g) for g in line_groups]
    single_count = sum(1 for g in line_groups if len(g) == 1)
    two_count = sum(1 for g in line_groups if len(g) == 2)
    three_plus_count = sum(1 for g in line_groups if len(g) >= 3)
    max_len = max(lengths)

    score = 0

    if max_len >= 3:
        score += 2
    if three_plus_count >= 2:
        score += 2

    score -= single_count * 2

    if two_count >= 2:
        score -= 1

    if score >= 3:
        return "固め"
    elif score <= -2:
        return "混戦"
    return "標準"


# =========================================
# 選手データ整形
# 入力例:
# 1 92.3 逃
# 2,88.1,追
# 3　90.7　両
# =========================================
def normalize_style(style):
    s = normalize_text(style).strip()
    if s in ["逃", "先", "先行"]:
        return "逃"
    if s in ["捲", "まくり"]:
        return "捲"
    if s in ["差", "差し"]:
        return "差"
    if s in ["追", "追込", "追い込み"]:
        return "追"
    if s in ["両", "自在"]:
        return "両"
    return s


def parse_rider_data(text):
    rider_data = {}
    for line in clean_lines(text):
        parts = [p.strip() for p in re.split(r"[,\s]+", line) if p.strip()]
        if len(parts) < 3:
            continue

        num = parts[0]
        score_text = parts[1]
        style = parts[2]

        if not num.isdigit():
            continue

        try:
            score_val = float(score_text)
        except ValueError:
            continue

        rider_data[num] = {
            "score": score_val,
            "style": normalize_style(style)
        }

    return rider_data


def format_rider_data_preview(text):
    rider_data = parse_rider_data(text)
    lines = []
    for num in sorted(rider_data.keys(), key=lambda x: int(x)):
        info = rider_data[num]
        lines.append(f"{num}: 得点 {info['score']} / 脚質 {info['style']}")
    return rider_data, lines


# =========================================
# オッズ整形
# 入力例:
# 4-2-6 12.5
# 5-3-1,18.2
# 7-4-2　55
# =========================================
def parse_odds_text(text):
    odds_map = {}

    for line in clean_lines(text):
        parts = [p.strip() for p in re.split(r"[,\s]+", line) if p.strip()]
        if len(parts) < 2:
            continue

        ticket = parts[0]
        ticket = normalize_text(ticket)

        try:
            odds_val = float(parts[1])
        except ValueError:
            continue

        odds_map[ticket] = odds_val

    return odds_map


def format_odds_preview(text):
    odds_map = parse_odds_text(text)
    lines = []
    for k, v in odds_map.items():
        lines.append(f"{k}: {v}")
    return odds_map, lines


# =========================================
# マップ
# =========================================
def build_position_maps(line_groups):
    line_index_map = {}
    pos_map = {}

    for li, group in enumerate(line_groups):
        for pi, num in enumerate(group):
            line_index_map[num] = li
            pos_map[num] = pi

    return line_index_map, pos_map


def pair_same_line(a, b, line_index_map):
    return line_index_map.get(a) == line_index_map.get(b)


def get_single_nums(line_groups):
    return {g[0] for g in line_groups if len(g) == 1}


# =========================================
# 基本点
# =========================================
def make_base_scores(line_groups, rider_data=None):
    rider_data = rider_data or {}
    scores = {}

    for group in line_groups:
        if len(group) == 3:
            scores[group[0]] = 96
            scores[group[1]] = 90
            scores[group[2]] = 78
        elif len(group) == 2:
            scores[group[0]] = 92
            scores[group[1]] = 84
        elif len(group) == 1:
            scores[group[0]] = 82

    for num, info in rider_data.items():
        if num not in scores:
            scores[num] = 70

        score_val = info.get("score", 0)
        scores[num] += (score_val - 85.0) * 1.8

        style = info.get("style", "")
        if style == "逃":
            scores[num] += 6
        elif style == "捲":
            scores[num] += 5
        elif style == "差":
            scores[num] += 4
        elif style == "両":
            scores[num] += 4
        elif style == "追":
            scores[num] += 1

    return scores


# =========================================
# 券種スコア
# =========================================
def score_sanrentan(ticket, line_groups, rider_data=None, mode="標準", mix_style="固め穴目ミックス"):
    a, b, c = ticket

    line_index_map, pos_map = build_position_maps(line_groups)
    base_scores = make_base_scores(line_groups, rider_data)
    single_nums = get_single_nums(line_groups)

    score = 0.0
    score += base_scores.get(a, 60) * 1.00
    score += base_scores.get(b, 60) * 0.72
    score += base_scores.get(c, 60) * 0.52

    if pair_same_line(a, b, line_index_map):
        score += 22
    if pair_same_line(b, c, line_index_map):
        score += 12
    if pair_same_line(a, c, line_index_map):
        score += 6

    if pos_map.get(a) == 0:
        score += 18
    if pos_map.get(a) == 1:
        score += 8

    if a in single_nums:
        score += 16
    if b in single_nums:
        score += 4
    if c in single_nums:
        score += 2

    if pair_same_line(a, b, line_index_map):
        if pos_map.get(a, 9) < pos_map.get(b, 9):
            score += 8
        else:
            score -= 3

    if pair_same_line(b, c, line_index_map):
        if pos_map.get(b, 9) < pos_map.get(c, 9):
            score += 5
        else:
            score -= 2

    rider_data = rider_data or {}
    a_style = rider_data.get(a, {}).get("style", "")
    b_style = rider_data.get(b, {}).get("style", "")

    if a_style == "逃":
        score += 7
    elif a_style == "捲":
        score += 6
    elif a_style == "両":
        score += 5

    if b_style == "差":
        score += 4
    elif b_style == "追":
        score += 2

    if mode == "固め":
        if pair_same_line(a, b, line_index_map):
            score += 12
        if pair_same_line(b, c, line_index_map):
            score += 6
        if a in single_nums:
            score -= 6
    elif mode == "混戦":
        if a in single_nums:
            score += 16
        if not pair_same_line(a, b, line_index_map):
            score += 9
        if not pair_same_line(b, c, line_index_map):
            score += 5
    else:
        if pair_same_line(a, b, line_index_map):
            score += 7
        if a in single_nums:
            score += 6

    if mix_style == "固め穴目ミックス":
        if a in single_nums:
            score += 5
    elif mix_style == "本線重視":
        if a in single_nums:
            score -= 8
        if pair_same_line(a, b, line_index_map):
            score += 8

    return round(score, 1)


def score_nishatan(ticket, line_groups, rider_data=None, mode="標準", mix_style="固め穴目ミックス"):
    a, b = ticket

    line_index_map, pos_map = build_position_maps(line_groups)
    base_scores = make_base_scores(line_groups, rider_data)
    single_nums = get_single_nums(line_groups)

    score = 0.0
    score += base_scores.get(a, 60) * 1.00
    score += base_scores.get(b, 60) * 0.78

    if pair_same_line(a, b, line_index_map):
        score += 24

    if pos_map.get(a) == 0:
        score += 16
    if pos_map.get(a) == 1:
        score += 7

    if a in single_nums:
        score += 18
    if b in single_nums:
        score += 4

    if pair_same_line(a, b, line_index_map):
        if pos_map.get(a, 9) < pos_map.get(b, 9):
            score += 9
        else:
            score -= 3

    rider_data = rider_data or {}
    a_style = rider_data.get(a, {}).get("style", "")
    b_style = rider_data.get(b, {}).get("style", "")

    if a_style == "逃":
        score += 7
    elif a_style == "捲":
        score += 6
    elif a_style == "両":
        score += 5

    if b_style == "差":
        score += 4
    elif b_style == "追":
        score += 2

    if mode == "固め":
        if pair_same_line(a, b, line_index_map):
            score += 12
        if a in single_nums:
            score -= 6
    elif mode == "混戦":
        if a in single_nums:
            score += 18
        if not pair_same_line(a, b, line_index_map):
            score += 10
    else:
        if a in single_nums:
            score += 7

    if mix_style == "本線重視":
        if pair_same_line(a, b, line_index_map):
            score += 8
        if a in single_nums:
            score -= 8

    return round(score, 1)


# =========================================
# 期待値・ランク
# =========================================
def softmax_probabilities(items):
    if not items:
        return {}

    scores = [x["スコア"] for x in items]
    max_score = max(scores)
    exps = [math.exp((s - max_score) / 12.0) for s in scores]
    total = sum(exps)

    probs = {}
    for item, e in zip(items, exps):
        probs[item["買い目"]] = e / total if total else 0
    return probs


def assign_ranks(items, odds_map=None):
    odds_map = odds_map or {}

    sorted_items = sorted(items, key=lambda x: x["スコア"], reverse=True)
    prob_map = softmax_probabilities(sorted_items)

    enriched = []
    for idx, item in enumerate(sorted_items):
        ticket = item["買い目"]
        prob = prob_map.get(ticket, 0)
        odds = odds_map.get(ticket)
        ev = round(prob * odds, 3) if odds is not None else None

        rank = "🟡 穴"
        if idx == 0:
            rank = "🔥 AI本命"
        elif idx <= 2:
            rank = "🟢 対抗"

        enriched.append({
            **item,
            "推定確率": round(prob * 100, 1),
            "オッズ": odds,
            "期待値": ev,
            "ランク": rank
        })

    if odds_map:
        ev_sorted = [x for x in enriched if x["期待値"] is not None]
        ev_sorted = sorted(ev_sorted, key=lambda x: x["期待値"], reverse=True)
        top_ev_tickets = {x["買い目"] for x in ev_sorted[:2]}

        for item in enriched:
            if item["買い目"] in top_ev_tickets and item["ランク"] == "🟡 穴":
                item["ランク"] = "💰 期待値"

    return enriched


# =========================================
# 買い目生成
# =========================================
def select_mixed_tickets(scored_tickets, ticket_count):
    scored_tickets = sorted(scored_tickets, key=lambda x: x[1], reverse=True)

    safe_n = max(1, int(ticket_count * 0.7))
    hole_n = ticket_count - safe_n

    safe_part = scored_tickets[:safe_n]
    start = safe_n
    end = min(len(scored_tickets), safe_n + max(hole_n * 4, hole_n))
    hole_pool = scored_tickets[start:end]
    hole_part = hole_pool[:hole_n]

    selected = safe_part + hole_part

    result = []
    seen = set()
    for ticket, score in selected:
        if ticket not in seen:
            seen.add(ticket)
            result.append((ticket, score))

    return result[:ticket_count]


def generate_sanrentan(line_groups, rider_data=None, mode="標準", ticket_count=10, mix_style="固め穴目ミックス"):
    nums = flatten_line_groups(line_groups)
    scored = []

    for ticket in itertools.permutations(nums, 3):
        score = score_sanrentan(ticket, line_groups, rider_data, mode, mix_style)
        scored.append((ticket, score))

    if mix_style == "固め穴目ミックス":
        selected = select_mixed_tickets(scored, ticket_count)
    else:
        selected = sorted(scored, key=lambda x: x[1], reverse=True)[:ticket_count]

    return [{"買い目": "-".join(ticket), "スコア": score} for ticket, score in selected]


def generate_nishatan(line_groups, rider_data=None, mode="標準", ticket_count=10, mix_style="固め穴目ミックス"):
    nums = flatten_line_groups(line_groups)
    scored = []

    for ticket in itertools.permutations(nums, 2):
        score = score_nishatan(ticket, line_groups, rider_data, mode, mix_style)
        scored.append((ticket, score))

    if mix_style == "固め穴目ミックス":
        selected = select_mixed_tickets(scored, ticket_count)
    else:
        selected = sorted(scored, key=lambda x: x[1], reverse=True)[:ticket_count]

    return [{"買い目": "-".join(ticket), "スコア": score} for ticket, score in selected]


# =========================================
# 保存
# =========================================
def save_result_to_csv(data, file_path=LOG_FILE):
    file_exists = os.path.exists(file_path)

    with open(file_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "保存日時", "レース名", "URL", "車立て", "券種", "モード", "買い方",
                "並び", "選手データ", "買い目点数", "買い目一覧", "オッズ入力",
                "レース結果", "的中買い目"
            ]
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)


def find_hit_tickets(tickets, result_text):
    result_text = normalize_text(result_text).strip()
    if not result_text:
        return []
    return [t["買い目"] for t in tickets if t["買い目"] == result_text]


# =========================================
# UI
# =========================================
st.title("🚴 競輪AI モバイル版")
st.caption("CSVなし・コピペ最適化版")

st.markdown("### 🔽 並び自動取得")
race_name = st.text_input("レース名（任意）", key="race_name")
url = st.text_input("KドリームスのレースURL", key="race_url")

if st.button("並び予想を取得"):
    if not url:
        st.warning("URLを入力してください")
    else:
        try:
            lineup = get_kdreams_lineup_from_url(url)
            if lineup:
                st.session_state["kari_narabi"] = lineup
                st.success(f"取得成功: {lineup}")
            else:
                st.error("並び予想を取得できませんでした")
        except Exception as e:
            st.error(f"取得エラー: {e}")

if "kari_narabi" in st.session_state:
    st.markdown("### ✏️ 並び編集")
    st.session_state["kari_narabi"] = st.text_input(
        "並び",
        value=st.session_state["kari_narabi"]
    )
    st.info(f"現在の並び: {st.session_state['kari_narabi']}")

    if st.button("並びをクリア"):
        del st.session_state["kari_narabi"]
        if "generated_tickets" in st.session_state:
            del st.session_state["generated_tickets"]
        st.rerun()

st.markdown("### ⚙️ 設定")
default_race_size = detect_race_size(st.session_state.get("kari_narabi", ""))
race_size = st.selectbox("車立て", [7, 9], index=0 if default_race_size == 7 else 1)
bet_type = st.selectbox("券種", ["3連単", "2車単"])
ticket_count = st.selectbox("買い目点数", list(range(3, 21)), index=7)
mix_style = st.selectbox("買い方", ["固め穴目ミックス", "本線重視"])
manual_mode = st.selectbox("モード", ["自動判定", "固め", "標準", "混戦"])

st.markdown("### 🧠 選手データをコピペ")
st.caption("例: 1 92.3 逃 / 2,88.1,追 / 3 90.7 両")
rider_text = st.text_area(
    "車番 得点 脚質",
    height=110,
    placeholder="1 92.3 逃\n2 88.1 追\n3 90.7 両"
)

rider_data, rider_preview = format_rider_data_preview(rider_text)
if rider_preview:
    st.write("読込プレビュー:")
    for line in rider_preview:
        st.write(line)

st.markdown("### 💰 オッズをコピペ")
st.caption("例: 4-2-6 12.5 / 5-3-1,18.2")
odds_text = st.text_area(
    "買い目 オッズ",
    height=110,
    placeholder="4-2-6 12.5\n5-3-1 18.2"
)

odds_map, odds_preview = format_odds_preview(odds_text)
if odds_preview:
    st.write("オッズプレビュー:")
    for line in odds_preview[:10]:
        st.write(line)

if "kari_narabi" in st.session_state:
    line_groups = parse_lineup(st.session_state["kari_narabi"])
    auto_mode = judge_mode(line_groups)
    mode = auto_mode if manual_mode == "自動判定" else manual_mode

    st.markdown("### 📊 判定")
    st.write(f"車立て: {race_size}車")
    st.write(f"自動モード判定: {auto_mode}")
    st.write(f"使用モード: {mode}")
    st.write(f"ライン構成: {line_groups}")

    if st.button("買い目生成"):
        if bet_type == "3連単":
            tickets = generate_sanrentan(
                line_groups=line_groups,
                rider_data=rider_data,
                mode=mode,
                ticket_count=ticket_count,
                mix_style=mix_style
            )
        else:
            tickets = generate_nishatan(
                line_groups=line_groups,
                rider_data=rider_data,
                mode=mode,
                ticket_count=ticket_count,
                mix_style=mix_style
            )

        ranked_tickets = assign_ranks(tickets, odds_map)

        st.session_state["generated_tickets"] = ranked_tickets
        st.session_state["generated_mode"] = mode
        st.session_state["generated_bet_type"] = bet_type
        st.session_state["generated_ticket_count"] = ticket_count
        st.session_state["generated_mix_style"] = mix_style
        st.session_state["generated_rider_text"] = rider_text
        st.session_state["generated_odds_text"] = odds_text

if "generated_tickets" in st.session_state:
    st.markdown("### 🎯 買い目")

    for i, t in enumerate(st.session_state["generated_tickets"], 1):
        line = f"{i}. {t['ランク']}  {t['買い目']}"
        extras = [f"AI評価 {t['スコア']}", f"推定確率 {t['推定確率']}%"]

        if t["オッズ"] is not None:
            extras.append(f"オッズ {t['オッズ']}")
        if t["期待値"] is not None:
            extras.append(f"期待値 {t['期待値']}")

        st.write(line + "  （" + " / ".join(extras) + "）")

    st.markdown("### 📝 レース結果保存")
    result_text = st.text_input(
        "レース結果を入力（3連単なら 1-2-3、2車単なら 1-2）",
        key="result_text"
    )

    if st.button("結果を保存"):
        tickets = st.session_state["generated_tickets"]
        hits = find_hit_tickets(tickets, result_text)

        row = {
            "保存日時": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "レース名": race_name,
            "URL": url,
            "車立て": race_size,
            "券種": st.session_state["generated_bet_type"],
            "モード": st.session_state["generated_mode"],
            "買い方": st.session_state["generated_mix_style"],
            "並び": st.session_state.get("kari_narabi", ""),
            "選手データ": st.session_state.get("generated_rider_text", ""),
            "買い目点数": st.session_state["generated_ticket_count"],
            "買い目一覧": " | ".join([t["買い目"] for t in tickets]),
            "オッズ入力": st.session_state.get("generated_odds_text", ""),
            "レース結果": result_text,
            "的中買い目": " | ".join(hits) if hits else ""
        }

        try:
            save_result_to_csv(row)
            if hits:
                st.success(f"結果を保存しました。的中: {' / '.join(hits)}")
            else:
                st.success("結果を保存しました。今回は的中なしです。")
        except Exception as e:
            st.error(f"保存エラー: {e}")

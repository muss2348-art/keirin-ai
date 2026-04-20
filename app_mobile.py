# app_mobile.py
# -*- coding: utf-8 -*-

import re
import csv
import json
import itertools
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

from predict import auto_detect_mode, generate_predictions


st.set_page_config(
    page_title="競輪AIアプリ Mobile",
    page_icon="🚴",
    layout="centered",
    initial_sidebar_state="collapsed",
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/146.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
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

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_PATH = SCRIPT_DIR / "log.csv"
SAVED_RACES_PATH = SCRIPT_DIR / "saved_races.json"


# =========================================================
# 共通
# =========================================================
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    table = str.maketrans(
        {
            "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
            "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
            "－": "-", "ー": "-", "―": "-", "‐": "-", "ｰ": "-",
            "／": "/", "　": " ", "，": ",", "．": ".",
            "（": "(", "）": ")", "｜": "|",
            "：": ":", "\xa0": " ",
        }
    )
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
        if v is None or v == "":
            return int(default)
        if isinstance(v, str):
            v = v.replace(",", "").strip()
        return int(float(v))
    except Exception:
        return int(default)


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_ticket(ticket: str) -> str:
    s = normalize_text(ticket)
    s = s.replace(" ", "")
    s = re.sub(r"[^0-9\-]", "", s)
    return s


def is_valid_player_name(name: str) -> bool:
    name = normalize_text(name)

    if not name:
        return False
    if re.fullmatch(r"\d+", name):
        return False
    if not re.search(r"[一-龥ぁ-んァ-ヶ々]", name):
        return False
    if len(name) < 2 or len(name) > 12:
        return False
    return True


def format_saved_result(item: dict) -> str:
    result = item.get("result", {}) or {}
    return str(result.get("result_text", "")).strip()


def format_saved_hit_ticket(item: dict) -> str:
    result = item.get("result", {}) or {}
    return str(result.get("hit_ticket", "")).strip()


# =========================================================
# 金額配分
# =========================================================
def rank_base_amount(rank_label: str, unit_bet: int) -> int:
    unit = max(100, int(unit_bet))

    if rank_label == "🔥 AI推奨":
        return unit * 3
    if rank_label == "🟢 本命":
        return unit * 2
    if rank_label == "💰 期待値高":
        return unit
    return unit


def apply_rank_based_amounts(pred_df: pd.DataFrame, unit_bet: int) -> pd.DataFrame:
    if pred_df is None or pred_df.empty:
        return pred_df

    out = pred_df.copy()
    total_budget = int(unit_bet) * len(out)

    if "買い目ランク" not in out.columns:
        out["買い目ランク"] = "🟡 穴"

    base_amounts = []
    thick_scores = []

    for _, row in out.iterrows():
        rank_label = str(row.get("買い目ランク", "🟡 穴"))
        amount = rank_base_amount(rank_label, unit_bet)
        base_amounts.append(amount)

        if rank_label == "🔥 AI推奨":
            thick_scores.append(3.0)
        elif rank_label == "🟢 本命":
            thick_scores.append(2.0)
        elif rank_label == "💰 期待値高":
            thick_scores.append(1.2)
        else:
            thick_scores.append(1.0)

    base_sum = sum(base_amounts)

    if base_sum <= total_budget:
        final_amounts = base_amounts[:]
    else:
        ratio = total_budget / base_sum if base_sum > 0 else 1.0
        scaled = [max(unit_bet, int(round((x * ratio) / 100.0) * 100)) for x in base_amounts]

        diff = total_budget - sum(scaled)

        if diff < 0:
            order = sorted(range(len(scaled)), key=lambda i: scaled[i], reverse=True)
            for i in order:
                while diff < 0 and scaled[i] - 100 >= unit_bet:
                    scaled[i] -= 100
                    diff += 100
                if diff == 0:
                    break
        elif diff > 0:
            order = sorted(range(len(scaled)), key=lambda i: thick_scores[i], reverse=True)
            idx = 0
            while diff > 0 and order:
                scaled[order[idx % len(order)]] += 100
                diff -= 100
                idx += 1

        final_amounts = scaled

    out["厚張り指数"] = [round(x, 2) for x in thick_scores]
    out["購入金額"] = final_amounts

    ev_num = pd.to_numeric(out.get("期待値", 0), errors="coerce").fillna(0)
    out["期待回収額(目安)"] = (ev_num / 100.0 * out["購入金額"]).round(0)

    return out


# =========================================================
# 的中判定
# =========================================================
def judge_hit(ticket_type: str, pred_df: pd.DataFrame, result_1: str, result_2: str, result_3: str):
    if pred_df is None or pred_df.empty:
        return {
            "status_label": "未結果",
            "hit_any": False,
            "hit_ticket": "",
            "result_text": "",
        }

    r1 = str(result_1).strip()
    r2 = str(result_2).strip()
    r3 = str(result_3).strip()

    if not r1 or not r2:
        return {
            "status_label": "未結果",
            "hit_any": False,
            "hit_ticket": "",
            "result_text": "",
        }

    if ticket_type == "2車単":
        result_ticket = f"{r1}-{r2}"
    else:
        if not r3:
            return {
                "status_label": "未結果",
                "hit_any": False,
                "hit_ticket": "",
                "result_text": "",
            }
        result_ticket = f"{r1}-{r2}-{r3}"

    tickets = pred_df["買い目"].astype(str).tolist() if "買い目" in pred_df.columns else []
    hit_any = result_ticket in tickets

    return {
        "status_label": "的中" if hit_any else "不的中",
        "hit_any": hit_any,
        "hit_ticket": result_ticket if hit_any else "",
        "result_text": result_ticket,
    }


# =========================================================
# 回収率集計
# =========================================================
def load_log_df() -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(LOG_PATH, encoding="utf-8-sig")
    except Exception:
        try:
            df = pd.read_csv(LOG_PATH, encoding="utf-8")
        except Exception:
            return pd.DataFrame()

    if df.empty:
        return df

    for col in ["購入金額", "オッズ", "期待値", "期待回収額(目安)"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    for col in ["レース名", "券種", "モード", "天候", "判定", "結果", "買い目"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    return df


def summarize_log_df(log_df: pd.DataFrame):
    if log_df is None or log_df.empty:
        return {
            "race_count": 0,
            "result_saved_race_count": 0,
            "hit_race_count": 0,
            "hit_rate": 0.0,
            "total_invest": 0,
            "total_return": 0,
            "recovery_rate": 0.0,
        }

    work = log_df.copy()

    for c in ["レース名", "券種", "モード", "天候", "結果"]:
        if c not in work.columns:
            work[c] = ""

    work["race_key"] = (
        work["レース名"].astype(str) + " | " +
        work["券種"].astype(str) + " | " +
        work["モード"].astype(str) + " | " +
        work["天候"].astype(str) + " | " +
        work["結果"].astype(str)
    )

    race_summary = (
        work.groupby("race_key", as_index=False)
        .agg(
            結果=("結果", "first"),
            判定=("判定", "first"),
            投資額=("購入金額", "sum"),
        )
    )

    return_map = []
    for _, row in race_summary.iterrows():
        race_rows = work[work["race_key"] == row["race_key"]].copy()
        hit_rows = race_rows[race_rows["判定"] == "的中"].copy()

        if hit_rows.empty:
            return_map.append(0)
            continue

        hit_rows["払戻候補"] = hit_rows["購入金額"] * hit_rows["オッズ"]
        return_map.append(float(hit_rows["払戻候補"].max()))

    race_summary["払戻額"] = return_map

    race_count = len(race_summary)
    result_saved_race_count = int((race_summary["結果"].astype(str).str.strip() != "").sum())
    hit_race_count = int((race_summary["判定"] == "的中").sum())
    hit_rate = round((hit_race_count / result_saved_race_count * 100.0), 1) if result_saved_race_count > 0 else 0.0

    total_invest = int(race_summary["投資額"].sum())
    total_return = int(round(race_summary["払戻額"].sum()))
    recovery_rate = round((total_return / total_invest * 100.0), 1) if total_invest > 0 else 0.0

    return {
        "race_count": race_count,
        "result_saved_race_count": result_saved_race_count,
        "hit_race_count": hit_race_count,
        "hit_rate": hit_rate,
        "total_invest": total_invest,
        "total_return": total_return,
        "recovery_rate": recovery_rate,
    }


# =========================================================
# 保存JSON
# =========================================================
def ensure_saved_races_file():
    if not SAVED_RACES_PATH.exists():
        with open(SAVED_RACES_PATH, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)


def load_saved_races() -> list:
    ensure_saved_races_file()
    try:
        with open(SAVED_RACES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def write_saved_races(data: list):
    with open(SAVED_RACES_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_race_record(record: dict):
    data = load_saved_races()
    data.insert(0, record)
    write_saved_races(data)


def update_saved_race(saved_id: str, updates: dict) -> bool:
    data = load_saved_races()
    ok = False

    for i, item in enumerate(data):
        if item.get("id") == saved_id:
            item.update(updates)
            item["updated_at"] = now_str()
            data[i] = item
            ok = True
            break

    if ok:
        write_saved_races(data)
    return ok


def delete_saved_race(saved_id: str) -> bool:
    data = load_saved_races()
    before = len(data)
    data = [x for x in data if x.get("id") != saved_id]
    if len(data) != before:
        write_saved_races(data)
        return True
    return False


def get_saved_race(saved_id: str):
    for item in load_saved_races():
        if item.get("id") == saved_id:
            return item
    return None


def saved_race_status_label(item: dict) -> str:
    if not item.get("result_saved", False):
        return "未結果"
    return item.get("hit_status", "未結果")


# =========================================================
# 状態管理
# =========================================================
def init_state(num_riders: int = 7):
    rows = []
    for i in range(1, num_riders + 1):
        rows.append(
            {
                "車番": i,
                "選手名": "",
                "競走得点": 0.0,
                "脚質": "",
                "ライン": 0,
                "ライン順": 0,
                "単騎": 0,
            }
        )

    st.session_state["race_rows"] = rows
    st.session_state["num_riders"] = num_riders
    st.session_state["lineup_string"] = ""
    st.session_state["message"] = ""
    st.session_state["pred_df"] = None
    st.session_state["player_debug_info"] = None
    st.session_state["odds_debug_info"] = None
    st.session_state["lineup_debug_info"] = None
    st.session_state["odds_dict"] = {}
    st.session_state["ticket_type"] = st.session_state.get("ticket_type", "3連単")


def get_df() -> pd.DataFrame:
    rows = st.session_state.get("race_rows", [])
    if not rows:
        init_state(7)
        rows = st.session_state.get("race_rows", [])

    df = pd.DataFrame(rows)

    for c in DEFAULT_COLUMNS:
        if c not in df.columns:
            df[c] = 0.0 if c == "競走得点" else ""

    df["車番"] = pd.to_numeric(df["車番"], errors="coerce").fillna(0).astype(int)
    df["競走得点"] = pd.to_numeric(df["競走得点"], errors="coerce").fillna(0.0)
    df["ライン"] = pd.to_numeric(df["ライン"], errors="coerce").fillna(0).astype(int)
    df["ライン順"] = pd.to_numeric(df["ライン順"], errors="coerce").fillna(0).astype(int)
    df["単騎"] = pd.to_numeric(df["単騎"], errors="coerce").fillna(0).astype(int)

    return df[DEFAULT_COLUMNS].copy()


def set_df(df: pd.DataFrame):
    st.session_state["race_rows"] = df[DEFAULT_COLUMNS].to_dict(orient="records")
    st.session_state["num_riders"] = len(df)


def restore_saved_race_to_session(item: dict):
    rows = item.get("race_rows", [])
    num_riders = item.get("num_riders", 7)

    if not rows:
        init_state(num_riders)
    else:
        st.session_state["race_rows"] = rows
        st.session_state["num_riders"] = num_riders

    st.session_state["race_name"] = item.get("race_name", "")
    st.session_state["last_url"] = item.get("url", "")
    st.session_state["lineup_string"] = item.get("lineup_string", "")
    st.session_state["pred_df"] = pd.DataFrame(item.get("pred_rows", [])) if item.get("pred_rows") else None
    st.session_state["odds_dict"] = item.get("odds_dict", {})
    st.session_state["ticket_type"] = item.get("ticket_type", "3連単")
    st.session_state["message"] = f"保存レースを読込: {item.get('race_name', '')}"


# =========================================================
# 並び処理
# =========================================================
def parse_lineup_groups(lineup_text: str):
    s = normalize_text(lineup_text)
    if not s:
        return []

    s = s.replace("|", "/").replace("・", "/").replace(">", "/").replace("→", "/")
    s = s.replace(",", "/").replace(";", "/")

    raw_groups = re.split(r"\s*/\s*", s)
    groups = []

    for g in raw_groups:
        g = normalize_text(g)
        if not g:
            continue
        nums = re.findall(r"[1-9]", g)
        if nums:
            groups.append([int(x) for x in nums])

    flat = list(itertools.chain.from_iterable(groups))
    if not flat or len(set(flat)) != len(flat):
        return []

    return groups


def groups_to_lineup_string(groups):
    return " / ".join("-".join(str(x) for x in g) for g in groups if g)


def apply_lineup_to_df(df: pd.DataFrame, lineup_text: str) -> pd.DataFrame:
    groups = parse_lineup_groups(lineup_text)
    if not groups:
        raise ValueError("並び文字列を解釈できませんでした。")

    flat = list(itertools.chain.from_iterable(groups))
    riders = sorted(df["車番"].astype(int).tolist())

    if set(flat) != set(riders):
        raise ValueError(f"並びの車番 {sorted(flat)} と出走表の車番 {riders} が一致しません。")

    out = df.copy()
    out["ライン"] = 0
    out["ライン順"] = 0
    out["単騎"] = 0

    line_id = 1
    for g in groups:
        if len(g) == 1:
            car = g[0]
            out.loc[out["車番"] == car, "ライン"] = 0
            out.loc[out["車番"] == car, "ライン順"] = 1
            out.loc[out["車番"] == car, "単騎"] = 1
        else:
            for order, car in enumerate(g, start=1):
                out.loc[out["車番"] == car, "ライン"] = line_id
                out.loc[out["車番"] == car, "ライン順"] = order
                out.loc[out["車番"] == car, "単騎"] = 0
            line_id += 1

    return out


# =========================================================
# URL候補
# =========================================================
def build_lineup_candidate_urls(url: str):
    u = normalize_text(url).rstrip("/")
    candidates = [u]

    if "/odds/" in u:
        candidates.append(u.replace("/odds/", "/racecard/"))

    if "/racecard/" not in u and "/odds/" not in u and "/keirin/" in u:
        candidates.append(u.replace("/keirin/", "/keirin/racecard/"))

    uniq = []
    seen = set()
    for x in candidates:
        if x and x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def build_player_candidate_urls(url: str):
    u = normalize_text(url).rstrip("/")
    candidates = [u]

    if "/odds/" in u:
        candidates.append(u.replace("/odds/", "/racecard/"))

    if "/racecard/" not in u and "/odds/" not in u and "/keirin/" in u:
        candidates.append(u.replace("/keirin/", "/keirin/racecard/"))

    uniq = []
    seen = set()
    for x in candidates:
        if x and x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def build_odds_candidate_urls(url: str):
    u = normalize_text(url).rstrip("/")
    candidates = [u]

    if "/racecard/" in u:
        candidates.append(u.replace("/racecard/", "/odds/"))

    if "/odds/" not in u and "/racecard/" not in u and "/keirin/" in u:
        candidates.append(u.replace("/keirin/", "/keirin/odds/"))

    uniq = []
    seen = set()
    for x in candidates:
        if x and x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


# =========================================================
# HTTP / HTML
# =========================================================
def fetch_response(url: str):
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or r.encoding
    return r


def get_html_text_title(url: str):
    r = fetch_response(url)
    html = r.text
    soup = BeautifulSoup(html, "html.parser")
    text = normalize_text(soup.get_text(" ", strip=True))
    title = normalize_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
    return {
        "url": url,
        "status_code": r.status_code,
        "html": html,
        "text": text,
        "title": title,
    }


# =========================================================
# 並び取得
# =========================================================
def extract_lineup_window(page_text: str):
    s = normalize_text(page_text)

    keywords = ["並び予想", "予想並び", "並び"]
    start = -1
    found_kw = ""

    for kw in keywords:
        pos = s.find(kw)
        if pos != -1:
            start = pos
            found_kw = kw
            break

    if start == -1:
        return ""

    tail = s[start + len(found_kw): start + len(found_kw) + 1500]
    end_keywords = [
        "基本情報", "直近成績", "前検コメ", "対戦成績",
        "オッズ一覧", "レース情報", "払戻", "結果", "出走表",
        "人気順", "3連単", "2車単", "2車複",
    ]

    end_pos = len(tail)
    for kw in end_keywords:
        pos = tail.find(kw)
        if pos != -1:
            end_pos = min(end_pos, pos)

    return normalize_text(tail[:end_pos])


def parse_lineup_candidate_string(candidate: str):
    groups = parse_lineup_groups(candidate)
    if not groups:
        return None

    flat = list(itertools.chain.from_iterable(groups))
    if len(flat) not in (6, 7, 9):
        return None

    expected = set(range(1, len(flat) + 1))
    if set(flat) != expected:
        return None

    return groups_to_lineup_string(groups)


def parse_lineup_from_page_text(page_text: str):
    window = extract_lineup_window(page_text)
    if window:
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
            if flat and len(set(flat)) == len(flat) and len(flat) in (6, 7, 9):
                expected = set(range(1, len(flat) + 1))
                if set(flat) == expected:
                    return groups_to_lineup_string(groups)

    s = normalize_text(page_text)
    pattern = re.compile(r'([1-9](?:\s*-\s*[1-9])*(?:\s*/\s*[1-9](?:\s*-\s*[1-9])*){1,8})')

    candidates = []
    for m in pattern.finditer(s):
        cand = normalize_text(m.group(1))
        parsed = parse_lineup_candidate_string(cand)
        if parsed:
            candidates.append(parsed)

    if candidates:
        return candidates[0]

    return None


def fetch_lineup_from_winticket(url: str):
    candidate_urls = build_lineup_candidate_urls(url)
    debug_items = []

    for target_url in candidate_urls:
        try:
            fetched = get_html_text_title(target_url)
            lineup = parse_lineup_from_page_text(fetched["text"])

            debug_items.append(
                {
                    "url": target_url,
                    "status_code": fetched["status_code"],
                    "title": fetched["title"],
                    "lineup_found": lineup if lineup else "",
                    "text_head": fetched["text"][:400],
                }
            )

            if lineup:
                st.session_state["lineup_debug_info"] = {
                    "source_type": "multi_candidate_url_lineup_parse",
                    "candidate_results": debug_items,
                }
                return lineup

        except Exception as e:
            debug_items.append({"url": target_url, "error": str(e)})

    st.session_state["lineup_debug_info"] = {
        "source_type": "multi_candidate_url_lineup_parse",
        "candidate_results": debug_items,
    }
    raise ValueError("URLから並びを抽出できませんでした。")


# =========================================================
# 選手情報取得
# =========================================================
def extract_players_section(page_text: str) -> str:
    text = normalize_text(page_text)

    start_keywords = ["AI 競走得点", "競走得点", "脚質"]
    end_keywords = ["並び予想", "予想並び", "並び", "オッズ一覧", "人気順", "2車単", "3連単"]

    start_pos = -1
    for kw in start_keywords:
        pos = text.find(kw)
        if pos != -1:
            start_pos = pos
            break

    if start_pos == -1:
        return text

    end_pos = len(text)
    for kw in end_keywords:
        pos = text.find(kw, start_pos)
        if pos != -1:
            end_pos = min(end_pos, pos)

    return normalize_text(text[start_pos:end_pos])


def extract_players_with_regex(text: str, num_riders: int):
    s = normalize_text(text)
    rows = []
    preview = []
    seen = set()

    patterns = [
        re.compile(
            r'(?<!\d)([1-9])\s+([1-9])\s+(?:【\d+†)?([^\]】\s]{2,12})(?:】)?\s+[^\s]{2,4}\s+A\d\s+\d{2}歳\s+\d{2,3}期\s+(?:本命|対抗|単穴|連下)?\s*([5-9]\d\.\d{1,2}).{0,20}?([逃捲追両自])'
        ),
        re.compile(
            r'(?<!\d)([1-9])\s+([1-9])\s+([一-龥ぁ-んァ-ヶ々]{2,12}).{0,120}?([5-9]\d\.\d{1,2}).{0,30}?([逃捲追両自])'
        ),
    ]

    for pat in patterns:
        for m in pat.finditer(s):
            car = safe_int(m.group(2))
            name = normalize_text(m.group(3))
            score = safe_float(m.group(4))
            style = normalize_text(m.group(5))

            if (
                1 <= car <= num_riders
                and score >= 50.0
                and style in ["逃", "捲", "追", "両", "自"]
                and is_valid_player_name(name)
            ):
                if car not in seen:
                    seen.add(car)
                    rows.append(
                        {
                            "車番": car,
                            "選手名": name,
                            "競走得点": score,
                            "脚質": style,
                        }
                    )
                    preview.append(
                        {
                            "車番": car,
                            "選手名": name,
                            "競走得点": score,
                            "脚質": style,
                            "source": "regex",
                        }
                    )

    if not rows:
        return pd.DataFrame(), {"hit_count": 0, "preview": []}

    df = pd.DataFrame(rows).groupby("車番", as_index=False).first()
    return df[["車番", "選手名", "競走得点", "脚質"]].copy(), {
        "hit_count": len(df),
        "preview": preview[:10],
    }


def build_sequential_player_blocks(section_text: str, num_riders: int):
    s = normalize_text(section_text)
    starts = []
    cursor = 0

    for car in range(1, num_riders + 1):
        patterns = [
            re.compile(rf'(?<!\d)[1-9]\s+{car}\s+'),
            re.compile(rf'(?<!\d){car}\s+'),
        ]

        found = None
        for pat in patterns:
            m = pat.search(s, cursor)
            if m:
                found = m
                break

        if not found:
            continue

        starts.append((car, found.start(), found.end()))
        cursor = found.end()

    if not starts:
        return []

    blocks = []
    for i, (car, start, match_end) in enumerate(starts):
        if i < len(starts) - 1:
            end = starts[i + 1][1]
        else:
            end = len(s)
            for kw in ["並び予想", "予想並び", "並び", "オッズ一覧", "人気順", "2車単", "3連単"]:
                pos = s.find(kw, start)
                if pos != -1:
                    end = min(end, pos)

        block = normalize_text(s[start:end])
        blocks.append({"車番": car, "block": block})

    return blocks


def parse_player_from_block(car: int, block: str):
    b = normalize_text(block)

    name = ""
    score = 0.0
    style = ""

    name_patterns = [
        re.compile(rf'(?<!\d)[1-9]\s+{car}\s+(?:【\d+†)?([^\]】\s]{{2,12}})(?:】)?\s+[^\s]{{2,4}}\s+A\d'),
        re.compile(rf'(?<!\d)[1-9]\s+{car}\s+([一-龥ぁ-んァ-ヶ々]{{2,12}})'),
        re.compile(rf'(?<!\d){car}\s+(?:【\d+†)?([一-龥ぁ-んァ-ヶ々]{{2,12}})(?:】)?'),
    ]

    for pat in name_patterns:
        m = pat.search(b)
        if m:
            candidate = normalize_text(m.group(1))
            if is_valid_player_name(candidate):
                name = candidate
                break

    score_m = re.search(r'([5-9]\d\.\d{1,2})', b)
    if score_m:
        score = safe_float(score_m.group(1))

    style_patterns = [
        re.compile(r'([5-9]\d\.\d{1,2}).{0,30}?([逃捲追両自])'),
        re.compile(r'([逃捲追両自])'),
    ]

    for pat in style_patterns:
        m = pat.search(b)
        if m:
            if len(m.groups()) == 2:
                style = normalize_text(m.group(2))
            else:
                style = normalize_text(m.group(1))
            break

    if not (
        is_valid_player_name(name)
        and score >= 50.0
        and style in ["逃", "捲", "追", "両", "自"]
    ):
        return None

    return {
        "車番": car,
        "選手名": name,
        "競走得点": score,
        "脚質": style,
    }


def extract_missing_players_by_car_blocks(section_text: str, num_riders: int, existing_cars: set[int]):
    blocks = build_sequential_player_blocks(section_text, num_riders)
    rows = []
    preview = []

    for item in blocks:
        car = item["車番"]
        if car in existing_cars:
            continue

        parsed = parse_player_from_block(car, item["block"])
        if parsed:
            rows.append(parsed)
            preview.append(
                {
                    "車番": parsed["車番"],
                    "選手名": parsed["選手名"],
                    "競走得点": parsed["競走得点"],
                    "脚質": parsed["脚質"],
                    "source": "block_rescue",
                }
            )

    if not rows:
        return pd.DataFrame(), {"hit_count": 0, "preview": []}

    df = pd.DataFrame(rows).groupby("車番", as_index=False).first()
    return df[["車番", "選手名", "競走得点", "脚質"]].copy(), {
        "hit_count": len(df),
        "preview": preview[:10],
    }


def merge_player_dfs(base_df: pd.DataFrame, add_df: pd.DataFrame) -> pd.DataFrame:
    if base_df is None or base_df.empty:
        return add_df.copy() if add_df is not None else pd.DataFrame()
    if add_df is None or add_df.empty:
        return base_df.copy()

    merged = pd.concat([base_df, add_df], ignore_index=True)
    merged = merged.groupby("車番", as_index=False).first()
    return merged[["車番", "選手名", "競走得点", "脚質"]].copy()


def fetch_players_from_winticket(url: str, num_riders: int):
    candidate_urls = build_player_candidate_urls(url)
    debug_items = []
    best_df = pd.DataFrame()

    for target_url in candidate_urls:
        try:
            fetched = get_html_text_title(target_url)
            full_text = fetched["text"]
            section_text = extract_players_section(full_text)

            df_section, dbg_section = extract_players_with_regex(section_text, num_riders)
            df_full, dbg_full = extract_players_with_regex(full_text, num_riders)

            chosen_df = df_section if len(df_section) >= len(df_full) else df_full
            chosen_dbg = dbg_section if len(df_section) >= len(df_full) else dbg_full

            existing_cars = set(chosen_df["車番"].tolist()) if not chosen_df.empty else set()
            missing_cars = set(range(1, num_riders + 1)) - existing_cars

            rescue_df, rescue_dbg = extract_missing_players_by_car_blocks(
                section_text,
                num_riders,
                existing_cars,
            )

            final_df = merge_player_dfs(chosen_df, rescue_df)

            debug_items.append(
                {
                    "url": target_url,
                    "status_code": fetched["status_code"],
                    "title": fetched["title"],
                    "section_hits": len(df_section),
                    "full_hits": len(df_full),
                    "rescue_hits": len(rescue_df),
                    "chosen_hits": len(chosen_df),
                    "final_hits": len(final_df),
                    "missing_before_rescue": sorted(list(missing_cars)),
                    "preview": chosen_dbg.get("preview", [])[:7] + rescue_dbg.get("preview", [])[:4],
                    "section_head": section_text[:300],
                }
            )

            if len(final_df) > len(best_df):
                best_df = final_df

        except Exception as e:
            debug_items.append({"url": target_url, "error": str(e)})

    debug_info = {
        "source_type": "player_regex_plus_section_rescue",
        "hit_count": len(best_df),
        "candidate_results": debug_items,
    }

    if best_df.empty:
        raise ValueError("選手情報を自動取得できませんでした。")

    return best_df, debug_info


def apply_players_to_df(df: pd.DataFrame, players_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    players_df = players_df.copy()

    players_df["車番"] = pd.to_numeric(players_df["車番"], errors="coerce").fillna(0).astype(int)
    players_df["競走得点"] = pd.to_numeric(players_df["競走得点"], errors="coerce").fillna(0.0)

    for _, row in players_df.iterrows():
        car = int(row["車番"])

        name = normalize_text(row.get("選手名", ""))
        if is_valid_player_name(name):
            out.loc[out["車番"] == car, "選手名"] = name

        score = safe_float(row.get("競走得点", 0.0))
        if score >= 50.0:
            out.loc[out["車番"] == car, "競走得点"] = score

        style = normalize_text(row.get("脚質", ""))
        if style in ["逃", "捲", "追", "両", "自"]:
            out.loc[out["車番"] == car, "脚質"] = style

    return out


# =========================================================
# オッズ取得
# =========================================================
def extract_script_texts(html: str):
    soup = BeautifulSoup(html, "html.parser")
    items = []
    for tag in soup.find_all("script"):
        txt = tag.string if tag.string else tag.get_text(" ", strip=True)
        txt = normalize_text(txt or "")
        if txt:
            items.append(txt)
    return items


def extract_odds_loose(text: str, ticket_type: str):
    s = normalize_text(text)
    results = {}

    if ticket_type == "2車単":
        patterns = [
            re.compile(r'(?<!\d)([1-9])\s*-\s*([1-9])\s+([0-9]+(?:\.[0-9]+)?)(?!\d)'),
            re.compile(r'(?<!\d)(\d{1,3})\s+([1-9])\s*-\s*([1-9])\s+([0-9]+(?:\.[0-9]+)?)(?!\d)'),
            re.compile(r'"combination"\s*:\s*"([1-9]-[1-9])".{0,80}?"odds"\s*:\s*([0-9]+(?:\.[0-9]+)?)'),
            re.compile(r'([1-9]-[1-9]).{0,80}?"odds"\s*:\s*([0-9]+(?:\.[0-9]+)?)'),
            re.compile(r'"odds"\s*:\s*([0-9]+(?:\.[0-9]+)?).{0,80}?([1-9]-[1-9])'),
        ]
    else:
        patterns = [
            re.compile(r'(?<!\d)([1-9])\s*-\s*([1-9])\s*-\s*([1-9])\s+([0-9]+(?:\.[0-9]+)?)(?!\d)'),
            re.compile(r'(?<!\d)(\d{1,3})\s+([1-9])\s*-\s*([1-9])\s*-\s*([1-9])\s+([0-9]+(?:\.[0-9]+)?)(?!\d)'),
            re.compile(r'(?<!\d)([1-9])\s+([1-9])\s+([1-9])\s+([0-9]+(?:\.[0-9]+)?)(?!\d)'),
            re.compile(r'"combination"\s*:\s*"([1-9]-[1-9]-[1-9])".{0,80}?"odds"\s*:\s*([0-9]+(?:\.[0-9]+)?)'),
            re.compile(r'([1-9]-[1-9]-[1-9]).{0,80}?"odds"\s*:\s*([0-9]+(?:\.[0-9]+)?)'),
            re.compile(r'"odds"\s*:\s*([0-9]+(?:\.[0-9]+)?).{0,80}?([1-9]-[1-9]-[1-9])'),
        ]

    for pat in patterns:
        for m in pat.finditer(s):
            groups = m.groups()

            if ticket_type == "2車単":
                if len(groups) == 3:
                    a, b, odds = groups
                    key = f"{a}-{b}"
                    val = safe_float(odds, 0.0)
                elif len(groups) == 4:
                    _, a, b, odds = groups
                    key = f"{a}-{b}"
                    val = safe_float(odds, 0.0)
                elif len(groups) == 2:
                    if "-" in groups[0]:
                        key = normalize_ticket(groups[0])
                        val = safe_float(groups[1], 0.0)
                    else:
                        val = safe_float(groups[0], 0.0)
                        key = normalize_ticket(groups[1])
                else:
                    continue

                if len(key.split("-")) == 2 and len(set(key.split("-"))) == 2 and val > 0:
                    results[key] = val

            else:
                if len(groups) == 4:
                    a, b, c, odds = groups
                    key = f"{a}-{b}-{c}"
                    val = safe_float(odds, 0.0)
                elif len(groups) == 5:
                    _, a, b, c, odds = groups
                    key = f"{a}-{b}-{c}"
                    val = safe_float(odds, 0.0)
                elif len(groups) == 2:
                    if "-" in groups[0]:
                        key = normalize_ticket(groups[0])
                        val = safe_float(groups[1], 0.0)
                    else:
                        val = safe_float(groups[0], 0.0)
                        key = normalize_ticket(groups[1])
                else:
                    continue

                if len(key.split("-")) == 3 and len(set(key.split("-"))) == 3 and val > 0:
                    results[key] = val

    return results


def fetch_odds_from_winticket(url: str, ticket_type: str):
    candidate_urls = build_odds_candidate_urls(url)
    all_debug = []
    best_results = {}

    for target_url in candidate_urls:
        try:
            fetched = get_html_text_title(target_url)
            html = fetched["html"]
            text = fetched["text"]
            title = fetched["title"]

            page_results = extract_odds_loose(text, ticket_type=ticket_type)

            scripts = extract_script_texts(html)
            script_results = {}
            for txt in scripts:
                script_results.update(extract_odds_loose(txt, ticket_type=ticket_type))

            merged = {}
            merged.update(page_results)
            merged.update(script_results)

            debug_item = {
                "url": target_url,
                "status_code": fetched["status_code"],
                "title": title,
                "ticket_type": ticket_type,
                "text_head": text[:500],
                "page_hits": len(page_results),
                "script_hits": len(script_results),
                "merged_hits": len(merged),
                "preview": list(sorted(merged.items(), key=lambda x: x[1]))[:15],
            }
            all_debug.append(debug_item)

            if len(merged) > len(best_results):
                best_results = merged

        except Exception as e:
            all_debug.append({"url": target_url, "ticket_type": ticket_type, "error": str(e)})

    debug_info = {
        "source_type": "multi_candidate_url_loose_parse",
        "ticket_type": ticket_type,
        "best_hit_count": len(best_results),
        "candidate_results": all_debug,
    }

    if not best_results:
        raise ValueError("オッズを抽出できませんでした。")

    return best_results, debug_info


# =========================================================
# ログ保存
# =========================================================
def save_result_log(
    race_name: str,
    mode: str,
    weather: str,
    lineup: str,
    ticket_type: str,
    pred_df: pd.DataFrame,
    result_1: str,
    result_2: str,
    result_3: str,
    hit_status: str,
):
    is_new = not LOG_PATH.exists()

    with open(LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(
                [
                    "保存日時",
                    "レース名",
                    "券種",
                    "モード",
                    "天候",
                    "並び",
                    "結果",
                    "判定",
                    "買い目",
                    "買い目ランク",
                    "AI評価",
                    "期待値",
                    "オッズ",
                    "購入金額",
                    "期待回収額(目安)",
                ]
            )

        if ticket_type == "2車単":
            result_text = "-".join([x for x in [result_1, result_2] if x])
        else:
            result_text = "-".join([x for x in [result_1, result_2, result_3] if x])

        for _, row in pred_df.iterrows():
            writer.writerow(
                [
                    now_str(),
                    race_name,
                    ticket_type,
                    mode,
                    weather,
                    lineup,
                    result_text,
                    hit_status,
                    row.get("買い目", ""),
                    row.get("買い目ランク", ""),
                    row.get("AI評価", ""),
                    row.get("期待値", ""),
                    row.get("オッズ", ""),
                    row.get("購入金額", ""),
                    row.get("期待回収額(目安)", ""),
                ]
            )


def save_current_prediction(
    race_name: str,
    url: str,
    mode: str,
    weather: str,
    lineup_string: str,
    ticket_type: str,
    current_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    odds_dict: dict,
    unit_bet: int,
    display_count: int,
):
    saved_id = datetime.now().strftime("%Y%m%d%H%M%S%f")

    record = {
        "id": saved_id,
        "created_at": now_str(),
        "updated_at": now_str(),
        "race_name": race_name,
        "url": url,
        "mode": mode,
        "weather": weather,
        "lineup_string": lineup_string,
        "ticket_type": ticket_type,
        "num_riders": len(current_df),
        "display_count": int(display_count),
        "unit_bet": int(unit_bet),
        "race_rows": current_df[DEFAULT_COLUMNS].to_dict(orient="records"),
        "pred_rows": pred_df.to_dict(orient="records"),
        "odds_dict": odds_dict,
        "result_saved": False,
        "hit_status": "未結果",
        "result": {},
    }
    save_race_record(record)


# =========================================================
# 表示用
# =========================================================
def show_message():
    msg = st.session_state.get("message", "")
    if not msg:
        return

    if "成功" in msg:
        st.success(msg)
    elif "失敗" in msg or "エラー" in msg:
        st.error(msg)
    else:
        st.info(msg)


def render_prediction_cards(pred_df: pd.DataFrame):
    if pred_df is None or pred_df.empty:
        st.caption("まだ買い目は出していません。")
        return

    for i, row in pred_df.reset_index(drop=True).iterrows():
        with st.container(border=True):
            st.markdown(f"### {row.get('買い目ランク', '')} {row.get('買い目', '')}")
            c1, c2 = st.columns(2)
            c1.write(f"AI評価: {row.get('AI評価', '-')}")
            c2.write(f"期待値: {row.get('期待値', '-')}")
            c3, c4 = st.columns(2)
            c3.write(f"オッズ: {row.get('オッズ', '-')}")
            c4.write(f"購入金額: {int(safe_float(row.get('購入金額', 0))):,}円")
            st.caption(f"厚張り指数: {row.get('厚張り指数', '-')}")
            if "期待回収額(目安)" in row:
                st.caption(f"期待回収額(目安): {int(safe_float(row.get('期待回収額(目安)', 0))):,}円")


def render_saved_race_card(item: dict):
    with st.container(border=True):
        st.markdown(f"### {item.get('race_name', '(名称未設定)')}")
        st.caption(f"{item.get('created_at', '')}")

        c1, c2 = st.columns(2)
        c1.write(f"券種: {item.get('ticket_type', '3連単')}")
        c2.write(f"判定: {saved_race_status_label(item)}")

        c3, c4 = st.columns(2)
        c3.write(f"モード: {item.get('mode', '')}")
        c4.write(f"天候: {item.get('weather', '')}")

        result_text = format_saved_result(item)
        hit_ticket = format_saved_hit_ticket(item)
        st.write(f"結果: {result_text if result_text else '-'}")
        st.write(f"的中買い目: {hit_ticket if hit_ticket else '-'}")


# =========================================================
# UI
# =========================================================
st.title("🚴 競輪AI Mobile")
st.caption("スマホ向け / GitHub運用向け")

if "race_rows" not in st.session_state:
    init_state(7)

show_message()

summary = summarize_log_df(load_log_df())
with st.expander("📊 回収率サマリー", expanded=False):
    a, b, c = st.columns(3)
    a.metric("保存", f"{summary['race_count']}件")
    b.metric("的中", f"{summary['hit_race_count']}件")
    c.metric("回収率", f"{summary['recovery_rate']}%")

with st.expander("⚙️ 基本設定", expanded=True):
    num_riders = st.radio(
        "車立て",
        options=[6, 7, 9],
        horizontal=True,
        index=[6, 7, 9].index(st.session_state.get("num_riders", 7)),
    )

    if num_riders != st.session_state.get("num_riders", 7):
        init_state(num_riders)
        st.rerun()

    ticket_type = st.selectbox(
        "券種",
        options=["3連単", "2車単"],
        index=0 if st.session_state.get("ticket_type", "3連単") == "3連単" else 1,
    )
    st.session_state["ticket_type"] = ticket_type

    display_count = st.selectbox("買い目点数", [3, 5, 10, 15, 20, 25, 30], index=2)
    weather = st.selectbox("天候", ["晴", "雨", "風強"], index=0)
    unit_bet = st.number_input("1点あたり金額", min_value=100, max_value=10000, step=100, value=100)

    if st.button("初期化", use_container_width=True):
        init_state(num_riders)
        st.rerun()

race_name = st.text_input("レース名", value=st.session_state.get("race_name", ""))
st.session_state["race_name"] = race_name

default_url = "https://www.winticket.jp/keirin/kumamoto/racecard/2026041487/1/7"
url = st.text_input("WINTICKET URL", value=st.session_state.get("last_url", default_url))
st.session_state["last_url"] = url

b1, b2, b3 = st.columns(3)

with b1:
    if st.button("並び取得", use_container_width=True):
        try:
            lineup = fetch_lineup_from_winticket(url)
            st.session_state["lineup_string"] = lineup

            df = get_df()
            groups = parse_lineup_groups(lineup)
            total = len(list(itertools.chain.from_iterable(groups)))
            if total != len(df):
                init_state(total)
                df = get_df()

            df = apply_lineup_to_df(df, lineup)
            set_df(df)
            st.session_state["message"] = f"並び取得成功: {lineup}"
            st.rerun()
        except Exception as e:
            st.session_state["message"] = f"読み込み失敗: {e}"
            st.rerun()

with b2:
    if st.button("選手取得", use_container_width=True):
        try:
            df = get_df()
            players_df, debug_info = fetch_players_from_winticket(url, len(df))
            df = apply_players_to_df(df, players_df)
            set_df(df)

            st.session_state["player_debug_info"] = debug_info
            st.session_state["message"] = f"選手情報取得成功: {len(players_df)}人"
            st.rerun()
        except Exception as e:
            st.session_state["player_debug_info"] = {"error": str(e)}
            st.session_state["message"] = f"選手情報取得失敗: {e}"
            st.rerun()

with b3:
    if st.button("オッズ取得", use_container_width=True):
        try:
            odds_dict, debug_info = fetch_odds_from_winticket(url, ticket_type=ticket_type)
            st.session_state["odds_dict"] = odds_dict
            st.session_state["odds_debug_info"] = debug_info
            st.session_state["message"] = f"オッズ取得成功: {len(odds_dict)}件"
            st.rerun()
        except Exception as e:
            st.session_state["odds_debug_info"] = {"error": str(e)}
            st.session_state["message"] = f"オッズ取得失敗: {e}"
            st.rerun()

lineup_string = st.text_input(
    "並び文字列",
    value=st.session_state.get("lineup_string", ""),
    placeholder="例: 4-2 / 3-5-1-6",
)

if st.button("並びを反映", use_container_width=True):
    try:
        df = get_df()
        df = apply_lineup_to_df(df, lineup_string)
        set_df(df)
        st.session_state["lineup_string"] = lineup_string
        st.session_state["message"] = f"並び反映成功: {lineup_string}"
        st.rerun()
    except Exception as e:
        st.session_state["message"] = f"反映失敗: {e}"
        st.rerun()

with st.expander("👥 出走表編集", expanded=False):
    current_df = get_df().sort_values("車番").reset_index(drop=True)
    updated_rows = []

    for i, row in current_df.iterrows():
        with st.container(border=True):
            st.markdown(f"### {int(row['車番'])}番車")
            name = st.text_input(f"選手名_{i}", value=str(row["選手名"]))
            c1, c2 = st.columns(2)
            score = c1.number_input(f"競走得点_{i}", min_value=0.0, max_value=200.0, value=float(row["競走得点"]), step=0.1)
            style = c2.selectbox(f"脚質_{i}", options=["", "逃", "捲", "追", "両", "自"], index=["", "逃", "捲", "追", "両", "自"].index(str(row["脚質"]) if str(row["脚質"]) in ["", "逃", "捲", "追", "両", "自"] else ""))
            c3, c4, c5 = st.columns(3)
            line_id = c3.number_input(f"ライン_{i}", min_value=0, max_value=9, value=int(row["ライン"]), step=1)
            line_order = c4.number_input(f"ライン順_{i}", min_value=0, max_value=9, value=int(row["ライン順"]), step=1)
            single = c5.selectbox(f"単騎_{i}", [0, 1], index=1 if int(row["単騎"]) == 1 else 0)

            updated_rows.append(
                {
                    "車番": int(row["車番"]),
                    "選手名": name,
                    "競走得点": score,
                    "脚質": style,
                    "ライン": line_id,
                    "ライン順": line_order,
                    "単騎": single,
                }
            )

    if st.button("出走表を更新", use_container_width=True):
        st.session_state["race_rows"] = updated_rows
        st.session_state["message"] = "出走表を更新しました。"
        st.rerun()

st.markdown("---")
st.subheader("🎯 AI予想")

current_df = get_df().sort_values("車番").reset_index(drop=True)
detected_mode = auto_detect_mode(current_df)
st.info(f"モード自動判定: {detected_mode}")

g1, g2 = st.columns(2)
with g1:
    if st.button("買い目を出す", type="primary", use_container_width=True):
        try:
            pred_df = generate_predictions(
                current_df,
                mode=detected_mode,
                weather=weather,
                top_n=display_count,
                odds_dict=st.session_state.get("odds_dict", {}),
                ticket_type=ticket_type,
            )
            pred_df = apply_rank_based_amounts(pred_df, unit_bet)
            st.session_state["pred_df"] = pred_df
            st.session_state["message"] = "買い目生成成功"
            st.rerun()
        except Exception as e:
            st.session_state["message"] = f"予想生成失敗: {e}"
            st.rerun()

with g2:
    if st.button("この予想を保存", use_container_width=True):
        pred_df = st.session_state.get("pred_df")
        if pred_df is None or pred_df.empty:
            st.session_state["message"] = "先に買い目を出してください。"
            st.rerun()
        else:
            try:
                save_current_prediction(
                    race_name=st.session_state.get("race_name", ""),
                    url=st.session_state.get("last_url", ""),
                    mode=detected_mode,
                    weather=weather,
                    lineup_string=st.session_state.get("lineup_string", ""),
                    ticket_type=ticket_type,
                    current_df=current_df,
                    pred_df=pred_df,
                    odds_dict=st.session_state.get("odds_dict", {}),
                    unit_bet=unit_bet,
                    display_count=display_count,
                )
                st.session_state["message"] = "予想レースを保存しました。"
                st.rerun()
            except Exception as e:
                st.session_state["message"] = f"保存失敗: {e}"
                st.rerun()

pred_df = st.session_state.get("pred_df")
render_prediction_cards(pred_df)

if pred_df is not None and isinstance(pred_df, pd.DataFrame) and not pred_df.empty:
    total_amount = int(pd.to_numeric(pred_df["購入金額"], errors="coerce").fillna(0).sum()) if "購入金額" in pred_df.columns else 0
    st.metric("合計購入額", f"{total_amount:,}円")

    with st.expander("結果保存", expanded=False):
        r1, r2, r3 = st.columns(3)
        result_1 = r1.text_input("1着", value="", key="mobile_result_1")
        result_2 = r2.text_input("2着", value="", key="mobile_result_2")
        result_3 = r3.text_input("3着", value="", key="mobile_result_3")

        if st.button("この予想の結果を保存", use_container_width=True):
            try:
                hit_info = judge_hit(
                    ticket_type=ticket_type,
                    pred_df=pred_df,
                    result_1=result_1,
                    result_2=result_2,
                    result_3=result_3,
                )

                save_result_log(
                    race_name=st.session_state.get("race_name", ""),
                    mode=detected_mode,
                    weather=weather,
                    lineup=st.session_state.get("lineup_string", ""),
                    ticket_type=ticket_type,
                    pred_df=pred_df,
                    result_1=result_1,
                    result_2=result_2,
                    result_3=result_3,
                    hit_status=hit_info["status_label"],
                )
                st.session_state["message"] = f"結果を保存しました / 判定: {hit_info['status_label']}"
                st.rerun()
            except Exception as e:
                st.session_state["message"] = f"保存失敗: {e}"
                st.rerun()

st.markdown("---")
st.subheader("💾 保存した予想")

saved_items = load_saved_races()
if not saved_items:
    st.caption("まだ保存レースはありません。")
else:
    labels = []
    label_to_id = {}

    for item in saved_items:
        result_text = format_saved_result(item)
        label = f"{item.get('race_name', '(名称未設定)')} | {item.get('ticket_type', '3連単')} | {saved_race_status_label(item)}"
        if result_text:
            label += f" | {result_text}"
        labels.append(label)
        label_to_id[label] = item.get("id")

    selected_label = st.selectbox("保存レースを選択", options=labels)
    selected_saved_id = label_to_id.get(selected_label, "")
    selected_item = get_saved_race(selected_saved_id)

    if selected_item:
        render_saved_race_card(selected_item)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("読込", use_container_width=True):
                restore_saved_race_to_session(selected_item)
                st.rerun()

        with c2:
            if st.button("削除", use_container_width=True):
                ok = delete_saved_race(selected_saved_id)
                if ok:
                    st.session_state["message"] = "削除しました。"
                else:
                    st.session_state["message"] = "削除に失敗しました。"
                st.rerun()

        with st.expander("保存済み買い目", expanded=False):
            saved_pred_rows = selected_item.get("pred_rows", [])
            if saved_pred_rows:
                render_prediction_cards(pd.DataFrame(saved_pred_rows))
            else:
                st.caption("保存済み買い目はありません。")

        with st.expander("一覧から結果保存", expanded=False):
            default_result = selected_item.get("result", {})
            selected_ticket_type = selected_item.get("ticket_type", "3連単")

            rr1, rr2, rr3 = st.columns(3)
            result_1 = rr1.text_input("1着", value=str(default_result.get("1着", "")), key="saved_result_1")
            result_2 = rr2.text_input("2着", value=str(default_result.get("2着", "")), key="saved_result_2")
            result_3 = rr3.text_input("3着", value=str(default_result.get("3着", "")), key="saved_result_3")

            if selected_ticket_type == "2車単":
                st.caption("2車単判定は 1着-2着 です。")

            if st.button("この保存レースに結果を保存", use_container_width=True):
                try:
                    saved_pred_df = pd.DataFrame(saved_pred_rows)

                    hit_info = judge_hit(
                        ticket_type=selected_ticket_type,
                        pred_df=saved_pred_df,
                        result_1=result_1,
                        result_2=result_2,
                        result_3=result_3,
                    )

                    save_result_log(
                        race_name=selected_item.get("race_name", ""),
                        mode=selected_item.get("mode", ""),
                        weather=selected_item.get("weather", ""),
                        lineup=selected_item.get("lineup_string", ""),
                        ticket_type=selected_ticket_type,
                        pred_df=saved_pred_df,
                        result_1=result_1,
                        result_2=result_2,
                        result_3=result_3,
                        hit_status=hit_info["status_label"],
                    )

                    update_saved_race(
                        selected_saved_id,
                        {
                            "result_saved": True,
                            "hit_status": hit_info["status_label"],
                            "result": {
                                "1着": result_1,
                                "2着": result_2,
                                "3着": result_3,
                                "result_text": hit_info["result_text"],
                                "hit_ticket": hit_info["hit_ticket"],
                                "saved_at": now_str(),
                            },
                        },
                    )

                    st.session_state["message"] = f"結果を保存しました / 判定: {hit_info['status_label']}"
                    st.rerun()
                except Exception as e:
                    st.session_state["message"] = f"保存失敗: {e}"
                    st.rerun()

with st.expander("デバッグ情報", expanded=False):
    if st.session_state.get("lineup_debug_info"):
        st.write("並び取得デバッグ")
        st.write(st.session_state.get("lineup_debug_info"))
    if st.session_state.get("player_debug_info"):
        st.write("選手情報取得デバッグ")
        st.write(st.session_state.get("player_debug_info"))
    if st.session_state.get("odds_debug_info"):
        st.write("オッズ取得デバッグ")
        st.write(st.session_state.get("odds_debug_info"))

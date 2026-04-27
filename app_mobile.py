# app.py
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
from learning import apply_learning_correction, learning_summary_text
from roi_learning import apply_roi_learning, roi_learning_summary_text
from race_filter import assess_race_buyability, apply_race_buyability_to_predictions, race_buyability_summary_text
from staking import apply_staking_ai, staking_summary_text


st.set_page_config(
    page_title="競輪AI mobile版",
    page_icon="🚴",
    layout="centered",
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

PREFECTURES = [
    "北海道",
    "青森", "岩手", "宮城", "秋田", "山形", "福島",
    "茨城", "栃木", "群馬", "埼玉", "千葉", "東京", "神奈川",
    "新潟", "富山", "石川", "福井",
    "山梨", "長野",
    "岐阜", "静岡", "愛知", "三重",
    "滋賀", "京都", "大阪", "兵庫", "奈良", "和歌山",
    "鳥取", "島根", "岡山", "広島", "山口",
    "徳島", "香川", "愛媛", "高知",
    "福岡", "佐賀", "長崎", "熊本", "大分", "宮崎", "鹿児島", "沖縄",
]
PREF_PATTERN = "|".join(sorted(PREFECTURES, key=len, reverse=True))


def generate_predictions_compat(
    current_df,
    detected_mode,
    weather,
    display_count,
    odds_dict,
    ticket_type,
    race_type,
):
    """predict.py が race_type 未対応でも完全版UIを壊さず予想生成する互換ラッパー。"""
    try:
        return generate_predictions(
            current_df,
            mode=detected_mode,
            weather=weather,
            top_n=display_count,
            odds_dict=odds_dict,
            ticket_type=ticket_type,
            race_type=race_type,
        )
    except TypeError as e:
        msg = str(e)
        if "race_type" in msg and "unexpected keyword" in msg:
            return generate_predictions(
                current_df,
                mode=detected_mode,
                weather=weather,
                top_n=display_count,
                odds_dict=odds_dict,
                ticket_type=ticket_type,
            )
        raise


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


def widget_key(name: str, idx: int) -> str:
    ver = st.session_state.get("widget_ver", 0)
    return f"{name}_{idx}_v{ver}"


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_ticket(ticket: str) -> str:
    s = normalize_text(ticket)
    s = s.replace(" ", "")
    s = re.sub(r"[^0-9\-]", "", s)
    return s


def format_saved_result(item: dict) -> str:
    result = item.get("result", {}) or {}
    return str(result.get("result_text", "")).strip()


def format_saved_hit_ticket(item: dict) -> str:
    result = item.get("result", {}) or {}
    return str(result.get("hit_ticket", "")).strip()


def is_valid_player_name(name: str) -> bool:
    """選手名としてあり得る文字列だけ通す。余計な本文・コメント欄の誤取得を減らす。"""
    name = normalize_text(name)
    if not name:
        return False
    if re.fullmatch(r"\d+", name):
        return False

    ng_words = set(PREFECTURES + [
        "勝率", "本命", "対抗", "単穴", "連下", "単騎で", "コメント", "ギヤ", "倍率",
        "ライン", "並び", "予想", "出走表", "人気順", "払戻", "結果", "レース",
        "オッズ", "前検", "成績", "基本情報", "直近成績",
    ])
    if name in ng_words:
        return False

    if not re.fullmatch(r"[一-龥ぁ-んァ-ヶ々]{2,8}", name):
        return False
    return True


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

    for col in ["レース名", "券種", "モード", "天候", "判定", "結果", "買い目", "レース種別"]:
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
            "by_ticket_type": pd.DataFrame(),
            "by_mode": pd.DataFrame(),
            "by_weather": pd.DataFrame(),
            "by_race_type": pd.DataFrame(),
            "recent_races": pd.DataFrame(),
        }

    work = log_df.copy()

    for c in ["レース名", "券種", "モード", "天候", "結果", "レース種別"]:
        if c not in work.columns:
            work[c] = ""

    work["race_key"] = (
        work["レース名"].astype(str) + " | " +
        work["券種"].astype(str) + " | " +
        work["モード"].astype(str) + " | " +
        work["天候"].astype(str) + " | " +
        work["レース種別"].astype(str) + " | " +
        work["結果"].astype(str)
    )

    race_summary = (
        work.groupby("race_key", as_index=False)
        .agg(
            保存日時=("保存日時", "max"),
            レース名=("レース名", "first"),
            券種=("券種", "first"),
            モード=("モード", "first"),
            天候=("天候", "first"),
            レース種別=("レース種別", "first"),
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

    def make_group_summary(base_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
        if base_df.empty or group_col not in base_df.columns:
            return pd.DataFrame()

        g = (
            base_df.groupby(group_col, as_index=False)
            .agg(
                レース数=("race_key", "count"),
                結果保存数=("結果", lambda x: int((x.astype(str).str.strip() != "").sum())),
                的中数=("判定", lambda x: int((x == "的中").sum())),
                投資額=("投資額", "sum"),
                払戻額=("払戻額", "sum"),
            )
        )

        g["的中率(%)"] = g.apply(
            lambda r: round((r["的中数"] / r["結果保存数"] * 100.0), 1) if r["結果保存数"] > 0 else 0.0,
            axis=1,
        )
        g["回収率(%)"] = g.apply(
            lambda r: round((r["払戻額"] / r["投資額"] * 100.0), 1) if r["投資額"] > 0 else 0.0,
            axis=1,
        )

        return g.sort_values(["回収率(%)", "的中率(%)", "レース数"], ascending=False).reset_index(drop=True)

    recent_cols = ["保存日時", "レース名", "券種", "モード", "天候", "レース種別", "結果", "判定", "投資額", "払戻額"]
    recent_races = race_summary.sort_values("保存日時", ascending=False)[recent_cols].head(20).reset_index(drop=True)

    return {
        "race_count": race_count,
        "result_saved_race_count": result_saved_race_count,
        "hit_race_count": hit_race_count,
        "hit_rate": hit_rate,
        "total_invest": total_invest,
        "total_return": total_return,
        "recovery_rate": recovery_rate,
        "by_ticket_type": make_group_summary(race_summary, "券種"),
        "by_mode": make_group_summary(race_summary, "モード"),
        "by_weather": make_group_summary(race_summary, "天候"),
        "by_race_type": make_group_summary(race_summary, "レース種別"),
        "recent_races": recent_races,
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


def saved_race_label(item: dict) -> str:
    race_name = item.get("race_name", "") or "(名称未設定)"
    created_at = item.get("created_at", "")
    mode = item.get("mode", "")
    ticket_type = item.get("ticket_type", "3連単")
    race_type = item.get("race_type", "通常")
    result_saved = saved_race_status_label(item)
    result_text = format_saved_result(item)

    if result_text:
        return f"{created_at} | {race_name} | {race_type} | {ticket_type} | {mode} | {result_saved} | 結果 {result_text}"
    return f"{created_at} | {race_name} | {race_type} | {ticket_type} | {mode} | {result_saved}"


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
    st.session_state["race_type"] = st.session_state.get("race_type", "通常")
    st.session_state["widget_ver"] = st.session_state.get("widget_ver", 0) + 1


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
        st.session_state["widget_ver"] = st.session_state.get("widget_ver", 0) + 1

    st.session_state["race_name"] = item.get("race_name", "")
    st.session_state["last_url"] = item.get("url", "")
    st.session_state["lineup_string"] = item.get("lineup_string", "")
    st.session_state["pred_df"] = pd.DataFrame(item.get("pred_rows", [])) if item.get("pred_rows") else None
    st.session_state["odds_dict"] = item.get("odds_dict", {})
    st.session_state["ticket_type"] = item.get("ticket_type", "3連単")
    st.session_state["race_type"] = item.get("race_type", "通常")
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
    return build_lineup_candidate_urls(url)


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
def extract_lineup_windows(page_text: str):
    """並び候補がありそうな窓を複数返す。最初の「ライン」などに引っかかって外す問題を避ける。"""
    s = normalize_text(page_text)

    keywords = [
        "並び予想", "予想並び", "周回予想", "予想周回",
        "ライン予想", "初手", "隊列", "並び",
    ]

    end_keywords = [
        "基本情報", "直近成績", "前検コメ", "対戦成績",
        "オッズ一覧", "レース情報", "払戻", "結果", "出走表",
        "人気順", "3連単", "2車単", "2車複", "3連複",
        "選手コメント", "ニュース",
    ]

    windows = []
    seen = set()

    for kw in keywords:
        for m in re.finditer(re.escape(kw), s):
            tail = s[m.start():m.start() + 3000]
            body = s[m.end():m.end() + 3000]

            end_pos = len(body)
            for end_kw in end_keywords:
                p = body.find(end_kw)
                if p != -1:
                    end_pos = min(end_pos, p)

            win = normalize_text(body[:end_pos])
            if not win:
                win = normalize_text(tail)

            if len(re.findall(r"[1-9]", win)) < 5:
                continue

            key = win[:300]
            if key not in seen:
                windows.append(win)
                seen.add(key)

    return windows


def extract_lineup_window(page_text: str):
    windows = extract_lineup_windows(page_text)
    return windows[0] if windows else ""


def parse_lineup_candidate_string(candidate: str):
    groups = parse_lineup_groups(candidate)
    if not groups:
        return None

    flat = list(itertools.chain.from_iterable(groups))
    if len(flat) not in (5, 6, 7, 9):
        return None

    expected = set(range(1, len(flat) + 1))
    if set(flat) != expected:
        return None

    return groups_to_lineup_string(groups)


def _lineup_from_token_window(text: str):
    """区切りや / がある近辺から、余計な数字を除いて並びを復元する保険。"""
    s = normalize_text(text)

    pretty_patterns = [
        re.compile(r'([1-9](?:\s*[-→]\s*[1-9])*(?:\s*/\s*[1-9](?:\s*[-→]\s*[1-9])*){1,8})'),
        re.compile(r'([1-9](?:\s+[1-9]){4,8})'),
    ]
    for pat in pretty_patterns:
        for m in pat.finditer(s):
            cand = normalize_text(m.group(1)).replace("→", "-")
            parsed = parse_lineup_candidate_string(cand)
            if parsed:
                return parsed

    tokens = re.findall(r"区切り|/|→|-|[1-9]", s)
    if not tokens:
        return None

    if "区切り" in tokens or "/" in tokens:
        groups = []
        current = []
        for t in tokens:
            if t in ["区切り", "/"]:
                if current:
                    groups.append(current)
                    current = []
            elif t == "-" or t == "→":
                continue
            else:
                current.append(int(t))
        if current:
            groups.append(current)

        flat = list(itertools.chain.from_iterable(groups))
        if len(flat) in (5, 6, 7, 9) and len(set(flat)) == len(flat):
            expected = set(range(1, len(flat) + 1))
            if set(flat) == expected:
                return groups_to_lineup_string(groups)

    nums = [int(t) for t in tokens if re.fullmatch(r"[1-9]", t)]
    for n in (9, 7, 6, 5):
        if len(nums) < n:
            continue
        expected = set(range(1, n + 1))
        for i in range(0, len(nums) - n + 1):
            chunk = nums[i:i + n]
            if len(set(chunk)) == n and set(chunk) == expected:
                return groups_to_lineup_string([[x] for x in chunk])

    return None


def parse_lineup_from_page_text(page_text: str):
    s = normalize_text(page_text)

    windows = extract_lineup_windows(s)
    for window in windows:
        parsed = _lineup_from_token_window(window)
        if parsed:
            return parsed

    parsed = _lineup_from_token_window(s)
    if parsed:
        return parsed

    return None


def fetch_lineup_from_winticket(url: str):
    candidate_urls = build_lineup_candidate_urls(url)
    debug_items = []

    for target_url in candidate_urls:
        try:
            fetched = get_html_text_title(target_url)
            windows = extract_lineup_windows(fetched["text"])
            lineup = parse_lineup_from_page_text(fetched["text"])

            debug_items.append(
                {
                    "url": target_url,
                    "status_code": fetched["status_code"],
                    "title": fetched["title"],
                    "lineup_found": lineup if lineup else "",
                    "lineup_window": windows[0][:600] if windows else "",
                    "lineup_windows_count": len(windows),
                    "lineup_windows_preview": [w[:300] for w in windows[:5]],
                    "text_head": fetched["text"][:400],
                }
            )

            if lineup:
                st.session_state["lineup_debug_info"] = {
                    "source_type": "multi_candidate_url_lineup_parse_v3",
                    "candidate_results": debug_items,
                }
                return lineup

        except Exception as e:
            debug_items.append({"url": target_url, "error": str(e)})

    st.session_state["lineup_debug_info"] = {
        "source_type": "multi_candidate_url_lineup_parse_v3",
        "candidate_results": debug_items,
    }
    raise ValueError("URLから並びを抽出できませんでした。デバッグの lineup_windows_preview を貼ってください。")


# =========================================================
# 選手情報抽出
# =========================================================
def extract_players_section(page_text: str) -> str:
    text = normalize_text(page_text)

    start_keywords = ["AI 競走得点", "競走得点", "脚質"]
    end_keywords = [
        "並び予想", "予想並び", "並び",
        "オッズ一覧", "人気順",
        "2車単", "3連単", "2車複", "3連複"
    ]

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
        pos = text.find(kw, start_pos + 1)
        if pos != -1:
            end_pos = min(end_pos, pos)

    section = normalize_text(text[start_pos:end_pos])

    if len(section) < 300:
        return text

    return section


def extract_name_from_block(block: str) -> str:
    b = normalize_text(block)

    m = re.search(rf'([一-龥ぁ-んァ-ヶ々]{{2,12}})\s+(?:{PREF_PATTERN})', b)
    if m:
        cand = normalize_text(m.group(1))
        if is_valid_player_name(cand):
            return cand

    candidates = re.findall(r'([一-龥ぁ-んァ-ヶ々]{2,12})', b)
    ng_words = set(PREFECTURES + ["本命", "対抗", "単穴", "連下", "勝率", "コメント", "倍率", "ギヤ", "単騎で"])

    for cand in candidates:
        cand = normalize_text(cand)
        if cand in ng_words:
            continue
        if is_valid_player_name(cand):
            return cand

    return ""


def extract_score_from_block(block: str) -> float:
    b = normalize_text(block)

    patterns = [
        re.compile(r'\d{2,3}期\s+(?:本命|対抗|単穴|連下)?\s*([4-9]\d(?:\.\d{1,3})?)'),
        re.compile(r'(?:本命|対抗|単穴|連下)\s*([4-9]\d(?:\.\d{1,3})?)'),
        re.compile(r'([4-9]\d(?:\.\d{1,3})?)\s+\d+\s+\d+\s+\d+\s+(?:逃|捲|追|両|自)'),
    ]

    for pat in patterns:
        m = pat.search(b)
        if m:
            v = safe_float(m.group(1), 0.0)
            if 40 <= v <= 130:
                return v

    candidates = []
    for m in re.finditer(r'([4-9]\d(?:\.\d{1,3})?)', b):
        raw = m.group(1)
        v = safe_float(raw, 0.0)

        if not (40 <= v <= 130):
            continue

        before = b[max(0, m.start() - 3):m.start()]
        after = b[m.end():m.end() + 3]

        if "期" in before or "期" in after:
            continue
        if "歳" in before or "歳" in after:
            continue

        candidates.append(v)

    if candidates:
        return candidates[-1]

    return 0.0


def extract_style_from_block(block: str) -> str:
    b = normalize_text(block)

    patterns = [
        re.compile(r'(?:本命|対抗|単穴|連下)?\s*[4-9]\d(?:\.\d{1,3})?\s+\d+\s+\d+\s+\d+\s+(逃|捲|追|両|自)'),
        re.compile(r'[4-9]\d(?:\.\d{1,3})?(?:\s+\d+){0,6}\s+(逃|捲|追|両|自)'),
    ]

    for pat in patterns:
        m = pat.search(b)
        if m:
            return normalize_text(m.group(1))

    m = re.search(r'(逃|捲|追|両|自)', b)
    if m:
        return normalize_text(m.group(1))

    return ""


def extract_single_player_by_car(text: str, car: int):
    s = normalize_text(text)

    patterns = [
        re.compile(
            rf'(?<!\d){car}\s+{car}\s+'
            rf'([一-龥ぁ-んァ-ヶ々]{{2,12}})\s+'
            rf'({PREF_PATTERN})\s+'
            rf'([ALS]\d)\s+'
            rf'(\d{{2}})歳\s+'
            rf'(\d{{2,3}})期\s+'
            rf'(?:本命|対抗|単穴|連下)?\s*'
            rf'([4-9]\d(?:\.\d{{1,3}})?)\s+'
            rf'(?:\d+\s+){{2,5}}'
            rf'(逃|捲|追|両|自)'
        ),
        re.compile(
            rf'(?<!\d){car}\s+'
            rf'([一-龥ぁ-んァ-ヶ々]{{2,12}})\s+'
            rf'({PREF_PATTERN})\s+'
            rf'([ALS]\d)\s+'
            rf'(\d{{2}})歳\s+'
            rf'(\d{{2,3}})期\s+'
            rf'(?:本命|対抗|単穴|連下)?\s*'
            rf'([4-9]\d(?:\.\d{{1,3}})?)\s+'
            rf'(?:\d+\s+){{2,5}}'
            rf'(逃|捲|追|両|自)'
        ),
    ]

    for idx, pat in enumerate(patterns, start=1):
        m = pat.search(s)
        if not m:
            continue

        name = normalize_text(m.group(1))
        score = safe_float(m.group(5), 0.0)
        style = normalize_text(m.group(6))

        if is_valid_player_name(name) and 40.0 <= score <= 130.0 and style in ["逃", "捲", "追", "両", "自"]:
            return {
                "車番": car,
                "選手名": name,
                "競走得点": score,
                "脚質": style,
                "source": f"single_pattern_{idx}",
            }

    next_car = car + 1
    block_patterns = []

    if next_car <= 9:
        block_patterns.append(re.compile(rf'(?<!\d){car}\s+{car}\s+(.*?)(?=(?<!\d){next_car}\s+{next_car}\s+|$)'))
        block_patterns.append(re.compile(rf'(?<!\d){car}\s+(.*?)(?=(?<!\d){next_car}\s+{next_car}\s+|$)'))
    else:
        block_patterns.append(re.compile(rf'(?<!\d){car}\s+{car}\s+(.*)$'))
        block_patterns.append(re.compile(rf'(?<!\d){car}\s+(.*)$'))

    for bpat in block_patterns:
        mm = bpat.search(s)
        if not mm:
            continue

        block = normalize_text(mm.group(1))[:500]
        name = extract_name_from_block(block)
        score = extract_score_from_block(block)
        style = extract_style_from_block(block)

        if is_valid_player_name(name) and 40.0 <= score <= 130.0 and style in ["逃", "捲", "追", "両", "自"]:
            return {
                "車番": car,
                "選手名": name,
                "競走得点": score,
                "脚質": style,
                "source": "single_block",
            }

    return None


def extract_players_with_regex(text: str, num_riders: int):
    s = normalize_text(text)
    rows = []
    preview = []
    seen = set()

    entry_pattern = re.compile(
        rf'(?<!\d)'
        rf'([1-9])\s+\1\s+'
        rf'([一-龥ぁ-んァ-ヶ々]{{2,12}})\s+'
        rf'({PREF_PATTERN})\s+'
        rf'([ALS]\d)\s+'
        rf'(\d{{2}})歳\s+'
        rf'(\d{{2,3}})期\s+'
        rf'(?:本命|対抗|単穴|連下)?\s*'
        rf'([4-9]\d(?:\.\d{{1,3}})?)\s+'
        rf'(\d+)\s+(\d+)\s+(\d+)\s+'
        rf'(逃|捲|追|両|自)'
    )

    for m in entry_pattern.finditer(s):
        car = safe_int(m.group(1))
        name = normalize_text(m.group(2))
        score = safe_float(m.group(7), 0.0)
        style = normalize_text(m.group(11))

        if (
            1 <= car <= num_riders
            and car not in seen
            and is_valid_player_name(name)
            and 40.0 <= score <= 130.0
            and style in ["逃", "捲", "追", "両", "自"]
        ):
            seen.add(car)
            rows.append({"車番": car, "選手名": name, "競走得点": score, "脚質": style})
            preview.append({"車番": car, "選手名": name, "競走得点": score, "脚質": style, "source": "entry_pattern"})

    if len(rows) < num_riders:
        for car in range(1, num_riders + 1):
            if car in seen:
                continue

            hit = extract_single_player_by_car(s, car)
            if hit:
                seen.add(car)
                rows.append(
                    {
                        "車番": hit["車番"],
                        "選手名": hit["選手名"],
                        "競走得点": hit["競走得点"],
                        "脚質": hit["脚質"],
                    }
                )
                preview.append(hit)

    if not rows:
        return pd.DataFrame(), {"hit_count": 0, "preview": []}

    df = pd.DataFrame(rows).groupby("車番", as_index=False).first()
    df = df.sort_values("車番").reset_index(drop=True)

    return df[["車番", "選手名", "競走得点", "脚質"]].copy(), {
        "hit_count": len(df),
        "preview": preview[:12],
    }


def extract_players_by_car_blocks(text: str, num_riders: int):
    s = normalize_text(text)
    rows = []
    preview = []

    for car in range(1, num_riders + 1):
        hit = extract_single_player_by_car(s, car)
        if not hit:
            continue

        rows.append(
            {
                "車番": hit["車番"],
                "選手名": hit["選手名"],
                "競走得点": hit["競走得点"],
                "脚質": hit["脚質"],
            }
        )
        hit["source"] = "car_block_safe"
        preview.append(hit)

    if not rows:
        return pd.DataFrame(), {"hit_count": 0, "preview": []}

    df = pd.DataFrame(rows).groupby("車番", as_index=False).first()
    df = df.sort_values("車番").reset_index(drop=True)

    return df[["車番", "選手名", "競走得点", "脚質"]].copy(), {
        "hit_count": len(df),
        "preview": preview[:12],
    }



def normalize_player_df(players_df: pd.DataFrame, num_riders: int) -> pd.DataFrame:
    """
    選手取得の最終安全整形 v6。

    - 同じ選手名が複数車番へ入る事故を防ぐ
    - 車番は 1〜num_riders の範囲だけ採用
    - 同じ車番に複数候補がある場合は、情報量が良い候補を採用
    - 同名候補しか無い車番は、誤表示せず不足エラーで止める
    """
    cols = ["車番", "選手名", "競走得点", "脚質"]
    if players_df is None or players_df.empty:
        return pd.DataFrame(columns=cols)

    df = players_df.copy()
    for col in cols:
        if col not in df.columns:
            df[col] = ""

    df["車番"] = pd.to_numeric(df["車番"], errors="coerce").fillna(0).astype(int)
    df["選手名"] = df["選手名"].astype(str).map(normalize_text)
    df["競走得点"] = pd.to_numeric(df["競走得点"], errors="coerce").fillna(0.0)
    df["脚質"] = df["脚質"].astype(str).map(normalize_text)

    df = df[(df["車番"] >= 1) & (df["車番"] <= int(num_riders))]
    df = df[df["選手名"].map(is_valid_player_name)]
    df = df[(df["競走得点"] >= 40.0) & (df["競走得点"] <= 130.0)]
    df = df[df["脚質"].isin(["逃", "捲", "追", "両", "自"])]

    if df.empty:
        return pd.DataFrame(columns=cols)

    def candidate_quality(row) -> float:
        q = 0.0
        name = str(row.get("選手名", ""))
        score = safe_float(row.get("競走得点", 0), 0)
        style = str(row.get("脚質", ""))
        if is_valid_player_name(name):
            q += 10.0
        if 60.0 <= score <= 125.0:
            q += 10.0
        if style in ["逃", "捲", "追", "両", "自"]:
            q += 5.0
        q += min(max(score - 40.0, 0.0), 80.0) / 20.0
        return q

    df["_quality"] = df.apply(candidate_quality, axis=1)
    df = df.sort_values(["車番", "_quality", "競走得点"], ascending=[True, False, False]).reset_index(drop=True)

    fixed = {}
    used_names = set()

    for car in range(1, int(num_riders) + 1):
        cand = df[df["車番"] == car].copy()
        if cand.empty:
            continue

        chosen = None
        for _, row in cand.iterrows():
            name = str(row["選手名"])
            if name in used_names:
                continue
            chosen = row
            break

        if chosen is None:
            continue

        fixed[car] = {
            "車番": car,
            "選手名": str(chosen["選手名"]),
            "競走得点": float(chosen["競走得点"]),
            "脚質": str(chosen["脚質"]),
        }
        used_names.add(str(chosen["選手名"]))

    rows = [fixed[k] for k in sorted(fixed.keys())]
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=cols)

    out = out.drop_duplicates(subset=["車番"], keep="first")
    out = out.drop_duplicates(subset=["選手名"], keep="first")
    out = out.sort_values("車番").reset_index(drop=True)
    return out[cols].copy()

def merge_player_dfs(base_df: pd.DataFrame, add_df: pd.DataFrame, num_riders: int | None = None) -> pd.DataFrame:
    if base_df is None or base_df.empty:
        merged = add_df.copy() if add_df is not None else pd.DataFrame()
    elif add_df is None or add_df.empty:
        merged = base_df.copy()
    else:
        merged = pd.concat([base_df, add_df], ignore_index=True)

    if num_riders is None:
        if merged is None or merged.empty or "車番" not in merged.columns:
            return pd.DataFrame(columns=["車番", "選手名", "競走得点", "脚質"])
        num_riders = int(pd.to_numeric(merged["車番"], errors="coerce").fillna(0).max())

    return normalize_player_df(merged, num_riders)


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
            df_block, dbg_block = extract_players_by_car_blocks(full_text, num_riders)

            chosen_df = df_section if len(df_section) >= len(df_full) else df_full
            chosen_dbg = dbg_section if len(df_section) >= len(df_full) else dbg_full

            final_df = merge_player_dfs(chosen_df, df_block, num_riders)

            debug_items.append(
                {
                    "url": target_url,
                    "status_code": fetched["status_code"],
                    "title": fetched["title"],
                    "section_hits": len(df_section),
                    "full_hits": len(df_full),
                    "block_hits": len(df_block),
                    "chosen_hits": len(chosen_df),
                    "final_hits": len(final_df),
                    "missing_after": sorted(list(set(range(1, num_riders + 1)) - set(final_df["車番"].tolist()))) if not final_df.empty else list(range(1, num_riders + 1)),
                    "preview": (
                        chosen_dbg.get("preview", [])[:8]
                        + dbg_block.get("preview", [])[:8]
                    ),
                    "section_head": section_text[:300],
                }
            )

            if len(final_df) > len(best_df):
                best_df = final_df

        except Exception as e:
            debug_items.append({"url": target_url, "error": str(e)})

    best_df = normalize_player_df(best_df, num_riders)

    debug_info = {
        "source_type": "player_regex_plus_5_6_7_9_unique_name_v6",
        "hit_count": len(best_df),
        "candidate_results": debug_items,
        "final_players": best_df.to_dict(orient="records") if not best_df.empty else [],
    }

    if best_df.empty:
        raise ValueError("選手情報を自動取得できませんでした。")
    if len(best_df) < num_riders:
        missing = sorted(list(set(range(1, num_riders + 1)) - set(best_df["車番"].astype(int).tolist())))
        raise ValueError(f"選手情報が不足しています。取得{len(best_df)}人 / 不足車番: {missing}")

    return best_df, debug_info


def apply_players_to_df(df: pd.DataFrame, players_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    players_df = normalize_player_df(players_df, len(out)).copy()

    players_df["車番"] = pd.to_numeric(players_df["車番"], errors="coerce").fillna(0).astype(int)
    players_df["競走得点"] = pd.to_numeric(players_df["競走得点"], errors="coerce").fillna(0.0)

    for _, row in players_df.iterrows():
        car = int(row["車番"])

        name = normalize_text(row.get("選手名", ""))
        if is_valid_player_name(name):
            out.loc[out["車番"] == car, "選手名"] = name

        score = safe_float(row.get("競走得点", 0.0))
        if 40.0 <= score <= 130.0:
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
            re.compile(r'"combination"\s*:\s*"([1-9]-[1-9])".{0,80}?"odds"\s*:\s*([0-9]+(?:\.[0-9]+)?)'),
            re.compile(r'([1-9]-[1-9]).{0,80}?"odds"\s*:\s*([0-9]+(?:\.[0-9]+)?)'),
            re.compile(r'"odds"\s*:\s*([0-9]+(?:\.[0-9]+)?).{0,80}?([1-9]-[1-9])'),
        ]
    else:
        patterns = [
            re.compile(r'(?<!\d)([1-9])\s*-\s*([1-9])\s*-\s*([1-9])\s+([0-9]+(?:\.[0-9]+)?)(?!\d)'),
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
    race_type: str,
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
                    "保存日時", "レース名", "券種", "モード", "天候", "レース種別", "並び", "結果", "判定",
                    "買い目", "買い目ランク", "AI評価", "期待値", "オッズ",
                    "購入金額", "期待回収額(目安)", "レース判定", "的中率評価", "レース評価点", "判定理由", "見送りAIコメント",
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
                    race_type,
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
                    row.get("レース判定", ""),
                    row.get("的中率評価", ""),
                    row.get("レース評価点", ""),
                    row.get("判定理由", ""),
                ]
            )


def save_current_prediction(
    race_name: str,
    url: str,
    mode: str,
    weather: str,
    race_type: str,
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
        "race_type": race_type,
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
# UI
# =========================================================
st.title("🚴 競輪AI mobile版")
st.caption("完全統合版 / ライン信頼度・崩れ対応 / 学習補正 / 見送りAI / 賭け金AI")

if "race_rows" not in st.session_state:
    init_state(7)

with st.sidebar:
    st.header("設定")

    rider_options = [5, 6, 7, 9]
    current_num = st.session_state.get("num_riders", 7)
    current_index = rider_options.index(current_num) if current_num in rider_options else 2

    num_riders = st.radio("車立て", options=rider_options, index=current_index, horizontal=True)

    if num_riders != st.session_state.get("num_riders", 7):
        init_state(num_riders)
        st.rerun()

    ticket_type = st.selectbox(
        "券種",
        options=["3連単", "2車単"],
        index=0 if st.session_state.get("ticket_type", "3連単") == "3連単" else 1,
    )
    st.session_state["ticket_type"] = ticket_type

    race_type_options = ["通常", "ガールズ", "G3"]
    race_type_default = st.session_state.get("race_type", "通常")
    race_type = st.selectbox(
        "レース種別",
        options=race_type_options,
        index=race_type_options.index(race_type_default) if race_type_default in race_type_options else 0,
    )
    st.session_state["race_type"] = race_type

    display_count = st.selectbox(
        "買い目点数",
        options=[5, 7, 10, 12, 15, 20],
        index=2,
    )

    weather = st.selectbox("天候", options=["晴", "雨", "風強"], index=0)
    unit_bet = st.number_input("1点あたり金額", min_value=100, max_value=10000, step=100, value=100)

    st.caption("厚張り基準")
    st.caption(f"🔥 AI推奨 = {unit_bet * 3:,}円")
    st.caption(f"🟢 本命 = {unit_bet * 2:,}円")
    st.caption(f"💰 期待値高 = {unit_bet:,}円")
    st.caption(f"🟡 穴 = {unit_bet:,}円")

    if st.button("初期化", use_container_width=True):
        init_state(num_riders)
        st.rerun()

c1, c2 = st.columns([1, 1])

with c1:
    race_name = st.text_input("レース名", value=st.session_state.get("race_name", ""))
    st.session_state["race_name"] = race_name

with c2:
    default_url = "https://www.winticket.jp/keirin/kumamoto/racecard/2026041487/1/7"
    url = st.text_input("WINTICKET URL", value=st.session_state.get("last_url", default_url))
    st.session_state["last_url"] = url

c3, c4, c5, c6 = st.columns([1, 1, 1, 2])

with c3:
    if st.button("URLから並びを読み込む", use_container_width=True):
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
            st.session_state["widget_ver"] = st.session_state.get("widget_ver", 0) + 1
            st.session_state["message"] = f"並び取得成功: {lineup}"
            st.rerun()
        except Exception as e:
            st.session_state["message"] = f"読み込み失敗: {e}"

with c4:
    if st.button("選手情報を自動取得", use_container_width=True):
        try:
            df = get_df()
            players_df, debug_info = fetch_players_from_winticket(url, len(df))
            df = apply_players_to_df(df, players_df)
            set_df(df)

            st.session_state["player_debug_info"] = debug_info
            st.session_state["widget_ver"] = st.session_state.get("widget_ver", 0) + 1
            st.session_state["message"] = f"選手情報取得成功: {len(players_df)}人"
            st.rerun()
        except Exception as e:
            st.session_state["player_debug_info"] = {"error": str(e)}
            st.session_state["message"] = f"選手情報取得失敗: {e}"

with c5:
    if st.button("オッズを自動取得", use_container_width=True):
        try:
            odds_dict, debug_info = fetch_odds_from_winticket(url, ticket_type=ticket_type)
            st.session_state["odds_dict"] = odds_dict
            st.session_state["odds_debug_info"] = debug_info
            st.session_state["message"] = f"オッズ取得成功: {len(odds_dict)}件"
            st.rerun()
        except Exception as e:
            st.session_state["odds_debug_info"] = {"error": str(e)}
            st.session_state["message"] = f"オッズ取得失敗: {e}"

with c6:
    msg = st.session_state.get("message", "")
    if msg:
        if "成功" in msg:
            st.success(msg)
        else:
            st.error(msg)

player_debug = st.session_state.get("player_debug_info", None)
if player_debug:
    with st.expander("選手情報取得デバッグ情報"):
        if player_debug.get("error"):
            st.error(player_debug["error"])
        else:
            st.write(player_debug)

lineup_debug = st.session_state.get("lineup_debug_info", None)
if lineup_debug:
    with st.expander("並び取得デバッグ情報"):
        st.write(lineup_debug)

odds_debug = st.session_state.get("odds_debug_info", None)
if odds_debug:
    with st.expander("オッズ取得デバッグ情報"):
        if odds_debug.get("error"):
            st.error(odds_debug["error"])
        else:
            st.write(odds_debug)

lineup_string = st.text_input(
    "並び文字列",
    value=st.session_state.get("lineup_string", ""),
    placeholder="例: 4-2 / 3-5-1-6",
)

c7, c8 = st.columns([1, 2])

with c7:
    if st.button("並びを反映", use_container_width=True):
        try:
            df = get_df()
            df = apply_lineup_to_df(df, lineup_string)
            set_df(df)
            st.session_state["lineup_string"] = lineup_string
            st.session_state["widget_ver"] = st.session_state.get("widget_ver", 0) + 1
            st.session_state["message"] = f"並び反映成功: {lineup_string}"
            st.rerun()
        except Exception as e:
            st.session_state["message"] = f"反映失敗: {e}"

with c8:
    if race_type == "ガールズ":
        st.caption(f"券種: {ticket_type} / レース種別: ガールズ / ライン評価なし / 取得オッズ: {len(st.session_state.get('odds_dict', {}))}件")
    else:
        st.caption(f"券種: {ticket_type} / レース種別: {race_type} / 取得オッズ: {len(st.session_state.get('odds_dict', {}))}件")

st.markdown("---")
st.subheader("出走表入力")

df = get_df().sort_values("車番").reset_index(drop=True)

with st.form("runner_form"):
    header = st.columns([0.8, 1.8, 1.2, 1.0, 0.8, 0.8, 0.8])
    header[0].markdown("**車番**")
    header[1].markdown("**選手名**")
    header[2].markdown("**競走得点**")
    header[3].markdown("**脚質**")
    header[4].markdown("**ライン**")
    header[5].markdown("**ライン順**")
    header[6].markdown("**単騎**")

    updated_rows = []
    style_options = ["", "逃", "捲", "追", "両", "自"]

    for i, row in df.iterrows():
        cols = st.columns([0.8, 1.8, 1.2, 1.0, 0.8, 0.8, 0.8])

        car_num = cols[0].number_input(
            f"車番_{i}", min_value=1, max_value=9, value=int(row["車番"]), step=1, key=widget_key("car", i)
        )
        name = cols[1].text_input(f"選手名_{i}", value=str(row["選手名"]), key=widget_key("name", i))
        score = cols[2].number_input(
            f"競走得点_{i}", min_value=0.0, max_value=200.0, value=float(row["競走得点"]), step=0.1, key=widget_key("score", i)
        )
        style_now = str(row["脚質"]) if str(row["脚質"]) in style_options else ""
        style = cols[3].selectbox(
            f"脚質_{i}", options=style_options, index=style_options.index(style_now), key=widget_key("style", i)
        )
        line_id = cols[4].number_input(
            f"ライン_{i}", min_value=0, max_value=9, value=int(row["ライン"]), step=1, key=widget_key("line", i)
        )
        line_order = cols[5].number_input(
            f"ライン順_{i}", min_value=0, max_value=9, value=int(row["ライン順"]), step=1, key=widget_key("line_order", i)
        )
        single = cols[6].selectbox(
            f"単騎_{i}", options=[0, 1], index=1 if int(row["単騎"]) == 1 else 0, key=widget_key("single", i)
        )

        updated_rows.append(
            {
                "車番": car_num,
                "選手名": name,
                "競走得点": score,
                "脚質": style,
                "ライン": line_id,
                "ライン順": line_order,
                "単騎": single,
            }
        )

    submit_rows = st.form_submit_button("出走表を反映", use_container_width=True)

if submit_rows:
    st.session_state["race_rows"] = updated_rows
    st.session_state["message"] = "出走表を反映しました。"
    st.rerun()

st.markdown("---")
st.subheader("現在の出走表")
current_df = get_df().sort_values("車番").reset_index(drop=True)
st.dataframe(current_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("AI予想")
st.caption(learning_summary_text(LOG_PATH))
st.caption(roi_learning_summary_text(LOG_PATH))
st.caption("見送りAIは買い/軽く買い/注意/見送りを判定します。")
st.caption("賭け金AIはAI評価・期待値・見送りAI判定から購入金額を自動配分します。")

detected_mode = auto_detect_mode(current_df)
if race_type == "ガールズ":
    st.info("モード: ガールズモード（ライン評価なし）")
else:
    st.info(f"モード自動判定: {detected_mode}")

st.caption(f"券種: {ticket_type} / 天候: {weather} / レース種別: {race_type} / 買い目点数: {display_count}点")

p1, p2 = st.columns([1, 1])

with p1:
    if st.button("買い目を出す", type="primary", use_container_width=True):
        try:
            pred_df = generate_predictions_compat(
                current_df=current_df,
                detected_mode=detected_mode,
                weather=weather,
                display_count=display_count,
                odds_dict=st.session_state.get("odds_dict", {}),
                ticket_type=ticket_type,
                race_type=race_type,
            )
            pred_df = apply_learning_correction(
                pred_df,
                LOG_PATH,
                mode=detected_mode,
                weather=weather,
                ticket_type=ticket_type,
            )
            pred_df = apply_roi_learning(
                pred_df,
                LOG_PATH,
                mode=detected_mode,
                weather=weather,
                ticket_type=ticket_type,
            )
            race_assessment = assess_race_buyability(
                current_df,
                pred_df=pred_df,
                log_path=LOG_PATH,
                mode=detected_mode,
                weather=weather,
                ticket_type=ticket_type,
                race_type=race_type,
            )
            pred_df = apply_race_buyability_to_predictions(pred_df, race_assessment)
            st.session_state["race_assessment"] = race_assessment
            pred_df = apply_rank_based_amounts(pred_df, unit_bet)
            pred_df = apply_staking_ai(
                pred_df,
                unit_bet=unit_bet,
                race_assessment=race_assessment,
            )
            st.session_state["pred_df"] = pred_df
            st.session_state["message"] = "買い目生成成功"
            st.rerun()
        except Exception as e:
            st.session_state["message"] = f"予想生成失敗: {e}"

with p2:
    if st.button("この予想を保存", use_container_width=True):
        pred_df = st.session_state.get("pred_df")
        if pred_df is None or pred_df.empty:
            st.error("先に買い目を出してください。")
        else:
            try:
                save_current_prediction(
                    race_name=st.session_state.get("race_name", ""),
                    url=st.session_state.get("last_url", ""),
                    mode=detected_mode,
                    weather=weather,
                    race_type=race_type,
                    lineup_string=st.session_state.get("lineup_string", ""),
                    ticket_type=ticket_type,
                    current_df=current_df,
                    pred_df=pred_df,
                    odds_dict=st.session_state.get("odds_dict", {}),
                    unit_bet=unit_bet,
                    display_count=display_count,
                )
                st.success("予想レースを保存しました。")
            except Exception as e:
                st.error(f"保存失敗: {e}")

pred_df = st.session_state.get("pred_df")

if st.session_state.get("race_assessment"):
    ra = st.session_state.get("race_assessment")
    if ra.get("decision") in ["買い", "軽く買い"]:
        st.success(race_buyability_summary_text(ra))
    elif ra.get("decision") == "注意":
        st.warning(race_buyability_summary_text(ra))
    else:
        st.error(race_buyability_summary_text(ra))
    if ra.get("advice"):
        st.caption(ra.get("advice"))

if pred_df is not None and isinstance(pred_df, pd.DataFrame) and not pred_df.empty:
    show_df = pred_df.copy()

    cols_order = [
        c for c in [
            "レース判定", "的中率評価", "レース評価点", "判定理由", "見送りAIコメント",
            "買い目ランク", "買い目", "AI評価", "期待値", "学習補正", "学習理由",
            "オッズ", "厚張り指数", "賭け金AI係数", "賭け金AI理由", "購入金額", "期待回収額(目安)"
        ]
        if c in show_df.columns
    ]
    remain = [c for c in show_df.columns if c not in cols_order]
    show_df = show_df[cols_order + remain]

    if "レース判定" in show_df.columns:
        race_decision = str(show_df.iloc[0]["レース判定"])
        hit_label = str(show_df.iloc[0].get("的中率評価", ""))
        race_score = str(show_df.iloc[0].get("レース評価点", ""))
        reason = str(show_df.iloc[0].get("判定理由", ""))

        if race_decision == "買い":
            st.success(f"レース判定: {race_decision} / 的中率評価: {hit_label} / 評価点: {race_score}")
        elif race_decision == "見送り":
            st.warning(f"レース判定: {race_decision} / 的中率評価: {hit_label} / 評価点: {race_score}")
        else:
            st.info(f"レース判定: {race_decision} / 的中率評価: {hit_label} / 評価点: {race_score}")

        if reason:
            st.caption(f"判定理由: {reason}")

    st.dataframe(show_df, use_container_width=True, hide_index=True)
    st.caption(staking_summary_text(show_df))
    st.metric("合計購入額", f"{int(pd.to_numeric(show_df['購入金額'], errors='coerce').fillna(0).sum()):,}円")
else:
    st.caption("まだ買い目は出していません。")

st.markdown("---")
st.subheader("回収率集計")

log_df = load_log_df()
summary = summarize_log_df(log_df)

m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
m1.metric("保存件数", f"{summary['race_count']}件")
m2.metric("結果保存件数", f"{summary['result_saved_race_count']}件")
m3.metric("的中件数", f"{summary['hit_race_count']}件")
m4.metric("的中率", f"{summary['hit_rate']}%")
m5.metric("投資額", f"{summary['total_invest']:,}円")
m6.metric("払戻額", f"{summary['total_return']:,}円")
m7.metric("回収率", f"{summary['recovery_rate']}%")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["券種別", "モード別", "天候別", "レース種別", "レース別直近20件"])

with tab1:
    if summary["by_ticket_type"].empty:
        st.caption("まだ集計データがありません。")
    else:
        st.dataframe(summary["by_ticket_type"], use_container_width=True, hide_index=True)

with tab2:
    if summary["by_mode"].empty:
        st.caption("まだ集計データがありません。")
    else:
        st.dataframe(summary["by_mode"], use_container_width=True, hide_index=True)

with tab3:
    if summary["by_weather"].empty:
        st.caption("まだ集計データがありません。")
    else:
        st.dataframe(summary["by_weather"], use_container_width=True, hide_index=True)

with tab4:
    if summary["by_race_type"].empty:
        st.caption("まだ集計データがありません。")
    else:
        st.dataframe(summary["by_race_type"], use_container_width=True, hide_index=True)

with tab5:
    if summary["recent_races"].empty:
        st.caption("まだ集計データがありません。")
    else:
        st.dataframe(summary["recent_races"], use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("保存した予想レース一覧")

saved_items = load_saved_races()
if not saved_items:
    st.caption("まだ保存レースはありません。")
else:
    labels = [saved_race_label(x) for x in saved_items]
    label_to_id = {saved_race_label(x): x.get("id") for x in saved_items}

    selected_label = st.selectbox("保存レースを選択", options=labels)
    selected_saved_id = label_to_id.get(selected_label, "")
    selected_item = get_saved_race(selected_saved_id)

    if selected_item:
        result_text = format_saved_result(selected_item)
        hit_ticket = format_saved_hit_ticket(selected_item)

        s1, s2, s3, s4, s5, s6, s7, s8 = st.columns([2, 1, 1, 1, 1, 1, 1.2, 1.2])
        s1.write(f"**レース名:** {selected_item.get('race_name', '')}")
        s2.write(f"**種別:** {selected_item.get('race_type', '通常')}")
        s3.write(f"**券種:** {selected_item.get('ticket_type', '3連単')}")
        s4.write(f"**モード:** {selected_item.get('mode', '')}")
        s5.write(f"**天候:** {selected_item.get('weather', '')}")
        s6.write(f"**判定:** {saved_race_status_label(selected_item)}")
        s7.write(f"**結果:** {result_text if result_text else '-'}")
        s8.write(f"**的中買い目:** {hit_ticket if hit_ticket else '-'}")

        b1, b2 = st.columns([1, 1])

        with b1:
            if st.button("この保存レースを読込", use_container_width=True):
                restore_saved_race_to_session(selected_item)
                st.rerun()

        with b2:
            if st.button("この保存レースを削除", use_container_width=True):
                ok = delete_saved_race(selected_saved_id)
                if ok:
                    st.success("削除しました。")
                    st.rerun()
                else:
                    st.error("削除に失敗しました。")

        saved_pred_rows = selected_item.get("pred_rows", [])
        if saved_pred_rows:
            st.markdown("#### 保存済み買い目")
            st.dataframe(pd.DataFrame(saved_pred_rows), use_container_width=True, hide_index=True)

        st.markdown("#### 一覧から結果保存")
        default_result = selected_item.get("result", {})
        selected_ticket_type = selected_item.get("ticket_type", "3連単")

        with st.form("result_save_from_list"):
            rr1, rr2, rr3 = st.columns(3)
            result_1 = rr1.text_input("1着", value=str(default_result.get("1着", "")))
            result_2 = rr2.text_input("2着", value=str(default_result.get("2着", "")))
            result_3 = rr3.text_input("3着", value=str(default_result.get("3着", "")))
            submit_result = st.form_submit_button("この保存レースに結果を保存", use_container_width=True)

        if selected_ticket_type == "2車単":
            st.caption("2車単判定は 1着-2着 で行います。3着は保存だけされます。")

        if submit_result:
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
                    race_type=selected_item.get("race_type", "通常"),
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
                st.success(f"結果を保存しました: {LOG_PATH.name} / 判定: {hit_info['status_label']}")
                st.rerun()
            except Exception as e:
                st.error(f"保存失敗: {e}")

st.markdown("---")
st.subheader("現在表示中の予想をそのまま結果保存")

if pred_df is not None and isinstance(pred_df, pd.DataFrame) and not pred_df.empty:
    with st.form("direct_result_form"):
        r1, r2, r3 = st.columns(3)
        result_1 = r1.text_input("1着", value="", key="result_1")
        result_2 = r2.text_input("2着", value="", key="result_2")
        result_3 = r3.text_input("3着", value="", key="result_3")
        save_now = st.form_submit_button("結果を保存", use_container_width=True)

    if ticket_type == "2車単":
        st.caption("2車単判定は 1着-2着 で行います。")

    if save_now:
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
                race_type=race_type,
                lineup=st.session_state.get("lineup_string", ""),
                ticket_type=ticket_type,
                pred_df=pred_df,
                result_1=result_1,
                result_2=result_2,
                result_3=result_3,
                hit_status=hit_info["status_label"],
            )
            st.success(f"結果を保存しました: {LOG_PATH.name} / 判定: {hit_info['status_label']}")
        except Exception as e:
            st.error(f"保存失敗: {e}")
else:
    st.caption("先に買い目を出してください。")

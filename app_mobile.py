# app_mobile.py
# -*- coding: utf-8 -*-

import re
import itertools
import csv
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

from predict import auto_detect_mode, generate_predictions
from learning import apply_learning_correction, learning_summary_text


st.set_page_config(page_title="競輪AI Mobile", page_icon="🚴", layout="centered")

st.title("🚴 競輪AI Mobile")
st.caption("安定版 / 5・6・7・9車 / ガールズ対応")

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
}

DEFAULT_COLUMNS = ["車番", "選手名", "競走得点", "脚質", "ライン", "ライン順", "単騎"]

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_PATH = SCRIPT_DIR / "log.csv"

PREFECTURES = [
    "北海道", "青森", "岩手", "宮城", "秋田", "山形", "福島",
    "茨城", "栃木", "群馬", "埼玉", "千葉", "東京", "神奈川",
    "新潟", "富山", "石川", "福井", "山梨", "長野",
    "岐阜", "静岡", "愛知", "三重", "滋賀", "京都", "大阪", "兵庫",
    "奈良", "和歌山", "鳥取", "島根", "岡山", "広島", "山口",
    "徳島", "香川", "愛媛", "高知", "福岡", "佐賀", "長崎",
    "熊本", "大分", "宮崎", "鹿児島", "沖縄",
]

PREF_PATTERN = "|".join(sorted(PREFECTURES, key=len, reverse=True))

NG_NAMES = set(PREFECTURES + [
    "競走", "得点", "脚質", "ライン", "ライン順", "単騎",
    "本命", "対抗", "単穴", "連下", "勝率", "倍率", "ギヤ",
    "コメント", "自力", "並び", "予想", "出走", "選手", "車番",
    "人気", "オッズ", "取れた", "位置", "取れた位置",
    "取れた位置から", "位置から", "取得", "情報", "更新",
])


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    table = str.maketrans({
        "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
        "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
        "－": "-", "ー": "-", "―": "-", "‐": "-", "ｰ": "-",
        "／": "/", "　": " ", "，": ",", "．": ".",
        "（": "(", "）": ")", "｜": "|", "：": ":", "\xa0": " ",
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


def is_valid_name(name: str) -> bool:
    name = normalize_text(name)
    if not name:
        return False
    if name in NG_NAMES:
        return False
    if re.fullmatch(r"\d+", name):
        return False
    if len(name) < 2 or len(name) > 8:
        return False
    if re.search(r"(取れた|位置|から|競走|得点|脚質|ライン|予想|取得)", name):
        return False
    if not re.search(r"[一-龥々]", name):
        return False
    return True


def normalize_ticket(ticket: str) -> str:
    s = normalize_text(ticket)
    s = s.replace(" ", "")
    s = re.sub(r"[^0-9\-]", "", s)
    return s


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


if "race_rows" not in st.session_state:
    init_state(7)


def get_df():
    df = pd.DataFrame(st.session_state.get("race_rows", []))

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


def build_racecard_urls(url: str):
    u = normalize_text(url).rstrip("/")
    candidates = [u]

    if "/odds/" in u:
        candidates.append(u.replace("/odds/", "/racecard/"))

    if "/racecard/" not in u and "/odds/" not in u and "/keirin/" in u:
        candidates.append(u.replace("/keirin/", "/keirin/racecard/"))

    out, seen = [], set()
    for x in candidates:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def build_odds_urls(url: str):
    u = normalize_text(url).rstrip("/")
    candidates = [u]

    if "/racecard/" in u:
        candidates.append(u.replace("/racecard/", "/odds/"))

    if "/odds/" not in u and "/racecard/" not in u and "/keirin/" in u:
        candidates.append(u.replace("/keirin/", "/keirin/odds/"))

    out, seen = [], set()
    for x in candidates:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def get_page(url: str):
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or r.encoding
    soup = BeautifulSoup(r.text, "html.parser")
    text = normalize_text(soup.get_text(" ", strip=True))
    title = normalize_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
    return {"url": url, "html": r.text, "text": text, "title": title, "status": r.status_code}


def parse_lineup_groups(lineup_text):
    s = normalize_text(lineup_text)
    if not s:
        return []

    s = s.replace("→", "/").replace("|", "/").replace("・", "/")
    s = s.replace(",", "/").replace(";", "/")

    groups = []
    for part in re.split(r"\s*/\s*", s):
        nums = re.findall(r"[1-9]", part)
        if nums:
            groups.append([int(x) for x in nums])

    flat = list(itertools.chain.from_iterable(groups))
    if not flat or len(flat) != len(set(flat)):
        return []

    return groups


def groups_to_lineup_string(groups):
    return " / ".join("-".join(str(x) for x in g) for g in groups if g)


def extract_lineup_window(text):
    s = normalize_text(text)

    for kw in ["並び予想", "予想並び", "並び"]:
        pos = s.find(kw)
        if pos != -1:
            tail = s[pos + len(kw): pos + len(kw) + 1800]
            end_keywords = [
                "基本情報", "直近成績", "前検コメ", "対戦成績",
                "オッズ一覧", "レース情報", "払戻", "結果", "出走表",
                "人気順", "3連単", "2車単", "2車複",
            ]

            end_pos = len(tail)
            for end_kw in end_keywords:
                p = tail.find(end_kw)
                if p != -1:
                    end_pos = min(end_pos, p)

            return normalize_text(tail[:end_pos])

    return ""


def parse_lineup_from_text(text):
    window = extract_lineup_window(text)

    if window:
        tokens = re.findall(r"区切り|/|[1-9]", window)
        groups, current = [], []

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
        if len(flat) in [5, 6, 7, 9] and len(flat) == len(set(flat)):
            if set(flat) == set(range(1, len(flat) + 1)):
                return groups_to_lineup_string(groups)

    pattern = re.compile(r"([1-9](?:\s*-\s*[1-9])*(?:\s*/\s*[1-9](?:\s*-\s*[1-9])*){1,8})")

    for m in pattern.finditer(normalize_text(text)):
        cand = normalize_text(m.group(1))
        groups = parse_lineup_groups(cand)
        flat = list(itertools.chain.from_iterable(groups))

        if len(flat) in [5, 6, 7, 9] and len(flat) == len(set(flat)):
            if set(flat) == set(range(1, len(flat) + 1)):
                return groups_to_lineup_string(groups)

    return ""


def fetch_lineup(url):
    debug = []

    for target_url in build_racecard_urls(url):
        try:
            page = get_page(target_url)
            lineup = parse_lineup_from_text(page["text"])

            debug.append({
                "url": target_url,
                "title": page["title"],
                "lineup": lineup,
                "text_head": page["text"][:500],
            })

            if lineup:
                st.session_state["debug"]["lineup"] = debug
                return lineup

        except Exception as e:
            debug.append({"url": target_url, "error": str(e)})

    st.session_state["debug"]["lineup"] = debug
    raise ValueError("URLから並びを抽出できませんでした。手入力してください。")


def apply_lineup_to_df(df, lineup_text):
    groups = parse_lineup_groups(lineup_text)
    if not groups:
        raise ValueError("並び文字列を解釈できませんでした。例: 1-4 / 2-5 / 3")

    flat = list(itertools.chain.from_iterable(groups))
    riders = set(df["車番"].astype(int).tolist())

    if set(flat) != riders:
        raise ValueError(f"並び {sorted(flat)} と車番 {sorted(riders)} が一致しません。")

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


def extract_players_section(text):
    s = normalize_text(text)

    start_pos = -1
    for kw in ["AI 競走得点", "競走得点", "脚質"]:
        p = s.find(kw)
        if p != -1:
            start_pos = p
            break

    if start_pos == -1:
        return s

    end_pos = len(s)
    for kw in [
        "並び予想", "予想並び", "並び",
        "オッズ一覧", "人気順", "2車単", "3連単", "2車複", "3連複",
    ]:
        p = s.find(kw, start_pos + 1)
        if p != -1:
            end_pos = min(end_pos, p)

    section = normalize_text(s[start_pos:end_pos])
    return section if len(section) >= 250 else s


def split_blocks_by_car(text, num_riders):
    s = normalize_text(text)
    blocks = {}
    positions = []

    for car in range(1, num_riders + 1):
        patterns = [
            rf"(?<!\d){car}\s+{car}\s+",
            rf"(?<!\d){car}\s+{car}(?=[一-龥ぁ-んァ-ヶ々])",
            rf"(?<!\d){car}\s+(?=[一-龥ぁ-んァ-ヶ々])",
            rf"(?<!\d){car}(?=[一-龥ぁ-んァ-ヶ々])",
        ]

        found = None
        for pat in patterns:
            m = re.search(pat, s)
            if m:
                found = m.start()
                break

        if found is not None:
            positions.append((car, found))

    positions = sorted(positions, key=lambda x: x[1])

    for i, (car, start) in enumerate(positions):
        end = positions[i + 1][1] if i + 1 < len(positions) else len(s)
        block = normalize_text(s[start:end])[:1800]
        blocks[car] = block

    return blocks


def extract_name(block):
    b = normalize_text(block)

    exact = re.search(
        rf"[1-9]?\s*([一-龥々]{{2,5}})\s+(?:{PREF_PATTERN})\s+(?:SS|S1|S2|A1|A2|L1|L2)\s+\d{{2}}歳\s+\d{{2,3}}期",
        b,
    )
    if exact:
        name = normalize_text(exact.group(1))
        if is_valid_name(name):
            return name

    pref_before = re.search(rf"([一-龥々]{{2,5}})\s+(?:{PREF_PATTERN})", b)
    if pref_before:
        name = normalize_text(pref_before.group(1))
        if is_valid_name(name):
            return name

    candidates = re.findall(r"[一-龥々]{2,5}", b)
    for cand in candidates:
        cand = normalize_text(cand)
        if is_valid_name(cand):
            return cand

    return ""


def extract_score(block):
    b = normalize_text(block)

    patterns = [
        re.compile(r"\d{2,3}期\s+(?:本命|対抗|単穴|連下)?\s*([4-9]\d(?:\.\d{1,3})?)"),
        re.compile(r"(?:本命|対抗|単穴|連下)\s*([4-9]\d(?:\.\d{1,3})?)"),
        re.compile(r"([4-9]\d(?:\.\d{1,3})?)\s+\d+\s+\d+\s+\d+\s+(?:逃|捲|追|両|自)"),
    ]

    for pat in patterns:
        m = pat.search(b)
        if m:
            v = safe_float(m.group(1), 0.0)
            if 40 <= v <= 130:
                return v

    candidates = []
    for m in re.finditer(r"([4-9]\d(?:\.\d{1,3})?)", b):
        v = safe_float(m.group(1), 0.0)
        before = b[max(0, m.start() - 4):m.start()]
        after = b[m.end():m.end() + 4]

        if "期" in before or "期" in after:
            continue
        if "歳" in before or "歳" in after:
            continue
        if "勝率" in before or "勝率" in after:
            continue

        if 40 <= v <= 130:
            candidates.append(v)

    return candidates[0] if candidates else 0.0


def extract_style(block):
    b = normalize_text(block)

    patterns = [
        re.compile(r"(?:本命|対抗|単穴|連下)?\s*[4-9]\d(?:\.\d{1,3})?\s+\d+\s+\d+\s+\d+\s+(逃|捲|追|両|自)"),
        re.compile(r"[4-9]\d(?:\.\d{1,3})?(?:\s+\d+){0,6}\s+(逃|捲|追|両|自)"),
    ]

    for pat in patterns:
        m = pat.search(b)
        if m:
            return normalize_text(m.group(1))

    m = re.search(r"(逃|捲|追|両|自)", b)
    return normalize_text(m.group(1)) if m else ""


def extract_single_player_by_car(text, car, num_riders):
    section = extract_players_section(text)
    blocks = split_blocks_by_car(section, num_riders)

    block = blocks.get(car, "")
    if not block:
        blocks = split_blocks_by_car(text, num_riders)
        block = blocks.get(car, "")

    if not block:
        return None

    name = extract_name(block)
    score = extract_score(block)
    style = extract_style(block)

    if is_valid_name(name) and 40 <= score <= 130:
        return {
            "車番": car,
            "選手名": name,
            "競走得点": score,
            "脚質": style,
            "source": "block_split_safe_v3",
            "block_head": block[:180],
        }

    return None


def extract_players_from_text(text, num_riders):
    rows = []
    preview = []

    for car in range(1, num_riders + 1):
        hit = extract_single_player_by_car(text, car, num_riders)
        if hit:
            rows.append({
                "車番": hit["車番"],
                "選手名": hit["選手名"],
                "競走得点": hit["競走得点"],
                "脚質": hit["脚質"],
            })
            preview.append(hit)

    if not rows:
        return pd.DataFrame(), preview

    df = pd.DataFrame(rows).groupby("車番", as_index=False).first()
    df = df.sort_values("車番").reset_index(drop=True)

    return df[["車番", "選手名", "競走得点", "脚質"]], preview


def fetch_players(url, num_riders):
    debug = []
    best_df = pd.DataFrame()

    for target_url in build_racecard_urls(url):
        try:
            page = get_page(target_url)
            df, preview = extract_players_from_text(page["text"], num_riders)

            missing = (
                sorted(list(set(range(1, num_riders + 1)) - set(df["車番"].tolist())))
                if not df.empty
                else list(range(1, num_riders + 1))
            )

            debug.append({
                "url": target_url,
                "title": page["title"],
                "hit_count": len(df),
                "missing": missing,
                "preview": preview,
                "text_head": page["text"][:500],
            })

            if len(df) > len(best_df):
                best_df = df

        except Exception as e:
            debug.append({"url": target_url, "error": str(e)})

    st.session_state["debug"]["players"] = debug

    if best_df.empty:
        raise ValueError("選手情報を自動取得できませんでした。")

    return best_df


def apply_players_to_df(df, players_df):
    out = df.copy()

    for _, row in players_df.iterrows():
        car = int(row["車番"])
        name = normalize_text(row.get("選手名", ""))
        score = safe_float(row.get("競走得点", 0.0))
        style = normalize_text(row.get("脚質", ""))

        if is_valid_name(name):
            out.loc[out["車番"] == car, "選手名"] = name
        if 40 <= score <= 130:
            out.loc[out["車番"] == car, "競走得点"] = score
        if style in ["逃", "捲", "追", "両", "自"]:
            out.loc[out["車番"] == car, "脚質"] = style

    return out


def extract_script_texts(html):
    soup = BeautifulSoup(html, "html.parser")
    items = []

    for tag in soup.find_all("script"):
        txt = tag.string if tag.string else tag.get_text(" ", strip=True)
        txt = normalize_text(txt or "")
        if txt:
            items.append(txt)

    return items


def extract_odds_loose(text, ticket_type):
    s = normalize_text(text)
    results = {}

    if ticket_type == "2車単":
        patterns = [
            re.compile(r"(?<!\d)([1-9])\s*-\s*([1-9])\s+([0-9]+(?:\.[0-9]+)?)(?!\d)"),
            re.compile(r'"combination"\s*:\s*"([1-9]-[1-9])".{0,100}?"odds"\s*:\s*([0-9]+(?:\.[0-9]+)?)'),
        ]
    else:
        patterns = [
            re.compile(r"(?<!\d)([1-9])\s*-\s*([1-9])\s*-\s*([1-9])\s+([0-9]+(?:\.[0-9]+)?)(?!\d)"),
            re.compile(r'"combination"\s*:\s*"([1-9]-[1-9]-[1-9])".{0,100}?"odds"\s*:\s*([0-9]+(?:\.[0-9]+)?)'),
        ]

    for pat in patterns:
        for m in pat.finditer(s):
            g = m.groups()

            if ticket_type == "2車単":
                if len(g) == 3:
                    key = f"{g[0]}-{g[1]}"
                    val = safe_float(g[2])
                elif len(g) == 2:
                    key = normalize_ticket(g[0])
                    val = safe_float(g[1])
                else:
                    continue

                parts = key.split("-")
                if len(parts) == 2 and len(set(parts)) == 2 and val > 0:
                    results[key] = val

            else:
                if len(g) == 4:
                    key = f"{g[0]}-{g[1]}-{g[2]}"
                    val = safe_float(g[3])
                elif len(g) == 2:
                    key = normalize_ticket(g[0])
                    val = safe_float(g[1])
                else:
                    continue

                parts = key.split("-")
                if len(parts) == 3 and len(set(parts)) == 3 and val > 0:
                    results[key] = val

    return results


def fetch_odds(url, ticket_type):
    debug = []
    best = {}

    for target_url in build_odds_urls(url):
        try:
            page = get_page(target_url)
            found = {}
            found.update(extract_odds_loose(page["text"], ticket_type))

            for script_text in extract_script_texts(page["html"]):
                found.update(extract_odds_loose(script_text, ticket_type))

            debug.append({
                "url": target_url,
                "title": page["title"],
                "hit_count": len(found),
                "preview": list(sorted(found.items(), key=lambda x: x[1]))[:10],
            })

            if len(found) > len(best):
                best = found

        except Exception as e:
            debug.append({"url": target_url, "error": str(e)})

    st.session_state["debug"]["odds"] = debug

    if not best:
        raise ValueError("オッズを抽出できませんでした。")

    return best




# =========================
# 結果保存 / 学習ログ
# =========================
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


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

    if ticket_type == "2車単":
        result_text = "-".join([x for x in [result_1, result_2] if x])
    else:
        result_text = "-".join([x for x in [result_1, result_2, result_3] if x])

    with open(LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow([
                "保存日時", "レース名", "券種", "モード", "天候", "レース種別", "並び", "結果", "判定",
                "買い目", "買い目ランク", "AI評価", "期待値", "オッズ",
                "購入金額", "期待回収額(目安)", "レース判定", "的中率評価", "レース評価点", "判定理由",
                "学習補正", "学習理由",
            ])

        for _, row in pred_df.iterrows():
            writer.writerow([
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
                row.get("学習補正", ""),
                row.get("学習理由", ""),
            ])


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

race_name = st.text_input("レース名", value=st.session_state.get("race_name", ""))
st.session_state["race_name"] = race_name

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
        style = st.selectbox(f"脚質_{i}", style_options, index=style_options.index(style_now))

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
st.caption(learning_summary_text(LOG_PATH))

detected_mode = auto_detect_mode(current_df)

if race_type == "ガールズ":
    st.info("ガールズモード（表示のみ / 予想生成はpredict.py準拠）")
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
        )

        if pred is None or pred.empty:
            st.session_state["message"] = "買い目が生成できませんでした。"
            st.rerun()

        pred = apply_learning_correction(
            pred,
            LOG_PATH,
            mode=detected_mode,
            weather=weather,
            ticket_type=ticket_type,
        )

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
        race_score = str(first.get("レース評価点", ""))
        reason = str(first.get("判定理由", ""))

        if decision == "買い":
            st.success(f"レース判定: {decision} / 的中率評価: {hit_label} / 評価点: {race_score}")
        elif decision == "見送り":
            st.warning(f"レース判定: {decision} / 的中率評価: {hit_label} / 評価点: {race_score}")
        else:
            st.info(f"レース判定: {decision} / 的中率評価: {hit_label} / 評価点: {race_score}")

        if reason:
            st.caption(f"判定理由: {reason}")

    for _, row in pred_df.iterrows():
        with st.container(border=True):
            st.markdown(f"### {row.get('買い目ランク', '')} {row.get('買い目', '')}")
            st.write(f"AI評価: {row.get('AI評価', '-')}")
            st.write(f"期待値: {row.get('期待値', '-')}")
            st.write(f"オッズ: {row.get('オッズ', '-')}")
            st.write(f"学習補正: {row.get('学習補正', '-')}")
            st.write(f"学習理由: {row.get('学習理由', '-')}")
            st.write(f"購入金額: {int(safe_float(row.get('購入金額', 0))):,}円")

    total = int(pd.to_numeric(pred_df["購入金額"], errors="coerce").fillna(0).sum())
    st.metric("合計購入額", f"{total:,}円")

    st.markdown("### 📝 結果保存")
    with st.form("mobile_result_form"):
        r1, r2, r3 = st.columns(3)
        result_1 = r1.text_input("1着", value="")
        result_2 = r2.text_input("2着", value="")
        result_3 = r3.text_input("3着", value="")
        save_result = st.form_submit_button("結果を保存", use_container_width=True)

    if ticket_type == "2車単":
        st.caption("2車単判定は 1着-2着 で行います。")

    if save_result:
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
            st.rerun()
        except Exception as e:
            st.error(f"保存失敗: {e}")

with st.expander("デバッグ", expanded=False):
    st.write(st.session_state.get("debug", {}))
    st.write("取得オッズ件数:", len(st.session_state.get("odds_dict", {})))

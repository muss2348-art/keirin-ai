# -*- coding: utf-8 -*-
"""
競輪レース選定AI Mobile版
買い目は出さず、その日のレースから「狙う価値がありそうなレース」を選別する専用アプリ。

使い方:
    streamlit run app_race_selector.py

入力:
    WINTICKETの開催ページURL、またはレースURL、または複数レースURL

出力:
    勝負 / 候補 / 見送り
    勝負度
    的中率重視向き / 回収率重視向き
    理由
"""

import re
import math
import itertools
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from urllib.parse import urljoin

import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

st.set_page_config(
    page_title="競輪レース選定AI",
    page_icon="🔥",
    layout="centered",
)

APP_VERSION = "race-selector-mobile v2.3（指数印表調整版）"

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

RACE_SELECTION_LOG_PATH = "race_selection_log.csv"

DEFAULT_TIMEOUT = 15

VENUE_NAME_MAP = {
    "hakodate": "函館", "aomori": "青森", "iwaki": "いわき平", "toride": "取手",
    "utsunomiya": "宇都宮", "maebashi": "前橋", "omiya": "大宮", "seibuen": "西武園",
    "keiokaku": "京王閣", "tachikawa": "立川", "matsudo": "松戸", "chiba": "千葉",
    "kawasaki": "川崎", "hiratsuka": "平塚", "odawara": "小田原", "ito": "伊東",
    "shizuoka": "静岡", "nagoya": "名古屋", "gifu": "岐阜", "ogaki": "大垣",
    "toyohashi": "豊橋", "toyama": "富山", "matsusaka": "松阪", "yokkaichi": "四日市",
    "fukui": "福井", "nara": "奈良", "kyoto": "京都向日町", "kishiwada": "岸和田",
    "wakayama": "和歌山", "tamano": "玉野", "hiroshima": "広島", "hofu": "防府",
    "takamatsu": "高松", "komatsushima": "小松島", "kochi": "高知", "matsuyama": "松山",
    "kokura": "小倉", "kurume": "久留米", "takeo": "武雄", "sasebo": "佐世保",
    "beppu": "別府", "kumamoto": "熊本",
}

PREFECTURES = [
    "北海道", "青森", "岩手", "宮城", "秋田", "山形", "福島",
    "茨城", "栃木", "群馬", "埼玉", "千葉", "東京", "神奈川",
    "新潟", "富山", "石川", "福井", "山梨", "長野", "岐阜", "静岡",
    "愛知", "三重", "滋賀", "京都", "大阪", "兵庫", "奈良", "和歌山",
    "鳥取", "島根", "岡山", "広島", "山口", "徳島", "香川", "愛媛", "高知",
    "福岡", "佐賀", "長崎", "熊本", "大分", "宮崎", "鹿児島", "沖縄",
]

@dataclass
class RaceScore:
    rank: int
    race_name: str
    race_url: str
    judge: str
    style: str
    confidence: int
    riders: int
    avg_score: float
    max_score: float
    score_gap: float
    line_count: int
    solo_count: int
    main_line_size: int
    max_b_count: int
    max_h_count: int
    b_data_count: int
    h_data_count: int
    advanced_data_count: int
    reason: str
    warnings: str


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    table = str.maketrans({
        "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
        "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
        "－": "-", "ー": "-", "―": "-", "‐": "-", "ｰ": "-",
        "／": "/", "　": " ", "，": ",", "．": ".", "（": "(", "）": ")",
        "：": ":", "\xa0": " ", "｜": "|",
    })
    s = str(s).translate(table)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def safe_float(v, default=0.0) -> float:
    try:
        if v is None or v == "":
            return float(default)
        return float(str(v).replace(",", "").replace("%", "").strip())
    except Exception:
        return float(default)


def fetch_html(url: str) -> Tuple[str, str]:
    res = requests.get(url, headers=HEADERS, timeout=DEFAULT_TIMEOUT)
    res.raise_for_status()
    return res.text, res.url


def clean_urls(raw: str) -> List[str]:
    urls = re.findall(r"https?://[^\s\n\r\t]+", raw or "")
    out = []
    for u in urls:
        u = u.rstrip("、,。)]}")
        if u not in out:
            out.append(u)
    return out


def race_number_from_url(url: str) -> Optional[int]:
    m = re.search(r"/racecard/[^/]+/(\d+)/(\d+)", url)
    if not m:
        return None
    return int(m.group(2))


def venue_from_url(url: str) -> str:
    m = re.search(r"/keirin/([^/]+)/", url)
    if not m:
        return ""
    slug = m.group(1)
    return VENUE_NAME_MAP.get(slug, slug)


def build_race_urls_from_one_url(url: str, max_race: int = 12) -> List[str]:
    """レースURLなら同開催の1R〜max_raceを生成。開催ページならページ内リンクを拾う。"""
    url = url.strip()
    if not url:
        return []

    m = re.search(r"(.*/racecard/[^/]+/\d+/)(\d+)(?:[/?#].*)?$", url)
    if m:
        base = m.group(1)
        return [f"{base}{i}" for i in range(1, max_race + 1)]

    # 開催ページ・出走表ページから racecard リンクを拾う
    try:
        html, final_url = fetch_html(url)
        soup = BeautifulSoup(html, "html.parser")
        urls = []
        for a in soup.find_all("a", href=True):
            href = urljoin(final_url, a.get("href"))
            if "/keirin/" in href and "/racecard/" in href:
                href = href.split("?")[0].split("#")[0]
                if re.search(r"/racecard/[^/]+/\d+/\d+", href) and href not in urls:
                    urls.append(href)
        urls = sorted(urls, key=lambda x: race_number_from_url(x) or 999)
        if urls:
            return urls[:max_race]
    except Exception:
        pass

    return [url]


def extract_scores(text: str) -> List[float]:
    """ページ本文から競走得点らしい80〜130台の数値を抽出。"""
    text = normalize_text(text)
    candidates = []
    for m in re.finditer(r"(?<!\d)([89]\d\.\d{1,2}|1[01]\d\.\d{1,2}|12[0-9]\.\d{1,2})(?!\d)", text):
        v = safe_float(m.group(1), 0)
        if 75 <= v <= 130:
            candidates.append(v)

    # 近接重複を削る。WINTICKET本文は同じ得点が複数出ることがある。
    unique = []
    for v in candidates:
        if not any(abs(v - x) < 0.001 for x in unique):
            unique.append(v)
        if len(unique) >= 9:
            break
    return unique


def _unique_limited_numbers(values: List[float], limit: int = 9, eps: float = 0.001) -> List[float]:
    """同じ数値の重複を抑えて、最大limit件にする。"""
    out = []
    for v in values:
        if not any(abs(v - x) < eps for x in out):
            out.append(v)
        if len(out) >= limit:
            break
    return out


def extract_rate_values(text: str, kind: str) -> List[float]:
    """
    2連対率・3連対率をページ本文から拾う。
    WINTICKET側の表記ゆれに備えて、ラベル周辺の%値を優先して抽出する。
    取れない時は空リストを返し、判定では強く減点しない。
    """
    text = normalize_text(text)
    if kind == "two":
        labels = ["2連対率", "二連対率", "連対率", "2連対", "二連対"]
    else:
        labels = ["3連対率", "三連対率", "三連率", "3連対", "三連対"]

    values = []
    for label in labels:
        # ラベルの直後に選手分の%が並ぶケース
        for m in re.finditer(re.escape(label) + r".{0,240}", text):
            block = m.group(0)
            for p in re.findall(r"(?<!\d)(\d{1,3}(?:\.\d{1,2})?)\s*%", block):
                v = safe_float(p, -1)
                if 0 <= v <= 100:
                    values.append(v)
        # ラベルの直前に%があるケースも保険で見る
        for m in re.finditer(r".{0,80}" + re.escape(label), text):
            block = m.group(0)
            for p in re.findall(r"(?<!\d)(\d{1,3}(?:\.\d{1,2})?)\s*%", block):
                v = safe_float(p, -1)
                if 0 <= v <= 100:
                    values.append(v)

    return _unique_limited_numbers(values, limit=9, eps=0.01)


def extract_bh_pairs_from_winticket_json(html: str) -> List[Dict[str, int]]:
    """
    WINTICKETのscript内JSONから、選手ごとのB数/H数を拾う。

    デバッグ結果では、
      "home": 7 ... "playerId": "xxxxx" ... "racePoint": 104 ... "back": 3
    のように、HがplayerIdの少し前、BがplayerIdの後ろに出るケースがある。
    そのため playerId の前後を小さく切り出して、直近のhomeとbackをセットで拾う。
    """
    if not html:
        return []

    player_matches = list(re.finditer(r'"playerId"\s*:\s*"([^"]+)"', html))
    pairs: List[Dict[str, int]] = []

    for i, m in enumerate(player_matches):
        pid = m.group(1)
        start = m.start()

        prev_start = player_matches[i - 1].start() if i > 0 else 0
        next_start = player_matches[i + 1].start() if i + 1 < len(player_matches) else len(html)

        before = html[max(prev_start, start - 1800):start]
        after = html[start:min(next_start, start + 2200)]
        around = before + after

        if '"racePoint"' not in after and '"style"' not in after:
            continue

        # HはplayerIdより前に出やすいので、前方の最後のhomeを採用
        home_candidates = re.findall(r'"home"\s*:\s*(\d{1,3})', before)
        # BはplayerIdより後ろに出やすいので、後方の最初のbackを採用
        back_candidates = re.findall(r'"back"\s*:\s*(\d{1,3})', after)

        home = int(home_candidates[-1]) if home_candidates else 0
        back = int(back_candidates[0]) if back_candidates else 0

        if not (0 <= back <= 60 and 0 <= home <= 60):
            continue

        race_point_m = re.search(r'"racePoint"\s*:\s*([0-9]+(?:\.[0-9]+)?)', after)
        style_m = re.search(r'"style"\s*:\s*"([^"]*)"', after)

        # playerIdだけの別JSONを拾わないように、得点か脚質があるものだけ採用
        if not race_point_m and not style_m:
            continue

        pairs.append({
            "player_id": pid,
            "race_point": safe_float(race_point_m.group(1), 0.0) if race_point_m else 0.0,
            "style": style_m.group(1) if style_m else "",
            "back": back,
            "home": home,
        })

    unique: List[Dict[str, int]] = []
    seen = set()
    for p in pairs:
        pid = str(p.get("player_id", ""))
        if pid in seen:
            continue
        seen.add(pid)
        unique.append(p)

    return unique[:9]


def extract_bh_values_from_html(html: str, label: str) -> List[int]:
    pairs = extract_bh_pairs_from_winticket_json(html)
    key = "back" if label == "B" else "home"
    vals = []
    for p in pairs:
        v = int(p.get(key, 0))
        if 0 <= v <= 60:
            vals.append(v)
    return vals



def extract_bh_values(text: str, label: str) -> List[int]:
    """
    B数/H数をなるべく確実に拾う。
    連対率など不安定なデータは使わず、ページ本文に出やすい
    「バック」「ホーム」「B」「H」周辺の整数だけを候補にする。
    取れない時は空リストを返し、勝負度では強く減点しない。
    """
    text = normalize_text(text)
    compact = re.sub(r"\s+", " ", text)
    values: List[int] = []

    if label == "B":
        jp_labels = ["バック", "B数", "B回", "Ｂ数"]
        en = "B"
    else:
        jp_labels = ["ホーム", "H数", "H回", "Ｈ数"]
        en = "H"

    # 1) 日本語ラベル直後の数字を最優先
    for lb in jp_labels:
        for m in re.finditer(re.escape(lb) + r"\s*[:：]?\s*(\d{1,2})", compact):
            v = int(m.group(1))
            if 0 <= v <= 40:
                values.append(v)

    # 2) ラベル周辺の短いブロックから拾う
    for lb in jp_labels:
        for m in re.finditer(re.escape(lb) + r".{0,140}", compact):
            block = m.group(0)
            for n in re.findall(r"(?<!\d)(\d{1,2})(?!\d)", block):
                v = int(n)
                if 0 <= v <= 40:
                    values.append(v)

    # 3) 英字B/Hの直後だけ拾う。短いラベルなので広く拾いすぎない。
    for m in re.finditer(rf"(?<![A-Za-z0-9]){en}\s*[:：]?\s*(\d{{1,2}})(?!\d)", compact):
        v = int(m.group(1))
        if 0 <= v <= 40:
            values.append(v)

    # 4) 「B H 12 8」のように見出し後に数字が続くケースの保険
    header_pat = r"B\s*H\s*((?:\d{1,2}\s*){2,24})" if label == "B" else r"B\s*H\s*((?:\d{1,2}\s*){2,24})"
    for m in re.finditer(header_pat, compact):
        nums = [int(x) for x in re.findall(r"\d{1,2}", m.group(1))]
        # B H が交互に並んでいる想定で分ける
        picked = nums[0::2] if label == "B" else nums[1::2]
        for v in picked:
            if 0 <= v <= 40:
                values.append(v)

    # レースは通常7〜9車。多すぎる場合は先頭側を優先。
    out = [int(x) for x in _unique_limited_numbers([float(v) for v in values], limit=9, eps=0.001)]
    return out

def parse_lineup_groups(lineup_text: str) -> List[List[int]]:
    s = normalize_text(lineup_text)
    if not s:
        return []

    s = s.replace("|", "/").replace("・", "/").replace(">", "/").replace("→", "/")
    s = s.replace(",", "/").replace(";", "/")

    raw_groups = re.split(r"\s*/\s*", s)
    groups: List[List[int]] = []

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


def groups_to_lineup_string(groups: List[List[int]]) -> str:
    return " / ".join("-".join(str(x) for x in g) for g in groups if g)


def extract_lineup_windows(page_text: str) -> List[str]:
    """
    買い目AI側の並び取得ロジックを移植。
    WINTICKET本文から「並び予想」「予想並び」「周回予想」などの周辺だけを切り出す。
    """
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

    windows: List[str] = []
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


def parse_lineup_candidate_string(candidate: str) -> Optional[str]:
    groups = parse_lineup_groups(candidate)
    if not groups:
        return None

    flat = list(itertools.chain.from_iterable(groups))
    if len(flat) not in (5, 6, 7, 8, 9):
        return None

    expected = set(range(1, len(flat) + 1))
    if set(flat) != expected:
        return None

    return groups_to_lineup_string(groups)


def _lineup_from_token_window(text: str) -> Optional[str]:
    """
    区切りや / がある近辺から、余計な数字を除いて並びを復元する保険。
    例: 1-3-5 / 2-7 / 4 / 6
    """
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
        groups: List[List[int]] = []
        current: List[int] = []
        for t in tokens:
            if t in ["区切り", "/"]:
                if current:
                    groups.append(current)
                    current = []
            elif t in ["-", "→"]:
                continue
            else:
                current.append(int(t))
        if current:
            groups.append(current)

        flat = list(itertools.chain.from_iterable(groups))
        if len(flat) in (5, 6, 7, 8, 9) and len(set(flat)) == len(flat):
            expected = set(range(1, len(flat) + 1))
            if set(flat) == expected:
                return groups_to_lineup_string(groups)

    nums = [int(t) for t in tokens if re.fullmatch(r"[1-9]", t)]
    for n in (9, 8, 7, 6, 5):
        if len(nums) < n:
            continue
        expected = set(range(1, n + 1))
        for i in range(0, len(nums) - n + 1):
            chunk = nums[i:i + n]
            if len(set(chunk)) == n and set(chunk) == expected:
                return groups_to_lineup_string([[x] for x in chunk])

    return None


def parse_lineup_from_page_text(page_text: str) -> Optional[str]:
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


def extract_lineup_text(text: str) -> str:
    lineup = parse_lineup_from_page_text(text)
    return lineup or ""


def estimate_rider_count(text: str, scores: List[float]) -> int:
    nums = set(int(x) for x in re.findall(r"(?<!\d)([1-9])番", text))
    if nums:
        return max(nums)
    return min(max(len(scores), 7), 9) if scores else 0


def estimate_rider_count_from_html(html: str) -> int:
    """
    人数推定を安定化。
    本文の数字やライン数字ではなく、WINTICKETの選手JSONを優先する。
    """
    if not html:
        return 0

    pairs = extract_bh_pairs_from_winticket_json(html)
    if 5 <= len(pairs) <= 9:
        return len(pairs)

    player_matches = list(re.finditer(r'"playerId"\s*:\s*"([^"]+)"', html))
    valid_ids = []
    for i, m in enumerate(player_matches):
        start = m.start()
        next_start = player_matches[i + 1].start() if i + 1 < len(player_matches) else min(len(html), start + 3000)
        block = html[start:next_start]
        if '"racePoint"' in block or '"style"' in block:
            pid = m.group(1)
            if pid not in valid_ids:
                valid_ids.append(pid)

    if 5 <= len(valid_ids) <= 9:
        return len(valid_ids)

    return 0


def analyze_bh_strategy(b_counts: List[int], h_counts: List[int]) -> Dict[str, object]:
    """
    B/Hから展開の読みやすさを判定する。
    先行候補が絞れるレースは的中率寄り、候補が多すぎるレースは回収率寄りにする。
    """
    b = [int(x) for x in b_counts if 0 <= int(x) <= 60]
    h = [int(x) for x in h_counts if 0 <= int(x) <= 60]
    n = max(len(b), len(h))

    info = {
        "score_delta": 0,
        "type_hint": "",
        "reasons": [],
        "warnings": [],
    }

    if n < 5:
        info["warnings"].append("B/Hデータ不足")
        return info

    max_b = max(b) if b else 0
    max_h = max(h) if h else 0
    avg_b = sum(b) / len(b) if b else 0.0
    avg_h = sum(h) / len(h) if h else 0.0

    high_b = sum(1 for x in b if x >= 10)
    high_h = sum(1 for x in h if x >= 10)
    mid_b = sum(1 for x in b if x >= 6)
    mid_h = sum(1 for x in h if x >= 6)

    if max_b >= 10 and high_b <= 2:
        info["score_delta"] += 8
        info["reasons"].append("主導権候補が絞りやすい")
        info["type_hint"] = "的中率重視向き"

    if max_h >= 10 and high_h <= 2:
        info["score_delta"] += 6
        info["reasons"].append("先行意欲の高い選手が見える")
        if not info["type_hint"]:
            info["type_hint"] = "的中率重視向き"

    if max_b >= 8 and max_h >= 8 and (high_b + high_h) <= 4:
        info["score_delta"] += 5
        info["reasons"].append("B/Hから展開軸を作りやすい")

    if high_b >= 4 or high_h >= 4:
        info["score_delta"] -= 7
        info["warnings"].append("主導権候補が多く展開が割れやすい")
        info["type_hint"] = "回収率重視向き"

    if mid_b >= 5 or mid_h >= 5:
        info["score_delta"] -= 3
        info["warnings"].append("自力候補が多め")
        if not info["type_hint"]:
            info["type_hint"] = "回収率重視向き"

    if max_b <= 4 and max_h <= 4:
        info["score_delta"] -= 6
        info["warnings"].append("B/Hが低く展開材料が少ない")

    if avg_b >= 4 or avg_h >= 4:
        info["score_delta"] += 2
        info["reasons"].append("B/H材料あり")

    return info


def analyze_race(url: str) -> RaceScore:
    html, final_url = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")
    title = normalize_text(soup.title.get_text(" ") if soup.title else "")
    text = normalize_text(soup.get_text(" "))

    venue = venue_from_url(final_url)
    rno = race_number_from_url(final_url)
    race_name = f"{venue} {rno}R" if venue and rno else (title[:40] or final_url)

    scores = extract_scores(text)

    # 人数は本文推定よりWINTICKET script内の選手JSONを優先する
    rider_count = estimate_rider_count_from_html(html)
    if rider_count <= 0:
        rider_count = estimate_rider_count(text, scores)
    avg_score = round(sum(scores) / len(scores), 2) if scores else 0.0
    max_score = round(max(scores), 2) if scores else 0.0
    sorted_scores = sorted(scores, reverse=True)
    score_gap = round((sorted_scores[0] - sorted_scores[2]), 2) if len(sorted_scores) >= 3 else 0.0
    top_gap = round((sorted_scores[0] - sorted_scores[1]), 2) if len(sorted_scores) >= 2 else 0.0

    lineup = extract_lineup_text(text)
    groups = parse_lineup_groups(lineup)
    line_count = len(groups)
    solo_count = sum(1 for g in groups if len(g) == 1)
    main_line_size = max([len(g) for g in groups], default=0)

    # 連対率は一旦使わない。まずはWINTICKET script内JSONのB/Hを優先取得する。
    b_counts = extract_bh_values_from_html(html, "B")
    h_counts = extract_bh_values_from_html(html, "H")
    if not b_counts:
        b_counts = extract_bh_values(text, "B")
    if not h_counts:
        h_counts = extract_bh_values(text, "H")

    # B/Hが5〜9車分取れている場合、人数推定が小さすぎたら補正する
    bh_count = max(len(b_counts), len(h_counts))
    if 5 <= bh_count <= 9 and (rider_count <= 0 or rider_count < bh_count):
        rider_count = bh_count

    max_b_count = int(max(b_counts)) if b_counts else 0
    max_h_count = int(max(h_counts)) if h_counts else 0
    b_data_count = int(len(b_counts))
    h_data_count = int(len(h_counts))
    advanced_data_count = int(len(b_counts) + len(h_counts))
    bh_strategy = analyze_bh_strategy(b_counts, h_counts)

    # AI自信度の中心は「得点の強弱」「B/H」「ラインのまとまり」。ライン数は減点材料止まり。
    confidence = 45
    reasons = []
    warnings = []

    if len(scores) >= 7:
        confidence += 12
        reasons.append("得点データ取得良好")
    elif len(scores) >= 5:
        confidence += 5
        warnings.append("得点データやや不足")
    else:
        confidence -= 18
        warnings.append("得点データ不足")

    if score_gap >= 4.5:
        confidence += 16
        reasons.append("上位得点差が大きい")
    elif score_gap >= 2.5:
        confidence += 9
        reasons.append("上位得点差あり")
    elif score_gap <= 1.5 and score_gap > 0:
        confidence -= 12
        warnings.append("得点差が小さく混戦")

    if top_gap >= 2.0:
        confidence += 6
        reasons.append("軸候補が作りやすい")

    # 追加データ: B数・H数。
    # 取得できたかだけでなく、先行候補が絞れるかまで見る。
    if b_counts or h_counts:
        confidence += 7
        reasons.append("B/Hデータあり")
        if len(b_counts) >= max(5, min(rider_count or 7, 7)) or len(h_counts) >= max(5, min(rider_count or 7, 7)):
            confidence += 3
            reasons.append("B/H取得数良好")

        bh_delta = int(bh_strategy.get("score_delta", 0))
        confidence += bh_delta

        for r in bh_strategy.get("reasons", []):
            if r not in reasons:
                reasons.append(r)
        for w in bh_strategy.get("warnings", []):
            if w not in warnings:
                warnings.append(w)

        if max_b_count >= 10 or max_h_count >= 10:
            reasons.append("主導権候補が強い")
        elif max_b_count >= 6 or max_h_count >= 6:
            reasons.append("主導権候補あり")
        elif max_b_count >= 3 or max_h_count >= 3:
            reasons.append("自力候補あり")
    else:
        warnings.append("B/Hデータ未取得")

    if main_line_size >= 4:
        confidence += 10
        reasons.append("主導ラインあり")
    elif main_line_size == 3:
        confidence += 6
        reasons.append("軸候補が作りやすい")
    elif main_line_size == 2:
        confidence -= 10
        warnings.append("最大ラインが短く展開不安")
    else:
        confidence -= 16
        warnings.append("ライン情報が弱い")

    # ライン数は重く見すぎない。単騎過多だけ軽く減点。
    if solo_count >= 3:
        confidence -= 12
        warnings.append("単騎が多く展開が難しい")
    elif solo_count == 2:
        confidence -= 8
        warnings.append("単騎2名")

    if line_count >= 4:
        confidence -= 4
        warnings.append("ラインが分散")

    if rider_count and rider_count < 7:
        confidence -= 12
        warnings.append("出走情報不足の可能性")

    # レースタイプ
    if score_gap >= 3.0 and main_line_size >= 2:
        style = "的中率重視向き"
    elif (score_gap < 2.0 and len(scores) >= 7) or max_b_count >= 8 or max_h_count >= 8:
        style = "回収率重視向き"
    else:
        style = "バランス型"

    confidence = int(max(0, min(100, confidence)))

    if confidence >= 76:
        judge = "🔥 勝負"
    elif confidence >= 62:
        judge = "○ 候補"
    elif confidence >= 52:
        judge = "△ 軽め"
    else:
        judge = "見送り"

    if not reasons:
        reasons.append("強い買い材料が少なめ")
    if not warnings:
        warnings.append("大きな不安材料なし")

    return RaceScore(
        rank=0,
        race_name=race_name,
        race_url=final_url,
        judge=judge,
        style=style,
        confidence=confidence,
        riders=rider_count,
        avg_score=avg_score,
        max_score=max_score,
        score_gap=score_gap,
        line_count=line_count,
        solo_count=solo_count,
        main_line_size=main_line_size,
        max_b_count=max_b_count,
        max_h_count=max_h_count,
        b_data_count=b_data_count,
        h_data_count=h_data_count,
        advanced_data_count=advanced_data_count,
        reason=" / ".join(reasons[:5]),
        warnings=" / ".join(warnings[:5]),
    )


def analyze_urls(urls: List[str], max_races: int) -> Tuple[pd.DataFrame, List[str]]:
    rows = []
    errors = []
    for url in urls[:max_races]:
        try:
            rows.append(analyze_race(url))
        except Exception as e:
            msg = str(e)
            # 1R〜12R自動生成時、存在しないレースの404は自然にスキップする。
            if "404" not in msg:
                errors.append(f"{url}：{e}")

    rows = sorted(rows, key=lambda r: r.confidence, reverse=True)

    # ===== 相対評価AI =====
    # 「単体で良い」ではなく、
    # "今日の中で本当に買う価値があるレース" を上位だけ残す。
    total = len(rows)

    top_battle = 2 if total >= 8 else 1
    top_candidate = min(max(5, top_battle + 2), total)

    if rows:
        top_conf = rows[0].confidence
    else:
        top_conf = 0

    for idx, row in enumerate(rows):
        score = int(row.confidence)

        # 順位による減点
        if idx >= top_candidate:
            score -= 18
        elif idx >= top_battle:
            score -= 8

        # TOPとの差による減点
        gap = top_conf - score
        if gap >= 18:
            score -= 10
        elif gap >= 10:
            score -= 5

        score = max(0, min(96, int(score)))
        row.confidence = score

        # 新しい厳選判定
        if idx < top_battle and score >= 80:
            row.judge = "🔥 勝負"
        elif idx < top_candidate and score >= 68:
            row.judge = "○ 候補"
        elif score >= 58:
            row.judge = "△ 軽め"
        else:
            row.judge = "見送り"

    # 相対評価後に再ソート
    rows = sorted(rows, key=lambda r: r.confidence, reverse=True)

    for i, row in enumerate(rows, start=1):
        row.rank = i

    df = pd.DataFrame([asdict(r) for r in rows])
    return df, errors


def make_post_text(df: pd.DataFrame, top_n: int = 5) -> str:
    if df is None or df.empty:
        return "今日の勝負候補は見つかりませんでした。"
    lines = ["🔥 今日の勝負レース候補"]
    for _, r in df.head(top_n).iterrows():
        lines.append(f"{int(r['rank'])}位 {r['race_name']}｜勝負度 {int(r['confidence'])}%｜{r['judge']}｜{r['style']}")
        lines.append(f"理由：{r['reason']}")
    return "\n".join(lines)



# =========================================================
# 全国開催URL 自動取得
# =========================================================
def fetch_today_winticket_racecard_urls() -> List[str]:
    """
    WINTICKETから今日開催っぽいracecard URLを自動探索する。
    取れない場合もあるので、その時は従来通りURL手入力で使う。
    """
    candidate_pages = [
        "https://www.winticket.jp/keirin",
        "https://www.winticket.jp/keirin/racecard",
    ]

    found: List[str] = []

    for page_url in candidate_pages:
        try:
            html, final_url = fetch_html(page_url)
        except Exception:
            continue

        patterns = [
            r'https://www\.winticket\.jp/keirin/[a-z0-9_-]+/racecard/\d+/\d+/\d+',
            r'/keirin/[a-z0-9_-]+/racecard/\d+/\d+/\d+',
            r'https://www\.winticket\.jp/keirin/[a-z0-9_-]+/racecard/\d+',
            r'/keirin/[a-z0-9_-]+/racecard/\d+',
        ]

        for pat in patterns:
            for m in re.finditer(pat, html):
                u = m.group(0)
                if u.startswith("/"):
                    u = "https://www.winticket.jp" + u
                if re.search(r'/racecard/\d+$', u):
                    u = u.rstrip("/") + "/1/1"
                if u not in found:
                    found.append(u)

    unique = []
    seen = set()
    for u in found:
        m = re.search(r'/keirin/([^/]+)/racecard/(\d+)', u)
        if not m:
            continue
        key = f"{m.group(1)}-{m.group(2)}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(u)

    return unique


def apply_national_relative_judgment(df: pd.DataFrame) -> pd.DataFrame:
    """
    全国ランキング用の相対評価。
    全国TOPだけを勝負扱いにする。
    """
    if df is None or df.empty or "confidence" not in df.columns:
        return df

    out = df.copy().sort_values("confidence", ascending=False).reset_index(drop=True)
    total = len(out)

    top_battle = 3 if total >= 20 else (2 if total >= 8 else 1)
    top_candidate = min(10 if total >= 20 else 5, total)
    top_score = int(out.iloc[0]["confidence"]) if total else 0

    new_conf = []
    new_judge = []

    for idx, row in out.iterrows():
        score = int(row.get("confidence", 0))

        if idx >= top_candidate:
            score -= 22
        elif idx >= top_battle:
            score -= 10

        gap = top_score - score
        if gap >= 20:
            score -= 12
        elif gap >= 12:
            score -= 6

        score = max(0, min(96, int(score)))

        if idx < top_battle and score >= 82:
            judge = "🔥 勝負"
        elif idx < top_candidate and score >= 70:
            judge = "○ 候補"
        elif score >= 60:
            judge = "△ 軽め"
        else:
            judge = "見送り"

        new_conf.append(score)
        new_judge.append(judge)

    out["confidence"] = new_conf
    out["judge"] = new_judge
    out = out.sort_values("confidence", ascending=False).reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    return out




# =========================================================
# レース形状ログ保存
# =========================================================
def safe_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default


def safe_float2(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def save_race_selection_log(df: pd.DataFrame):
    """
    AIが見たレース形状をCSV保存する。
    将来の展開学習AI用。
    """
    if df is None or df.empty:
        return

    rows = []

    now = datetime.now()
    saved_at = now.strftime("%Y-%m-%d %H:%M:%S")

    for _, row in df.iterrows():
        race_name = str(row.get("race_name", ""))

        venue = ""
        race_no = ""

        m = re.search(r"(.+?)\s*(\d+)R", race_name)
        if m:
            venue = m.group(1).strip()
            race_no = m.group(2)

        b_high_count = 0
        h_high_count = 0

        try:
            if safe_int(row.get("max_b_count", 0)) >= 10:
                b_high_count = 1
        except Exception:
            pass

        try:
            if safe_int(row.get("max_h_count", 0)) >= 10:
                h_high_count = 1
        except Exception:
            pass

        log_row = {
            "saved_at": saved_at,
            "date": now.strftime("%Y-%m-%d"),
            "venue": venue,
            "race_no": race_no,
            "race_name": race_name,
            "rank": safe_int(row.get("rank", 0)),
            "judge": str(row.get("judge", "")),
            "confidence": safe_int(row.get("confidence", 0)),
            "race_style": str(row.get("race_style", "")),
            "riders": safe_int(row.get("riders", 0)),
            "score_gap": round(safe_float2(row.get("score_gap", 0.0)), 2),
            "line_count": safe_int(row.get("line_count", 0)),
            "solo_count": safe_int(row.get("solo_count", 0)),
            "main_line_size": safe_int(row.get("main_line_size", 0)),
            "max_b_count": safe_int(row.get("max_b_count", 0)),
            "max_h_count": safe_int(row.get("max_h_count", 0)),
            "b_data_count": safe_int(row.get("b_data_count", 0)),
            "h_data_count": safe_int(row.get("h_data_count", 0)),
            "advanced_data_count": safe_int(row.get("advanced_data_count", 0)),
            "b_high_count": b_high_count,
            "h_high_count": h_high_count,
            "reasons": str(row.get("reason", "")),
            "warnings": str(row.get("warnings", "")),
            "url": str(row.get("url", "")),
            # 将来の学習用
            "hit_result": "",
            "payout": "",
            "memo": "",
        }

        rows.append(log_row)

    if not rows:
        return

    log_df = pd.DataFrame(rows)

    path = Path(RACE_SELECTION_LOG_PATH)

    try:
        if path.exists():
            old = pd.read_csv(path)
            merged = pd.concat([old, log_df], ignore_index=True)

            # 同一レース重複除去
            dedup_cols = ["date", "race_name", "url"]
            merged = merged.drop_duplicates(subset=dedup_cols, keep="last")

            merged.to_csv(path, index=False, encoding="utf-8-sig")
        else:
            log_df.to_csv(path, index=False, encoding="utf-8-sig")

    except Exception:
        pass




# =========================================================
# AI指数＋印表 noteコピペ用
# =========================================================
def extract_player_blocks_for_mark_table(html: str) -> List[Dict[str, object]]:
    if not html:
        return []

    player_matches = list(re.finditer(r'"playerId"\s*:\s*"([^"]+)"', html))
    players: List[Dict[str, object]] = []

    for i, m in enumerate(player_matches):
        pid = m.group(1)
        start = m.start()
        prev_start = player_matches[i - 1].start() if i > 0 else 0
        next_start = player_matches[i + 1].start() if i + 1 < len(player_matches) else len(html)

        before = html[max(prev_start, start - 1800):start]
        after = html[start:min(next_start, start + 2600)]
        around = before + after

        if '"racePoint"' not in after and '"style"' not in after:
            continue

        race_point_m = re.search(r'"racePoint"\s*:\s*([0-9]+(?:\.[0-9]+)?)', after)
        style_m = re.search(r'"style"\s*:\s*"([^"]*)"', after)

        home_candidates = re.findall(r'"home"\s*:\s*(\d{1,3})', before)
        back_candidates = re.findall(r'"back"\s*:\s*(\d{1,3})', after)

        home = int(home_candidates[-1]) if home_candidates else 0
        back = int(back_candidates[0]) if back_candidates else 0

        name = ""
        for pat in [r'"name"\s*:\s*"([^"]{2,14})"', r'"playerName"\s*:\s*"([^"]{2,14})"', r'"fullName"\s*:\s*"([^"]{2,14})"']:
            nm = re.search(pat, around)
            if nm:
                cand = normalize_text(nm.group(1))
                if re.search(r"[一-龥ぁ-んァ-ヶ々]", cand):
                    name = cand
                    break

        players.append({
            "player_id": pid,
            "name": name,
            "race_point": safe_float(race_point_m.group(1), 0.0) if race_point_m else 0.0,
            "style": style_m.group(1) if style_m else "",
            "back": back,
            "home": home,
        })

    unique = []
    seen = set()
    for p in players:
        pid = str(p.get("player_id", ""))
        if pid in seen:
            continue
        seen.add(pid)
        unique.append(p)

    return unique[:9]


def parse_lineup_positions_for_mark(lineup: str) -> Dict[int, Dict[str, int]]:
    groups = parse_lineup_groups(lineup)
    info: Dict[int, Dict[str, int]] = {}
    for line_idx, group in enumerate(groups, start=1):
        size = len(group)
        for pos, car in enumerate(group, start=1):
            info[int(car)] = {
                "line_no": line_idx,
                "line_pos": pos,
                "line_size": size,
                "is_single": 1 if size == 1 else 0,
            }
    return info


def mark_from_rank_for_note(rank: int) -> str:
    return {1: "◎", 2: "○", 3: "▲", 4: "△", 5: "☆"}.get(rank, "")


def make_player_index_mark_table(url: str) -> Tuple[pd.DataFrame, str]:
    html, final_url = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")
    text = normalize_text(soup.get_text(" "))

    lineup = extract_lineup_text(text)
    line_info = parse_lineup_positions_for_mark(lineup)
    players = extract_player_blocks_for_mark_table(html)

    if not players:
        return pd.DataFrame(), lineup

    rows = []
    for idx, p in enumerate(players, start=1):
        car_no = idx
        li = line_info.get(car_no, {})

        race_point = safe_float(p.get("race_point", 0.0), 0.0)
        back = safe_int(p.get("back", 0), 0)
        home = safe_int(p.get("home", 0), 0)
        style = str(p.get("style", ""))

        line_size = safe_int(li.get("line_size", 0), 0)
        line_pos = safe_int(li.get("line_pos", 0), 0)
        is_single = safe_int(li.get("is_single", 0), 0)

        index = race_point

        if line_size >= 4:
            index += 5.0
        elif line_size == 3:
            index += 3.0
        elif line_size == 2:
            index += 1.0
        elif is_single:
            index -= 3.0

        if line_pos == 1:
            index += min(back, 14) * 0.35
            index += min(home, 14) * 0.25
        elif line_pos == 2:
            index += 3.0
        elif line_pos >= 3:
            index += 1.0

        if back >= 10:
            index += 3.0
        elif back >= 6:
            index += 1.5

        if home >= 10:
            index += 2.0
        elif home >= 6:
            index += 1.0

        if "逃" in style:
            index += 2.0
        elif "両" in style:
            index += 1.2
        elif "追" in style and line_pos == 2:
            index += 2.0

        if is_single and (back < 6 and home < 6):
            index -= 2.0

        index = max(0.0, min(120.0, round(index, 1)))

        comment_parts = []
        if line_size >= 3:
            comment_parts.append(f"{line_size}車ライン")
        if line_pos == 1 and (back >= 6 or home >= 6):
            comment_parts.append("主導権候補")
        if line_pos == 2:
            comment_parts.append("番手有利")
        if is_single:
            comment_parts.append("単騎")
        if back >= 10 or home >= 10:
            comment_parts.append("B/H強め")
        if not comment_parts:
            comment_parts.append("展開待ち")

        rows.append({
            "車番": car_no,
            "選手名": f"{car_no}番",
            "AI指数": index,
            "印": "",
            "競走得点": round(race_point, 2),
            "脚質": style,
            "B": back,
            "H": home,
            "ライン": line_size,
            "ライン順": line_pos,
            "単騎": "Yes" if is_single else "",
            "AIコメント": " / ".join(comment_parts[:3]),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df, lineup

    df = df.sort_values("AI指数", ascending=False).reset_index(drop=True)
    # note表示では%やバーではなく、1位から始まる順位型の数字にする
    df["指数順位"] = [i + 1 for i in range(len(df))]
    df["印"] = [mark_from_rank_for_note(i + 1) for i in range(len(df))]

    for i, row in df.iterrows():
        if row["印"] == "" and (safe_int(row["B"], 0) >= 8 or safe_int(row["H"], 0) >= 8):
            df.at[i, "印"] = "穴"
            break

    return df, lineup


def make_copyable_index_mark_text(mark_df: pd.DataFrame, race_name: str = "", lineup: str = "") -> str:
    if mark_df is None or mark_df.empty:
        return "指数表を作成できませんでした。"

    lines = []

    if race_name:
        lines.append(f"## {race_name} 指数・印")
        lines.append("")

    if lineup:
        lines.append(f"想定並び：{lineup}")
        lines.append("")

    lines.append("※指数表・印とは連動していない場合もございます。")
    lines.append("")
    lines.append("| 印 | 車番 | 選手名 | 指数 | 得点 | 脚質 | B | H | AIコメント |")
    lines.append("|---|---:|---|---:|---:|---|---:|---:|---|")

    for _, r in mark_df.iterrows():
        lines.append(
            f"| {r.get('印','')} | {r.get('車番','')} | {r.get('選手名','')} | "
            f"{r.get('指数順位','')} | {r.get('競走得点','')} | {r.get('脚質','')} | "
            f"{r.get('B','')} | {r.get('H','')} | {r.get('AIコメント','')} |"
        )

    lines.append("")
    lines.append("※指数は、競走得点・ライン構成・B/H・脚質などをもとにAIが順位化した参考値です。")
    return "\n".join(lines)

st.title("🔥 レース選定AI")
st.caption(f"{APP_VERSION} / 買い目は出さず、狙うレースだけを選ぶ専用サイト")

with st.sidebar:
    st.header("設定")
    max_races = st.slider("最大チェックレース数", 1, 12, 12)
    min_display_conf = st.slider("一覧に表示する最低勝負度", 0, 100, 45)
    only_hot = st.checkbox("勝負・候補だけ表示", value=False)
    national_mode = st.checkbox("全国開催を自動取得して比較", value=False)
    max_national_venues = st.slider("全国取得する最大開催数", 1, 12, 6)
    show_mark_table = st.checkbox("指数・印表を表示", value=True)
    st.caption("レース形状ログ: race_selection_log.csv")
    st.divider()
    st.caption("レースURLを1つ入れると、同開催の1R〜12Rを自動チェックします。複数URLもOKです。")

url_text = st.text_area(
    "WINTICKETの開催URL、レースURL、または複数レースURL",
    height=110,
    placeholder="https://www.winticket.jp/keirin/ito/racecard/2026041837/3/1",
)

col_a, col_b = st.columns([1, 3])
with col_a:
    run = st.button("レース選定する", type="primary", use_container_width=True)
with col_b:
    st.info("買い目は今の完全版/Mobile版で出す前提。ここでは“どのレースを買うか”だけ判定します。")

if run:
    input_urls = clean_urls(url_text)

    all_urls = []
    base_urls = []

    with st.spinner("レースURLを準備中..."):
        if national_mode:
            base_urls = fetch_today_winticket_racecard_urls()[:max_national_venues]
            if not base_urls and input_urls:
                base_urls = input_urls
        else:
            base_urls = input_urls

        if not base_urls:
            st.error("URLを入力してください。全国自動取得も失敗しました。")
            st.stop()

        for u in base_urls:
            for ru in build_race_urls_from_one_url(u, max_race=max_races):
                if ru not in all_urls:
                    all_urls.append(ru)

    if national_mode:
        st.caption(f"全国比較モード: {len(base_urls)}開催 / チェック対象: {len(all_urls)}レース")
    else:
        st.caption(f"チェック対象: {len(all_urls)}レース")

    with st.spinner("AIがレースを選定中..."):
        df, errors = analyze_urls(all_urls, max_races=len(all_urls))

    if national_mode and df is not None and not df.empty:
        df = apply_national_relative_judgment(df)

    # AIが見たレース形状を保存
    try:
        save_race_selection_log(df)
    except Exception:
        pass

    if errors:
        with st.expander("取得できなかったレース"):
            for e in errors:
                st.write(e)

    if df.empty:
        st.warning("判定できるレースがありませんでした。URLを確認してください。")
        st.stop()

    show_df = df[df["confidence"] >= min_display_conf].copy()
    if only_hot:
        show_df = show_df[show_df["judge"].isin(["🔥 勝負", "○ 候補"])]

    st.subheader("🔥 全国勝負候補ランキング" if national_mode else "🔥 今日の勝負候補ランキング")

    if show_df.empty:
        st.warning("表示条件に合うレースがありません。最低勝負度を下げてください。")
    else:
        for _, r in show_df.iterrows():
            with st.container(border=True):
                st.markdown(f"### {int(r['rank'])}位　{r['race_name']}")
                st.progress(int(r['confidence']) / 100.0, text=f"勝負度 {int(r['confidence'])}%")
                st.write(f"**判定:** {r['judge']}  /  **タイプ:** {r['style']}")
                st.write(f"**買い材料:** {r['reason']}")
                st.caption(f"注意点: {r['warnings']}")
                st.caption(f"人数 {int(r['riders'])} / 得点差 {r['score_gap']} / ライン数 {int(r['line_count'])} / 単騎 {int(r['solo_count'])}")
                st.link_button("WINTICKETで開く", r['race_url'], use_container_width=True)


    if show_mark_table and not show_df.empty:
        st.subheader("🧠 指数・印表")
        st.caption("noteにそのままコピペしやすい形式で表示します。")

        race_options = [
            f"{int(r['rank'])}位 {r['race_name']}｜{r['judge']}｜{int(r['confidence'])}%"
            for _, r in show_df.head(10).iterrows()
        ]

        selected_label = st.selectbox("指数表を作るレース", race_options)
        selected_idx = race_options.index(selected_label)
        selected_row = show_df.head(10).iloc[selected_idx]

        try:
            mark_df, lineup_for_mark = make_player_index_mark_table(str(selected_row["race_url"]))
            if mark_df.empty:
                st.warning("指数表を作成できませんでした。")
            else:
                display_mark_df = mark_df.copy()
                # 表示は%やバーではなく、1位から始まる「指数順位」をメインにする
                display_cols = [
                    "印", "車番", "選手名", "指数順位", "競走得点", "脚質",
                    "B", "H", "ライン", "ライン順", "単騎", "AIコメント"
                ]
                display_cols = [c for c in display_cols if c in display_mark_df.columns]
                st.dataframe(
                    display_mark_df[display_cols],
                    use_container_width=True,
                    hide_index=True,
                )

                copy_text = make_copyable_index_mark_text(
                    mark_df,
                    race_name=str(selected_row["race_name"]),
                    lineup=lineup_for_mark,
                )

                st.text_area(
                    "noteコピペ用：指数表・印",
                    value=copy_text,
                    height=360,
                )

        except Exception as e:
            st.warning(f"指数表の作成に失敗しました: {e}")


    c1, c2, c3, c4 = st.columns(4)
    c1.metric("勝負", int((df["judge"] == "🔥 勝負").sum()))
    c2.metric("候補", int((df["judge"] == "○ 候補").sum()))
    c3.metric("軽め", int((df["judge"] == "△ 軽め").sum()))
    c4.metric("見送り", int((df["judge"] == "見送り").sum()))

    st.subheader("投稿用メモ")
    st.code(make_post_text(show_df, top_n=5), language="text")

else:
    st.subheader("このアプリの役割")
    st.write("今日の開催から、AIが“買う価値のありそうなレース”だけをランキングします。")
    st.write("買い目は出さないので、選ばれたレースURLを今の完全版/Mobile版に入れて予想を出してください。")

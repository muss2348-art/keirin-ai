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

APP_VERSION = "race-selector-mobile v1.0"

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


def extract_lineup_text(text: str) -> str:
    text = normalize_text(text)
    # 例: 1-3-5 / 2-7 / 4 / 6 のような並びを拾う
    patterns = re.findall(r"(?:[1-9](?:-[1-9]){0,3})(?:\s*/\s*(?:[1-9](?:-[1-9]){0,3})){1,5}", text)
    if not patterns:
        patterns = re.findall(r"(?:[1-9](?:-[1-9]){1,3})(?:\s+|　)+(?:[1-9](?:-[1-9]){1,3})", text)
    if patterns:
        return max(patterns, key=len)
    return ""


def parse_lineup_groups(lineup: str) -> List[List[int]]:
    lineup = normalize_text(lineup).replace(" ", "")
    if not lineup:
        return []
    groups = []
    for part in re.split(r"[/|,、 ]+", lineup):
        nums = [int(x) for x in re.findall(r"[1-9]", part)]
        if nums:
            groups.append(nums)
    return groups


def estimate_rider_count(text: str, scores: List[float]) -> int:
    nums = set(int(x) for x in re.findall(r"(?<!\d)([1-9])番", text))
    if nums:
        return max(nums)
    return min(max(len(scores), 7), 9) if scores else 0


def analyze_race(url: str) -> RaceScore:
    html, final_url = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")
    title = normalize_text(soup.title.get_text(" ") if soup.title else "")
    text = normalize_text(soup.get_text(" "))

    venue = venue_from_url(final_url)
    rno = race_number_from_url(final_url)
    race_name = f"{venue} {rno}R" if venue and rno else (title[:40] or final_url)

    scores = extract_scores(text)
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

    # AI自信度の中心は「得点の強弱」「データ取得量」「ラインのまとまり」。ライン数は減点材料止まり。
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
    elif score_gap <= 1.0 and score_gap > 0:
        confidence -= 8
        warnings.append("得点差が小さく混戦")

    if top_gap >= 2.0:
        confidence += 6
        reasons.append("軸候補が作りやすい")

    if main_line_size >= 3:
        confidence += 8
        reasons.append("主導ラインあり")
    elif main_line_size == 2:
        confidence += 3
        reasons.append("ライン構成は最低限あり")
    else:
        confidence -= 4
        warnings.append("ライン情報が弱い")

    # ライン数は重く見すぎない。単騎過多だけ軽く減点。
    if solo_count >= 3:
        confidence -= 7
        warnings.append("単騎が多め")
    elif solo_count == 2:
        confidence -= 3
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
    elif score_gap < 2.0 and len(scores) >= 7:
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
        reason=" / ".join(reasons[:4]),
        warnings=" / ".join(warnings[:4]),
    )


def analyze_urls(urls: List[str], max_races: int) -> Tuple[pd.DataFrame, List[str]]:
    rows = []
    errors = []
    for url in urls[:max_races]:
        try:
            rows.append(analyze_race(url))
        except Exception as e:
            errors.append(f"{url}：{e}")

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


st.title("🔥 レース選定AI")
st.caption(f"{APP_VERSION} / 買い目は出さず、狙うレースだけを選ぶ専用サイト")

with st.sidebar:
    st.header("設定")
    max_races = st.slider("最大チェックレース数", 1, 12, 12)
    min_display_conf = st.slider("一覧に表示する最低勝負度", 0, 100, 45)
    only_hot = st.checkbox("勝負・候補だけ表示", value=False)
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
    if not input_urls:
        st.error("URLを入力してください。")
        st.stop()

    all_urls = []
    with st.spinner("レースURLを準備中..."):
        for u in input_urls:
            for ru in build_race_urls_from_one_url(u, max_race=max_races):
                if ru not in all_urls:
                    all_urls.append(ru)

    st.caption(f"チェック対象: {len(all_urls)}レース")

    with st.spinner("AIがレースを選定中..."):
        df, errors = analyze_urls(all_urls, max_races=max_races)

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

    st.subheader("🔥 今日の勝負候補ランキング")

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

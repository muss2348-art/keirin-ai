import itertools
import re

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

st.set_page_config(page_title="競輪AI Mobile", layout="centered")

st.title("🚴 競輪AI Mobile")
st.caption("軽量版 / URL読込 / 7車9車対応 / モード判定 / アツバリ / チャッピー買い目")

# ------------------------
# 初期値
# ------------------------

default_lines_7 = "147\n25\n3\n6"
default_lines_9 = "123\n456\n78\n9"

if "mobile_lines_text" not in st.session_state:
    st.session_state.mobile_lines_text = default_lines_7
if "mobile_odds_text" not in st.session_state:
    st.session_state.mobile_odds_text = ""
if "mobile_race_url_text" not in st.session_state:
    st.session_state.mobile_race_url_text = ""
if "mobile_race_size" not in st.session_state:
    st.session_state.mobile_race_size = "7車"

# ------------------------
# URL読込
# ------------------------

def normalize_winticket_race_urls(url: str) -> tuple[str, str]:
    url = url.strip()
    m = re.search(
        r"https?://www\.winticket\.jp/keirin/([^/]+)/(racecard|odds)/([^/]+)/([^/]+)/([^/?#]+)",
        url,
    )
    if not m:
        raise ValueError("WINTICKETのレースURL形式を読み取れませんでした")

    stadium = m.group(1)
    cup_id = m.group(3)
    day_no = m.group(4)
    race_no = m.group(5)

    racecard_url = f"https://www.winticket.jp/keirin/{stadium}/racecard/{cup_id}/{day_no}/{race_no}"
    odds_url = f"https://www.winticket.jp/keirin/{stadium}/odds/{cup_id}/{day_no}/{race_no}"
    return racecard_url, odds_url


def fetch_html(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    return r.text


def extract_lines_from_racecard_html(html: str) -> str:
    text = BeautifulSoup(html, "lxml").get_text("\n")
    lines = text.splitlines()

    try:
        idx = next(i for i, line in enumerate(lines) if "並び予想" in line)
    except StopIteration:
        raise ValueError("並び予想が見つかりませんでした")

    buf = []
    current = []

    for raw in lines[idx + 1: idx + 80]:
        s = raw.strip()
        if not s:
            continue

        if s == "区切り":
            if current:
                buf.append("".join(current))
                current = []
            continue

        if re.fullmatch(r"[1-9]", s):
            current.append(s)
            continue

        if "基本情報" in s or "オッズ一覧" in s or "投票" in s:
            break

    if current:
        buf.append("".join(current))

    buf = [x for x in buf if x.isdigit()]
    if not buf:
        raise ValueError("並びを抽出できませんでした")

    return "\n".join(buf)


def extract_odds_from_odds_html(html: str, max_items: int = 100) -> str:
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text("\n")
    script_text = "\n".join(s.get_text(" ", strip=True) for s in soup.find_all("script"))

    candidates = []

    pattern1 = re.findall(
        r"([1-9])\s*-\s*([1-9])\s*-\s*([1-9])\s+(\d+(?:\.\d+)?)",
        text
    )
    for a, b, c, odds in pattern1:
        candidates.append((f"{a}-{b}-{c}", float(odds)))

    pattern2 = re.findall(
        r"([1-9]-[1-9]-[1-9])\s+(\d+(?:\.\d+)?)",
        text
    )
    for ticket, odds in pattern2:
        candidates.append((ticket, float(odds)))

    pattern3 = re.findall(
        r"([1-9])\s*-\s*([1-9])\s*-\s*([1-9]).{0,80}?(\d+(?:\.\d+)?)",
        script_text
    )
    for a, b, c, odds in pattern3:
        candidates.append((f"{a}-{b}-{c}", float(odds)))

    odds_map = {}
    for ticket, odds in candidates:
        if ticket not in odds_map:
            odds_map[ticket] = odds
        else:
            odds_map[ticket] = min(odds_map[ticket], odds)

    cleaned = {}
    for ticket, odds in odds_map.items():
        if 1.0 <= odds <= 999999:
            cleaned[ticket] = odds

    if not cleaned:
        return ""

    items = sorted(cleaned.items(), key=lambda x: x[1])[:max_items]
    return "\n".join(f"{ticket} {odds}" for ticket, odds in items)


def load_winticket_race_from_url(url: str) -> tuple[str, str]:
    racecard_url, odds_url = normalize_winticket_race_urls(url)
    racecard_html = fetch_html(racecard_url)
    odds_html = fetch_html(odds_url)

    lines_text = extract_lines_from_racecard_html(racecard_html)
    odds_text = extract_odds_from_odds_html(odds_html)

    return lines_text, odds_text

# ------------------------
# 基本処理
# ------------------------

def parse_lines(text: str) -> list[str]:
    lines = []
    for raw in text.splitlines():
        raw = raw.strip().replace("-", "").replace(" ", "").replace("　", "")
        if raw:
            lines.append(raw)
    return lines


def parse_odds(text: str) -> dict:
    odds_dict = {}
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue

        parts = raw.split()
        if len(parts) >= 2:
            ticket = parts[0]
            try:
                odds = float(parts[-1])
                odds_dict[ticket] = odds
            except ValueError:
                pass

    return odds_dict

# ------------------------
# モード判定
# ------------------------

def judge_mode_fit(lines: list[str]) -> tuple[str, int, str]:
    line_list = [line for line in lines if len(line) >= 2]
    solo_list = [line for line in lines if len(line) == 1]

    line3_count = sum(1 for x in line_list if len(x) == 3)
    line2_count = sum(1 for x in line_list if len(x) == 2)
    solo_count = len(solo_list)

    normal_score = 50
    kosen_score = 50
    ana_score = 50

    normal_score += line3_count * 18
    normal_score += max(0, 2 - solo_count) * 8
    normal_score -= line2_count * 4
    normal_score -= solo_count * 8

    kosen_score += line2_count * 10
    kosen_score += solo_count * 10
    kosen_score -= line3_count * 6

    ana_score += solo_count * 12
    ana_score += line2_count * 7
    ana_score += max(0, solo_count - 1) * 4
    ana_score -= line3_count * 4

    scores = {
        "通常モード": max(0, min(100, normal_score)),
        "混戦モード": max(0, min(100, kosen_score)),
        "穴モード": max(0, min(100, ana_score)),
    }

    best_mode = max(scores, key=scores.get)
    best_score = scores[best_mode]

    if best_mode == "通常モード":
        comment = "3車ライン主体・単騎少なめで本線寄り"
    elif best_mode == "混戦モード":
        comment = "2車ラインや単騎があり混戦寄り"
    else:
        comment = "単騎やハサミが出やすく穴寄り"

    return best_mode, best_score, comment

# ------------------------
# データフレーム作成
# ------------------------

def build_dataframe(lines: list[str], mode: str) -> pd.DataFrame:
    data = []
    line_no = 1

    for line in lines:
        for pos, r in enumerate(line):
            r = int(r)

            if mode == "通常モード":
                score = 90 - pos * 2
            elif mode == "混戦モード":
                score = 92 - pos * 3
            else:
                score = 91 - pos * 2.5

            data.append({
                "車番": r,
                "得点": score,
                "ライン": line_no if len(line) > 1 else 0,
                "ライン順": pos + 1 if len(line) > 1 else 0,
                "ライン人数": len(line)
            })

        line_no += 1

    df = pd.DataFrame(data)
    df["評価"] = df["得点"]

    if mode == "通常モード":
        df.loc[df["ライン順"] == 1, "評価"] += 5
        df.loc[df["ライン順"] == 2, "評価"] += 6
        df.loc[df["ライン順"] == 3, "評価"] += 2
        df.loc[df["ライン人数"] == 3, "評価"] += 2

    elif mode == "混戦モード":
        df.loc[df["ライン順"] == 1, "評価"] += 5
        df.loc[df["ライン順"] == 2, "評価"] += 5
        df.loc[df["ライン順"] == 3, "評価"] += 1
        df.loc[df["ライン"] == 0, "評価"] += 2

    else:
        df.loc[df["ライン順"] == 1, "評価"] += 4
        df.loc[df["ライン順"] == 2, "評価"] += 4
        df.loc[df["ライン順"] == 3, "評価"] += 2
        df.loc[df["ライン"] == 0, "評価"] += 3
        df.loc[df["ライン人数"] == 3, "評価"] += 1

    return df

# ------------------------
# 予想ロジック
# ------------------------

def predict(df: pd.DataFrame, mode: str, race_size: str, odds_dict: dict) -> pd.DataFrame:
    if mode == "通常モード":
        top_count = 5 if race_size == "7車" else 6
    elif mode == "混戦モード":
        top_count = 7
    else:
        top_count = 6 if race_size == "7車" else 7

    top = df.sort_values("評価", ascending=False).head(top_count)
    riders_list = top["車番"].tolist()
    tickets = list(itertools.permutations(riders_list, 3))

    result = []
    line_strength = df.groupby("ライン")["評価"].sum().to_dict()

    sorted_lines = sorted(
        [(k, v) for k, v in line_strength.items() if k != 0],
        key=lambda x: x[1],
        reverse=True
    )
    line_gap = 0
    if len(sorted_lines) >= 2:
        line_gap = sorted_lines[0][1] - sorted_lines[1][1]

    for t in tickets:
        r1 = df[df["車番"] == t[0]].iloc[0]
        r2 = df[df["車番"] == t[1]].iloc[0]
        r3 = df[df["車番"] == t[2]].iloc[0]

        weights = [1.2, 0.95, 0.65] if mode != "穴モード" else [1.15, 1.0, 0.75]
        score = r1["評価"] * weights[0] + r2["評価"] * weights[1] + r3["評価"] * weights[2]

        # ライン強弱補正
        if r1["ライン"] != 0:
            score += line_strength.get(r1["ライン"], 0) * 0.05

        # 同ライン決着
        if r1["ライン"] == r2["ライン"] and r1["ライン"] != 0:
            score += 10
        if r2["ライン"] == r3["ライン"] and r2["ライン"] != 0:
            score += 5

        # 先頭→番手厚め
        if (
            r1["ライン"] == r2["ライン"]
            and r1["ライン"] != 0
            and r1["ライン順"] == 1
            and r2["ライン順"] == 2
        ):
            score += 6

        # 3車ラインの素直決着
        if (
            r1["ライン"] == r2["ライン"] == r3["ライン"]
            and r1["ライン"] != 0
            and r1["ライン順"] == 1
            and r2["ライン順"] == 2
            and r3["ライン順"] == 3
        ):
            score += 8

        # 単騎強化
        if r1["ライン"] == 0:
            score += 2
            if mode in ["混戦モード", "穴モード"]:
                score += 3

        if r2["ライン"] == 0:
            score += 1
            if mode in ["混戦モード", "穴モード"]:
                score += 2

        if r3["ライン"] == 0 and mode == "穴モード":
            score += 1

        # ハサミ強化 A-B-A
        if (
            r1["ライン"] != 0
            and r2["ライン"] != 0
            and r3["ライン"] != 0
            and r1["ライン"] == r3["ライン"]
            and r1["ライン"] != r2["ライン"]
        ):
            score += 7
            if mode == "混戦モード":
                score += 3

        # ライン崩れ検知
        if line_gap < 8:
            if r1["ライン"] != r2["ライン"]:
                score += 3
            if r1["ライン"] != 0 and r2["ライン"] != 0 and r1["ライン"] != r2["ライン"]:
                score += 2

        if line_gap >= 12:
            if r1["ライン"] == r2["ライン"] and r1["ライン"] != 0:
                score += 4
            if r1["ライン"] != 0 and r1["ライン順"] == 1:
                score += 2

        # モード補正
        if mode == "通常モード":
            if r1["ライン順"] == 1:
                score += 3
        elif mode == "混戦モード":
            if r1["ライン"] != r2["ライン"]:
                score += 3
            if r3["ライン"] == 0:
                score += 1
        else:
            if r1["ライン"] != r2["ライン"]:
                score += 3
            if r1["ライン"] == 0:
                score += 2

        ticket = f"{t[0]}-{t[1]}-{t[2]}"
        odds = odds_dict.get(ticket, None)
        expected = score * odds if odds is not None else None

        result.append([ticket, score, odds, expected])

    result_df = pd.DataFrame(result, columns=["買い目", "AI評価", "オッズ", "期待値"])
    return result_df.sort_values("AI評価", ascending=False).reset_index(drop=True)

# ------------------------
# 補助
# ------------------------

def classify_tickets(result: pd.DataFrame) -> pd.DataFrame:
    ranked = result.copy()

    ai_rank = ranked["AI評価"].rank(ascending=False, method="min")
    if ranked["期待値"].notna().any():
        ev_rank = ranked["期待値"].rank(ascending=False, method="min")
    else:
        ev_rank = pd.Series([999] * len(ranked), index=ranked.index)

    labels = []
    for i in range(len(ranked)):
        a = ai_rank.iloc[i]
        e = ev_rank.iloc[i]

        if a <= 3 and e <= 5:
            labels.append("🔥 AI推奨")
        elif a <= 5:
            labels.append("🟢 本命")
        elif e <= 5:
            labels.append("💰 期待値高")
        elif a <= 12:
            labels.append("🟡 穴")
        else:
            labels.append("⚪ その他")

    ranked["ランク"] = labels
    return ranked


def build_final_tickets(ranked_result: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    picks = []

    for label in ["🔥 AI推奨", "🟢 本命", "💰 期待値高", "🟡 穴"]:
        subset = ranked_result[ranked_result["ランク"] == label]
        for _, row in subset.iterrows():
            if row["買い目"] not in [p["買い目"] for p in picks]:
                picks.append(row)

    final_df = pd.DataFrame(picks)
    if len(final_df) > limit:
        final_df = final_df.head(limit)

    return final_df


def get_atsubari_tickets(final_df: pd.DataFrame, race_size: str) -> pd.DataFrame:
    if len(final_df) == 0:
        return final_df.copy()
    pick_n = 2 if race_size == "7車" else 3
    return final_df.sort_values(["ランク", "AI評価"], ascending=[True, False]).head(pick_n).reset_index(drop=True)


def get_chappy_tickets(ranked_result: pd.DataFrame) -> pd.DataFrame:
    if len(ranked_result) == 0:
        return ranked_result.copy()

    picks = []
    for label in ["🔥 AI推奨", "🟢 本命", "🟡 穴", "💰 期待値高"]:
        subset = ranked_result[ranked_result["ランク"] == label].head(1)
        for _, row in subset.iterrows():
            if row["買い目"] not in [p["買い目"] for p in picks]:
                picks.append(row)

    if not picks:
        return ranked_result.head(3).copy()

    return pd.DataFrame(picks).head(3)


def tickets_to_text(df: pd.DataFrame) -> str:
    if len(df) == 0:
        return ""
    return "\n".join(df["買い目"].tolist())


def show_rank_cards(ranked_result: pd.DataFrame):
    st.markdown("## 🃏 買い目カード")

    ai_push = ranked_result[ranked_result["ランク"] == "🔥 AI推奨"].head(2)
    honmei = ranked_result[ranked_result["ランク"] == "🟢 本命"].head(2)
    ana = ranked_result[ranked_result["ランク"] == "🟡 穴"].head(2)

    for title, df_part in [
        ("🔥 AI推奨", ai_push),
        ("🟢 本命", honmei),
        ("🟡 穴", ana),
    ]:
        st.markdown(f"### {title}")
        if len(df_part) > 0:
            for _, row in df_part.iterrows():
                st.write(f"{row['買い目']}  | AI:{row['AI評価']:.1f}")
        else:
            st.write("該当なし")

# ------------------------
# UI
# ------------------------

race_size = st.radio(
    "車立て",
    ["7車", "9車"],
    horizontal=True,
    key="mobile_race_size"
)

mode = st.radio(
    "モード",
    ["通常モード", "混戦モード", "穴モード"],
    horizontal=True,
    key="mobile_mode"
)

st.text_input(
    "WINTICKETレースURL",
    key="mobile_race_url_text",
    placeholder="https://www.winticket.jp/keirin/..."
)

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("URL読込", use_container_width=True):
        try:
            loaded_lines, loaded_odds = load_winticket_race_from_url(st.session_state.mobile_race_url_text)
            st.session_state.mobile_lines_text = loaded_lines
            if loaded_odds:
                st.session_state.mobile_odds_text = loaded_odds
            st.success("読み込みOK")
        except Exception as e:
            st.error(f"失敗: {e}")

with col2:
    if st.button("7車初期化", use_container_width=True):
        st.session_state.mobile_lines_text = default_lines_7
        st.session_state.mobile_race_size = "7車"

with col3:
    if st.button("9車初期化", use_container_width=True):
        st.session_state.mobile_lines_text = default_lines_9
        st.session_state.mobile_race_size = "9車"

st.text_area("並び", key="mobile_lines_text", height=120)

run_button = st.button("予想する", use_container_width=True)

if run_button:
    lines = parse_lines(st.session_state.mobile_lines_text)
    odds_dict = parse_odds(st.session_state.mobile_odds_text)

    if not lines:
        st.error("並びを入力してください")
    else:
        fit_mode, fit_score, fit_comment = judge_mode_fit(lines)

        st.markdown("## 🧭 モード判定")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("推奨モード", fit_mode)
        with m2:
            st.metric("相性点", fit_score)
        with m3:
            st.metric("現在のモード", mode)
        st.caption(fit_comment)

        df = build_dataframe(lines, mode)
        result = predict(df, mode, race_size, odds_dict)
        ranked_result = classify_tickets(result)
        final_df = build_final_tickets(ranked_result, limit=5)
        atsubari_df = get_atsubari_tickets(final_df, race_size)
        chappy_df = get_chappy_tickets(ranked_result)

        st.markdown("## ✅ 最終買い目候補")
        st.dataframe(
            final_df[["買い目", "ランク", "AI評価", "オッズ", "期待値"]],
            use_container_width=True,
            hide_index=True
        )

        st.text_area(
            "最終買い目 コピー用",
            value=tickets_to_text(final_df),
            height=110
        )

        st.markdown("## 💥 アツバリ買い目")
        st.dataframe(
            atsubari_df[["買い目", "ランク", "AI評価", "オッズ", "期待値"]],
            use_container_width=True,
            hide_index=True
        )

        st.text_area(
            "アツバリ コピー用",
            value=tickets_to_text(atsubari_df),
            height=80
        )

        st.markdown("## 😎 チャッピー買い目")
        st.dataframe(
            chappy_df[["買い目", "ランク", "AI評価", "オッズ", "期待値"]],
            use_container_width=True,
            hide_index=True
        )

        st.text_area(
            "チャッピー買い目 コピー用",
            value=tickets_to_text(chappy_df),
            height=90
        )

        show_rank_cards(ranked_result)

        st.markdown("## 🎯 買い目ランク")
        st.dataframe(
            ranked_result[["買い目", "ランク", "AI評価", "オッズ", "期待値"]].head(12),
            use_container_width=True,
            hide_index=True
        )

else:
    st.info("URLを入れて『URL読込』→『予想する』")

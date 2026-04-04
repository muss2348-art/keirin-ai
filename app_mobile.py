import re
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

# =====================================
# selenium
# =====================================
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait
    SELENIUM_AVAILABLE = True
except Exception:
    SELENIUM_AVAILABLE = False


# =====================================
# 基本設定
# =====================================
st.set_page_config(page_title="競輪AI mobile 完全版", layout="centered")

BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = BASE_DIR / "log.csv"
RESULT_LOG_PATH = BASE_DIR / "result_log.csv"
DEBUG_POS_PATH = BASE_DIR / "winticket_debug_positions.csv"
DEBUG_TEXT_PATH = BASE_DIR / "winticket_debug_text.txt"

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 0.8rem;
        padding-bottom: 4rem;
        max-width: 760px;
    }
    .stButton > button {
        width: 100%;
        min-height: 48px;
        border-radius: 12px;
        font-weight: 700;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================
# CSV 初期化
# =====================================
def ensure_csv_files():
    if not LOG_PATH.exists():
        pd.DataFrame(columns=[
            "ID",
            "保存日時",
            "レース名",
            "レースURL",
            "券種",
            "モード",
            "モード判定理由",
            "買い目点数",
            "1点金額",
            "合計金額",
            "買い目一覧",
            "メモ",
            "並び",
            "AIサマリー",
        ]).to_csv(LOG_PATH, index=False, encoding="utf-8-sig")

    if not RESULT_LOG_PATH.exists():
        pd.DataFrame(columns=[
            "予想ID",
            "登録日時",
            "レース名",
            "着順1",
            "着順2",
            "着順3",
            "的中",
            "払戻",
            "投資額",
            "収支",
        ]).to_csv(RESULT_LOG_PATH, index=False, encoding="utf-8-sig")


# =====================================
# セッション初期化
# =====================================
def init_session():
    defaults = {
        "screen": "予想",
        "race_name": "",
        "race_url": "",
        "car_count": "7車",
        "mode_setting": "自動判定",
        "mode": "通常モード",
        "selected_mode_reason": "初期状態",
        "display_count": 10,
        "bet_amount": 100,
        "line_text": "",
        "fetch_reason": "",
        "final_bets": [],
        "memo_input": "",
        "selected_pred_id": "",
        "selected_race_name": "",
        "selected_investment": 0,
        "ai_summary": "",
        "line_score_table": pd.DataFrame(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# =====================================
# 既存CSVの列補完
# =====================================
def upgrade_existing_csv():
    ensure_csv_files()
    try:
        pred_df = pd.read_csv(LOG_PATH, encoding="utf-8-sig")
        changed = False
        for col in ["モード判定理由", "並び", "AIサマリー"]:
            if col not in pred_df.columns:
                pred_df[col] = ""
                changed = True
        if changed:
            pred_df.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")
    except Exception:
        pass


ensure_csv_files()
init_session()
upgrade_existing_csv()


# =====================================
# 共通関数
# =====================================
def parse_line_text(line_text: str):
    lines = []
    single_count = 0

    if not line_text.strip():
        return lines, single_count

    normalized = (
        line_text
        .replace("　", " ")
        .replace("-", " ")
        .replace("/", "|")
        .replace("
", "|")
    )
    groups = [g.strip() for g in normalized.split("|") if g.strip()]

    for group in groups:
        nums = [x for x in group.split() if x.isdigit()]
        if nums:
            lines.append(nums)
            if len(nums) == 1:
                single_count += 1

    return lines, single_count


def auto_detect_mode(line_text: str, car_count: str):
    lines, single_count = parse_line_text(line_text)

    if not lines:
        return "通常モード", "並び未入力のため通常モードに設定"

    total_lines = len(lines)
    max_line_len = max(len(line) for line in lines) if lines else 0
    two_line_count = sum(1 for line in lines if len(line) == 2)
    three_plus_count = sum(1 for line in lines if len(line) >= 3)

    reasons = [
        f"ライン数:{total_lines}",
        f"単騎数:{single_count}",
        f"最長ライン:{max_line_len}車",
        f"2車ライン数:{two_line_count}",
        f"3車以上ライン数:{three_plus_count}",
    ]

    if single_count >= 3:
        return "穴モード", " / ".join(reasons) + " → 単騎が多いため穴モード"
    if total_lines >= 4 and max_line_len <= 2:
        return "穴モード", " / ".join(reasons) + " → 細切れ戦で荒れやすいため穴モード"
    if single_count >= 2:
        return "混戦モード", " / ".join(reasons) + " → 単騎が複数いて混戦寄り"
    if two_line_count >= 2 and three_plus_count == 0:
        return "混戦モード", " / ".join(reasons) + " → 2車ライン中心で混戦モード"
    if car_count == "9車" and total_lines >= 4:
        return "混戦モード", " / ".join(reasons) + " → 9車でライン数が多く混戦モード"

    return "通常モード", " / ".join(reasons) + " → 主力ラインがあり通常モード"


def analyze_lines(line_text: str) -> tuple[pd.DataFrame, str]:
    lines, _ = parse_line_text(line_text)
    if not lines:
        return pd.DataFrame(), "並び未入力"

    rows = []
    for idx, line in enumerate(lines, start=1):
        line_len = len(line)
        for pos, num in enumerate(line, start=1):
            role = "先頭"
            if line_len == 1:
                role = "単騎"
            elif pos == 2:
                role = "番手"
            elif pos >= 3:
                role = "3番手以降"

            score = 50
            if line_len >= 3:
                score += 18
            elif line_len == 2:
                score += 8
            else:
                score -= 5

            if role == "番手":
                score += 18
            elif role == "先頭":
                score += 10
            elif role == "単騎":
                score += 12
            else:
                score += 4

            rows.append({
                "ライン": idx,
                "車番": str(num),
                "位置": pos,
                "役割": role,
                "ライン人数": line_len,
                "ラインAI": score,
            })

    df = pd.DataFrame(rows).sort_values(["ライン", "位置"]).reset_index(drop=True)

    strong_lines = df.groupby("ライン")["ラインAI"].sum().sort_values(ascending=False)
    best_line = int(strong_lines.index[0]) if not strong_lines.empty else 1
    best_line_nums = "-".join(lines[best_line - 1]) if lines else ""

    banker_candidates = df[df["役割"] == "番手"].sort_values("ラインAI", ascending=False)
    banker_text = banker_candidates.iloc[0]["車番"] if not banker_candidates.empty else "なし"

    single_df = df[df["役割"] == "単騎"].sort_values("ラインAI", ascending=False)
    single_text = ", ".join(single_df["車番"].astype(str).tolist()) if not single_df.empty else "なし"

    summary = f"主力ライン候補:{best_line}ライン({best_line_nums}) / 番手注目:{banker_text} / 単騎穴候補:{single_text}"
    return df, summary


def pick_key_numbers(line_text: str):
    df, _ = analyze_lines(line_text)
    if df.empty:
        return [], [], []

    heads = df[df["役割"].isin(["先頭", "単騎"])].sort_values("ラインAI", ascending=False)
    seconds = df[df["役割"].isin(["番手", "単騎"])].sort_values("ラインAI", ascending=False)
    thirds = df.sort_values("ラインAI", ascending=False)

    head_nums = heads["車番"].astype(str).tolist()
    second_nums = seconds["車番"].astype(str).tolist()
    third_nums = thirds["車番"].astype(str).tolist()
    return head_nums, second_nums, third_nums


def estimate_odds_from_structure(combo: str, line_text: str, mode_name: str) -> float:
    lines, _ = parse_line_text(line_text)
    if not lines:
        return 20.0

    nums = combo.split("-")
    line_map = {}
    pos_map = {}
    for line_idx, line in enumerate(lines, start=1):
        for pos_idx, num in enumerate(line, start=1):
            line_map[str(num)] = line_idx
            pos_map[str(num)] = pos_idx

    same_line_pairs = 0
    for a, b in [(nums[0], nums[1]), (nums[1], nums[2]), (nums[0], nums[2])]:
        if line_map.get(a) == line_map.get(b):
            same_line_pairs += 1

    odds = 55.0
    odds -= same_line_pairs * 12

    for n in nums:
        if pos_map.get(n) == 2:
            odds -= 4
        if line_map.get(n):
            target_line = lines[line_map[n] - 1]
            if len(target_line) == 1:
                odds += 10

    if mode_name == "穴モード":
        odds += 20
    elif mode_name == "混戦モード":
        odds += 8
    else:
        odds -= 8

    return max(6.0, round(odds, 1))


def score_combo(combo: str, line_text: str, mode_name: str):
    head_nums, second_nums, third_nums = pick_key_numbers(line_text)
    a, b, c = combo.split("-")

    score = 0.0
    if a in head_nums[:3]:
        score += 34 - head_nums.index(a) * 5
    elif a in head_nums:
        score += 8

    if b in second_nums[:4]:
        score += 24 - second_nums.index(b) * 4
    elif b in second_nums:
        score += 6

    if c in third_nums[:6]:
        score += 18 - third_nums.index(c) * 2
    elif c in third_nums:
        score += 4

    lines, _ = parse_line_text(line_text)
    line_map = {}
    pos_map = {}
    for line_idx, line in enumerate(lines, start=1):
        for pos_idx, num in enumerate(line, start=1):
            line_map[str(num)] = line_idx
            pos_map[str(num)] = pos_idx

    if line_map.get(a) == line_map.get(b):
        score += 14
    if line_map.get(b) == line_map.get(c):
        score += 8
    if line_map.get(a) == line_map.get(c):
        score += 5

    if pos_map.get(b) == 2:
        score += 6
    if pos_map.get(a) == 1:
        score += 5
    if line_map.get(a):
        target_line = lines[line_map[a] - 1]
        if len(target_line) == 1:
            score += 7

    if mode_name == "通常モード" and pos_map.get(a) == 2:
        score += 2
    if mode_name == "穴モード" and line_map.get(a):
        target_line = lines[line_map[a] - 1]
        if len(target_line) == 1:
            score += 6

    odds = estimate_odds_from_structure(combo, line_text, mode_name)
    hit_rate = min(85.0, max(8.0, round(score * 0.9, 1)))
    value = round((hit_rate / 100.0) * odds, 2)
    return score, hit_rate, odds, value


def generate_bets(mode_name: str, display_count: int, line_text: str):
    head_nums, second_nums, third_nums = pick_key_numbers(line_text)
    rows = []

    if head_nums and second_nums and third_nums:
        candidate_set = set()
        for a in head_nums[:5]:
            for b in second_nums[:6]:
                for c in third_nums[:7]:
                    if len({a, b, c}) == 3:
                        candidate_set.add(f"{a}-{b}-{c}")

        for combo in candidate_set:
            score, hit_rate, odds, value = score_combo(combo, line_text, mode_name)
            rows.append({
                "買い目": combo,
                "AI評価": round(score, 1),
                "的中率": hit_rate,
                "想定オッズ": odds,
                "期待値": value,
            })
    else:
        fallback = ["1-2-3", "1-3-2", "2-1-3", "2-3-1", "3-1-2", "3-2-1"]
        for combo in fallback[:display_count]:
            score, hit_rate, odds, value = score_combo(combo, line_text, mode_name)
            rows.append({
                "買い目": combo,
                "AI評価": round(score, 1),
                "的中率": hit_rate,
                "想定オッズ": odds,
                "期待値": value,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if mode_name == "通常モード":
        df = df.sort_values(["AI評価", "期待値"], ascending=[False, False])
    elif mode_name == "混戦モード":
        df = df.sort_values(["期待値", "AI評価"], ascending=[False, False])
    else:
        df = df.sort_values(["想定オッズ", "期待値", "AI評価"], ascending=[False, False, False])

    return df.head(display_count).reset_index(drop=True)


def build_rank_label(row):
    if row["期待値"] >= 20:
        return "💰 期待値高"
    if row["AI評価"] >= 65:
        return "🔥 AI推奨"
    if row["的中率"] >= 40:
        return "🟢 本命"
    return "🟡 穴"


def save_prediction_log(race_name, race_url, predict_type, mode_name, mode_reason, buy_points, bet_amount, final_bets, memo=""):
    total_amount = buy_points * bet_amount
    bet_text = " / ".join(final_bets) if final_bets else ""
    save_id = datetime.now().strftime("%Y%m%d%H%M%S%f")

    row = pd.DataFrame([{
        "ID": save_id,
        "保存日時": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "レース名": race_name,
        "レースURL": race_url,
        "券種": predict_type,
        "モード": mode_name,
        "モード判定理由": mode_reason,
        "買い目点数": buy_points,
        "1点金額": bet_amount,
        "合計金額": total_amount,
        "買い目一覧": bet_text,
        "メモ": memo,
        "並び": st.session_state.get("line_text", ""),
        "AIサマリー": st.session_state.get("ai_summary", ""),
    }])

    old = pd.read_csv(LOG_PATH, encoding="utf-8-sig")
    pd.concat([old, row], ignore_index=True).to_csv(LOG_PATH, index=False, encoding="utf-8-sig")
    return save_id


def load_prediction_log():
    try:
        df = pd.read_csv(LOG_PATH, encoding="utf-8-sig")
        if df.empty:
            return df
        return df.sort_values("保存日時", ascending=False)
    except Exception as e:
        st.error(f"予想履歴の読込エラー: {e}")
        return pd.DataFrame()


def load_result_log():
    try:
        df = pd.read_csv(RESULT_LOG_PATH, encoding="utf-8-sig")
        if df.empty:
            return df
        return df.sort_values("登録日時", ascending=False)
    except Exception as e:
        st.error(f"結果履歴の読込エラー: {e}")
        return pd.DataFrame()


def save_result_log(pred_id, race_name, rank1, rank2, rank3, payout, investment, hit):
    row = pd.DataFrame([{
        "予想ID": pred_id,
        "登録日時": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "レース名": race_name,
        "着順1": rank1,
        "着順2": rank2,
        "着順3": rank3,
        "的中": "的中" if hit else "ハズレ",
        "払戻": payout,
        "投資額": investment,
        "収支": payout - investment,
    }])

    old = pd.read_csv(RESULT_LOG_PATH, encoding="utf-8-sig")
    pd.concat([old, row], ignore_index=True).to_csv(RESULT_LOG_PATH, index=False, encoding="utf-8-sig")


def calc_summary(pred_df, result_df):
    total_races = len(pred_df) if not pred_df.empty else 0
    total_investment = 0
    total_payout = 0
    hit_count = 0

    if not result_df.empty:
        total_investment = pd.to_numeric(result_df["投資額"], errors="coerce").fillna(0).sum()
        total_payout = pd.to_numeric(result_df["払戻"], errors="coerce").fillna(0).sum()
        hit_count = (result_df["的中"] == "的中").sum()

    hit_rate = (hit_count / len(result_df) * 100) if len(result_df) > 0 else 0
    recovery_rate = (total_payout / total_investment * 100) if total_investment > 0 else 0
    profit = total_payout - total_investment

    return {
        "総予想数": total_races,
        "結果登録数": len(result_df),
        "的中数": hit_count,
        "的中率": hit_rate,
        "投資額": total_investment,
        "払戻": total_payout,
        "回収率": recovery_rate,
        "収支": profit,
    }


# =====================================
# WINTICKET 並び取得
# =====================================
def fetch_line_from_winticket(url: str, car_count: str):
    if not SELENIUM_AVAILABLE:
        return "", "selenium未インストール"

    def looks_like_single_number(text: str) -> bool:
        return bool(re.fullmatch(r"[1-9]", text.strip()))

    def dedupe_candidates(items):
        seen = set()
        result = []
        for item in items:
            key = (item["num"], round(item["x"], 1), round(item["y"], 1))
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result

    def cluster_by_gap(sorted_items, gap_threshold=22):
        if not sorted_items:
            return []
        groups = [[sorted_items[0]]]
        for item in sorted_items[1:]:
            prev = groups[-1][-1]
            if (item["x"] - prev["x"]) <= gap_threshold:
                groups[-1].append(item)
            else:
                groups.append([item])
        return groups

    def build_line_from_row(row_items, need):
        groups = cluster_by_gap(sorted(row_items, key=lambda v: v["x"]), gap_threshold=22)
        parts = []
        used = set()
        total = 0
        for g in groups:
            nums = []
            for item in g:
                if item["num"] not in used:
                    nums.append(item["num"])
                    used.add(item["num"])
            if nums:
                parts.append(nums)
                total += len(nums)
            if total >= need:
                break
        if total < need:
            return ""

        trimmed = []
        count = 0
        for p in parts:
            remain = need - count
            if remain <= 0:
                break
            seg = p[:remain]
            if seg:
                trimmed.append(seg)
                count += len(seg)
        if count != need:
            return ""

        return " / ".join(" ".join(seg) for seg in trimmed)

    driver = None
    try:
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1400,2200")
        options.add_argument("--lang=ja-JP")

        driver = webdriver.Chrome(options=options)
        driver.get(url)

        WebDriverWait(driver, 20).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )
        time.sleep(5)

        body_text = driver.find_element(By.TAG_NAME, "body").text
        with open(DEBUG_TEXT_PATH, "w", encoding="utf-8") as f:
            f.write(body_text)

        elements = driver.find_elements(By.XPATH, "//*")
        candidates = []
        for el in elements:
            try:
                txt = el.text.strip()
                if not looks_like_single_number(txt):
                    continue
                rect = el.rect
                x = float(rect.get("x", 0))
                y = float(rect.get("y", 0))
                w = float(rect.get("width", 0))
                h = float(rect.get("height", 0))
                if w <= 0 or h <= 0:
                    continue
                if w > 80 or h > 80:
                    continue
                candidates.append({"num": txt, "x": x, "y": y, "w": w, "h": h})
            except Exception:
                pass

        candidates = dedupe_candidates(candidates)
        if candidates:
            pd.DataFrame(candidates).sort_values(["y", "x"]).to_csv(DEBUG_POS_PATH, index=False, encoding="utf-8-sig")

        need = 7 if car_count == "7車" else 9
        y_groups = {}
        for c in candidates:
            y_key = round(c["y"] / 8) * 8
            y_groups.setdefault(y_key, []).append(c)

        row_candidates = []
        for y_key, group in y_groups.items():
            group = sorted(group, key=lambda v: v["x"])
            unique_nums = []
            seen_nums = set()
            for g in group:
                if g["num"] not in seen_nums:
                    unique_nums.append(g)
                    seen_nums.add(g["num"])
            if len(unique_nums) < 3:
                continue
            x_span = max(v["x"] for v in unique_nums) - min(v["x"] for v in unique_nums)
            if x_span < 40:
                continue
            row_candidates.append((y_key, unique_nums))

        row_candidates = sorted(row_candidates, key=lambda t: t[0])
        for y_key, row in row_candidates[:8]:
            line_text = build_line_from_row(row, need)
            if line_text:
                return line_text, f"上位横行から抽出: {line_text} (y={y_key})"

        top_band = [c for c in candidates if 300 <= c["y"] <= 520]
        top_band = sorted(top_band, key=lambda v: (round(v["y"] / 8) * 8, v["x"]))
        top_y_groups = {}
        for c in top_band:
            y_key = round(c["y"] / 8) * 8
            top_y_groups.setdefault(y_key, []).append(c)

        for y_key in sorted(top_y_groups.keys()):
            row = top_y_groups[y_key]
            unique_row = []
            seen_nums = set()
            for r in sorted(row, key=lambda v: v["x"]):
                if r["num"] not in seen_nums:
                    unique_row.append(r)
                    seen_nums.add(r["num"])
            line_text = build_line_from_row(unique_row, need)
            if line_text:
                return line_text, f"上部帯から抽出: {line_text} (y={y_key})"

        return "", "並びを正しく組めませんでした"

    except Exception as e:
        return "", f"取得エラー: {e}"

    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass


# =====================================
# UI 部品
# =====================================
def show_prediction_cards(pred_df: pd.DataFrame):
    if pred_df.empty:
        st.info("まだ予想履歴はありません")
        return

    for _, row in pred_df.iterrows():
        with st.container(border=True):
            st.markdown(f"**{row['レース名']}**")
            st.caption(f"{row['保存日時']} / {row['モード']}")
            if isinstance(row.get("AIサマリー", ""), str) and row.get("AIサマリー", ""):
                st.write(row["AIサマリー"])
            st.write(f"買い目: {row['買い目一覧']}")
            st.write(f"合計: ¥{int(row['合計金額']):,}")
            if st.button("この予想で結果登録", key=f"select_{row['ID']}"):
                st.session_state.selected_pred_id = str(row["ID"])
                st.session_state.selected_race_name = row["レース名"]
                st.session_state.selected_investment = int(row["合計金額"])
                st.session_state.screen = "結果入力"
                st.rerun()


def show_result_cards(result_df: pd.DataFrame):
    if result_df.empty:
        st.info("まだ結果履歴はありません")
        return

    for _, row in result_df.iterrows():
        with st.container(border=True):
            st.markdown(f"**{row['レース名']}**")
            st.caption(row["登録日時"])
            st.write(f"着順: {row['着順1']}-{row['着順2']}-{row['着順3']}")
            st.write(f"判定: {row['的中']}")
            st.write(f"払戻: ¥{int(row['払戻']):,} / 収支: ¥{int(row['収支']):,}")


# =====================================
# 画面: 予想
# =====================================
def render_predict_screen():
    st.title("🚴 競輪AI mobile 完全版")

    st.session_state.race_name = st.text_input("レース名", value=st.session_state.race_name, placeholder="例: 宇都宮 4R")
    st.session_state.race_url = st.text_input("レースURL", value=st.session_state.race_url, placeholder="WINTICKETのURL")

    c1, c2 = st.columns(2)
    with c1:
        st.session_state.car_count = st.selectbox("車立て", ["7車", "9車"], index=["7車", "9車"].index(st.session_state.car_count))
    with c2:
        st.session_state.mode_setting = st.selectbox("モード", ["自動判定", "手動選択"], index=["自動判定", "手動選択"].index(st.session_state.mode_setting))

    if st.session_state.mode_setting == "手動選択":
        st.session_state.mode = st.selectbox("手動モード", ["通常モード", "混戦モード", "穴モード"], index=["通常モード", "混戦モード", "穴モード"].index(st.session_state.mode))
        st.session_state.selected_mode_reason = "手動選択"

    st.session_state.line_text = st.text_area("並び", value=st.session_state.line_text, height=110, placeholder="例: 4 5 7 / 2 / 1 3 / 6")

    auto_mode, auto_reason = auto_detect_mode(st.session_state.line_text, st.session_state.car_count)
    if st.session_state.mode_setting == "手動選択":
        final_mode = st.session_state.mode
        final_reason = "手動選択"
    else:
        final_mode = auto_mode
        final_reason = auto_reason
        st.session_state.mode = auto_mode
        st.session_state.selected_mode_reason = auto_reason

    line_df, ai_summary = analyze_lines(st.session_state.line_text)
    st.session_state.ai_summary = ai_summary
    st.session_state.line_score_table = line_df

    mc1, mc2 = st.columns(2)
    mc1.metric("現在モード", final_mode)
    mc2.metric("買い目点数", st.session_state.display_count)
    st.caption(final_reason)
    st.caption(f"ラインAI: {ai_summary}")

    if st.button("並び自動取得"):
        if not st.session_state.race_url.strip():
            st.warning("URLを入力してね")
        else:
            with st.spinner("WINTICKETから並び取得中..."):
                line_result, reason = fetch_line_from_winticket(st.session_state.race_url, st.session_state.car_count)
            st.session_state.fetch_reason = reason
            if line_result:
                st.session_state.line_text = line_result
                st.success(f"並び取得: {line_result}")
                st.rerun()
            else:
                st.error(reason)

    if st.session_state.fetch_reason:
        st.caption(f"取得ログ: {st.session_state.fetch_reason}")

    if not line_df.empty:
        with st.expander("ラインAI 詳細"):
            st.dataframe(line_df, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        display_options = [5, 10, 15, 20]
        current_display = st.session_state.display_count if st.session_state.display_count in display_options else 10
        st.session_state.display_count = st.selectbox("表示点数", display_options, index=display_options.index(current_display))
    with c4:
        st.session_state.bet_amount = st.number_input("1点金額", min_value=100, max_value=10000, step=100, value=st.session_state.bet_amount)

    if st.button("予想生成", type="primary"):
        if not st.session_state.race_name.strip():
            st.warning("レース名を入力してね")
        else:
            bets_df = generate_bets(final_mode, st.session_state.display_count, st.session_state.line_text)
            if bets_df.empty:
                st.warning("買い目を生成できませんでした")
            else:
                bets_df["買い目ランク"] = bets_df.apply(build_rank_label, axis=1)
                st.session_state.final_bets = bets_df.to_dict("records")
                st.session_state.mode = final_mode
                st.session_state.selected_mode_reason = final_reason
            st.rerun()

    if st.session_state.final_bets:
        st.subheader("買い目")
        total_amount = len(st.session_state.final_bets) * st.session_state.bet_amount
        st.caption(f"合計: ¥{total_amount:,}")

        for idx, row in enumerate(st.session_state.final_bets, start=1):
            with st.container(border=True):
                st.write(f"{idx}. {row['買い目']}  {row['買い目ランク']}")
                st.caption(f"AI評価 {row['AI評価']} / 的中率 {row['的中率']}% / 想定オッズ {row['想定オッズ']} / 期待値 {row['期待値']}")

        st.session_state.memo_input = st.text_input("メモ", value=st.session_state.memo_input)
        if st.button("予想履歴に保存"):
            bet_list = [r["買い目"] for r in st.session_state.final_bets]
            save_prediction_log(
                race_name=st.session_state.race_name,
                race_url=st.session_state.race_url,
                predict_type="3連単",
                mode_name=st.session_state.mode,
                mode_reason=st.session_state.selected_mode_reason,
                buy_points=len(bet_list),
                bet_amount=st.session_state.bet_amount,
                final_bets=bet_list,
                memo=st.session_state.memo_input,
            )
            st.success("保存しました")
            st.session_state.screen = "結果一覧"
            st.session_state.memo_input = ""
            st.rerun()


# =====================================
# 画面: 結果一覧
# =====================================
def render_results_list_screen():
    st.title("🧾 保存済み予想")
    pred_df = load_prediction_log()
    show_prediction_cards(pred_df)
    st.divider()
    if st.button("予想画面に戻る"):
        st.session_state.screen = "予想"
        st.rerun()


# =====================================
# 画面: 結果入力
# =====================================
def render_result_input_screen():
    st.title("🎯 結果入力")

    pred_df = load_prediction_log()
    if pred_df.empty:
        st.info("先に予想を保存してください")
        if st.button("戻る"):
            st.session_state.screen = "予想"
            st.rerun()
        return

    selected_id = st.session_state.get("selected_pred_id", "")
    if not selected_id or selected_id not in pred_df["ID"].astype(str).tolist():
        st.warning("対象の予想が見つかりません")
        if st.button("保存済み予想へ"):
            st.session_state.screen = "結果一覧"
            st.rerun()
        return

    row = pred_df[pred_df["ID"].astype(str) == selected_id].iloc[0]
    race_name = str(row["レース名"])
    buy_list_text = str(row["買い目一覧"])
    investment = int(row["合計金額"])

    with st.container(border=True):
        st.markdown(f"**{race_name}**")
        st.write(f"買い目: {buy_list_text}")
        st.write(f"投資額: ¥{investment:,}")

    c1, c2, c3 = st.columns(3)
    with c1:
        rank1 = st.number_input("1着", min_value=1, max_value=9, value=1, step=1)
    with c2:
        rank2 = st.number_input("2着", min_value=1, max_value=9, value=2, step=1)
    with c3:
        rank3 = st.number_input("3着", min_value=1, max_value=9, value=3, step=1)

    payout = st.number_input("払戻金", min_value=0, step=100, value=0)
    result_combo = f"{rank1}-{rank2}-{rank3}"
    hit = result_combo in buy_list_text.split(" / ")

    st.write(f"今回の着順: {result_combo}")
    st.write(f"判定: {'的中' if hit else 'ハズレ'}")

    c4, c5 = st.columns(2)
    with c4:
        if st.button("結果を保存する", type="primary"):
            save_result_log(pred_id=selected_id, race_name=race_name, rank1=rank1, rank2=rank2, rank3=rank3, payout=payout, investment=investment, hit=hit)
            st.success("結果を保存しました")
            st.session_state.screen = "結果履歴"
            st.rerun()
    with c5:
        if st.button("保存済み予想へ戻る"):
            st.session_state.screen = "結果一覧"
            st.rerun()


# =====================================
# 画面: 結果履歴
# =====================================
def render_result_history_screen():
    st.title("📊 結果履歴")

    pred_df = load_prediction_log()
    result_df = load_result_log()
    summary = calc_summary(pred_df, result_df)

    c1, c2 = st.columns(2)
    c1.metric("的中率", f"{summary['的中率']:.1f}%")
    c2.metric("回収率", f"{summary['回収率']:.1f}%")

    c3, c4 = st.columns(2)
    c3.metric("投資額", f"¥{int(summary['投資額']):,}")
    c4.metric("収支", f"¥{int(summary['収支']):,}")

    show_result_cards(result_df)

    c5, c6 = st.columns(2)
    with c5:
        if st.button("保存済み予想へ"):
            st.session_state.screen = "結果一覧"
            st.rerun()
    with c6:
        if st.button("予想画面へ"):
            st.session_state.screen = "予想"
            st.rerun()


# =====================================
# 下部ナビ
# =====================================
def render_bottom_nav():
    st.divider()
    n1, n2, n3 = st.columns(3)
    with n1:
        if st.button("予想", key="nav_predict"):
            st.session_state.screen = "予想"
            st.rerun()
    with n2:
        if st.button("結果入力", key="nav_results"):
            st.session_state.screen = "結果一覧"
            st.rerun()
    with n3:
        if st.button("結果履歴", key="nav_history"):
            st.session_state.screen = "結果履歴"
            st.rerun()


# =====================================
# メイン描画
# =====================================
if st.session_state.screen == "予想":
    render_predict_screen()
elif st.session_state.screen == "結果一覧":
    render_results_list_screen()
elif st.session_state.screen == "結果入力":
    render_result_input_screen()
elif st.session_state.screen == "結果履歴":
    render_result_history_screen()
else:
    render_predict_screen()

render_bottom_nav()

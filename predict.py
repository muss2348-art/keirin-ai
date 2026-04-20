# predict.py
# -*- coding: utf-8 -*-

import itertools
import pandas as pd


# ----------------------------------
# 安全変換
# ----------------------------------
def to_int(v, default=0):
    try:
        if v is None or v == "":
            return int(default)
        return int(float(v))
    except Exception:
        return int(default)


def to_float(v, default=0.0):
    try:
        if v is None or v == "":
            return float(default)
        return float(v)
    except Exception:
        return float(default)


# ----------------------------------
# モード自動判定
# ----------------------------------
def auto_detect_mode(df):
    """
    混戦モードを拾いやすくする判定
    """

    if df is None or len(df) == 0:
        return "通常モード"

    work = df.copy()

    for col in ["ライン", "ライン順", "単騎", "脚質", "競走得点"]:
        if col not in work.columns:
            if col in ["ライン", "ライン順", "単騎"]:
                work[col] = 0
            elif col == "競走得点":
                work[col] = 0.0
            else:
                work[col] = ""

    work["ライン"] = work["ライン"].apply(to_int)
    work["ライン順"] = work["ライン順"].apply(to_int)
    work["単騎"] = work["単騎"].apply(to_int)
    work["競走得点"] = work["競走得点"].apply(to_float)
    work["脚質"] = work["脚質"].fillna("").astype(str)

    tanki_cnt = int((work["単騎"] == 1).sum())

    line_df = work[work["ライン"] != 0].copy()
    line_counts = line_df.groupby("ライン").size().to_dict() if not line_df.empty else {}

    line_cnt = len(line_counts)
    two_line_cnt = sum(1 for _, cnt in line_counts.items() if cnt == 2)
    head_df = work[(work["ライン"] != 0) & (work["ライン順"] == 1)].copy()
    head_cnt = len(head_df)

    attack_cnt = int(work["脚質"].isin(["逃", "捲"]).sum())
    strong_heads = int((head_df["競走得点"] >= 68).sum()) if not head_df.empty else 0

    if tanki_cnt >= 2:
        return "混戦モード"

    if line_cnt >= 3:
        return "混戦モード"

    if two_line_cnt >= 2:
        return "混戦モード"

    if head_cnt >= 3 and attack_cnt >= 3:
        return "混戦モード"

    if strong_heads >= 2:
        return "混戦モード"

    if tanki_cnt >= 1 and line_cnt >= 2 and two_line_cnt >= 1:
        return "混戦モード"

    if len(work) == 7 and tanki_cnt >= 1 and line_cnt >= 2 and attack_cnt >= 2:
        return "混戦モード"

    return "通常モード"


# ----------------------------------
# スコア計算
# ----------------------------------
def calc_score(a, b, c, mode, weather):
    score = 0.0

    # 競走得点
    score += to_float(a["競走得点"]) * 0.45
    score += to_float(b["競走得点"]) * 0.35
    score += to_float(c["競走得点"]) * 0.20

    # 同ライン評価
    if to_int(a["ライン"]) != 0 and to_int(a["ライン"]) == to_int(b["ライン"]):
        score += 8

    if to_int(b["ライン"]) != 0 and to_int(b["ライン"]) == to_int(c["ライン"]):
        score += 3

    # 番手優遇
    if to_int(b["ライン順"]) == 2:
        score += 4

    # 単騎頭
    if to_int(a["単騎"]) == 1:
        score += 5

    # 脚質
    if str(a["脚質"]) == "逃":
        score += 5
    if str(a["脚質"]) == "捲":
        score += 4
    if str(a["脚質"]) == "追":
        score += 1

    # 混戦モード補正
    if mode == "混戦モード":
        if to_int(a["単騎"]) == 1:
            score += 8

        if to_int(a["ライン"]) != to_int(b["ライン"]):
            score += 6

        if str(a["脚質"]) in ["捲", "両"]:
            score += 3

    # 天候補正
    if weather == "風強":
        if str(a["脚質"]) == "逃":
            score -= 3
        if str(a["脚質"]) == "追":
            score += 3

    if weather == "雨":
        score *= 0.97

    return round(score, 2)


# ----------------------------------
# 相対ランク付け
# ----------------------------------
def assign_relative_ranks(pred: pd.DataFrame) -> pd.DataFrame:
    """
    固定閾値ではなく、今回の候補内で相対的にランク分けする
    """
    if pred is None or pred.empty:
        return pred

    out = pred.copy().reset_index(drop=True)
    n = len(out)

    if n == 1:
        out["買い目ランク"] = ["🔥 AI推奨"]
        out["AI評価"] = ["S"]
        return out

    # 上位から順にランクを振る
    rank_labels = []
    ai_labels = []

    top_ai = max(1, int(round(n * 0.15)))
    top_main = max(1, int(round(n * 0.30)))
    top_value = max(1, int(round(n * 0.60)))

    for i in range(n):
        if i < top_ai:
            rank_labels.append("🔥 AI推奨")
            ai_labels.append("S")
        elif i < top_main:
            rank_labels.append("🟢 本命")
            ai_labels.append("A")
        elif i < top_value:
            rank_labels.append("💰 期待値高")
            ai_labels.append("B")
        else:
            rank_labels.append("🟡 穴")
            ai_labels.append("C")

    out["買い目ランク"] = rank_labels
    out["AI評価"] = ai_labels
    return out


# ----------------------------------
# 買い目生成
# ----------------------------------
def generate_predictions(
    df,
    mode="通常モード",
    weather="晴",
    top_n=10,
    odds_dict=None,
    ticket_type="3連単"
):
    if odds_dict is None:
        odds_dict = {}

    rows = []
    players = df.to_dict("records")

    # ------------------------
    # 2車単
    # ------------------------
    if ticket_type == "2車単":
        for a, b in itertools.permutations(players, 2):
            ticket = f"{a['車番']}-{b['車番']}"

            dummy_c = a

            score = calc_score(
                a, b, dummy_c,
                mode,
                weather
            )

            odds = float(odds_dict.get(ticket, 10))
            ev = round(score * odds / 10, 1)

            rows.append(
                {
                    "買い目": ticket,
                    "score": score,
                    "期待値": ev,
                    "オッズ": odds
                }
            )

    # ------------------------
    # 3連単
    # ------------------------
    else:
        for a, b, c in itertools.permutations(players, 3):
            ticket = f"{a['車番']}-{b['車番']}-{c['車番']}"

            score = calc_score(
                a, b, c,
                mode,
                weather
            )

            odds = float(odds_dict.get(ticket, 30))
            ev = round(score * odds / 10, 1)

            rows.append(
                {
                    "買い目": ticket,
                    "score": score,
                    "期待値": ev,
                    "オッズ": odds
                }
            )

    pred = pd.DataFrame(rows)

    pred = pred.sort_values(
        ["score", "期待値"],
        ascending=False
    ).head(top_n)

    pred = pred.reset_index(drop=True)

    pred = assign_relative_ranks(pred)

    return pred

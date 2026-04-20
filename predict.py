# predict.py
# -*- coding: utf-8 -*-

import itertools
import pandas as pd


# ----------------------------------
# モード自動判定
# ----------------------------------
def auto_detect_mode(df):

    if df is None or len(df) == 0:
        return "通常モード"

    try:
        line_cnt = len(
            [
                x for x in df["ライン"].unique()
                if x != 0
            ]
        )
    except:
        line_cnt = 0

    try:
        tanki_cnt = int(
            (df["単騎"] == 1).sum()
        )
    except:
        tanki_cnt = 0

    # 混戦条件
    if tanki_cnt >= 2:
        return "混戦モード"

    if line_cnt >= 3:
        return "混戦モード"

    return "通常モード"


# ----------------------------------
# スコア計算
# ----------------------------------
def calc_score(a, b, c, mode, weather):

    score = 0

    # 競走得点
    score += a["競走得点"] * 0.45
    score += b["競走得点"] * 0.35
    score += c["競走得点"] * 0.20

    # 同ライン評価
    if a["ライン"] != 0 and a["ライン"] == b["ライン"]:
        score += 8

    if b["ライン"] != 0 and b["ライン"] == c["ライン"]:
        score += 3

    # 番手優遇
    if b["ライン順"] == 2:
        score += 4

    # 単騎頭少し拾う
    if a["単騎"] == 1:
        score += 5

    # 脚質
    if a["脚質"] == "逃":
        score += 5

    if a["脚質"] == "捲":
        score += 4

    # 混戦モード補正
    if mode == "混戦モード":

        if a["単騎"] == 1:
            score += 8

        if a["ライン"] != b["ライン"]:
            score += 6

    # 天候補正
    if weather == "風強":

        if a["脚質"] == "逃":
            score -= 3

        if a["脚質"] == "追":
            score += 3

    if weather == "雨":
        score *= 0.97

    return round(score,2)


# ----------------------------------
# ランク付け
# ----------------------------------
def get_rank(score):

    if score >= 88:
        return "🔥 AI推奨"

    elif score >= 82:
        return "🟢 本命"

    elif score >= 75:
        return "💰 期待値高"

    else:
        return "🟡 穴"


# ----------------------------------
# AI評価
# ----------------------------------
def get_ai_eval(score):

    if score >= 88:
        return "S"

    elif score >= 82:
        return "A"

    elif score >= 75:
        return "B"

    else:
        return "C"


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
    if ticket_type=="2車単":

        for a,b in itertools.permutations(players,2):

            ticket = f"{a['車番']}-{b['車番']}"

            dummy_c = a

            score = calc_score(
                a,b,dummy_c,
                mode,
                weather
            )

            odds = float(
                odds_dict.get(ticket,10)
            )

            ev = round(
                score * odds / 10,
                1
            )

            rows.append(
                {
                    "買い目":ticket,
                    "score":score,
                    "買い目ランク":get_rank(score),
                    "AI評価":get_ai_eval(score),
                    "期待値":ev,
                    "オッズ":odds
                }
            )

    # ------------------------
    # 3連単
    # ------------------------
    else:

        for a,b,c in itertools.permutations(players,3):

            ticket = (
                f"{a['車番']}-"
                f"{b['車番']}-"
                f"{c['車番']}"
            )

            score = calc_score(
                a,b,c,
                mode,
                weather
            )

            odds = float(
                odds_dict.get(ticket,30)
            )

            ev = round(
                score * odds / 10,
                1
            )

            rows.append(
                {
                    "買い目":ticket,
                    "score":score,
                    "買い目ランク":get_rank(score),
                    "AI評価":get_ai_eval(score),
                    "期待値":ev,
                    "オッズ":odds
                }
            )

    pred = pd.DataFrame(rows)

    pred = pred.sort_values(
        ["score","期待値"],
        ascending=False
    ).head(top_n)

    pred = pred.reset_index(
        drop=True
    )

    return pred

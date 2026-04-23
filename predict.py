# predict.py
import pandas as pd
import itertools


# =========================================
# モード判定
# =========================================
def auto_detect_mode(df: pd.DataFrame) -> str:
    line_counts = df["ライン"].value_counts()

    # 単騎多い → 混戦
    if (df["単騎"] == 1).sum() >= 3:
        return "混戦モード"

    # ラインがバラバラ
    if len(line_counts) >= 4:
        return "混戦モード"

    # 強い1ラインあり
    if max(line_counts) >= 3:
        return "通常モード"

    return "混戦モード"


# =========================================
# スコア計算
# =========================================
def calc_score(r1, r2, r3, mode):
    score = 0

    # ===== 基本 =====
    score += r1["競走得点"] * 1.5
    score += r2["競走得点"] * 1.0
    score += r3["競走得点"] * 0.7

    # ===== ライン =====
    if r1["ライン"] == r2["ライン"] and r1["ライン"] != 0:
        score += 15

    if r2["ライン"] == r3["ライン"] and r2["ライン"] != 0:
        score += 8

    # ===== 番手強化 =====
    if r2["ライン順"] == 2:
        score += 10

    # ===== 逃げ有利 =====
    if r1["脚質"] == "逃":
        score += 8

    # ===== 単騎 =====
    if r1["単騎"] == 1:
        score += 12   # ←かなり重要

    # =========================================
    # モード別調整
    # =========================================
    if mode == "混戦モード":
        # 単騎さらに強化
        if r1["単騎"] == 1:
            score += 20

        # 別ライン絡み
        if r1["ライン"] != r2["ライン"]:
            score += 10

        if r2["ライン"] != r3["ライン"]:
            score += 5

    return score


# =========================================
# ランク付け
# =========================================
def rank_ticket(score, odds):
    if odds == 0:
        return "🟡 穴"

    ev = score * odds

    if ev > 8000:
        return "🔥 AI推奨"
    elif ev > 4000:
        return "🟢 本命"
    elif ev > 2000:
        return "💰 期待値高"
    else:
        return "🟡 穴"


# =========================================
# メイン予想
# =========================================
def generate_predictions(
    df: pd.DataFrame,
    mode="通常モード",
    weather="晴",
    top_n=10,
    odds_dict=None,
    ticket_type="3連単"
):
    if odds_dict is None:
        odds_dict = {}

    results = []

    riders = df.to_dict("records")

    # =========================================
    # 組み合わせ生成
    # =========================================
    if ticket_type == "3連単":
        combos = itertools.permutations(riders, 3)
    else:
        combos = itertools.permutations(riders, 2)

    for combo in combos:
        if ticket_type == "3連単":
            r1, r2, r3 = combo
            ticket = f"{r1['車番']}-{r2['車番']}-{r3['車番']}"
            score = calc_score(r1, r2, r3, mode)
        else:
            r1, r2 = combo
            ticket = f"{r1['車番']}-{r2['車番']}"
            score = r1["競走得点"] * 1.5 + r2["競走得点"]

        odds = odds_dict.get(ticket, 0)

        results.append({
            "買い目": ticket,
            "スコア": score,
            "オッズ": odds,
        })

    result_df = pd.DataFrame(results)

    # =========================================
    # スコア順
    # =========================================
    result_df = result_df.sort_values("スコア", ascending=False)

    # =========================================
    # 期待値
    # =========================================
    result_df["期待値"] = result_df["スコア"] * result_df["オッズ"]

    # =========================================
    # ランク
    # =========================================
    result_df["買い目ランク"] = result_df.apply(
        lambda x: rank_ticket(x["スコア"], x["オッズ"]),
        axis=1
    )

    # =========================================
    # AI評価（見やすさ）
    # =========================================
    def ai_label(score):
        if score > 250:
            return "S"
        elif score > 200:
            return "A"
        elif score > 150:
            return "B"
        else:
            return "C"

    result_df["AI評価"] = result_df["スコア"].apply(ai_label)

    # =========================================
    # 上位抽出（バランス）
    # =========================================
    main = result_df.head(int(top_n * 0.6))
    mid = result_df.iloc[int(top_n * 0.6):int(top_n * 1.2)]
    mid = mid.sample(min(len(mid), int(top_n * 0.4))) if len(mid) > 0 else mid

    final_df = pd.concat([main, mid]).drop_duplicates()

    # =========================================
    # 最終整形
    # =========================================
    final_df = final_df.sort_values("期待値", ascending=False)
    final_df = final_df.head(top_n)

    return final_df.reset_index(drop=True)

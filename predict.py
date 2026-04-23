# predict.py
import itertools
import pandas as pd


def safe_float(v, default=0.0):
    try:
        if v is None or v == "":
            return float(default)
        if isinstance(v, str):
            v = v.replace(",", "").strip()
        return float(v)
    except Exception:
        return float(default)


def auto_detect_mode(df: pd.DataFrame) -> str:
    line_counts = df["ライン"].value_counts()
    single_count = int((df["単騎"] == 1).sum())

    if single_count >= 3:
        return "混戦モード"

    if len(line_counts) >= 4:
        return "混戦モード"

    if len(line_counts) > 0 and max(line_counts) >= 3:
        return "通常モード"

    return "混戦モード"


def get_single_bonus_weights(df: pd.DataFrame, mode: str):
    single_count = int((df["単騎"] == 1).sum())

    if single_count <= 1:
        head_bonus = 20
        second_bonus = 8
        third_bonus = 4
        strong_bonus = 14
        chaos_bonus = 18
    elif single_count == 2:
        head_bonus = 12
        second_bonus = 6
        third_bonus = 3
        strong_bonus = 8
        chaos_bonus = 10
    else:
        head_bonus = 6
        second_bonus = 4
        third_bonus = 2
        strong_bonus = 4
        chaos_bonus = 5

    if mode == "通常モード":
        head_bonus = max(4, head_bonus - 2)
        strong_bonus = max(2, strong_bonus - 2)

    return {
        "single_count": single_count,
        "head_bonus": head_bonus,
        "second_bonus": second_bonus,
        "third_bonus": third_bonus,
        "strong_bonus": strong_bonus,
        "chaos_bonus": chaos_bonus,
    }


def calc_race_balance_factor(df: pd.DataFrame) -> float:
    line_only = df[df["ライン"] > 0].copy()
    if line_only.empty:
        return 0.9

    line_sizes = line_only.groupby("ライン").size().tolist()
    max_line = max(line_sizes) if line_sizes else 0
    single_count = int((df["単騎"] == 1).sum())

    if max_line >= 3 and single_count <= 1:
        return 1.12

    if single_count >= 3:
        return 0.93

    return 1.0


def calc_score_3tan(r1, r2, r3, mode, bonus_cfg, race_balance_factor):
    score = 0.0

    score += r1["競走得点"] * 1.45
    score += r2["競走得点"] * 1.00
    score += r3["競走得点"] * 0.72

    if r1["ライン"] == r2["ライン"] and r1["ライン"] != 0:
        score += 15

    if r2["ライン"] == r3["ライン"] and r2["ライン"] != 0:
        score += 8

    if (
        r1["ライン"] != 0
        and r1["ライン"] == r2["ライン"] == r3["ライン"]
        and r1["ライン順"] == 1
        and r2["ライン順"] == 2
        and r3["ライン順"] == 3
    ):
        score += 10

    if r2["ライン順"] == 2:
        score += 8

    if r1["脚質"] == "逃":
        score += 7
    elif r1["脚質"] == "両":
        score += 5

    if r2["脚質"] in ["追", "両"]:
        score += 4
    if r3["脚質"] in ["追", "両"]:
        score += 2

    if r1["単騎"] == 1:
        score += bonus_cfg["head_bonus"]

        if r1["競走得点"] >= 90:
            score += bonus_cfg["strong_bonus"]

        if mode == "混戦モード":
            score += bonus_cfg["chaos_bonus"]

    if r2["単騎"] == 1:
        score += bonus_cfg["second_bonus"]

    if r3["単騎"] == 1:
        score += bonus_cfg["third_bonus"]

    if bonus_cfg["single_count"] >= 2:
        single_in_ticket = int(r1["単騎"] == 1) + int(r2["単騎"] == 1) + int(r3["単騎"] == 1)
        if single_in_ticket >= 2:
            score -= 8
        if single_in_ticket == 3:
            score -= 14

    if r1["ライン"] != 0 and r1["ライン"] == r2["ライン"]:
        score *= race_balance_factor

    return score


def calc_score_2tan(r1, r2, mode, bonus_cfg, race_balance_factor):
    score = 0.0

    score += r1["競走得点"] * 1.5
    score += r2["競走得点"] * 1.0

    if r1["ライン"] == r2["ライン"] and r1["ライン"] != 0:
        score += 14

    if r1["脚質"] == "逃":
        score += 6
    elif r1["脚質"] == "両":
        score += 4

    if r2["ライン順"] == 2:
        score += 8

    if r1["単騎"] == 1:
        score += bonus_cfg["head_bonus"]

        if r1["競走得点"] >= 90:
            score += bonus_cfg["strong_bonus"]

        if mode == "混戦モード":
            score += bonus_cfg["chaos_bonus"]

    if r2["単騎"] == 1:
        score += bonus_cfg["second_bonus"]

    if bonus_cfg["single_count"] >= 2:
        single_in_ticket = int(r1["単騎"] == 1) + int(r2["単騎"] == 1)
        if single_in_ticket >= 2:
            score -= 8

    if r1["ライン"] != 0 and r1["ライン"] == r2["ライン"]:
        score *= race_balance_factor

    return score


def rank_ticket(score, odds):
    if odds is None or odds <= 0:
        if score >= 185:
            return "🟢 本命"
        if score >= 160:
            return "🔥 AI推奨"
        return "🟡 穴"

    ev = score * odds

    if ev >= 9000:
        return "💰 期待値高"
    elif score >= 190:
        return "🔥 AI推奨"
    elif score >= 170:
        return "🟢 本命"
    else:
        return "🟡 穴"


def ai_label(score):
    if score >= 210:
        return "S"
    elif score >= 185:
        return "A"
    elif score >= 160:
        return "B"
    else:
        return "C"


def evaluate_race(df: pd.DataFrame, pred_df: pd.DataFrame, mode: str):
    if pred_df.empty:
        return {
            "レース判定": "見送り",
            "的中率評価": "低",
            "レース評価点": 0,
            "判定理由": "買い目が生成できませんでした",
        }

    single_count = int((df["単騎"] == 1).sum())
    top_score = float(pred_df["スコア"].max())
    mean_top5 = float(pred_df.head(min(5, len(pred_df)))["スコア"].mean())

    score = 50
    reasons = []

    if mode == "通常モード":
        score += 10
        reasons.append("通常モード")

    if single_count == 0:
        score += 8
        reasons.append("単騎なし")
    elif single_count == 1:
        score += 4
        reasons.append("単騎1車")
    elif single_count >= 3:
        score -= 10
        reasons.append("単騎多め")

    if top_score >= 205:
        score += 12
        reasons.append("本線強め")
    elif top_score >= 185:
        score += 6
        reasons.append("軸はある")
    else:
        score -= 6
        reasons.append("決め手弱め")

    if mean_top5 >= 180:
        score += 8
        reasons.append("上位買い目安定")
    else:
        score -= 4
        reasons.append("買い目分散")

    if score >= 75:
        decision = "買い"
        hit_label = "高"
    elif score >= 62:
        decision = "様子見"
        hit_label = "中"
    else:
        decision = "見送り"
        hit_label = "低"

    return {
        "レース判定": decision,
        "的中率評価": hit_label,
        "レース評価点": int(score),
        "判定理由": " / ".join(reasons),
    }


def rebalance_single_head_tickets(df: pd.DataFrame, result_df: pd.DataFrame, ticket_type: str, top_n: int):
    if result_df.empty:
        return result_df

    single_cars = set(df.loc[df["単騎"] == 1, "車番"].astype(int).tolist())
    single_count = len(single_cars)

    if single_count <= 1:
        return result_df.head(top_n).copy()

    out_rows = []
    max_single_head = max(2, int(top_n * 0.4))
    current_single_head = 0

    for _, row in result_df.iterrows():
        ticket = str(row["買い目"])
        head = int(ticket.split("-")[0])

        is_single_head = head in single_cars

        if is_single_head:
            if current_single_head >= max_single_head:
                continue
            current_single_head += 1

        out_rows.append(row)

        if len(out_rows) >= top_n:
            break

    if len(out_rows) < top_n:
        used_tickets = {str(r["買い目"]) for r in out_rows}
        for _, row in result_df.iterrows():
            if str(row["買い目"]) in used_tickets:
                continue
            out_rows.append(row)
            if len(out_rows) >= top_n:
                break

    return pd.DataFrame(out_rows).reset_index(drop=True)


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

    bonus_cfg = get_single_bonus_weights(df, mode)
    race_balance_factor = calc_race_balance_factor(df)

    if ticket_type == "3連単":
        combos = itertools.permutations(riders, 3)
    else:
        combos = itertools.permutations(riders, 2)

    for combo in combos:
        if ticket_type == "3連単":
            r1, r2, r3 = combo
            ticket = f"{r1['車番']}-{r2['車番']}-{r3['車番']}"
            score = calc_score_3tan(r1, r2, r3, mode, bonus_cfg, race_balance_factor)
        else:
            r1, r2 = combo
            ticket = f"{r1['車番']}-{r2['車番']}"
            score = calc_score_2tan(r1, r2, mode, bonus_cfg, race_balance_factor)

        if weather == "雨":
            if r1["脚質"] in ["逃", "両"]:
                score -= 4
            if ticket_type == "3連単" and r2["脚質"] in ["追", "両"]:
                score += 2
        elif weather == "風強":
            if r1["脚質"] == "逃":
                score -= 6
            if ticket_type == "3連単" and r2["脚質"] == "追":
                score += 3

        odds = safe_float(odds_dict.get(ticket, 0), 0.0)

        results.append({
            "買い目": ticket,
            "スコア": round(score, 2),
            "オッズ": odds,
        })

    result_df = pd.DataFrame(results)
    if result_df.empty:
        return result_df

    result_df["期待値"] = (result_df["スコア"] * result_df["オッズ"]).round(1)
    result_df["AI評価"] = result_df["スコア"].apply(ai_label)
    result_df["買い目ランク"] = result_df.apply(
        lambda x: rank_ticket(x["スコア"], x["オッズ"]),
        axis=1
    )

    result_df = result_df.sort_values(
        ["スコア", "期待値", "オッズ"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    result_df = rebalance_single_head_tickets(df, result_df, ticket_type, top_n)

    race_info = evaluate_race(df, result_df, mode)
    for col, val in race_info.items():
        result_df[col] = val

    return result_df.head(top_n).reset_index(drop=True)

# staking.py
# -*- coding: utf-8 -*-

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


def _round_100(x: float) -> int:
    return int(max(0, round(float(x) / 100.0) * 100))


def apply_staking_ai(
    pred_df: pd.DataFrame,
    unit_bet: int = 100,
    race_assessment: dict | None = None,
    max_multiplier: float = 3.0,
) -> pd.DataFrame:
    """
    賭け金AI。
    - 見送りなら購入金額を0円にする
    - 軽く買い/注意は金額を抑える
    - AI評価・期待値・ランクで厚張り候補を調整
    """
    if pred_df is None or pred_df.empty:
        return pred_df

    out = pred_df.copy()
    unit = max(100, int(unit_bet))

    if "AI評価" not in out.columns:
        out["AI評価"] = 0
    if "期待値" not in out.columns:
        out["期待値"] = 0
    if "買い目ランク" not in out.columns:
        out["買い目ランク"] = "🟡 穴"

    out["AI評価"] = pd.to_numeric(out["AI評価"], errors="coerce").fillna(0.0)
    out["期待値"] = pd.to_numeric(out["期待値"], errors="coerce").fillna(0.0)

    assessment = race_assessment or {}
    race_label = assessment.get("レース判定", "")
    race_multiplier = safe_float(assessment.get("賭け金倍率", 1.0), 1.0)

    if race_label == "見送り":
        out["購入金額"] = 0
        out["賭け金AI"] = "見送り推奨"
        out["資金配分理由"] = assessment.get("判定理由", "見送り判定")
        return out

    if race_label == "注意":
        race_multiplier = min(race_multiplier, 0.7)
    elif race_label == "軽く買い":
        race_multiplier = min(race_multiplier, 1.0)
    elif race_label == "買い":
        race_multiplier = max(race_multiplier, 1.15)

    amounts = []
    reasons = []

    ai_min = float(out["AI評価"].min())
    ai_max = float(out["AI評価"].max())
    ai_spread = max(ai_max - ai_min, 1.0)

    for _, row in out.iterrows():
        rank = str(row.get("買い目ランク", "🟡 穴"))
        ai = safe_float(row.get("AI評価", 0))
        ev = safe_float(row.get("期待値", 0))

        score_ratio = (ai - ai_min) / ai_spread

        mult = 1.0
        reason = []

        if rank == "🔥 AI推奨":
            mult += 1.2
            reason.append("AI推奨")
        elif rank == "🟢 本命":
            mult += 0.65
            reason.append("本命")
        elif rank == "💰 期待値高":
            mult += 0.45
            reason.append("期待値高")
        else:
            mult += 0.15
            reason.append("穴抑え")

        if ev >= 120:
            mult += 0.45
            reason.append("EV高")
        elif ev < 75:
            mult -= 0.35
            reason.append("EV低め")

        mult += score_ratio * 0.55
        mult *= race_multiplier
        mult = min(max(mult, 0.0), max_multiplier)

        amount = _round_100(unit * mult)
        if amount > 0:
            amount = max(unit, amount)

        amounts.append(amount)
        reasons.append(" / ".join(reason))

    out["購入金額"] = amounts
    out["賭け金AI"] = race_label if race_label else "通常配分"
    out["資金配分理由"] = reasons

    if "期待回収額(目安)" in out.columns:
        out["期待回収額(目安)"] = pd.to_numeric(out["期待値"], errors="coerce").fillna(0) / 100.0 * out["購入金額"]
        out["期待回収額(目安)"] = out["期待回収額(目安)"].round(0)

    return out


def staking_summary_text(pred_df: pd.DataFrame | None) -> str:
    if pred_df is None or pred_df.empty or "購入金額" not in pred_df.columns:
        return "賭け金AI: 未計算"

    total = int(pd.to_numeric(pred_df["購入金額"], errors="coerce").fillna(0).sum())
    if total <= 0:
        return "賭け金AI: 見送り推奨（購入金額0円）"

    max_amt = int(pd.to_numeric(pred_df["購入金額"], errors="coerce").fillna(0).max())
    return f"賭け金AI: 合計 {total:,}円 / 最大 {max_amt:,}円"

# staking.py
# -*- coding: utf-8 -*-

import pandas as pd


def _safe_float(v, default=0.0):
    try:
        if v is None or v == "":
            return float(default)
        if isinstance(v, str):
            v = v.replace(",", "").strip()
        return float(v)
    except Exception:
        return float(default)


def _round_100(x: float) -> int:
    if x <= 0:
        return 0
    return int(round(x / 100.0) * 100)


def _rank_multiplier(rank_label: str) -> float:
    rank = str(rank_label)
    if "AI推奨" in rank:
        return 2.2
    if "本命" in rank:
        return 1.6
    if "期待値" in rank:
        return 1.3
    if "穴" in rank:
        return 0.9
    return 1.0


def _race_multiplier(assessment: dict) -> tuple[float, str]:
    decision = str((assessment or {}).get("decision", ""))

    if decision == "買い":
        return 1.25, "見送りAI=買い"
    if decision == "軽く買い":
        return 1.0, "見送りAI=軽く買い"
    if decision == "注意":
        return 0.55, "見送りAI=注意のため抑え"
    if decision == "見送り":
        return 0.0, "見送りAI=見送りのため0円"
    return 1.0, "見送りAI判定なし"


def apply_staking_ai(
    pred_df: pd.DataFrame,
    unit_bet: int = 100,
    race_assessment: dict | None = None,
    max_per_ticket: int | None = None,
) -> pd.DataFrame:
    """
    買い目ごとの購入金額を自動配分する軽量AI。
    - 見送りAIが「見送り」なら全買い目0円
    - 注意なら抑えめ
    - AI評価・期待値・買い目ランクで厚みを調整
    """
    if pred_df is None or pred_df.empty:
        return pred_df

    out = pred_df.copy()
    unit = max(100, int(unit_bet))
    if max_per_ticket is None:
        max_per_ticket = unit * 8

    race_mult, race_reason = _race_multiplier(race_assessment or {})

    out["賭け金AI係数"] = 0.0
    out["賭け金AI理由"] = ""

    if race_mult <= 0:
        out["購入金額"] = 0
        out["期待回収額(目安)"] = 0
        out["賭け金AI係数"] = 0.0
        out["賭け金AI理由"] = race_reason
        return out

    ai_series = pd.to_numeric(out.get("AI評価", 0), errors="coerce").fillna(0)
    ev_series = pd.to_numeric(out.get("期待値", 0), errors="coerce").fillna(0)
    odds_series = pd.to_numeric(out.get("オッズ", 0), errors="coerce").fillna(0)

    ai_max = float(ai_series.max()) if len(ai_series) else 0.0
    ev_max = float(ev_series.max()) if len(ev_series) else 0.0

    amounts = []
    coeffs = []
    reasons = []

    for i, row in out.iterrows():
        coeff = race_mult
        reason_parts = [race_reason]

        rank = str(row.get("買い目ランク", ""))
        rm = _rank_multiplier(rank)
        coeff *= rm
        if rank:
            reason_parts.append(f"ランク係数{xformat(rm)}")

        ai = _safe_float(row.get("AI評価", 0))
        ev = _safe_float(row.get("期待値", 0))
        odds = _safe_float(row.get("オッズ", 0))

        if ai_max > 0:
            ai_ratio = ai / ai_max
            if ai_ratio >= 0.95:
                coeff *= 1.25
                reason_parts.append("AI評価上位")
            elif ai_ratio >= 0.85:
                coeff *= 1.10
                reason_parts.append("AI評価高め")
            elif ai_ratio < 0.65:
                coeff *= 0.75
                reason_parts.append("AI評価控えめ")

        if ev > 0:
            if ev >= 115:
                coeff *= 1.25
                reason_parts.append("期待値高")
            elif ev >= 105:
                coeff *= 1.10
                reason_parts.append("期待値やや高")
            elif ev < 90:
                coeff *= 0.70
                reason_parts.append("期待値低め")

        if odds > 0:
            if odds >= 100:
                coeff *= 0.65
                reason_parts.append("高配当すぎるため抑え")
            elif odds >= 60:
                coeff *= 0.80
                reason_parts.append("高配当寄り")
            elif 5 <= odds <= 35:
                coeff *= 1.05
                reason_parts.append("買いやすいオッズ帯")

        amount = _round_100(unit * coeff)

        if amount > 0:
            amount = max(unit, amount)
            amount = min(max_per_ticket, amount)

        amounts.append(int(amount))
        coeffs.append(round(coeff, 2))
        reasons.append(" / ".join(reason_parts))

    out["賭け金AI係数"] = coeffs
    out["賭け金AI理由"] = reasons
    out["購入金額"] = amounts

    if "期待値" in out.columns:
        ev_num = pd.to_numeric(out["期待値"], errors="coerce").fillna(0)
        out["期待回収額(目安)"] = (ev_num / 100.0 * out["購入金額"]).round(0)
    else:
        out["期待回収額(目安)"] = 0

    return out


def xformat(v: float) -> str:
    try:
        return f"{float(v):.2f}"
    except Exception:
        return str(v)


def staking_summary_text(pred_df: pd.DataFrame) -> str:
    if pred_df is None or pred_df.empty or "購入金額" not in pred_df.columns:
        return "賭け金AI: 未計算"

    amounts = pd.to_numeric(pred_df["購入金額"], errors="coerce").fillna(0)
    total = int(amounts.sum())

    if total <= 0:
        return "賭け金AI: 見送り推奨のため購入金額0円"

    max_amt = int(amounts.max())
    min_amt = int(amounts[amounts > 0].min()) if (amounts > 0).any() else 0

    return f"賭け金AI: 合計{total:,}円 / 最小{min_amt:,}円 / 最大{max_amt:,}円"

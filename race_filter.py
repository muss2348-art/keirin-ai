# race_filter.py
# -*- coding: utf-8 -*-

from pathlib import Path
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


def safe_int(v, default=0):
    try:
        if v is None or v == "":
            return int(default)
        if isinstance(v, str):
            v = v.replace(",", "").strip()
        return int(float(v))
    except Exception:
        return int(default)


def _load_log(log_path) -> pd.DataFrame:
    path = Path(log_path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        try:
            return pd.read_csv(path, encoding="utf-8")
        except Exception:
            return pd.DataFrame()


def _line_shape_score(df: pd.DataFrame) -> tuple[float, list[str]]:
    reasons = []
    score = 50.0

    if df is None or df.empty:
        return 0.0, ["出走表なし"]

    d = df.copy()
    for c in ["車番", "競走得点", "ライン", "ライン順", "単騎"]:
        if c not in d.columns:
            d[c] = 0
    d["競走得点"] = pd.to_numeric(d["競走得点"], errors="coerce").fillna(0.0)
    d["ライン"] = pd.to_numeric(d["ライン"], errors="coerce").fillna(0).astype(int)
    d["単騎"] = pd.to_numeric(d["単騎"], errors="coerce").fillna(0).astype(int)

    singles = int((d["単騎"] == 1).sum()) + int((d["ライン"] == 0).sum())
    line_counts = d[d["ライン"] > 0].groupby("ライン")["車番"].count().tolist()
    max_line = max(line_counts) if line_counts else 0
    two_lines = sum(1 for x in line_counts if x == 2)
    spread = float(d["競走得点"].max() - d["競走得点"].min()) if len(d) else 0.0

    if max_line >= 3:
        score += 12
        reasons.append("3車ラインあり")
    if max_line >= 4:
        score += 6
        reasons.append("長いラインで軸を作りやすい")
    if singles >= 2:
        score -= 12
        reasons.append("単騎多め")
    elif singles == 1:
        score -= 4
        reasons.append("単騎あり")
    if two_lines >= 2:
        score -= 6
        reasons.append("2車ライン多めで混戦")
    if spread >= 10:
        score += 8
        reasons.append("得点差あり")
    elif 0 < spread <= 4:
        score -= 8
        reasons.append("得点差小さく混戦")

    return score, reasons


def _prediction_score(pred_df: pd.DataFrame) -> tuple[float, list[str]]:
    if pred_df is None or pred_df.empty:
        return 0.0, ["買い目候補なし"]

    d = pred_df.copy()
    for c in ["AI評価", "期待値", "オッズ"]:
        if c not in d.columns:
            d[c] = 0
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)

    top = d.head(min(5, len(d)))
    avg_ai = float(top["AI評価"].mean()) if not top.empty else 0.0
    avg_ev = float(top["期待値"].mean()) if not top.empty else 0.0

    score = 0.0
    reasons = []

    if avg_ai >= 150:
        score += 24
        reasons.append("上位AI評価が高い")
    elif avg_ai >= 130:
        score += 14
        reasons.append("上位AI評価まずまず")
    elif avg_ai < 105:
        score -= 12
        reasons.append("上位AI評価が低い")

    if avg_ev >= 115:
        score += 12
        reasons.append("期待値高め")
    elif avg_ev < 80:
        score -= 8
        reasons.append("期待値低め")

    # 上位買い目が同じ頭に偏りすぎるなら本命寄り、分散なら混戦寄り
    if "買い目" in d.columns:
        heads = d.head(min(8, len(d)))["買い目"].astype(str).str.split("-").str[0]
        top_head_rate = heads.value_counts(normalize=True).max() if len(heads) else 0
        if top_head_rate >= 0.5:
            score += 6
            reasons.append("軸候補が見える")
        elif top_head_rate <= 0.25:
            score -= 5
            reasons.append("頭が分散")

    return score, reasons


def _log_score(log_path, mode: str, weather: str, ticket_type: str) -> tuple[float, list[str]]:
    log = _load_log(log_path)
    if log.empty or len(log) < 10:
        return 0.0, ["ログ少なめ"]

    work = log.copy()
    for col in ["判定", "券種", "モード", "天候", "購入金額", "オッズ"]:
        if col not in work.columns:
            work[col] = 0 if col in ["購入金額", "オッズ"] else ""
    work["購入金額"] = pd.to_numeric(work["購入金額"], errors="coerce").fillna(0.0)
    work["オッズ"] = pd.to_numeric(work["オッズ"], errors="coerce").fillna(0.0)
    work["払戻額"] = work.apply(lambda r: r["購入金額"] * r["オッズ"] if str(r["判定"]) == "的中" else 0.0, axis=1)

    filt = work.copy()
    if ticket_type:
        filt = filt[filt["券種"].astype(str) == str(ticket_type)]
    if mode:
        mode_f = filt[filt["モード"].astype(str) == str(mode)]
        if len(mode_f) >= 5:
            filt = mode_f
    if weather:
        weather_f = filt[filt["天候"].astype(str) == str(weather)]
        if len(weather_f) >= 5:
            filt = weather_f

    if len(filt) < 5:
        return 0.0, ["条件別ログ不足"]

    bet = float(filt["購入金額"].sum())
    ret = float(filt["払戻額"].sum())
    roi = ret / bet if bet > 0 else 0.0
    hit_rate = float((filt["判定"].astype(str) == "的中").mean())

    score = 0.0
    reasons = []
    if roi >= 1.2:
        score += 10
        reasons.append(f"条件別ROI良好 {roi*100:.1f}%")
    elif roi < 0.65:
        score -= 10
        reasons.append(f"条件別ROI低め {roi*100:.1f}%")

    if hit_rate >= 0.25:
        score += 5
        reasons.append(f"条件別的中率良好 {hit_rate*100:.1f}%")
    elif hit_rate < 0.08:
        score -= 5
        reasons.append(f"条件別的中率低め {hit_rate*100:.1f}%")

    return score, reasons


def assess_race_buyability(
    df: pd.DataFrame,
    pred_df: pd.DataFrame | None = None,
    log_path=None,
    mode: str = "",
    weather: str = "",
    ticket_type: str = "",
    race_type: str = "通常",
) -> dict:
    base_score, base_reasons = _line_shape_score(df)
    pred_score, pred_reasons = _prediction_score(pred_df)
    log_score, log_reasons = _log_score(log_path, mode, weather, ticket_type) if log_path else (0.0, [])

    score = base_score + pred_score + log_score

    if race_type == "G3":
        # G3は荒れ許容。完全見送りにしすぎない。
        score += 3
    elif race_type == "ガールズ":
        # ガールズは得点差が出やすいが、ライン情報なし前提
        score += 2

    if score >= 82:
        label = "買い"
        hit_eval = "高"
        multiplier = 1.25
    elif score >= 66:
        label = "軽く買い"
        hit_eval = "中"
        multiplier = 1.0
    elif score >= 52:
        label = "注意"
        hit_eval = "低〜中"
        multiplier = 0.7
    else:
        label = "見送り"
        hit_eval = "低"
        multiplier = 0.0

    reasons = base_reasons + pred_reasons + log_reasons

    return {
        "レース判定": label,
        "的中率評価": hit_eval,
        "レース評価点": round(score, 1),
        "賭け金倍率": multiplier,
        "判定理由": " / ".join([r for r in reasons if r]) if reasons else "総合評価",
    }


def apply_race_buyability_to_predictions(pred_df: pd.DataFrame, assessment: dict) -> pd.DataFrame:
    if pred_df is None or pred_df.empty:
        return pred_df

    out = pred_df.copy()
    out["レース判定"] = assessment.get("レース判定", "")
    out["的中率評価"] = assessment.get("的中率評価", "")
    out["レース評価点"] = assessment.get("レース評価点", 0)
    out["見送り理由"] = assessment.get("判定理由", "")

    if "AI評価" in out.columns:
        out["AI評価"] = pd.to_numeric(out["AI評価"], errors="coerce").fillna(0.0)
        if assessment.get("レース判定") == "見送り":
            out["AI評価"] = (out["AI評価"] - 12).round(2)
        elif assessment.get("レース判定") == "買い":
            out["AI評価"] = (out["AI評価"] + 5).round(2)

    return out


def race_buyability_summary_text(assessment: dict | None) -> str:
    if not assessment:
        return "見送りAI: 未判定"
    return (
        f"見送りAI: {assessment.get('レース判定', '')} / "
        f"的中率評価 {assessment.get('的中率評価', '')} / "
        f"評価点 {assessment.get('レース評価点', 0)} / "
        f"{assessment.get('判定理由', '')}"
    )

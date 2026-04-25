# race_filter.py
# -*- coding: utf-8 -*-

from pathlib import Path
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


def _safe_int(v, default=0):
    try:
        if v is None or v == "":
            return int(default)
        return int(float(v))
    except Exception:
        return int(default)


def _load_log(log_path):
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


def _rate_bonus(log_df, col, value, min_count=8):
    if log_df is None or log_df.empty or col not in log_df.columns:
        return 0.0, ""
    work = log_df.copy()
    work[col] = work[col].fillna("").astype(str)
    target = work[work[col] == str(value)].copy()
    if len(target) < min_count:
        return 0.0, ""
    if "判定" not in target.columns:
        return 0.0, ""
    hit_rate = (target["判定"].astype(str) == "的中").mean()
    if hit_rate >= 0.22:
        return 5.0, f"過去の{col}成績が良好"
    if hit_rate <= 0.06:
        return -7.0, f"過去の{col}成績が低調"
    return 0.0, ""


def assess_race_buyability(
    current_df: pd.DataFrame,
    pred_df: pd.DataFrame = None,
    log_path=None,
    mode: str = "",
    weather: str = "",
    ticket_type: str = "",
    race_type: str = "通常",
) -> dict:
    """レース自体を買うか見送るか判定する軽量スコアリング。"""
    score = 60.0
    reasons = []

    if current_df is None or current_df.empty:
        return {
            "decision": "見送り",
            "hit_label": "低",
            "score": 0,
            "reason": "出走表が空のため見送り",
            "advice": "選手情報を取得・入力してから予想してください。",
        }

    df = current_df.copy()
    n = len(df)

    for col in ["車番", "選手名", "競走得点", "脚質", "ライン", "ライン順", "単騎"]:
        if col not in df.columns:
            df[col] = "" if col in ["選手名", "脚質"] else 0

    scores = pd.to_numeric(df["競走得点"], errors="coerce").fillna(0)
    missing_score = int((scores <= 0).sum())
    missing_name = int((df["選手名"].astype(str).str.strip() == "").sum())
    missing_style = int((df["脚質"].astype(str).str.strip() == "").sum())

    if missing_name > 0:
        score -= min(18, missing_name * 4)
        reasons.append(f"選手名未入力{missing_name}人")
    if missing_score > 0:
        score -= min(20, missing_score * 5)
        reasons.append(f"競走得点未入力{missing_score}人")
    if missing_style > 0:
        score -= min(12, missing_style * 3)
        reasons.append(f"脚質未入力{missing_style}人")

    valid_scores = scores[scores > 0]
    if len(valid_scores) >= 2:
        spread = float(valid_scores.max() - valid_scores.min())
        std = float(valid_scores.std()) if len(valid_scores) >= 3 else 0.0
        sorted_scores = valid_scores.sort_values(ascending=False).tolist()
        top_gap = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) >= 2 else 0.0

        if top_gap >= 4.0:
            score += 8
            reasons.append("上位得点差があり軸を作りやすい")
        elif top_gap <= 1.0:
            score -= 5
            reasons.append("上位得点差が小さく混戦")

        if std >= 4.5:
            score += 4
            reasons.append("得点差が大きく序列あり")
        elif std <= 1.8 and n >= 6:
            score -= 6
            reasons.append("得点が接近して荒れやすい")

        if spread <= 3.0 and n >= 6:
            score -= 4
            reasons.append("全体の得点差が小さい")

    singles = pd.to_numeric(df["単騎"], errors="coerce").fillna(0).astype(int)
    single_count = int((singles == 1).sum())

    lines = pd.to_numeric(df["ライン"], errors="coerce").fillna(0).astype(int)
    line_sizes = []
    for lid, cnt in lines[lines > 0].value_counts().items():
        line_sizes.append(int(cnt))
    max_line = max(line_sizes) if line_sizes else 0
    two_car_lines = sum(1 for x in line_sizes if x == 2)

    if race_type == "通常":
        if max_line >= 3:
            score += 7
            reasons.append("3車以上ラインあり")
        elif max_line == 2 and two_car_lines >= 2:
            score += 2
            reasons.append("2車ライン中心")
        elif max_line <= 1:
            score -= 8
            reasons.append("ラインが薄く展開読みづらい")

        if single_count >= 3:
            score -= 10
            reasons.append("単騎が多く展開不安")
        elif single_count == 2:
            score -= 5
            reasons.append("単騎2人で波乱含み")
        elif single_count == 1:
            score -= 1
            reasons.append("単騎あり")
    else:
        score -= 1
        reasons.append("ガールズは展開より個力重視")

    if weather == "雨":
        score -= 3
        reasons.append("雨で不確定要素あり")
    elif weather == "風強":
        score -= 5
        reasons.append("風強で展開乱れやすい")

    if pred_df is not None and not pred_df.empty:
        pred = pred_df.copy()
        ai = pd.to_numeric(pred.get("AI評価", 0), errors="coerce").fillna(0)
        ev = pd.to_numeric(pred.get("期待値", 0), errors="coerce").fillna(0)
        odds = pd.to_numeric(pred.get("オッズ", 0), errors="coerce").fillna(0)

        if len(ai) > 0:
            top_ai = float(ai.max())
            avg_ai = float(ai.head(min(10, len(ai))).mean())
            if top_ai >= 75:
                score += 5
                reasons.append("AI評価上位が強い")
            elif avg_ai < 45:
                score -= 6
                reasons.append("AI評価平均が低い")

        if len(ev) > 0 and ev.max() > 0:
            avg_ev = float(ev.head(min(10, len(ev))).mean())
            if avg_ev >= 105:
                score += 5
                reasons.append("期待値が高め")
            elif avg_ev < 90:
                score -= 5
                reasons.append("期待値が低め")

        if len(odds) > 0 and (odds > 0).sum() >= 3:
            median_odds = float(odds[odds > 0].head(min(10, len(odds))).median())
            if median_odds >= 80 and ticket_type == "3連単":
                score -= 6
                reasons.append("上位買い目のオッズが高く荒れ気味")
            elif 5 <= median_odds <= 35:
                score += 3
                reasons.append("オッズ帯が買いやすい")

    if log_path:
        log_df = _load_log(log_path)
        for col, value in [("モード", mode), ("天候", weather), ("券種", ticket_type), ("レース種別", race_type)]:
            b, r = _rate_bonus(log_df, col, value)
            if b:
                score += b
                reasons.append(r)

    score = max(0, min(100, round(score, 1)))

    if score >= 72:
        decision = "買い"
        hit_label = "高"
        advice = "本命寄りで点数を絞ってもよいレース。"
    elif score >= 58:
        decision = "軽く買い"
        hit_label = "中"
        advice = "買うなら通常点数。無理に厚張りしない。"
    elif score >= 45:
        decision = "注意"
        hit_label = "低〜中"
        advice = "買うなら少額・点数控えめ。見送りも候補。"
    else:
        decision = "見送り"
        hit_label = "低"
        advice = "無理に買わず見送り推奨。買うなら遊び程度。"

    reason = " / ".join(reasons[:8]) if reasons else "大きな不安材料なし"

    return {
        "decision": decision,
        "hit_label": hit_label,
        "score": score,
        "reason": reason,
        "advice": advice,
    }


def apply_race_buyability_to_predictions(
    pred_df: pd.DataFrame,
    assessment: dict,
) -> pd.DataFrame:
    if pred_df is None or pred_df.empty:
        return pred_df
    out = pred_df.copy()
    out["レース判定"] = assessment.get("decision", "")
    out["的中率評価"] = assessment.get("hit_label", "")
    out["レース評価点"] = assessment.get("score", "")
    out["判定理由"] = assessment.get("reason", "")
    out["見送りAIコメント"] = assessment.get("advice", "")
    return out


def race_buyability_summary_text(assessment: dict) -> str:
    return (
        f"見送りAI: {assessment.get('decision', '-')} / "
        f"的中率評価: {assessment.get('hit_label', '-')} / "
        f"評価点: {assessment.get('score', '-')} / "
        f"理由: {assessment.get('reason', '-')}"
    )

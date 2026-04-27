# predict.py
# -*- coding: utf-8 -*-

import itertools
import math
from typing import Dict, List, Tuple, Any

import pandas as pd


DEFAULT_COLUMNS = ["車番", "選手名", "競走得点", "脚質", "ライン", "ライン順", "単騎"]


def safe_float(v, default=0.0) -> float:
    try:
        if v is None or v == "":
            return float(default)
        if isinstance(v, str):
            v = v.replace(",", "").strip()
        return float(v)
    except Exception:
        return float(default)


def safe_int(v, default=0) -> int:
    try:
        if v is None or v == "":
            return int(default)
        if isinstance(v, str):
            v = v.replace(",", "").strip()
        return int(float(v))
    except Exception:
        return int(default)


def normalize_ticket(ticket: str) -> str:
    return str(ticket).replace(" ", "").strip()


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=DEFAULT_COLUMNS)

    out = df.copy()

    for c in DEFAULT_COLUMNS:
        if c not in out.columns:
            if c == "競走得点":
                out[c] = 0.0
            elif c in ["車番", "ライン", "ライン順", "単騎"]:
                out[c] = 0
            else:
                out[c] = ""

    out["車番"] = pd.to_numeric(out["車番"], errors="coerce").fillna(0).astype(int)
    out["競走得点"] = pd.to_numeric(out["競走得点"], errors="coerce").fillna(0.0)
    out["ライン"] = pd.to_numeric(out["ライン"], errors="coerce").fillna(0).astype(int)
    out["ライン順"] = pd.to_numeric(out["ライン順"], errors="coerce").fillna(0).astype(int)
    out["単騎"] = pd.to_numeric(out["単騎"], errors="coerce").fillna(0).astype(int)
    out["脚質"] = out["脚質"].fillna("").astype(str)

    return out[DEFAULT_COLUMNS].sort_values("車番").reset_index(drop=True)


def auto_detect_mode(df: pd.DataFrame) -> str:
    d = prepare_df(df)
    if d.empty:
        return "通常モード"

    singles = int((d["単騎"] == 1).sum())
    line_counts = d[d["ライン"] > 0].groupby("ライン")["車番"].count().tolist()
    max_line = max(line_counts) if line_counts else 0
    two_lines = sum(1 for x in line_counts if x == 2)
    avg_score = float(d["競走得点"].mean()) if len(d) else 0
    score_spread = float(d["競走得点"].max() - d["競走得点"].min()) if len(d) else 0

    if singles >= 2:
        return "混戦モード"
    if max_line <= 2 and two_lines >= 2:
        return "混戦モード"
    if score_spread <= 4.0 and avg_score > 0:
        return "混戦モード"
    if score_spread >= 12.0:
        return "穴モード"

    return "通常モード"


def get_line_groups(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    d = prepare_df(df)
    groups = {}
    for line_id in sorted(d["ライン"].unique()):
        if int(line_id) <= 0:
            continue
        g = d[d["ライン"] == int(line_id)].sort_values("ライン順").reset_index(drop=True)
        if not g.empty:
            groups[int(line_id)] = g
    return groups


def calc_line_reliability(df: pd.DataFrame) -> Dict[str, Any]:
    """
    ラインをどこまで信用するかを判定する。
    scoreが低いほど「番手千切れ・ライン崩れ」を買い目に入れる。
    """
    d = prepare_df(df)
    groups = get_line_groups(d)

    total_score = 0.0
    details = []
    line_scores = {}

    for line_id, g in groups.items():
        score = 0.0
        reasons = []

        if len(g) < 2:
            continue

        first = g.iloc[0]
        second = g.iloc[1]

        first_style = str(first.get("脚質", ""))
        second_style = str(second.get("脚質", ""))

        first_score = safe_float(first.get("競走得点", 0))
        second_score = safe_float(second.get("競走得点", 0))

        if first_style in ["逃", "両", "自"]:
            score += 2.0
            reasons.append("先頭自力")
        elif first_style in ["追"]:
            score -= 1.5
            reasons.append("先頭追込みで不安")

        if second_style == "追":
            score += 2.0
            reasons.append("番手追込み")
        elif second_style in ["逃", "両", "自"]:
            score -= 0.5
            reasons.append("番手自力型で連携やや不安")

        diff = abs(first_score - second_score)
        if diff <= 3:
            score += 1.5
            reasons.append("前後得点差小")
        elif diff <= 6:
            score += 0.5
            reasons.append("前後得点差普通")
        else:
            score -= 1.2
            reasons.append("前後得点差大で千切れ注意")

        if len(g) >= 3:
            third = g.iloc[2]
            third_score = safe_float(third.get("競走得点", 0))
            if abs(second_score - third_score) <= 5:
                score += 0.6
                reasons.append("3番手まで連携可")
            else:
                score -= 0.6
                reasons.append("3番手千切れ注意")

        line_scores[line_id] = round(score, 2)
        total_score += score
        details.append(f"L{line_id}:{score:.1f}({','.join(reasons)})")

    singles = int((d["単騎"] == 1).sum())
    if singles >= 2:
        total_score -= 2.0
        details.append("単騎2車以上でライン信頼低下")
    elif singles == 1:
        total_score -= 0.6
        details.append("単騎あり")

    if not groups:
        total_score -= 3.0
        details.append("ラインなし")

    if total_score >= 5:
        level = "高"
    elif total_score >= 2:
        level = "中"
    else:
        level = "低"

    return {
        "score": round(total_score, 2),
        "level": level,
        "line_scores": line_scores,
        "details": " / ".join(details) if details else "ライン情報少",
    }


def calc_rider_base_scores(df: pd.DataFrame, race_type: str = "通常") -> Dict[int, float]:
    d = prepare_df(df)
    if d.empty:
        return {}

    max_score = max(float(d["競走得点"].max()), 1.0)
    min_score = float(d["競走得点"].min())
    spread = max(max_score - min_score, 1.0)

    base = {}

    for _, r in d.iterrows():
        car = int(r["車番"])
        point = safe_float(r["競走得点"])
        style = str(r.get("脚質", ""))
        line = safe_int(r.get("ライン", 0))
        order = safe_int(r.get("ライン順", 0))
        single = safe_int(r.get("単騎", 0))

        # 得点基礎
        s = 50.0 + ((point - min_score) / spread) * 35.0

        # 脚質補正
        if style == "逃":
            s += 5.0
        elif style in ["両", "自"]:
            s += 6.0
        elif style == "捲":
            s += 4.5
        elif style == "追":
            s += 2.5

        # ライン位置補正
        if single == 1 or line == 0:
            s += 0.5
        else:
            if order == 1:
                s += 5.0
            elif order == 2:
                s += 4.0
            elif order == 3:
                s += 1.5

        # G3は点数上位・自力・単騎も来るので少し穴寄り
        if race_type == "G3":
            if style in ["両", "自", "捲"]:
                s += 3.0
            if single == 1 and point >= d["競走得点"].median():
                s += 2.0
        elif race_type == "ガールズ":
            # ガールズはライン評価をほぼ使わず、得点・自力寄り
            if style in ["逃", "両", "自", "捲"]:
                s += 3.0
            if style == "追":
                s -= 1.0

        base[car] = round(s, 3)

    return base


def get_rider(df: pd.DataFrame, car: int) -> pd.Series:
    d = prepare_df(df)
    hit = d[d["車番"] == int(car)]
    if hit.empty:
        return pd.Series(dtype=object)
    return hit.iloc[0]


def is_same_line(df: pd.DataFrame, a: int, b: int) -> bool:
    ra = get_rider(df, a)
    rb = get_rider(df, b)
    if ra.empty or rb.empty:
        return False
    la = safe_int(ra.get("ライン", 0))
    lb = safe_int(rb.get("ライン", 0))
    return la > 0 and la == lb


def line_order(df: pd.DataFrame, car: int) -> int:
    r = get_rider(df, car)
    if r.empty:
        return 0
    return safe_int(r.get("ライン順", 0))


def is_single(df: pd.DataFrame, car: int) -> bool:
    r = get_rider(df, car)
    if r.empty:
        return False
    return safe_int(r.get("単騎", 0)) == 1 or safe_int(r.get("ライン", 0)) == 0


def is_self_type(df: pd.DataFrame, car: int) -> bool:
    r = get_rider(df, car)
    if r.empty:
        return False
    return str(r.get("脚質", "")) in ["逃", "両", "自", "捲"]


def is_chasing_type(df: pd.DataFrame, car: int) -> bool:
    r = get_rider(df, car)
    if r.empty:
        return False
    return str(r.get("脚質", "")) == "追"


def score_ticket(
    df: pd.DataFrame,
    ticket: Tuple[int, ...],
    base_scores: Dict[int, float],
    odds_dict: Dict[str, float],
    mode: str,
    weather: str,
    ticket_type: str,
    race_type: str,
    line_info: Dict[str, Any],
) -> Dict[str, Any]:
    d = prepare_df(df)
    line_level = line_info.get("level", "中")
    line_score = safe_float(line_info.get("score", 0))
    n = len(ticket)

    score = 0.0
    reasons = []

    # 着順ごとの基礎評価
    weights = [1.0, 0.72, 0.52] if n == 3 else [1.0, 0.72]
    for i, car in enumerate(ticket):
        score += base_scores.get(int(car), 0.0) * weights[i]

    head = int(ticket[0])
    second = int(ticket[1]) if len(ticket) >= 2 else None
    third = int(ticket[2]) if len(ticket) >= 3 else None

    # ライン素直パターン
    if second is not None and is_same_line(d, head, second):
        h_order = line_order(d, head)
        s_order = line_order(d, second)
        if h_order > 0 and s_order == h_order + 1:
            if line_level == "高":
                score += 18.0
                reasons.append("ライン信頼高の素直")
            elif line_level == "中":
                score += 10.0
                reasons.append("ライン素直")
            else:
                score += 4.0
                reasons.append("ライン低信頼で素直は控えめ")

    if third is not None and second is not None and is_same_line(d, second, third):
        s_order = line_order(d, second)
        t_order = line_order(d, third)
        if s_order > 0 and t_order == s_order + 1:
            if line_level == "高":
                score += 7.0
            elif line_level == "中":
                score += 3.5
            else:
                score += 1.0

    # ライン崩れパターン
    break_bonus = 0.0
    if second is not None and not is_same_line(d, head, second):
        if line_level == "低":
            break_bonus += 12.0
            reasons.append("ライン崩れ想定")
        elif line_level == "中":
            break_bonus += 5.0
            reasons.append("ライン崩れ少し")
        else:
            break_bonus -= 2.0

    # 番手飛び・捲り差し
    if third is not None:
        if is_same_line(d, head, third) and not is_same_line(d, head, second):
            if line_level in ["低", "中"]:
                break_bonus += 7.0
                reasons.append("番手飛び/別線割込み")
        if is_self_type(d, head) and is_chasing_type(d, third) and not is_same_line(d, head, second):
            if line_level == "低":
                break_bonus += 4.0

    score += break_bonus

    # 単騎頭は厳選。強い単騎だけ残す。
    if is_single(d, head):
        head_point = safe_float(get_rider(d, head).get("競走得点", 0))
        median = float(d["競走得点"].median()) if not d.empty else 0.0
        top_rank = d["競走得点"].rank(ascending=False, method="min")
        head_rank = int(top_rank[d["車番"] == head].iloc[0]) if (d["車番"] == head).any() else 99

        if race_type == "G3":
            if head_point >= median and (is_self_type(d, head) or head_rank <= 3):
                score += 8.0
                reasons.append("G3単騎穴頭")
            else:
                score -= 8.0
                reasons.append("単騎頭抑制")
        else:
            if line_level == "低" and head_rank <= 3 and is_self_type(d, head):
                score += 6.0
                reasons.append("厳選単騎頭")
            elif head_rank <= 2 and head_point >= median + 3:
                score += 3.0
                reasons.append("得点上位単騎")
            else:
                score -= 14.0
                reasons.append("単騎頭過多抑制")

    # 単騎2着3着は少し許容
    for pos, car in enumerate(ticket[1:], start=2):
        if is_single(d, int(car)):
            if race_type == "G3":
                score += 2.5
            elif line_level == "低":
                score += 1.5
            else:
                score -= 1.0

    # 天候
    if weather == "雨":
        if is_self_type(d, head):
            score += 1.0
        if third is not None and is_chasing_type(d, third):
            score += 1.0
    elif weather == "風強":
        if str(get_rider(d, head).get("脚質", "")) == "逃":
            score -= 4.0
            reasons.append("風強で逃げ頭抑制")
        if is_chasing_type(d, head):
            score += 2.0

    # モード
    if "混戦" in str(mode):
        if line_level == "低":
            score += break_bonus * 0.4
        if not is_same_line(d, head, second):
            score += 3.0
    elif "穴" in str(mode):
        if not is_same_line(d, head, second):
            score += 4.0
        if is_single(d, head) and race_type == "G3":
            score += 4.0
    else:
        if line_level == "高" and second is not None and is_same_line(d, head, second):
            score += 3.0

    # G3は穴・崩れ・別線を許容
    if race_type == "G3":
        if not is_same_line(d, head, second):
            score += 5.0
        if third is not None and len({head, second, third}) == 3:
            # 上位だけで固まり過ぎないよう3着穴に少し加点
            third_score = safe_float(get_rider(d, third).get("競走得点", 0))
            median = float(d["競走得点"].median()) if not d.empty else 0.0
            if third_score <= median:
                score += 3.0
                reasons.append("G3三着穴")

    # オッズ/期待値
    ticket_key = "-".join(str(x) for x in ticket)
    odds = safe_float((odds_dict or {}).get(ticket_key, 0.0), 0.0)

    if odds > 0:
        if race_type == "G3":
            if 15 <= odds <= 80:
                score += 7.0
                reasons.append("G3妙味オッズ")
            elif odds > 150:
                score -= 4.0
            elif odds < 5:
                score -= 2.0
        else:
            if 3 <= odds <= 40:
                score += 5.0
                reasons.append("現実的オッズ")
            elif 40 < odds <= 80:
                score += 1.0
            elif odds > 100:
                score -= 6.0
                reasons.append("高配当すぎ抑制")
            elif odds < 2.0:
                score -= 2.0

    # 期待値はAI評価とオッズがあればそれっぽく
    if odds > 0:
        expected_value = min(180.0, max(40.0, score * 0.75 + odds * 0.35))
    else:
        expected_value = max(40.0, min(150.0, score * 0.8))

    return {
        "ticket": ticket_key,
        "score": round(score, 2),
        "odds": round(odds, 2) if odds > 0 else 0.0,
        "expected_value": round(expected_value, 1),
        "reason": " / ".join(reasons) if reasons else "総合評価",
    }


def rank_label(score: float, expected_value: float, odds: float, race_type: str = "通常") -> str:
    if race_type == "G3":
        if expected_value >= 115 or (odds >= 25 and score >= 120):
            return "💰 期待値高"
        if score >= 145:
            return "🔥 AI推奨"
        if score >= 130:
            return "🟢 本命"
        return "🟡 穴"

    if score >= 150 and expected_value >= 95:
        return "🔥 AI推奨"
    if score >= 132:
        return "🟢 本命"
    if expected_value >= 110:
        return "💰 期待値高"
    return "🟡 穴"


def race_decision_from_candidates(
    candidates: List[Dict[str, Any]],
    df: pd.DataFrame,
    line_info: Dict[str, Any],
    race_type: str,
) -> Dict[str, Any]:
    if not candidates:
        return {
            "レース判定": "見送り",
            "的中率評価": "低",
            "レース評価点": 0,
            "判定理由": "候補が作れない",
        }

    top_scores = [safe_float(x.get("score", 0)) for x in candidates[:5]]
    avg_top = sum(top_scores) / len(top_scores)
    spread = max(top_scores) - min(top_scores) if len(top_scores) >= 2 else 0

    line_level = line_info.get("level", "中")
    singles = int((prepare_df(df)["単騎"] == 1).sum()) if df is not None else 0

    point = avg_top / 2.0
    if line_level == "高":
        point += 8
    elif line_level == "低":
        point -= 4

    if spread <= 8:
        point += 4
    elif spread >= 25:
        point -= 4

    if singles >= 2:
        point -= 3

    if race_type == "G3":
        point += 4  # G3は荒れ含みでも買い候補を残す

    point = round(max(0, min(100, point)), 1)

    if race_type == "G3":
        if point >= 68:
            decision = "買い"
            hit_label = "中"
        elif point >= 55:
            decision = "軽く買い"
            hit_label = "中穴"
        elif point >= 45:
            decision = "注意"
            hit_label = "穴"
        else:
            decision = "見送り"
            hit_label = "低"
    else:
        if point >= 72:
            decision = "買い"
            hit_label = "高"
        elif point >= 60:
            decision = "軽く買い"
            hit_label = "中"
        elif point >= 48:
            decision = "注意"
            hit_label = "低中"
        else:
            decision = "見送り"
            hit_label = "低"

    reason = f"上位平均{avg_top:.1f} / ライン信頼{line_level}({line_info.get('score')}) / {line_info.get('details')}"

    return {
        "レース判定": decision,
        "的中率評価": hit_label,
        "レース評価点": point,
        "判定理由": reason,
    }


def diversify_tickets(
    ranked: List[Dict[str, Any]],
    top_n: int,
    df: pd.DataFrame,
    race_type: str,
) -> List[Dict[str, Any]]:
    """
    同じ頭・単騎頭ばかりを抑える。
    G3は穴寄り許容なので少し緩くする。
    """
    if not ranked:
        return []

    selected = []
    head_counts = {}
    single_head_count = 0

    if race_type == "G3":
        max_same_head = max(3, math.ceil(top_n * 0.35))
        max_single_head = max(2, math.ceil(top_n * 0.25))
    else:
        max_same_head = max(2, math.ceil(top_n * 0.25))
        max_single_head = max(1, math.ceil(top_n * 0.15))

    for item in ranked:
        parts = [int(x) for x in str(item["ticket"]).split("-") if x]
        if not parts:
            continue

        head = parts[0]
        is_single_head = is_single(df, head)

        if head_counts.get(head, 0) >= max_same_head:
            continue
        if is_single_head and single_head_count >= max_single_head:
            continue

        selected.append(item)
        head_counts[head] = head_counts.get(head, 0) + 1
        if is_single_head:
            single_head_count += 1

        if len(selected) >= top_n:
            break

    if len(selected) < top_n:
        seen = {x["ticket"] for x in selected}
        for item in ranked:
            if item["ticket"] in seen:
                continue
            selected.append(item)
            seen.add(item["ticket"])
            if len(selected) >= top_n:
                break

    return selected[:top_n]


def generate_ticket_candidates(
    df: pd.DataFrame,
    ticket_type: str,
    base_scores: Dict[int, float],
    odds_dict: Dict[str, float],
    mode: str,
    weather: str,
    race_type: str,
    line_info: Dict[str, Any],
) -> List[Dict[str, Any]]:
    d = prepare_df(df)
    cars = [int(x) for x in d["車番"].tolist() if int(x) > 0]

    candidates = []
    if ticket_type == "2車単":
        perms = itertools.permutations(cars, 2)
    else:
        perms = itertools.permutations(cars, 3)

    for ticket in perms:
        item = score_ticket(
            d,
            tuple(ticket),
            base_scores,
            odds_dict,
            mode,
            weather,
            ticket_type,
            race_type,
            line_info,
        )
        candidates.append(item)

    candidates.sort(key=lambda x: (safe_float(x["score"]), safe_float(x["expected_value"])), reverse=True)
    return candidates


def generate_predictions(
    df: pd.DataFrame,
    mode: str = "通常モード",
    weather: str = "晴",
    top_n: int = 10,
    odds_dict: Dict[str, float] | None = None,
    ticket_type: str = "3連単",
    race_type: str = "通常",
) -> pd.DataFrame:
    d = prepare_df(df)

    if d.empty:
        return pd.DataFrame()

    odds_dict = odds_dict or {}
    top_n = max(1, safe_int(top_n, 10))

    # ガールズはラインをほぼ無効化
    if race_type == "ガールズ":
        d = d.copy()
        d["ライン"] = 0
        d["ライン順"] = 0
        d["単騎"] = 1

    line_info = calc_line_reliability(d)
    base_scores = calc_rider_base_scores(d, race_type=race_type)

    candidates = generate_ticket_candidates(
        d,
        ticket_type=ticket_type,
        base_scores=base_scores,
        odds_dict=odds_dict,
        mode=mode,
        weather=weather,
        race_type=race_type,
        line_info=line_info,
    )

    # 多めに見てから偏り補正
    pre_take = max(top_n * 6, 40)
    ranked = candidates[:pre_take]
    selected = diversify_tickets(ranked, top_n, d, race_type=race_type)

    race_eval = race_decision_from_candidates(selected, d, line_info, race_type)

    rows = []
    for item in selected:
        rank = rank_label(
            safe_float(item["score"]),
            safe_float(item["expected_value"]),
            safe_float(item["odds"]),
            race_type=race_type,
        )
        rows.append(
            {
                "レース判定": race_eval["レース判定"],
                "的中率評価": race_eval["的中率評価"],
                "レース評価点": race_eval["レース評価点"],
                "判定理由": race_eval["判定理由"],
                "ライン信頼度": line_info.get("level", ""),
                "ライン信頼点": line_info.get("score", 0),
                "買い目ランク": rank,
                "買い目": item["ticket"],
                "AI評価": item["score"],
                "期待値": item["expected_value"],
                "オッズ": item["odds"],
                "展開メモ": item["reason"],
            }
        )

    return pd.DataFrame(rows)

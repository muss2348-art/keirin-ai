# app.py
# -*- coding: utf-8 -*-

import re
import csv
import json
import itertools
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

from predict import auto_detect_mode, generate_predictions
from learning import apply_learning_correction, learning_summary_text
from roi_learning import apply_roi_learning, roi_learning_summary_text
from race_filter import assess_race_buyability, apply_race_buyability_to_predictions, race_buyability_summary_text
from staking import apply_staking_ai, staking_summary_text


st.set_page_config(
    page_title="競輪AI モバイル版",
    page_icon="🚴",
    layout="centered",
)

st.caption("✅ mobile ROIランク分岐版 v16 起動中（レース選別AI・見送り買い目非表示版）")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/146.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

DEFAULT_COLUMNS = [
    "車番",
    "選手名",
    "競走得点",
    "脚質",
    "ライン",
    "ライン順",
    "単騎",
]

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_PATH = SCRIPT_DIR / "log.csv"
SAVED_RACES_PATH = SCRIPT_DIR / "saved_races.json"

PREFECTURES = [
    "北海道",
    "青森", "岩手", "宮城", "秋田", "山形", "福島",
    "茨城", "栃木", "群馬", "埼玉", "千葉", "東京", "神奈川",
    "新潟", "富山", "石川", "福井",
    "山梨", "長野",
    "岐阜", "静岡", "愛知", "三重",
    "滋賀", "京都", "大阪", "兵庫", "奈良", "和歌山",
    "鳥取", "島根", "岡山", "広島", "山口",
    "徳島", "香川", "愛媛", "高知",
    "福岡", "佐賀", "長崎", "熊本", "大分", "宮崎", "鹿児島", "沖縄",
]
PREF_PATTERN = "|".join(sorted(PREFECTURES, key=len, reverse=True))


def generate_predictions_compat(
    current_df,
    detected_mode,
    weather,
    display_count,
    odds_dict,
    ticket_type,
    race_type,
):
    """predict.py が race_type 未対応でも完全版UIを壊さず予想生成する互換ラッパー。"""
    try:
        return generate_predictions(
            current_df,
            mode=detected_mode,
            weather=weather,
            top_n=display_count,
            odds_dict=odds_dict,
            ticket_type=ticket_type,
            race_type=race_type,
        )
    except TypeError as e:
        msg = str(e)
        if "race_type" in msg and "unexpected keyword" in msg:
            return generate_predictions(
                current_df,
                mode=detected_mode,
                weather=weather,
                top_n=display_count,
                odds_dict=odds_dict,
                ticket_type=ticket_type,
            )
        raise


# =========================================================
# 共通
# =========================================================
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    table = str.maketrans(
        {
            "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
            "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
            "－": "-", "ー": "-", "―": "-", "‐": "-", "ｰ": "-",
            "／": "/", "　": " ", "，": ",", "．": ".",
            "（": "(", "）": ")", "｜": "|",
            "：": ":", "\xa0": " ",
        }
    )
    s = str(s).translate(table)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


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

# =========================================================
# 発信用 厳選AI / 新ランク判定
# =========================================================
def detect_roi_column(df: pd.DataFrame):
    """
    ROI/回収率系の列を優先して探す。
    注意: 「期待値」は全買い目が似た数値になりやすいので、最後の保険にする。
    """
    if df is None or df.empty:
        return None

    preferred = [
        "ROI補正後期待値", "ROI期待値", "期待ROI", "期待回収率", "回収率", "ROI", "roi",
    ]
    for col in preferred:
        if col in df.columns:
            return col

    for col in df.columns:
        name = str(col).lower()
        if "roi" in name or "回収" in str(col):
            return col

    for col in ["期待値", "EV", "ev", "score", "スコア", "AI評価"]:
        if col in df.columns:
            return col

    for col in df.columns:
        if "期待" in str(col):
            return col
    return None


def detect_confidence_column(df: pd.DataFrame):
    """的中率・確率・信頼度系の列を自動検出する。"""
    if df is None or df.empty:
        return None
    preferred = ["的中率", "予想的中率", "信頼度", "confidence", "prob", "確率", "勝率", "AI信頼度"]
    for col in preferred:
        if col in df.columns:
            return col
    for col in df.columns:
        name = str(col).lower()
        if "confidence" in name or "prob" in name or "的中" in str(col) or "信頼" in str(col) or "確率" in str(col):
            return col
    return None


def normalize_roi_value(value) -> float:
    """ROI/期待値を100基準に寄せる。1.25なら125扱い。"""
    roi = safe_float(value, 0.0)
    if 0 < roi <= 3:
        roi *= 100.0
    return roi


def normalize_confidence_value(value) -> float:
    """的中率/信頼度を0〜100に寄せる。0.35なら35扱い。"""
    conf = safe_float(value, 0.0)
    if 0 < conf <= 1:
        conf *= 100.0
    return conf


def _get_odds_value(row) -> float:
    for col in ["オッズ", "odds"]:
        if col in row.index:
            return safe_float(row.get(col), 0.0)
    return 0.0


def _publish_rank_label(position: int, total: int, roi: float, confidence: float, odds: float) -> str:
    """
    発信用ランク。
    熱🔥: 的中率もROIも強い本命寄り
    堅: 的中率優先で堅い
    穴: オッズがありつつROI・的中率も最低ライン以上
    抑え: 中間〜保険
    """
    rate = position / max(total - 1, 1) if total > 1 else 0.0

    if confidence >= 38 and roi >= 103 and (odds <= 0 or odds <= 18):
        return "熱🔥"
    if confidence >= 33 and (odds <= 0 or odds <= 12):
        return "堅"
    if confidence >= 24 and roi >= 115 and odds >= 8:
        return "穴"

    # 確率列が無い/弱い場合でも、順位で最低限散らす
    if confidence <= 0:
        if rate <= 0.12 and roi >= 112:
            return "熱🔥"
        if rate <= 0.35 and roi >= 105:
            return "堅"
        if roi >= 118 and odds >= 8:
            return "穴"
        return "抑え"

    if rate <= 0.15 and roi >= 108:
        return "熱🔥"
    if rate <= 0.40 and confidence >= 28:
        return "堅"
    if roi >= 110 and odds >= 7:
        return "穴"
    return "抑え"


def apply_roi_ticket_ranking(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    ROI学習結果・的中率・オッズを使って、発信用ランクへ再判定する。
    旧表示: AI推奨/期待値高/本命/穴/抑え
    新表示: 熱🔥/堅/穴/抑え
    """
    if pred_df is None or pred_df.empty:
        return pred_df

    out = pred_df.copy()
    roi_col = detect_roi_column(out)
    confidence_col = detect_confidence_column(out)

    roi_values, conf_values, odds_values, scores = [], [], [], []
    for _, row in out.iterrows():
        roi = normalize_roi_value(row.get(roi_col, 0.0)) if roi_col else 0.0
        conf = normalize_confidence_value(row.get(confidence_col, 0.0)) if confidence_col else 0.0
        odds = _get_odds_value(row)
        current_ev = normalize_roi_value(row.get("期待値", 0.0)) if "期待値" in out.columns else roi

        roi_values.append(roi)
        conf_values.append(conf)
        odds_values.append(odds)

        score = roi
        if conf > 0:
            score += conf * 1.2
        if current_ev > 0 and current_ev != roi:
            score += current_ev * 0.25
        if odds > 0:
            # 高すぎる穴は発信用では少し減点、ほどよいオッズを加点
            score += min(odds, 30) * 0.12
            if odds >= 45:
                score -= 8
        scores.append(score)

    total = len(out)
    order = sorted(range(total), key=lambda i: scores[i], reverse=True)
    position_map = {idx: pos for pos, idx in enumerate(order)}

    ranks = []
    for i in range(total):
        pos = position_map[i]
        ranks.append(_publish_rank_label(pos, total, roi_values[i], conf_values[i], odds_values[i]))

    if total >= 4 and len(set(ranks)) == 1:
        # 全部同じ表示になるのを防ぐ
        for idx, pos in position_map.items():
            if pos == 0:
                ranks[idx] = "熱🔥"
            elif pos <= max(1, int(total * 0.35)):
                ranks[idx] = "堅"
            elif pos <= max(2, int(total * 0.65)):
                ranks[idx] = "穴" if odds_values[idx] >= 7 and roi_values[idx] >= 105 else "抑え"
            else:
                ranks[idx] = "抑え"

    out["買い目ランク"] = ranks
    thick_map = {"熱🔥": 3.0, "堅": 2.1, "穴": 1.7, "抑え": 1.0}
    out["厚張り指数"] = [
        round(thick_map.get(ranks[i], 1.0) + min(max((roi_values[i] - 100.0) / 90.0, 0.0), 0.8), 2)
        for i in range(total)
    ]

    rank_order = {"熱🔥": 0, "堅": 1, "穴": 2, "抑え": 3}
    out["_rank_order"] = out["買い目ランク"].map(rank_order).fillna(9)
    out["_publish_score"] = scores
    out = out.sort_values(["_rank_order", "_publish_score"], ascending=[True, False]).drop(columns=["_rank_order", "_publish_score"])
    return out.reset_index(drop=True)


# =========================================================
# 厳選AI（買い目フィルタ＋レース判定を厳しめにする）
# =========================================================
def calc_selection_score(row, roi_col=None, confidence_col=None) -> float:
    """買い目を残す/切るための総合スコア。ROI・的中率・オッズ・元期待値を合わせる。"""
    roi = normalize_roi_value(row.get(roi_col, 0.0)) if roi_col else 0.0
    conf = normalize_confidence_value(row.get(confidence_col, 0.0)) if confidence_col else 0.0
    ev = normalize_roi_value(row.get("期待値", 0.0)) if "期待値" in row.index else roi
    odds = _get_odds_value(row)

    score = 0.0
    score += roi * 1.0
    score += conf * 1.15
    if ev > 0 and ev != roi:
        score += ev * 0.20
    if odds > 0:
        score += min(odds, 30.0) * 0.15
        if odds >= 45:
            score -= 10
    return round(score, 3)


def _strict_race_decision(filtered: pd.DataFrame, before_count: int, min_roi: float) -> dict:
    if filtered is None or filtered.empty:
        return {
            "decision": "見送り",
            "confidence": 0,
            "message": "厳選AI: 発信用に残せる買い目が無いため見送り推奨です。",
        }

    top = filtered.head(5).copy()
    roi_col = detect_roi_column(top)
    conf_col = detect_confidence_column(top)

    roi_vals = [normalize_roi_value(r.get(roi_col, 0.0)) if roi_col else normalize_roi_value(r.get("期待値", 0.0)) for _, r in top.iterrows()]
    conf_vals = [normalize_confidence_value(r.get(conf_col, 0.0)) if conf_col else 0.0 for _, r in top.iterrows()]
    odds_vals = [_get_odds_value(r) for _, r in top.iterrows()]
    ranks = top["買い目ランク"].astype(str).tolist() if "買い目ランク" in top.columns else []

    avg_roi = sum(roi_vals) / len(roi_vals) if roi_vals else 0.0
    max_roi = max(roi_vals) if roi_vals else 0.0
    avg_conf = sum(conf_vals) / len(conf_vals) if conf_vals else 0.0
    max_conf = max(conf_vals) if conf_vals else 0.0
    avg_odds = sum([x for x in odds_vals if x > 0]) / len([x for x in odds_vals if x > 0]) if any(x > 0 for x in odds_vals) else 0.0

    hot_count = ranks.count("熱🔥")
    solid_count = ranks.count("堅")
    hole_count = ranks.count("穴")

    confidence = 0
    confidence += min(max_roi, 150) * 0.22
    confidence += min(avg_roi, 140) * 0.18
    confidence += max_conf * 0.70
    confidence += avg_conf * 0.45
    confidence += hot_count * 12
    confidence += solid_count * 6
    confidence -= hole_count * 2
    if avg_odds >= 35:
        confidence -= 10
    if len(filtered) > max(12, before_count * 0.75):
        confidence -= 5
    confidence = int(max(0, min(100, round(confidence))))

    # 発信用なのでかなり厳しめ。勝負は熱ありが基本。
    if confidence >= 70 and (hot_count >= 1 or solid_count >= 2) and avg_roi >= max(100, float(min_roi) - 3):
        return {
            "decision": "勝負",
            "confidence": confidence,
            "message": f"厳選AI: {before_count}点→{len(filtered)}点。🔥勝負候補です。信頼度{confidence}% / 熱{hot_count}点",
        }
    if confidence >= 56 and (hot_count >= 1 or solid_count >= 1 or hole_count >= 1) and avg_roi >= 96:
        return {
            "decision": "厳選候補",
            "confidence": confidence,
            "message": f"厳選AI: {before_count}点→{len(filtered)}点。△軽め候補です。信頼度{confidence}%",
        }
    return {
        "decision": "見送り",
        "confidence": confidence,
        "message": f"厳選AI: {before_count}点→{len(filtered)}点。見送り推奨です。信頼度{confidence}% / 発信用には弱め",
    }


def apply_strict_selection_ai(
    pred_df: pd.DataFrame,
    max_count: int = 12,
    min_roi: float = 110.0,
    min_score: float = 0.0,
    keep_min: int = 2,
):
    """
    厳選AI。
    1) ROI/期待値/的中率で総合スコアを作る
    2) ROIが低すぎる買い目を削る
    3) 発信用に点数を絞る
    4) レース自体の勝負/厳選候補/見送りを厳しめに返す
    """
    info = {
        "enabled": True,
        "before_count": 0,
        "after_count": 0,
        "removed_count": 0,
        "min_roi": float(min_roi),
        "max_count": int(max_count),
        "avg_roi_before": 0.0,
        "avg_roi_after": 0.0,
        "decision": "見送り",
        "confidence": 0,
        "message": "",
    }

    if pred_df is None or pred_df.empty:
        info["message"] = "厳選AI: 買い目がありません。"
        return pred_df, info

    out = apply_roi_ticket_ranking(pred_df.copy())
    info["before_count"] = len(out)

    roi_col = detect_roi_column(out)
    confidence_col = detect_confidence_column(out)

    out["厳選スコア"] = out.apply(
        lambda r: calc_selection_score(r, roi_col=roi_col, confidence_col=confidence_col),
        axis=1,
    )
    out["厳選ROI"] = out.apply(
        lambda r: normalize_roi_value(r.get(roi_col, 0.0)) if roi_col else normalize_roi_value(r.get("期待値", 0.0)),
        axis=1,
    )

    out["厳選ROI"] = pd.to_numeric(out["厳選ROI"], errors="coerce").fillna(0.0)
    out["厳選スコア"] = pd.to_numeric(out["厳選スコア"], errors="coerce").fillna(0.0)
    info["avg_roi_before"] = round(float(out["厳選ROI"].mean()), 1) if len(out) else 0.0

    sorted_out = out.sort_values(["厳選スコア", "厳選ROI"], ascending=False).reset_index(drop=True)

    # ROIが取れている場合だけ低ROIを削る。発信用なので初期値はやや厳しめ。
    if (sorted_out["厳選ROI"] > 0).any():
        filtered = sorted_out[sorted_out["厳選ROI"] >= float(min_roi)].copy()
    else:
        filtered = sorted_out.copy()

    if float(min_score) > 0:
        filtered = filtered[filtered["厳選スコア"] >= float(min_score)].copy()

    keep_min = max(1, int(keep_min))
    if filtered.empty:
        # 完全ゼロにすると保存・確認ができないので、上位を少しだけ残して見送り判定にする
        filtered = sorted_out.head(min(keep_min, len(sorted_out))).copy()
    elif len(filtered) < keep_min and len(sorted_out) >= keep_min:
        comeback = sorted_out.head(keep_min).copy()
        if "買い目" in filtered.columns:
            filtered = pd.concat([filtered, comeback], ignore_index=True).drop_duplicates(subset=["買い目"])
        else:
            filtered = pd.concat([filtered, comeback], ignore_index=True).drop_duplicates()

    max_count = max(1, int(max_count))
    filtered = filtered.sort_values(["厳選スコア", "厳選ROI"], ascending=False).head(max_count).reset_index(drop=True)
    filtered = apply_roi_ticket_ranking(filtered)

    info["after_count"] = len(filtered)
    info["removed_count"] = max(0, info["before_count"] - info["after_count"])
    info["avg_roi_after"] = round(float(filtered["厳選ROI"].mean()), 1) if len(filtered) and "厳選ROI" in filtered.columns else 0.0

    decision = _strict_race_decision(filtered, before_count=info["before_count"], min_roi=min_roi)
    info.update(decision)
    return filtered, info


# =========================================================
# 予想スタイル別フィルタ（的中率重視 / 回収率重視）
# =========================================================
def get_prediction_style_settings(prediction_style: str) -> dict:
    """予想スタイルごとの点数・ROI・ランク配分。"""
    style = str(prediction_style or "的中率重視").strip()
    if style == "回収率重視":
        return {
            "style": "回収率重視",
            "strict_min_roi": 100,
            "strict_keep_min": 4,
            "default_max_count": 12,
            "rank_limits": {"熱🔥": 2, "堅": 3, "穴": 5, "抑え": 3},
            "hole_min": 2,
            "gate_confidence_min": 58,
            "message": "回収率重視: 穴・期待値を多めに残しつつ、無謀な穴は削ります。",
        }
    return {
        "style": "的中率重視",
        "strict_min_roi": 102,
        "strict_keep_min": 3,
        "default_max_count": 9,
        "rank_limits": {"熱🔥": 3, "堅": 4, "穴": 2, "抑え": 2},
        "hole_min": 1,
        "gate_confidence_min": 62,
        "message": "的中率重視: 熱🔥・堅を中心に、夢のある穴を少しだけ混ぜます。",
    }


def _style_sort_columns(df: pd.DataFrame):
    sort_cols = []
    ascending = []
    if "厳選スコア" in df.columns:
        sort_cols.append("厳選スコア")
        ascending.append(False)
    if "厳選ROI" in df.columns:
        sort_cols.append("厳選ROI")
        ascending.append(False)
    if "期待値" in df.columns:
        sort_cols.append("期待値")
        ascending.append(False)
    if "AI評価" in df.columns:
        sort_cols.append("AI評価")
        ascending.append(False)
    if not sort_cols:
        sort_cols = [df.columns[0]]
        ascending = [True]
    return sort_cols, ascending


def apply_prediction_style_filter(
    pred_df: pd.DataFrame,
    prediction_style: str = "的中率重視",
    max_count: int = 8,
):
    """
    予想スタイル別に買い目を残す。
    的中率重視: 熱🔥・堅中心、穴は少しだけ。
    回収率重視: 穴・期待値を多め、ただし熱🔥・堅も残す。
    """
    info = {
        "style": prediction_style,
        "before_count": 0,
        "after_count": 0,
        "message": "",
    }
    if pred_df is None or pred_df.empty:
        info["message"] = f"{prediction_style}: 買い目がありません。"
        return pred_df, info

    settings = get_prediction_style_settings(prediction_style)
    out = apply_roi_ticket_ranking(pred_df.copy())
    info["before_count"] = len(out)

    if "買い目ランク" not in out.columns:
        out["買い目ランク"] = "抑え"

    sort_cols, ascending = _style_sort_columns(out)
    out = out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    max_count = max(1, int(max_count))
    limits = dict(settings.get("rank_limits", {}))

    selected_parts = []
    used_keys = set()

    def row_key(row):
        if "買い目" in row.index:
            return str(row.get("買い目", ""))
        return str(row.name)

    # まずランクごとの上限に沿って選ぶ
    for rank_label in ["熱🔥", "堅", "穴", "抑え"]:
        limit = int(limits.get(rank_label, 0))
        if limit <= 0:
            continue
        part = out[out["買い目ランク"].astype(str) == rank_label].copy()
        if part.empty:
            continue
        part = part.sort_values(sort_cols, ascending=ascending).head(limit)
        rows = []
        for _, r in part.iterrows():
            k = row_key(r)
            if k not in used_keys:
                used_keys.add(k)
                rows.append(r)
        if rows:
            selected_parts.append(pd.DataFrame(rows))

    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame(columns=out.columns)

    # 的中率重視でも穴を最低1点、回収率重視なら最低2点入れやすくする
    hole_min = int(settings.get("hole_min", 0))
    current_holes = int((selected.get("買い目ランク", pd.Series(dtype=str)).astype(str) == "穴").sum()) if not selected.empty else 0
    if hole_min > 0 and current_holes < hole_min and len(out) >= 5:
        hole_candidates = out[out["買い目ランク"].astype(str) == "穴"].copy()
        if not hole_candidates.empty:
            hole_candidates = hole_candidates.sort_values(sort_cols, ascending=ascending)
            add_rows = []
            for _, r in hole_candidates.iterrows():
                if len(add_rows) >= (hole_min - current_holes):
                    break
                k = row_key(r)
                if k not in used_keys:
                    used_keys.add(k)
                    add_rows.append(r)
            if add_rows:
                selected = pd.concat([selected, pd.DataFrame(add_rows)], ignore_index=True)

    # まだ少ない場合は全体上位から補充
    if len(selected) < min(max_count, len(out)):
        add_rows = []
        for _, r in out.iterrows():
            if len(selected) + len(add_rows) >= min(max_count, len(out)):
                break
            k = row_key(r)
            if k not in used_keys:
                used_keys.add(k)
                add_rows.append(r)
        if add_rows:
            selected = pd.concat([selected, pd.DataFrame(add_rows)], ignore_index=True)

    selected = selected.head(max_count).reset_index(drop=True)
    selected = apply_roi_ticket_ranking(selected)
    selected["予想スタイル"] = settings["style"]

    info["after_count"] = len(selected)
    rank_counts = selected["買い目ランク"].astype(str).value_counts().to_dict() if not selected.empty and "買い目ランク" in selected.columns else {}
    info["message"] = (
        f"{settings['style']}: {info['before_count']}点→{info['after_count']}点。"
        f"熱🔥{rank_counts.get('熱🔥', 0)} / 堅{rank_counts.get('堅', 0)} / "
        f"穴{rank_counts.get('穴', 0)} / 抑え{rank_counts.get('抑え', 0)}"
    )
    return selected, info


def widget_key(name: str, idx: int) -> str:
    ver = st.session_state.get("widget_ver", 0)
    return f"{name}_{idx}_v{ver}"


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_ticket(ticket: str) -> str:
    s = normalize_text(ticket)
    s = s.replace(" ", "")
    s = re.sub(r"[^0-9\-]", "", s)
    return s


def format_saved_result(item: dict) -> str:
    result = item.get("result", {}) or {}
    return str(result.get("result_text", "")).strip()


def format_saved_hit_ticket(item: dict) -> str:
    result = item.get("result", {}) or {}
    return str(result.get("hit_ticket", "")).strip()


def is_valid_player_name(name: str) -> bool:
    """選手名としてあり得る文字列だけ通す。余計な本文・コメント欄の誤取得を減らす。"""
    name = normalize_text(name)
    if not name:
        return False
    if re.fullmatch(r"\d+", name):
        return False

    ng_words = set(PREFECTURES + [
        "勝率", "本命", "対抗", "単穴", "連下", "単騎で", "コメント", "ギヤ", "倍率",
        "ライン", "並び", "予想", "出走表", "人気順", "払戻", "結果", "レース",
        "オッズ", "前検", "成績", "基本情報", "直近成績",
    ])
    if name in ng_words:
        return False

    if not re.fullmatch(r"[一-龥ぁ-んァ-ヶ々]{2,8}", name):
        return False
    return True


# =========================================================
# 金額配分
# =========================================================
def rank_base_amount(rank_label: str, unit_bet: int) -> int:
    unit = max(100, int(unit_bet))
    if rank_label == "熱🔥":
        return unit * 3
    if rank_label == "堅":
        return unit * 2
    if rank_label == "穴":
        return unit * 2
    return unit


def apply_rank_based_amounts(pred_df: pd.DataFrame, unit_bet: int) -> pd.DataFrame:
    if pred_df is None or pred_df.empty:
        return pred_df

    # ここで必ずROI連動ランクを再判定する
    out = apply_roi_ticket_ranking(pred_df)
    total_budget = int(unit_bet) * len(out)

    if "買い目ランク" not in out.columns:
        out["買い目ランク"] = "抑え"

    if "厚張り指数" not in out.columns:
        out["厚張り指数"] = 1.0

    rank_weight = {
        "熱🔥": 3.0,
        "堅": 2.1,
        "穴": 1.7,
        "抑え": 1.0,
    }

    weights = []
    for _, row in out.iterrows():
        rank_label = str(row.get("買い目ランク", "抑え"))
        thick_score = safe_float(row.get("厚張り指数", 1.0), 1.0)
        weight = rank_weight.get(rank_label, 1.0) * max(thick_score, 0.1)
        weights.append(weight)

    total_weight = sum(weights)
    unit = max(100, int(unit_bet))

    if total_weight <= 0:
        out["購入金額"] = unit
        return out

    amounts = []
    for weight in weights:
        raw_amount = total_budget * (weight / total_weight)
        rounded_amount = int(raw_amount // 100) * 100
        rounded_amount = max(rounded_amount, unit)
        amounts.append(rounded_amount)

    # 予算超過時は金額が大きいところから100円ずつ削る
    while sum(amounts) > total_budget and max(amounts) > unit:
        max_index = amounts.index(max(amounts))
        amounts[max_index] -= 100

    # 予算に余りがある場合は厚張り指数が高い順に100円ずつ足す
    diff = total_budget - sum(amounts)
    if diff >= 100 and len(amounts) > 0:
        order = sorted(range(len(amounts)), key=lambda i: weights[i], reverse=True)
        idx = 0
        while diff >= 100 and order:
            amounts[order[idx % len(order)]] += 100
            diff -= 100
            idx += 1

    out["購入金額"] = [int(x) for x in amounts]

    ev_num = pd.to_numeric(out.get("期待値", 0), errors="coerce").fillna(0)
    out["期待回収額(目安)"] = (ev_num / 100.0 * out["購入金額"]).round(0)

    return out


# =========================================================
# 的中判定
# =========================================================
def judge_hit(ticket_type: str, pred_df: pd.DataFrame, result_1: str, result_2: str, result_3: str):
    if pred_df is None or pred_df.empty:
        return {
            "status_label": "未結果",
            "hit_any": False,
            "hit_ticket": "",
            "result_text": "",
        }

    r1 = str(result_1).strip()
    r2 = str(result_2).strip()
    r3 = str(result_3).strip()

    if not r1 or not r2:
        return {
            "status_label": "未結果",
            "hit_any": False,
            "hit_ticket": "",
            "result_text": "",
        }

    if ticket_type == "2車単":
        result_ticket = f"{r1}-{r2}"
    else:
        if not r3:
            return {
                "status_label": "未結果",
                "hit_any": False,
                "hit_ticket": "",
                "result_text": "",
            }
        result_ticket = f"{r1}-{r2}-{r3}"

    tickets = pred_df["買い目"].astype(str).tolist() if "買い目" in pred_df.columns else []
    hit_any = result_ticket in tickets

    return {
        "status_label": "的中" if hit_any else "不的中",
        "hit_any": hit_any,
        "hit_ticket": result_ticket if hit_any else "",
        "result_text": result_ticket,
    }


# =========================================================
# 回収率集計
# =========================================================
def load_log_df() -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(LOG_PATH, encoding="utf-8-sig")
    except Exception:
        try:
            df = pd.read_csv(LOG_PATH, encoding="utf-8")
        except Exception:
            return pd.DataFrame()

    if df.empty:
        return df

    for col in ["購入金額", "オッズ", "期待値", "期待回収額(目安)"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    for col in ["保存日時", "レース名", "券種", "モード", "天候", "判定", "結果", "買い目", "レース種別"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    return df


def repair_log_purchase_amounts() -> None:
    """
    Streamlit Cloud/mobile用のログ修復。
    既存ログに購入金額が0/空で残っているとROI学習が「投資0円」と判断するため、
    買い目がある行は最低100円に補正してlog.csvへ書き戻す。
    """
    if not LOG_PATH.exists():
        return

    try:
        df = pd.read_csv(LOG_PATH, encoding="utf-8-sig")
    except Exception:
        try:
            df = pd.read_csv(LOG_PATH, encoding="utf-8")
        except Exception:
            return

    if df is None or df.empty:
        return

    # 必須列が無い古い/壊れたログは無理に触らない
    if "買い目" not in df.columns:
        return

    if "購入金額" not in df.columns:
        df["購入金額"] = 0

    amount = pd.to_numeric(df["購入金額"], errors="coerce").fillna(0)
    ticket_exists = df["買い目"].fillna("").astype(str).str.strip() != ""
    target = ticket_exists & (amount <= 0)

    if target.any():
        amount = amount.copy()
        amount.loc[target] = 100
        df["購入金額"] = amount.astype(int)

        if "期待回収額(目安)" not in df.columns:
            df["期待回収額(目安)"] = 0
        df["期待回収額(目安)"] = pd.to_numeric(df["期待回収額(目安)"], errors="coerce").fillna(0)

        df.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")


def ensure_prediction_amounts(pred_df: pd.DataFrame, unit_bet: int = 100) -> pd.DataFrame:
    """買い目データに購入金額を必ず付ける。ROI学習の投資0円防止用。"""
    if pred_df is None or pred_df.empty:
        return pred_df

    out = pred_df.copy()
    if "購入金額" not in out.columns:
        out = apply_rank_based_amounts(out, unit_bet=unit_bet)

    if "購入金額" not in out.columns:
        out["購入金額"] = unit_bet

    out["購入金額"] = pd.to_numeric(out["購入金額"], errors="coerce").fillna(0)
    out.loc[out["購入金額"] <= 0, "購入金額"] = max(100, int(unit_bet))
    out["購入金額"] = out["購入金額"].astype(int)

    if "期待回収額(目安)" not in out.columns:
        out["期待回収額(目安)"] = 0
    out["期待回収額(目安)"] = pd.to_numeric(out["期待回収額(目安)"], errors="coerce").fillna(0)

    return out


def summarize_log_df(log_df: pd.DataFrame):
    if log_df is None or log_df.empty:
        return {
            "race_count": 0,
            "result_saved_race_count": 0,
            "hit_race_count": 0,
            "hit_rate": 0.0,
            "total_invest": 0,
            "total_return": 0,
            "recovery_rate": 0.0,
            "by_ticket_type": pd.DataFrame(),
            "by_mode": pd.DataFrame(),
            "by_weather": pd.DataFrame(),
            "by_race_type": pd.DataFrame(),
            "recent_races": pd.DataFrame(),
        }

    work = log_df.copy()

    for c in ["レース名", "券種", "モード", "天候", "結果", "レース種別"]:
        if c not in work.columns:
            work[c] = ""

    work["race_key"] = (
        work["レース名"].astype(str) + " | " +
        work["券種"].astype(str) + " | " +
        work["モード"].astype(str) + " | " +
        work["天候"].astype(str) + " | " +
        work["レース種別"].astype(str) + " | " +
        work["結果"].astype(str)
    )

    race_summary = (
        work.groupby("race_key", as_index=False)
        .agg(
            保存日時=("保存日時", "max"),
            レース名=("レース名", "first"),
            券種=("券種", "first"),
            モード=("モード", "first"),
            天候=("天候", "first"),
            レース種別=("レース種別", "first"),
            結果=("結果", "first"),
            判定=("判定", "first"),
            投資額=("購入金額", "sum"),
        )
    )

    return_map = []
    for _, row in race_summary.iterrows():
        race_rows = work[work["race_key"] == row["race_key"]].copy()
        hit_rows = race_rows[race_rows["判定"] == "的中"].copy()

        if hit_rows.empty:
            return_map.append(0)
            continue

        hit_rows["払戻候補"] = hit_rows["購入金額"] * hit_rows["オッズ"]
        return_map.append(float(hit_rows["払戻候補"].max()))

    race_summary["払戻額"] = return_map

    race_count = len(race_summary)
    result_saved_race_count = int((race_summary["結果"].astype(str).str.strip() != "").sum())
    hit_race_count = int((race_summary["判定"] == "的中").sum())
    hit_rate = round((hit_race_count / result_saved_race_count * 100.0), 1) if result_saved_race_count > 0 else 0.0

    total_invest = int(race_summary["投資額"].sum())
    total_return = int(round(race_summary["払戻額"].sum()))
    recovery_rate = round((total_return / total_invest * 100.0), 1) if total_invest > 0 else 0.0

    def make_group_summary(base_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
        if base_df.empty or group_col not in base_df.columns:
            return pd.DataFrame()

        g = (
            base_df.groupby(group_col, as_index=False)
            .agg(
                レース数=("race_key", "count"),
                結果保存数=("結果", lambda x: int((x.astype(str).str.strip() != "").sum())),
                的中数=("判定", lambda x: int((x == "的中").sum())),
                投資額=("投資額", "sum"),
                払戻額=("払戻額", "sum"),
            )
        )

        g["的中率(%)"] = g.apply(
            lambda r: round((r["的中数"] / r["結果保存数"] * 100.0), 1) if r["結果保存数"] > 0 else 0.0,
            axis=1,
        )
        g["回収率(%)"] = g.apply(
            lambda r: round((r["払戻額"] / r["投資額"] * 100.0), 1) if r["投資額"] > 0 else 0.0,
            axis=1,
        )

        return g.sort_values(["回収率(%)", "的中率(%)", "レース数"], ascending=False).reset_index(drop=True)

    recent_cols = ["保存日時", "レース名", "券種", "モード", "天候", "レース種別", "結果", "判定", "投資額", "払戻額"]
    recent_races = race_summary.sort_values("保存日時", ascending=False)[recent_cols].head(20).reset_index(drop=True)

    return {
        "race_count": race_count,
        "result_saved_race_count": result_saved_race_count,
        "hit_race_count": hit_race_count,
        "hit_rate": hit_rate,
        "total_invest": total_invest,
        "total_return": total_return,
        "recovery_rate": recovery_rate,
        "by_ticket_type": make_group_summary(race_summary, "券種"),
        "by_mode": make_group_summary(race_summary, "モード"),
        "by_weather": make_group_summary(race_summary, "天候"),
        "by_race_type": make_group_summary(race_summary, "レース種別"),
        "recent_races": recent_races,
    }


# =========================================================
# 保存JSON
# =========================================================
def ensure_saved_races_file():
    if not SAVED_RACES_PATH.exists():
        with open(SAVED_RACES_PATH, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)


def load_saved_races() -> list:
    ensure_saved_races_file()
    try:
        with open(SAVED_RACES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def write_saved_races(data: list):
    with open(SAVED_RACES_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_race_record(record: dict):
    data = load_saved_races()
    data.insert(0, record)
    write_saved_races(data)


def update_saved_race(saved_id: str, updates: dict) -> bool:
    data = load_saved_races()
    ok = False

    for i, item in enumerate(data):
        if item.get("id") == saved_id:
            item.update(updates)
            item["updated_at"] = now_str()
            data[i] = item
            ok = True
            break

    if ok:
        write_saved_races(data)
    return ok


def delete_saved_race(saved_id: str) -> bool:
    data = load_saved_races()
    before = len(data)
    data = [x for x in data if x.get("id") != saved_id]
    if len(data) != before:
        write_saved_races(data)
        return True
    return False


def get_saved_race(saved_id: str):
    for item in load_saved_races():
        if item.get("id") == saved_id:
            return item
    return None


def saved_race_status_label(item: dict) -> str:
    if not item.get("result_saved", False):
        return "未結果"
    return item.get("hit_status", "未結果")


def saved_race_label(item: dict) -> str:
    race_name = item.get("race_name", "") or "(名称未設定)"
    created_at = item.get("created_at", "")
    mode = item.get("mode", "")
    ticket_type = item.get("ticket_type", "3連単")
    race_type = item.get("race_type", "通常")
    result_saved = saved_race_status_label(item)
    result_text = format_saved_result(item)

    if result_text:
        return f"{created_at} | {race_name} | {race_type} | {ticket_type} | {mode} | {result_saved} | 結果 {result_text}"
    return f"{created_at} | {race_name} | {race_type} | {ticket_type} | {mode} | {result_saved}"


# =========================================================
# 状態管理
# =========================================================
def init_state(num_riders: int = 7):
    rows = []
    for i in range(1, num_riders + 1):
        rows.append(
            {
                "車番": i,
                "選手名": "",
                "競走得点": 0.0,
                "脚質": "",
                "ライン": 0,
                "ライン順": 0,
                "単騎": 0,
            }
        )

    st.session_state["race_rows"] = rows
    st.session_state["num_riders"] = num_riders
    st.session_state["lineup_string"] = ""
    st.session_state["message"] = ""
    st.session_state["pred_df"] = None
    st.session_state["player_debug_info"] = None
    st.session_state["odds_debug_info"] = None
    st.session_state["lineup_debug_info"] = None
    st.session_state["odds_dict"] = {}
    st.session_state["ticket_type"] = st.session_state.get("ticket_type", "3連単")
    st.session_state["race_type"] = st.session_state.get("race_type", "通常")
    st.session_state["widget_ver"] = st.session_state.get("widget_ver", 0) + 1


def get_df() -> pd.DataFrame:
    rows = st.session_state.get("race_rows", [])
    if not rows:
        init_state(7)
        rows = st.session_state.get("race_rows", [])

    df = pd.DataFrame(rows)

    for c in DEFAULT_COLUMNS:
        if c not in df.columns:
            df[c] = 0.0 if c == "競走得点" else ""

    df["車番"] = pd.to_numeric(df["車番"], errors="coerce").fillna(0).astype(int)
    df["競走得点"] = pd.to_numeric(df["競走得点"], errors="coerce").fillna(0.0)
    df["ライン"] = pd.to_numeric(df["ライン"], errors="coerce").fillna(0).astype(int)
    df["ライン順"] = pd.to_numeric(df["ライン順"], errors="coerce").fillna(0).astype(int)
    df["単騎"] = pd.to_numeric(df["単騎"], errors="coerce").fillna(0).astype(int)

    return df[DEFAULT_COLUMNS].copy()


def set_df(df: pd.DataFrame):
    st.session_state["race_rows"] = df[DEFAULT_COLUMNS].to_dict(orient="records")
    st.session_state["num_riders"] = len(df)


def restore_saved_race_to_session(item: dict):
    rows = item.get("race_rows", [])
    num_riders = item.get("num_riders", 7)

    if not rows:
        init_state(num_riders)
    else:
        st.session_state["race_rows"] = rows
        st.session_state["num_riders"] = num_riders
        st.session_state["widget_ver"] = st.session_state.get("widget_ver", 0) + 1

    st.session_state["race_name"] = item.get("race_name", "")
    st.session_state["last_url"] = item.get("url", "")
    st.session_state["lineup_string"] = item.get("lineup_string", "")
    st.session_state["pred_df"] = pd.DataFrame(item.get("pred_rows", [])) if item.get("pred_rows") else None
    st.session_state["odds_dict"] = item.get("odds_dict", {})
    st.session_state["ticket_type"] = item.get("ticket_type", "3連単")
    st.session_state["race_type"] = item.get("race_type", "通常")
    st.session_state["message"] = f"保存レースを読込: {item.get('race_name', '')}"


# =========================================================
# 並び処理
# =========================================================
def parse_lineup_groups(lineup_text: str):
    s = normalize_text(lineup_text)
    if not s:
        return []

    s = s.replace("|", "/").replace("・", "/").replace(">", "/").replace("→", "/")
    s = s.replace(",", "/").replace(";", "/")

    raw_groups = re.split(r"\s*/\s*", s)
    groups = []

    for g in raw_groups:
        g = normalize_text(g)
        if not g:
            continue
        nums = re.findall(r"[1-9]", g)
        if nums:
            groups.append([int(x) for x in nums])

    flat = list(itertools.chain.from_iterable(groups))
    if not flat or len(set(flat)) != len(flat):
        return []

    return groups


def groups_to_lineup_string(groups):
    return " / ".join("-".join(str(x) for x in g) for g in groups if g)


def apply_lineup_to_df(df: pd.DataFrame, lineup_text: str) -> pd.DataFrame:
    groups = parse_lineup_groups(lineup_text)
    if not groups:
        raise ValueError("並び文字列を解釈できませんでした。")

    flat = list(itertools.chain.from_iterable(groups))
    riders = sorted(df["車番"].astype(int).tolist())

    if set(flat) != set(riders):
        raise ValueError(f"並びの車番 {sorted(flat)} と出走表の車番 {riders} が一致しません。")

    out = df.copy()
    out["ライン"] = 0
    out["ライン順"] = 0
    out["単騎"] = 0

    line_id = 1
    for g in groups:
        if len(g) == 1:
            car = g[0]
            out.loc[out["車番"] == car, "ライン"] = 0
            out.loc[out["車番"] == car, "ライン順"] = 1
            out.loc[out["車番"] == car, "単騎"] = 1
        else:
            for order, car in enumerate(g, start=1):
                out.loc[out["車番"] == car, "ライン"] = line_id
                out.loc[out["車番"] == car, "ライン順"] = order
                out.loc[out["車番"] == car, "単騎"] = 0
            line_id += 1

    return out


# =========================================================
# URL候補
# =========================================================
def build_lineup_candidate_urls(url: str):
    u = normalize_text(url).rstrip("/")
    candidates = [u]

    if "/odds/" in u:
        candidates.append(u.replace("/odds/", "/racecard/"))

    if "/racecard/" not in u and "/odds/" not in u and "/keirin/" in u:
        candidates.append(u.replace("/keirin/", "/keirin/racecard/"))

    uniq = []
    seen = set()
    for x in candidates:
        if x and x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def build_player_candidate_urls(url: str):
    return build_lineup_candidate_urls(url)


def build_odds_candidate_urls(url: str):
    u = normalize_text(url).rstrip("/")
    candidates = [u]

    if "/racecard/" in u:
        candidates.append(u.replace("/racecard/", "/odds/"))

    if "/odds/" not in u and "/racecard/" not in u and "/keirin/" in u:
        candidates.append(u.replace("/keirin/", "/keirin/odds/"))

    uniq = []
    seen = set()
    for x in candidates:
        if x and x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


# =========================================================
# HTTP / HTML
# =========================================================
def fetch_response(url: str):
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or r.encoding
    return r


def get_html_text_title(url: str):
    r = fetch_response(url)
    html = r.text
    soup = BeautifulSoup(html, "html.parser")
    text = normalize_text(soup.get_text(" ", strip=True))
    title = normalize_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
    return {
        "url": url,
        "status_code": r.status_code,
        "html": html,
        "text": text,
        "title": title,
    }


# =========================================================
# 並び取得
# =========================================================
def extract_lineup_windows(page_text: str):
    """並び候補がありそうな窓を複数返す。最初の「ライン」などに引っかかって外す問題を避ける。"""
    s = normalize_text(page_text)

    keywords = [
        "並び予想", "予想並び", "周回予想", "予想周回",
        "ライン予想", "初手", "隊列", "並び",
    ]

    end_keywords = [
        "基本情報", "直近成績", "前検コメ", "対戦成績",
        "オッズ一覧", "レース情報", "払戻", "結果", "出走表",
        "人気順", "3連単", "2車単", "2車複", "3連複",
        "選手コメント", "ニュース",
    ]

    windows = []
    seen = set()

    for kw in keywords:
        for m in re.finditer(re.escape(kw), s):
            tail = s[m.start():m.start() + 3000]
            body = s[m.end():m.end() + 3000]

            end_pos = len(body)
            for end_kw in end_keywords:
                p = body.find(end_kw)
                if p != -1:
                    end_pos = min(end_pos, p)

            win = normalize_text(body[:end_pos])
            if not win:
                win = normalize_text(tail)

            if len(re.findall(r"[1-9]", win)) < 5:
                continue

            key = win[:300]
            if key not in seen:
                windows.append(win)
                seen.add(key)

    return windows


def extract_lineup_window(page_text: str):
    windows = extract_lineup_windows(page_text)
    return windows[0] if windows else ""


def parse_lineup_candidate_string(candidate: str):
    groups = parse_lineup_groups(candidate)
    if not groups:
        return None

    flat = list(itertools.chain.from_iterable(groups))
    if len(flat) not in (5, 6, 7, 9):
        return None

    expected = set(range(1, len(flat) + 1))
    if set(flat) != expected:
        return None

    return groups_to_lineup_string(groups)


def _lineup_from_token_window(text: str):
    """区切りや / がある近辺から、余計な数字を除いて並びを復元する保険。"""
    s = normalize_text(text)

    pretty_patterns = [
        re.compile(r'([1-9](?:\s*[-→]\s*[1-9])*(?:\s*/\s*[1-9](?:\s*[-→]\s*[1-9])*){1,8})'),
        re.compile(r'([1-9](?:\s+[1-9]){4,8})'),
    ]
    for pat in pretty_patterns:
        for m in pat.finditer(s):
            cand = normalize_text(m.group(1)).replace("→", "-")
            parsed = parse_lineup_candidate_string(cand)
            if parsed:
                return parsed

    tokens = re.findall(r"区切り|/|→|-|[1-9]", s)
    if not tokens:
        return None

    if "区切り" in tokens or "/" in tokens:
        groups = []
        current = []
        for t in tokens:
            if t in ["区切り", "/"]:
                if current:
                    groups.append(current)
                    current = []
            elif t == "-" or t == "→":
                continue
            else:
                current.append(int(t))
        if current:
            groups.append(current)

        flat = list(itertools.chain.from_iterable(groups))
        if len(flat) in (5, 6, 7, 9) and len(set(flat)) == len(flat):
            expected = set(range(1, len(flat) + 1))
            if set(flat) == expected:
                return groups_to_lineup_string(groups)

    nums = [int(t) for t in tokens if re.fullmatch(r"[1-9]", t)]
    for n in (9, 7, 6, 5):
        if len(nums) < n:
            continue
        expected = set(range(1, n + 1))
        for i in range(0, len(nums) - n + 1):
            chunk = nums[i:i + n]
            if len(set(chunk)) == n and set(chunk) == expected:
                return groups_to_lineup_string([[x] for x in chunk])

    return None


def parse_lineup_from_page_text(page_text: str):
    s = normalize_text(page_text)

    windows = extract_lineup_windows(s)
    for window in windows:
        parsed = _lineup_from_token_window(window)
        if parsed:
            return parsed

    parsed = _lineup_from_token_window(s)
    if parsed:
        return parsed

    return None


def fetch_lineup_from_winticket(url: str):
    candidate_urls = build_lineup_candidate_urls(url)
    debug_items = []

    for target_url in candidate_urls:
        try:
            fetched = get_html_text_title(target_url)
            windows = extract_lineup_windows(fetched["text"])
            lineup = parse_lineup_from_page_text(fetched["text"])

            debug_items.append(
                {
                    "url": target_url,
                    "status_code": fetched["status_code"],
                    "title": fetched["title"],
                    "lineup_found": lineup if lineup else "",
                    "lineup_window": windows[0][:600] if windows else "",
                    "lineup_windows_count": len(windows),
                    "lineup_windows_preview": [w[:300] for w in windows[:5]],
                    "text_head": fetched["text"][:400],
                }
            )

            if lineup:
                st.session_state["lineup_debug_info"] = {
                    "source_type": "multi_candidate_url_lineup_parse_v3",
                    "candidate_results": debug_items,
                }
                return lineup

        except Exception as e:
            debug_items.append({"url": target_url, "error": str(e)})

    st.session_state["lineup_debug_info"] = {
        "source_type": "multi_candidate_url_lineup_parse_v3",
        "candidate_results": debug_items,
    }
    raise ValueError("URLから並びを抽出できませんでした。デバッグの lineup_windows_preview を貼ってください。")


# =========================================================
# 選手情報抽出
# =========================================================
def extract_players_section(page_text: str) -> str:
    text = normalize_text(page_text)

    start_keywords = ["AI 競走得点", "競走得点", "脚質"]
    end_keywords = [
        "並び予想", "予想並び", "並び",
        "オッズ一覧", "人気順",
        "2車単", "3連単", "2車複", "3連複"
    ]

    start_pos = -1
    for kw in start_keywords:
        pos = text.find(kw)
        if pos != -1:
            start_pos = pos
            break

    if start_pos == -1:
        return text

    end_pos = len(text)
    for kw in end_keywords:
        pos = text.find(kw, start_pos + 1)
        if pos != -1:
            end_pos = min(end_pos, pos)

    section = normalize_text(text[start_pos:end_pos])

    if len(section) < 300:
        return text

    return section


def extract_name_from_block(block: str) -> str:
    b = normalize_text(block)

    m = re.search(rf'([一-龥ぁ-んァ-ヶ々]{{2,12}})\s+(?:{PREF_PATTERN})', b)
    if m:
        cand = normalize_text(m.group(1))
        if is_valid_player_name(cand):
            return cand

    candidates = re.findall(r'([一-龥ぁ-んァ-ヶ々]{2,12})', b)
    ng_words = set(PREFECTURES + ["本命", "対抗", "単穴", "連下", "勝率", "コメント", "倍率", "ギヤ", "単騎で"])

    for cand in candidates:
        cand = normalize_text(cand)
        if cand in ng_words:
            continue
        if is_valid_player_name(cand):
            return cand

    return ""


def extract_score_from_block(block: str) -> float:
    b = normalize_text(block)

    patterns = [
        re.compile(r'\d{2,3}期\s+(?:本命|対抗|単穴|連下)?\s*([4-9]\d(?:\.\d{1,3})?)'),
        re.compile(r'(?:本命|対抗|単穴|連下)\s*([4-9]\d(?:\.\d{1,3})?)'),
        re.compile(r'([4-9]\d(?:\.\d{1,3})?)\s+\d+\s+\d+\s+\d+\s+(?:逃|捲|追|両|自)'),
    ]

    for pat in patterns:
        m = pat.search(b)
        if m:
            v = safe_float(m.group(1), 0.0)
            if 40 <= v <= 130:
                return v

    candidates = []
    for m in re.finditer(r'([4-9]\d(?:\.\d{1,3})?)', b):
        raw = m.group(1)
        v = safe_float(raw, 0.0)

        if not (40 <= v <= 130):
            continue

        before = b[max(0, m.start() - 3):m.start()]
        after = b[m.end():m.end() + 3]

        if "期" in before or "期" in after:
            continue
        if "歳" in before or "歳" in after:
            continue

        candidates.append(v)

    if candidates:
        return candidates[-1]

    return 0.0


def extract_style_from_block(block: str) -> str:
    b = normalize_text(block)

    patterns = [
        re.compile(r'(?:本命|対抗|単穴|連下)?\s*[4-9]\d(?:\.\d{1,3})?\s+\d+\s+\d+\s+\d+\s+(逃|捲|追|両|自)'),
        re.compile(r'[4-9]\d(?:\.\d{1,3})?(?:\s+\d+){0,6}\s+(逃|捲|追|両|自)'),
    ]

    for pat in patterns:
        m = pat.search(b)
        if m:
            return normalize_text(m.group(1))

    m = re.search(r'(逃|捲|追|両|自)', b)
    if m:
        return normalize_text(m.group(1))

    return ""


def extract_single_player_by_car(text: str, car: int):
    s = normalize_text(text)

    patterns = [
        re.compile(
            rf'(?<!\d){car}\s+{car}\s+'
            rf'([一-龥ぁ-んァ-ヶ々]{{2,12}})\s+'
            rf'({PREF_PATTERN})\s+'
            rf'([ALS]\d)\s+'
            rf'(\d{{2}})歳\s+'
            rf'(\d{{2,3}})期\s+'
            rf'(?:本命|対抗|単穴|連下)?\s*'
            rf'([4-9]\d(?:\.\d{{1,3}})?)\s+'
            rf'(?:\d+\s+){{2,5}}'
            rf'(逃|捲|追|両|自)'
        ),
        re.compile(
            rf'(?<!\d){car}\s+'
            rf'([一-龥ぁ-んァ-ヶ々]{{2,12}})\s+'
            rf'({PREF_PATTERN})\s+'
            rf'([ALS]\d)\s+'
            rf'(\d{{2}})歳\s+'
            rf'(\d{{2,3}})期\s+'
            rf'(?:本命|対抗|単穴|連下)?\s*'
            rf'([4-9]\d(?:\.\d{{1,3}})?)\s+'
            rf'(?:\d+\s+){{2,5}}'
            rf'(逃|捲|追|両|自)'
        ),
    ]

    for idx, pat in enumerate(patterns, start=1):
        m = pat.search(s)
        if not m:
            continue

        name = normalize_text(m.group(1))
        score = safe_float(m.group(5), 0.0)
        style = normalize_text(m.group(6))

        if is_valid_player_name(name) and 40.0 <= score <= 130.0 and style in ["逃", "捲", "追", "両", "自"]:
            return {
                "車番": car,
                "選手名": name,
                "競走得点": score,
                "脚質": style,
                "source": f"single_pattern_{idx}",
            }

    next_car = car + 1
    block_patterns = []

    if next_car <= 9:
        block_patterns.append(re.compile(rf'(?<!\d){car}\s+{car}\s+(.*?)(?=(?<!\d){next_car}\s+{next_car}\s+|$)'))
        block_patterns.append(re.compile(rf'(?<!\d){car}\s+(.*?)(?=(?<!\d){next_car}\s+{next_car}\s+|$)'))
    else:
        block_patterns.append(re.compile(rf'(?<!\d){car}\s+{car}\s+(.*)$'))
        block_patterns.append(re.compile(rf'(?<!\d){car}\s+(.*)$'))

    for bpat in block_patterns:
        mm = bpat.search(s)
        if not mm:
            continue

        block = normalize_text(mm.group(1))[:500]
        name = extract_name_from_block(block)
        score = extract_score_from_block(block)
        style = extract_style_from_block(block)

        if is_valid_player_name(name) and 40.0 <= score <= 130.0 and style in ["逃", "捲", "追", "両", "自"]:
            return {
                "車番": car,
                "選手名": name,
                "競走得点": score,
                "脚質": style,
                "source": "single_block",
            }

    return None


def extract_players_with_regex(text: str, num_riders: int):
    s = normalize_text(text)
    rows = []
    preview = []
    seen = set()

    entry_pattern = re.compile(
        rf'(?<!\d)'
        rf'([1-9])\s+\1\s+'
        rf'([一-龥ぁ-んァ-ヶ々]{{2,12}})\s+'
        rf'({PREF_PATTERN})\s+'
        rf'([ALS]\d)\s+'
        rf'(\d{{2}})歳\s+'
        rf'(\d{{2,3}})期\s+'
        rf'(?:本命|対抗|単穴|連下)?\s*'
        rf'([4-9]\d(?:\.\d{{1,3}})?)\s+'
        rf'(\d+)\s+(\d+)\s+(\d+)\s+'
        rf'(逃|捲|追|両|自)'
    )

    for m in entry_pattern.finditer(s):
        car = safe_int(m.group(1))
        name = normalize_text(m.group(2))
        score = safe_float(m.group(7), 0.0)
        style = normalize_text(m.group(11))

        if (
            1 <= car <= num_riders
            and car not in seen
            and is_valid_player_name(name)
            and 40.0 <= score <= 130.0
            and style in ["逃", "捲", "追", "両", "自"]
        ):
            seen.add(car)
            rows.append({"車番": car, "選手名": name, "競走得点": score, "脚質": style})
            preview.append({"車番": car, "選手名": name, "競走得点": score, "脚質": style, "source": "entry_pattern"})

    if len(rows) < num_riders:
        for car in range(1, num_riders + 1):
            if car in seen:
                continue

            hit = extract_single_player_by_car(s, car)
            if hit:
                seen.add(car)
                rows.append(
                    {
                        "車番": hit["車番"],
                        "選手名": hit["選手名"],
                        "競走得点": hit["競走得点"],
                        "脚質": hit["脚質"],
                    }
                )
                preview.append(hit)

    if not rows:
        return pd.DataFrame(), {"hit_count": 0, "preview": []}

    df = pd.DataFrame(rows).groupby("車番", as_index=False).first()
    df = df.sort_values("車番").reset_index(drop=True)

    return df[["車番", "選手名", "競走得点", "脚質"]].copy(), {
        "hit_count": len(df),
        "preview": preview[:12],
    }


def extract_players_by_car_blocks(text: str, num_riders: int):
    s = normalize_text(text)
    rows = []
    preview = []

    for car in range(1, num_riders + 1):
        hit = extract_single_player_by_car(s, car)
        if not hit:
            continue

        rows.append(
            {
                "車番": hit["車番"],
                "選手名": hit["選手名"],
                "競走得点": hit["競走得点"],
                "脚質": hit["脚質"],
            }
        )
        hit["source"] = "car_block_safe"
        preview.append(hit)

    if not rows:
        return pd.DataFrame(), {"hit_count": 0, "preview": []}

    df = pd.DataFrame(rows).groupby("車番", as_index=False).first()
    df = df.sort_values("車番").reset_index(drop=True)

    return df[["車番", "選手名", "競走得点", "脚質"]].copy(), {
        "hit_count": len(df),
        "preview": preview[:12],
    }




def normalize_player_df(players_df: pd.DataFrame, num_riders: int) -> pd.DataFrame:
    """
    通常版・選手取得の最終安全整形 v8。

    重要：名前が取れている選手を、得点0・脚質空欄という理由で消さない。
    - 必須：車番、選手名
    - 任意：競走得点、脚質
    - 同じ車番は品質の高い候補を採用
    - 同じ選手名が複数車番に出た場合は、品質の高い方を残す
    """
    cols = ["車番", "選手名", "競走得点", "脚質"]
    if players_df is None or players_df.empty:
        return pd.DataFrame(columns=cols)

    df = players_df.copy()

    for col in cols:
        if col not in df.columns:
            df[col] = ""
    if "source" not in df.columns:
        df["source"] = ""

    df["車番"] = pd.to_numeric(df["車番"], errors="coerce").fillna(0).astype(int)
    df["選手名"] = df["選手名"].astype(str).map(normalize_text)
    df["競走得点"] = pd.to_numeric(df["競走得点"], errors="coerce").fillna(0.0)
    df["脚質"] = df["脚質"].astype(str).map(normalize_text)
    df["source"] = df["source"].astype(str)

    # 車番と名前だけは必須
    df = df[(df["車番"] >= 1) & (df["車番"] <= int(num_riders))]
    df = df[df["選手名"].map(is_valid_player_name)]

    # 得点・脚質は任意。怪しい値だけ空に戻す。
    df.loc[~df["競走得点"].between(40.0, 130.0), "競走得点"] = 0.0
    df.loc[~df["脚質"].isin(["逃", "捲", "追", "両", "自"]), "脚質"] = ""

    if df.empty:
        return pd.DataFrame(columns=cols)

    source_weight = {
        "json_exact": 120,
        "json_recursive": 115,
        "html_card": 100,
        "entry_pattern": 90,
        "loose_entry": 75,
        "single_block": 65,
        "car_block_safe": 60,
        "sequence_card": 45,
        "sequence_text": 35,
    }

    def candidate_quality(row) -> float:
        q = 0.0
        name = str(row.get("選手名", ""))
        score = safe_float(row.get("競走得点", 0), 0)
        style = str(row.get("脚質", ""))
        src = str(row.get("source", ""))

        q += source_weight.get(src, 20)
        if is_valid_player_name(name):
            q += 10
        if 60.0 <= score <= 125.0:
            q += 12
        elif 40.0 <= score <= 130.0:
            q += 5
        if style in ["逃", "捲", "追", "両", "自"]:
            q += 8
        return q

    df["_quality"] = df.apply(candidate_quality, axis=1)
    df = df.sort_values(["車番", "_quality", "競走得点"], ascending=[True, False, False]).reset_index(drop=True)

    # 車番ごとに最良候補を採用
    picked = {}
    for car in range(1, int(num_riders) + 1):
        cand = df[df["車番"] == car]
        if cand.empty:
            continue
        picked[car] = cand.iloc[0]

    # 同名重複は品質の高い方を残し、可能なら代替候補へ差し替え
    name_to_cars = {}
    for car, row in picked.items():
        name_to_cars.setdefault(str(row["選手名"]), []).append(car)

    for name, cars in list(name_to_cars.items()):
        if len(cars) <= 1:
            continue

        keep_car = max(cars, key=lambda c: safe_float(picked[c].get("_quality", 0), 0))
        for car in cars:
            if car == keep_car:
                continue

            used_names = {str(v["選手名"]) for k, v in picked.items() if k != car}
            cand = df[df["車番"] == car]
            replacement = None
            for _, row in cand.iterrows():
                if str(row["選手名"]) not in used_names:
                    replacement = row
                    break

            if replacement is not None:
                picked[car] = replacement
            else:
                picked.pop(car, None)

    rows = []
    for car in sorted(picked.keys()):
        row = picked[car]
        rows.append({
            "車番": int(car),
            "選手名": str(row["選手名"]),
            "競走得点": float(safe_float(row.get("競走得点", 0), 0)),
            "脚質": str(row.get("脚質", "")),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=cols)

    out = out.drop_duplicates(subset=["車番"], keep="first")
    out = out.drop_duplicates(subset=["選手名"], keep="first")
    out = out.sort_values("車番").reset_index(drop=True)
    return out[cols].copy()

def _walk_json(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk_json(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_json(v)


def _pick_from_keys(d: dict, keys):
    for k in keys:
        if k in d and d[k] not in [None, ""]:
            return d[k]
    lower = {str(k).lower(): k for k in d.keys()}
    for k in keys:
        lk = str(k).lower()
        if lk in lower and d[lower[lk]] not in [None, ""]:
            return d[lower[lk]]
    return None


def _normalize_style_value(v) -> str:
    s = normalize_text(v)
    style_map = {
        "nige": "逃", "escape": "逃", "逃げ": "逃",
        "makuri": "捲", "捲り": "捲", "まくり": "捲",
        "oi": "追", "chase": "追", "追込": "追", "追い": "追",
        "ryo": "両", "both": "両", "自在": "自", "jizai": "自", "自力": "自",
    }
    if s in ["逃", "捲", "追", "両", "自"]:
        return s
    sl = s.lower()
    for key, val in style_map.items():
        if key in sl or key in s:
            return val
    m = re.search(r"(逃|捲|追|両|自)", s)
    return m.group(1) if m else ""


def extract_players_from_json_html(html: str, num_riders: int):
    """
    WINTICKETのHTML内JSONから選手情報を拾う。
    テキスト抽出よりこちらを優先する。
    """
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    preview = []

    json_texts = []
    for tag in soup.find_all("script"):
        txt = tag.string if tag.string else tag.get_text(" ", strip=True)
        txt = txt or ""
        if not txt:
            continue
        typ = (tag.get("type") or "").lower()
        sid = (tag.get("id") or "").lower()
        if "json" in typ or "__next_data__" in sid:
            json_texts.append(txt)
        elif ("racer" in txt.lower() or "player" in txt.lower() or "選手" in txt) and ("score" in txt.lower() or "競走得点" in txt):
            json_texts.append(txt)

    car_keys = ["車番", "racerNumber", "racerNo", "riderNumber", "number", "bikeNumber", "carNumber", "bracketNumber", "frameNumber"]
    name_keys = ["選手名", "racerName", "riderName", "playerName", "name", "fullName"]
    score_keys = ["競走得点", "raceScore", "racerScore", "racingScore", "evaluationPoint", "currentPoint", "racerPoint", "racePoint", "competitionPoint", "rankPoint"]
    style_keys = ["脚質", "legType", "legTypeName", "style", "ridingStyle"]

    for txt in json_texts:
        try:
            data = json.loads(txt)
            for d in _walk_json(data):
                car = safe_int(_pick_from_keys(d, car_keys), 0)
                name = normalize_text(_pick_from_keys(d, name_keys) or "")
                score = safe_float(_pick_from_keys(d, score_keys), 0.0)
                style = _normalize_style_value(_pick_from_keys(d, style_keys) or "")
                if 1 <= car <= num_riders and is_valid_player_name(name) and 40 <= score <= 130 and style in ["逃", "捲", "追", "両", "自"]:
                    item = {"車番": car, "選手名": name, "競走得点": score, "脚質": style, "source": "json_recursive"}
                    rows.append(item)
                    preview.append(item)
        except Exception:
            pass

        nt = normalize_text(txt)
        for car in range(1, num_riders + 1):
            pattern = re.compile(
                rf'(?:racerNumber|racerNo|riderNumber|bikeNumber|carNumber|車番)["\':\s]{{1,10}}{car}.{{0,900}}?'
                rf'([一-龥ぁ-んァ-ヶ々]{{2,12}}).{{0,900}}?'
                rf'([4-9]\d(?:\.\d{{1,3}})?).{{0,300}}?'
                rf'(逃|捲|追|両|自|逃げ|捲り|追込|自在)',
                re.DOTALL,
            )
            m = pattern.search(nt)
            if m:
                name = normalize_text(m.group(1))
                score = safe_float(m.group(2), 0.0)
                style = _normalize_style_value(m.group(3))
                if is_valid_player_name(name) and 40 <= score <= 130 and style:
                    item = {"車番": car, "選手名": name, "競走得点": score, "脚質": style, "source": "json_exact"}
                    rows.append(item)
                    preview.append(item)

    if not rows:
        return pd.DataFrame(columns=["車番", "選手名", "競走得点", "脚質"]), {"hit_count": 0, "preview": []}

    df = normalize_player_df(pd.DataFrame(rows), num_riders)
    return df, {"hit_count": len(df), "preview": preview[:20]}


def extract_players_from_html_cards(html: str, num_riders: int):
    """HTMLタグのカード単位で選手を拾う保険。"""
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    preview = []
    selectors = [
        '[class*="RaceCard"]', '[class*="race-card"]', '[class*="Player"]', '[class*="player"]',
        '[class*="Racer"]', '[class*="racer"]', 'li', 'tr'
    ]
    seen_texts = set()
    for sel in selectors:
        for tag in soup.select(sel):
            text = normalize_text(tag.get_text(" ", strip=True))
            if not text or text in seen_texts or len(text) < 12 or len(text) > 1200:
                continue
            seen_texts.add(text)
            for car in range(1, num_riders + 1):
                pats = [
                    rf'(?<!\d){car}\s+{car}\s+([一-龥ぁ-んァ-ヶ々]{{2,12}})\s+(?:{PREF_PATTERN})\s+[ALS]\d\s+\d{{2}}歳\s+\d{{2,3}}期.*?([4-9]\d(?:\.\d{{1,3}})?).*?(逃|捲|追|両|自)',
                    rf'(?<!\d){car}\s+([一-龥ぁ-んァ-ヶ々]{{2,12}})\s+(?:{PREF_PATTERN}).*?([4-9]\d(?:\.\d{{1,3}})?).*?(逃|捲|追|両|自)',
                ]
                for pat in pats:
                    m = re.search(pat, text)
                    if not m:
                        continue
                    name = normalize_text(m.group(1))
                    score = safe_float(m.group(2), 0.0)
                    style = normalize_text(m.group(3))
                    if is_valid_player_name(name) and 40 <= score <= 130 and style in ["逃", "捲", "追", "両", "自"]:
                        item = {"車番": car, "選手名": name, "競走得点": score, "脚質": style, "source": "html_card"}
                        rows.append(item)
                        preview.append({**item, "text": text[:160]})
    if not rows:
        return pd.DataFrame(columns=["車番", "選手名", "競走得点", "脚質"]), {"hit_count": 0, "preview": []}
    df = normalize_player_df(pd.DataFrame(rows), num_riders)
    return df, {"hit_count": len(df), "preview": preview[:20]}


def extract_players_loose_entries(text: str, num_riders: int):
    """既存regexで漏れた車番を拾う緩めの保険。"""
    s = normalize_text(text)
    rows = []
    preview = []
    for car in range(1, num_riders + 1):
        patterns = [
            rf'(?<!\d){car}\s+{car}\s+([一-龥ぁ-んァ-ヶ々]{{2,12}})\s+(?:{PREF_PATTERN})\s+[ALS]\d\s+\d{{2}}歳\s+\d{{2,3}}期\s+(?:本命|対抗|単穴|連下)?\s*([4-9]\d(?:\.\d{{1,3}})?).{{0,160}}?(逃|捲|追|両|自)',
            rf'(?<!\d){car}\s+([一-龥ぁ-んァ-ヶ々]{{2,12}})\s+(?:{PREF_PATTERN}).{{0,220}}?([4-9]\d(?:\.\d{{1,3}})?).{{0,160}}?(逃|捲|追|両|自)',
        ]
        for pat in patterns:
            m = re.search(pat, s)
            if not m:
                continue
            name = normalize_text(m.group(1))
            score = safe_float(m.group(2), 0.0)
            style = normalize_text(m.group(3))
            if is_valid_player_name(name) and 40 <= score <= 130 and style in ["逃", "捲", "追", "両", "自"]:
                item = {"車番": car, "選手名": name, "競走得点": score, "脚質": style, "source": "loose_entry"}
                rows.append(item)
                preview.append(item)
                break
    if not rows:
        return pd.DataFrame(columns=["車番", "選手名", "競走得点", "脚質"]), {"hit_count": 0, "preview": []}
    df = normalize_player_df(pd.DataFrame(rows), num_riders)
    return df, {"hit_count": len(df), "preview": preview[:20]}


def merge_player_dfs(base_df: pd.DataFrame, add_df: pd.DataFrame, num_riders=None) -> pd.DataFrame:
    frames = []
    for x in [base_df, add_df]:
        if x is not None and not x.empty:
            frames.append(x.copy())
    if not frames:
        return pd.DataFrame(columns=["車番", "選手名", "競走得点", "脚質"])
    merged = pd.concat(frames, ignore_index=True)
    if num_riders is None:
        num_riders = int(pd.to_numeric(merged.get("車番", 0), errors="coerce").fillna(0).max())
    return normalize_player_df(merged, num_riders)


def fetch_players_from_winticket(url: str, num_riders: int):
    """
    WINTICKET選手取得 v8 通常版安全取得。

    方針：
    1. まず既存の高精度ルート（JSON/HTMLカード/regex）を使う
    2. 足りない車番だけ、ページ上の選手カードらしき並びから補完する
    3. それでも足りない場合もアプリを止めず、取れた分だけ返す
    """
    candidate_urls = build_player_candidate_urls(url)
    debug_items = []
    best_df = pd.DataFrame(columns=["車番", "選手名", "競走得点", "脚質"])

    def extract_sequence_candidates(html: str, page_text: str, n: int):
        """車番がうまく拾えない時の保険。出走表らしい順番で名前だけでも拾う。"""
        soup = BeautifulSoup(html, "html.parser")
        selectors = [
            '[class*="RaceCard"]',
            '[class*="race-card"]',
            '[class*="Player"]',
            '[class*="player"]',
            '[class*="Racer"]',
            '[class*="racer"]',
            'tr',
            'li',
        ]

        candidates = []
        used_names = set()
        used_texts = set()

        def pick_name(text: str) -> str:
            text = normalize_text(text)
            m = re.search(rf'([一-龥ぁ-んァ-ヶ々]{{2,8}})\s+(?:{PREF_PATTERN})', text)
            if m:
                cand = normalize_text(m.group(1))
                if is_valid_player_name(cand):
                    return cand
            for cand in re.findall(r'[一-龥ぁ-んァ-ヶ々]{2,8}', text):
                cand = normalize_text(cand)
                if is_valid_player_name(cand):
                    return cand
            return ""

        def pick_score(text: str) -> float:
            vals = []
            for m in re.finditer(r'([6-9]\d(?:\.\d{1,3})?|1[0-2]\d(?:\.\d{1,3})?)', text):
                v = safe_float(m.group(1), 0.0)
                if 60.0 <= v <= 130.0:
                    vals.append(v)
            return vals[0] if vals else 0.0

        def pick_style(text: str) -> str:
            m = re.search(r'(逃|捲|追|両|自)', text)
            return normalize_text(m.group(1)) if m else ""

        for sel in selectors:
            for tag in soup.select(sel):
                text = normalize_text(tag.get_text(" ", strip=True))
                if not text or text in used_texts:
                    continue
                used_texts.add(text)

                if len(text) < 8 or len(text) > 1200:
                    continue

                name = pick_name(text)
                if not name or name in used_names:
                    continue

                # 選手カードらしさが薄すぎるものは除外
                has_pref = bool(re.search(PREF_PATTERN, text))
                has_style = bool(re.search(r'(逃|捲|追|両|自)', text))
                has_score = bool(re.search(r'([6-9]\d(?:\.\d{1,3})?|1[0-2]\d(?:\.\d{1,3})?)', text))
                if not (has_pref or has_style or has_score):
                    continue

                candidates.append({
                    "車番": len(candidates) + 1,
                    "選手名": name,
                    "競走得点": pick_score(text),
                    "脚質": pick_style(text),
                    "source": "sequence_card",
                })
                used_names.add(name)

                if len(candidates) >= n:
                    break
            if len(candidates) >= n:
                break

        if candidates:
            return pd.DataFrame(candidates), candidates[:12]

        return pd.DataFrame(columns=["車番", "選手名", "競走得点", "脚質", "source"]), []

    for target_url in candidate_urls:
        try:
            fetched = get_html_text_title(target_url)
            full_text = fetched["text"]
            html = fetched.get("html", "")
            section_text = extract_players_section(full_text)

            df_json, dbg_json = extract_players_from_json_html(html, num_riders)
            df_cards, dbg_cards = extract_players_from_html_cards(html, num_riders)
            df_section, dbg_section = extract_players_with_regex(section_text, num_riders)
            df_full, dbg_full = extract_players_with_regex(full_text, num_riders)
            df_block, dbg_block = extract_players_by_car_blocks(full_text, num_riders)
            df_loose, dbg_loose = extract_players_loose_entries(full_text, num_riders)
            df_sequence, seq_preview = extract_sequence_candidates(html, full_text, num_riders)

            # まず高精度候補を統合
            valid_frames = [
                x for x in [df_json, df_cards, df_section, df_full, df_block, df_loose]
                if x is not None and not x.empty
            ]
            combined = pd.concat(valid_frames, ignore_index=True) if valid_frames else pd.DataFrame()
            final_df = normalize_player_df(combined, num_riders)

            # 足りない場合は、順番候補で不足車番だけ補完
            if len(final_df) < num_riders and df_sequence is not None and not df_sequence.empty:
                seq_df = normalize_player_df(df_sequence, num_riders)
                if not seq_df.empty:
                    existing_cars = set(final_df["車番"].astype(int).tolist()) if not final_df.empty else set()
                    existing_names = set(final_df["選手名"].astype(str).tolist()) if not final_df.empty else set()
                    fill_rows = []
                    for _, row in seq_df.iterrows():
                        car = safe_int(row.get("車番", 0), 0)
                        name = normalize_text(row.get("選手名", ""))
                        if car in existing_cars or name in existing_names:
                            continue
                        fill_rows.append(row.to_dict())

                    if fill_rows:
                        final_df = normalize_player_df(
                            pd.concat([final_df, pd.DataFrame(fill_rows)], ignore_index=True),
                            num_riders,
                        )

            missing = sorted(list(set(range(1, num_riders + 1)) - set(final_df["車番"].astype(int).tolist()))) if not final_df.empty else list(range(1, num_riders + 1))

            debug_items.append({
                "url": target_url,
                "status_code": fetched["status_code"],
                "title": fetched["title"],
                "json_hits": len(df_json),
                "card_hits": len(df_cards),
                "section_hits": len(df_section),
                "full_hits": len(df_full),
                "block_hits": len(df_block),
                "loose_hits": len(df_loose),
                "sequence_hits": len(df_sequence),
                "final_hits": len(final_df),
                "missing_after": missing,
                "final_players": final_df.to_dict(orient="records") if not final_df.empty else [],
                "preview_json": dbg_json.get("preview", [])[:6],
                "preview_cards": dbg_cards.get("preview", [])[:6],
                "preview_regex": (dbg_section.get("preview", [])[:4] + dbg_full.get("preview", [])[:4] + dbg_block.get("preview", [])[:4] + dbg_loose.get("preview", [])[:4]),
                "preview_sequence": seq_preview[:8],
                "section_head": section_text[:300],
            })

            if len(final_df) > len(best_df):
                best_df = final_df.copy()
            if len(best_df) >= num_riders:
                break

        except Exception as e:
            debug_items.append({"url": target_url, "error": str(e)})

    best_df = normalize_player_df(best_df, num_riders)
    missing = sorted(list(set(range(1, num_riders + 1)) - set(best_df["車番"].astype(int).tolist()))) if not best_df.empty else list(range(1, num_riders + 1))

    debug_info = {
        "source_type": "winticket_player_auto_v8_safe_normal",
        "hit_count": len(best_df),
        "missing": missing,
        "candidate_results": debug_items,
        "final_players": best_df.to_dict(orient="records") if not best_df.empty else [],
    }

    if best_df.empty:
        raise ValueError("選手情報を自動取得できませんでした。デバッグ情報を確認してください。")

    # 通常版は止めない。取れた分だけ反映して、不足は手入力できるようにする。
    if len(best_df) < num_riders:
        debug_info["warning"] = f"部分取得: {len(best_df)}人 / 不足車番: {missing}"

    return best_df, debug_info

def apply_players_to_df(df: pd.DataFrame, players_df: pd.DataFrame) -> pd.DataFrame:
    """
    取得した選手情報を画面の表へ反映する。
    得点0・脚質空欄でも、名前が取れていれば消さずに反映する。
    """
    out = df.copy()

    if players_df is None or players_df.empty:
        return out

    players_df = players_df.copy()

    for col in ["車番", "選手名", "競走得点", "脚質"]:
        if col not in players_df.columns:
            players_df[col] = ""

    players_df["車番"] = pd.to_numeric(players_df["車番"], errors="coerce").fillna(0).astype(int)
    players_df["競走得点"] = pd.to_numeric(players_df["競走得点"], errors="coerce").fillna(0.0)

    for _, row in players_df.iterrows():
        car = int(row["車番"])
        if car <= 0 or car not in out["車番"].astype(int).tolist():
            continue

        name = normalize_text(row.get("選手名", ""))
        if is_valid_player_name(name):
            out.loc[out["車番"] == car, "選手名"] = name

        score = safe_float(row.get("競走得点", 0.0), 0.0)
        # 0は「未取得」としてそのまま。怪しい40台整数は反映しない。
        if 45.0 <= score <= 130.0:
            out.loc[out["車番"] == car, "競走得点"] = score

        style = normalize_text(row.get("脚質", ""))
        if style in ["逃", "捲", "追", "両", "自"]:
            out.loc[out["車番"] == car, "脚質"] = style

    return out

def extract_script_texts(html: str):
    soup = BeautifulSoup(html, "html.parser")
    items = []
    for tag in soup.find_all("script"):
        txt = tag.string if tag.string else tag.get_text(" ", strip=True)
        txt = normalize_text(txt or "")
        if txt:
            items.append(txt)
    return items


def extract_odds_loose(text: str, ticket_type: str):
    s = normalize_text(text)
    results = {}

    if ticket_type == "2車単":
        patterns = [
            re.compile(r'(?<!\d)([1-9])\s*-\s*([1-9])\s+([0-9]+(?:\.[0-9]+)?)(?!\d)'),
            re.compile(r'"combination"\s*:\s*"([1-9]-[1-9])".{0,80}?"odds"\s*:\s*([0-9]+(?:\.[0-9]+)?)'),
            re.compile(r'([1-9]-[1-9]).{0,80}?"odds"\s*:\s*([0-9]+(?:\.[0-9]+)?)'),
            re.compile(r'"odds"\s*:\s*([0-9]+(?:\.[0-9]+)?).{0,80}?([1-9]-[1-9])'),
        ]
    else:
        patterns = [
            re.compile(r'(?<!\d)([1-9])\s*-\s*([1-9])\s*-\s*([1-9])\s+([0-9]+(?:\.[0-9]+)?)(?!\d)'),
            re.compile(r'"combination"\s*:\s*"([1-9]-[1-9]-[1-9])".{0,80}?"odds"\s*:\s*([0-9]+(?:\.[0-9]+)?)'),
            re.compile(r'([1-9]-[1-9]-[1-9]).{0,80}?"odds"\s*:\s*([0-9]+(?:\.[0-9]+)?)'),
            re.compile(r'"odds"\s*:\s*([0-9]+(?:\.[0-9]+)?).{0,80}?([1-9]-[1-9]-[1-9])'),
        ]

    for pat in patterns:
        for m in pat.finditer(s):
            groups = m.groups()

            if ticket_type == "2車単":
                if len(groups) == 3:
                    a, b, odds = groups
                    key = f"{a}-{b}"
                    val = safe_float(odds, 0.0)
                elif len(groups) == 2:
                    if "-" in groups[0]:
                        key = normalize_ticket(groups[0])
                        val = safe_float(groups[1], 0.0)
                    else:
                        val = safe_float(groups[0], 0.0)
                        key = normalize_ticket(groups[1])
                else:
                    continue

                if len(key.split("-")) == 2 and len(set(key.split("-"))) == 2 and val > 0:
                    results[key] = val

            else:
                if len(groups) == 4:
                    a, b, c, odds = groups
                    key = f"{a}-{b}-{c}"
                    val = safe_float(odds, 0.0)
                elif len(groups) == 2:
                    if "-" in groups[0]:
                        key = normalize_ticket(groups[0])
                        val = safe_float(groups[1], 0.0)
                    else:
                        val = safe_float(groups[0], 0.0)
                        key = normalize_ticket(groups[1])
                else:
                    continue

                if len(key.split("-")) == 3 and len(set(key.split("-"))) == 3 and val > 0:
                    results[key] = val

    return results


def fetch_odds_from_winticket(url: str, ticket_type: str):
    candidate_urls = build_odds_candidate_urls(url)
    all_debug = []
    best_results = {}

    for target_url in candidate_urls:
        try:
            fetched = get_html_text_title(target_url)
            html = fetched["html"]
            text = fetched["text"]
            title = fetched["title"]

            page_results = extract_odds_loose(text, ticket_type=ticket_type)

            scripts = extract_script_texts(html)
            script_results = {}
            for txt in scripts:
                script_results.update(extract_odds_loose(txt, ticket_type=ticket_type))

            merged = {}
            merged.update(page_results)
            merged.update(script_results)

            debug_item = {
                "url": target_url,
                "status_code": fetched["status_code"],
                "title": title,
                "ticket_type": ticket_type,
                "text_head": text[:500],
                "page_hits": len(page_results),
                "script_hits": len(script_results),
                "merged_hits": len(merged),
                "preview": list(sorted(merged.items(), key=lambda x: x[1]))[:15],
            }
            all_debug.append(debug_item)

            if len(merged) > len(best_results):
                best_results = merged

        except Exception as e:
            all_debug.append({"url": target_url, "ticket_type": ticket_type, "error": str(e)})

    debug_info = {
        "source_type": "multi_candidate_url_loose_parse",
        "ticket_type": ticket_type,
        "best_hit_count": len(best_results),
        "candidate_results": all_debug,
    }

    if not best_results:
        raise ValueError("オッズを抽出できませんでした。")

    return best_results, debug_info


# =========================================================
# ログ保存
# =========================================================
def save_result_log(
    race_name: str,
    mode: str,
    weather: str,
    race_type: str,
    lineup: str,
    ticket_type: str,
    pred_df: pd.DataFrame,
    result_1: str,
    result_2: str,
    result_3: str,
    hit_status: str,
):
    is_new = not LOG_PATH.exists()

    with open(LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(
                [
                    "保存日時", "レース名", "券種", "モード", "天候", "レース種別", "並び", "結果", "判定",
                    "買い目", "買い目ランク", "AI評価", "期待値", "オッズ",
                    "購入金額", "期待回収額(目安)", "レース判定", "的中率評価", "レース評価点", "判定理由", "見送りAIコメント",
                ]
            )

        if ticket_type == "2車単":
            result_text = "-".join([x for x in [result_1, result_2] if x])
        else:
            result_text = "-".join([x for x in [result_1, result_2, result_3] if x])

        # ROI学習用の安全補正。
        # モバイル版では保存時に購入金額が空/0になりやすいため、
        # ログへ書く直前に最低100円を必ず入れる。
        if pred_df is None or pred_df.empty:
            return

        pred_df = ensure_prediction_amounts(pred_df, unit_bet=100)

        for _, row in pred_df.iterrows():
            purchase_amount = max(100, int(safe_float(row.get("購入金額", 100), 100)))
            expected_return = safe_float(row.get("期待回収額(目安)", 0), 0)

            writer.writerow(
                [
                    now_str(),
                    race_name,
                    ticket_type,
                    mode,
                    weather,
                    race_type,
                    lineup,
                    result_text,
                    hit_status,
                    row.get("買い目", ""),
                    row.get("買い目ランク", ""),
                    row.get("AI評価", ""),
                    row.get("期待値", ""),
                    row.get("オッズ", ""),
                    purchase_amount,
                    expected_return,
                    row.get("レース判定", ""),
                    row.get("的中率評価", ""),
                    row.get("レース評価点", ""),
                    row.get("判定理由", ""),
                    row.get("見送りAIコメント", ""),
                ]
            )

    # 保存後にも既存0円ログを修復
    repair_log_purchase_amounts()


def save_current_prediction(
    race_name: str,
    url: str,
    mode: str,
    weather: str,
    race_type: str,
    lineup_string: str,
    ticket_type: str,
    current_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    odds_dict: dict,
    unit_bet: int,
    display_count: int,
):
    saved_id = datetime.now().strftime("%Y%m%d%H%M%S%f")

    record = {
        "id": saved_id,
        "created_at": now_str(),
        "updated_at": now_str(),
        "race_name": race_name,
        "url": url,
        "mode": mode,
        "weather": weather,
        "race_type": race_type,
        "lineup_string": lineup_string,
        "ticket_type": ticket_type,
        "num_riders": len(current_df),
        "display_count": int(display_count),
        "unit_bet": int(unit_bet),
        "race_rows": current_df[DEFAULT_COLUMNS].to_dict(orient="records"),
        "pred_rows": pred_df.to_dict(orient="records"),
        "odds_dict": odds_dict,
        "result_saved": False,
        "hit_status": "未結果",
        "result": {},
    }
    save_race_record(record)


# =========================================================
# UI
# =========================================================
st.title("🚴 競輪AI モバイル版")
st.caption("モバイル版 / 通常版の安全取得ロジック反映 / 学習補正・ROI・見送りAI・賭け金AI")

if "race_rows" not in st.session_state:
    init_state(7)

with st.sidebar:
    st.header("設定")

    rider_options = [5, 6, 7, 9]
    current_num = st.session_state.get("num_riders", 7)
    current_index = rider_options.index(current_num) if current_num in rider_options else 2

    num_riders = st.radio("車立て", options=rider_options, index=current_index, horizontal=True)

    if num_riders != st.session_state.get("num_riders", 7):
        init_state(num_riders)
        st.rerun()

    ticket_type = st.selectbox(
        "券種",
        options=["3連単", "2車単"],
        index=0 if st.session_state.get("ticket_type", "3連単") == "3連単" else 1,
    )
    st.session_state["ticket_type"] = ticket_type

    race_type_options = ["通常", "ガールズ", "G3"]
    race_type_default = st.session_state.get("race_type", "通常")
    race_type = st.selectbox(
        "レース種別",
        options=race_type_options,
        index=race_type_options.index(race_type_default) if race_type_default in race_type_options else 0,
    )
    st.session_state["race_type"] = race_type

    prediction_style_options = ["的中率重視", "回収率重視"]
    prediction_style_default = st.session_state.get("prediction_style", "的中率重視")
    prediction_style = st.radio(
        "予想スタイル",
        options=prediction_style_options,
        index=prediction_style_options.index(prediction_style_default) if prediction_style_default in prediction_style_options else 0,
        horizontal=True,
    )
    st.session_state["prediction_style"] = prediction_style
    style_settings = get_prediction_style_settings(prediction_style)
    st.caption(style_settings["message"])

    display_count = st.selectbox(
        "買い目点数",
        options=list(range(3, 31)),
        index=7,
    )

    st.divider()
    strict_ai_on = st.checkbox("厳選AIを使う", value=True)
    default_strict_max = int(style_settings.get("default_max_count", 8))
    strict_max_options = list(range(3, 21))
    strict_max_count = st.selectbox(
        "厳選後の最大点数",
        options=strict_max_options,
        index=strict_max_options.index(default_strict_max) if default_strict_max in strict_max_options else 7,
        disabled=not strict_ai_on,
    )
    strict_min_roi = st.slider(
        "厳選ROI下限",
        min_value=90,
        max_value=140,
        value=int(style_settings.get("strict_min_roi", 110)),
        step=5,
        disabled=not strict_ai_on,
    )

    race_gate_on = st.checkbox(
        "レース選別AI（見送り判定を表示）",
        value=True,
        help="ONにすると、AIの自信度を見て勝負/様子見/見送りを表示します。買い目は基本表示します。",
    )
    st.caption("ON: ライン数だけでは切らず、AI信頼度・熱🔥/堅/穴の強さを中心に判定します。※買い目は基本表示します。")

    weather = st.selectbox("天候", options=["晴", "雨", "風強"], index=0)
    unit_bet = st.number_input("1点あたり金額", min_value=100, max_value=10000, step=100, value=100)

    st.caption("厚張り基準")
    st.caption(f"熱🔥 = {unit_bet * 3:,}円")
    st.caption(f"堅 = {unit_bet * 2:,}円")
    st.caption(f"穴 = {unit_bet * 2:,}円")
    st.caption(f"抑え = {unit_bet:,}円")

    if st.button("初期化", use_container_width=True):
        init_state(num_riders)
        st.rerun()

c1, c2 = st.columns([1, 1])

with c1:
    race_name = st.text_input("レース名", value=st.session_state.get("race_name", ""))
    st.session_state["race_name"] = race_name

with c2:
    default_url = "https://www.winticket.jp/keirin/kumamoto/racecard/2026041487/1/7"
    url = st.text_input("WINTICKET URL", value=st.session_state.get("last_url", default_url))
    st.session_state["last_url"] = url

c3, c4, c5, c6 = st.columns([1, 1, 1, 2])

with c3:
    if st.button("URLから並びを読み込む", use_container_width=True):
        try:
            lineup = fetch_lineup_from_winticket(url)
            st.session_state["lineup_string"] = lineup

            df = get_df()
            groups = parse_lineup_groups(lineup)
            total = len(list(itertools.chain.from_iterable(groups)))
            if total != len(df):
                init_state(total)
                df = get_df()

            df = apply_lineup_to_df(df, lineup)
            set_df(df)
            st.session_state["widget_ver"] = st.session_state.get("widget_ver", 0) + 1
            st.session_state["message"] = f"並び取得成功: {lineup}"
            st.rerun()
        except Exception as e:
            st.session_state["message"] = f"読み込み失敗: {e}"

with c4:
    if st.button("選手情報を自動取得", use_container_width=True):
        try:
            df = get_df()
            players_df, debug_info = fetch_players_from_winticket(url, len(df))
            df = apply_players_to_df(df, players_df)
            set_df(df)

            st.session_state["player_debug_info"] = debug_info
            st.session_state["widget_ver"] = st.session_state.get("widget_ver", 0) + 1
            st.session_state["message"] = f"選手情報取得成功: {len(players_df)}人"
            st.rerun()
        except Exception as e:
            st.session_state["player_debug_info"] = {"error": str(e)}
            st.session_state["message"] = f"選手情報取得失敗: {e}"

with c5:
    if st.button("オッズを自動取得", use_container_width=True):
        try:
            odds_dict, debug_info = fetch_odds_from_winticket(url, ticket_type=ticket_type)
            st.session_state["odds_dict"] = odds_dict
            st.session_state["odds_debug_info"] = debug_info
            st.session_state["message"] = f"オッズ取得成功: {len(odds_dict)}件"
            st.rerun()
        except Exception as e:
            st.session_state["odds_debug_info"] = {"error": str(e)}
            st.session_state["message"] = f"オッズ取得失敗: {e}"

with c6:
    msg = st.session_state.get("message", "")
    if msg:
        if "成功" in msg:
            st.success(msg)
        else:
            st.error(msg)

player_debug = st.session_state.get("player_debug_info", None)
if player_debug:
    with st.expander("選手情報取得デバッグ情報"):
        if player_debug.get("error"):
            st.error(player_debug["error"])
        else:
            st.write(player_debug)

lineup_debug = st.session_state.get("lineup_debug_info", None)
if lineup_debug:
    with st.expander("並び取得デバッグ情報"):
        st.write(lineup_debug)

odds_debug = st.session_state.get("odds_debug_info", None)
if odds_debug:
    with st.expander("オッズ取得デバッグ情報"):
        if odds_debug.get("error"):
            st.error(odds_debug["error"])
        else:
            st.write(odds_debug)

lineup_string = st.text_input(
    "並び文字列",
    value=st.session_state.get("lineup_string", ""),
    placeholder="例: 4-2 / 3-5-1-6",
)

c7, c8 = st.columns([1, 2])

with c7:
    if st.button("並びを反映", use_container_width=True):
        try:
            df = get_df()
            df = apply_lineup_to_df(df, lineup_string)
            set_df(df)
            st.session_state["lineup_string"] = lineup_string
            st.session_state["widget_ver"] = st.session_state.get("widget_ver", 0) + 1
            st.session_state["message"] = f"並び反映成功: {lineup_string}"
            st.rerun()
        except Exception as e:
            st.session_state["message"] = f"反映失敗: {e}"

with c8:
    if race_type == "ガールズ":
        st.caption(f"券種: {ticket_type} / レース種別: ガールズ / ライン評価なし / 取得オッズ: {len(st.session_state.get('odds_dict', {}))}件")
    else:
        st.caption(f"券種: {ticket_type} / レース種別: {race_type} / 取得オッズ: {len(st.session_state.get('odds_dict', {}))}件")

st.markdown("---")
st.subheader("出走表入力")

df = get_df().sort_values("車番").reset_index(drop=True)

with st.form("runner_form"):
    header = st.columns([0.8, 1.8, 1.2, 1.0, 0.8, 0.8, 0.8])
    header[0].markdown("**車番**")
    header[1].markdown("**選手名**")
    header[2].markdown("**競走得点**")
    header[3].markdown("**脚質**")
    header[4].markdown("**ライン**")
    header[5].markdown("**ライン順**")
    header[6].markdown("**単騎**")

    updated_rows = []
    style_options = ["", "逃", "捲", "追", "両", "自"]

    for i, row in df.iterrows():
        cols = st.columns([0.8, 1.8, 1.2, 1.0, 0.8, 0.8, 0.8])

        car_num = cols[0].number_input(
            f"車番_{i}", min_value=1, max_value=9, value=int(row["車番"]), step=1, key=widget_key("car", i)
        )
        name = cols[1].text_input(f"選手名_{i}", value=str(row["選手名"]), key=widget_key("name", i))
        score = cols[2].number_input(
            f"競走得点_{i}", min_value=0.0, max_value=200.0, value=float(row["競走得点"]), step=0.1, key=widget_key("score", i)
        )
        style_now = str(row["脚質"]) if str(row["脚質"]) in style_options else ""
        style = cols[3].selectbox(
            f"脚質_{i}", options=style_options, index=style_options.index(style_now), key=widget_key("style", i)
        )
        line_id = cols[4].number_input(
            f"ライン_{i}", min_value=0, max_value=9, value=int(row["ライン"]), step=1, key=widget_key("line", i)
        )
        line_order = cols[5].number_input(
            f"ライン順_{i}", min_value=0, max_value=9, value=int(row["ライン順"]), step=1, key=widget_key("line_order", i)
        )
        single = cols[6].selectbox(
            f"単騎_{i}", options=[0, 1], index=1 if int(row["単騎"]) == 1 else 0, key=widget_key("single", i)
        )

        updated_rows.append(
            {
                "車番": car_num,
                "選手名": name,
                "競走得点": score,
                "脚質": style,
                "ライン": line_id,
                "ライン順": line_order,
                "単騎": single,
            }
        )

    submit_rows = st.form_submit_button("出走表を反映", use_container_width=True)

if submit_rows:
    st.session_state["race_rows"] = updated_rows
    st.session_state["message"] = "出走表を反映しました。"
    st.rerun()

st.markdown("---")
st.subheader("現在の出走表")
current_df = get_df().sort_values("車番").reset_index(drop=True)
st.dataframe(current_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("AI予想")
st.caption(learning_summary_text(LOG_PATH))
repair_log_purchase_amounts()
st.caption(roi_learning_summary_text(LOG_PATH))
st.caption("見送りAIは買い/軽く買い/注意/見送りを判定します。")
st.caption("賭け金AIはAI評価・期待値・見送りAI判定から購入金額を自動配分します。")

detected_mode = auto_detect_mode(current_df)
if race_type == "ガールズ":
    st.info("モード: ガールズモード（ライン評価なし）")
else:
    st.info(f"モード自動判定: {detected_mode}")

st.caption(f"券種: {ticket_type} / 天候: {weather} / レース種別: {race_type} / 予想スタイル: {prediction_style} / 買い目点数: {display_count}点")

p1, p2 = st.columns([1, 1])

with p1:
    if st.button("買い目を出す", type="primary", use_container_width=True):
        try:
            pred_df = generate_predictions_compat(
                current_df=current_df,
                detected_mode=detected_mode,
                weather=weather,
                display_count=display_count,
                odds_dict=st.session_state.get("odds_dict", {}),
                ticket_type=ticket_type,
                race_type=race_type,
            )
            pred_df = apply_learning_correction(
                pred_df,
                LOG_PATH,
                mode=detected_mode,
                weather=weather,
                ticket_type=ticket_type,
            )
            repair_log_purchase_amounts()
            pred_df = apply_roi_learning(
                pred_df,
                LOG_PATH,
                mode=detected_mode,
                weather=weather,
                ticket_type=ticket_type,
            )
            pred_df = apply_roi_ticket_ranking(pred_df)

            if strict_ai_on:
                pred_df, selection_info = apply_strict_selection_ai(
                    pred_df,
                    max_count=min(int(strict_max_count), int(display_count)),
                    min_roi=float(strict_min_roi),
                    keep_min=int(style_settings.get("strict_keep_min", 2)),
                )
            else:
                selection_info = {
                    "enabled": False,
                    "before_count": len(pred_df) if pred_df is not None else 0,
                    "after_count": len(pred_df) if pred_df is not None else 0,
                    "removed_count": 0,
                    "message": "厳選AIはOFFです。",
                }
            pred_df, style_info = apply_prediction_style_filter(
                pred_df,
                prediction_style=prediction_style,
                max_count=min(int(strict_max_count), int(display_count)),
            )
            selection_info["style"] = prediction_style
            selection_info["style_message"] = style_info.get("message", "")
            selection_info["after_count"] = len(pred_df) if pred_df is not None else 0
            st.session_state["selection_info"] = selection_info

            race_assessment = assess_race_buyability(
                current_df,
                pred_df=ensure_prediction_amounts(pred_df, unit_bet=unit_bet),
                log_path=LOG_PATH,
                mode=detected_mode,
                weather=weather,
                ticket_type=ticket_type,
                race_type=race_type,
            )
            pred_df = apply_race_buyability_to_predictions(pred_df, race_assessment)
            st.session_state["race_assessment"] = race_assessment

            # =========================================================
            # レース選別AI：AIの自信が弱いレースは買い目を表示しない
            # ライン数は参考材料にするが、それだけでは見送りにしない。
            # 主役は「厳選AIの信頼度」「熱🔥/堅の強さ」「上位買い目の質」。
            # 見送りAIは強い警告として扱い、AI信頼度が十分高い場合は買い目表示を許可する。
            # =========================================================
            if race_gate_on:
                strict_decision = str(selection_info.get("decision", ""))
                buyability_decision = str(race_assessment.get("decision", ""))
                strict_confidence = safe_int(selection_info.get("confidence", 0), 0)
                gate_min_confidence = safe_int(style_settings.get("gate_confidence_min", 78), 78)

                ai_decision_ok = strict_decision in ["勝負", "厳選候補"]
                ai_confidence_ok = strict_confidence >= gate_min_confidence

                # 少し緩め版：ライン数・見送りAIだけでは切らない。
                # 明らかにAI信頼度が低い時だけ買い目非表示にする。
                rank_list = pred_df["買い目ランク"].astype(str).tolist() if pred_df is not None and not pred_df.empty and "買い目ランク" in pred_df.columns else []
                strong_rank_exists = any(r in ["熱🔥", "堅", "穴"] for r in rank_list[:5])
                buyability_hard_warning = (buyability_decision == "見送り" and strict_confidence < 45 and not strong_rank_exists)

                show_bets = (
                    (ai_decision_ok or ai_confidence_ok or strong_rank_exists)
                    and strict_confidence >= max(42, gate_min_confidence - 15)
                    and not buyability_hard_warning
                )

                gate_reasons = []
                if not ai_decision_ok:
                    gate_reasons.append(f"AI判定={strict_decision or '不明'}")
                if not ai_confidence_ok:
                    gate_reasons.append(f"AI信頼度{strict_confidence}% < 基準{gate_min_confidence}%")
                if buyability_hard_warning:
                    gate_reasons.append(f"見送りAI={buyability_decision}（AI信頼度がかなり低いため）")

                selection_info["race_gate_on"] = True
                selection_info["gate_confidence_min"] = gate_min_confidence
                selection_info["buyability_decision"] = buyability_decision

                if not show_bets:
                    selection_info["race_gate_blocked"] = True
                    selection_info["gate_reason"] = " / ".join(gate_reasons) if gate_reasons else "AI信頼度不足"
                    selection_info["message"] = (
                        "レース選別AI: 見送り寄りです。"
                        "ただし買い目は確認用に表示します。発信・購入は慎重にしてください。"
                    )
                else:
                    selection_info["race_gate_blocked"] = False
                    selection_info["gate_reason"] = (
                        f"AI信頼度{strict_confidence}% / 基準{gate_min_confidence}% / 見送りAI={buyability_decision or '不明'}"
                    )

            selection_info["race_gate_on"] = bool(race_gate_on)
            if not race_gate_on:
                selection_info["race_gate_blocked"] = False
            st.session_state["selection_info"] = selection_info

            pred_df = apply_rank_based_amounts(pred_df, unit_bet)
            pred_df = ensure_prediction_amounts(pred_df, unit_bet=unit_bet)
            pred_df = apply_staking_ai(
                pred_df,
                unit_bet=unit_bet,
                race_assessment=race_assessment,
            )
            pred_df = apply_roi_ticket_ranking(pred_df)
            pred_df = apply_rank_based_amounts(pred_df, unit_bet)
            pred_df = ensure_prediction_amounts(pred_df, unit_bet=unit_bet)
            st.session_state["pred_df"] = pred_df
            st.session_state["message"] = "買い目生成成功"
            st.rerun()
        except Exception as e:
            st.session_state["message"] = f"予想生成失敗: {e}"

with p2:
    if st.button("この予想を保存", use_container_width=True):
        pred_df = st.session_state.get("pred_df")
        if pred_df is None or pred_df.empty:
            st.error("先に買い目を出してください。")
        else:
            try:
                save_current_prediction(
                    race_name=st.session_state.get("race_name", ""),
                    url=st.session_state.get("last_url", ""),
                    mode=detected_mode,
                    weather=weather,
                    race_type=race_type,
                    lineup_string=st.session_state.get("lineup_string", ""),
                    ticket_type=ticket_type,
                    current_df=current_df,
                    pred_df=pred_df,
                    odds_dict=st.session_state.get("odds_dict", {}),
                    unit_bet=unit_bet,
                    display_count=display_count,
                )
                st.success("予想レースを保存しました。")
            except Exception as e:
                st.error(f"保存失敗: {e}")

pred_df = st.session_state.get("pred_df")

if st.session_state.get("selection_info"):
    si = st.session_state.get("selection_info") or {}
    msg = si.get("message", "")
    if msg:
        if si.get("decision") in ["見送り", "見送り寄り"]:
            st.warning(msg)
        elif si.get("decision") == "勝負":
            st.success(msg)
        else:
            st.info(msg)
        if si.get("enabled"):
            st.caption(
                f"削除 {si.get('removed_count', 0)}点 / "
                f"平均ROI {si.get('avg_roi_before', 0)} → {si.get('avg_roi_after', 0)} / "
                f"下限 {si.get('min_roi', 0)}"
            )
        if si.get("style_message"):
            st.caption(si.get("style_message"))
        if si.get("race_gate_blocked"):
            st.warning("レース選別AI: 見送り寄りです。ただし買い目は表示しています。")
            if si.get("gate_reason"):
                st.caption(f"理由: {si.get('gate_reason')}")

if st.session_state.get("race_assessment"):
    ra = st.session_state.get("race_assessment")
    if ra.get("decision") in ["買い", "軽く買い"]:
        st.success(race_buyability_summary_text(ra))
    elif ra.get("decision") == "注意":
        st.warning(race_buyability_summary_text(ra))
    else:
        st.error(race_buyability_summary_text(ra))
    if ra.get("advice"):
        st.caption(ra.get("advice"))

if pred_df is not None and isinstance(pred_df, pd.DataFrame) and not pred_df.empty:
    show_df = pred_df.copy()

    cols_order = [
        c for c in [
            "レース判定", "的中率評価", "レース評価点", "判定理由", "見送りAIコメント",
            "予想スタイル", "買い目ランク", "買い目", "AI評価", "期待値", "学習補正", "学習理由",
            "オッズ", "厚張り指数", "賭け金AI係数", "賭け金AI理由", "購入金額", "期待回収額(目安)"
        ]
        if c in show_df.columns
    ]
    remain = [c for c in show_df.columns if c not in cols_order]
    show_df = show_df[cols_order + remain]

    if "レース判定" in show_df.columns:
        race_decision = str(show_df.iloc[0]["レース判定"])
        hit_label = str(show_df.iloc[0].get("的中率評価", ""))
        race_score = str(show_df.iloc[0].get("レース評価点", ""))
        reason = str(show_df.iloc[0].get("判定理由", ""))

        if race_decision == "買い":
            st.success(f"レース判定: {race_decision} / 的中率評価: {hit_label} / 評価点: {race_score}")
        elif race_decision == "見送り":
            st.warning(f"レース判定: {race_decision} / 的中率評価: {hit_label} / 評価点: {race_score}")
        else:
            st.info(f"レース判定: {race_decision} / 的中率評価: {hit_label} / 評価点: {race_score}")

        if reason:
            st.caption(f"判定理由: {reason}")

    st.dataframe(show_df, use_container_width=True, hide_index=True)
    st.caption(staking_summary_text(show_df))
    st.metric("合計購入額", f"{int(pd.to_numeric(show_df['購入金額'], errors='coerce').fillna(0).sum()):,}円")
else:
    st.caption("まだ買い目は出していません。")

st.markdown("---")
st.subheader("回収率集計")

repair_log_purchase_amounts()
log_df = load_log_df()
summary = summarize_log_df(log_df)

m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
m1.metric("保存件数", f"{summary['race_count']}件")
m2.metric("結果保存件数", f"{summary['result_saved_race_count']}件")
m3.metric("的中件数", f"{summary['hit_race_count']}件")
m4.metric("的中率", f"{summary['hit_rate']}%")
m5.metric("投資額", f"{summary['total_invest']:,}円")
m6.metric("払戻額", f"{summary['total_return']:,}円")
m7.metric("回収率", f"{summary['recovery_rate']}%")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["券種別", "モード別", "天候別", "レース種別", "レース別直近20件"])

with tab1:
    if summary["by_ticket_type"].empty:
        st.caption("まだ集計データがありません。")
    else:
        st.dataframe(summary["by_ticket_type"], use_container_width=True, hide_index=True)

with tab2:
    if summary["by_mode"].empty:
        st.caption("まだ集計データがありません。")
    else:
        st.dataframe(summary["by_mode"], use_container_width=True, hide_index=True)

with tab3:
    if summary["by_weather"].empty:
        st.caption("まだ集計データがありません。")
    else:
        st.dataframe(summary["by_weather"], use_container_width=True, hide_index=True)

with tab4:
    if summary["by_race_type"].empty:
        st.caption("まだ集計データがありません。")
    else:
        st.dataframe(summary["by_race_type"], use_container_width=True, hide_index=True)

with tab5:
    if summary["recent_races"].empty:
        st.caption("まだ集計データがありません。")
    else:
        st.dataframe(summary["recent_races"], use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("保存した予想レース一覧")

saved_items = load_saved_races()
if not saved_items:
    st.caption("まだ保存レースはありません。")
else:
    labels = [saved_race_label(x) for x in saved_items]
    label_to_id = {saved_race_label(x): x.get("id") for x in saved_items}

    selected_label = st.selectbox("保存レースを選択", options=labels)
    selected_saved_id = label_to_id.get(selected_label, "")
    selected_item = get_saved_race(selected_saved_id)

    if selected_item:
        result_text = format_saved_result(selected_item)
        hit_ticket = format_saved_hit_ticket(selected_item)

        s1, s2, s3, s4, s5, s6, s7, s8 = st.columns([2, 1, 1, 1, 1, 1, 1.2, 1.2])
        s1.write(f"**レース名:** {selected_item.get('race_name', '')}")
        s2.write(f"**種別:** {selected_item.get('race_type', '通常')}")
        s3.write(f"**券種:** {selected_item.get('ticket_type', '3連単')}")
        s4.write(f"**モード:** {selected_item.get('mode', '')}")
        s5.write(f"**天候:** {selected_item.get('weather', '')}")
        s6.write(f"**判定:** {saved_race_status_label(selected_item)}")
        s7.write(f"**結果:** {result_text if result_text else '-'}")
        s8.write(f"**的中買い目:** {hit_ticket if hit_ticket else '-'}")

        b1, b2 = st.columns([1, 1])

        with b1:
            if st.button("この保存レースを読込", use_container_width=True):
                restore_saved_race_to_session(selected_item)
                st.rerun()

        with b2:
            if st.button("この保存レースを削除", use_container_width=True):
                ok = delete_saved_race(selected_saved_id)
                if ok:
                    st.success("削除しました。")
                    st.rerun()
                else:
                    st.error("削除に失敗しました。")

        saved_pred_rows = selected_item.get("pred_rows", [])
        if saved_pred_rows:
            st.markdown("#### 保存済み買い目")
            st.dataframe(pd.DataFrame(saved_pred_rows), use_container_width=True, hide_index=True)

        st.markdown("#### 一覧から結果保存")
        default_result = selected_item.get("result", {})
        selected_ticket_type = selected_item.get("ticket_type", "3連単")

        with st.form("result_save_from_list"):
            rr1, rr2, rr3 = st.columns(3)
            result_1 = rr1.text_input("1着", value=str(default_result.get("1着", "")))
            result_2 = rr2.text_input("2着", value=str(default_result.get("2着", "")))
            result_3 = rr3.text_input("3着", value=str(default_result.get("3着", "")))
            submit_result = st.form_submit_button("この保存レースに結果を保存", use_container_width=True)

        if selected_ticket_type == "2車単":
            st.caption("2車単判定は 1着-2着 で行います。3着は保存だけされます。")

        if submit_result:
            try:
                saved_pred_df = pd.DataFrame(saved_pred_rows)

                hit_info = judge_hit(
                    ticket_type=selected_ticket_type,
                    pred_df=ensure_prediction_amounts(saved_pred_df, unit_bet=100),
                    result_1=result_1,
                    result_2=result_2,
                    result_3=result_3,
                )

                save_result_log(
                    race_name=selected_item.get("race_name", ""),
                    mode=selected_item.get("mode", ""),
                    weather=selected_item.get("weather", ""),
                    race_type=selected_item.get("race_type", "通常"),
                    lineup=selected_item.get("lineup_string", ""),
                    ticket_type=selected_ticket_type,
                    pred_df=saved_pred_df,
                    result_1=result_1,
                    result_2=result_2,
                    result_3=result_3,
                    hit_status=hit_info["status_label"],
                )

                update_saved_race(
                    selected_saved_id,
                    {
                        "result_saved": True,
                        "hit_status": hit_info["status_label"],
                        "result": {
                            "1着": result_1,
                            "2着": result_2,
                            "3着": result_3,
                            "result_text": hit_info["result_text"],
                            "hit_ticket": hit_info["hit_ticket"],
                            "saved_at": now_str(),
                        },
                    },
                )
                st.success(f"結果を保存しました: {LOG_PATH.name} / 判定: {hit_info['status_label']}")
                st.rerun()
            except Exception as e:
                st.error(f"保存失敗: {e}")

st.markdown("---")
st.subheader("現在表示中の予想をそのまま結果保存")

if pred_df is not None and isinstance(pred_df, pd.DataFrame) and not pred_df.empty:
    with st.form("direct_result_form"):
        r1, r2, r3 = st.columns(3)
        result_1 = r1.text_input("1着", value="", key="result_1")
        result_2 = r2.text_input("2着", value="", key="result_2")
        result_3 = r3.text_input("3着", value="", key="result_3")
        save_now = st.form_submit_button("結果を保存", use_container_width=True)

    if ticket_type == "2車単":
        st.caption("2車単判定は 1着-2着 で行います。")

    if save_now:
        try:
            hit_info = judge_hit(
                ticket_type=ticket_type,
                pred_df=pred_df,
                result_1=result_1,
                result_2=result_2,
                result_3=result_3,
            )

            save_result_log(
                race_name=st.session_state.get("race_name", ""),
                mode=detected_mode,
                weather=weather,
                race_type=race_type,
                lineup=st.session_state.get("lineup_string", ""),
                ticket_type=ticket_type,
                pred_df=pred_df,
                result_1=result_1,
                result_2=result_2,
                result_3=result_3,
                hit_status=hit_info["status_label"],
            )
            st.success(f"結果を保存しました: {LOG_PATH.name} / 判定: {hit_info['status_label']}")
        except Exception as e:
            st.error(f"保存失敗: {e}")
else:
    st.caption("先に買い目を出してください。")

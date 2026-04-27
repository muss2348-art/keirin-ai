# roi_learning.py
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


def normalize_ticket(ticket: str) -> str:
    return str(ticket).replace(" ", "").strip()


def load_roi_log(log_path) -> pd.DataFrame:
    path = Path(log_path)
    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except Exception:
            return pd.DataFrame()

    if df.empty:
        return df

    for col in ["買い目", "判定", "券種", "モード", "天候", "結果"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    for col in ["購入金額", "オッズ", "期待値", "AI評価"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["買い目"] = df["買い目"].astype(str).map(normalize_ticket)
    df["的中フラグ"] = df["判定"].astype(str).eq("的中")
    df["払戻額"] = df.apply(
        lambda r: safe_float(r["購入金額"]) * safe_float(r["オッズ"]) if r["的中フラグ"] else 0.0,
        axis=1,
    )

    return df


def build_roi_profile(log_df: pd.DataFrame) -> dict:
    profile = {
        "ready": False,
        "ticket_roi_bonus": {},
        "head_roi_bonus": {},
        "mode_roi_bonus": {},
        "weather_roi_bonus": {},
        "ticket_type_roi_bonus": {},
        "summary": {
            "total_rows": 0,
            "total_bet": 0,
            "total_return": 0,
            "roi": 0.0,
        },
    }

    if log_df is None or log_df.empty:
        return profile

    work = log_df.copy()

    total_rows = len(work)
    total_bet = float(work["購入金額"].sum())
    total_return = float(work["払戻額"].sum())
    roi = total_return / total_bet if total_bet > 0 else 0.0

    profile["summary"] = {
        "total_rows": total_rows,
        "total_bet": int(total_bet),
        "total_return": int(total_return),
        "roi": round(roi, 3),
    }

    if total_rows < 10 or total_bet <= 0:
        return profile

    profile["ready"] = True

    def roi_bonus_from_value(group_roi: float, count: int) -> float:
        if count < 2:
            return 0.0

        if group_roi >= 1.8:
            return 10.0
        if group_roi >= 1.3:
            return 6.0
        if group_roi >= 1.05:
            return 3.0
        if group_roi < 0.45 and count >= 4:
            return -8.0
        if group_roi < 0.75 and count >= 3:
            return -4.0
        if group_roi < 0.95 and count >= 3:
            return -2.0

        return 0.0

    # 買い目別ROI
    g_ticket = work.groupby("買い目").agg(
        count=("買い目", "count"),
        bet=("購入金額", "sum"),
        ret=("払戻額", "sum"),
    ).reset_index()

    for _, r in g_ticket.iterrows():
        ticket = str(r["買い目"])
        count = int(r["count"])
        bet = safe_float(r["bet"])
        ret = safe_float(r["ret"])

        if not ticket or bet <= 0:
            continue

        group_roi = ret / bet
        bonus = roi_bonus_from_value(group_roi, count)

        if bonus != 0:
            profile["ticket_roi_bonus"][ticket] = round(bonus, 2)

    # 頭別ROI
    work["頭"] = work["買い目"].str.split("-").str[0]

    g_head = work.groupby("頭").agg(
        count=("頭", "count"),
        bet=("購入金額", "sum"),
        ret=("払戻額", "sum"),
    ).reset_index()

    for _, r in g_head.iterrows():
        head = str(r["頭"])
        count = int(r["count"])
        bet = safe_float(r["bet"])
        ret = safe_float(r["ret"])

        if not head or bet <= 0:
            continue

        group_roi = ret / bet

        if count < 4:
            continue

        if group_roi >= 1.4:
            bonus = 5.0
        elif group_roi >= 1.1:
            bonus = 3.0
        elif group_roi < 0.6:
            bonus = -5.0
        elif group_roi < 0.85:
            bonus = -3.0
        else:
            bonus = 0.0

        if bonus != 0:
            profile["head_roi_bonus"][head] = round(bonus, 2)

    # モード別ROI
    if "モード" in work.columns:
        g_mode = work.groupby("モード").agg(
            count=("モード", "count"),
            bet=("購入金額", "sum"),
            ret=("払戻額", "sum"),
        ).reset_index()

        for _, r in g_mode.iterrows():
            key = str(r["モード"])
            count = int(r["count"])
            bet = safe_float(r["bet"])
            ret = safe_float(r["ret"])

            if not key or bet <= 0 or count < 5:
                continue

            group_roi = ret / bet

            if group_roi >= 1.25:
                bonus = 3.0
            elif group_roi < 0.75:
                bonus = -3.0
            else:
                bonus = 0.0

            if bonus != 0:
                profile["mode_roi_bonus"][key] = round(bonus, 2)

    # 天候別ROI
    if "天候" in work.columns:
        g_weather = work.groupby("天候").agg(
            count=("天候", "count"),
            bet=("購入金額", "sum"),
            ret=("払戻額", "sum"),
        ).reset_index()

        for _, r in g_weather.iterrows():
            key = str(r["天候"])
            count = int(r["count"])
            bet = safe_float(r["bet"])
            ret = safe_float(r["ret"])

            if not key or bet <= 0 or count < 5:
                continue

            group_roi = ret / bet

            if group_roi >= 1.25:
                bonus = 2.0
            elif group_roi < 0.75:
                bonus = -2.0
            else:
                bonus = 0.0

            if bonus != 0:
                profile["weather_roi_bonus"][key] = round(bonus, 2)

    # 券種別ROI
    if "券種" in work.columns:
        g_type = work.groupby("券種").agg(
            count=("券種", "count"),
            bet=("購入金額", "sum"),
            ret=("払戻額", "sum"),
        ).reset_index()

        for _, r in g_type.iterrows():
            key = str(r["券種"])
            count = int(r["count"])
            bet = safe_float(r["bet"])
            ret = safe_float(r["ret"])

            if not key or bet <= 0 or count < 5:
                continue

            group_roi = ret / bet

            if group_roi >= 1.25:
                bonus = 2.0
            elif group_roi < 0.75:
                bonus = -2.0
            else:
                bonus = 0.0

            if bonus != 0:
                profile["ticket_type_roi_bonus"][key] = round(bonus, 2)

    return profile


def apply_roi_learning(
    pred_df: pd.DataFrame,
    log_path,
    mode: str = "",
    weather: str = "",
    ticket_type: str = "",
) -> pd.DataFrame:
    if pred_df is None or pred_df.empty:
        return pred_df

    log_df = load_roi_log(log_path)
    profile = build_roi_profile(log_df)

    out = pred_df.copy()

    if "買い目" not in out.columns:
        return out

    if "AI評価" not in out.columns:
        out["AI評価"] = 0

    out["AI評価"] = pd.to_numeric(out["AI評価"], errors="coerce").fillna(0)
    out["ROI補正"] = 0.0
    out["ROI理由"] = ""

    if not profile.get("ready"):
        out["ROI理由"] = "ROIログ不足のため補正なし"
        return out

    for idx, row in out.iterrows():
        ticket = normalize_ticket(row.get("買い目", ""))
        head = ticket.split("-")[0] if ticket else ""

        bonus = 0.0
        reasons = []

        if ticket in profile["ticket_roi_bonus"]:
            b = safe_float(profile["ticket_roi_bonus"][ticket])
            bonus += b
            reasons.append(f"買い目ROI補正 {b:+.1f}")

        if head in profile["head_roi_bonus"]:
            b = safe_float(profile["head_roi_bonus"][head])
            bonus += b
            reasons.append(f"頭{head}ROI補正 {b:+.1f}")

        if mode in profile["mode_roi_bonus"]:
            b = safe_float(profile["mode_roi_bonus"][mode])
            bonus += b
            reasons.append(f"モードROI補正 {b:+.1f}")

        if weather in profile["weather_roi_bonus"]:
            b = safe_float(profile["weather_roi_bonus"][weather])
            bonus += b
            reasons.append(f"天候ROI補正 {b:+.1f}")

        if ticket_type in profile["ticket_type_roi_bonus"]:
            b = safe_float(profile["ticket_type_roi_bonus"][ticket_type])
            bonus += b
            reasons.append(f"券種ROI補正 {b:+.1f}")

        out.at[idx, "ROI補正"] = round(bonus, 2)
        out.at[idx, "AI評価"] = round(safe_float(out.at[idx, "AI評価"]) + bonus, 2)
        out.at[idx, "ROI理由"] = " / ".join(reasons) if reasons else "ROI補正なし"

    if "期待値" in out.columns:
        out["期待値"] = pd.to_numeric(out["期待値"], errors="coerce").fillna(0)
        out["期待値"] = (out["期待値"] + out["ROI補正"] * 1.8).round(1)

    sort_cols = ["AI評価"]
    if "期待値" in out.columns:
        sort_cols.append("期待値")

    out = out.sort_values(sort_cols, ascending=False).reset_index(drop=True)

    return out


def roi_learning_summary_text(log_path) -> str:
    log_df = load_roi_log(log_path)
    profile = build_roi_profile(log_df)

    total = profile["summary"]["total_rows"]
    total_bet = profile["summary"]["total_bet"]
    total_return = profile["summary"]["total_return"]
    roi = profile["summary"]["roi"]

    if total == 0:
        return "ROI学習ログなし"
    if not profile["ready"]:
        return f"ROI学習ログ {total}件 / 投資{total_bet:,}円 / 払戻{total_return:,}円：まだ補正なし（10件以上で開始）"

    return f"ROI学習ON：ログ{total}件 / 投資{total_bet:,}円 / 払戻{total_return:,}円 / ROI {roi * 100:.1f}%"

# learning.py
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


def load_learning_log(log_path) -> pd.DataFrame:
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

    for col in ["オッズ", "購入金額", "期待値"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def build_learning_profile(log_df: pd.DataFrame) -> dict:
    profile = {
        "ready": False,
        "ticket_bonus": {},
        "head_bonus": {},
        "mode_bonus": {},
        "weather_bonus": {},
        "ticket_type_bonus": {},
        "summary": {
            "total_rows": 0,
            "hit_rows": 0,
        },
    }

    if log_df is None or log_df.empty:
        return profile

    work = log_df.copy()
    work["買い目"] = work["買い目"].astype(str).map(normalize_ticket)
    work["hit"] = work["判定"].astype(str).eq("的中")

    total_rows = len(work)
    hit_rows = int(work["hit"].sum())

    profile["summary"]["total_rows"] = total_rows
    profile["summary"]["hit_rows"] = hit_rows

    if total_rows < 10:
        return profile

    profile["ready"] = True

    # 買い目別補正
    g_ticket = work.groupby("買い目").agg(
        count=("買い目", "count"),
        hits=("hit", "sum"),
    ).reset_index()

    for _, r in g_ticket.iterrows():
        ticket = str(r["買い目"])
        count = int(r["count"])
        hits = int(r["hits"])

        if not ticket or count < 2:
            continue

        rate = hits / count if count else 0

        if rate >= 0.35:
            profile["ticket_bonus"][ticket] = 8.0
        elif rate >= 0.2:
            profile["ticket_bonus"][ticket] = 4.0
        elif rate == 0 and count >= 4:
            profile["ticket_bonus"][ticket] = -5.0

    # 頭別補正
    work["頭"] = work["買い目"].str.split("-").str[0]

    g_head = work.groupby("頭").agg(
        count=("頭", "count"),
        hits=("hit", "sum"),
    ).reset_index()

    for _, r in g_head.iterrows():
        head = str(r["頭"])
        count = int(r["count"])
        hits = int(r["hits"])

        if not head or count < 4:
            continue

        rate = hits / count if count else 0

        if rate >= 0.25:
            profile["head_bonus"][head] = 5.0
        elif rate <= 0.05:
            profile["head_bonus"][head] = -4.0

    # モード別補正
    if "モード" in work.columns:
        g_mode = work.groupby("モード").agg(
            count=("モード", "count"),
            hits=("hit", "sum"),
        ).reset_index()

        for _, r in g_mode.iterrows():
            key = str(r["モード"])
            count = int(r["count"])
            hits = int(r["hits"])

            if not key or count < 5:
                continue

            rate = hits / count
            if rate >= 0.25:
                profile["mode_bonus"][key] = 3.0
            elif rate <= 0.05:
                profile["mode_bonus"][key] = -3.0

    # 天候別補正
    if "天候" in work.columns:
        g_weather = work.groupby("天候").agg(
            count=("天候", "count"),
            hits=("hit", "sum"),
        ).reset_index()

        for _, r in g_weather.iterrows():
            key = str(r["天候"])
            count = int(r["count"])
            hits = int(r["hits"])

            if not key or count < 5:
                continue

            rate = hits / count
            if rate >= 0.25:
                profile["weather_bonus"][key] = 2.0
            elif rate <= 0.05:
                profile["weather_bonus"][key] = -2.0

    # 券種別補正
    if "券種" in work.columns:
        g_type = work.groupby("券種").agg(
            count=("券種", "count"),
            hits=("hit", "sum"),
        ).reset_index()

        for _, r in g_type.iterrows():
            key = str(r["券種"])
            count = int(r["count"])
            hits = int(r["hits"])

            if not key or count < 5:
                continue

            rate = hits / count
            if rate >= 0.25:
                profile["ticket_type_bonus"][key] = 2.0
            elif rate <= 0.05:
                profile["ticket_type_bonus"][key] = -2.0

    return profile


def apply_learning_correction(
    pred_df: pd.DataFrame,
    log_path,
    mode: str = "",
    weather: str = "",
    ticket_type: str = "",
) -> pd.DataFrame:
    if pred_df is None or pred_df.empty:
        return pred_df

    log_df = load_learning_log(log_path)
    profile = build_learning_profile(log_df)

    out = pred_df.copy()

    if "買い目" not in out.columns:
        return out

    if "AI評価" not in out.columns:
        out["AI評価"] = 0

    out["AI評価"] = pd.to_numeric(out["AI評価"], errors="coerce").fillna(0)
    out["学習補正"] = 0.0
    out["学習理由"] = ""

    if not profile.get("ready"):
        out["学習理由"] = "ログ不足のため補正なし"
        return out

    for idx, row in out.iterrows():
        ticket = normalize_ticket(row.get("買い目", ""))
        head = ticket.split("-")[0] if ticket else ""

        bonus = 0.0
        reasons = []

        if ticket in profile["ticket_bonus"]:
            b = safe_float(profile["ticket_bonus"][ticket])
            bonus += b
            reasons.append(f"買い目履歴補正 {b:+.1f}")

        if head in profile["head_bonus"]:
            b = safe_float(profile["head_bonus"][head])
            bonus += b
            reasons.append(f"頭{head}履歴補正 {b:+.1f}")

        if mode in profile["mode_bonus"]:
            b = safe_float(profile["mode_bonus"][mode])
            bonus += b
            reasons.append(f"モード補正 {b:+.1f}")

        if weather in profile["weather_bonus"]:
            b = safe_float(profile["weather_bonus"][weather])
            bonus += b
            reasons.append(f"天候補正 {b:+.1f}")

        if ticket_type in profile["ticket_type_bonus"]:
            b = safe_float(profile["ticket_type_bonus"][ticket_type])
            bonus += b
            reasons.append(f"券種補正 {b:+.1f}")

        out.at[idx, "学習補正"] = round(bonus, 2)
        out.at[idx, "AI評価"] = round(safe_float(out.at[idx, "AI評価"]) + bonus, 2)
        out.at[idx, "学習理由"] = " / ".join(reasons) if reasons else "補正なし"

    if "期待値" in out.columns:
        out["期待値"] = pd.to_numeric(out["期待値"], errors="coerce").fillna(0)
        out["期待値"] = (out["期待値"] + out["学習補正"] * 1.5).round(1)

    out = out.sort_values(["AI評価", "期待値"], ascending=False).reset_index(drop=True)

    return out


def learning_summary_text(log_path) -> str:
    log_df = load_learning_log(log_path)
    profile = build_learning_profile(log_df)

    total = profile["summary"]["total_rows"]
    hits = profile["summary"]["hit_rows"]

    if total == 0:
        return "学習ログなし"
    if not profile["ready"]:
        return f"学習ログ {total}件 / 的中 {hits}件：まだ補正なし（10件以上で開始）"

    return f"学習補正ON：ログ {total}件 / 的中 {hits}件"

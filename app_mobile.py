if "race_rows" not in st.session_state:
    init_state(7)

def extract_single_player_by_car(text, car, num_riders):
    s = normalize_text(text)

    # --------------------------------------------------
    # ★ ① 完全一致（最強）
    # --------------------------------------------------
    exact = re.search(
        rf"{car}\s+{car}\s+"
        rf"([一-龥ぁ-んァ-ヶ々]{{2,12}})\s+"
        rf"({PREF_PATTERN}).{{0,50}}?"
        rf"([4-9]\d(?:\.\d{{1,3}})?)\s+"
        rf"(?:\d+\s+){{2,6}}"
        rf"(逃|捲|追|両|自)",
        s
    )

    if exact:
        name = normalize_text(exact.group(1))
        score = safe_float(exact.group(3))
        style = normalize_text(exact.group(4))

        if is_valid_player_name(name):
            return {
                "車番": car,
                "選手名": name,
                "競走得点": score,
                "脚質": style,
                "source": "exact"
            }

    # --------------------------------------------------
    # ★ ② ブロック抽出（ここが重要）
    # --------------------------------------------------
    next_car = car + 1

    if next_car <= num_riders:
        block_match = re.search(
            rf"{car}\s+{car}\s+(.*?)(?={next_car}\s+{next_car}\s+)",
            s
        )
    else:
        block_match = re.search(
            rf"{car}\s+{car}\s+(.*)$",
            s
        )

    if not block_match:
        return None

    block = normalize_text(block_match.group(1))[:600]

    # -------------------------
    # 名前
    # -------------------------
    name = extract_name(block)

    # -------------------------
    # 点数（←ここ修正ポイント）
    # -------------------------
    score_candidates = []

    for m in re.finditer(r"([4-9]\d(?:\.\d{1,3})?)", block):
        v = safe_float(m.group(1))

        # 除外ルール
        before = block[max(0, m.start()-3):m.start()]
        after = block[m.end():m.end()+3]

        if "期" in before or "期" in after:
            continue
        if "歳" in before or "歳" in after:
            continue

        if 40 <= v <= 130:
            score_candidates.append(v)

    score = max(score_candidates) if score_candidates else 0.0

    # -------------------------
    # 脚質
    # -------------------------
    style_match = re.search(r"(逃|捲|追|両|自)", block)
    style = style_match.group(1) if style_match else ""

    if is_valid_player_name(name) and score > 0:
        return {
            "車番": car,
            "選手名": name,
            "競走得点": score,
            "脚質": style,
            "source": "block_fixed"
        }

    return None

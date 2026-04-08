import streamlit as st
import pandas as pd

st.set_page_config(page_title="競輪AIモバイル", layout="centered")

st.title("🚴 競輪AI予想 モバイル版")
st.caption("軽量版：出走表取得 → モード判定 → 買い目表示 → 結果保存")

# ========================
# 入力エリア
# ========================
st.subheader("レース設定")

race_url = st.text_input("レースURLを入力", placeholder="https://...")

buy_count = st.selectbox(
    "表示する買い目点数",
    [5, 10, 15, 20, 25, 30],
    index=1
)

bet_amount = st.number_input(
    "1点あたり金額",
    min_value=100,
    max_value=10000,
    value=100,
    step=100
)

mode = st.selectbox(
    "予想モード",
    ["自動", "通常", "混戦", "穴"],
    index=0
)

load_button = st.button("出走表を読み込む")

# ========================
# 仮の処理
# ========================
if load_button:
    if not race_url:
        st.warning("レースURLを入力してね")
    else:
        st.success("URLを受け取ったよ")
        st.write("URL:", race_url)
        st.write("買い目点数:", buy_count)
        st.write("1点金額:", bet_amount)
        st.write("モード:", mode)

        # 仮の出走表
        dummy_data = pd.DataFrame([
            {"車番": 1, "選手名": "選手A", "競走得点": 98.5, "脚質": "逃"},
            {"車番": 2, "選手名": "選手B", "競走得点": 95.2, "脚質": "追"},
            {"車番": 3, "選手名": "選手C", "競走得点": 92.8, "脚質": "両"},
            {"車番": 4, "選手名": "選手D", "競走得点": 90.1, "脚質": "追"},
            {"車番": 5, "選手名": "選手E", "競走得点": 89.7, "脚質": "逃"},
            {"車番": 6, "選手名": "選手F", "競走得点": 87.3, "脚質": "両"},
            {"車番": 7, "選手名": "選手G", "競走得点": 85.9, "脚質": "追"},
        ])

        st.subheader("出走表")
        st.dataframe(dummy_data, use_container_width=True)

        # 仮の車立て判定
        num_racers = len(dummy_data)
        if num_racers == 7:
            race_type = "7車"
        elif num_racers == 9:
            race_type = "9車"
        else:
            race_type = "不明"

        st.info(f"車立て判定：{race_type}")

        # 仮のモード判定
        auto_mode = "混戦" if num_racers == 7 else "通常"
        final_mode = auto_mode if mode == "自動" else mode
        st.info(f"予想モード：{final_mode}")

        # 仮の買い目
        st.subheader("AI推奨買い目")
        dummy_bets = [
            "2車単 1-2",
            "2車単 1-3",
            "2車単 2-1",
            "2車単 2-3",
            "2車単 3-1",
        ]

        for bet in dummy_bets[:buy_count]:
            st.write("・", bet)

        total_investment = min(len(dummy_bets), buy_count) * bet_amount
        st.success(f"合計投資額：{total_investment}円")

        # 仮の結果保存欄
        st.subheader("結果保存")
        result = st.selectbox("結果", ["未入力", "的中", "不的中"])
        payout = st.number_input("払戻金", min_value=0, value=0, step=100)

        if st.button("結果を保存"):
            st.success("保存機能は次のステップで追加するよ")

# ===============================
# 競輪AI mobile 完全版（並び取得強化）
# ===============================
import re
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

# ===============================
# selenium
# ===============================
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait
    SELENIUM_AVAILABLE = True
except:
    SELENIUM_AVAILABLE = False

# ===============================
# 設定
# ===============================
st.set_page_config(page_title="競輪AI mobile", layout="centered")

BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = BASE_DIR / "log.csv"
RESULT_LOG_PATH = BASE_DIR / "result_log.csv"
DEBUG_POS_PATH = BASE_DIR / "winticket_debug_positions.csv"

# ===============================
# CSV
# ===============================
def ensure_csv():
    if not LOG_PATH.exists():
        pd.DataFrame(columns=["ID","保存日時","レース名","レースURL","買い目一覧"]).to_csv(LOG_PATH,index=False,encoding="utf-8-sig")
    if not RESULT_LOG_PATH.exists():
        pd.DataFrame(columns=["レース名","着順","払戻"]).to_csv(RESULT_LOG_PATH,index=False,encoding="utf-8-sig")

ensure_csv()

# ===============================
# 並び取得（最新版）
# ===============================
def fetch_line(url, car_count="7車"):
    if not SELENIUM_AVAILABLE:
        return "", "selenium未インストール"

    def cluster(items, gap=22):
        if not items:
            return []
        groups = [[items[0]]]
        for i in items[1:]:
            if i["x"] - groups[-1][-1]["x"] <= gap:
                groups[-1].append(i)
            else:
                groups.append([i])
        return groups

    driver = None
    try:
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")

        driver = webdriver.Chrome(options=options)
        driver.get(url)
        WebDriverWait(driver,20).until(lambda d: d.execute_script("return document.readyState") == "complete")
        time.sleep(5)

        elements = driver.find_elements(By.XPATH,"//*")
        data = []

        for el in elements:
            try:
                txt = el.text.strip()
                if not re.fullmatch(r"[1-9]", txt):
                    continue

                r = el.rect
                x = float(r.get("x",0))
                y = float(r.get("y",0))
                w = float(r.get("width",0))
                h = float(r.get("height",0))

                if w>80 or h>80:
                    continue

                data.append({"num":txt,"x":x,"y":y})
            except:
                pass

        driver.quit()

        if not data:
            return "","取得失敗"

        df = pd.DataFrame(data).drop_duplicates()
        df.to_csv(DEBUG_POS_PATH,index=False)

        need = 7 if car_count=="7車" else 9

        # yでグループ化
        df["y_group"] = (df["y"]//8)*8
        rows = []

        for y, g in df.groupby("y_group"):
            g = g.sort_values("x")
            if len(g) >= 3:
                span = g["x"].max() - g["x"].min()
                if span > 40:
                    rows.append((y,g))

        rows = sorted(rows, key=lambda x:x[0])

        for y, g in rows[:6]:
            items = g.to_dict("records")
            groups = cluster(items)

            parts = []
            used = set()
            total = 0

            for gr in groups:
                nums = []
                for i in gr:
                    if i["num"] not in used:
                        nums.append(i["num"])
                        used.add(i["num"])
                if nums:
                    parts.append(nums)
                    total += len(nums)
                if total>=need:
                    break

            if total>=need:
                result = []
                count = 0
                for p in parts:
                    if count>=need:
                        break
                    seg = p[:need-count]
                    result.append(seg)
                    count += len(seg)

                if count==need:
                    line = " / ".join(" ".join(r) for r in result)
                    return line, f"抽出成功 y={y}"

        return "","並び不明"

    except Exception as e:
        return "",str(e)

    finally:
        try:
            driver.quit()
        except:
            pass

# ===============================
# UI
# ===============================
st.title("🚴 競輪AI mobile")

race = st.text_input("レース名")
url = st.text_input("URL")

if "line" not in st.session_state:
    st.session_state.line = ""

st.session_state.line = st.text_area("並び", value=st.session_state.line)

if st.button("並び自動取得"):
    l, reason = fetch_line(url)
    if l:
        st.session_state.line = l
        st.success(l)
        st.caption(reason)
        st.rerun()
    else:
        st.error(reason)

# ===============================
# 予想（簡易）
# ===============================
if st.button("予想生成"):
    bets = ["1-2-3","1-3-2","2-1-3"]
    st.session_state.bets = bets

if "bets" in st.session_state:
    for b in st.session_state.bets:
        st.write(b)

    if st.button("保存"):
        df = pd.read_csv(LOG_PATH)
        new = pd.DataFrame([{
            "ID":datetime.now().strftime("%Y%m%d%H%M%S"),
            "保存日時":datetime.now(),
            "レース名":race,
            "レースURL":url,
            "買い目一覧":" / ".join(st.session_state.bets)
        }])
        pd.concat([df,new]).to_csv(LOG_PATH,index=False)
        st.success("保存完了")

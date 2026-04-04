# Cloudでも自動取得ON版（完全版）
import re
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

# selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait

st.set_page_config(page_title="競輪AI mobile", layout="centered")

BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = BASE_DIR / "log.csv"
RESULT_LOG_PATH = BASE_DIR / "result_log.csv"

# =====================================
# CSV
# =====================================
def ensure_csv():
    if not LOG_PATH.exists():
        pd.DataFrame(columns=["ID","保存日時","レース名","レースURL","券種","モード","モード判定理由","買い目点数","1点金額","合計金額","買い目一覧","メモ"]).to_csv(LOG_PATH,index=False,encoding="utf-8-sig")
    if not RESULT_LOG_PATH.exists():
        pd.DataFrame(columns=["予想ID","登録日時","レース名","着順1","着順2","着順3","的中","払戻","投資額","収支"]).to_csv(RESULT_LOG_PATH,index=False,encoding="utf-8-sig")

ensure_csv()

# =====================================
# 並び解析
# =====================================
def parse_line(text):
    text = text.replace("/","|").replace("\n","|")
    groups = [g.strip() for g in text.split("|") if g.strip()]
    return [g.split() for g in groups]

# =====================================
# モード判定
# =====================================
def detect_mode(text):
    lines = parse_line(text)
    single = sum(1 for l in lines if len(l)==1)
    if single>=3:
        return "穴モード"
    if single>=2:
        return "混戦モード"
    return "通常モード"

# =====================================
# 並び取得（Cloud対応）
# =====================================
def fetch_line(url):
    try:
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")

        driver = webdriver.Chrome(options=options)
        driver.get(url)
        WebDriverWait(driver,10).until(lambda d: d.execute_script("return document.readyState")=="complete")
        time.sleep(5)

        body = driver.find_element(By.TAG_NAME,"body").text
        driver.quit()

        nums = re.findall(r"\b[1-9]\b",body)

        if len(nums)>=7:
            # 超簡易整形
            return f"{' '.join(nums[:3])} / {nums[3]} / {' '.join(nums[4:6])} / {nums[6]}", "簡易取得"

        return "","取得失敗"

    except Exception as e:
        return "",str(e)

# =====================================
# 保存
# =====================================
def save_pred(name,url,bets,mode):
    df = pd.read_csv(LOG_PATH,encoding="utf-8-sig")
    row = {
        "ID":datetime.now().strftime("%Y%m%d%H%M%S"),
        "保存日時":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "レース名":name,
        "レースURL":url,
        "券種":"3連単",
        "モード":mode,
        "モード判定理由":"",
        "買い目点数":len(bets),
        "1点金額":100,
        "合計金額":len(bets)*100,
        "買い目一覧":" / ".join(bets),
        "メモ":""
    }
    pd.concat([df,pd.DataFrame([row])]).to_csv(LOG_PATH,index=False,encoding="utf-8-sig")

# =====================================
# UI
# =====================================
st.title("🚴 競輪AI mobile")

name = st.text_input("レース名")
url = st.text_input("URL")
line = st.text_area("並び")

if st.button("並び自動取得"):
    l,reason = fetch_line(url)
    if l:
        line = l
        st.success(l)
    else:
        st.error(reason)

mode = detect_mode(line)
st.write("モード:",mode)

if st.button("予想生成"):
    bets = ["1-2-3","1-3-2","2-1-3","3-1-2"]
    st.session_state["bets"] = bets

if "bets" in st.session_state:
    for b in st.session_state["bets"]:
        st.write(b)

    if st.button("保存"):
        save_pred(name,url,st.session_state["bets"],mode)
        st.success("保存完了")

# =====================================
# 結果入力
# =====================================
st.divider()
st.subheader("結果入力")

rank1 = st.number_input("1着",1,9,1)
rank2 = st.number_input("2着",1,9,2)
rank3 = st.number_input("3着",1,9,3)
payout = st.number_input("払戻",0,100000,0)

if st.button("結果保存"):
    df = pd.read_csv(RESULT_LOG_PATH,encoding="utf-8-sig")
    row = {
        "予想ID":"-",
        "登録日時":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "レース名":name,
        "着順1":rank1,
        "着順2":rank2,
        "着順3":rank3,
        "的中":"-",
        "払戻":payout,
        "投資額":0,
        "収支":payout
    }
    pd.concat([df,pd.DataFrame([row])]).to_csv(RESULT_LOG_PATH,index=False,encoding="utf-8-sig")
    st.success("結果保存完了")


# ★ Cloud対応版（selenium分岐あり）
import re
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

# ===== seleniumは環境で分岐 =====
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait
    SELENIUM_AVAILABLE = True
except:
    SELENIUM_AVAILABLE = False

# ===== Cloud判定 =====
def is_cloud():
    try:
        return "streamlit.app" in st.runtime.get_instance()._runtime_config.server_address
    except:
        return False

# =====================================
st.set_page_config(page_title="競輪AI mobile", layout="centered")

BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = BASE_DIR / "log.csv"
RESULT_LOG_PATH = BASE_DIR / "result_log.csv"

# =====================================
def ensure_csv():
    if not LOG_PATH.exists():
        pd.DataFrame(columns=["ID","保存日時","レース名","レースURL","券種","モード","モード判定理由","買い目点数","1点金額","合計金額","買い目一覧","メモ"]).to_csv(LOG_PATH,index=False,encoding="utf-8-sig")
    if not RESULT_LOG_PATH.exists():
        pd.DataFrame(columns=["予想ID","登録日時","レース名","着順1","着順2","着順3","的中","払戻","投資額","収支"]).to_csv(RESULT_LOG_PATH,index=False,encoding="utf-8-sig")

ensure_csv()

# =====================================
def parse_line_text(text):
    text = text.replace("/","|").replace("\n","|")
    groups = [g.strip() for g in text.split("|") if g.strip()]
    lines = []
    for g in groups:
        nums = [x for x in g.split() if x.isdigit()]
        if nums:
            lines.append(nums)
    return lines


def auto_detect_mode(text):
    lines = parse_line_text(text)
    if not lines:
        return "通常モード"
    single = sum(1 for l in lines if len(l)==1)
    if single>=3:
        return "穴モード"
    if single>=2:
        return "混戦モード"
    return "通常モード"

# =====================================
def fetch_line(url):
    if is_cloud():
        return "", "Cloudでは自動取得不可（手入力してください）"

    if not SELENIUM_AVAILABLE:
        return "", "selenium未インストール"

    try:
        options = Options()
        driver = webdriver.Chrome(options=options)
        driver.get(url)
        time.sleep(5)
        body = driver.find_element(By.TAG_NAME, "body").text
        driver.quit()

        nums = re.findall(r"\b[1-9]\b", body)
        if len(nums)>=7:
            return " ".join(nums[:7]), "簡易取得"
        return "", "取得失敗"
    except Exception as e:
        return "", str(e)

# =====================================
def save_log(name,url,bets,mode):
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
st.title("🚴 競輪AI mobile")

name = st.text_input("レース名")
url = st.text_input("URL")
line = st.text_area("並び")

if st.button("並び取得"):
    l,reason = fetch_line(url)
    if l:
        st.success(l)
        line = l
    else:
        st.error(reason)

mode = auto_detect_mode(line)
st.write("モード:",mode)

if st.button("予想生成"):
    bets = ["1-2-3","1-3-2","2-1-3","3-1-2"]
    st.session_state["bets"] = bets

if "bets" in st.session_state:
    for b in st.session_state["bets"]:
        st.write(b)

    if st.button("保存"):
        save_log(name,url,st.session_state["bets"],mode)
        st.success("保存完了")

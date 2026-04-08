import streamlit as st

st.set_page_config(page_title="MOBILE TEST", layout="centered")
st.title("MOBILE TEST 12345")
st.write("これは app_mobile.py です")
st.stop()



import os
import streamlit as st

st.title("確認用アプリ")
st.write("これは新しい app_mobile.py です")
st.write("実行ファイル:", os.path.abspath(__file__))

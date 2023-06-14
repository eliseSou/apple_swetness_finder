import streamlit as st
import pandas as pd
import numpy as np

from time import sleep

# 페이지 기본 설정
st.set_page_config(
    page_title="Team 9",
    page_icon=":rocket:",
    layout="centered",
    initial_sidebar_state="auto",
    )

# 로딩바 구현하기
with st.spinner(text="Loading..."):
    sleep(2)

# 페이지 헤더, 서브헤더 제목 설정
st.header("👋Welcome")
st.subheader("Hello. We are team9")

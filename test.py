import streamlit as st

st.set_page_config(layout="wide")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Dashboard", "About"]
)

st.title(page)
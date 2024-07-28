import streamlit as st

# Page Set Up
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="images/bird.png",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={'About': 'NLP Sentiment Analysis'}
)

# Styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #DDEEFF; 
    }
    .title {
        text-align: center;
        font-size: 2em;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .header {
        text-align: center;
        font-size: 1em;
        font-weight: bold;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

IMAGE_PATH = "images/bird.png"
# Logo
st.logo(IMAGE_PATH)

# Navigation and it's paages
pg = st.navigation([
    st.Page("page1.py", title="Prediction Page", icon=":material/thumb_up:"), 
    st.Page("page2.py", title="About Page", icon=":material/analytics:"), 
    st.Page("page3.py", title="Contacts Page", icon=":material/contact_page:")])
pg.run()
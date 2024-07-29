import streamlit as st
import numpy as np
from classes import execute_flow
import time

st.header("SentimentFlow Sentiment Analysis", divider=True)
st.write("This application uses the Unsupervised Machine Learning - Natural Language Preprocessing to analyse the sentiment behind a text.")

st.image("app/images/word_cloud_short.png")

# The Form Container
tweet_container = st.empty()

with tweet_container.container():

   col1, col2 = st.columns([7,3])

   with col1:
       st.subheader("Tweet Here :bird:")
       with st.form(key='form1'):
           username = st.text_input("Username", placeholder="Enter your Username")
           products = ['Google', 'Apple', 'iPad', 'iPhone', 'Apple Product', 'Google Product']
           product = st.selectbox("Which product do you want to talk about", products)
           sentiment = st.text_area("Tweet", placeholder="Post your Tweet")
           predict_button = st.form_submit_button('Analyze')


   with col2:
        st.subheader("Prediction :telescope:")
        info_text = st.info("Enter text to predict sentiment")
        if predict_button:
            info_text.empty()
            if username == "" and sentiment == "":
                st.error("Please fill in the previous form - Username and Tweet Fields")
            elif username == "":
                st.info("Please fill in the previous form - Username Field", icon="ℹ️")
            elif sentiment == "":
                st.info("Please fill in the previous form - Tweet Field", icon="ℹ️")
            else:
                st.write(f"The user @{username.lower().replace(' ', '_')} focused on the product: {product} and said '{sentiment}'")
                randoms = ['Thinking', 'Processing', 'Considering', 'Wizardy Applying', 'Hmmm', 'Contacting the Mothership', "Predicting",
                "Checking under the bed for an answer"]
                answers = randoms[np.random.choice(len(randoms))]
                with st.spinner(f"{answers}..."):
                    time.sleep(3)
                    st.success(f"Our model predicted the sentiment to be a {execute_flow(sentiment)}!")

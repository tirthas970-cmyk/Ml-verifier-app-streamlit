
from AIAssistant import Assistant
import streamlit as st

assistant = Assistant() 

st.set_page_config(page_title="MLFactCheckingbot", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #1a1a1a; color: white; }
    h1 { color: #FFBE46; text-align: center; font-family: 'Helvetica'; }
    .stButton>button { background-color: #FFBE46; color: black; font-weight: bold; width: 100%; }
    </style>
    """, unsafe_allow_html=True)

st.title("💡 High-Precision Topic Verifier Bot")

action = st.selectbox("Select Action", ["Select Action", "Find Definition of words or sentences", "Get Information on a Certain Topic"])

if action == "Select Action":
    st.markdown("### 👋 Welcome to the Verifier Bot!")
    st.write("Please choose an action from the dropdown menu above to get started.")
    
    st.info("""
    **What can this bot do?**
    - **Find Definitions:** Get precise meanings for complex words or phrases.
    - **Topic Information:** Verify and cross-check information on specific topics. Done through cosine similairty and machine learning!
    """)
elif action == "Find Definition of words or sentences":
    phrase = st.text_input("Enter word or phrase: ")

    if st.button("ENTER"):
        result = assistant.FindDefinition(phrase)
        st.success(result)

elif action == "Get Information on a Certain Topic":
    topic = st.text_input("Enter topic:")
    sentnum = st.number_input("How much sentences?", min_value=0, max_value=None, value=None, step=1)

    if st.button("ENTER"):
        generated_text = assistant.Crosscheck(topic, sentnum)
        st.success(generated_text)



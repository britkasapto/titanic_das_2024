import streamlit as st
import streamlit.web.cli as stcli
import pandas as pd
import streamlit.components.v1 as stc 
import joblib

from prediction import run_predict_app
from eda import run_eda_app

st.set_page_config(page_title="Titanic ML",
		   page_icon="🚢",
		   layout="wide")

html_temp = """
		<div style="background-color:#3872fb;padding:5px;border-radius:10px">
		<h3 style="color:white;text-align:center;font-family:arial;">Titanic Machine Learning </h3>
		<h3 style="color:white;text-align:center;font-family:arial;">PJJ DAS I 2024</h3>
		</div>
		"""

def main():
    stc.html(html_temp)
    menu = ["Home","EDA","Prediction"]
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "Home":
        st.subheader('Home Menu')
        st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/d/db/Titanic-Cobh-Harbour-1912.JPG/1600px-Titanic-Cobh-Harbour-1912.JPG')
        st.write("""
                ###### APP Content:
                EDA Section: Exploratory Data Analysis of Data | Prediction Section: ML Predictor App
			""")
    elif choice == "EDA":
        run_eda_app()

    elif choice == "Prediction" :
        run_predict_app()

if __name__=='__main__':
    main()
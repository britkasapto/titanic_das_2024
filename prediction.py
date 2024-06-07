import streamlit as st
import joblib
import pandas as pd
import numpy as np
from PIL import Image

def load_model():
    model_path = '/Users/britkasapto/Python Machine Learning/PJJ Data Analitik/MP 6 Project Deployment/day_2/bahan_titanic/model.joblib'
    model_gb = joblib.load(model_path)
    return model_gb

def run_predict_app():
    st.subheader("Prediction Section")
    model = load_model()
    with st.sidebar:
        st.title("Features")
        pclass = st.selectbox('PClass :', (1, 2, 3))
        fare = st.number_input('Fare :', min_value=0.0, max_value=263.0)
        sex_label = st.selectbox('Sex :', ('Male', 'Female'))
        sex = 0 if sex_label == 'Male' else 1
        
        Embarked = st.selectbox('Embarked :', ('Cherbourg', 'Queenstown', 'Southampton'))
        embarked_enc = 0 if Embarked == 'Cherbourg' else (1 if Embarked == 'Queenstown' else 2)
       
        initial = st.selectbox('Initial :', ('Mr', 'Mrs', 'Miss', 'Master', 'Other'))
        mr, mrs, miss, master, other = (1, 0, 0, 0, 0) if initial == 'Mr' else \
                                       (0, 1, 0, 0, 0) if initial == 'Mrs' else \
                                       (0, 0, 1, 0, 0) if initial == 'Miss' else \
                                       (0, 0, 0, 1, 0) if initial == 'Master' else \
                                       (0, 0, 0, 0, 1)

        age = st.slider('Age', 0, 100)
        cut_points = [0, 15, 30, 50, 100]
        age_label = ['child', 'young_adult', 'adult', 'elderly']
        age_group = pd.cut([age], bins=cut_points, labels=age_label, include_lowest=True)
        child, young_adult, adult, elderly = (1, 0, 0, 0) if age_group[0] == 'child' else \
                                             (0, 1, 0, 0) if age_group[0] == 'young_adult' else \
                                             (0, 0, 1, 0) if age_group[0] == 'adult' else \
                                             (0, 0, 0, 1)

    if st.button("Click here to predict"):
        st.info('Input :')
        st.write(f'Class : {pclass}')
        st.write(f'Fare : ${fare} ')
        st.write(f'Sex : {sex_label}')
        st.write(f'Embarked : {Embarked}')
        st.write(f'Initial : {initial}')
        st.write(f'Age : {age}')
        
        dfvalues = pd.DataFrame(list(zip([pclass], [fare], [sex], [embarked_enc], [mr], [mrs], [miss], [master], [other],
                                         [young_adult], [adult], [elderly], [child])), 
                                columns=['pclass', 'fare', 'sex', 'embarked', 'mr', 'mrs', 'miss', 'master', 'other',
                                         'young_adult', 'adult', 'elderly', 'child'])
       
        input_variables = np.array(dfvalues)

        st.info('Convertion :')
        st.dataframe(dfvalues)
        prediction = model.predict(input_variables)
        pred_prob = model.predict_proba(input_variables)
        st.info('Result :')
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write('Prediction :')
            hasil = prediction[0]
            if hasil == 1:
                st.success('Survived')
            else:
                st.warning('Not Survived')
        
        with col2:
            if prediction == 1:
                pred_probability_score = pred_prob[0][1] * 100
                st.write("Prediction Probability Score :")
                st.success(f"There is a : {pred_probability_score:.2f} % you will survived like Rose")
                img = "https://www.thelist.com/img/gallery/things-only-adults-notice-in-titanic/why-would-rose-display-paintings-for-a-week-long-trip-on-the-titanic-1575316189.jpg"
                st.image(img, width=460)
            else:
                pred_probability_score = pred_prob[0][0] * 100
                st.write("Prediction Probability Score")
                st.warning(f"There is a : {pred_probability_score:.2f} % you will end up like Jack")
                image_jack = "https://imgix.bustle.com/uploads/image/2017/7/4/e60d1805-b01f-4f02-a155-13f33814639a-jack-dawson-dirtbag.jpg?w=800&fit=crop&crop=faces&auto=format%2Ccompress&q=50&dpr=2"
                st.image(image_jack, width=460)

if __name__ == '__main__':
    run_predict_app()

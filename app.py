import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBRegressor

# Sayfa ayarları (en başta)
st.set_page_config(page_title="Abalone Yaş Tahmini", layout="wide")

@st.cache_resource
def load_model():
    try:
        model = joblib.load('xgb_model.pkl')
        if not hasattr(model, '_Booster'):
            model = XGBRegressor()
            model.load_model('xgb_model.json')
        return model
    except Exception as e:
        st.error(f"Model yüklenemedi: {str(e)}")
        st.stop()

model = load_model()

# Arayüz
st.title("🐚 Abalone Yaş Tahmin Aracı")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        sex = st.selectbox('Cinsiyet', ['Dişi', 'Bebek', 'Erkek'])
        length = st.slider('Uzunluk (mm)', 0.0, 1.0, 0.5)
        diameter = st.slider('Çap (mm)', 0.0, 1.0, 0.4)
        height = st.slider('Yükseklik (mm)', 0.0, 1.0, 0.1)
        
    with col2:
        whole_weight = st.slider('Toplam ağırlık (gram)', 0.0, 3.0, 0.8)
        shucked_weight = st.slider('Ayıklanmış ağırlık (gram)', 0.0, 2.0, 0.3)
        viscera_weight = st.slider('İç organ ağırlığı (gram)', 0.0, 1.0, 0.2)
        shell_weight = st.slider('Kabuk ağırlığı (gram)', 0.0, 1.0, 0.2)
    
    submitted = st.form_submit_button("TAHMİN YAP")

if submitted:
    input_data = pd.DataFrame({
        'Sex': {'Dişi':0, 'Bebek':1, 'Erkek':2}[sex],
        'Length': [length],
        'Diameter': [diameter],
        'Height': [height],
        'Whole weight': [whole_weight],
        'Whole weight.1': [shucked_weight],
        'Whole weight.2': [viscera_weight],
        'Shell weight': [shell_weight]
    })
    
    try:
        prediction = model.predict(input_data)
        age = prediction[0] + 1.5
        st.success(f"Tahmini Yaş: {age:.1f} yıl")
        st.info(f"Halka Sayısı: {prediction[0]:.1f}")
    except Exception as e:
        st.error(f"Hata oluştu: {str(e)}")
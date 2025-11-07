import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ---------- PAGE SETUP, STYLE, AND LANGUAGES (keep as in your previous template) ----------------

st.set_page_config(page_title="Khet Sahayak - Fertilizer Recommendation", layout="wide")

st.markdown("""
    <style>
    label, .stSelectbox label, .stTextInput label, .stNumberInput label, .stSlider label {
        color: #18683A !important;
        font-weight: bold !important;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown("""
    <style>
    html, body, .main {background: #fff!important;}
    body {font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #222;}
    .fertilizer-section {
        background: #fff; border-radius: 18px; padding: 50px 25px; box-shadow: 0 10px 30px rgba(44,68,116,0.08);
        margin:40px auto 24px auto; max-width:920px;
    }
    .fertilizer-header {text-align:center; margin-bottom:36px;}
    .fertilizer-header h1 {font-size:34px; color:#18683A; font-weight:700; letter-spacing:0.5px;}
    .main-title {font-size:26px; font-weight:bold; color:#249f56; text-align:center; 
                 margin:24px 0 10px 0; letter-spacing:0.3px;}
    .footer {background:#fff; color:#999; text-align:center; padding:18px;}
    .footer p {margin:0; font-size:15px;}
    .navbar-inner {height:60px; display:flex; align-items:center; justify-content:space-between; max-width:950px; margin:0 auto; padding:0 18px;}
    @media (max-width: 650px) {.fertilizer-section {padding:15px 2px;} .fertilizer-header h1 {font-size:22px;} .main-title {font-size:18px;}}
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="navbar-inner">
    <span class="main-title">Khet Sahayak · Fertilizer Recommendation</span>
</div>
""", unsafe_allow_html=True)

# ------------- LABELS IN THREE LANGUAGES -----------------
langs = {"English": "en", "हिन्दी": "hi", "ਪੰਜਾਬੀ": "pa"}
lang = st.selectbox("Language / भाषा / ਭਾਸ਼ਾ", list(langs.keys()), index=0, key="langbox")
cur_lang = langs[lang]
labels = {
    'en': {
        'title': "Fertilizer Recommendation",
        'desc': "Get personalized fertilizer advice by filling the form below.",
        'temperature': "Temperature (°C)",
        'humidity': "Humidity (%)",
        'soil_moisture': "Soil Moisture (%)",
        'soil_type': "Soil Type",
        'crop': "Crop Type",
        'nitrogen': "Nitrogen (N)",
        'phosphorus': "Phosphorus (P)",
        'potassium': "Potassium (K)",
        'submit': "Get Recommendation",
        'result': "Recommended Fertilizer:"
    },
    'hi': {
        'title': "उर्वरक सिफारिश",
        'desc': "निम्न फॉर्म भरकर व्यक्तिगत उर्वरक सलाह प्राप्त करें।",
        'temperature': "तापमान (°C)",
        'humidity': "आर्द्रता (%)",
        'soil_moisture': "मृदा नमी (%)",
        'soil_type': "मिट्टी का प्रकार",
        'crop': "फसल का प्रकार",
        'nitrogen': "नाइट्रोजन (N)",
        'phosphorus': "फास्फोरस (P)",
        'potassium': "पोटेशियम (K)",
        'submit': "सिफारिश प्राप्त करें",
        'result': "अनुशंसित उर्वरक:"
    },
    'pa': {
        'title': "ਖਾਦ ਦੀ ਸਿਫਾਰਸ਼",
        'desc': "ਹੇਠਾਂ ਦਿੱਤੇ ਫਾਰਮ ਨੂੰ ਭਰ ਕੇ ਨਿਜੀਖੇਡ ਖਾਦ ਸਲਾਹ ਲਵੋ。",
        'temperature': "ਤਾਪਮਾਨ (°C)",
        'humidity': "ਨਮੀਆਂ (%)",
        'soil_moisture': "ਮਿੱਟੀ ਦੀ ਨਮੀ (%)",
        'soil_type': "ਮਿੱਟੀ ਦੀ ਕਿਸਮ",
        'crop': "ਫਸਲ ਦੀ ਕਿਸਮ",
        'nitrogen': "ਨਾਈਟ੍ਰੋਜਨ (N)",
        'phosphorus': "ਫਾਸਫੋਰਸ (P)",
        'potassium': "ਪੋਟਾਸ਼ੀਅਮ (K)",
        'submit': "ਸਿਫਾਰਸ਼ ਲਵੋ",
        'result': "ਸਿਫਾਰਸ਼ ਖਾਦ:"
    }
}[cur_lang]

# ------------- LOAD DATA AND TRAIN RANDOMFOREST ------------

df = pd.read_csv("Fertilizer_recommendation.csv")
df.columns = df.columns.str.strip()
X = df.drop(["Fertilizer"], axis=1)
y = df["Fertilizer"]

le_soil = LabelEncoder()
le_crop = LabelEncoder()
X['Soil Type'] = le_soil.fit_transform(X['Soil Type'])
X['Crop Type'] = le_crop.fit_transform(X['Crop Type'])

rf = RandomForestClassifier(random_state=42, n_estimators=60)
rf.fit(X, y)

# ------------- UI CARD AND INPUT FORM HTML -----------------

st.markdown(f"""
<div class="fertilizer-section">
    <div class="fertilizer-header">
        <span style="font-size:36px;">🧪</span>
        <h1>{labels['title']}</h1>
    </div>
    <div style="margin-bottom:28px;"><p>{labels['desc']}</p></div>
""", unsafe_allow_html=True)

# ---------- STREAMLIT INPUT WIDGETS ------------
temp = st.number_input("Temperature (in Celsius)", min_value=0.0, max_value=60.0, value=25.0, step=0.1)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
nitrogen = st.number_input("Nitrogen Content in Soil (ppm)", min_value=0, max_value=200, value=100)
potassium = st.number_input("Potassium Content in Soil (ppm)", min_value=0, max_value=200, value=100)
phosphorus = st.number_input("Phosphorous Content in Soil (ppm)", min_value=0, max_value=200, value=100)
soil_type = st.selectbox("Select Soil Type", le_soil.classes_)
crop_type = st.selectbox("Select Crop Type", le_crop.classes_)
moisture = st.number_input("Soil Moisture (%)", min_value=0.0, max_value=100.0, value=30.0, step=0.1)

if st.button(labels['submit']):
    row = [[
        temp, humidity, moisture,
        le_soil.transform([soil_type])[0],
        le_crop.transform([crop_type])[0],
        nitrogen, phosphorus, potassium
    ]]
    fertil = rf.predict(np.array(row))[0]
    st.success(f"{labels['result']} {fertil}")

st.markdown("</div>", unsafe_allow_html=True)
st.markdown('<div class="footer"><p>© 2025 Khet Sahayak. All rights reserved.</p></div>', unsafe_allow_html=True)

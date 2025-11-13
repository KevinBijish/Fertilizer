import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ---------- PAGE SETUP, STYLE, AND LANGUAGES ----------------

st.set_page_config(page_title="Khet Sahayak - Fertilizer Recommendation", layout="wide")

st.markdown("""
    <style>
    label, .stSelectbox label, .stTextInput label, .stNumberInput label, .stSlider label {
        color: #00a859 !important;
        font-weight: bold !important;
        font-size: 18px !important;
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

# Translation dictionaries for soil and crop types
soil_translations = {
    'en': {
        'Alluvial': 'Alluvial',
        'Black': 'Black',
        'Chalky': 'Chalky',
        'Clay': 'Clay',
        'Clayey': 'Clayey',
        'Loamy': 'Loamy',
        'Peaty': 'Peaty',
        'Red': 'Red',
        'Sandy': 'Sandy'
    },
    'hi': {
        'Alluvial': 'जलोढ़',
        'Black': 'काली',
        'Chalky': 'चॉकी',
        'Clay': 'मृत्तिका',
        'Clayey': 'मृत्तिकायुक्त',
        'Loamy': 'दोमट',
        'Peaty': 'पीटयुक्त',
        'Red': 'लाल',
        'Sandy': 'बलुई'
    },
    'pa': {
        'Alluvial': 'ਜਲੋਢ',
        'Black': 'ਕਾਲੀ',
        'Chalky': 'ਚਾਕੀ',
        'Clay': 'ਚਿਕਨੀ',
        'Clayey': 'ਚਿਕਨੀ ਮਿੱਟੀ',
        'Loamy': 'ਦੁਮਟ',
        'Peaty': 'ਪੀਟ ਵਾਲੀ',
        'Red': 'ਲਾਲ',
        'Sandy': 'ਰੇਤਲੀ'
    }
}

crop_translations = {
    'en': {
        'Arhar': 'Arhar',
        'Bajra': 'Bajra',
        'Barley': 'Barley',
        'Cotton': 'Cotton',
        'Gram': 'Gram',
        'Groundnut': 'Groundnut',
        'Jowar': 'Jowar',
        'Maize': 'Maize',
        'Millets': 'Millets',
        'Moong': 'Moong',
        'Paddy': 'Paddy',
        'Ragi': 'Ragi',
        'Rice': 'Rice',
        'Sugarcane': 'Sugarcane',
        'Tobacco': 'Tobacco',
        'Urad': 'Urad',
        'Wheat': 'Wheat'
    },
    'hi': {
        'Arhar': 'अरहर',
        'Bajra': 'बाजरा',
        'Barley': 'जौ',
        'Cotton': 'कपास',
        'Gram': 'चना',
        'Groundnut': 'मूंगफली',
        'Jowar': 'ज्वार',
        'Maize': 'मक्का',
        'Millets': 'बाजरा',
        'Moong': 'मूंग',
        'Paddy': 'धान',
        'Ragi': 'रागी',
        'Rice': 'चावल',
        'Sugarcane': 'गन्ना',
        'Tobacco': 'तंबाकू',
        'Urad': 'उड़द',
        'Wheat': 'गेहूं'
    },
    'pa': {
        'Arhar': 'ਅਰਹਰ',
        'Bajra': 'ਬਾਜਰਾ',
        'Barley': 'ਜੌਂ',
        'Cotton': 'ਕਪਾਹ',
        'Gram': 'ਚਣਾ',
        'Groundnut': 'ਮੂੰਗਫਲੀ',
        'Jowar': 'ਜੁਆਰ',
        'Maize': 'ਮੱਕੀ',
        'Millets': 'ਬਾਜਰਾ',
        'Moong': 'ਮੂੰਗ',
        'Paddy': 'ਝੋਨਾ',
        'Ragi': 'ਰਾਗੀ',
        'Rice': 'ਚੌਲ',
        'Sugarcane': 'ਗੰਨਾ',
        'Tobacco': 'ਤੰਬਾਕੂ',
        'Urad': 'ਉੜਦ',
        'Wheat': 'ਕਣਕ'
    }
}

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
        'desc': "ਹੇਠਾਂ ਦਿੱਤੇ ਫਾਰਮ ਨੂੰ ਭਰ ਕੇ ਨਿਜੀਖੇਡ ਖਾਦ ਸਲਾਹ ਲਵੋ।",
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
df['Soil Type'] = df['Soil Type'].str.strip()
df['Crop Type'] = df['Crop Type'].str.strip()
X = df.drop(["Fertilizer"], axis=1)
y = df["Fertilizer"]

le_soil = LabelEncoder()
le_crop = LabelEncoder()
X['Soil Type'] = le_soil.fit_transform(X['Soil Type'])
X['Crop Type'] = le_crop.fit_transform(X['Crop Type'])

rf = RandomForestClassifier(random_state=42, n_estimators=60)
rf.fit(X, y)

# Helper functions to get only matching translation options
def get_valid_translated_options(le_classes, translations):
    valid = []
    keys_original = []
    for s in le_classes:
        if s in translations[cur_lang]:
            valid.append(translations[cur_lang][s])
            keys_original.append(s)
    return valid, keys_original

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
temp = st.number_input(labels['temperature'], min_value=0.0, max_value=60.0, value=25.0, step=0.1)
humidity = st.number_input(labels['humidity'], min_value=0.0, max_value=100.0, value=50.0, step=0.1)
moisture = st.number_input(labels['soil_moisture'], min_value=0.0, max_value=100.0, value=30.0, step=0.1)

# Get only matched soil types for translation display
soil_options_translated, soil_keys = get_valid_translated_options(le_soil.classes_, soil_translations)
soil_type_display = st.selectbox(labels['soil_type'], soil_options_translated)
soil_type = soil_keys[soil_options_translated.index(soil_type_display)]

crop_options_translated, crop_keys = get_valid_translated_options(le_crop.classes_, crop_translations)
crop_type_display = st.selectbox(labels['crop'], crop_options_translated)
crop_type = crop_keys[crop_options_translated.index(crop_type_display)]

nitrogen = st.number_input(labels['nitrogen'], min_value=0, max_value=200, value=100)
phosphorus = st.number_input(labels['phosphorus'], min_value=0, max_value=200, value=100)
potassium = st.number_input(labels['potassium'], min_value=0, max_value=200, value=100)

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


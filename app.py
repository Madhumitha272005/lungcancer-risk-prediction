import streamlit as st
from PIL import Image
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Lung Cancer Risk Prediction",
    layout="wide"
)

# =====================================
# LOAD MODEL
# =====================================
try:
    model = pickle.load(open("lung_model.pkl", "rb"))
except:
    model = None

# =====================================
# LOAD DATASET
# =====================================
data = pd.read_csv("lung_cancer.csv")
data.columns = data.columns.str.strip().str.replace(" ", "_")

# =====================================
# CUSTOM CSS (UNCHANGED)
# =====================================
st.markdown("""
<style>
.stApp { background-color: white; }

.main-title {
    font-family: "Times New Roman", serif;
    font-size: 50px;
    font-weight: bold;
    font-style: italic;
    color: black;
}

.description {
    font-size: 20px;
    font-family:"Century";
    color: #333333;
    margin-top: 15px;
    line-height: 1.7;
}

.section-header {
    font-size: 26px;
    font-family:"Sitka Display Semibold";
    font-weight: bold;
    margin-top: 30px;
    margin-bottom: 20px;
    color: black;
}

.stButton>button {
    background-color: #c40000;
    color: white;
    font-size: 18px;
    font-family:"Sitka Display Semibold";
    font-weight: bold;
    padding: 12px 25px;
    border-radius: 8px;
    border: none;
}

.stButton>button:hover {
    background-color: black;
    color: white;
}

label {
    color: black !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# =====================================
# HEADER
# =====================================
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<div class='main-title'>Lung Cancer Risk Prediction</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='description'>
   Welcome to the Lung Cancer Risk Prediction System, an AI-powered tool designed to assess the potential risk of lung cancer using clinical and lifestyle factors such as age, smoking habits, air pollution exposure, and respiratory symptoms. 
                By analyzing these inputs through advanced Machine Learning algorithms, the system predicts whether a person falls into a low, medium, or high-risk category. 
                This platform supports early awareness and preventive healthcare. It is not a medical diagnosis tool but helps users understand their risk level and seek timely medical consultation if necessary.
    </div>
    """, unsafe_allow_html=True)

with col2:
    try:
        image = Image.open("lc3.png")
        st.image(image, width="stretch")
    except:
        pass

st.markdown("<br><br>", unsafe_allow_html=True)

# =====================================
# INPUT SECTION
# =====================================
st.markdown("<div class='section-header'>Enter Patient Dataset Information</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 1, 120, 30)
    gender = st.selectbox("Gender (1=Male, 0=Female)", [1, 0])
    smoking = st.selectbox("Smoking", [1, 0])
    yellow_fingers = st.selectbox("Yellow Fingers", [1, 0])
    anxiety = st.selectbox("Anxiety", [1, 0])

with col2:
    peer_pressure = st.selectbox("Peer Pressure", [1, 0])
    chronic_disease = st.selectbox("Chronic Disease", [1, 0])
    fatigue = st.selectbox("Fatigue", [1, 0])
    allergy = st.selectbox("Allergy", [1, 0])
    wheezing = st.selectbox("Wheezing", [1, 0])

with col3:
    alcohol = st.selectbox("Alcohol Consuming", [1, 0])
    coughing = st.selectbox("Coughing", [1, 0])
    short_breath = st.selectbox("Shortness of Breath", [1, 0])
    swallowing = st.selectbox("Swallowing Difficulty", [1, 0])
    chest_pain = st.selectbox("Chest Pain", [1, 0])

# =====================================
# PREDICTION
# =====================================
if st.button("Predict Risk") and model is not None:

    input_data = pd.DataFrame([[
        gender, age, smoking, yellow_fingers, anxiety,
        peer_pressure, chronic_disease, fatigue, allergy,
        wheezing, alcohol, coughing, short_breath,
        swallowing, chest_pain
    ]], columns=[
        "GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY",
        "PEER_PRESSURE", "CHRONIC_DISEASE", "FATIGUE", "ALLERGY",
        "WHEEZING", "ALCOHOL_CONSUMING", "COUGHING",
        "SHORTNESS_OF_BREATH", "SWALLOWING_DIFFICULTY",
        "CHEST_PAIN"
    ])

    probability = model.predict_proba(input_data)[0][1]
    risk_percentage = round(probability * 100, 2)

    st.markdown(f"""
    <div style='text-align:center; font-family:Times New Roman;
                padding:20px; border-radius:15px;
                background-color:#111827; color:white;'>
        <h2>Predicted Risk Percentage</h2>
        <h1 style='font-size:50px;'>{risk_percentage}%</h1>
    </div>
    """, unsafe_allow_html=True)

    if risk_percentage >= 70:
        st.error("HIGH RISK")
    elif risk_percentage >= 40:
        st.warning("MEDIUM RISK")
    else:
        st.success("LOW RISK")

# =====================================
# GRAPH SECTION
# =====================================
st.markdown("---")
st.markdown(
    "<h2 style='text-align:center; font-family:Times New Roman;'>Positive Cases Analysis</h2>",
    unsafe_allow_html=True
)

positive_cases = data[data['LUNG_CANCER'] == 'YES'].copy()
positive_cases['GENDER'] = positive_cases['GENDER'].map({'M': 'Male', 'F': 'Female'})

# -------- PIE CHART (CENTERED & SMALL) --------
st.markdown("### Gender Distribution")

gender_counts = positive_cases['GENDER'].value_counts()

colA, colB, colC = st.columns([1,2,1])
with colB:
    fig4, ax4 = plt.subplots(figsize=(3,3))
    ax4.pie(gender_counts,
            labels=gender_counts.index,
            autopct='%1.1f%%',
            startangle=90)
    ax4.axis('equal')
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

# -------- MALE GRAPH --------
st.markdown("### Male Age Distribution")

male_cases = positive_cases[positive_cases['GENDER'] == 'Male']

colA, colB, colC = st.columns([1,2,1])
with colB:
    fig1, ax1 = plt.subplots(figsize=(5,3))
    sns.histplot(male_cases["AGE"], bins=15, kde=True, ax=ax1, color="gold")
    ax1.set_title("Male Positive Cases Age Distribution", fontname="Times New Roman")
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

# -------- FEMALE GRAPH --------
st.markdown("### Female Age Distribution")

female_cases = positive_cases[positive_cases['GENDER'] == 'Female']

colA, colB, colC = st.columns([1,2,1])
with colB:
    fig2, ax2 = plt.subplots(figsize=(5,3))
    sns.histplot(female_cases["AGE"], bins=15, kde=True, ax=ax2, color="black")
    ax2.set_title("Female Positive Cases Age Distribution", fontname="Times New Roman")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

# -------- COMBINED GRAPH --------
st.markdown("### Combined Age Distribution")

colA, colB, colC = st.columns([1,2,1])
with colB:
    fig3, ax3 = plt.subplots(figsize=(5,3))
    sns.histplot(data=positive_cases, x="AGE", hue="GENDER",
                 bins=15, kde=True,
                 palette={"Male": "gold", "Female": "black"},
                 ax=ax3)
    ax3.set_title("Combined Positive Cases Age Distribution", fontname="Times New Roman")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)






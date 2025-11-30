import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# CONFIG
# -----------------------------

st.set_page_config(
    page_title="Prédiction du risque de maladie cardiaque",
    layout="centered"
)

# -----------------------------
# MEDICAL THEME (CSS)
# -----------------------------
st.markdown("""
    <style>

        body {
            background-color: #eef3f8;
        }

        .main {
            background-color: #ffffff;
            padding: 2rem 3rem;
            border-radius: 12px;
            box-shadow: 0 2px 15px rgba(0,0,0,0.1);
        }

        .header {
            background-color: #d3e3fc;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 20px;
            border-left: 6px solid #4a90e2;
        }

        .header h1 {
            color: #2c4a63;
            font-size: 30px;
            margin: 0;
            font-weight: 700;
        }

        .header p {
            color: #436280;
            font-size: 15px;
            margin-top: 5px;
        }

        label {
            color: #2c4a63 !important;
            font-weight: 600 !important;
        }

        .stNumberInput input, .stSelectbox div div {
            border-radius: 8px !important;
            border: 1px solid #bcd0e7 !important;
            background-color: #f7fbff !important;
        }

        .stButton button {
            background-color: #4a90e2 !important;
            color: white !important;
            padding: 0.6rem 1.2rem !important;
            border-radius: 8px !important;
            border: none !important;
            font-size: 15px !important;
        }

        .stButton button:hover {
            background-color: #3d7bc4 !important;
        }

    </style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<div class="header">
    <h1>🩺 Prédiction du risque de maladie cardiaque</h1>
    <p>Analyse automatique basée sur un modèle Machine Learning validé.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# MODEL LOADING
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("Model.pkl")

model = load_model()

# -----------------------------
# BOX START
# -----------------------------
st.markdown('<div class="main">', unsafe_allow_html=True)

st.markdown("### 🔍 Paramètres cliniques du patient")

# -----------------------------
# FORM INPUTS
# -----------------------------
with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Âge", 10, 100, 50)
        sbp = st.number_input("Pression systolique (mmHg)", 80.0, 250.0, 140.0)
        ldl = st.number_input("Taux de LDL (mmol/L)", 0.0, 10.0, 4.0)

    with col2:
        adiposity = st.number_input("Adiposity", 0.0, 60.0, 25.0)
        obesity = st.number_input("Obesity", 0.0, 60.0, 30.0)
        famhist = st.selectbox("Antécédents familiaux", ["Present", "Absent"])

    submitted = st.form_submit_button("Analyser le risque")

# -----------------------------
# PREDICTION
# -----------------------------
if submitted:

    data = pd.DataFrame([{
        "sbp": sbp,
        "ldl": ldl,
        "adiposity": adiposity,
        "obesity": obesity,
        "age": age,
        "famhist": famhist
    }])

    st.markdown("### 📄 Données saisies")
    st.dataframe(data, use_container_width=True)

    proba = model.predict_proba(data)[0, 1]
    pred = model.predict(data)[0]

    st.markdown("### 🧪 Résultat de l'analyse")

    if pred == 1:
        st.error(
            f"⚠️ Risque ÉLEVÉ détecté.\n\nProbabilité estimée : **{proba:.2f}**"
        )
    else:
        st.success(
            f"🟢 Risque FAIBLE détecté.\n\nProbabilité estimée : **{proba:.2f}**"
        )

    st.info(
        "Cette application fournit une estimation algorithmique et ne constitue pas un avis médical."
    )

st.markdown('</div>', unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import requests
from azure.storage.blob import BlobServiceClient
import io
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MIMIC-III Medical Dashboard",
    page_icon="🏥",
    layout="wide"
)

# ── CUSTOM CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main { background-color: #0f1117; color: #ffffff; }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e, #252d3d);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px;
    }
    .metric-number { font-size: 2.5rem; font-weight: 700; color: #4fc3f7; }
    .metric-label  { font-size: 0.85rem; color: #90a4ae; margin-top: 4px; }
    
    .chat-message-user {
        background: linear-gradient(135deg, #1565c0, #1976d2);
        border-radius: 12px 12px 2px 12px;
        padding: 12px 16px;
        margin: 8px 0;
        margin-left: 20%;
        color: white;
    }
    .chat-message-ai {
        background: linear-gradient(135deg, #1a2332, #1e2a3a);
        border: 1px solid #2d3748;
        border-radius: 12px 12px 12px 2px;
        padding: 12px 16px;
        margin: 8px 0;
        margin-right: 20%;
        color: #e0e0e0;
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4fc3f7, #29b6f6, #81d4fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .header-sub {
        color: #90a4ae;
        font-size: 0.95rem;
        margin-top: 4px;
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #4fc3f7;
        border-bottom: 2px solid #1565c0;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1565c0, #1976d2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        width: 100%;
    }
    .stTextInput > div > input {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        color: white;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── CONFIG ─────────────────────────────────────────────────────────────────
STORAGE_ACCOUNT  = "akshit"
st.secrets["STORAGE_KEY"]
CONTAINER        = "mimic-raw"
st.secrets["GROQ_API_KEY"]
GROQ_URL         = "https://api.groq.com/openai/v1/chat/completions"

CONNECTION_STRING = (
    f"DefaultEndpointsProtocol=https;"
    f"AccountName={STORAGE_ACCOUNT};"
    f"AccountKey={STORAGE_KEY};"
    f"EndpointSuffix=core.windows.net"
)

# ── DATA LOADER ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    def read(filename):
        blob = client.get_blob_client(container=CONTAINER, blob=filename)
        data = blob.download_blob().readall()
        df   = pd.read_csv(io.BytesIO(data))
        df.columns = df.columns.str.upper()
        return df

    patients      = read("PATIENTS.csv")
    admissions    = read("ADMISSIONS.csv")
    icustays      = read("ICUSTAYS.csv")
    diagnoses     = read("DIAGNOSES_ICD.csv")
    prescriptions = read("PRESCRIPTIONS.csv")
    labevents     = read("LABEVENTS.csv")
    return patients, admissions, icustays, diagnoses, prescriptions, labevents

# ── GROQ AI ────────────────────────────────────────────────────────────────
def ask_ai(question, summary):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": summary},
            {"role": "user",   "content": question}
        ]
    }
    response = requests.post(GROQ_URL, headers=headers, json=payload)
    result   = response.json()
    if "choices" in result:
        return result["choices"][0]["message"]["content"]
    return f"Error: {result}"

# ── HEADER ─────────────────────────────────────────────────────────────────
st.markdown('<p class="header-title">🏥 MIMIC-III Medical Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="header-sub">AI-powered clinical data analysis · Beth Israel Deaconess Medical Center · 2001–2012</p>', unsafe_allow_html=True)
st.markdown("---")

# ── LOAD DATA ──────────────────────────────────────────────────────────────
with st.spinner("Loading MIMIC-III data from Azure Blob Storage..."):
    df_patients, df_admissions, df_icustays, df_diagnoses, df_prescriptions, df_labevents = load_data()

# ── METRICS ROW ────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">📊 Dataset Overview</p>', unsafe_allow_html=True)
c1, c2, c3, c4, c5, c6 = st.columns(6)

metrics = [
    (c1, len(df_patients),      "Patients"),
    (c2, len(df_admissions),    "Admissions"),
    (c3, len(df_icustays),      "ICU Stays"),
    (c4, len(df_diagnoses),     "Diagnoses"),
    (c5, len(df_prescriptions), "Prescriptions"),
    (c6, len(df_labevents),     "Lab Events"),
]
for col, num, label in metrics:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{num:,}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── CHARTS ────────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">📈 Exploratory Data Analysis</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Patient Gender Distribution**")
    gender = df_patients["GENDER"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="#0f1117")
    ax.set_facecolor("#0f1117")
    ax.pie(gender, labels=gender.index, autopct="%1.1f%%",
           colors=["#4fc3f7", "#f06292"], startangle=90,
           textprops={"color": "white"})
    st.pyplot(fig)
    plt.close()

with col2:
    st.markdown("**Admission Types**")
    adm_types = df_admissions["ADMISSION_TYPE"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="#0f1117")
    ax.set_facecolor("#1a1f2e")
    bars = ax.bar(adm_types.index, adm_types.values,
                  color=["#4fc3f7", "#f06292", "#81c784"])
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#2d3748")
    ax.set_ylabel("Count", color="white")
    st.pyplot(fig)
    plt.close()

col3, col4 = st.columns(2)

with col3:
    st.markdown("**Top 10 Diagnoses (ICD9)**")
    top_dx = df_diagnoses["ICD9_CODE"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0f1117")
    ax.set_facecolor("#1a1f2e")
    ax.barh(top_dx.index.astype(str), top_dx.values, color="#4fc3f7")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#2d3748")
    ax.set_xlabel("Count", color="white")
    ax.invert_yaxis()
    st.pyplot(fig)
    plt.close()

with col4:
    st.markdown("**Top 10 Medications**")
    top_meds = df_prescriptions["DRUG"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0f1117")
    ax.set_facecolor("#1a1f2e")
    ax.barh(top_meds.index, top_meds.values, color="#f06292")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#2d3748")
    ax.set_xlabel("Count", color="white")
    ax.invert_yaxis()
    st.pyplot(fig)
    plt.close()

# ICU Stay
st.markdown("**ICU Stay Duration Distribution**")
df_icu = df_icustays.copy()
df_icu["INTIME"]    = pd.to_datetime(df_icu["INTIME"])
df_icu["OUTTIME"]   = pd.to_datetime(df_icu["OUTTIME"])
df_icu["LOS_HOURS"] = (df_icu["OUTTIME"] - df_icu["INTIME"]).dt.total_seconds() / 3600
df_icu = df_icu[df_icu["LOS_HOURS"] < 1000]
fig, ax = plt.subplots(figsize=(10, 3), facecolor="#0f1117")
ax.set_facecolor("#1a1f2e")
ax.hist(df_icu["LOS_HOURS"].dropna(), bins=40, color="#81c784", edgecolor="#0f1117")
ax.tick_params(colors="white")
ax.spines[:].set_color("#2d3748")
ax.set_xlabel("Hours", color="white")
ax.set_ylabel("Count", color="white")
st.pyplot(fig)
plt.close()

st.markdown("---")

# ── AI CHAT ────────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">🤖 Ask AI About the Data</p>', unsafe_allow_html=True)

ai_summary = f"""
You are a medical data assistant for MIMIC-III hospital data.
PATIENTS: {len(df_patients)} total | Gender: {df_patients['GENDER'].value_counts().to_dict()}
ADMISSIONS: {len(df_admissions)} total | Types: {df_admissions['ADMISSION_TYPE'].value_counts().to_dict()}
ICU STAYS: {len(df_icustays)} total
DIAGNOSES: {len(df_diagnoses)} total | Top ICD9: {df_diagnoses['ICD9_CODE'].value_counts().head(5).to_dict()}
PRESCRIPTIONS: {len(df_prescriptions)} total | Top Drugs: {df_prescriptions['DRUG'].value_counts().head(5).to_dict()}
Answer clearly and concisely.
"""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-message-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message-ai">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

question = st.text_input("Ask a question about the MIMIC-III data...",
                         placeholder="e.g. What is the most common diagnosis?")

quick_col1, quick_col2, quick_col3 = st.columns(3)
with quick_col1:
    if st.button("👥 Gender breakdown"):
        question = "How many male vs female patients are there?"
with quick_col2:
    if st.button("💊 Top medications"):
        question = "What is the most prescribed medication?"
with quick_col3:
    if st.button("🏥 Common diagnoses"):
        question = "What are the most common diagnoses?"

if question:
    with st.spinner("AI is thinking..."):
        answer = ask_ai(question, ai_summary)
    st.session_state.chat_history.append({"role": "user",      "content": question})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.rerun()

st.markdown("---")
st.markdown('<p style="text-align:center; color:#546e7a; font-size:0.8rem;">MIMIC-III Clinical Database Demo · Built with Streamlit + Azure + Groq AI</p>', unsafe_allow_html=True)

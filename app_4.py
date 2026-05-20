import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
from azure.storage.blob import BlobServiceClient
import io
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="ICU AI Medical Assistant", page_icon="🏥", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #0d1b2a, #1b2838);
        border: 1px solid #1e3a5f;
        border-radius: 14px;
        padding: 18px 12px;
        text-align: center;
    }
    .metric-num   { font-size: 2rem; font-weight: 700; color: #38bdf8; }
    .metric-lbl   { font-size: 0.78rem; color: #64748b; margin-top: 3px; letter-spacing:.5px; text-transform:uppercase; }
    .section-hdr  { font-size: 1.1rem; font-weight: 600; color: #38bdf8; border-left: 3px solid #0ea5e9; padding-left: 10px; margin: 18px 0 12px; }
    .patient-card {
        background: linear-gradient(135deg, #0d1b2a, #1b2838);
        border: 1px solid #1e3a5f;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 12px;
    }
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px;
    }
    .badge-blue  { background:#0c4a6e; color:#7dd3fc; }
    .badge-green { background:#064e3b; color:#6ee7b7; }
    .badge-red   { background:#7f1d1d; color:#fca5a5; }
    .chat-user {
        background: linear-gradient(135deg,#1d4ed8,#2563eb);
        border-radius: 16px 16px 4px 16px;
        padding: 12px 16px; margin: 6px 0; margin-left: 15%;
        color: white; font-size: 0.92rem;
    }
    .chat-ai {
        background: #0d1b2a; border: 1px solid #1e3a5f;
        border-radius: 16px 16px 16px 4px;
        padding: 12px 16px; margin: 6px 0; margin-right: 15%;
        color: #cbd5e1; font-size: 0.92rem;
    }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 500; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stButton > button {
        background: linear-gradient(135deg,#0369a1,#0284c7);
        color: white; border: none; border-radius: 8px;
        padding: 8px 20px; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── SECURE CONFIG via st.secrets ──────────────────────────────────────────
STORAGE_ACCOUNT = "akshit"
CONTAINER       = "mimic-raw"
try:
    STORAGE_KEY  = st.secrets["STORAGE_KEY"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    STORAGE_KEY  = ""
    GROQ_API_KEY = ""

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
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
    return (read("PATIENTS.csv"), read("ADMISSIONS.csv"), read("ICUSTAYS.csv"),
            read("DIAGNOSES_ICD.csv"), read("PRESCRIPTIONS.csv"), read("LABEVENTS.csv"))

# ── AI ─────────────────────────────────────────────────────────────────────
def ask_ai(messages):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "llama-3.3-70b-versatile", "messages": messages, "max_tokens": 800}
    r = requests.post(GROQ_URL, headers=headers, json=payload)
    res = r.json()
    if "choices" in res:
        return res["choices"][0]["message"]["content"]
    return f"Error: {res}"

def dark_fig(w=6, h=4):
    fig, ax = plt.subplots(figsize=(w, h), facecolor="#0d1b2a")
    ax.set_facecolor("#1b2838")
    ax.tick_params(colors="#94a3b8")
    ax.spines[:].set_color("#1e3a5f")
    return fig, ax

# ── HEADER ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:16px 0 8px">
  <span style="font-size:2rem;font-weight:700;background:linear-gradient(90deg,#38bdf8,#818cf8);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent">
  🏥 AI-Powered ICU Analysis & Medical Assistant</span><br>
  <span style="color:#64748b;font-size:0.9rem">MIMIC-III Clinical Database · Beth Israel Deaconess Medical Center · 2001–2012</span>
</div>
""", unsafe_allow_html=True)

# ── LOAD ───────────────────────────────────────────────────────────────────
with st.spinner("Loading MIMIC-III data from Azure Blob Storage..."):
    df_p, df_a, df_icu, df_dx, df_rx, df_lab = load_data()

df_icu2 = df_icu.copy()
df_icu2["INTIME"]    = pd.to_datetime(df_icu2["INTIME"])
df_icu2["OUTTIME"]   = pd.to_datetime(df_icu2["OUTTIME"])
df_icu2["LOS_HOURS"] = (df_icu2["OUTTIME"] - df_icu2["INTIME"]).dt.total_seconds() / 3600
df_icu2 = df_icu2[df_icu2["LOS_HOURS"] < 1000]

# ── METRICS ────────────────────────────────────────────────────────────────
cols = st.columns(6)
for col, num, lbl in zip(cols, [len(df_p), len(df_a), len(df_icu), len(df_dx), len(df_rx), len(df_lab)],
                               ["Patients","Admissions","ICU Stays","Diagnoses","Prescriptions","Lab Events"]):
    with col:
        st.markdown(f'<div class="metric-card"><div class="metric-num">{num:,}</div><div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 ICU Dashboard", "🔍 Patient Search", "🔬 Filter & Explore", "🤖 AI Medical Assistant"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — ICU DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-hdr">ICU Overview</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        avg_los = df_icu2["LOS_HOURS"].mean()
        st.markdown(f'<div class="metric-card"><div class="metric-num">{avg_los:.1f}h</div><div class="metric-lbl">Avg ICU Stay</div></div>', unsafe_allow_html=True)
    with c2:
        max_los = df_icu2["LOS_HOURS"].max()
        st.markdown(f'<div class="metric-card"><div class="metric-num">{max_los:.1f}h</div><div class="metric-lbl">Longest ICU Stay</div></div>', unsafe_allow_html=True)
    with c3:
        emerg = (df_a["ADMISSION_TYPE"] == "EMERGENCY").sum()
        st.markdown(f'<div class="metric-card"><div class="metric-num">{emerg}</div><div class="metric-lbl">Emergency Admissions</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-hdr">Gender Distribution</div>', unsafe_allow_html=True)
        gender = df_p["GENDER"].value_counts()
        fig, ax = dark_fig(5, 3.5)
        ax.pie(gender, labels=gender.index, autopct="%1.1f%%",
               colors=["#38bdf8","#f472b6"], startangle=90,
               textprops={"color":"white"})
        st.pyplot(fig); plt.close()

    with col2:
        st.markdown('<div class="section-hdr">Admission Types</div>', unsafe_allow_html=True)
        adm = df_a["ADMISSION_TYPE"].value_counts()
        fig, ax = dark_fig(5, 3.5)
        ax.bar(adm.index, adm.values, color=["#38bdf8","#f472b6","#34d399"])
        ax.set_ylabel("Count", color="#94a3b8")
        plt.xticks(rotation=15)
        st.pyplot(fig); plt.close()

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-hdr">Top 10 Diagnoses (ICD9)</div>', unsafe_allow_html=True)
        top_dx = df_dx["ICD9_CODE"].value_counts().head(10)
        fig, ax = dark_fig(5, 4)
        ax.barh(top_dx.index.astype(str), top_dx.values, color="#38bdf8")
        ax.set_xlabel("Count", color="#94a3b8")
        ax.invert_yaxis()
        st.pyplot(fig); plt.close()

    with col4:
        st.markdown('<div class="section-hdr">Top 10 Medications</div>', unsafe_allow_html=True)
        top_rx = df_rx["DRUG"].value_counts().head(10)
        fig, ax = dark_fig(5, 4)
        ax.barh(top_rx.index, top_rx.values, color="#f472b6")
        ax.set_xlabel("Count", color="#94a3b8")
        ax.invert_yaxis()
        st.pyplot(fig); plt.close()

    st.markdown('<div class="section-hdr">ICU Stay Duration Distribution</div>', unsafe_allow_html=True)
    fig, ax = dark_fig(10, 3)
    ax.hist(df_icu2["LOS_HOURS"].dropna(), bins=40, color="#34d399", edgecolor="#0d1b2a")
    ax.set_xlabel("Hours", color="#94a3b8")
    ax.set_ylabel("Count", color="#94a3b8")
    st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — PATIENT SEARCH
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-hdr">Search Patient by ID</div>', unsafe_allow_html=True)

    all_ids = sorted(df_p["SUBJECT_ID"].unique().tolist())
    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        patient_id = st.selectbox("Select or type a Patient ID", all_ids)
    with col_s2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_btn = st.button("🔍 Search Patient")

    if search_btn or patient_id:
        pid = int(patient_id)
        patient = df_p[df_p["SUBJECT_ID"] == pid]

        if patient.empty:
            st.error("Patient not found.")
        else:
            row = patient.iloc[0]
            st.markdown(f"""
            <div class="patient-card">
                <div style="font-size:1.3rem;font-weight:700;color:#38bdf8;margin-bottom:10px">
                    👤 Patient #{pid}
                </div>
                <span class="badge badge-blue">Gender: {row.get('GENDER','N/A')}</span>
                <span class="badge badge-green">DOB: {str(row.get('DOB','N/A'))[:10]}</span>
                <span class="badge badge-red">Expires: {'Yes' if pd.notna(row.get('DOD')) else 'No'}</span>
            </div>
            """, unsafe_allow_html=True)

            r1, r2, r3 = st.columns(3)

            with r1:
                st.markdown('<div class="section-hdr">🏥 Admissions</div>', unsafe_allow_html=True)
                adm_p = df_a[df_a["SUBJECT_ID"] == pid][["ADMITTIME","ADMISSION_TYPE","DIAGNOSIS"]].head(5)
                if adm_p.empty:
                    st.info("No admissions found.")
                else:
                    st.dataframe(adm_p, use_container_width=True, hide_index=True)

            with r2:
                st.markdown('<div class="section-hdr">🧬 Diagnoses</div>', unsafe_allow_html=True)
                dx_p = df_dx[df_dx["SUBJECT_ID"] == pid][["ICD9_CODE","SEQ_NUM"]].head(10)
                if dx_p.empty:
                    st.info("No diagnoses found.")
                else:
                    st.dataframe(dx_p, use_container_width=True, hide_index=True)

            with r3:
                st.markdown('<div class="section-hdr">💊 Prescriptions</div>', unsafe_allow_html=True)
                rx_p = df_rx[df_rx["SUBJECT_ID"] == pid][["DRUG","DOSE_VAL_RX","ROUTE"]].head(10)
                if rx_p.empty:
                    st.info("No prescriptions found.")
                else:
                    st.dataframe(rx_p, use_container_width=True, hide_index=True)

            st.markdown('<div class="section-hdr">🛏️ ICU Stays</div>', unsafe_allow_html=True)
            icu_p = df_icu2[df_icu2["SUBJECT_ID"] == pid][["ICUSTAY_ID","INTIME","OUTTIME","LOS_HOURS","FIRST_CAREUNIT"]].head(5)
            if icu_p.empty:
                st.info("No ICU stays found.")
            else:
                st.dataframe(icu_p, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — FILTER & EXPLORE
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-hdr">Filter Dashboard</div>', unsafe_allow_html=True)

    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        adm_filter = st.selectbox("Admission Type", ["All"] + df_a["ADMISSION_TYPE"].unique().tolist())
    with fc2:
        gender_filter = st.selectbox("Gender", ["All", "M", "F"])
    with fc3:
        icu_min, icu_max = st.slider("ICU Stay (hours)", 0, 500, (0, 500))
    with fc4:
        top_n = st.slider("Top N Diagnoses", 5, 20, 10)

    df_a_f = df_a.copy()
    if adm_filter != "All":
        df_a_f = df_a_f[df_a_f["ADMISSION_TYPE"] == adm_filter]

    df_p_f = df_p.copy()
    if gender_filter != "All":
        df_p_f = df_p_f[df_p_f["GENDER"] == gender_filter]

    df_icu_f = df_icu2[(df_icu2["LOS_HOURS"] >= icu_min) & (df_icu2["LOS_HOURS"] <= icu_max)]

    filtered_pids = set(df_a_f["SUBJECT_ID"]) & set(df_p_f["SUBJECT_ID"]) & set(df_icu_f["SUBJECT_ID"])
    df_dx_f = df_dx[df_dx["SUBJECT_ID"].isin(filtered_pids)]
    df_rx_f = df_rx[df_rx["SUBJECT_ID"].isin(filtered_pids)]

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-num">{len(filtered_pids)}</div><div class="metric-lbl">Filtered Patients</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-num">{len(df_a_f)}</div><div class="metric-lbl">Filtered Admissions</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-num">{len(df_icu_f)}</div><div class="metric-lbl">Filtered ICU Stays</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    fc_col1, fc_col2 = st.columns(2)

    with fc_col1:
        st.markdown('<div class="section-hdr">Top Diagnoses (Filtered)</div>', unsafe_allow_html=True)
        if not df_dx_f.empty:
            top_dx_f = df_dx_f["ICD9_CODE"].value_counts().head(top_n)
            fig, ax = dark_fig(5, 4)
            ax.barh(top_dx_f.index.astype(str), top_dx_f.values, color="#38bdf8")
            ax.set_xlabel("Count", color="#94a3b8")
            ax.invert_yaxis()
            st.pyplot(fig); plt.close()
        else:
            st.info("No data for selected filters.")

    with fc_col2:
        st.markdown('<div class="section-hdr">Top Medications (Filtered)</div>', unsafe_allow_html=True)
        if not df_rx_f.empty:
            top_rx_f = df_rx_f["DRUG"].value_counts().head(top_n)
            fig, ax = dark_fig(5, 4)
            ax.barh(top_rx_f.index, top_rx_f.values, color="#f472b6")
            ax.set_xlabel("Count", color="#94a3b8")
            ax.invert_yaxis()
            st.pyplot(fig); plt.close()
        else:
            st.info("No data for selected filters.")

    st.markdown('<div class="section-hdr">Filtered ICU Stay Distribution</div>', unsafe_allow_html=True)
    if not df_icu_f.empty:
        fig, ax = dark_fig(10, 3)
        ax.hist(df_icu_f["LOS_HOURS"].dropna(), bins=30, color="#34d399", edgecolor="#0d1b2a")
        ax.set_xlabel("Hours", color="#94a3b8")
        ax.set_ylabel("Count", color="#94a3b8")
        st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — AI MEDICAL ASSISTANT
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-hdr">💬 Chat with ICU Data</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#64748b;font-size:0.88rem">Ask anything about the MIMIC-III dataset. The AI has access to all patient, diagnosis, ICU, and medication data.</p>', unsafe_allow_html=True)

    system_prompt = f"""You are an expert ICU medical data analyst assistant.
You have access to the MIMIC-III clinical database with the following data:
- {len(df_p)} patients | Gender breakdown: {df_p['GENDER'].value_counts().to_dict()}
- {len(df_a)} admissions | Types: {df_a['ADMISSION_TYPE'].value_counts().to_dict()}
- {len(df_icu)} ICU stays | Avg duration: {df_icu2['LOS_HOURS'].mean():.1f} hours
- {len(df_dx)} diagnoses | Top 5 ICD9 codes: {df_dx['ICD9_CODE'].value_counts().head(5).to_dict()}
- {len(df_rx)} prescriptions | Top 5 drugs: {df_rx['DRUG'].value_counts().head(5).to_dict()}
- {len(df_lab)} lab events

Answer clearly and concisely. When mentioning ICD9 codes, always explain what the diagnosis means in plain English.
If asked about a specific patient, explain you can provide population-level insights."""

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.markdown('<div class="section-hdr">Quick Questions</div>', unsafe_allow_html=True)
    qc1, qc2, qc3, qc4 = st.columns(4)
    quick_q = None
    with qc1:
        if st.button("👥 Patient demographics"):
            quick_q = "Give me a full demographic breakdown of all patients including gender, and age insights."
    with qc2:
        if st.button("🏥 ICU statistics"):
            quick_q = "What are the key ICU statistics? Include average stay, longest stay, and patterns."
    with qc3:
        if st.button("🧬 Top diagnoses"):
            quick_q = "What are the most common diagnoses? Explain each ICD9 code in plain English."
    with qc4:
        if st.button("💊 Medication insights"):
            quick_q = "What are the most prescribed medications and what are they typically used for?"

    if quick_q:
        st.session_state.messages.append({"role": "user", "content": quick_q})
        with st.spinner("AI is analyzing the data..."):
            msgs = [{"role": "system", "content": system_prompt}] + st.session_state.messages
            reply = ask_ai(msgs)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-ai">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    user_input = st.text_input("Type your question...", placeholder="e.g. How many patients had heart failure?", key="chat_input")

    if st.button("Send ➤") and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("AI is thinking..."):
            msgs = [{"role": "system", "content": system_prompt}] + st.session_state.messages
            reply = ask_ai(msgs)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

st.markdown("---")
st.markdown('<p style="text-align:center;color:#334155;font-size:0.78rem">AI-Powered ICU Analysis & Medical Assistant · MIMIC-III · Built with Streamlit + Azure + Groq AI</p>', unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import requests
import json
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="ICU AI Medical Assistant", page_icon="🏥", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 40%, #0a1628 70%, #080d1a 100%);
    min-height: 100vh;
}

.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* GLASSMORPHISM CARDS */
.glass-card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

.glass-metric {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 16px;
    padding: 20px 12px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(56,189,248,0.05), inset 0 1px 0 rgba(255,255,255,0.1);
    transition: transform 0.2s;
}
.glass-metric:hover { transform: translateY(-2px); }
.metric-num { font-size: 2rem; font-weight: 800; color: #38bdf8; letter-spacing: -1px; }
.metric-lbl { font-size: 0.7rem; color: #64748b; margin-top: 4px; letter-spacing: 1px; text-transform: uppercase; }

/* HEADER */
.app-header {
    background: rgba(255,255,255,0.03);
    backdrop-filter: blur(30px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px;
    padding: 28px 32px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.app-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
}
.app-subtitle { color: #475569; font-size: 0.9rem; margin-top: 6px; }

/* SECTION HEADERS */
.section-hdr {
    font-size: 1rem;
    font-weight: 600;
    color: #38bdf8;
    border-left: 3px solid #0ea5e9;
    padding-left: 12px;
    margin: 20px 0 14px;
    letter-spacing: 0.3px;
}

/* RISK SCORE */
.risk-high   { background: rgba(239,68,68,0.15);  border: 1px solid rgba(239,68,68,0.3);  border-radius: 12px; padding: 16px; text-align:center; }
.risk-medium { background: rgba(245,158,11,0.15); border: 1px solid rgba(245,158,11,0.3); border-radius: 12px; padding: 16px; text-align:center; }
.risk-low    { background: rgba(34,197,94,0.15);  border: 1px solid rgba(34,197,94,0.3);  border-radius: 12px; padding: 16px; text-align:center; }
.risk-score  { font-size: 3rem; font-weight: 800; }
.risk-label  { font-size: 0.85rem; font-weight: 600; margin-top: 4px; }

/* PATIENT CARD */
.patient-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(56,189,248,0.15);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 12px;
}
.badge { display:inline-block; padding:4px 12px; border-radius:20px; font-size:0.73rem; font-weight:600; margin:2px; }
.badge-blue  { background:rgba(56,189,248,0.15); color:#7dd3fc; border:1px solid rgba(56,189,248,0.2); }
.badge-green { background:rgba(34,197,94,0.15);  color:#86efac; border:1px solid rgba(34,197,94,0.2); }
.badge-red   { background:rgba(239,68,68,0.15);  color:#fca5a5; border:1px solid rgba(239,68,68,0.2); }

/* CHAT */
.chat-user {
    background: linear-gradient(135deg, rgba(29,78,216,0.6), rgba(37,99,235,0.6));
    backdrop-filter: blur(10px);
    border: 1px solid rgba(96,165,250,0.2);
    border-radius: 18px 18px 4px 18px;
    padding: 14px 18px; margin: 8px 0; margin-left: 18%;
    color: white; font-size: 0.92rem;
}
.chat-ai {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px 18px 18px 4px;
    padding: 14px 18px; margin: 8px 0; margin-right: 18%;
    color: #cbd5e1; font-size: 0.92rem; line-height: 1.6;
}

/* BUTTONS */
.stButton > button {
    background: rgba(255,255,255,0.06) !important;
    backdrop-filter: blur(10px) !important;
    color: #38bdf8 !important;
    border: 1px solid rgba(56,189,248,0.3) !important;
    border-radius: 10px !important;
    padding: 8px 18px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: rgba(56,189,248,0.15) !important;
    border-color: rgba(56,189,248,0.6) !important;
}

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    backdrop-filter: blur(10px);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid rgba(255,255,255,0.07);
}
.stTabs [data-baseweb="tab"] { color: #64748b; font-weight: 500; border-radius: 8px; }
.stTabs [aria-selected="true"] { background: rgba(56,189,248,0.15) !important; color: #38bdf8 !important; }

/* INPUTS */
.stTextInput > div > input, .stSelectbox > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: white !important;
    border-radius: 10px !important;
}

/* DATAFRAME */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* DIVIDER */
hr { border-color: rgba(255,255,255,0.06) !important; }
</style>
""", unsafe_allow_html=True)

# ── SECURE CONFIG ──────────────────────────────────────────────────────────
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    GROQ_API_KEY = ""

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# ── DATA LOADER ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    with open("mimic_data.json","r") as f:
        raw = json.load(f)
    dfs = {}
    for k in ["patients","admissions","icustays","diagnoses","prescriptions","labevents"]:
        df = pd.DataFrame(json.loads(raw[k]))
        df.columns = df.columns.str.upper()
        dfs[k] = df
    return dfs

# ── AI ─────────────────────────────────────────────────────────────────────
def ask_ai(messages):
    headers = {"Authorization":f"Bearer {GROQ_API_KEY}","Content-Type":"application/json"}
    payload = {"model":"llama-3.3-70b-versatile","messages":messages,"max_tokens":900}
    r = requests.post(GROQ_URL,headers=headers,json=payload)
    res = r.json()
    return res["choices"][0]["message"]["content"] if "choices" in res else f"Error: {res}"

# ── CHART HELPER ───────────────────────────────────────────────────────────
def glass_fig(w=6,h=4):
    fig,ax = plt.subplots(figsize=(w,h),facecolor="#0d1b2a")
    ax.set_facecolor("#111827")
    ax.tick_params(colors="#94a3b8",labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3a5f")
    return fig,ax

# ── RISK SCORE ─────────────────────────────────────────────────────────────
def compute_risk(pid, df_a, df_dx, df_icu2, df_lab):
    score = 0
    adm_p = df_a[df_a["SUBJECT_ID"]==pid]
    if len(adm_p) > 2: score += 20
    dx_p = df_dx[df_dx["SUBJECT_ID"]==pid]
    high_risk_codes = ["4019","42731","5849","4280","25000","41401","2724","5990"]
    matched = dx_p["ICD9_CODE"].astype(str).isin(high_risk_codes).sum()
    score += min(matched * 10, 40)
    icu_p = df_icu2[df_icu2["SUBJECT_ID"]==pid]
    if not icu_p.empty:
        avg_los = icu_p["LOS_HOURS"].mean()
        if avg_los > 200: score += 30
        elif avg_los > 100: score += 20
        elif avg_los > 50: score += 10
    score = min(score, 100)
    if score >= 70: level,color = "HIGH RISK","#ef4444"
    elif score >= 40: level,color = "MEDIUM RISK","#f59e0b"
    else: level,color = "LOW RISK","#22c55e"
    return score,level,color

# ── LOAD ───────────────────────────────────────────────────────────────────
with st.spinner("🔄 Loading MIMIC-III data..."):
    dfs = load_data()

df_p  = dfs["patients"]
df_a  = dfs["admissions"]
df_icu= dfs["icustays"]
df_dx = dfs["diagnoses"]
df_rx = dfs["prescriptions"]
df_lab= dfs["labevents"]

df_icu2 = df_icu.copy()
df_icu2["INTIME"]    = pd.to_datetime(df_icu2["INTIME"])
df_icu2["OUTTIME"]   = pd.to_datetime(df_icu2["OUTTIME"])
df_icu2["LOS_HOURS"] = (df_icu2["OUTTIME"]-df_icu2["INTIME"]).dt.total_seconds()/3600
df_icu2 = df_icu2[df_icu2["LOS_HOURS"]<1000]

# ── HEADER ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="app-title">🏥 AI-Powered ICU Analysis & Medical Assistant</div>
  <div class="app-subtitle">MIMIC-III Clinical Database · Beth Israel Deaconess Medical Center · 2001–2012 · Powered by LLaMA 3.3 + Groq</div>
</div>
""", unsafe_allow_html=True)

# ── METRICS ────────────────────────────────────────────────────────────────
cols = st.columns(6)
for col,num,lbl in zip(cols,
    [len(df_p),len(df_a),len(df_icu),len(df_dx),len(df_rx),len(df_lab)],
    ["Patients","Admissions","ICU Stays","Diagnoses","Prescriptions","Lab Events"]):
    with col:
        st.markdown(f'<div class="glass-metric"><div class="metric-num">{num:,}</div><div class="metric-lbl">{lbl}</div></div>',unsafe_allow_html=True)

st.markdown("<br>",unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs([
    "📊 Dashboard","🔍 Patient Search","⚠️ Risk Score",
    "🧪 Lab Trends","🔄 Readmission","🔬 Filter & Explore","🤖 AI Assistant"
])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════
with tab1:
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="glass-metric"><div class="metric-num">{df_icu2["LOS_HOURS"].mean():.1f}h</div><div class="metric-lbl">Avg ICU Stay</div></div>',unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="glass-metric"><div class="metric-num">{df_icu2["LOS_HOURS"].max():.0f}h</div><div class="metric-lbl">Max ICU Stay</div></div>',unsafe_allow_html=True)
    with c3:
        emerg=(df_a["ADMISSION_TYPE"]=="EMERGENCY").sum()
        st.markdown(f'<div class="glass-metric"><div class="metric-num">{emerg}</div><div class="metric-lbl">Emergency Admissions</div></div>',unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    col1,col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-hdr">Gender Distribution</div>',unsafe_allow_html=True)
        gender=df_p["GENDER"].value_counts()
        fig,ax=glass_fig(5,3.5)
        wedges,_,autotexts=ax.pie(gender,labels=gender.index,autopct="%1.1f%%",
            colors=["#38bdf8","#f472b6"],startangle=90,
            textprops={"color":"white","fontsize":11},
            wedgeprops={"edgecolor":"#0d1b2a","linewidth":2})
        st.pyplot(fig,transparent=True); plt.close()

    with col2:
        st.markdown('<div class="section-hdr">Admission Types</div>',unsafe_allow_html=True)
        adm=df_a["ADMISSION_TYPE"].value_counts()
        fig,ax=glass_fig(5,3.5)
        bars=ax.bar(adm.index,adm.values,color=["#38bdf8","#f472b6","#34d399"],
                    edgecolor="#1e3a5f",linewidth=1.5,width=0.6)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,
                   str(int(bar.get_height())),ha="center",color="white",fontsize=10,fontweight="bold")
        ax.set_ylabel("Count",color="#94a3b8",fontsize=9)
        plt.xticks(rotation=10,color="white",fontsize=9)
        ax.set_ylim(0,adm.max()*1.15)
        st.pyplot(fig,transparent=True); plt.close()

    col3,col4=st.columns(2)
    with col3:
        st.markdown('<div class="section-hdr">Top 10 Diagnoses (ICD9)</div>',unsafe_allow_html=True)
        top_dx=df_dx["ICD9_CODE"].value_counts().head(10)
        fig,ax=glass_fig(5,4.5)
        colors=plt.cm.Blues(np.linspace(0.4,0.9,len(top_dx)))[::-1]
        bars=ax.barh(top_dx.index.astype(str),top_dx.values,color=colors,edgecolor="none",height=0.7)
        ax.set_xlabel("Count",color="#94a3b8",fontsize=9)
        ax.invert_yaxis()
        for bar in bars:
            ax.text(bar.get_width()+0.3,bar.get_y()+bar.get_height()/2,
                   str(int(bar.get_width())),va="center",color="#94a3b8",fontsize=8)
        st.pyplot(fig,transparent=True); plt.close()

    with col4:
        st.markdown('<div class="section-hdr">Top 10 Medications</div>',unsafe_allow_html=True)
        top_rx=df_rx["DRUG"].value_counts().head(10)
        fig,ax=glass_fig(5,4.5)
        colors=plt.cm.RdPu(np.linspace(0.4,0.9,len(top_rx)))[::-1]
        bars=ax.barh(top_rx.index,top_rx.values,color=colors,edgecolor="none",height=0.7)
        ax.set_xlabel("Count",color="#94a3b8",fontsize=9)
        ax.invert_yaxis()
        for bar in bars:
            ax.text(bar.get_width()+0.3,bar.get_y()+bar.get_height()/2,
                   str(int(bar.get_width())),va="center",color="#94a3b8",fontsize=8)
        st.pyplot(fig,transparent=True); plt.close()

    st.markdown('<div class="section-hdr">ICU Stay Duration Distribution</div>',unsafe_allow_html=True)
    fig,ax=glass_fig(12,3)
    n,bins,patches=ax.hist(df_icu2["LOS_HOURS"].dropna(),bins=40,edgecolor="none")
    for i,(patch,val) in enumerate(zip(patches,n)):
        patch.set_facecolor(plt.cm.cool(i/len(patches)))
        patch.set_alpha(0.8)
    ax.set_xlabel("Hours in ICU",color="#94a3b8",fontsize=9)
    ax.set_ylabel("Number of Stays",color="#94a3b8",fontsize=9)
    ax.axvline(df_icu2["LOS_HOURS"].mean(),color="#f59e0b",linestyle="--",linewidth=1.5,label=f'Mean: {df_icu2["LOS_HOURS"].mean():.0f}h')
    ax.legend(facecolor="none",edgecolor="#1e3a5f",labelcolor="white",fontsize=9)
    st.pyplot(fig,transparent=True); plt.close()

# ══════════════════════════════════════════════════════════════════════
# TAB 2 — PATIENT SEARCH
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-hdr">Search Patient by ID</div>',unsafe_allow_html=True)
    all_ids=sorted(df_p["SUBJECT_ID"].unique().tolist())
    cs1,cs2=st.columns([3,1])
    with cs1: patient_id=st.selectbox("Select Patient ID",all_ids)
    with cs2: st.markdown("<br>",unsafe_allow_html=True); search_btn=st.button("🔍 Search")

    if patient_id:
        pid=int(patient_id)
        row=df_p[df_p["SUBJECT_ID"]==pid].iloc[0]
        score,level,color=compute_risk(pid,df_a,df_dx,df_icu2,df_lab)

        st.markdown(f"""
        <div class="patient-card">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <div>
                    <div style="font-size:1.4rem;font-weight:700;color:#38bdf8">👤 Patient #{pid}</div>
                    <div style="margin-top:8px">
                        <span class="badge badge-blue">Gender: {row.get('GENDER','N/A')}</span>
                        <span class="badge badge-green">DOB: {str(row.get('DOB','N/A'))[:10]}</span>
                        <span class="badge badge-red">Deceased: {'Yes' if pd.notna(row.get('DOD')) else 'No'}</span>
                    </div>
                </div>
                <div style="text-align:center;border:2px solid {color};border-radius:12px;padding:12px 20px;background:#1a1f2e">
                    <div style="font-size:2rem;font-weight:800;color:{color}">{score}</div>
                    <div style="font-size:0.75rem;font-weight:600;color:{color}">{level}</div>
                </div>
            </div>
        </div>""",unsafe_allow_html=True)

        r1,r2,r3=st.columns(3)
        with r1:
            st.markdown('<div class="section-hdr">🏥 Admissions</div>',unsafe_allow_html=True)
            adm_p=df_a[df_a["SUBJECT_ID"]==pid][["ADMITTIME","ADMISSION_TYPE","DIAGNOSIS"]].head(5)
            st.dataframe(adm_p,use_container_width=True,hide_index=True) if not adm_p.empty else st.info("No admissions.")
        with r2:
            st.markdown('<div class="section-hdr">🧬 Diagnoses</div>',unsafe_allow_html=True)
            dx_p=df_dx[df_dx["SUBJECT_ID"]==pid][["ICD9_CODE","SEQ_NUM"]].head(10)
            st.dataframe(dx_p,use_container_width=True,hide_index=True) if not dx_p.empty else st.info("No diagnoses.")
        with r3:
            st.markdown('<div class="section-hdr">💊 Prescriptions</div>',unsafe_allow_html=True)
            rx_p=df_rx[df_rx["SUBJECT_ID"]==pid][["DRUG","DOSE_VAL_RX","ROUTE"]].head(10)
            st.dataframe(rx_p,use_container_width=True,hide_index=True) if not rx_p.empty else st.info("No prescriptions.")

        st.markdown('<div class="section-hdr">🛏️ ICU Stays</div>',unsafe_allow_html=True)
        icu_p=df_icu2[df_icu2["SUBJECT_ID"]==pid][["ICUSTAY_ID","INTIME","OUTTIME","LOS_HOURS","FIRST_CAREUNIT"]].head(5)
        st.dataframe(icu_p,use_container_width=True,hide_index=True) if not icu_p.empty else st.info("No ICU stays.")

# ══════════════════════════════════════════════════════════════════════
# TAB 3 — RISK SCORE
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-hdr">⚠️ Patient Risk Score & Mortality Predictor</div>',unsafe_allow_html=True)
    st.markdown('<p style="color:#64748b;font-size:0.88rem">Risk score based on number of admissions, high-risk ICD9 diagnoses, and ICU stay duration.</p>',unsafe_allow_html=True)

    risk_pid=st.selectbox("Select Patient for Risk Analysis",all_ids,key="risk_pid")

    if risk_pid:
        pid=int(risk_pid)
        score,level,color=compute_risk(pid,df_a,df_dx,df_icu2,df_lab)

        rc1,rc2,rc3=st.columns(3)
        with rc1:
            admissions_count=len(df_a[df_a["SUBJECT_ID"]==pid])
            st.markdown(f'<div class="glass-metric"><div class="metric-num">{admissions_count}</div><div class="metric-lbl">Total Admissions</div></div>',unsafe_allow_html=True)
        with rc2:
            dx_count=len(df_dx[df_dx["SUBJECT_ID"]==pid])
            st.markdown(f'<div class="glass-metric"><div class="metric-num">{dx_count}</div><div class="metric-lbl">Total Diagnoses</div></div>',unsafe_allow_html=True)
        with rc3:
            icu_count=len(df_icu2[df_icu2["SUBJECT_ID"]==pid])
            st.markdown(f'<div class="glass-metric"><div class="metric-num">{icu_count}</div><div class="metric-lbl">ICU Stays</div></div>',unsafe_allow_html=True)

        st.markdown("<br>",unsafe_allow_html=True)
        sc1,sc2=st.columns([1,2])

        with sc1:
            css_class = "risk-high" if score>=70 else ("risk-medium" if score>=40 else "risk-low")
            st.markdown(f"""
            <div class="{css_class}" style="margin-top:10px">
                <div class="risk-score" style="color:{color}">{score}/100</div>
                <div class="risk-label" style="color:{color}">{level}</div>
                <div style="margin-top:12px;color:#94a3b8;font-size:0.8rem">
                    {'⚠️ Requires immediate attention' if score>=70 else ('📋 Monitor closely' if score>=40 else '✅ Stable condition')}
                </div>
            </div>""",unsafe_allow_html=True)

        with sc2:
            st.markdown('<div class="section-hdr">Risk Breakdown</div>',unsafe_allow_html=True)
            adm_score=min(len(df_a[df_a["SUBJECT_ID"]==pid])*10,20) if len(df_a[df_a["SUBJECT_ID"]==pid])>2 else 0
            high_risk=["4019","42731","5849","4280","25000","41401","2724","5990"]
            dx_score=min(df_dx[df_dx["SUBJECT_ID"]==pid]["ICD9_CODE"].astype(str).isin(high_risk).sum()*10,40)
            icu_avg=df_icu2[df_icu2["SUBJECT_ID"]==pid]["LOS_HOURS"].mean() if not df_icu2[df_icu2["SUBJECT_ID"]==pid].empty else 0
            icu_score=30 if icu_avg>200 else (20 if icu_avg>100 else (10 if icu_avg>50 else 0))

            categories=["Readmissions","High-Risk Diagnoses","ICU Duration"]
            values=[adm_score,dx_score,icu_score]
            max_vals=[20,40,40]

            fig,ax=glass_fig(7,3)
            colors_bar=["#38bdf8","#f472b6","#34d399"]
            y=range(len(categories))
            ax.barh(y,[m for m in max_vals],color="#111827",edgecolor="#1e3a5f",height=0.5)
            ax.barh(y,values,color=colors_bar,edgecolor="none",height=0.5,alpha=0.85)
            ax.set_yticks(y); ax.set_yticklabels(categories,color="white",fontsize=10)
            ax.set_xlabel("Score",color="#94a3b8",fontsize=9)
            for i,v in enumerate(values):
                ax.text(v+0.5,i,f"{v}pts",va="center",color="white",fontsize=9,fontweight="bold")
            st.pyplot(fig,transparent=True); plt.close()

    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown('<div class="section-hdr">📊 Risk Distribution Across All Patients</div>',unsafe_allow_html=True)

    with st.spinner("Computing risk scores for all patients..."):
        sample_pids=df_p["SUBJECT_ID"].head(50).tolist()
        risk_scores=[compute_risk(p,df_a,df_dx,df_icu2,df_lab)[0] for p in sample_pids]
        high=sum(1 for s in risk_scores if s>=70)
        med=sum(1 for s in risk_scores if 40<=s<70)
        low=sum(1 for s in risk_scores if s<40)

    rd1,rd2,rd3=st.columns(3)
    with rd1: st.markdown(f'<div class="risk-high"><div class="risk-score" style="color:#ef4444">{high}</div><div class="risk-label" style="color:#ef4444">HIGH RISK</div></div>',unsafe_allow_html=True)
    with rd2: st.markdown(f'<div class="risk-medium"><div class="risk-score" style="color:#f59e0b">{med}</div><div class="risk-label" style="color:#f59e0b">MEDIUM RISK</div></div>',unsafe_allow_html=True)
    with rd3: st.markdown(f'<div class="risk-low"><div class="risk-score" style="color:#22c55e">{low}</div><div class="risk-label" style="color:#22c55e">LOW RISK</div></div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 4 — LAB TRENDS
# ══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-hdr">🧪 Lab Results Trend Analysis</div>',unsafe_allow_html=True)

    lt1,lt2=st.columns([2,1])
    with lt1: lab_pid=st.selectbox("Select Patient",all_ids,key="lab_pid")
    with lt2:
        available_items=df_lab["ITEMID"].value_counts().head(20).index.tolist()
        selected_item=st.selectbox("Lab Test (ITEMID)",available_items)

    lab_p=df_lab[(df_lab["SUBJECT_ID"]==int(lab_pid))].copy() if lab_pid else pd.DataFrame()

    if not lab_p.empty and "CHARTTIME" in lab_p.columns:
        lab_p["CHARTTIME"]=pd.to_datetime(lab_p["CHARTTIME"])
        lab_item=lab_p[lab_p["ITEMID"]==selected_item].sort_values("CHARTTIME")

        if not lab_item.empty and "VALUENUM" in lab_item.columns:
            lab_clean=lab_item.dropna(subset=["VALUENUM"])
            if not lab_clean.empty:
                st.markdown(f'<div class="section-hdr">Lab Test {selected_item} — Patient #{lab_pid}</div>',unsafe_allow_html=True)

                lc1,lc2,lc3=st.columns(3)
                with lc1: st.markdown(f'<div class="glass-metric"><div class="metric-num">{lab_clean["VALUENUM"].mean():.2f}</div><div class="metric-lbl">Mean Value</div></div>',unsafe_allow_html=True)
                with lc2: st.markdown(f'<div class="glass-metric"><div class="metric-num">{lab_clean["VALUENUM"].min():.2f}</div><div class="metric-lbl">Min Value</div></div>',unsafe_allow_html=True)
                with lc3: st.markdown(f'<div class="glass-metric"><div class="metric-num">{lab_clean["VALUENUM"].max():.2f}</div><div class="metric-lbl">Max Value</div></div>',unsafe_allow_html=True)

                st.markdown("<br>",unsafe_allow_html=True)
                fig,ax=glass_fig(12,4)
                ax.plot(lab_clean["CHARTTIME"],lab_clean["VALUENUM"],
                       color="#38bdf8",linewidth=2,marker="o",markersize=5,markerfacecolor="#f472b6",markeredgecolor="none")
                ax.fill_between(lab_clean["CHARTTIME"],lab_clean["VALUENUM"],
                               alpha=0.15,color="#38bdf8")
                mean_val=lab_clean["VALUENUM"].mean()
                ax.axhline(mean_val,color="#f59e0b",linestyle="--",linewidth=1.2,alpha=0.7,label=f"Mean: {mean_val:.2f}")
                ax.set_xlabel("Time",color="#94a3b8",fontsize=9)
                ax.set_ylabel("Value",color="#94a3b8",fontsize=9)
                ax.legend(facecolor="none",edgecolor="#1e3a5f",labelcolor="white",fontsize=9)
                plt.xticks(rotation=30,color="#94a3b8",fontsize=8)
                st.pyplot(fig,transparent=True); plt.close()
            else:
                st.info("No numeric values for this lab test.")
        else:
            st.info("No data for selected lab test.")
    else:
        st.info("No lab data found for this patient.")

    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown('<div class="section-hdr">📊 Most Common Lab Tests (Overall)</div>',unsafe_allow_html=True)
    top_labs=df_lab["ITEMID"].value_counts().head(15)
    fig,ax=glass_fig(12,4)
    colors=plt.cm.viridis(np.linspace(0.3,0.9,len(top_labs)))
    bars=ax.bar(top_labs.index.astype(str),top_labs.values,color=colors,edgecolor="none",width=0.7)
    ax.set_xlabel("Lab Item ID",color="#94a3b8",fontsize=9)
    ax.set_ylabel("Count",color="#94a3b8",fontsize=9)
    plt.xticks(rotation=45,color="#94a3b8",fontsize=8)
    st.pyplot(fig,transparent=True); plt.close()

# ══════════════════════════════════════════════════════════════════════
# TAB 5 — READMISSION
# ══════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-hdr">🔄 ICU Readmission Analysis</div>',unsafe_allow_html=True)

    adm_counts=df_a.groupby("SUBJECT_ID").size().reset_index(name="ADMISSION_COUNT")
    readmitted=adm_counts[adm_counts["ADMISSION_COUNT"]>1]
    not_readmitted=adm_counts[adm_counts["ADMISSION_COUNT"]==1]

    ra1,ra2,ra3,ra4=st.columns(4)
    with ra1: st.markdown(f'<div class="glass-metric"><div class="metric-num">{len(readmitted)}</div><div class="metric-lbl">Readmitted Patients</div></div>',unsafe_allow_html=True)
    with ra2: st.markdown(f'<div class="glass-metric"><div class="metric-num">{len(not_readmitted)}</div><div class="metric-lbl">Single Admission</div></div>',unsafe_allow_html=True)
    with ra3:
        rate=len(readmitted)/len(adm_counts)*100
        st.markdown(f'<div class="glass-metric"><div class="metric-num">{rate:.1f}%</div><div class="metric-lbl">Readmission Rate</div></div>',unsafe_allow_html=True)
    with ra4:
        avg_readm=readmitted["ADMISSION_COUNT"].mean()
        st.markdown(f'<div class="glass-metric"><div class="metric-num">{avg_readm:.1f}</div><div class="metric-lbl">Avg Admissions</div></div>',unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    rr1,rr2=st.columns(2)

    with rr1:
        st.markdown('<div class="section-hdr">Readmission vs Single Admission</div>',unsafe_allow_html=True)
        fig,ax=glass_fig(5,4)
        sizes=[len(readmitted),len(not_readmitted)]
        labels=[f"Readmitted\n({len(readmitted)})",f"Single\n({len(not_readmitted)})"]
        wedges,_,autotexts=ax.pie(sizes,labels=labels,autopct="%1.1f%%",
            colors=["#f472b6","#38bdf8"],startangle=90,
            textprops={"color":"white","fontsize":10},
            wedgeprops={"edgecolor":"#0d1b2a","linewidth":2})
        st.pyplot(fig,transparent=True); plt.close()

    with rr2:
        st.markdown('<div class="section-hdr">Admission Count Distribution</div>',unsafe_allow_html=True)
        fig,ax=glass_fig(5,4)
        count_dist=adm_counts["ADMISSION_COUNT"].value_counts().sort_index()
        bars=ax.bar(count_dist.index.astype(str),count_dist.values,
               color=plt.cm.cool(np.linspace(0.2,0.9,len(count_dist))),edgecolor="none",width=0.7)
        ax.set_xlabel("Number of Admissions",color="#94a3b8",fontsize=9)
        ax.set_ylabel("Number of Patients",color="#94a3b8",fontsize=9)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.2,
                   str(int(bar.get_height())),ha="center",color="white",fontsize=9)
        st.pyplot(fig,transparent=True); plt.close()

    st.markdown('<div class="section-hdr">Top 10 Most Readmitted Patients</div>',unsafe_allow_html=True)
    top_readmitted=adm_counts.nlargest(10,"ADMISSION_COUNT").reset_index(drop=True)
    top_readmitted.index+=1
    fig,ax=glass_fig(12,4)
    colors=plt.cm.YlOrRd(np.linspace(0.3,0.9,len(top_readmitted)))[::-1]
    bars=ax.barh(top_readmitted["SUBJECT_ID"].astype(str),
                top_readmitted["ADMISSION_COUNT"],color=colors,edgecolor="none",height=0.6)
    ax.set_xlabel("Number of Admissions",color="#94a3b8",fontsize=9)
    ax.set_ylabel("Patient ID",color="#94a3b8",fontsize=9)
    ax.invert_yaxis()
    for bar in bars:
        ax.text(bar.get_width()+0.05,bar.get_y()+bar.get_height()/2,
               str(int(bar.get_width())),va="center",color="white",fontsize=9,fontweight="bold")
    st.pyplot(fig,transparent=True); plt.close()

    st.markdown('<div class="section-hdr">Readmission by Admission Type</div>',unsafe_allow_html=True)
    readmitted_pids=readmitted["SUBJECT_ID"].tolist()
    df_a_readm=df_a[df_a["SUBJECT_ID"].isin(readmitted_pids)]
    adm_type_readm=df_a_readm["ADMISSION_TYPE"].value_counts()
    fig,ax=glass_fig(8,3)
    ax.bar(adm_type_readm.index,adm_type_readm.values,
           color=["#38bdf8","#f472b6","#34d399"],edgecolor="none",width=0.5)
    ax.set_ylabel("Count",color="#94a3b8",fontsize=9)
    plt.xticks(color="white",fontsize=10)
    st.pyplot(fig,transparent=True); plt.close()

# ══════════════════════════════════════════════════════════════════════
# TAB 6 — FILTER & EXPLORE
# ══════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-hdr">Filter Dashboard</div>',unsafe_allow_html=True)
    fc1,fc2,fc3,fc4=st.columns(4)
    with fc1: adm_filter=st.selectbox("Admission Type",["All"]+df_a["ADMISSION_TYPE"].unique().tolist())
    with fc2: gender_filter=st.selectbox("Gender",["All","M","F"])
    with fc3: icu_min,icu_max=st.slider("ICU Stay (hours)",0,500,(0,500))
    with fc4: top_n=st.slider("Top N",5,20,10)

    df_a_f=df_a if adm_filter=="All" else df_a[df_a["ADMISSION_TYPE"]==adm_filter]
    df_p_f=df_p if gender_filter=="All" else df_p[df_p["GENDER"]==gender_filter]
    df_icu_f=df_icu2[(df_icu2["LOS_HOURS"]>=icu_min)&(df_icu2["LOS_HOURS"]<=icu_max)]
    filtered_pids=set(df_a_f["SUBJECT_ID"])&set(df_p_f["SUBJECT_ID"])&set(df_icu_f["SUBJECT_ID"])
    df_dx_f=df_dx[df_dx["SUBJECT_ID"].isin(filtered_pids)]
    df_rx_f=df_rx[df_rx["SUBJECT_ID"].isin(filtered_pids)]

    m1,m2,m3=st.columns(3)
    with m1: st.markdown(f'<div class="glass-metric"><div class="metric-num">{len(filtered_pids)}</div><div class="metric-lbl">Filtered Patients</div></div>',unsafe_allow_html=True)
    with m2: st.markdown(f'<div class="glass-metric"><div class="metric-num">{len(df_a_f)}</div><div class="metric-lbl">Admissions</div></div>',unsafe_allow_html=True)
    with m3: st.markdown(f'<div class="glass-metric"><div class="metric-num">{len(df_icu_f)}</div><div class="metric-lbl">ICU Stays</div></div>',unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    fc_col1,fc_col2=st.columns(2)
    with fc_col1:
        st.markdown('<div class="section-hdr">Top Diagnoses (Filtered)</div>',unsafe_allow_html=True)
        if not df_dx_f.empty:
            top_dx_f=df_dx_f["ICD9_CODE"].value_counts().head(top_n)
            fig,ax=glass_fig(5,4)
            ax.barh(top_dx_f.index.astype(str),top_dx_f.values,
                   color=plt.cm.Blues(np.linspace(0.4,0.9,len(top_dx_f)))[::-1],edgecolor="none",height=0.7)
            ax.set_xlabel("Count",color="#94a3b8",fontsize=9); ax.invert_yaxis()
            st.pyplot(fig,transparent=True); plt.close()
        else: st.info("No data for selected filters.")
    with fc_col2:
        st.markdown('<div class="section-hdr">Top Medications (Filtered)</div>',unsafe_allow_html=True)
        if not df_rx_f.empty:
            top_rx_f=df_rx_f["DRUG"].value_counts().head(top_n)
            fig,ax=glass_fig(5,4)
            ax.barh(top_rx_f.index,top_rx_f.values,
                   color=plt.cm.RdPu(np.linspace(0.4,0.9,len(top_rx_f)))[::-1],edgecolor="none",height=0.7)
            ax.set_xlabel("Count",color="#94a3b8",fontsize=9); ax.invert_yaxis()
            st.pyplot(fig,transparent=True); plt.close()
        else: st.info("No data for selected filters.")

# ══════════════════════════════════════════════════════════════════════
# TAB 7 — AI ASSISTANT
# ══════════════════════════════════════════════════════════════════════
with tab7:
    st.markdown('<div class="section-hdr">💬 Chat with ICU Data</div>',unsafe_allow_html=True)
    st.markdown('<p style="color:#64748b;font-size:0.88rem">Ask anything about MIMIC-III in plain English. AI powered by LLaMA 3.3 70B.</p>',unsafe_allow_html=True)

    system_prompt=f"""You are an expert ICU medical data analyst for MIMIC-III.
Data: {len(df_p)} patients | {len(df_a)} admissions | {len(df_icu)} ICU stays
Gender: {df_p['GENDER'].value_counts().to_dict()}
Admission types: {df_a['ADMISSION_TYPE'].value_counts().to_dict()}
Avg ICU stay: {df_icu2['LOS_HOURS'].mean():.1f} hours
Top diagnoses: {df_dx['ICD9_CODE'].value_counts().head(5).to_dict()}
Top medications: {df_rx['DRUG'].value_counts().head(5).to_dict()}
Readmission rate: {len(df_a.groupby('SUBJECT_ID').filter(lambda x: len(x)>1)['SUBJECT_ID'].unique())/len(df_p)*100:.1f}%
Always explain ICD9 codes in plain English. Be concise but thorough."""

    if "messages" not in st.session_state:
        st.session_state.messages=[]

    qc1,qc2,qc3,qc4=st.columns(4)
    quick_q=None
    with qc1:
        if st.button("👥 Demographics"): quick_q="Give me a full demographic breakdown."
    with qc2:
        if st.button("🏥 ICU Stats"): quick_q="What are the key ICU statistics?"
    with qc3:
        if st.button("🧬 Top Diagnoses"): quick_q="What are the most common diagnoses? Explain each."
    with qc4:
        if st.button("🔄 Readmissions"): quick_q="Analyze the readmission patterns in this dataset."

    if quick_q:
        st.session_state.messages.append({"role":"user","content":quick_q})
        with st.spinner("🤖 AI is analyzing..."):
            reply=ask_ai([{"role":"system","content":system_prompt}]+st.session_state.messages)
        st.session_state.messages.append({"role":"assistant","content":reply})
        st.rerun()

    for msg in st.session_state.messages:
        cls="chat-user" if msg["role"]=="user" else "chat-ai"
        icon="👤" if msg["role"]=="user" else "🤖"
        st.markdown(f'<div class="{cls}">{icon} {msg["content"]}</div>',unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    ci1,ci2=st.columns([5,1])
    with ci1: user_input=st.text_input("",placeholder="e.g. How many patients had heart failure? What is the mortality rate?",key="chat_input",label_visibility="collapsed")
    with ci2: send=st.button("Send ➤",use_container_width=True)

    if send and user_input.strip():
        st.session_state.messages.append({"role":"user","content":user_input})
        with st.spinner("🤖 Thinking..."):
            reply=ask_ai([{"role":"system","content":system_prompt}]+st.session_state.messages)
        st.session_state.messages.append({"role":"assistant","content":reply})
        st.rerun()

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages=[]
        st.rerun()

st.markdown("---")
st.markdown('<p style="text-align:center;color:#1e3a5f;font-size:0.75rem">AI-Powered ICU Analysis & Medical Assistant · MIMIC-III · Built with Streamlit + Groq LLaMA 3.3</p>',unsafe_allow_html=True)

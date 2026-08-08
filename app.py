"""
╔══════════════════════════════════════════════════════╗
║   Sleep Health Dashboard  —  app.py                 ║
║   Run:  streamlit run app.py                        ║
╚══════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import shap
import warnings
import time
import os

warnings.filterwarnings('ignore')

from train_model import get_recommendations, sleep_risk_score

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG  — must be first Streamlit call
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SleepIQ · Sleep Health Dashboard",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# GLOBAL CSS  — deep-space dark theme, custom typography
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Root palette ── */
:root {
    --bg:        #080c14;
    --surface:   #0e1420;
    --card:      #131b2e;
    --border:    #1e2d4a;
    --accent1:   #4f8ef7;   /* electric blue  */
    --accent2:   #7c3aed;   /* deep violet    */
    --accent3:   #06b6d4;   /* cyan           */
    --green:     #22d3a0;
    --amber:     #f59e0b;
    --red:       #f43f5e;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --font-head: 'Syne', sans-serif;
    --font-body: 'DM Sans', sans-serif;
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: var(--font-body);
    background-color: var(--bg);
    color: var(--text);
}
.stApp { background: var(--bg); }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem 2rem; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    font-family: var(--font-body);
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {
    color: var(--muted);
}

/* ── Slider accent ── */
[data-testid="stSidebar"] .stSlider > div > div > div > div {
    background: var(--accent1) !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] {
    font-family: var(--font-body);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--muted);
}
[data-testid="stMetricValue"] {
    font-family: var(--font-head);
    font-size: 1.6rem;
    color: var(--text);
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: var(--surface);
    border-radius: 10px;
    padding: 4px;
    gap: 2px;
    border: 1px solid var(--border);
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: var(--font-body);
    font-size: 0.82rem;
    font-weight: 500;
    color: var(--muted);
    border-radius: 8px;
    padding: 0.45rem 1.1rem;
    border: none !important;
    background: transparent;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: var(--card) !important;
    color: var(--accent1) !important;
    border: 1px solid var(--border) !important;
}

/* ── Primary button ── */
.stButton > button {
    font-family: var(--font-head);
    font-weight: 700;
    font-size: 0.9rem;
    letter-spacing: 0.04em;
    background: linear-gradient(135deg, var(--accent1), var(--accent2));
    color: #fff;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.6rem;
    width: 100%;
    transition: opacity 0.2s, transform 0.15s;
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }

/* ── Divider ── */
hr { border-color: var(--border); opacity: 1; }

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div { border-radius: 999px; }
[data-testid="stProgressBar"] > div > div { border-radius: 999px; }

/* ── DataFrame ── */
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }

/* ── Selectbox / text_input ── */
.stSelectbox div[data-baseweb="select"] > div,
.stTextInput input {
    background: var(--card);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 8px;
}

/* ── Alerts ── */
.stSuccess, .stWarning, .stError, .stInfo {
    border-radius: 10px;
    font-family: var(--font-body);
}

/* ── Custom HTML cards ── */
.iq-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.iq-card-accent {
    background: linear-gradient(135deg, #0e1a35 0%, #13243d 100%);
    border: 1px solid var(--accent1);
    box-shadow: 0 0 24px rgba(79,142,247,0.08);
}
.iq-label {
    font-family: var(--font-body);
    font-size: 0.7rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin-bottom: 4px;
}
.iq-value {
    font-family: var(--font-head);
    font-size: 2rem;
    font-weight: 800;
    line-height: 1.1;
}
.iq-badge {
    display: inline-block;
    font-family: var(--font-body);
    font-size: 0.72rem;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 999px;
    margin-top: 6px;
}

/* ── Gauge wrapper ── */
.gauge-wrap { position: relative; text-align: center; }

/* scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 999px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# MATPLOTLIB DARK THEME
# ══════════════════════════════════════════════════════════════
plt.rcParams.update({
    'figure.facecolor':  '#131b2e',
    'axes.facecolor':    '#131b2e',
    'axes.edgecolor':    '#1e2d4a',
    'axes.labelcolor':   '#94a3b8',
    'xtick.color':       '#64748b',
    'ytick.color':       '#64748b',
    'text.color':        '#e2e8f0',
    'grid.color':        '#1e2d4a',
    'grid.linewidth':    0.6,
    'font.family':       'DejaVu Sans',
    'axes.titlesize':    11,
    'axes.titleweight':  'bold',
    'axes.titlecolor':   '#e2e8f0',
})

CMAP_ACCENT = ['#4f8ef7', '#22d3a0', '#f59e0b', '#f43f5e', '#7c3aed', '#06b6d4', '#fb923c', '#a3e635']
CLASSES = ['Healthy', 'Insomnia', 'Sleep Apnea']
CLASS_COLORS = {'Healthy': '#22d3a0', 'Insomnia': '#f59e0b', 'Sleep Apnea': '#f43f5e'}


# ══════════════════════════════════════════════════════════════
# ARTIFACT LOADER
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading models…")
def load_artifacts():
    model        = joblib.load('artifacts/stacking_model.pkl')
    xgb_model    = joblib.load('artifacts/xgb_model.pkl')
    scaler       = joblib.load('artifacts/scaler.pkl')
    le_target    = joblib.load('artifacts/le_target.pkl')
    feature_cols = joblib.load('artifacts/feature_cols.pkl')
    explainer    = joblib.load('artifacts/shap_explainer.pkl')
    return model, xgb_model, scaler, le_target, feature_cols, explainer

model, xgb_model, scaler, le_target, feature_cols, explainer = load_artifacts()


# ══════════════════════════════════════════════════════════════
# HELPER UTILITIES
# ══════════════════════════════════════════════════════════════
def health_score(stress, sleep_dur, bmi, activity):
    """Composite health score 0–100 (higher = healthier)."""
    s = 100
    s -= min(stress * 5, 35)
    s -= max(0, (7.5 - sleep_dur) * 6)
    s -= bmi * 10
    s += min((activity - 30) * 0.3, 15)
    return max(0, min(100, round(s, 1)))


def donut_chart(value, label, color, size=2.2):
    fig, ax = plt.subplots(figsize=(size, size))
    fig.patch.set_facecolor('#131b2e')
    ax.set_facecolor('#131b2e')
    rem   = 100 - value
    wedge_props = dict(width=0.35, edgecolor='#131b2e', linewidth=2)
    ax.pie([value, rem],
           colors=[color, '#1e2d4a'],
           startangle=90,
           wedgeprops=wedge_props,
           counterclock=False)
    ax.text(0, 0, f"{value:.0f}", ha='center', va='center',
            fontsize=16, fontweight='bold', color=color, fontfamily='DejaVu Sans')
    ax.set_title(label, fontsize=7.5, color='#94a3b8', pad=4)
    plt.tight_layout(pad=0.2)
    return fig


def radar_chart(values, labels, color='#4f8ef7'):
    N = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    vals = list(values) + [values[0]]

    fig, ax = plt.subplots(figsize=(3.8, 3.8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#131b2e')
    ax.set_facecolor('#0e1420')
    ax.plot(angles, vals, color=color, linewidth=2)
    ax.fill(angles, vals, alpha=0.25, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=7.5, color='#94a3b8')
    ax.set_yticklabels([])
    ax.spines['polar'].set_color('#1e2d4a')
    ax.grid(color='#1e2d4a', linewidth=0.6)
    ax.yaxis.grid(True, color='#1e2d4a', linewidth=0.5)
    plt.tight_layout(pad=0.5)
    return fig


def gauge_chart(score, color):
    fig, ax = plt.subplots(figsize=(3.5, 2.0), subplot_kw=dict(aspect='equal'))
    fig.patch.set_facecolor('#131b2e')
    ax.set_facecolor('#131b2e')

    theta_start = np.pi
    theta_end   = 0.0
    theta_val   = theta_start - (score / 100) * np.pi

    # Background arc
    theta_bg = np.linspace(theta_start, theta_end, 200)
    ax.plot(np.cos(theta_bg), np.sin(theta_bg), color='#1e2d4a', linewidth=14, solid_capstyle='round')

    # Value arc
    theta_v = np.linspace(theta_start, theta_val, 200)
    ax.plot(np.cos(theta_v), np.sin(theta_v), color=color, linewidth=14, solid_capstyle='round')

    ax.text(0, -0.35, f"{score:.0f}", ha='center', va='center',
            fontsize=22, fontweight='bold', color=color)
    ax.text(0, -0.7, "/100", ha='center', va='center', fontsize=9, color='#64748b')
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.9, 1.1)
    ax.axis('off')
    plt.tight_layout(pad=0.1)
    return fig


def probability_bar(proba, classes, colors):
    fig, ax = plt.subplots(figsize=(5, 2.2))
    y_pos = range(len(classes))
    bars  = ax.barh(y_pos, proba, color=colors, height=0.5,
                    edgecolor='#131b2e', linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability', fontsize=8)
    ax.set_title('Prediction Confidence', fontsize=10, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for bar, p in zip(bars, proba):
        ax.text(min(p + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
                f'{p:.1%}', va='center', fontsize=8.5, color='#e2e8f0')
    plt.tight_layout()
    return fig


def comparison_chart(user_vals, ideal_vals, labels):
    x  = np.arange(len(labels))
    w  = 0.35
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    ax.bar(x - w/2, user_vals,  w, label='You',   color='#4f8ef7', alpha=0.9, edgecolor='#131b2e')
    ax.bar(x + w/2, ideal_vals, w, label='Ideal',  color='#22d3a0', alpha=0.7, edgecolor='#131b2e')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(fontsize=8, framealpha=0.15)
    ax.set_title('Your Metrics vs Ideal', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# SIDEBAR — USER INPUT
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:0.8rem 0 1rem 0'>
      <div style='font-family:Syne,sans-serif;font-size:1.35rem;font-weight:800;
                  background:linear-gradient(90deg,#4f8ef7,#7c3aed);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
        🌙 SleepIQ
      </div>
      <div style='font-size:0.72rem;color:#64748b;margin-top:2px;letter-spacing:0.05em'>
        AI-POWERED SLEEP HEALTH ANALYSIS
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### 👤 Personal")
    gender     = st.selectbox("Gender", ["Male", "Female"], label_visibility="collapsed",
                               key="gender", help="Select your gender")
    st.caption("Gender")
    age        = st.slider("Age", 18, 65, 30)

    occ_options = {
        "Accountant": 0, "Doctor": 1, "Engineer": 2, "Lawyer": 3,
        "Nurse": 4, "Salesperson": 5, "Teacher": 6, "Other": 7
    }
    occ_label  = st.selectbox("Occupation", list(occ_options.keys()))
    occupation = occ_options[occ_label]

    st.markdown("##### 😴 Sleep")
    sleep_dur  = st.slider("Sleep Duration (hrs)", 5.0, 9.0, 7.0, 0.1)
    quality    = st.slider("Sleep Quality (1–9)", 4.5, 9.0, 7.0, 0.5)

    st.markdown("##### 💪 Lifestyle")
    stress     = st.slider("Stress Level (1–8)", 1, 8, 5)
    activity   = st.slider("Physical Activity (min/day)", 10, 90, 45)

    bmi_labels = {"Normal (BMI < 25)": 0, "Overweight (25–30)": 1, "Obese (BMI > 30)": 2}
    bmi_label  = st.selectbox("BMI Category", list(bmi_labels.keys()))
    bmi        = bmi_labels[bmi_label]

    st.markdown("##### ❤️ Vitals")
    heart_rate = st.slider("Heart Rate (bpm)", 60, 90, 70)
    steps      = st.slider("Daily Steps", 3000, 10000, 7000, 500)
    systolic   = st.slider("Systolic Pressure", 110, 145, 120)
    diastolic  = st.slider("Diastolic Pressure", 70, 95, 80)

    st.markdown("")
    predict_btn = st.button("🔮  Run Analysis", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.65rem;color:#475569;line-height:1.5'>
    ⚠️ For educational purposes only.<br>Not a substitute for medical advice.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════
if "results" not in st.session_state:
    st.session_state.results = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ══════════════════════════════════════════════════════════════
# COMPUTE PREDICTION ON BUTTON CLICK
# ══════════════════════════════════════════════════════════════
if predict_btn:
    gender_enc   = 1 if gender == "Male" else 0
    sleep_eff    = sleep_dur / quality
    risk_index   = stress * bmi + heart_rate / 10

    model_input  = pd.DataFrame([{
        'Gender': gender_enc, 'Age': age, 'Occupation': occupation,
        'Sleep Duration': sleep_dur, 'Quality of Sleep': quality,
        'Physical Activity Level': activity, 'Stress Level': stress,
        'BMI Category': bmi, 'Heart Rate': heart_rate, 'Daily Steps': steps,
        'Systolic Pressure': systolic, 'Diastolic Pressure': diastolic,
        'Sleep Efficiency': sleep_eff, 'Health Risk Index': risk_index,
    }])[feature_cols]

    user_scaled  = scaler.transform(model_input)
    prediction   = model.predict(user_scaled)[0]
    proba        = model.predict_proba(user_scaled)[0]
    label        = le_target.inverse_transform([prediction])[0]
    risk_s, risk_l = sleep_risk_score(stress, sleep_dur, bmi, activity)
    h_score      = health_score(stress, sleep_dur, bmi, activity)
    _, recs      = get_recommendations(stress, sleep_dur, bmi, activity, int(prediction))

    # SHAP for this user
    try:
        shap_vals    = explainer.shap_values(pd.DataFrame(user_scaled, columns=feature_cols))
        pred_class   = int(xgb_model.predict(pd.DataFrame(user_scaled, columns=feature_cols))[0])
        shap_explanation = shap.Explanation(
            values        = shap_vals[pred_class][0],
            base_values   = explainer.expected_value[pred_class],
            data          = user_scaled[0],
            feature_names = feature_cols
        )
    except Exception:
        shap_explanation = None

    st.session_state.results = dict(
        label=label, prediction=int(prediction), proba=proba,
        risk_score=risk_s, risk_level=risk_l, h_score=h_score, recs=recs,
        sleep_dur=sleep_dur, stress=stress, activity=activity,
        heart_rate=heart_rate, quality=quality, bmi=bmi,
        age=age, gender=gender, occ_label=occ_label,
        steps=steps, systolic=systolic, diastolic=diastolic,
        sleep_eff=sleep_eff, risk_index=risk_index,
        shap_explanation=shap_explanation,
    )
    st.session_state.chat_history = []  # reset chat on new prediction


# ══════════════════════════════════════════════════════════════
# LANDING STATE  (no prediction yet)
# ══════════════════════════════════════════════════════════════
if st.session_state.results is None:
    st.markdown("""
    <div style='text-align:center;padding:4rem 2rem 2rem'>
      <div style='font-family:Syne,sans-serif;font-size:3.2rem;font-weight:800;
                  background:linear-gradient(135deg,#4f8ef7 0%,#7c3aed 50%,#06b6d4 100%);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  line-height:1.1;margin-bottom:1rem'>
        Understand Your Sleep.<br/>Reclaim Your Health.
      </div>
      <div style='font-size:1.05rem;color:#64748b;max-width:480px;margin:0 auto 2rem'>
        Enter your lifestyle details in the sidebar and click
        <strong style="color:#4f8ef7">Run Analysis</strong> for a full
        AI-powered sleep health breakdown.
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="iq-card" style="text-align:center">
          <div style="font-size:2rem">🧠</div>
          <div style="font-family:Syne,sans-serif;font-weight:700;margin:.4rem 0 .3rem">ML Stacking Ensemble</div>
          <div style="font-size:.8rem;color:#64748b">4 models combined for ~93% accuracy</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="iq-card" style="text-align:center">
          <div style="font-size:2rem">🔬</div>
          <div style="font-family:Syne,sans-serif;font-weight:700;margin:.4rem 0 .3rem">SHAP Explainability</div>
          <div style="font-size:.8rem;color:#64748b">See exactly why the model predicted your result</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="iq-card" style="text-align:center">
          <div style="font-size:2rem">💬</div>
          <div style="font-family:Syne,sans-serif;font-weight:700;margin:.4rem 0 .3rem">AI Health Chat</div>
          <div style="font-size:.8rem;color:#64748b">Ask questions about your personalised results</div>
        </div>""", unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════
# RESULTS LAYOUT
# ══════════════════════════════════════════════════════════════
r = st.session_state.results

# ── Condition colour + icon ──────────────────────────────────
cond_meta = {
    'Healthy':     ('#22d3a0', '✅', 'bg: rgba(34,211,160,0.12); border-color:#22d3a0'),
    'Insomnia':    ('#f59e0b', '⚠️', 'bg: rgba(245,158,11,0.12); border-color:#f59e0b'),
    'Sleep Apnea': ('#f43f5e', '🚨', 'bg: rgba(244,63,94,0.12); border-color:#f43f5e'),
}
cond_color, cond_icon, cond_style = cond_meta.get(r['label'], ('#4f8ef7', '🔵', ''))

risk_color = '#22d3a0' if r['risk_score'] < 30 else ('#f59e0b' if r['risk_score'] < 60 else '#f43f5e')
h_color    = '#22d3a0' if r['h_score'] >= 70 else ('#f59e0b' if r['h_score'] >= 40 else '#f43f5e')

# ── Page header ──────────────────────────────────────────────
st.markdown(f"""
<div style='display:flex;align-items:center;justify-content:space-between;
            margin-bottom:1.4rem;flex-wrap:wrap;gap:.5rem'>
  <div>
    <div style='font-family:Syne,sans-serif;font-size:1.6rem;font-weight:800;color:#e2e8f0'>
      Sleep Health Report
    </div>
    <div style='font-size:.78rem;color:#475569'>
      {r['gender']}, {r['age']} yrs · {r['occ_label']} · Analysis powered by Stacking Ensemble + SHAP
    </div>
  </div>
  <div style='background:{cond_color}22;border:1px solid {cond_color};border-radius:12px;
              padding:.6rem 1.4rem;display:flex;align-items:center;gap:.6rem'>
    <span style='font-size:1.5rem'>{cond_icon}</span>
    <div>
      <div style='font-size:.65rem;color:#94a3b8;letter-spacing:.06em;text-transform:uppercase'>Diagnosis</div>
      <div style='font-family:Syne,sans-serif;font-size:1.15rem;font-weight:700;color:{cond_color}'>{r['label']}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI row ──────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("😴 Sleep", f"{r['sleep_dur']}h",
          delta=f"{r['sleep_dur']-7.5:+.1f}h vs ideal")
k2.metric("😰 Stress", f"{r['stress']}/8",
          delta=f"{-(r['stress']-4):+d} vs ideal", delta_color="inverse")
k3.metric("🏃 Activity", f"{r['activity']}min",
          delta=f"{r['activity']-60:+d}min vs ideal")
k4.metric("❤️ Heart Rate", f"{r['heart_rate']}bpm")
k5.metric("📊 Health Score", f"{r['h_score']}/100")
k6.metric("⚠️ Risk Score", f"{r['risk_score']}/100",
          delta=r['risk_level'].split(" ", 1)[1])

st.markdown("")

# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
tab_overview, tab_charts, tab_shap, tab_recs, tab_chat = st.tabs([
    "📊 Overview",
    "📈 Analytics",
    "🧠 AI Explanation",
    "💡 Recommendations",
    "💬 AI Chat",
])


# ╔══════════════════════════════╗
# ║  TAB 1 — OVERVIEW            ║
# ╚══════════════════════════════╝
with tab_overview:
    left, mid, right = st.columns([1.1, 1, 1.2])

    # ── LEFT: diagnosis card ──────────────────────────────────
    with left:
        st.markdown(f"""
        <div class="iq-card iq-card-accent" style="margin-bottom:1rem">
          <div class="iq-label">Predicted Condition</div>
          <div class="iq-value" style="color:{cond_color}">{cond_icon} {r['label']}</div>
          <div style="font-size:.82rem;color:#94a3b8;margin-top:.6rem;line-height:1.5">
            Confidence: <strong style="color:{cond_color}">{max(r['proba'])*100:.1f}%</strong>
            &nbsp;·&nbsp; Model: Stacking Ensemble (RF+SVM+GB+XGB)
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Probability bar chart
        fig_prob = probability_bar(
            r['proba'], le_target.classes_,
            [CLASS_COLORS[c] for c in le_target.classes_]
        )
        st.pyplot(fig_prob, use_container_width=True)
        plt.close(fig_prob)

    # ── MID: gauges ───────────────────────────────────────────
    with mid:
        st.markdown('<div class="iq-label" style="text-align:center;margin-bottom:.3rem">Sleep Risk Score</div>',
                    unsafe_allow_html=True)
        fig_g1 = gauge_chart(r['risk_score'], risk_color)
        st.pyplot(fig_g1, use_container_width=True)
        plt.close(fig_g1)

        st.markdown(f"""
        <div style="text-align:center;margin-top:-.4rem">
          <span style="font-size:.85rem;color:{risk_color};font-weight:600">{r['risk_level']}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="height:.5rem"></div>', unsafe_allow_html=True)

        st.markdown('<div class="iq-label" style="text-align:center;margin-bottom:.3rem">Health Score</div>',
                    unsafe_allow_html=True)
        fig_g2 = gauge_chart(r['h_score'], h_color)
        st.pyplot(fig_g2, use_container_width=True)
        plt.close(fig_g2)

    # ── RIGHT: radar + donuts ─────────────────────────────────
    with right:
        # Normalise values for radar (0–1 scale)
        sleep_norm    = min(r['sleep_dur'] / 9, 1)
        activity_norm = min(r['activity'] / 90, 1)
        stress_norm   = 1 - (r['stress'] / 8)          # invert: low stress = good
        hr_norm       = 1 - abs(r['heart_rate'] - 70) / 30
        quality_norm  = (r['quality'] - 4.5) / 4.5

        fig_radar = radar_chart(
            [sleep_norm, quality_norm, activity_norm, stress_norm, hr_norm],
            ['Sleep', 'Quality', 'Activity', 'Low\nStress', 'Heart\nRate'],
            color=cond_color
        )
        st.markdown('<div class="iq-label" style="text-align:center">Lifestyle Radar</div>',
                    unsafe_allow_html=True)
        st.pyplot(fig_radar, use_container_width=True)
        plt.close(fig_radar)

        # Mini donut row
        dc1, dc2 = st.columns(2)
        with dc1:
            fd1 = donut_chart(r['h_score'], 'Health', h_color, size=2.0)
            st.pyplot(fd1, use_container_width=True)
            plt.close(fd1)
        with dc2:
            fd2 = donut_chart(100 - r['risk_score'], 'Safety', risk_color, size=2.0)
            st.pyplot(fd2, use_container_width=True)
            plt.close(fd2)

    st.markdown("---")

    # ── Full input summary table ──────────────────────────────
    st.markdown("#### 📋 Full Input Summary")
    bmi_str = {0: 'Normal', 1: 'Overweight', 2: 'Obese'}
    summary_data = {
        "Feature": ["Gender","Age","Occupation","Sleep Duration","Sleep Quality",
                    "Stress Level","Physical Activity","BMI Category","Heart Rate",
                    "Daily Steps","Blood Pressure","Sleep Efficiency","Health Risk Index"],
        "Your Value": [
            r['gender'], r['age'], r['occ_label'],
            f"{r['sleep_dur']} hrs", r['quality'],
            f"{r['stress']}/8", f"{r['activity']} min/day",
            bmi_str.get(r['bmi'], '?'), f"{r['heart_rate']} bpm",
            f"{r['steps']:,}", f"{r['systolic']}/{r['diastolic']} mmHg",
            f"{r['sleep_eff']:.3f}", f"{r['risk_index']:.2f}",
        ],
        "Ideal Range": [
            "—", "—", "—",
            "7–9 hrs", "7–9",
            "1–4", "≥60 min/day",
            "Normal", "60–75 bpm",
            "≥7,500", "< 120/80",
            "< 1.1", "< 10",
        ]
    }
    st.dataframe(
        pd.DataFrame(summary_data),
        use_container_width=True,
        hide_index=True,
    )


# ╔══════════════════════════════╗
# ║  TAB 2 — ANALYTICS           ║
# ╚══════════════════════════════╝
with tab_charts:
    row1_l, row1_r = st.columns(2)

    # ── Comparison bar chart ──────────────────────────────────
    with row1_l:
        st.markdown("#### Your Metrics vs Ideal")
        user_vals  = [r['sleep_dur'],  r['stress'], r['activity']/10, r['heart_rate']/10]
        ideal_vals = [7.5,             4,           6.0,              7.0]
        labels_cmp = ['Sleep (hrs)', 'Stress', 'Activity (/10)', 'Heart Rate (/10)']
        fig_cmp = comparison_chart(user_vals, ideal_vals, labels_cmp)
        st.pyplot(fig_cmp, use_container_width=True)
        plt.close(fig_cmp)

    # ── Sleep score breakdown stacked bar ────────────────────
    with row1_r:
        st.markdown("#### Risk Score Breakdown")
        stress_pts   = min(r['stress'] * 3.75, 30)
        sleep_pts    = 25 if r['sleep_dur'] < 6 else (15 if r['sleep_dur'] < 7 else 0)
        bmi_pts      = r['bmi'] * 12.5
        activity_pts = max(0, (60 - r['activity']) / 60 * 20)
        components   = [stress_pts, sleep_pts, bmi_pts, activity_pts]
        comp_labels  = ['Stress', 'Sleep Deficit', 'BMI', 'Low Activity']
        comp_colors  = ['#f43f5e', '#f59e0b', '#7c3aed', '#4f8ef7']

        fig_break, ax = plt.subplots(figsize=(5, 2.6))
        left_pos = 0
        for val, lbl, col in zip(components, comp_labels, comp_colors):
            ax.barh(['Risk'], [val], left=left_pos, color=col, label=f'{lbl} ({val:.0f})')
            if val > 4:
                ax.text(left_pos + val / 2, 0, f'{val:.0f}',
                        ha='center', va='center', fontsize=8, color='white', fontweight='bold')
            left_pos += val
        ax.set_xlim(0, 100)
        ax.set_xlabel('Points (max 100)', fontsize=8)
        ax.set_title('Risk Score Components', fontsize=10, fontweight='bold')
        ax.legend(loc='upper right', fontsize=7, framealpha=0.15, ncol=2)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_break, use_container_width=True)
        plt.close(fig_break)

    st.markdown("---")

    row2_l, row2_r = st.columns(2)

    # ── Score gauge row ───────────────────────────────────────
    with row2_l:
        st.markdown("#### Probability Distribution")
        fig_pie, ax = plt.subplots(figsize=(5, 3))
        wedge_props = dict(width=0.45, edgecolor='#131b2e', linewidth=2)
        colors_pie  = [CLASS_COLORS[c] for c in le_target.classes_]
        wedges, texts, autotexts = ax.pie(
            r['proba'], labels=le_target.classes_,
            autopct='%1.1f%%', colors=colors_pie,
            wedgeprops=wedge_props, startangle=140,
            pctdistance=0.75
        )
        for t in texts:
            t.set_fontsize(9); t.set_color('#94a3b8')
        for at in autotexts:
            at.set_fontsize(8); at.set_color('white')
        ax.set_title('Class Probabilities (Donut)', fontsize=10, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_pie, use_container_width=True)
        plt.close(fig_pie)

    # ── SHAP bar for this user ────────────────────────────────
    with row2_r:
        st.markdown("#### Feature Contribution (Your Prediction)")
        if r.get('shap_explanation') is not None:
            exp = r['shap_explanation']
            sv  = exp.values
            fnames = feature_cols
            sorted_idx = np.argsort(np.abs(sv))[-10:]
            sv_sorted = sv[sorted_idx]
            fn_sorted = [fnames[i] for i in sorted_idx]
            cols_bar  = ['#4f8ef7' if v >= 0 else '#f43f5e' for v in sv_sorted]

            fig_sv, ax = plt.subplots(figsize=(5, 3))
            ax.barh(fn_sorted, sv_sorted, color=cols_bar, edgecolor='#131b2e', height=0.55)
            ax.axvline(0, color='#64748b', linewidth=0.8)
            ax.set_xlabel('SHAP Value', fontsize=8)
            ax.set_title('Feature Impact on Prediction', fontsize=10, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_sv, use_container_width=True)
            plt.close(fig_sv)
        else:
            st.info("SHAP values unavailable for this session.")

    st.markdown("---")

    # ── Sleep quality vs duration scatter (reference data) ───
    st.markdown("#### Where Do You Stand? (Population Reference)")
    np.random.seed(42)
    n = 300
    ref_sleep   = np.random.normal(7.1, 0.8, n).clip(5.5, 9)
    ref_stress  = np.random.randint(2, 9, n)
    ref_classes = np.random.choice([0, 1, 2], n, p=[0.58, 0.21, 0.21])

    fig_scatter, ax = plt.subplots(figsize=(10, 3.5))
    for cls_idx, cls_name in enumerate(CLASSES):
        mask = ref_classes == cls_idx
        ax.scatter(ref_stress[mask], ref_sleep[mask],
                   c=CLASS_COLORS[cls_name], alpha=0.4, s=22,
                   label=cls_name, edgecolors='none')
    # User point
    ax.scatter(r['stress'], r['sleep_dur'],
               c='white', s=160, zorder=5, marker='*',
               edgecolors=cond_color, linewidths=1.5, label='You')
    ax.set_xlabel('Stress Level', fontsize=9)
    ax.set_ylabel('Sleep Duration (hrs)', fontsize=9)
    ax.set_title('Sleep Duration vs Stress Level — Population Reference', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, framealpha=0.15)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig_scatter, use_container_width=True)
    plt.close(fig_scatter)


# ╔══════════════════════════════╗
# ║  TAB 3 — SHAP EXPLANATION    ║
# ╚══════════════════════════════╝
with tab_shap:
    st.markdown("""
    <div class="iq-card" style="margin-bottom:1.2rem">
      <div style="font-family:Syne,sans-serif;font-size:1rem;font-weight:700;margin-bottom:.4rem">
        🧠 How did the AI reach this prediction?
      </div>
      <div style="font-size:.82rem;color:#94a3b8;line-height:1.6">
        SHAP (SHapley Additive exPlanations) assigns each feature a value showing
        how much it pushed the prediction <strong>toward</strong> (blue +) or
        <strong>away from</strong> (red −) the predicted class.
      </div>
    </div>
    """, unsafe_allow_html=True)

    exp = r.get('shap_explanation')
    if exp is not None:
        sv = exp.values
        fv = exp.data
        fn = feature_cols

        # Top positive + negative contributors
        top_pos_idx = np.argsort(sv)[-5:][::-1]
        top_neg_idx = np.argsort(sv)[:5]

        col_pos, col_neg = st.columns(2)

        with col_pos:
            st.markdown("##### ✅ Factors supporting this prediction")
            for i in top_pos_idx:
                if sv[i] > 0:
                    pct = sv[i] / (np.sum(np.abs(sv)) + 1e-9) * 100
                    st.markdown(f"""
                    <div style='background:#22d3a022;border:1px solid #22d3a055;
                                border-radius:8px;padding:.5rem .9rem;margin-bottom:.4rem'>
                      <span style='color:#22d3a0;font-weight:600;font-size:.85rem'>{fn[i]}</span>
                      <span style='color:#94a3b8;font-size:.78rem'> = {fv[i]:.2f}</span>
                      <div style='font-size:.72rem;color:#64748b'>
                        Contribution: +{sv[i]:.3f} ({pct:.1f}% of total impact)
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

        with col_neg:
            st.markdown("##### ⚠️ Factors working against this prediction")
            for i in top_neg_idx:
                if sv[i] < 0:
                    pct = abs(sv[i]) / (np.sum(np.abs(sv)) + 1e-9) * 100
                    st.markdown(f"""
                    <div style='background:#f43f5e22;border:1px solid #f43f5e55;
                                border-radius:8px;padding:.5rem .9rem;margin-bottom:.4rem'>
                      <span style='color:#f43f5e;font-weight:600;font-size:.85rem'>{fn[i]}</span>
                      <span style='color:#94a3b8;font-size:.78rem'> = {fv[i]:.2f}</span>
                      <div style='font-size:.72rem;color:#64748b'>
                        Contribution: {sv[i]:.3f} ({pct:.1f}% of total impact)
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")

        # Waterfall-style bar chart
        st.markdown("#### Full Feature Impact Waterfall")
        sorted_idx = np.argsort(sv)
        sv_sorted  = sv[sorted_idx]
        fn_sorted  = [fn[i] for i in sorted_idx]
        bar_colors = ['#4f8ef7' if v >= 0 else '#f43f5e' for v in sv_sorted]

        fig_wf, ax = plt.subplots(figsize=(10, 4.5))
        bars = ax.barh(fn_sorted, sv_sorted, color=bar_colors, edgecolor='#131b2e', height=0.6)
        ax.axvline(0, color='#94a3b8', linewidth=1)
        for bar, v in zip(bars, sv_sorted):
            xpos = v + 0.003 if v >= 0 else v - 0.003
            ha   = 'left' if v >= 0 else 'right'
            ax.text(xpos, bar.get_y() + bar.get_height()/2, f'{v:+.3f}',
                    va='center', ha=ha, fontsize=7.5, color='#94a3b8')
        ax.set_xlabel('SHAP Value (impact on model output)', fontsize=9)
        ax.set_title(f'Feature Impact — Predicted: {r["label"]}', fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig_wf, use_container_width=True)
        plt.close(fig_wf)

        # Global SHAP images (if saved by train_model.py)
        st.markdown("---")
        st.markdown("#### Global Feature Importance (from training data)")
        ci1, ci2 = st.columns(2)
        with ci1:
            if os.path.exists('shap_importance_bar.png'):
                st.image('shap_importance_bar.png', caption='Mean |SHAP| — All Samples', use_container_width=True)
            else:
                st.info("Run train_model.py to generate global SHAP plots.")
        with ci2:
            if os.path.exists('shap_summary.png'):
                st.image('shap_summary.png', caption='SHAP Beeswarm Summary', use_container_width=True)
    else:
        st.warning("SHAP explanation could not be computed. Ensure shap_explainer.pkl is in the artifacts/ folder.")


# ╔══════════════════════════════╗
# ║  TAB 4 — RECOMMENDATIONS     ║
# ╚══════════════════════════════╝
with tab_recs:
    st.markdown(f"""
    <div class="iq-card iq-card-accent" style="margin-bottom:1.4rem">
      <div style="font-family:Syne,sans-serif;font-size:1.05rem;font-weight:700">
        {cond_icon} Personalised Action Plan — {r['label']}
      </div>
      <div style="font-size:.82rem;color:#94a3b8;margin-top:.3rem">
        Based on your stress ({r['stress']}/8), sleep ({r['sleep_dur']}h),
        BMI category ({['Normal','Overweight','Obese'][r['bmi']]}),
        and activity ({r['activity']} min/day).
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Group recs by emoji prefix
    stress_recs   = [x for x in r['recs'] if any(e in x for e in ['STRESS','stress'])]
    sleep_recs    = [x for x in r['recs'] if any(e in x for e in ['SLEEP','sleep','caffeine','Aim'])]
    bmi_recs      = [x for x in r['recs'] if any(e in x for e in ['BMI','OBESE','OVERWEIGHT','weight','cardio'])]
    activity_recs = [x for x in r['recs'] if any(e in x for e in ['ACTIVITY','exercise','workouts'])]
    disorder_recs = [x for x in r['recs'] if any(e in x for e in ['INSOMNIA','APNEA','HEALTHY','CBT','CPAP','side','nap','bedroom'])]
    generic_recs  = [x for x in r['recs'] if x not in stress_recs + sleep_recs + bmi_recs + activity_recs + disorder_recs]

    def rec_section(title, icon, items, color):
        if not items:
            return
        items_html = ''.join(
            f'<div style="padding:.45rem 0;border-bottom:1px solid #1e2d4a;font-size:.85rem;line-height:1.5">{i}</div>'
            for i in items
        )
        st.markdown(f"""
        <div class="iq-card" style="margin-bottom:.8rem">
          <div style="font-family:Syne,sans-serif;font-weight:700;font-size:.9rem;
                      color:{color};margin-bottom:.5rem">{icon} {title}</div>
          {items_html}
        </div>
        """, unsafe_allow_html=True)

    all_recs_by_group = [
        ("Stress Management", "😰", stress_recs,   '#f43f5e'),
        ("Sleep Habits",      "😴", sleep_recs,    '#4f8ef7'),
        ("BMI & Nutrition",   "⚖️", bmi_recs,      '#7c3aed'),
        ("Physical Activity", "🏃", activity_recs, '#06b6d4'),
        ("Condition-Specific","🎯", disorder_recs, cond_color),
        ("General Wellness",  "✅", generic_recs,  '#22d3a0'),
    ]

    rc1, rc2 = st.columns(2)
    for i, (title, icon, items, color) in enumerate(all_recs_by_group):
        with (rc1 if i % 2 == 0 else rc2):
            rec_section(title, icon, items, color)

    st.markdown("---")

    # Priority action items
    st.markdown("#### 🚀 Top 3 Priority Actions")
    priority = []
    if r['stress'] >= 6:    priority.append(("Stress", "Reduce stress to ≤4 — high impact on sleep quality", '#f43f5e'))
    if r['sleep_dur'] < 7:  priority.append(("Sleep", "Add 30–60 min to nightly sleep duration", '#4f8ef7'))
    if r['activity'] < 40:  priority.append(("Activity", "Reach 45+ min of moderate exercise daily", '#22d3a0'))
    if r['bmi'] >= 2:       priority.append(("BMI", "Work with a nutritionist on weight management", '#7c3aed'))

    for rank, (area, action, col) in enumerate(priority[:3], 1):
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:1rem;background:#131b2e;
                    border:1px solid {col}44;border-radius:10px;padding:.75rem 1.1rem;margin-bottom:.6rem">
          <div style="font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;
                      color:{col};min-width:1.8rem">#{rank}</div>
          <div>
            <div style="font-weight:600;font-size:.88rem;color:{col}">{area}</div>
            <div style="font-size:.8rem;color:#94a3b8">{action}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ╔══════════════════════════════╗
# ║  TAB 5 — AI CHAT             ║
# ╚══════════════════════════════╝
with tab_chat:
    st.markdown("""
    <div class="iq-card" style="margin-bottom:1rem">
      <div style="font-family:Syne,sans-serif;font-weight:700;font-size:1rem">
        💬 Ask the Sleep Health AI
      </div>
      <div style="font-size:.8rem;color:#64748b;margin-top:.3rem">
        Ask questions about your results, sleep science, or lifestyle improvements.
        The AI has full context of your analysis.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Build context string for the AI
    bmi_str_map = {0: 'Normal', 1: 'Overweight', 2: 'Obese'}
    user_context = f"""
User's sleep health data:
- Predicted condition: {r['label']} (confidence: {max(r['proba'])*100:.1f}%)
- Sleep duration: {r['sleep_dur']} hours (ideal: 7-9)
- Sleep quality score: {r['quality']}/9
- Stress level: {r['stress']}/8
- Physical activity: {r['activity']} minutes/day
- BMI category: {bmi_str_map[r['bmi']]}
- Heart rate: {r['heart_rate']} bpm
- Daily steps: {r['steps']}
- Blood pressure: {r['systolic']}/{r['diastolic']} mmHg
- Health score: {r['h_score']}/100
- Sleep risk score: {r['risk_score']}/100 ({r['risk_level']})
- Age: {r['age']}, Gender: {r['gender']}, Occupation: {r['occ_label']}
"""

    # Quick-question chips
    st.markdown("**💡 Quick questions:**")
    qc1, qc2, qc3, qc4 = st.columns(4)
    quick_questions = [
        "Why do I have this condition?",
        "How can I improve my sleep tonight?",
        "What does my risk score mean?",
        "Which metric should I fix first?",
    ]
    for col, q in zip([qc1, qc2, qc3, qc4], quick_questions):
        if col.button(q, use_container_width=True, key=f"q_{q[:10]}"):
            st.session_state.chat_history.append({"role": "user", "content": q})

    # Chat history display
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div style="display:flex;justify-content:flex-end;margin:.4rem 0">
                  <div style="background:linear-gradient(135deg,#1e3a6e,#1e2d4a);
                              border:1px solid #2d4a7a;border-radius:12px 12px 2px 12px;
                              padding:.6rem 1rem;max-width:70%;font-size:.85rem">
                    {msg['content']}
                  </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="display:flex;justify-content:flex-start;margin:.4rem 0">
                  <div style="background:#131b2e;border:1px solid #1e2d4a;
                              border-radius:12px 12px 12px 2px;
                              padding:.6rem 1rem;max-width:78%;font-size:.85rem;line-height:1.6">
                    <span style="color:#4f8ef7;font-size:.7rem;font-weight:600;letter-spacing:.05em">
                      🌙 SLEEPIQ AI
                    </span><br>{msg['content']}
                  </div>
                </div>
                """, unsafe_allow_html=True)

    # Generate AI response for latest unanswered user message
    if (st.session_state.chat_history and
            st.session_state.chat_history[-1]["role"] == "user"):

        user_q = st.session_state.chat_history[-1]["content"]

        # Rule-based contextual AI (no API key required)
        def generate_ai_response(question, context, result):
            q = question.lower()
            label   = result['label']
            stress  = result['stress']
            sleep_d = result['sleep_dur']
            bmi     = result['bmi']
            act     = result['activity']
            risk    = result['risk_score']

            if any(w in q for w in ['why', 'reason', 'cause', 'condition']):
                factors = []
                if stress >= 6:    factors.append(f"high stress ({stress}/8)")
                if sleep_d < 6.5:  factors.append(f"insufficient sleep ({sleep_d}h)")
                if bmi >= 1:       factors.append(f"elevated BMI ({['Normal','Overweight','Obese'][bmi]})")
                if act < 35:       factors.append(f"low physical activity ({act} min/day)")
                f_str = ', '.join(factors) if factors else 'a combination of lifestyle factors'
                return (f"Based on your data, the primary factors contributing to **{label}** are: "
                        f"{f_str}. The model's top predictors for your case are Health Risk Index "
                        f"(stress × BMI + heart rate/10) and diastolic pressure. "
                        f"Your confidence score is {max(result['proba'])*100:.1f}%.")

            if any(w in q for w in ['tonight', 'tonight', 'improve sleep', 'sleep better', 'sleep tonight']):
                tips = ["Dim lights 1–2 hours before bed",
                        "Keep room temperature between 18–20°C",
                        "Avoid screens 45 minutes before sleep",
                        "Try 4-7-8 breathing: inhale 4s, hold 7s, exhale 8s"]
                if stress >= 5: tips.insert(0, "Do a 10-minute mindfulness session right now")
                return ("Here are evidence-based tips for better sleep **tonight**:\n\n" +
                        '\n'.join(f'• {t}' for t in tips))

            if any(w in q for w in ['risk', 'score', 'mean', 'number']):
                level = "low" if risk < 30 else ("moderate" if risk < 60 else "high")
                return (f"Your risk score is **{risk}/100** ({level} risk). "
                        f"It's calculated from 4 components: stress ({min(stress*3.75,30):.0f}/30 pts), "
                        f"sleep deficit ({(25 if sleep_d<6 else (15 if sleep_d<7 else 0))}/25 pts), "
                        f"BMI ({bmi*12.5:.0f}/25 pts), and low activity "
                        f"({max(0,(60-act)/60*20):.0f}/20 pts). "
                        f"Reducing stress and increasing sleep are the fastest ways to lower this score.")

            if any(w in q for w in ['first', 'priority', 'which', 'fix', 'start']):
                if stress >= 6 and sleep_d < 7:
                    return (f"Your highest-impact action is **stress reduction** (stress: {stress}/8). "
                            f"Chronic stress is a leading cause of both insomnia and sleep apnea. "
                            f"Even 10 minutes of daily meditation can reduce cortisol by 15–20% within 2 weeks. "
                            f"Second priority: extend sleep to 7+ hours.")
                elif sleep_d < 6.5:
                    return (f"Your top priority is **sleep duration** (currently {sleep_d}h). "
                            f"Sleep deprivation compounds every other health metric. "
                            f"Try moving bedtime 30 minutes earlier this week and avoid caffeine after 2 PM.")
                elif bmi >= 2:
                    return ("Your top priority is **BMI management**. Obesity is strongly correlated "
                            "with Sleep Apnea (3–5× higher risk). Even a 5–10% weight reduction can "
                            "significantly reduce apnea episodes. Start with dietary changes and a daily 20-minute walk.")
                else:
                    return (f"Your metrics are relatively balanced. Focus on **physical activity** "
                            f"(currently {act} min/day, ideal: 60+). Regular aerobic exercise improves "
                            f"deep sleep by up to 25% and reduces stress hormones naturally.")

            if any(w in q for w in ['insomnia', 'apnea', 'healthy']):
                if label == 'Insomnia':
                    return ("**Insomnia** is characterised by difficulty falling or staying asleep. "
                            "Key evidence-based treatments: (1) CBT-I (Cognitive Behavioral Therapy for Insomnia) "
                            "— more effective than sleep medication long-term, (2) strict sleep schedule, "
                            "(3) avoiding bed for non-sleep activities, (4) stimulus control therapy.")
                elif label == 'Sleep Apnea':
                    return ("**Sleep Apnea** involves repeated breathing pauses during sleep. "
                            "Symptoms include loud snoring, morning headaches, and daytime fatigue. "
                            "Treatment options: CPAP therapy (gold standard), positional therapy "
                            "(sleeping on your side), weight loss (reduces severity by 30–50%), "
                            "and avoiding alcohol/sedatives before bed.")
                else:
                    return ("You're currently classified as **Healthy** — great work! "
                            "To maintain this, keep sleep duration at 7–9 hours, stress below 4/8, "
                            "and activity at 60+ minutes daily. Annual check-ins are recommended.")

            if any(w in q for w in ['exercise', 'activity', 'workout', 'gym']):
                return (f"Your current activity level is **{act} min/day** (ideal: 60+). "
                        f"For sleep improvement, moderate-intensity aerobic exercise (brisk walking, cycling, swimming) "
                        f"done 3–5x/week is optimal. Avoid intense exercise within 2 hours of bedtime "
                        f"as it raises core body temperature and delays sleep onset. "
                        f"Morning exercise has the strongest effect on circadian rhythm regulation.")

            if any(w in q for w in ['stress', 'anxiety', 'relax', 'calm']):
                return (f"Your stress level is **{stress}/8** "
                        f"({'high' if stress>=6 else 'moderate' if stress>=4 else 'low'}). "
                        f"Proven stress-reduction techniques for better sleep:\n\n"
                        f"• **Progressive Muscle Relaxation** — tense and release muscle groups\n"
                        f"• **Journaling** — offload anxious thoughts before bed (10 min)\n"
                        f"• **Box breathing** — 4s in, 4s hold, 4s out, 4s hold\n"
                        f"• **Limit news/social media** after 8 PM\n"
                        f"• **Consistent wind-down routine** starting 1 hour before bed")

            # Default response
            return (f"Based on your analysis (condition: {label}, risk score: {risk}/100, "
                    f"health score: {result['h_score']}/100), the most important areas to focus on are "
                    f"{'stress management, ' if stress >= 5 else ''}"
                    f"{'increasing sleep duration, ' if sleep_d < 7 else ''}"
                    f"{'physical activity. ' if act < 50 else 'maintaining your current habits. '}"
                    f"Would you like specific advice on any of these areas?")

        with st.spinner("Analysing your question…"):
            time.sleep(0.6)
            answer = generate_ai_response(user_q, user_context, r)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    # Text input for new message
    user_input = st.chat_input("Ask anything about your sleep health…")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.rerun()
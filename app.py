import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
from dotenv import load_dotenv
from groq import Groq
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
from groq import Groq

# 👇 Add here
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ── Page config — must be first Streamlit call ────────────────────
st.set_page_config(
    page_title="StressLens",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

GROQ_KEY = os.getenv("GROQ_API_KEY", "")

if GROQ_KEY:
    groq_client = Groq(api_key=GROQ_KEY)
else:
    groq_client = None

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

@st.cache_resource
def load_artifacts():
    ARTIFACT_PATH = os.path.join(BASE_PATH, "data")

    model = joblib.load(os.path.join(ARTIFACT_PATH, "best_model.pkl"))
    scaler = joblib.load(os.path.join(ARTIFACT_PATH, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(ARTIFACT_PATH, "feature_names.pkl"))

    with open(os.path.join(ARTIFACT_PATH, "thresholds.json")) as f:
        thresholds = json.load(f)

    metrics_df = pd.read_csv(
        os.path.join(ARTIFACT_PATH, "model_metrics.csv")
    )

    return model, scaler, feature_names, thresholds, metrics_df

model, scaler, feature_names, thresholds, metrics_df = load_artifacts()

CHEST_HZ    = 700
WINDOW_SIZE = CHEST_HZ * 10
STEP_SIZE   = CHEST_HZ * 5

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main { background-color: #0f1117; }

    .stApp {
        background: linear-gradient(135deg, #0f1117 0%, #1a1d2e 100%);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #16192a;
        border-right: 1px solid #2a2d3e;
    }

    /* Cards */
    .metric-card {
        background: #1e2235;
        border: 1px solid #2a2d3e;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 12px;
    }

    .metric-card h3 {
        color: #7c8db0;
        font-size: 12px;
        font-weight: 500;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin: 0 0 8px 0;
    }

    .metric-card .value {
        font-size: 36px;
        font-weight: 600;
        margin: 0;
        line-height: 1;
    }

    .low-stress    { color: #4ade80; }
    .moderate-stress { color: #fb923c; }
    .high-stress   { color: #f87171; }

    /* Advice box */
    .advice-box {
        background: #1e2235;
        border-left: 3px solid #6366f1;
        border-radius: 0 12px 12px 0;
        padding: 20px 24px;
        margin-top: 16px;
        color: #c9d1e9;
        font-size: 15px;
        line-height: 1.75;
    }

    .advice-box h4 {
        color: #818cf8;
        font-size: 13px;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin: 0 0 12px 0;
    }

    /* Upload area */
    .upload-hint {
        background: #1e2235;
        border: 1px dashed #2a2d3e;
        border-radius: 12px;
        padding: 32px;
        text-align: center;
        color: #7c8db0;
        font-size: 14px;
    }

    /* Header */
    .app-header {
        padding: 8px 0 24px 0;
        border-bottom: 1px solid #2a2d3e;
        margin-bottom: 28px;
    }

    .app-header h1 {
        font-size: 28px;
        font-weight: 600;
        color: #e2e8f0;
        margin: 0;
    }

    .app-header p {
        color: #7c8db0;
        font-size: 14px;
        margin: 6px 0 0 0;
    }

    /* Tag pill */
    .tag {
        display: inline-block;
        background: #252840;
        color: #818cf8;
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 12px;
        margin-right: 6px;
        border: 1px solid #3a3d5c;
    }

    /* Streamlit overrides */
    .stButton > button {
        background: #6366f1;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        width: 100%;
        transition: background 0.2s;
    }
    .stButton > button:hover { background: #4f52d4; }

    div[data-testid="stFileUploader"] {
        background: #1e2235;
        border-radius: 12px;
        border: 1px dashed #2a2d3e;
        padding: 12px;
    }

    .stSelectbox > div { background: #1e2235; }
/* Sidebar selection mode & Headings */
section[data-testid="stSidebar"] label p {
    color: #ffffff !important; /* Pure white for visibility */
    font-size: 16px !important;
    font-weight: 600 !important;
}

section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}
/* Tab Headings, AI Recommendations, and Manual Check Headings */
h1, h2, h3, h4 {
    color: #ffffff !important; /* Force all headings to white */
    opacity: 1 !important;
}

/* AI Recommendation specifically */
.advice-box h4 {
    color: #818cf8 !important; /* Bright Indigo for visibility */
    font-weight: 700 !important;
    font-size: 16px !important;
}

.advice-box {
    color: #ffffff !important;
    background: rgba(255, 255, 255, 0.08) !important;
}
/* Slider Labels & Window breakdown link */
div[data-testid="stSlider"] label p, 
div[data-testid="stSelectbox"] label p {
    color: #ffffff !important;
    font-weight: 500 !important;
}

/* "View window-by-window breakdown" text */
.streamlit-expanderHeader {
    color: #ffffff !important;
    background-color: #1e2235 !important;
}

/* Model Performance Tab Headings */
[data-baseweb="tab-list"] button p {
    color: #ffffff !important;
    font-size: 16px !important;
}
/* Expander (View window breakdown) visibility */
.streamlit-expanderHeader {
    background-color: #1e2235 !important;
    color: #ffffff !important; /* Header text white */
    border: 1px solid #2a2d3e !important;
    border-radius: 8px !important;
}

.streamlit-expanderHeader p {
    color: #ffffff !important;
}

/* Expander icon (arrow) color */
.streamlit-expanderHeader svg {
    fill: #ffffff !important;
}


</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# Feature engineering (same as NB02 — must match exactly)
# ════════════════════════════════════════════════════════════════════════════════
def signal_features(sig, name):
    sig = sig.flatten()
    return {
        f'{name}_mean'    : np.mean(sig),
        f'{name}_std'     : np.std(sig),
        f'{name}_min'     : np.min(sig),
        f'{name}_max'     : np.max(sig),
        f'{name}_range'   : np.max(sig) - np.min(sig),
        f'{name}_skew'    : skew(sig),
        f'{name}_kurtosis': kurtosis(sig),
    }

def hrv_features(ecg_window):
    ecg = ecg_window.flatten()
    ecg_norm = (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-8)
    peaks, _ = find_peaks(ecg_norm, distance=int(CHEST_HZ * 0.4), height=0.3)
    if len(peaks) < 3:
        return {'hrv_mean_rr': 0.0, 'hrv_std_rr': 0.0, 'hrv_rmssd': 0.0}
    rr    = np.diff(peaks) / CHEST_HZ * 1000
    rmssd = np.sqrt(np.mean(np.diff(rr) ** 2))
    return {'hrv_mean_rr': np.mean(rr), 'hrv_std_rr': np.std(rr),
            'hrv_rmssd': rmssd}

def extract_features(chest_window):
    feats = {}
    for sig_name in ['ECG', 'EDA', 'Temp', 'Resp', 'EMG']:
        feats.update(signal_features(chest_window[sig_name], sig_name))
    acc     = chest_window['ACC']
    acc_mag = np.linalg.norm(acc, axis=1)
    feats.update(signal_features(acc_mag, 'ACC_mag'))
    for i, axis in enumerate(['x', 'y', 'z']):
        feats.update(signal_features(acc[:, i], f'ACC_{axis}'))
    feats.update(hrv_features(chest_window['ECG']))
    return feats


def predict_from_csv(df):
    """
    Expects CSV with columns: ECG, EDA, Temp, Resp, EMG, ACC_x, ACC_y, ACC_z
    Returns list of (stress_index, category) per window.
    """
    required = ['ECG', 'EDA', 'Temp', 'Resp', 'EMG', 'ACC_x', 'ACC_y', 'ACC_z']
    for col in required:
        if col not in df.columns:
            return None, f"Missing column: {col}"

    data = df[required].values.astype(float)
    n    = len(data)

    if n < WINDOW_SIZE:
        return None, f"Need at least {WINDOW_SIZE} rows ({WINDOW_SIZE/CHEST_HZ:.0f}s at 700Hz). Got {n}."

    results = []
    for start in range(0, n - WINDOW_SIZE + 1, STEP_SIZE):
        end    = start + WINDOW_SIZE
        chunk  = data[start:end]

        chest_window = {
            'ECG' : chunk[:, 0:1],
            'EDA' : chunk[:, 1:2],
            'Temp': chunk[:, 2:3],
            'Resp': chunk[:, 3:4],
            'EMG' : chunk[:, 4:5],
            'ACC' : chunk[:, 5:8],
        }

        feats      = extract_features(chest_window)
        feat_vec   = np.array([feats.get(f, 0.0) for f in feature_names])
        feat_vec   = np.nan_to_num(feat_vec, nan=0.0, posinf=0.0, neginf=0.0)
        feat_scaled = scaler.transform(feat_vec.reshape(1, -1))

        prob        = model.predict_proba(feat_scaled)[0][1]
        score       = round(float(np.clip(prob * 100, 0, 100)), 1)

        if score <= thresholds['low']:       cat = "Low"
        elif score <= thresholds['moderate']: cat = "Moderate"
        else:                                 cat = "High"

        results.append({'score': score, 'category': cat,
                        'window_start_sec': round(start / CHEST_HZ, 1)})

    return results, None


# ════════════════════════════════════════════════════════════════════════════════
# Gemini advice
# ════════════════════════════════════════════════════════════════════════════════
def get_gemini_advice(stress_index, category, duration_sec):

    if not groq_client:
        return "⚠️ Groq API key not configured."

    prompt = f"""
You are a stress management expert.

Stress Index: {stress_index}/100
Category: {category}
Duration: {duration_sec:.0f} sec

Give:
- Meaning
- 3 quick actions
- 2 daily habits
Keep under 120 words.
"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"


# ════════════════════════════════════════════════════════════════════════════════
# Gauge chart
# ════════════════════════════════════════════════════════════════════════════════
def stress_gauge(score, category):
    color = '#4ade80' if category == 'Low' else \
            '#fb923c' if category == 'Moderate' else '#f87171'

    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = score,
        domain= {'x': [0, 1], 'y': [0, 1]},
        number= {'font': {'size': 48, 'color': color,
                          'family': 'DM Sans'}},
        gauge = {
            'axis'      : {'range': [0, 100],
                           'tickcolor': '#4a5568',
                           'tickfont' : {'color': '#7c8db0', 'size': 11}},
            'bar'       : {'color': color, 'thickness': 0.25},
            'bgcolor'   : '#1e2235',
            'bordercolor': '#2a2d3e',
            'steps'     : [
                {'range': [0, 35],  'color': '#162420'},
                {'range': [35, 65], 'color': '#1e1a14'},
                {'range': [65, 100],'color': '#1e1414'},
            ],
            'threshold' : {
                'line' : {'color': color, 'width': 3},
                'value': score
            }
        }
    ))
    fig.update_layout(
        height          = 260,
        margin          = dict(t=20, b=10, l=20, r=20),
        paper_bgcolor   = 'rgba(0,0,0,0)',
        font            = {'family': 'DM Sans'},
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════════
# Timeline chart
# ════════════════════════════════════════════════════════════════════════════════
def stress_timeline(results):
    times  = [r['window_start_sec'] for r in results]
    scores = [r['score'] for r in results]
    cats   = [r['category'] for r in results]

    color_map = {'Low': '#4ade80', 'Moderate': '#fb923c', 'High': '#f87171'}
    colors    = [color_map[c] for c in cats]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=scores,
        mode='lines+markers',
        line=dict(color='#6366f1', width=2),
        marker=dict(color=colors, size=8, line=dict(color='#1e2235', width=1)),
        hovertemplate='Time: %{x}s<br>Stress Index: %{y}<extra></extra>'
    ))
    fig.add_hrect(y0=0,   y1=35,  fillcolor='#162420', opacity=0.3, line_width=0)
    fig.add_hrect(y0=35,  y1=65,  fillcolor='#1e1a14', opacity=0.3, line_width=0)
    fig.add_hrect(y0=65,  y1=100, fillcolor='#1e1414', opacity=0.3, line_width=0)

    fig.update_layout(
        xaxis_title    = 'Time (seconds)',
        yaxis_title    = 'Stress Index',
        yaxis_range    = [0, 100],
        height         = 280,
        margin         = dict(t=10, b=40, l=50, r=20),
        paper_bgcolor  = 'rgba(0,0,0,0)',
        plot_bgcolor   = 'rgba(0,0,0,0)',
        font           = dict(color='#7c8db0', family='DM Sans'),
        xaxis          = dict(gridcolor='#2a2d3e', color='#7c8db0'),
        yaxis          = dict(gridcolor='#2a2d3e', color='#7c8db0'),
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════════
# UI Layout
# ════════════════════════════════════════════════════════════════════════════════
# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:

    mode = st.sidebar.radio(
        "Choose Input Mode",
        ["📊 ML (Sensor Data)", "📝 Manual Check"]
    )

    st.markdown("""
<div style='text-align:center; padding:20px'>
<h1 style='color:#ffffff; font-size:36px;'>🧠 StressLens AI</h1>
<p style='color:#a0aec0;'>Real-time stress detection using ML + AI insights</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='color:#7c8db0; font-size:12px; "
                "text-transform:uppercase; letter-spacing:1px;'>"
                "How to use</p>", unsafe_allow_html=True)
    st.markdown("""
    <div style='color:#c9d1e9; font-size:13px; line-height:1.8;'>
    1. Export your wearable CSV<br>
    2. Upload below<br>
    3. View stress timeline<br>
    4. Read AI-powered advice
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='color:#7c8db0; font-size:12px; "
                "text-transform:uppercase; letter-spacing:1px;'>"
                "CSV format required</p>", unsafe_allow_html=True)
    st.code("ECG, EDA, Temp, Resp,\nEMG, ACC_x, ACC_y, ACC_z",
            language="text")
    st.markdown("<p style='color:#7c8db0; font-size:12px;'>"
                "700 Hz · min 10 seconds</p>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='color:#7c8db0; font-size:12px; "
                "text-transform:uppercase; letter-spacing:1px;'>"
                "Model info</p>", unsafe_allow_html=True)

    if not metrics_df.empty:
        rf_row = metrics_df[metrics_df['Model'] == 'Random Forest']
        if not rf_row.empty:
            acc = rf_row['Accuracy (%)'].values[0]
            f1  = rf_row['F1 (macro)'].values[0]
            st.markdown(f"""
            <div style='color:#c9d1e9; font-size:13px; line-height:1.8;'>
            Algorithm: Random Forest<br>
            Dataset: WESAD (15 subjects)<br>
            Validation: LOSO cross-val<br>
            Accuracy: {acc}%<br>
            F1 Score: {f1}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='color:#4a5568; font-size:11px; text-align:center;'>"
                "Built with WESAD dataset<br>🤖 Llama 3 (Groq AI)</p>",
                unsafe_allow_html=True)


# ── Main header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class='app-header'>
    <h1>🧠 StressLens</h1>
    <p>Upload physiological signals — get instant stress analysis + AI-powered advice</p>
</div>
""", unsafe_allow_html=True)

# ── Demo mode or Upload ───────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📂  Upload CSV", "📊  Model Performance"])
if mode == "📝 Manual Check":

    st.markdown("### 📝 Quick Stress Check")

    hr = st.slider("Heart Rate (bpm)", 50, 140, 80)
    sleep = st.slider("Sleep Hours", 0, 12, 6)
    workload = st.selectbox("Workload", ["Low", "Medium", "High"])
    mood = st.selectbox("Mood", ["Calm", "Neutral", "Anxious"])

    if st.button("Analyze Stress"):

        score = 0

        if hr > 90: score += 30
        elif hr > 75: score += 15

        if sleep < 5: score += 30
        elif sleep < 7: score += 15

        if workload == "High": score += 25
        elif workload == "Medium": score += 10

        if mood == "Anxious": score += 25
        elif mood == "Neutral": score += 10

        score = min(score, 100)

        if score <= 35:
            cat = "Low"
        elif score <= 65:
            cat = "Moderate"
        else:
            cat = "High"

        st.success(f"Stress Level: {score} ({cat})")

        advice = get_gemini_advice(score, cat,0)

        st.markdown("### 🤖 AI Recommendations")
        st.markdown(f"""
    <div class='advice-box'>
        <h4 style='color: #818cf8 !important; font-weight: bold;'>🤖 AI Powered Advice</h4>
        <p style='color: #ffffff !important;'>{advice.replace(chr(10), '<br>')}</p>
    </div>
    """, unsafe_allow_html=True)
        

with tab1:
    uploaded = st.file_uploader(
        "Upload your physiological signal CSV",
        type=['csv'],
        help="Columns: ECG, EDA, Temp, Resp, EMG, ACC_x, ACC_y, ACC_z at 700 Hz"
    )

    # ── Demo button ───────────────────────────────────────────────────────────
    st.markdown("<div style='text-align:center; margin: 12px 0; "
                "color:#7c8db0; font-size:13px;'>— or —</div>",
                unsafe_allow_html=True)

    demo_clicked = st.button("▶  Run Demo (synthetic data)")

    if demo_clicked or uploaded is not None:
        # Load data
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.success(f"Loaded {len(df):,} rows from {uploaded.name}")
        else:
            # Synthetic demo — simulate realistic signal shapes
            np.random.seed(42)
            n = WINDOW_SIZE * 4   # 40 seconds
            t = np.arange(n)

            ecg  = np.sin(2 * np.pi * 1.2 * t / CHEST_HZ) + 0.08 * np.random.randn(n)
            eda  = 2.5 + 0.5 * np.sin(2 * np.pi * 0.05 * t / CHEST_HZ) \
                   + 0.1 * np.random.randn(n)
            temp = 34.0 + 0.3 * np.sin(2 * np.pi * 0.02 * t / CHEST_HZ) \
                   + 0.05 * np.random.randn(n)
            resp = np.sin(2 * np.pi * 0.25 * t / CHEST_HZ) \
                   + 0.05 * np.random.randn(n)
            emg  = 0.1 * np.random.randn(n)
            acc_x = 0.02 * np.random.randn(n)
            acc_y = 0.02 * np.random.randn(n)
            acc_z = 1.0  + 0.02 * np.random.randn(n)

            df = pd.DataFrame({
                'ECG': ecg, 'EDA': eda, 'Temp': temp, 'Resp': resp,
                'EMG': emg, 'ACC_x': acc_x, 'ACC_y': acc_y, 'ACC_z': acc_z
            })
            st.info("Running on synthetic demo data.")

        # ── Predict ───────────────────────────────────────────────────────────
        with st.spinner("Analyzing signals..."):
            results, error = predict_from_csv(df)

        if error:
            st.error(f"Error: {error}")
        else:
            mean_score = np.mean([r['score'] for r in results])
            mean_score = round(mean_score, 1)
            if mean_score <= thresholds['low']:       final_cat = "Low"
            elif mean_score <= thresholds['moderate']: final_cat = "Moderate"
            else:                                      final_cat = "High"

            duration_sec = results[-1]['window_start_sec'] + 10

            # ── Results row ──────────────────────────────────────────────────
            col1, col2, col3 = st.columns([1, 1, 1])

            cat_class = final_cat.lower() + '-stress'

            with col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>Stress Index</h3>
                    <p class='value {cat_class}'>{mean_score}</p>
                </div>""", unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>Category</h3>
                    <p class='value {cat_class}'>{final_cat}</p>
                </div>""", unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>Windows Analyzed</h3>
                    <p class='value' style='color:#818cf8;'>{len(results)}</p>
                </div>""", unsafe_allow_html=True)

            # ── Gauge ─────────────────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            g_col1, g_col2 = st.columns([1, 1])

            with g_col1:
                st.markdown("<p style='color:#7c8db0; font-size:12px; "
                            "text-transform:uppercase; letter-spacing:1px; "
                            "margin-bottom:4px;'>Stress gauge</p>",
                            unsafe_allow_html=True)
                st.plotly_chart(stress_gauge(mean_score, final_cat),
                                use_container_width=True)

            with g_col2:
                st.markdown("<p style='color:#7c8db0; font-size:12px; "
                            "text-transform:uppercase; letter-spacing:1px; "
                            "margin-bottom:4px;'>Stress over time</p>",
                            unsafe_allow_html=True)
                st.plotly_chart(stress_timeline(results),
                                use_container_width=True)

            # ── Gemini advice ─────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("<p style='color:#7c8db0; font-size:12px; "
                        "text-transform:uppercase; letter-spacing:1px;'>"
                        "AI-powered recommendations</p>",
                        unsafe_allow_html=True)

            with st.spinner("Generating personalized advice..."):
                advice = get_gemini_advice(mean_score, final_cat, duration_sec)

            st.markdown(f"""
            <div class='advice-box'>
                <h4>🤖 Llama 3 (Groq AI)</h4>
                {advice.replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)

            # ── Raw window table ──────────────────────────────────────────────
            with st.expander("View window-by-window breakdown"):
                results_df = pd.DataFrame(results)
                results_df.columns = ['Stress Index', 'Category',
                                      'Window Start (sec)']
                st.dataframe(results_df, use_container_width=True)

with tab2:
    st.markdown("<p style='color:#7c8db0; font-size:12px; "
                "text-transform:uppercase; letter-spacing:1px; "
                "margin-bottom:16px;'>Model comparison — LOSO validation</p>",
                unsafe_allow_html=True)

    if not metrics_df.empty:
        # Styled table
        st.dataframe(
            metrics_df.style.background_gradient(
                subset=['Accuracy (%)'], cmap='Blues'
            ).format({'Accuracy (%)': '{:.2f}',
                      'F1 (macro)': '{:.3f}'}),
            use_container_width=True, height=200
        )

        # Bar chart
        fig = px.bar(
            metrics_df, x='Model', y='Accuracy (%)',
            color='Accuracy (%)',
            color_continuous_scale=['#1e2235', '#6366f1'],
            text='Accuracy (%)'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(
            height         = 320,
            paper_bgcolor  = 'rgba(0,0,0,0)',
            plot_bgcolor   = 'rgba(0,0,0,0)',
            font           = dict(color='#7c8db0', family='DM Sans'),
            xaxis          = dict(gridcolor='#2a2d3e'),
            yaxis          = dict(gridcolor='#2a2d3e', range=[0, 110]),
            coloraxis_showscale=False,
            margin         = dict(t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run NB04 first to generate model_metrics.csv")
# ─────────────────────────────────────────────
# FUTURE ENHANCEMENTS SECTION
# ─────────────────────────────────────────────

st.markdown("---")

st.markdown("""
<div style='text-align:center; padding:20px'>

<h2 style='color:#e2e8f0;'>🚀 Future Enhancements</h2>

</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='background:#1e2235; padding:20px; border-radius:12px; border:1px solid #2a2d3e'>

<h4 style='color:#818cf8;'>⌚ Smartwatch Integration</h4>

<p style='color:#c9d1e9; font-size:14px; line-height:1.6;'>
This system can be extended to real-time wearable devices such as
Apple Watch, Fitbit, or Empatica E4 to continuously capture
physiological signals like heart rate, EDA, and motion data.
</p>

<h4 style='color:#818cf8; margin-top:16px;'>📡 Real-Time Streaming</h4>

<p style='color:#c9d1e9; font-size:14px; line-height:1.6;'>
Instead of batch CSV input, live sensor data streams can be processed
using APIs/WebSockets for real-time stress detection.
</p>

<h4 style='color:#818cf8; margin-top:16px;'>🤖 AI Personal Stress Coach</h4>

<p style='color:#c9d1e9; font-size:14px; line-height:1.6;'>
Future version can act as a personal AI wellness assistant that
tracks stress trends and provides proactive health recommendations.
</p>

<h4 style='color:#818cf8; margin-top:16px;'>📊 Cloud Deployment</h4>

<p style='color:#c9d1e9; font-size:14px; line-height:1.6;'>
Can be deployed on cloud platforms with user login and long-term
stress history tracking dashboards.
</p>

</div>
""", unsafe_allow_html=True)

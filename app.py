"""
Customer Behaviour Analysis & Churn Prediction System
Main Streamlit Application
"""

import sys
import os
import json
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Custom plotly chart wrapper to ensure transparent backgrounds and correct text theme
_original_plotly_chart = st.plotly_chart

def custom_plotly_chart(fig, *args, **kwargs):
    if fig is not None:
        theme = st.session_state.get("theme", "light")
        text_color = "#e5e9f2" if theme == "dark" else "#0f172a"
        grid_color = "rgba(255,255,255,0.08)" if theme == "dark" else "rgba(0,0,0,0.08)"
        
        # Apply transparent backgrounds and theme-based fonts
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=text_color),
        )
        # Update axes grid colors if they exist
        fig.update_xaxes(gridcolor=grid_color, zerolinecolor=grid_color)
        fig.update_yaxes(gridcolor=grid_color, zerolinecolor=grid_color)
        
    return _original_plotly_chart(fig, *args, **kwargs)

st.plotly_chart = custom_plotly_chart

from generate_data import generate_dataset
from preprocessing import clean_transactions, clean_customers, build_customer_features, rfm_segment_label
from segmentation import elbow_method, run_kmeans, label_clusters, cluster_summary
from churn_model import (train_and_compare, save_model, load_model, predict_churn_probability,
                          risk_level, MODEL_FEATURES)
from recommendations import get_segment_recommendations, get_risk_recommendation
from reports import to_csv_bytes, to_excel_bytes, to_pdf_bytes, make_bar_chart_image
from storage import saved_model_exists, ensure_dirs
from archetype import assign_archetypes, archetype_counts, VALID_ARCHETYPES
from customer_management import (ensure_raw_files, load_raw_customers, load_raw_transactions,
                                  add_customer, add_transaction, customer_purchase_stats)
from email_alerts import (load_email_config, save_email_config, config_is_complete,
                           send_alert_email, send_bulk_alerts, check_and_alert_new_at_risk,
                           load_alert_log)

# ----------------------------------------------------------------------------
# PAGE CONFIG  (must be the very first Streamlit call)
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Behaviour & Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("churn_app")
ensure_dirs()

# ----------------------------------------------------------------------------
# GLOBAL STYLE  (light / dark theme support)
# ----------------------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"


def build_css(theme: str) -> str:
    if theme == "dark":
        tokens = {
            "ink":        "#e9ecf5",
            "ink-soft":   "#9aa5b8",
            "surface":    "#0F172A",
            "card":       "rgba(255,255,255,0.045)",
            "card-solid": "#141b2d",
            "border":     "rgba(255,255,255,0.09)",
            "sidebar-bg": "#070b16",
            "sidebar-card-bg": "rgba(255,255,255,0.045)",
            "sidebar-card-border": "rgba(255,255,255,0.09)",
            "sidebar-text": "#cbd5e1",
            "sidebar-border": "rgba(255,255,255,0.07)",
            "shadow": "0 8px 30px rgba(0,0,0,0.45)",
            "glass-blur": "blur(18px)",
        }
    else:
        tokens = {
            "ink":        "#0f172a",
            "ink-soft":   "#54607a",
            "surface":    "#F8FAFC",
            "card":       "rgba(255,255,255,0.72)",
            "card-solid": "#ffffff",
            "border":     "rgba(15,23,42,0.07)",
            "sidebar-bg": "#0b1120",
            "sidebar-card-bg": "rgba(255,255,255,0.045)",
            "sidebar-card-border": "rgba(255,255,255,0.08)",
            "sidebar-text": "#cbd5e1",
            "sidebar-border": "rgba(255,255,255,0.07)",
            "shadow": "0 8px 30px rgba(79,70,229,0.09)",
            "glass-blur": "blur(18px)",
        }

    dark_bg_css = ""
    if theme == "dark":
        dark_bg_css = """
/* ---- Dark constellation network background (dark theme only) -------- */
[data-testid="stAppViewContainer"] {
    background:
      radial-gradient(650px 320px at 12% 8%, rgba(59,130,246,0.22), transparent 60%),
      radial-gradient(550px 280px at 88% 12%, rgba(236,72,153,0.18), transparent 60%),
      radial-gradient(700px 340px at 45% 95%, rgba(139,92,246,0.16), transparent 60%),
      radial-gradient(400px 220px at 75% 70%, rgba(249,115,22,0.10), transparent 60%),
      repeating-radial-gradient(circle at 18% 25%, rgba(255,255,255,0.05) 0px, transparent 1.5px, transparent 46px),
      repeating-radial-gradient(circle at 68% 55%, rgba(255,255,255,0.045) 0px, transparent 1.5px, transparent 58px),
      repeating-radial-gradient(circle at 40% 85%, rgba(255,255,255,0.04) 0px, transparent 1.5px, transparent 64px),
      var(--surface) !important;
    background-attachment: fixed !important;
}
"""

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=Inter:wght@400;500;600;700;800&display=swap');

:root {{
    --ink:        {tokens['ink']};
    --ink-soft:   {tokens['ink-soft']};
    --surface:    {tokens['surface']};
    --card:       {tokens['card']};
    --card-solid: {tokens['card-solid']};
    --border:     {tokens['border']};

    --accent:      #4F46E5;
    --accent-2:    #7C3AED;
    --blue:        #3B82F6;
    --purple:      #8B5CF6;
    --pink:        #EC4899;
    --orange:      #F97316;
    --yellow:      #FACC15;
    --green:       #22C55E;
    --teal:        #14B8A6;
    --red:         #EF4444;

    --grad-primary: linear-gradient(135deg,#4F46E5,#7C3AED);
    --grad-blue:    linear-gradient(135deg,#3B82F6,#06B6D4);
    --grad-pink:    linear-gradient(135deg,#EC4899,#F97316);
    --grad-green:   linear-gradient(135deg,#22C55E,#14B8A6);
    --grad-yellow:  linear-gradient(135deg,#FACC15,#F97316);
    --grad-purple:  linear-gradient(135deg,#8B5CF6,#EC4899);

    --good:  #16A34A;
    --warn:  #D97706;
    --bad:   #DC2626;
    --nav-bg: {tokens['sidebar-bg']};
    --radius: 20px;
    --radius-sm: 12px;
    --shadow: {tokens['shadow']};
    --glass-blur: {tokens['glass-blur']};
}}

html, body, [class*="css"]  {{ font-family: 'Inter', sans-serif; }}
h1, h2, h3, h4 {{ font-family: 'Sora', sans-serif; color: var(--ink); letter-spacing: -0.01em; }}
p, span, label, div {{ color: var(--ink); }}

::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: var(--grad-primary); border-radius: 10px; }}

@keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(14px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes floatY {{
    0%, 100% {{ transform: translateY(0px) rotate(0deg); }}
    50%      {{ transform: translateY(-18px) rotate(6deg); }}
}}
@keyframes gradientShift {{
    0%   {{ background-position: 0% 50%; }}
    50%  {{ background-position: 100% 50%; }}
    100% {{ background-position: 0% 50%; }}
}}
@keyframes glowPulse {{
    0%, 100% {{ box-shadow: 0 0 0px rgba(124,58,237,0.0); }}
    50%      {{ box-shadow: 0 0 26px rgba(124,58,237,0.35); }}
}}

/* Lock the page to a stable width so nothing reflows/jumps between reruns */
.main .block-container, [data-testid="stMainBlockContainer"] {{
    max-width: 100% !important;
    width: 100% !important;
    padding: 1.1rem 2rem 1.5rem 2rem !important;
    animation: fadeInUp 0.45s ease both;
}}
/* Columns stretch to fill available height so cards in a row line up with no gaps */
div[data-testid="stHorizontalBlock"] {{
    align-items: stretch;
}}
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {{
    display: flex;
    flex-direction: column;
}}
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] > div {{
    flex: 1 1 auto;
}}
div[data-testid="stVerticalBlockBorderWrapper"] {{
    height: 100%;
}}
/* Tighten default vertical rhythm so sections don't leave big dead gaps */
hr {{ margin: 0.85rem 0 !important; }}
div[data-testid="stVerticalBlock"] > div[data-testid="element-container"] {{ margin-bottom: 0.35rem; }}
.main {{
    background:
      radial-gradient(1100px 480px at 12% -6%, rgba(79,70,229,0.10), transparent 60%),
      radial-gradient(900px 420px at 100% 0%, rgba(236,72,153,0.08), transparent 55%),
      var(--surface);
}}
[data-testid="stAppViewContainer"] {{
    min-height: 100vh;
    background:
      radial-gradient(1100px 480px at 12% -6%, rgba(79,70,229,0.10), transparent 60%),
      radial-gradient(900px 420px at 100% 0%, rgba(236,72,153,0.08), transparent 55%),
      var(--surface);
}}
[data-testid="stHeader"] {{ background: transparent; }}

/* Gradient headline accent for page titles */
h1 {{ font-weight: 800 !important; }}
[data-testid="stAppViewContainer"] > .main h1:first-of-type {{
    background: var(--grad-primary);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}

/* ---- Sidebar ------------------------------------------------------- */
[data-testid="stSidebarContent"] {{
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
}}
section[data-testid="stSidebar"] {{
    background:
      radial-gradient(500px 260px at 0% 0%, rgba(124,58,237,0.35), transparent 60%),
      radial-gradient(400px 240px at 100% 100%, rgba(59,130,246,0.20), transparent 60%),
      var(--nav-bg);
    border-right: 1px solid {tokens['sidebar-border']};
    transition: min-width 0.2s, width 0.2s;
}}
section[data-testid="stSidebar"] div[data-testid="stHtml"],
section[data-testid="stSidebar"] iframe,
section[data-testid="stSidebar"] div.element-container {{
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}}
section[data-testid="stSidebar"][aria-expanded="true"] {{
    width: 300px !important;
    min-width: 300px !important;
}}
section[data-testid="stSidebar"][aria-expanded="true"] > div {{
    width: 300px !important;
    padding-top: 0.5rem;
}}
section[data-testid="stSidebar"] * {{ color: {tokens['sidebar-text']}; }}
section[data-testid="stSidebar"] h3 {{ color: #ffffff; }}
section[data-testid="stSidebar"] hr {{ border-color: {tokens['sidebar-border']}; }}

.sidebar-brand {{
    display: flex; align-items: center; gap: 12px;
    padding: 4px 4px 18px 4px; margin-bottom: 8px;
    border-bottom: 1px solid {tokens['sidebar-border']};
}}
.sidebar-brand .logo {{
    width: 42px; height: 42px; border-radius: 13px;
    background: var(--grad-primary);
    background-size: 200% 200%;
    animation: gradientShift 6s ease infinite, glowPulse 3.5s ease infinite;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; flex-shrink: 0;
}}
.sidebar-brand .name {{ font-family: 'Sora', sans-serif; font-weight: 700; color: #fff; font-size: 15.5px; line-height: 1.2; }}
.sidebar-brand .sub  {{ font-size: 11.5px; background: linear-gradient(90deg,#a5b4fc,#f0abfc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 600; }}

.sidebar-user {{
    display: flex; align-items: center; gap: 10px;
    background: {tokens['sidebar-card-bg']};
    border: 1px solid {tokens['sidebar-card-border']};
    border-radius: var(--radius-sm); padding: 10px 12px; margin: 14px 0 18px 0;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, border-color 0.2s ease;
}}
.sidebar-user:hover {{ transform: translateY(-2px); border-color: rgba(124,58,237,0.5); }}
.sidebar-user .avatar {{
    width: 32px; height: 32px; border-radius: 50%;
    background: var(--grad-pink);
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; color: #fff; font-size: 13px; flex-shrink: 0;
    box-shadow: 0 0 14px rgba(236,72,153,0.55);
}}
.sidebar-user .uname {{ color: #fff; font-weight: 700; font-size: 13.5px; line-height: 1.1; }}
.sidebar-user .urole {{ color: #94a3b8; font-size: 11.5px; display: flex; align-items: center; gap: 5px; }}
.status-dot {{
    width: 7px; height: 7px; border-radius: 50%; background: #22C55E; display: inline-block;
    box-shadow: 0 0 6px rgba(34,197,94,0.8);
    animation: glowPulse 2.2s ease infinite;
}}

/* Sidebar logout button — quiet, not competing with nav */
section[data-testid="stSidebar"] .stButton>button {{
    background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.25);
    color: #fca5a5; border-radius: var(--radius-sm); font-weight: 600; font-size: 13px;
    transition: all 0.2s ease;
}}
section[data-testid="stSidebar"] .stButton>button:hover {{
    background: rgba(220,38,38,0.22); border-color: rgba(220,38,38,0.55); color: #fecaca;
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(220,38,38,0.25);
}}

/* Theme toggle control in sidebar */
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
    color: {tokens['sidebar-text']} !important; font-size: 13px;
}}

/* ---- Metric cards (glass + gradient top accent + hover lift) --------- */
div[data-testid="stMetric"] {{
    position: relative;
    background: var(--card);
    backdrop-filter: var(--glass-blur);
    -webkit-backdrop-filter: var(--glass-blur);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 18px 14px 18px;
    box-shadow: var(--shadow);
    min-height: 104px;
    overflow: hidden;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    animation: fadeInUp 0.5s ease both;
}}
div[data-testid="stMetric"]::before {{
    content: "";
    position: absolute; top: 0; left: 0; right: 0; height: 4px;
    background: var(--grad-primary);
}}
div[data-testid="stMetric"]:hover {{
    transform: translateY(-4px);
    box-shadow: 0 14px 32px rgba(79,70,229,0.18);
}}
div[data-testid="column"]:nth-of-type(4n+2) div[data-testid="stMetric"]::before {{ background: var(--grad-blue); }}
div[data-testid="column"]:nth-of-type(4n+3) div[data-testid="stMetric"]::before {{ background: var(--grad-pink); }}
div[data-testid="column"]:nth-of-type(4n+4) div[data-testid="stMetric"]::before {{ background: var(--grad-green); }}

div[data-testid="stMetricLabel"] {{
    font-weight: 700;
    color: var(--ink-soft);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}}
div[data-testid="stMetricLabel"] > div {{
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: unset !important;
    line-height: 1.2 !important;
}}
div[data-testid="stMetricValue"] {{
    font-family: 'Sora', sans-serif;
    color: var(--ink);
    font-size: 1.7rem !important;
    font-weight: 700;
}}
div[data-testid="stMetricValue"] > div {{
    overflow: visible !important;
    white-space: nowrap !important;
    text-overflow: unset !important;
}}

/* ---- General components ---------------------------------------------- */
.stButton>button {{
    border-radius: var(--radius-sm); font-weight: 700; border: 1px solid var(--border);
    background: var(--card-solid); color: var(--ink);
    transition: all 0.2s ease;
}}
.stButton>button:hover {{ transform: translateY(-1px); box-shadow: 0 8px 18px rgba(79,70,229,0.14); }}
.stButton>button[kind="primary"], .stButton>button[kind="primaryFormSubmit"] {{
    background: var(--grad-primary); background-size: 200% 200%;
    color: #fff !important; border: none;
    animation: gradientShift 6s ease infinite;
    box-shadow: 0 8px 22px rgba(79,70,229,0.35);
}}
.stButton>button[kind="primary"]:hover, .stButton>button[kind="primaryFormSubmit"]:hover {{
    box-shadow: 0 10px 28px rgba(79,70,229,0.5);
    transform: translateY(-2px) scale(1.01);
}}
div[data-testid="stForm"] {{
    background: var(--card); backdrop-filter: var(--glass-blur);
    border: 1px solid var(--border);
    border-radius: var(--radius); padding: 22px 26px;
    box-shadow: var(--shadow);
}}
[data-testid="stDataFrame"] {{
    border-radius: var(--radius-sm); overflow: hidden; border: 1px solid var(--border);
    box-shadow: var(--shadow);
}}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
.stTabs [data-baseweb="tab"] {{
    border-radius: 999px; font-weight: 700; color: var(--ink);
    background: var(--card); border: 1px solid var(--border); padding: 6px 16px;
    transition: all 0.2s ease;
}}
.stTabs [aria-selected="true"] {{
    background: var(--grad-primary) !important; color: #fff !important;
}}

/* Chart & element containers get a soft glass card treatment */
div[data-testid="stPlotlyChart"] {{
    background: var(--card);
    backdrop-filter: var(--glass-blur);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 10px;
    box-shadow: var(--shadow);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}}
div[data-testid="stPlotlyChart"]:hover {{
    transform: translateY(-3px);
    box-shadow: 0 16px 34px rgba(79,70,229,0.14);
}}

/* Inputs adapt to theme */
.stTextInput input, .stNumberInput input, .stSelectbox [data-baseweb="select"] > div,
.stTextArea textarea {{
    background: var(--card-solid) !important; color: var(--ink) !important;
    border-color: var(--border) !important;
    border-radius: 10px !important;
}}
.stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {{
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.15) !important;
}}

.badge-low    {{ background: var(--grad-green);  color:#fff; padding:4px 14px; border-radius:20px; font-weight:700; font-size:12.5px; box-shadow:0 4px 12px rgba(34,197,94,0.35); }}
.badge-medium {{ background: var(--grad-yellow); color:#7c2d12; padding:4px 14px; border-radius:20px; font-weight:700; font-size:12.5px; box-shadow:0 4px 12px rgba(250,204,21,0.35); }}
.badge-high   {{ background: linear-gradient(135deg,#EF4444,#EC4899); color:#fff; padding:4px 14px; border-radius:20px; font-weight:700; font-size:12.5px; box-shadow:0 4px 12px rgba(239,68,68,0.35); }}

/* ---- Custom KPI grid (used on Dashboard) ------------------------------ */
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 18px;
    margin: 6px 0 8px 0;
}}
.kpi-card {{
    position: relative;
    overflow: hidden;
    border-radius: var(--radius);
    padding: 20px 20px 18px 20px;
    color: #fff;
    box-shadow: 0 12px 28px rgba(0,0,0,0.14);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    animation: fadeInUp 0.55s ease both;
}}
.kpi-card::after {{
    content: "";
    position: absolute; top: -30px; right: -30px;
    width: 110px; height: 110px; border-radius: 50%;
    background: rgba(255,255,255,0.16);
}}
.kpi-card:hover {{ transform: translateY(-6px) scale(1.015); box-shadow: 0 20px 40px rgba(0,0,0,0.24); }}
.kpi-grid > .kpi-card:nth-child(1) {{ animation-delay: 0.03s; }}
.kpi-grid > .kpi-card:nth-child(2) {{ animation-delay: 0.09s; }}
.kpi-grid > .kpi-card:nth-child(3) {{ animation-delay: 0.15s; }}
.kpi-grid > .kpi-card:nth-child(4) {{ animation-delay: 0.21s; }}
.kpi-grid > .kpi-card:nth-child(5) {{ animation-delay: 0.27s; }}
.kpi-icon {{ font-size: 24px; margin-bottom: 10px; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2)); }}
.kpi-value {{ font-family: 'Sora', sans-serif; font-size: 1.65rem; font-weight: 800; line-height: 1.1; margin-bottom: 4px; }}
.kpi-label {{ font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; opacity: 0.92; }}

.kpi-purple {{ background: var(--grad-primary); }}
.kpi-blue   {{ background: var(--grad-blue); }}
.kpi-pink   {{ background: var(--grad-pink); }}
.kpi-green  {{ background: var(--grad-green); }}
.kpi-yellow {{ background: linear-gradient(135deg,#FACC15,#F97316); color:#5a2e00; }}
.kpi-teal   {{ background: linear-gradient(135deg,#14B8A6,#3B82F6); }}

/* ---- Glass panel / AI card / timeline --------------------------------- */
.glass-panel {{
    background: var(--card);
    backdrop-filter: var(--glass-blur);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 22px 24px;
    box-shadow: var(--shadow);
    animation: fadeInUp 0.5s ease both;
}}
.ai-card {{
    background: linear-gradient(135deg, rgba(79,70,229,0.14), rgba(236,72,153,0.10));
    border: 1px solid rgba(124,58,237,0.35);
    border-radius: var(--radius);
    padding: 22px 24px;
    box-shadow: 0 10px 30px rgba(124,58,237,0.12);
    animation: fadeInUp 0.5s ease both, glowPulse 4s ease infinite;
}}
.ai-card .ai-tag {{
    display: inline-block; padding: 3px 12px; border-radius: 999px;
    background: var(--grad-primary); color: #fff; font-size: 11.5px; font-weight: 800;
    letter-spacing: 0.04em; text-transform: uppercase; margin-bottom: 10px;
}}
.timeline-item {{
    display: flex; gap: 12px; align-items: flex-start;
    padding: 10px 0; border-bottom: 1px dashed var(--border);
}}
.timeline-item:last-child {{ border-bottom: none; }}
.timeline-dot {{
    width: 10px; height: 10px; border-radius: 50%; margin-top: 5px; flex-shrink: 0;
    background: var(--grad-primary); box-shadow: 0 0 8px rgba(124,58,237,0.6);
}}
.timeline-text {{ font-size: 13.5px; color: var(--ink); line-height: 1.4; }}
.timeline-meta {{ font-size: 11.5px; color: var(--ink-soft); }}

/* Responsive tweaks */
@media (max-width: 900px) {{
    .main .block-container {{ padding: 1.1rem 1rem 2rem 1rem; }}
    .kpi-value {{ font-size: 1.35rem; }}
}}

{dark_bg_css}

/* ---- Stat cards (bordered containers) with mini sparklines ----------- */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: var(--card);
    backdrop-filter: var(--glass-blur);
    -webkit-backdrop-filter: var(--glass-blur);
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 4px 6px;
    box-shadow: var(--shadow);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    animation: fadeInUp 0.5s ease both;
}}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
    transform: translateY(-4px);
    box-shadow: 0 16px 34px rgba(79,70,229,0.16);
}}
/* Charts nested inside a stat card shouldn't double up on card styling */
div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stPlotlyChart"] {{
    background: transparent; border: none; box-shadow: none; padding: 0;
}}
div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stPlotlyChart"]:hover {{
    transform: none; box-shadow: none;
}}

.trend-pill {{
    display: inline-flex; align-items: center; gap: 3px;
    padding: 2px 9px; border-radius: 999px; font-size: 11.5px; font-weight: 800;
}}
.trend-up   {{ background: rgba(34,197,94,0.16);  color: #22C55E; }}
.trend-down {{ background: rgba(239,68,68,0.16);  color: #EF4444; }}
</style>
<script>
(function() {{
    const styleIframes = () => {{
        const sidebar = document.querySelector('section[data-testid="stSidebar"]');
        if (!sidebar) return;
        const iframes = sidebar.querySelectorAll('iframe');
        iframes.forEach(iframe => {{
            try {{
                const doc = iframe.contentDocument || iframe.contentWindow.document;
                if (doc && doc.head) {{
                    let styleTag = doc.getElementById('custom-iframe-style');
                    if (!styleTag) {{
                        styleTag = doc.createElement('style');
                        styleTag.id = 'custom-iframe-style';
                        styleTag.innerHTML = `
                            html, body, #root, .container-fluid, .nav, .nav-item, ul, li {{
                                background-color: transparent !important;
                                background: transparent !important;
                                border: none !important;
                                box-shadow: none !important;
                            }}
                            .nav-link {{
                                background-color: transparent !important;
                                background: transparent !important;
                                border: none !important;
                                box-shadow: none !important;
                                transition: transform 0.15s ease, background 0.2s ease !important;
                            }}
                            .nav-link:hover {{
                                background: rgba(124,58,237,0.14) !important;
                                transform: translateX(3px);
                            }}
                            .nav-link.active, .nav-link-selected, .active {{
                                background-image: linear-gradient(135deg, #4F46E5, #7C3AED) !important;
                                background-color: #4f46e5 !important;
                                color: #ffffff !important;
                                box-shadow: 0 4px 16px rgba(124,58,237,0.55) !important;
                            }}
                        `;
                        doc.head.appendChild(styleTag);
                    }}
                }}
            }} catch (e) {{
                // Ignore same-origin loading issues
            }}
        }});
    }};
    setInterval(styleIframes, 200);
}})();

/* ---- Animated count-up for KPI cards & metrics ----------------------- */
(function() {{
    function animateValue(el) {{
        if (!el || el.dataset.counted === "1") return;
        const raw = el.textContent.trim();
        const match = raw.match(/^([^\\d\\-]*)([\\d,]+\\.?\\d*)(.*)$/);
        if (!match) return;
        el.dataset.counted = "1";
        const prefix = match[1], suffix = match[3];
        const numStr = match[2].replace(/,/g, "");
        const end = parseFloat(numStr);
        if (isNaN(end)) return;
        const decimals = (numStr.split(".")[1] || "").length;
        const duration = 850;
        const t0 = performance.now();
        function step(now) {{
            const p = Math.min((now - t0) / duration, 1);
            const eased = 1 - Math.pow(1 - p, 3);
            const current = end * eased;
            el.textContent = prefix + current.toLocaleString("en-IN", {{
                minimumFractionDigits: decimals, maximumFractionDigits: decimals
            }}) + suffix;
            if (p < 1) requestAnimationFrame(step);
            else el.textContent = raw;
        }}
        requestAnimationFrame(step);
    }}
    function scan() {{
        document.querySelectorAll(
            '.kpi-value, div[data-testid="stMetricValue"] > div'
        ).forEach(animateValue);
    }}
    setInterval(scan, 350);
}})();

/* ---- Ripple micro-interaction on buttons ------------------------------ */
(function() {{
    document.addEventListener("click", function(e) {{
        const btn = e.target.closest(".stButton>button, button");
        if (!btn) return;
        const circle = document.createElement("span");
        const d = Math.max(btn.clientWidth, btn.clientHeight);
        circle.style.cssText = `
            position:absolute; border-radius:50%; pointer-events:none;
            width:${{d}}px; height:${{d}}px;
            left:${{e.clientX - btn.getBoundingClientRect().left - d/2}}px;
            top:${{e.clientY - btn.getBoundingClientRect().top - d/2}}px;
            background: rgba(255,255,255,0.45);
            transform: scale(0); opacity: 1;
            transition: transform 0.5s ease, opacity 0.6s ease;
        `;
        btn.style.position = "relative";
        btn.style.overflow = "hidden";
        btn.appendChild(circle);
        requestAnimationFrame(() => {{
            circle.style.transform = "scale(2.5)";
            circle.style.opacity = "0";
        }});
        setTimeout(() => circle.remove(), 650);
    }});
}})();
</script>
"""


st.markdown(build_css(st.session_state.theme), unsafe_allow_html=True)

px.defaults.template = "plotly_dark" if st.session_state.theme == "dark" else "plotly_white"


# ----------------------------------------------------------------------------
# REUSABLE UI COMPONENTS (presentation-only helpers — no data/logic here)
# ----------------------------------------------------------------------------
def page_header(title_text: str, subtitle: str = "") -> None:
    """Top bar used on every page: gradient title + user pill / bell / date badge,
    matching the reference dashboard's header."""
    _today_str = datetime.now().strftime("%b %d, %Y")
    _username_display = (st.session_state.auth.get("username") or "Admin").title()
    _initial = _username_display[0].upper()
    subtitle_html = (
        f"<div style='color:var(--ink-soft); font-size:13px; margin-top:2px;'>{subtitle}</div>"
        if subtitle else ""
    )
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px; margin-bottom:4px;">
            <div>
                <div style="font-family:'Sora',sans-serif; font-weight:800; font-size:1.9rem;
                            background: var(--grad-primary); -webkit-background-clip:text;
                            -webkit-text-fill-color:transparent; background-clip:text;">{title_text}</div>
                {subtitle_html}
            </div>
            <div style="display:flex; align-items:center; gap:10px;">
                <div style="display:flex; align-items:center; gap:8px; background:var(--card);
                            border:1px solid var(--border); border-radius:999px; padding:5px 14px 5px 6px;
                            backdrop-filter: var(--glass-blur); box-shadow: var(--shadow);">
                    <div style="width:26px; height:26px; border-radius:50%; background:var(--grad-pink);
                                display:flex; align-items:center; justify-content:center; color:#fff;
                                font-weight:700; font-size:11px;">{_initial}</div>
                    <span style="font-size:13px; font-weight:700; color:var(--ink);">{_username_display}</span>
                </div>
                <div style="width:36px; height:36px; border-radius:50%; background:var(--card);
                            border:1px solid var(--border); display:flex; align-items:center; justify-content:center;
                            font-size:15px; position:relative; backdrop-filter: var(--glass-blur); box-shadow: var(--shadow);">
                    🔔
                    <span style="position:absolute; top:7px; right:8px; width:7px; height:7px; border-radius:50%;
                                 background:#EF4444; box-shadow:0 0 6px rgba(239,68,68,0.8);"></span>
                </div>
                <div style="background:var(--card); border:1px solid var(--border); border-radius:999px;
                            padding:7px 16px; font-size:12.5px; font-weight:700; color:var(--ink);
                            backdrop-filter: var(--glass-blur); box-shadow: var(--shadow);">
                    📅 {_today_str}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_row(cards: list) -> None:
    """Render a row of colorful gradient KPI cards.
    Each item in `cards` is a dict: {"label", "value", "icon", "grad"}
    where grad is one of: purple, blue, pink, green, yellow, teal."""
    html = '<div class="kpi-grid">'
    for c in cards:
        html += f"""
        <div class="kpi-card kpi-{c.get('grad', 'purple')}">
            <div class="kpi-icon">{c.get('icon', '📊')}</div>
            <div class="kpi-value">{c['value']}</div>
            <div class="kpi-label">{c['label']}</div>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def ai_insight_card(tag: str, title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="ai-card">
            <span class="ai-tag">{tag}</span>
            <div style="font-family:'Sora',sans-serif; font-weight:700; font-size:16px; margin-bottom:6px;">{title}</div>
            <div style="font-size:13.5px; color:var(--ink-soft); line-height:1.55;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def activity_timeline(items: list) -> None:
    """items: list of (text, meta) tuples."""
    html = '<div class="glass-panel">'
    for text, meta in items:
        html += f"""
        <div class="timeline-item">
            <div class="timeline-dot"></div>
            <div>
                <div class="timeline-text">{text}</div>
                <div class="timeline-meta">{meta}</div>
            </div>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _hex_to_rgba(hex_color: str, alpha: float = 0.18) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def sparkline_fig(x, y, kind: str = "area", color: str = "#8B5CF6", height: int = 64):
    """A minimal, axis-free Plotly figure used as an inline 'sparkline' inside stat cards."""
    fig = go.Figure()
    if kind == "bar":
        fig.add_trace(go.Bar(
            x=list(range(len(y))), y=y,
            marker=dict(color=y, colorscale=[[0, "#3B82F6"], [1, "#8B5CF6"]], showscale=False),
        ))
    else:
        fig.add_trace(go.Scatter(
            x=list(range(len(y))), y=y, mode="lines",
            line=dict(color=color, width=2.5, shape="spline"),
            fill="tozeroy", fillcolor=_hex_to_rgba(color, 0.18),
        ))
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=2, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, showgrid=False),
        yaxis=dict(visible=False, showgrid=False),
        showlegend=False,
        hovermode=False,
    )
    return fig


def window_change(dates: pd.Series, values: pd.Series, ref_date, days: int = 30, agg: str = "sum"):
    """Compare a metric over the last `days` vs the prior `days`-day window.
    Returns (current_total, pct_change). Purely descriptive — computed from real dates."""
    dates = pd.to_datetime(dates)
    cur_mask = (dates > ref_date - pd.Timedelta(days=days)) & (dates <= ref_date)
    prev_mask = (dates > ref_date - pd.Timedelta(days=2 * days)) & (dates <= ref_date - pd.Timedelta(days=days))
    if agg == "count":
        cur = int(cur_mask.sum())
        prev = int(prev_mask.sum())
    else:
        cur = float(values[cur_mask].sum())
        prev = float(values[prev_mask].sum())
    pct = ((cur - prev) / prev * 100) if prev else 0.0
    return cur, pct


def stat_card(label: str, value: str, delta_pct: float, fig, icon: str = "📈") -> None:
    """A dark-glass bordered card with a label, big value, real trend badge, and a mini sparkline —
    mirrors the reference dashboard's 'Overall Performance' style card."""
    with st.container(border=True):
        top_l, top_r = st.columns([1, 0.15])
        with top_l:
            st.markdown(
                f"<div style='font-size:13px; font-weight:700; color:var(--ink-soft);'>{icon} {label}</div>",
                unsafe_allow_html=True,
            )
        with top_r:
            st.markdown(
                "<div style='text-align:right; color:var(--ink-soft); font-size:16px;'>⋯</div>",
                unsafe_allow_html=True,
            )
        arrow = "▲" if delta_pct >= 0 else "▼"
        cls = "trend-up" if delta_pct >= 0 else "trend-down"
        st.markdown(
            f"""<div style="display:flex; align-items:baseline; gap:10px; margin:2px 0 4px 0;">
                <span style="font-family:'Sora',sans-serif; font-weight:800; font-size:1.55rem; color:var(--ink);">{value}</span>
                <span class="trend-pill {cls}">{arrow} {abs(delta_pct):.1f}%</span>
            </div>""",
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ----------------------------------------------------------------------------
# AUTH
# ----------------------------------------------------------------------------
USERS_FILE = os.path.join("data", "users.json")

_DEFAULT_USERS = {
    "admin": {"password": "admin123", "role": "Admin"},
    "user":  {"password": "user123",  "role": "User"},
}


def load_users() -> dict:
    """Load users (with any password changes) from disk, seeding the
    default accounts the first time the app runs."""
    os.makedirs("data", exist_ok=True)
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    save_users(_DEFAULT_USERS)
    return dict(_DEFAULT_USERS)


def save_users(users: dict) -> None:
    os.makedirs("data", exist_ok=True)
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


if "auth" not in st.session_state:
    st.session_state.auth = {"logged_in": False, "username": None, "role": None}


def login_screen():
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(-45deg, #4F46E5, #7C3AED, #EC4899, #3B82F6);
            background-size: 400% 400%;
            animation: loginGradient 16s ease infinite;
        }
        @keyframes loginGradient {
            0%   { background-position: 0% 50%; }
            50%  { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        [data-testid="stHeader"] { background: transparent; }
        .float-shape {
            position: fixed; border-radius: 50%;
            background: rgba(255,255,255,0.14);
            filter: blur(2px);
            animation: floatY 7s ease-in-out infinite;
            z-index: 0;
        }
        .fs1 { width: 140px; height: 140px; top: 8%;  left: 8%;  animation-duration: 8s; }
        .fs2 { width: 90px;  height: 90px;  top: 70%; left: 12%; animation-duration: 6.5s; animation-delay: 1s; }
        .fs3 { width: 180px; height: 180px; top: 12%; left: 82%; animation-duration: 9s;  animation-delay: 0.5s; }
        .fs4 { width: 110px; height: 110px; top: 68%; left: 80%; animation-duration: 7.5s; animation-delay: 1.5s; }

        .login-wrap { display:flex; justify-content:center; margin-top:48px; position:relative; z-index:1; }
        .login-card {
            width:100%; max-width:420px;
            background: rgba(255,255,255,0.85);
            backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
            border:1px solid rgba(255,255,255,0.5); border-radius:24px;
            padding:40px 36px 28px 36px;
            box-shadow:0 25px 60px rgba(15,23,42,0.35);
            animation: fadeInUp 0.6s ease both;
        }
        .login-logo {
            width:58px; height:58px; border-radius:16px; margin:0 auto 18px auto;
            background: linear-gradient(135deg,#4F46E5,#7C3AED,#EC4899);
            background-size: 200% 200%;
            animation: gradientShift 5s ease infinite, glowPulse 3s ease infinite;
            display:flex; align-items:center; justify-content:center; font-size:28px;
        }
        .login-title {
            text-align:center; font-family:'Sora',sans-serif; font-weight:800;
            font-size:21px; margin-bottom:2px;
            background: linear-gradient(135deg,#4F46E5,#7C3AED);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .login-sub { text-align:center; color:#64748b; font-size:13.5px; margin-bottom:22px; }
        </style>

        <div class="float-shape fs1"></div>
        <div class="float-shape fs2"></div>
        <div class="float-shape fs3"></div>
        <div class="float-shape fs4"></div>

        <div class="login-wrap"><div class="login-card">
            <div class="login-logo">📊</div>
            <div class="login-title">Customer Behaviour &amp; Churn Analytics</div>
            <div class="login-sub">Sign in to access your dashboard</div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input(
            "Password", type="password", placeholder="Enter your password"
        )
        r1, r2 = st.columns([1, 1])
        with r1:
            remember = st.checkbox("Remember me", value=True)
        with r2:
            st.markdown(
                "<div style='text-align:right; margin-top:6px;'>"
                "<span style='font-size:12.5px; color:#4F46E5; font-weight:600;'>Forgot password?</span>"
                "</div>",
                unsafe_allow_html=True,
            )
        if st.button("Sign in", use_container_width=True, type="primary"):
            with st.spinner("Verifying credentials..."):
                import time as _time
                _time.sleep(0.5)
                users = load_users()
                user = users.get(username)
                if user and user["password"] == password:
                    st.session_state.auth = {
                        "logged_in": True,
                        "username": username,
                        "role": user["role"],
                        "remember": remember,
                    }
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
        st.caption(
            "🔒 Forgot your password? Ask an admin to reset it, or sign in and use "
            "**Change Password** from the sidebar. Password visibility can be toggled "
            "with the 👁 icon inside the password field."
        )
    st.markdown("</div></div>", unsafe_allow_html=True)


if not st.session_state.auth.get("logged_in", False):
    login_screen()
    st.stop()

role = st.session_state.auth["role"]

# ----------------------------------------------------------------------------
# DATA PIPELINE  (canonical raw CSVs → cleaned → features → archetypes)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_pipeline(version: int, n_customers: int):
    """Rebuild the full feature pipeline from the canonical raw CSVs.
    `version` is bumped any time a customer/transaction is added or the
    dataset is regenerated, which invalidates this cache automatically."""
    if not os.path.exists("data/customers.csv") or not os.path.exists("data/transactions.csv"):
        seed_customers, seed_tx = generate_dataset(n_customers)
        ensure_raw_files(seed_customers, seed_tx)

    customers_raw = load_raw_customers()
    tx_raw        = load_raw_transactions()

    customers  = clean_customers(customers_raw)
    tx_clean, clean_stats = clean_transactions(tx_raw)
    features   = build_customer_features(customers, tx_clean)
    features["RFM_segment"] = features.apply(rfm_segment_label, axis=1)

    reference_date = (
        tx_clean["transaction_date"].max() + pd.Timedelta(days=1)
        if len(tx_clean) else pd.Timestamp.now()
    )
    features = assign_archetypes(features, reference_date)

    return customers, tx_clean, features, clean_stats


if "n_customers"  not in st.session_state:
    st.session_state.n_customers  = 1000
if "data_version" not in st.session_state:
    st.session_state.data_version = 0

with st.spinner("Loading customer data and refreshing analytics..."):
    customers, tx, features, clean_stats = build_pipeline(
        st.session_state.data_version, st.session_state.n_customers
    )

st.session_state.customers   = customers
st.session_state.tx          = tx
st.session_state.features    = features
st.session_state.clean_stats = clean_stats

# Restore a previously saved churn model on first load
if saved_model_exists() and "ml_result" not in st.session_state:
    try:
        best_model, scaler = load_model()
        st.session_state.ml_result = {
            "best_model": best_model, "scaler": scaler,
            "best_name": "Restored Model",
            "comparison_df": pd.DataFrame(), "results": {},
            "feature_importances": pd.Series(dtype=float),
        }
        logger.info("Restored previously trained churn model from disk")
    except Exception as exc:
        logger.warning("Could not restore saved model: %s", exc)


def refresh_all(retrain_model: bool = True):
    """Bump data version (invalidates cache), rebuild pipeline into
    session state, optionally retrain the churn model, and automatically
    send at-risk transition alerts for any customer whose archetype just
    changed to 'at_risk'."""
    st.session_state.data_version += 1
    st.cache_data.clear()
    st.session_state.pop("clustered", None)

    # snapshot archetypes BEFORE rebuild so we can detect transitions
    old_features = st.session_state.get("features", pd.DataFrame())

    fresh_c, fresh_tx, fresh_f, fresh_s = build_pipeline(
        st.session_state.data_version, st.session_state.n_customers
    )
    st.session_state.customers   = fresh_c
    st.session_state.tx          = fresh_tx
    st.session_state.features    = fresh_f
    st.session_state.clean_stats = fresh_s

    # auto-alert: customers who just became at_risk
    if not old_features.empty and "archetype" in old_features.columns:
        cfg = load_email_config()
        if config_is_complete(cfg):
            alerts = check_and_alert_new_at_risk(old_features, fresh_f, cfg)
            if alerts:
                for msg in alerts:
                    st.toast(msg, icon="📧")

    if retrain_model:
        with st.spinner("Refreshing churn model..."):
            try:
                result = train_and_compare(fresh_f)
                save_model(result["best_model"], result["scaler"])
                st.session_state.ml_result = result
            except Exception as exc:
                logger.warning("Auto-retrain skipped: %s", exc)


# ----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------------
from streamlit_option_menu import option_menu

username_display = (st.session_state.auth.get("username") or "Admin").title()
initial = username_display[0].upper()

st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <div class="logo">📊</div>
        <div>
            <div class="name">Churn Analytics</div>
            <div class="sub">Customer Behaviour Suite</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    f"""
    <div class="sidebar-user">
        <div class="avatar">{initial}</div>
        <div>
            <div class="uname">{username_display}</div>
            <div class="urole"><span class="status-dot"></span>{role}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

dark_on = st.sidebar.toggle(
    "🌙 Dark mode" if st.session_state.theme == "light" else "☀️ Light mode",
    value=(st.session_state.theme == "dark"),
    key="theme_toggle",
)
new_theme = "dark" if dark_on else "light"
if new_theme != st.session_state.theme:
    st.session_state.theme = new_theme
    st.rerun()

st.sidebar.markdown("<div style='margin:10px 0 4px 0'></div>", unsafe_allow_html=True)

PAGE_ICONS = {
    "Dashboard": "speedometer2",
    "Dataset Management": "database",
    "Add New Customer": "person-plus",
    "Add New Transaction": "cart-plus",
    "Customer Archetypes": "tags",
    "Exploratory Data Analysis": "bar-chart-line",
    "Customer Segmentation": "diagram-3",
    "Churn Prediction (ML Results)": "cpu",
    "Individual Customer Lookup": "search",
    "Business Recommendations": "lightbulb",
    "Email Alerts": "envelope",
    "Reports": "file-earmark-text",
    "Change Password": "key",
}

if role == "Admin":
    pages = [
        "Dashboard", "Dataset Management", "Add New Customer", "Add New Transaction",
        "Customer Archetypes", "Exploratory Data Analysis",
        "Customer Segmentation", "Churn Prediction (ML Results)",
        "Individual Customer Lookup", "Business Recommendations",
        "Email Alerts", "Reports", "Change Password",
    ]
else:
    pages = [
        "Dashboard", "Customer Archetypes", "Exploratory Data Analysis",
        "Customer Segmentation", "Individual Customer Lookup", "Business Recommendations",
        "Change Password",
    ]

with st.sidebar:
    sidebar_bg = "#070b16" if st.session_state.theme == "dark" else "#0b1120"
    page = option_menu(
        menu_title=None,
        options=pages,
        icons=[PAGE_ICONS[p] for p in pages],
        default_index=0,
        styles={
            "container": {
                "padding": "0!important",
                "background-color": "transparent",
                "border": "none",
            },
            "icon": {"color": "#a5b4fc", "font-size": "15px"},
            "nav-link": {
                "font-size": "14px",
                "font-weight": "600",
                "color": "#cbd5e1",
                "text-align": "left",
                "margin": "3px 0",
                "padding": "11px 12px",
                "border-radius": "12px",
                "background-color": "transparent",
                "--hover-color": "rgba(124,58,237,0.14)",
            },
            "nav-link-selected": {
                "background-image": "linear-gradient(135deg, #4F46E5, #7C3AED)",
                "background-color": "#4f46e5",
                "color": "#ffffff",
                "font-weight": "700",
                "box-shadow": "0 4px 16px rgba(124,58,237,0.55)",
            },
        },
    )

st.sidebar.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)
if st.sidebar.button("Logout", use_container_width=True):
    st.session_state.auth = {"logged_in": False, "username": None, "role": None}
    st.rerun()

# Keep local references up to date (refresh_all writes into session_state)
customers   = st.session_state.customers
tx          = st.session_state.tx
features    = st.session_state.features
clean_stats = st.session_state.clean_stats

# ----------------------------------------------------------------------------
# DASHBOARD
# ----------------------------------------------------------------------------
if page == "Dashboard":
    page_header("📊 Dashboard", "Executive overview — refreshed automatically whenever data changes.")

    total_customers = len(features)
    total_transactions = len(tx)
    revenue = tx["amount"].sum()
    active_customers = (features["recency_days"] <= 90).sum()
    churn_rate = features["churned"].mean() * 100
    aov = tx["amount"].mean()
    avg_freq = features["purchase_frequency_rate"].mean()
    clv = features["CLV"].mean()

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Customers", f"{total_customers:,}")
    m2.metric("Transactions", f"{total_transactions:,}")
    m3.metric("Active (90d)", f"{active_customers:,}")
    m4.metric("Churn Rate", f"{churn_rate:.1f}%")
    m5.metric("Avg Freq/mo", f"{avg_freq:.2f}")
    m6.metric("Avg CLV", f"₹{clv:,.0f}")

    # Real trend windows — last 30 days vs the prior 30 days, from actual dates in the data.
    ref_date = tx["transaction_date"].max()
    daily_rev = tx.set_index("transaction_date").resample("D")["amount"].sum().reset_index()
    daily_rev_recent = daily_rev[daily_rev["transaction_date"] > ref_date - pd.Timedelta(days=45)]
    revenue_now, revenue_pct = window_change(tx["transaction_date"], tx["amount"], ref_date, 30, "sum")

    weekly_tx = tx.set_index("transaction_date").resample("W")["transaction_id"].count().reset_index()
    weekly_tx_recent = weekly_tx.tail(10)
    txcount_now, txcount_pct = window_change(tx["transaction_date"], tx["amount"], ref_date, 30, "count")

    signup_dates = pd.to_datetime(features["signup_date"])
    weekly_signups = (
        features.assign(signup_date=signup_dates)
        .set_index("signup_date")["customer_id"].resample("W").count().reset_index()
    )
    weekly_signups_recent = weekly_signups.tail(10)
    signups_now, signups_pct = window_change(signup_dates, pd.Series(1, index=signup_dates.index), ref_date, 30, "count")

    weekly_aov = tx.set_index("transaction_date").resample("W")["amount"].mean().reset_index()
    weekly_aov_recent = weekly_aov.tail(10)
    cur_win = tx[(tx["transaction_date"] > ref_date - pd.Timedelta(days=30)) & (tx["transaction_date"] <= ref_date)]
    prev_win = tx[(tx["transaction_date"] > ref_date - pd.Timedelta(days=60)) & (tx["transaction_date"] <= ref_date - pd.Timedelta(days=30))]
    aov_now = cur_win["amount"].mean() if len(cur_win) else aov
    aov_prev = prev_win["amount"].mean() if len(prev_win) else aov_now
    aov_pct = ((aov_now - aov_prev) / aov_prev * 100) if aov_prev else 0.0

    st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)
    colL, colR = st.columns([1.35, 1])
    with colL:
        fig_perf = sparkline_fig(
            daily_rev_recent["transaction_date"], daily_rev_recent["amount"],
            kind="area", color="#8B5CF6", height=190,
        )
        stat_card("Overall Performance", f"₹{revenue_now:,.0f}", revenue_pct, fig_perf, icon="📈")
    with colR:
        fig_active = sparkline_fig(
            weekly_tx_recent["transaction_date"], weekly_tx_recent["transaction_id"],
            kind="bar", height=70,
        )
        stat_card("Transaction Volume (30d)", f"{txcount_now:,}", txcount_pct, fig_active, icon="⚡")
        r1, r2 = st.columns(2)
        with r1:
            fig_growth = sparkline_fig(
                weekly_signups_recent["signup_date"], weekly_signups_recent["customer_id"],
                kind="area", color="#3B82F6", height=64,
            )
            stat_card("Customer Growth", f"{signups_now:,}", signups_pct, fig_growth, icon="👥")
        with r2:
            fig_aov = sparkline_fig(
                weekly_aov_recent["transaction_date"], weekly_aov_recent["amount"],
                kind="area", color="#EC4899", height=64,
            )
            stat_card("Avg Order Value", f"₹{aov_now:,.0f}", aov_pct, fig_aov, icon="🛒")

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    st.subheader("🏷️ Customer Archetypes")
    a_counts = archetype_counts(features)
    kpi_row([
        {"label": "New",        "value": a_counts["new"],        "icon": "🆕", "grad": "blue"},
        {"label": "Regular",    "value": a_counts["regular"],    "icon": "🙂", "grad": "teal"},
        {"label": "Loyal",      "value": a_counts["loyal"],      "icon": "⭐", "grad": "purple"},
        {"label": "At Risk",    "value": a_counts["at_risk"],    "icon": "⚠️", "grad": "yellow"},
        {"label": "High Value", "value": a_counts["high_value"], "icon": "💎", "grad": "pink"},
    ])

    st.divider()
    with st.container(border=True):
        st.markdown(
            "<div style='font-weight:800; font-family:Sora,sans-serif; font-size:16px; margin-bottom:6px;'>"
            "📊 Analytics Overview</div>",
            unsafe_allow_html=True,
        )
        colA, colB = st.columns([1.6, 1])
        with colA:
            monthly = tx.set_index("transaction_date").resample("ME")["amount"].sum().reset_index()
            fig_bar = px.bar(monthly, x="transaction_date", y="amount", title="Monthly Revenue")
            fig_bar.update_traces(marker=dict(color=monthly["amount"], colorscale=[[0, "#3B82F6"], [1, "#EC4899"]]))
            fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_bar, use_container_width=True)
        with colB:
            seg_counts = features["RFM_segment"].value_counts().reset_index()
            seg_counts.columns = ["segment", "count"]
            seg_colors = ["#8B5CF6", "#EC4899", "#3B82F6", "#22C55E", "#F97316", "#FACC15"]
            fig_donut = px.pie(seg_counts, names="segment", values="count",
                                title="Customer Segments", hole=0.6,
                                color_discrete_sequence=seg_colors)
            fig_donut.update_traces(showlegend=False, textinfo="none")
            fig_donut.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=40, b=0), height=210,
            )
            st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})
            total_segs = seg_counts["count"].sum()
            legend_html = "<div style='padding:0 4px;'>"
            for i, row in seg_counts.iterrows():
                pct = (row["count"] / total_segs * 100) if total_segs else 0
                color = seg_colors[i % len(seg_colors)]
                legend_html += f"""
                <div style="display:flex; align-items:center; justify-content:space-between; padding:4px 0; font-size:12.5px;">
                    <div style="display:flex; align-items:center; gap:7px;">
                        <span style="width:9px; height:9px; border-radius:50%; background:{color}; display:inline-block;"></span>
                        <span style="color:var(--ink-soft); font-weight:600;">{row['segment']}</span>
                    </div>
                    <span style="color:var(--ink); font-weight:700;">{pct:.0f}%</span>
                </div>"""
            legend_html += "</div>"
            st.markdown(legend_html, unsafe_allow_html=True)

    st.divider()
    colC, colD = st.columns([1.3, 1])
    with colC:
        st.subheader("🤖 AI Insight")
        top_archetype = max(a_counts, key=a_counts.get)
        if churn_rate >= 20:
            tag, headline = "High Risk", "Churn is trending high — act now"
            body = (
                f"Churn rate is <b>{churn_rate:.1f}%</b>, with "
                f"<b>{a_counts['at_risk']:,}</b> customers currently flagged "
                f"<b>at_risk</b>. Consider prioritizing retention offers for this "
                f"segment and reviewing recent support tickets and engagement drops."
            )
        elif churn_rate >= 10:
            tag, headline = "Moderate Risk", "Churn is stable but worth watching"
            body = (
                f"Churn rate sits at <b>{churn_rate:.1f}%</b>. The largest customer "
                f"group is currently <b>{top_archetype.replace('_', ' ')}</b> "
                f"({a_counts[top_archetype]:,} customers). Light-touch retention "
                f"campaigns for the at-risk segment should keep this trend flat."
            )
        else:
            tag, headline = "Healthy", "Churn is low — good time to grow"
            body = (
                f"Churn rate is a healthy <b>{churn_rate:.1f}%</b>. With "
                f"<b>{a_counts['high_value']:,}</b> high-value customers active, "
                f"this is a good window to invest in acquisition and loyalty rewards."
            )
        ai_insight_card(tag, headline, body)
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        st.markdown("##### ⚡ Quick Actions")
        qa1, qa2 = st.columns(2)
        with qa1:
            if st.button("🔄 Refresh Data", use_container_width=True, type="primary"):
                refresh_all(retrain_model=False)
                st.success("Dashboard data refreshed.")
                st.rerun()
        with qa2:
            st.download_button(
                "⬇️ Export Snapshot",
                data=features.to_csv(index=False).encode("utf-8"),
                file_name="customer_snapshot.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with colD:
        st.subheader("🕒 Recent Activity")
        recent_tx = tx.sort_values("transaction_date", ascending=False).head(5)
        items = []
        for _, r in recent_tx.iterrows():
            items.append((
                f"Transaction <b>{r.get('transaction_id', '')}</b> — "
                f"₹{r['amount']:,.0f} in <b>{r.get('category', 'N/A')}</b>",
                pd.to_datetime(r["transaction_date"]).strftime("%d %b %Y"),
            ))
        if items:
            activity_timeline(items)
        else:
            st.info("No transactions recorded yet.")

# ----------------------------------------------------------------------------
# DATASET MANAGEMENT  (Admin only)
# ----------------------------------------------------------------------------
elif page == "Dataset Management":
    page_header("🗂️ Dataset Management", "Generate synthetic data or upload your own CSVs.")

    st.subheader("Generate / Upload Data")
    col1, col2 = st.columns(2)
    with col1:
        n = st.slider("Number of synthetic customers", 200, 5000,
                      st.session_state.n_customers, step=100)
        if st.button("🔄 Regenerate Synthetic Dataset"):
            st.session_state.n_customers = n
            for p in ("data/customers.csv", "data/transactions.csv"):
                if os.path.exists(p):
                    os.remove(p)
            st.session_state.pop("ml_result", None)
            refresh_all(retrain_model=False)
            st.success("Dataset regenerated.")
            st.rerun()
    with col2:
        up_customers = st.file_uploader("Upload customers CSV", type=["csv"])
        up_tx        = st.file_uploader("Upload transactions CSV", type=["csv"])
        if up_customers and up_tx and st.button("📤 Use Uploaded Data"):
            try:
                cust_df = pd.read_csv(up_customers)
                tx_df   = pd.read_csv(up_tx)
                for col, default in [("support_tickets", 0), ("engagement_score", 50.0),
                                     ("email_opt_in", 1), ("satisfaction_rating", 3.5)]:
                    if col not in cust_df.columns:
                        cust_df[col] = default
                cust_df.to_csv("data/customers.csv", index=False)
                tx_df.to_csv("data/transactions.csv",   index=False)
                st.session_state.pop("ml_result", None)
                refresh_all(retrain_model=False)
                st.success("Uploaded dataset loaded and saved to disk successfully.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not load uploaded files: {e}")

    st.divider()
    st.subheader("Preview")
    t1, t2 = st.tabs(["Customers", "Transactions"])
    with t1:
        st.dataframe(customers.head(50), use_container_width=True)
    with t2:
        st.dataframe(tx.head(50), use_container_width=True)

    st.divider()
    st.subheader("🧹 Preprocessing Statistics")
    s = clean_stats
    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("Rows Before",              f"{s['rows_before']:,}")
    cc2.metric("Duplicates Removed",       f"{s['duplicates_removed']:,}")
    cc3.metric("Invalid Records Removed",  f"{s['invalid_removed']:,}")
    cc4.metric("Rows After",               f"{s['rows_after']:,}")
    st.caption(
        f"Outliers capped via IQR method: {s['outliers_capped']:,}  |  "
        f"Missing values found pre-clean: {s['missing_values_before']:,}"
    )

    st.subheader("Customer Feature Table (RFM + CLV)")
    st.dataframe(
        features[["customer_id", "name", "recency_days", "frequency", "monetary",
                  "RFM_score", "RFM_segment", "CLV", "churned"]].head(50),
        use_container_width=True,
    )

# ----------------------------------------------------------------------------
# ADD NEW CUSTOMER  (Admin only)
# ----------------------------------------------------------------------------
elif page == "Add New Customer":
    page_header("➕ Add New Customer")
    st.caption(
        "New customers are automatically assigned the **new** archetype until they "
        "start purchasing — no manual archetype selection needed."
    )

    with st.form("add_customer_form"):
        col1, col2 = st.columns(2)
        with col1:
            name   = st.text_input("Full Name")
            age    = st.number_input("Age", min_value=18, max_value=100, value=30)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        with col2:
            income       = st.number_input("Annual Income", min_value=0.0,
                                           value=50000.0, step=1000.0)
            city         = st.text_input("City")
            email_opt_in = st.checkbox("Email opt-in", value=True)

        submitted = st.form_submit_button("✅ Create Customer", use_container_width=True)
        if submitted:
            if not name.strip() or not city.strip():
                st.error("Please provide both name and city.")
            else:
                try:
                    new_id = add_customer(name.strip(), int(age), gender,
                                          float(income), city.strip(), email_opt_in)
                    refresh_all(retrain_model=False)
                    st.success(
                        f"✅ Customer **{new_id}** created successfully and assigned "
                        f"the **new** archetype. Dashboard and analytics refreshed."
                    )
                except Exception as e:
                    st.error(f"❌ Failed to create customer: {e}")

# ----------------------------------------------------------------------------
# ADD NEW TRANSACTION  (Admin only)
# ----------------------------------------------------------------------------
elif page == "Add New Transaction":
    page_header("➕ Add New Transaction")
    st.caption(
        "Adding a transaction automatically recalculates that customer's purchase "
        "stats, archetype, and refreshes the dashboard, segmentation and churn model."
    )

    customer_choices = features["customer_id"] + " — " + features["name"]
    with st.form("add_transaction_form"):
        selection   = st.selectbox("Customer", customer_choices.tolist())
        customer_id = selection.split(" — ")[0]

        col1, col2 = st.columns(2)
        with col1:
            category   = st.selectbox("Category", ["Electronics", "Fashion", "Grocery",
                                                    "Home & Kitchen", "Beauty", "Sports",
                                                    "Books", "Toys", "Automotive", "Health"])
            product_id = st.text_input("Product ID", value="PRD-001")
            quantity   = st.number_input("Quantity", min_value=1, value=1)
        with col2:
            amount         = st.number_input("Amount (₹)", min_value=1.0,
                                              value=1000.0, step=100.0)
            payment_method = st.selectbox("Payment Method",
                                          ["Credit Card", "Debit Card", "UPI",
                                           "Net Banking", "Cash on Delivery", "Wallet"])
            channel        = st.selectbox("Channel", ["Web", "Mobile App", "In-Store"])

        submitted = st.form_submit_button("✅ Record Transaction", use_container_width=True)
        if submitted:
            try:
                new_txn_id = add_transaction(customer_id, category, product_id.strip(),
                                              int(quantity), float(amount),
                                              payment_method, channel)
                refresh_all(retrain_model=True)

                fresh_tx      = st.session_state.tx
                stats         = customer_purchase_stats(customer_id, fresh_tx)
                new_archetype = st.session_state.features.loc[
                    st.session_state.features["customer_id"] == customer_id, "archetype"
                ]
                st.success(
                    f"✅ Transaction **{new_txn_id}** recorded for **{customer_id}**.\n\n"
                    f"- Total purchases: **{stats['total_purchases']}**\n"
                    f"- Total spending: **₹{stats['total_spending']:,.2f}**\n"
                    f"- Average order value: **₹{stats['avg_order_value']:,.2f}**\n"
                    f"- Last purchase: **{stats['last_purchase_date']}**\n"
                    f"- Updated archetype: **{new_archetype.iloc[0] if len(new_archetype) else 'n/a'}**\n\n"
                    f"Dashboard, segmentation, and churn model have been refreshed."
                )
            except Exception as e:
                st.error(f"❌ Failed to record transaction: {e}")

# ----------------------------------------------------------------------------
# CUSTOMER ARCHETYPES ANALYTICS
# ----------------------------------------------------------------------------
elif page == "Customer Archetypes":
    page_header("🏷️ Customer Archetype Analytics")
    st.caption(
        "Archetypes are recalculated automatically from each customer's signup recency "
        "and purchase history (no manual assignment)."
    )

    a_counts = archetype_counts(features)
    cols  = st.columns(5)
    icons = {"new": "🆕", "regular": "🙂", "loyal": "⭐",
             "at_risk": "⚠️", "high_value": "💎"}
    for col, arch in zip(cols, VALID_ARCHETYPES):
        col.metric(f"{icons[arch]} {arch.replace('_', ' ').title()}", a_counts[arch])

    arch_df = pd.DataFrame({
        "archetype": list(a_counts.keys()),
        "count":     list(a_counts.values()),
    })

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            px.pie(arch_df, names="archetype", values="count",
                   title="Archetype Distribution"),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            px.bar(arch_df, x="archetype", y="count", color="archetype",
                   title="Customers per Archetype"),
            use_container_width=True,
        )

    st.subheader("Archetype Trend (by signup month)")
    trend_df = features.copy()
    trend_df["signup_month"] = (
        pd.to_datetime(trend_df["signup_date"]).dt.to_period("M").astype(str)
    )
    trend = trend_df.groupby(["signup_month", "archetype"]).size().reset_index(name="customers")
    st.plotly_chart(
        px.line(trend, x="signup_month", y="customers", color="archetype", markers=True,
                title="Customers Signed Up per Month, by Current Archetype"),
        use_container_width=True,
    )

    st.subheader("Customers by Archetype")
    arch_filter = st.selectbox("Filter by archetype", ["All"] + VALID_ARCHETYPES)
    table = features if arch_filter == "All" else features[features["archetype"] == arch_filter]
    st.dataframe(
        table[["customer_id", "name", "archetype", "recency_days",
               "frequency", "monetary", "tenure_days"]],
        use_container_width=True,
    )

# ----------------------------------------------------------------------------
# EXPLORATORY DATA ANALYSIS
# ----------------------------------------------------------------------------
elif page == "Exploratory Data Analysis":
    page_header("🔍 Exploratory Data Analysis", "Distributions, trends, and relationships in your data.")

    tx_m = tx.copy()
    tx_m["month"]   = tx_m["transaction_date"].dt.to_period("M").astype(str)
    tx_m["weekday"] = tx_m["transaction_date"].dt.day_name()

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            px.histogram(customers, x="age", nbins=25,
                         title="Customer Age Distribution",
                         color_discrete_sequence=["#4C72B0"]),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            px.pie(customers, names="gender", title="Gender Distribution"),
            use_container_width=True,
        )

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(
            px.histogram(customers, x="income", nbins=30, title="Income Analysis",
                         color_discrete_sequence=["#55A868"]),
            use_container_width=True,
        )
    with col4:
        monthly_sales = tx_m.groupby("month")["amount"].sum().reset_index()
        st.plotly_chart(
            px.bar(monthly_sales, x="month", y="amount", title="Monthly Sales"),
            use_container_width=True,
        )

    col5, col6 = st.columns(2)
    with col5:
        weekly = (
            tx_m.groupby("weekday")["amount"].sum()
            .reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
            .reset_index()
        )
        st.plotly_chart(
            px.bar(weekly, x="weekday", y="amount", title="Weekly Sales Pattern"),
            use_container_width=True,
        )
    with col6:
        cat_sales = tx.groupby("category")["amount"].sum().sort_values(ascending=False).reset_index()
        st.plotly_chart(
            px.bar(cat_sales, x="category", y="amount", title="Category-wise Sales"),
            use_container_width=True,
        )

    col7, col8 = st.columns(2)
    with col7:
        st.plotly_chart(
            px.pie(tx, names="payment_method", title="Payment Methods"),
            use_container_width=True,
        )
    with col8:
        st.plotly_chart(
            px.pie(tx, names="channel", title="Sales Channels"),
            use_container_width=True,
        )

    col9, col10 = st.columns(2)
    with col9:
        st.plotly_chart(
            px.histogram(features, x="frequency", nbins=30,
                         title="Purchase Frequency Distribution",
                         color_discrete_sequence=["#C44E52"]),
            use_container_width=True,
        )
    with col10:
        st.plotly_chart(
            px.pie(features, names="churned", title="Customer Churn Distribution", hole=0.45),
            use_container_width=True,
        )

    st.subheader("Correlation Heatmap")
    num_cols = ["age", "income", "recency_days", "frequency", "monetary",
                "avg_order_value", "CLV", "churned"]
    corr = features[num_cols].corr()
    st.plotly_chart(
        px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                  title="Feature Correlation"),
        use_container_width=True,
    )

    col11, col12 = st.columns(2)
    with col11:
        top_products = (
            tx.groupby("product_id")["amount"].sum()
            .sort_values(ascending=False).head(10).reset_index()
        )
        st.plotly_chart(
            px.bar(top_products, x="product_id", y="amount", title="Top 10 Products"),
            use_container_width=True,
        )
    with col12:
        top_customers = features.sort_values("monetary", ascending=False).head(10)
        st.plotly_chart(
            px.bar(top_customers, x="name", y="monetary",
                   title="Top 10 Customers by Spend"),
            use_container_width=True,
        )

    st.subheader("Customer Location Analysis")
    loc = customers.groupby("city").size().sort_values(ascending=False).reset_index(name="customers")
    st.plotly_chart(
        px.bar(loc, x="city", y="customers", title="Customers by City"),
        use_container_width=True,
    )
    st.info("💡 Tip: use Plotly's built-in camera icon on any chart toolbar to download it as a PNG.")

# ----------------------------------------------------------------------------
# CUSTOMER SEGMENTATION
# ----------------------------------------------------------------------------
elif page == "Customer Segmentation":
    page_header("🧩 Customer Segmentation", "K-Means clustering over recency, frequency, and monetary value.")

    n_clusters = st.slider("Number of clusters (K)", 2, 8, 5)

    with st.spinner("Running K-Means clustering..."):
        ks, inertias, sil_scores = elbow_method(features)
        clustered, sil, model    = run_kmeans(features, n_clusters=n_clusters)
        clustered                = label_clusters(clustered)
        st.session_state.clustered = clustered

    c1, c2 = st.columns(2)
    with c1:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(x=ks, y=inertias, mode="lines+markers", name="Inertia"))
        fig_elbow.update_layout(title="Elbow Method", xaxis_title="K", yaxis_title="Inertia",
                                 template=px.defaults.template)
        st.plotly_chart(fig_elbow, use_container_width=True)
    with c2:
        st.plotly_chart(
            px.scatter(clustered, x="pca_x", y="pca_y", color="segment",
                       title=f"PCA Cluster Visualization (Silhouette = {sil:.3f})",
                       hover_data=["customer_id", "monetary", "frequency"]),
            use_container_width=True,
        )

    st.metric("Silhouette Score", f"{sil:.3f}")

    st.subheader("Cluster / Segment Profiles")
    summary = cluster_summary(clustered)
    st.dataframe(
        summary.style.format({
            "avg_spending":          "₹{:,.2f}",
            "avg_frequency":         "{:.2f}",
            "avg_recency":           "{:.1f} days",
            "revenue_contribution":  "₹{:,.0f}",
            "churn_rate":            "{:.2%}",
        }),
        use_container_width=True,
    )
    st.plotly_chart(
        px.bar(summary, x="segment", y="revenue_share_%",
               title="Revenue Contribution by Segment", color="segment"),
        use_container_width=True,
    )

# ----------------------------------------------------------------------------
# CHURN PREDICTION / ML RESULTS  (Admin only)
# ----------------------------------------------------------------------------
elif page == "Churn Prediction (ML Results)":
    page_header("🤖 Churn Prediction", "Model training & comparison.")

    if st.button("🚀 Train / Retrain Models"):
        with st.spinner("Training Logistic Regression, Decision Tree & Random Forest..."):
            result = train_and_compare(features)
            st.session_state.ml_result = result
            save_model(result["best_model"], result["scaler"])
        st.success(
            f"Training complete. Best model: **{result['best_name']}** "
            f"(ROC-AUC = {result['comparison_df'].loc[result['best_name'], 'roc_auc']:.3f})"
        )

    if "ml_result" in st.session_state:
        result = st.session_state.ml_result
        if result["best_name"] == "Restored Model":
            st.info(
                "ℹ️ A previously trained model was restored from disk. "
                "Click **Train / Retrain Models** above to refresh comparison metrics."
            )
        else:
            st.subheader("Model Comparison")
            st.dataframe(
                result["comparison_df"].style
                    .format("{:.4f}")
                    .highlight_max(subset=["accuracy","precision","recall","f1","roc_auc"],
                                   color="#c6f6d5"),
                use_container_width=True,
            )

            best = result["best_name"]
            st.success(f"🏆 Best performing model: **{best}**")

            cmat = result["results"][best]["confusion_matrix"]
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    px.imshow(cmat, text_auto=True, color_continuous_scale="Blues",
                              x=["Predicted: Stay", "Predicted: Churn"],
                              y=["Actual: Stay",    "Actual: Churn"],
                              title=f"Confusion Matrix — {best}"),
                    use_container_width=True,
                )
            with c2:
                fig_roc = go.Figure()
                for name, r in result["results"].items():
                    fpr, tpr = r["fpr_tpr"]
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr, mode="lines",
                        name=f"{name} (AUC={r['roc_auc']:.3f})",
                    ))
                fig_roc.add_trace(go.Scatter(
                    x=[0,1], y=[0,1], mode="lines", name="Random",
                    line=dict(dash="dash", color="gray"),
                ))
                fig_roc.update_layout(title="ROC Curves",
                                      xaxis_title="False Positive Rate",
                                      yaxis_title="True Positive Rate",
                                      template=px.defaults.template)
                st.plotly_chart(fig_roc, use_container_width=True)

            st.subheader("Feature Importance")
            fi = result["feature_importances"].reset_index()
            fi.columns = ["feature", "importance"]
            st.plotly_chart(
                px.bar(fi, x="importance", y="feature", orientation="h",
                       title=f"Feature Importance — {best}"),
                use_container_width=True,
            )
    else:
        st.info("Click **Train / Retrain Models** to build the churn prediction pipeline.")

# ----------------------------------------------------------------------------
# INDIVIDUAL CUSTOMER LOOKUP
# ----------------------------------------------------------------------------
elif page == "Individual Customer Lookup":
    page_header("🔎 Individual Customer Prediction", "Look up a customer and see their churn risk.")

    search = st.text_input("Search by Customer ID or Name")
    if search:
        matches = features[
            features["customer_id"].str.contains(search, case=False, na=False) |
            features["name"].str.contains(search, case=False, na=False)
        ]
    else:
        matches = features.head(20)

    st.caption(f"{len(matches)} matching customers")
    selected_id = st.selectbox("Select a customer", matches["customer_id"].tolist()) if len(matches) else None

    if selected_id:
        cust    = features[features["customer_id"] == selected_id].iloc[0]
        cust_tx = tx[tx["customer_id"] == selected_id].sort_values("transaction_date", ascending=False)

        col1, col2, col3 = st.columns(3)
        col1.metric("Name",         cust["name"])
        col2.metric("Segment (RFM)", cust["RFM_segment"])
        col3.metric("CLV",          f"₹{cust['CLV']:,.0f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Recency",   f"{int(cust['recency_days'])} days")
        col5.metric("Frequency", f"{int(cust['frequency'])} orders")
        col6.metric("Monetary",  f"₹{cust['monetary']:,.0f}")

        if "ml_result" in st.session_state:
            result = st.session_state.ml_result
            prob   = predict_churn_probability(
                result["best_model"], result["scaler"],
                features[features["customer_id"] == selected_id],
            )
            risk         = risk_level(prob)
            badge_class  = {"Low Risk": "badge-low", "Medium Risk": "badge-medium",
                             "High Risk": "badge-high"}[risk]
            st.markdown(
                f"### Churn Probability: **{prob:.1%}**  "
                f"<span class='{badge_class}'>{risk}</span>",
                unsafe_allow_html=True,
            )
            st.caption(get_risk_recommendation(risk))
        else:
            st.warning(
                "Train the churn model on the **Churn Prediction (ML Results)** page first "
                "to get a live churn probability for this customer."
            )

        st.subheader("Purchase History")
        st.dataframe(
            cust_tx[["transaction_date","category","product_id",
                     "quantity","amount","payment_method","channel"]],
            use_container_width=True,
        )

        st.subheader("Recommended Actions")
        for rec in get_segment_recommendations(cust["RFM_segment"]):
            st.markdown(f"- {rec}")

# ----------------------------------------------------------------------------
# BUSINESS RECOMMENDATIONS
# ----------------------------------------------------------------------------
elif page == "Business Recommendations":
    page_header("💡 Business Recommendations", "Suggested actions for each customer segment.")

    seg_counts = features["RFM_segment"].value_counts()
    for seg in seg_counts.index:
        with st.expander(f"**{seg}** — {seg_counts[seg]} customers"):
            for rec in get_segment_recommendations(seg):
                st.markdown(f"- {rec}")

# ----------------------------------------------------------------------------
# EMAIL ALERTS  (Admin only)
# ----------------------------------------------------------------------------
elif page == "Email Alerts":
    page_header("📧 Email Alert System")
    st.caption(
        "Automatically sends email alerts when a customer transitions to **at_risk**. "
        "Gmail recommended — use an **App Password**, not your main password."
    )

    cfg = load_email_config()

    # ── Configuration form ──────────────────────────────────────────────────
    st.subheader("⚙️ Email Configuration")
    with st.form("email_config_form"):
        col1, col2 = st.columns(2)
        with col1:
            sender_email    = st.text_input("Sender Email (Gmail)",
                                             value=cfg.get("sender_email",""))
            sender_password = st.text_input("App Password",
                                             value=cfg.get("sender_password",""),
                                             type="password",
                                             help="Gmail: Settings → Security → 2-Step Verification → App Passwords")
            recipient_email = st.text_input("Recipient Email (who gets the alerts)",
                                             value=cfg.get("recipient_email",""))
        with col2:
            smtp_host    = st.text_input("SMTP Host",  value=cfg.get("smtp_host","smtp.gmail.com"))
            smtp_port    = st.number_input("SMTP Port", value=int(cfg.get("smtp_port", 587)),
                                            min_value=1, max_value=65535)
            alerts_enabled = st.checkbox("Enable automatic at-risk alerts",
                                          value=cfg.get("alerts_enabled", False))

        save_cfg = st.form_submit_button("💾 Save Configuration", use_container_width=True)
        if save_cfg:
            new_cfg = {
                "smtp_host":       smtp_host,
                "smtp_port":       smtp_port,
                "sender_email":    sender_email,
                "sender_password": sender_password,
                "recipient_email": recipient_email,
                "alerts_enabled":  alerts_enabled,
            }
            save_email_config(new_cfg)
            cfg = new_cfg
            st.success("✅ Email configuration saved.")

    # ── Status banner ────────────────────────────────────────────────────────
    if config_is_complete(cfg):
        st.success(f"✅ Alerts are **enabled**. Alerts will be sent to: `{cfg['recipient_email']}`")
    else:
        st.warning("⚠️ Email alerts are **disabled**. Fill in all fields above and enable alerts.")

    st.divider()

    # ── Test email ───────────────────────────────────────────────────────────
    st.subheader("🧪 Send Test Alert")
    st.caption("Sends a sample alert using the first at-risk customer in the dataset.")
    if st.button("Send Test Email", use_container_width=False):
        at_risk_customers = features[features["archetype"] == "at_risk"]
        if at_risk_customers.empty:
            st.info("No at-risk customers found in the current dataset.")
        elif not config_is_complete(cfg):
            st.error("Please save a complete email configuration first.")
        else:
            test_customer = at_risk_customers.iloc[0].to_dict()
            ok, msg = send_alert_email(test_customer, cfg)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    st.divider()

    # ── At-risk summary ──────────────────────────────────────────────────────
    at_risk_df = features[features["archetype"] == "at_risk"][
        ["customer_id", "name", "recency_days", "frequency", "monetary", "CLV", "RFM_segment"]
    ].copy()

    st.subheader(f"⚠️ Current At-Risk Customers ({len(at_risk_df)})")
    if at_risk_df.empty:
        st.info("No at-risk customers right now.")
    else:
        st.dataframe(at_risk_df, use_container_width=True)

        col1, col2 = st.columns([1, 3])
        with col1:
            bulk_limit = st.number_input(
                "Number of customers to email (start small to test!)",
                min_value=1, max_value=len(at_risk_df),
                value=min(5, len(at_risk_df)),
                help="Gmail can rate-limit or flag bulk sends. Test with a "
                     "small number first before sending to everyone.",
            )
            send_all = st.checkbox(
                f"Send to ALL {len(at_risk_df)} at-risk customers instead",
            )
            if st.button("📨 Send Bulk Alerts", use_container_width=True):
                if not config_is_complete(cfg):
                    st.error("Configure email settings first.")
                else:
                    n_to_send = len(at_risk_df) if send_all else int(bulk_limit)
                    progress_bar = st.progress(0, text="Starting...")

                    def _update_progress(sent, failed, total):
                        progress_bar.progress(
                            (sent + failed) / total,
                            text=f"Sent: {sent}  Failed: {failed}  /  {total}",
                        )

                    sent, failed, messages = send_bulk_alerts(
                        features, cfg,
                        limit=n_to_send,
                        delay_seconds=1.5,
                        progress_callback=_update_progress,
                    )
                    progress_bar.empty()
                    st.success(f"✅ Sent: {sent}  |  ❌ Failed: {failed}")
                    with st.expander("Details"):
                        for m in messages:
                            st.write(m)

        # individual send buttons
        with st.expander("Send alert for a specific customer"):
            sel = st.selectbox(
                "Select customer",
                at_risk_df["customer_id"] + " — " + at_risk_df["name"],
            )
            if st.button("Send Alert for Selected Customer"):
                cid = sel.split(" — ")[0]
                row = features[features["customer_id"] == cid].iloc[0].to_dict()
                ok, msg = send_alert_email(row, cfg)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

    st.divider()

    # ── Alert history ────────────────────────────────────────────────────────
    st.subheader("📋 Alert History")
    log = load_alert_log()
    if log.empty:
        st.info("No alerts have been sent yet.")
    else:
        # summary metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Alerts Sent",   len(log))
        m2.metric("Successful",          int((log["status"] == "sent").sum()))
        m3.metric("Failed",              int((log["status"] == "failed").sum()))
        st.dataframe(
            log.sort_values("sent_at", ascending=False),
            use_container_width=True,
        )
        st.download_button(
            "⬇️ Download Alert Log (CSV)",
            log.to_csv(index=False).encode("utf-8"),
            file_name="alert_log.csv", mime="text/csv",
        )

# ----------------------------------------------------------------------------
# REPORTS  (Admin only)
# ----------------------------------------------------------------------------
elif page == "Reports":
    page_header("📑 Reports", "Export summaries as PDF, Excel, or CSV.")

    report_choice = st.selectbox("Select report", [
        "Customer Summary Report", "Segment Report", "Churn Report",
        "Sales Report", "Revenue Report", "Business Insights Report",
    ])

    if report_choice == "Customer Summary Report":
        df = features[["customer_id","name","age","gender","city","recency_days",
                        "frequency","monetary","CLV","RFM_segment","churned"]]
        chart_df = features["RFM_segment"].value_counts().reset_index()
        chart_df.columns = ["segment", "customers"]
        chart = (make_bar_chart_image(chart_df, "segment", "customers", "Customers by Segment"),
                 "Customer count by RFM segment")

    elif report_choice == "Segment Report":
        df = cluster_summary(st.session_state.get(
            "clustered", features.assign(cluster=0, segment=features["RFM_segment"])
        ))
        chart = (make_bar_chart_image(df, "segment", "revenue_contribution", "Revenue by Segment"),
                 "Revenue contribution by segment")

    elif report_choice == "Churn Report":
        df = features[["customer_id","name","recency_days","frequency","monetary",
                        "RFM_segment","churned"]]
        chart_df = features.groupby("RFM_segment")["churned"].mean().reset_index()
        chart_df["churn_rate_%"] = (chart_df["churned"] * 100).round(1)
        chart = (make_bar_chart_image(chart_df, "RFM_segment", "churn_rate_%",
                                       "Churn Rate by Segment"),
                 "Churn rate (%) by segment")

    elif report_choice == "Sales Report":
        df = tx.groupby("category").agg(
            orders=("transaction_id", "count"), revenue=("amount", "sum")
        ).reset_index()
        chart = (make_bar_chart_image(df, "category", "revenue", "Revenue by Category"),
                 "Revenue by product category")

    elif report_choice == "Revenue Report":
        df = tx.set_index("transaction_date").resample("ME")["amount"].sum().reset_index()
        df.columns = ["month", "revenue"]
        chart = (make_bar_chart_image(df, "month", "revenue", "Monthly Revenue"),
                 "Revenue trend by month")

    else:  # Business Insights Report
        df = pd.DataFrame({
            "metric": ["Total Customers","Total Revenue","Churn Rate","Avg CLV","Avg Order Value"],
            "value":  [
                len(features),
                f"₹{tx['amount'].sum():,.0f}",
                f"{features['churned'].mean():.1%}",
                f"₹{features['CLV'].mean():,.0f}",
                f"₹{tx['amount'].mean():,.2f}",
            ],
        })
        chart_df = features["RFM_segment"].value_counts().reset_index()
        chart_df.columns = ["segment", "customers"]
        chart = (make_bar_chart_image(chart_df, "segment", "customers", "Customer Segments"),
                 "Customer distribution across segments")

    st.dataframe(df, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "⬇️ Download CSV", to_csv_bytes(df),
            file_name=f"{report_choice.replace(' ','_')}.csv", mime="text/csv",
        )
    with c2:
        st.download_button(
            "⬇️ Download Excel", to_excel_bytes({report_choice[:31]: df}),
            file_name=f"{report_choice.replace(' ','_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with c3:
        try:
            pdf_bytes = to_pdf_bytes(report_choice, [(report_choice, df.head(40))], chart=chart)
            st.download_button(
                "⬇️ Download PDF", pdf_bytes,
                file_name=f"{report_choice.replace(' ','_')}.pdf", mime="application/pdf",
            )
        except Exception as e:
            st.caption(f"PDF export unavailable: {e}")

# ----------------------------------------------------------------------------
# CHANGE PASSWORD  (available to Admin and User)
# ----------------------------------------------------------------------------
elif page == "Change Password":
    page_header("🔑 Change Password")
    st.caption(f"Update the password for account: **{st.session_state.auth.get('username')}**")

    with st.form("change_password_form", clear_on_submit=True):
        current_password = st.text_input("Current password", type="password")
        new_password = st.text_input("New password", type="password")
        confirm_password = st.text_input("Confirm new password", type="password")
        submitted = st.form_submit_button("Update Password", type="primary")

    if submitted:
        users = load_users()
        username = st.session_state.auth.get("username")
        user_record = users.get(username)

        if not user_record or user_record["password"] != current_password:
            st.error("Current password is incorrect.")
        elif not new_password:
            st.error("New password cannot be empty.")
        elif len(new_password) < 6:
            st.error("New password must be at least 6 characters long.")
        elif new_password != confirm_password:
            st.error("New password and confirmation do not match.")
        elif new_password == current_password:
            st.warning("New password must be different from the current password.")
        else:
            users[username]["password"] = new_password
            save_users(users)
            st.success("Password updated successfully. Use your new password next time you sign in.")

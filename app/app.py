import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.sparse import hstack, csr_matrix


# PAGE CONFIG

st.set_page_config(
    page_title="SafeApply",
    page_icon="🛡️",
    layout="wide",
   initial_sidebar_state="auto"
)

st.markdown("""
    <style>
    /* ── Main background ── */
    .stApp { background-color: #0f1117; }

    /* ── Result boxes ── */
    .result-fake {
        background: linear-gradient(135deg, #c0392b, #e74c3c);
        color: white; padding: 1.6rem 1rem; border-radius: 14px;
        text-align: center; font-size: 1.4rem; font-weight: 800;
        margin: 0.8rem 0; letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(231,76,60,0.4);
    }
    .result-real {
        background: linear-gradient(135deg, #1e8449, #27ae60);
        color: white; padding: 1.6rem 1rem; border-radius: 14px;
        text-align: center; font-size: 1.4rem; font-weight: 800;
        margin: 0.8rem 0; letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(39,174,96,0.4);
    }

    /* ── Info cards ── */
    .info-card {
        background: #1a1d27; border: 1px solid #2d3148;
        border-radius: 12px; padding: 1rem 1.2rem; margin: 0.4rem 0;
    }

    /* ── Layer badge ── */
    .layer-badge {
        display: inline-block; padding: 3px 10px;
        border-radius: 20px; font-size: 0.78rem; font-weight: 600;
        margin-bottom: 4px;
    }

    /* ── Page header ── */
    .page-header {
        text-align: center; padding: 1.2rem 0 0.5rem 0;
        font-size: 2.3rem; font-weight: 800; color: #e8eaf6;
        letter-spacing: -0.5px;
    }
    .page-sub {
        text-align: center; color: #9e9e9e;
        font-size: 1rem; margin-bottom: 1.2rem;
    }

    /* ── Metric cards ── */
    div[data-testid="metric-container"] {
        background: #1a1d27; border: 1px solid #2d3148;
        border-radius: 10px; padding: 0.6rem;
    }

   /* ── Hide streamlit chrome ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ── Sidebar toggle — force visible across all Streamlit versions ── */
    [data-testid="collapsedControl"] {
        visibility: visible !important;
        display: flex !important;
        opacity: 1 !important;
    }

    /* Hide other header elements but NOT the toggle ── */
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 2.5rem;
    }
    header[data-testid="stHeader"] > div:not(:has([data-testid="collapsedControl"])) {
        visibility: hidden;
    }


    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #1a1d2e;
        border-right: 2px solid #3d4172;
    }

    /* ── Divider ── */
    hr { border-color: #2d3148; }
    </style>
""", unsafe_allow_html=True)



# LOAD MODEL

@st.cache_resource
def load_model_and_vectorizer():
    model_paths = ['../models/best_model.pkl',       'models/best_model.pkl',       'best_model.pkl']
    tfidf_paths = ['../models/tfidf_vectorizer.pkl', 'models/tfidf_vectorizer.pkl', 'tfidf_vectorizer.pkl']
    info_paths  = ['../models/model_info.json',      'models/model_info.json',      'model_info.json']

    model, tfidf, info = None, None, {}
    for p in model_paths:
        if os.path.exists(p): model = joblib.load(p); break
    for p in tfidf_paths:
        if os.path.exists(p): tfidf = joblib.load(p); break
    for p in info_paths:
        if os.path.exists(p):
            with open(p) as f: info = json.load(f); break
    return model, tfidf, info


# TEXT CLEANING — identical to notebook cell 28

def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)           # remove HTML tags
    text = re.sub(r'http\S+|www\S+', ' ', text)  # remove URLs
    text = re.sub(r'[^a-z0-9\s]', ' ', text)     # remove special chars (₹, $, !, etc.)
    text = re.sub(r'\s+', ' ', text).strip()      # remove extra whitespace
    return text



# PREDICTION — exact same pipeline as training

def predict_job_posting(title, company_profile, description,
                         requirements, benefits,
                         has_logo, telecommuting, has_questions,
                         model, tfidf):
    """
    Three-layer hybrid fraud detection:
    Layer 1 — Rule-based: fee/financial fraud → instant FAKE
    Layer 2 — Scam counter: 3+ patterns OR 2+ patterns + ML suspicious → FAKE
    Layer 3 — ML model: XGBoost on 20,005 features, threshold=0.35
    """

    # ── Step 1: Combine text  ──
    combined = (
        str(title)           + ' ' +
        str(company_profile) + ' ' +
        str(description)     + ' ' +
        str(requirements)    + ' ' +
        str(benefits)
    )

    # ── Step 2: Clean text  ──
    cleaned = clean_text(combined)

    # ── Step 3: TF-IDF vectorization (same vectorizer from training) ──
    text_vector = tfidf.transform([cleaned])

    # ── Step 4: Engineered features ) ──
    # df[['text_length', 'word_count', 'has_logo', 'telecommute', 'has_questions']]
    text_length = len(cleaned)
    word_count  = len(cleaned.split())
    extra = csr_matrix([[
        text_length,
        word_count,
        int(has_logo),
        int(telecommuting),
        int(has_questions)
    ]])

    # ── Step 5: Combine TF-IDF + extra features ) ──
    X = hstack([text_vector, extra])

    # ── Step 6: Get fraud probability from ML model ──
    proba = float(model.predict_proba(X)[0][1])

    # ── Step 7: Rule-based detection (Layer 1) ──
    # Only 100% certain fraud indicators — no real employer ever uses these
    text_lower = combined.lower()

    critical_keywords = [
        # Fee fraud — real employers NEVER charge candidates
        'registration fee', 'joining fee',  'training fee',
        'processing fee',   'deposit fee',  'pay to start',
        'fee to apply',     'upfront fee',
        # Financial fraud — no legitimate job uses these
        'western union', 'wire transfer', 'money transfer',
    ]
    critical_hit = any(kw in text_lower for kw in critical_keywords)

    # ── Step 8: Scam pattern counter (for Layer 2) ──
    # These phrases individually may appear in real jobs but
    # 2+ together significantly increases fraud probability
    obvious_scam_phrases = [
        'no experience required', 'no experience needed',
        'no qualification',       'no skills required',
        'earn weekly',            'earn daily',
        'earn from home',         'work from home',
        'work from mobile',       'guaranteed income',
        'guaranteed salary',      'immediate joining',
        'urgent hiring',          'whatsapp',
        'send your details',      'typing job',
        'copy paste',             'form filling',
        'data entry work from home',
        'lakh per month',         'lakhs per month',
        '1 hour',                 '2 hours',
        'no interview',           'direct joining',
    ]
    scam_count = sum(1 for phrase in obvious_scam_phrases if phrase in text_lower)

    # ── Step 9: Hybrid decision ──
    # Layer 1: Fee/financial fraud — 100% certain, override ML
    if critical_hit:
        prediction = 1
        proba = max(proba, 0.90)

    # Layer 2a: 3+ scam patterns = very high confidence fake
    elif scam_count >= 3:
        prediction = 1
        proba = max(proba, 0.75)

    # Layer 2b: 2 scam patterns AND ML also flags as suspicious
    elif scam_count >= 2 and proba >= 0.20:
        prediction = 1
        proba = max(proba, 0.70)

    # Layer 3: ML model alone (threshold=0.35, lower than default 0.5)
    # Lower threshold = higher Recall = catch more frauds
    # Acceptable trade-off: slightly more false alarms vs missing real fraud
    elif proba >= 0.35:
        prediction = 1

    else:
        prediction = 0

    # ── Step 10: Warning flags (informational — do NOT change prediction) ──
    # These are displayed to help user make their own informed decision
    warning_flags = []
    suspicious_phrases = [
        # Generic
        ('work from home',           'Promotes remote work with no skills — common scam pattern'),
        ('no experience required',   'Legitimate jobs usually require some experience or qualification'),
        ('no experience needed',     'Legitimate jobs usually require some experience or qualification'),
        ('no skills required',       'Every real job needs some skills or qualification'),
        ('earn money fast',          'No legitimate company promises fast earnings'),
        ('guaranteed income',        'No job can guarantee income — suspicious promise'),
        ('guaranteed salary',        'Unconditional salary promise — suspicious'),
        ('send your details',        'Real companies use formal application portals, not this phrase'),
        ('whatsapp',                 'Professional recruiters use email/portals, not WhatsApp'),
        # Fee fraud
        ('upfront fee',              '🚨 Real employers NEVER charge upfront fees'),
        ('registration fee',         '🚨 Real employers NEVER charge registration fees'),
        ('training fee',             '🚨 Real employers NEVER charge training fees'),
        ('processing fee',           '🚨 Real employers NEVER charge processing fees'),
        ('joining fee',              '🚨 Real employers NEVER charge joining fees'),
        ('deposit fee',              '🚨 Real employers NEVER charge deposit fees'),
        ('pay to start',             '🚨 You should NEVER pay money to get a job'),
        ('fee to apply',             '🚨 Applying for jobs is always free'),
        # Financial fraud
        ('western union',            '🚨 Financial fraud indicator — never send money via Western Union'),
        ('wire transfer',            '🚨 Financial scam — no real employer asks for wire transfers'),
        ('money transfer',           '🚨 Suspicious — no real employer asks you to transfer money'),
        # Unrealistic promises
        ('earn weekly',              'Legitimate jobs pay monthly with proper payslips, not weekly'),
        ('earn daily',               'No salaried job pays daily — suspicious earning promise'),
        ('earn from home',           'Vague earning promise with no real job details'),
        ('daily income',             'Vague daily income promise — typical scam language'),
        ('weekly income',            'Vague weekly income promise — typical scam language'),
        # Indian scam patterns
        ('lakh per month',           'Unrealistically high salary with no skills required'),
        ('lakhs per month',          'Unrealistically high salary with no skills required'),
        ('work from mobile',         'Mobile-based scam — no real professional job works from just a mobile'),
        ('typing job',               'Common fake job type with no real employer or contract'),
        ('copy paste',               'Not real employment — extremely common Indian scam job type'),
        ('data entry work from home','Very common fake job pattern — rarely a legitimate opportunity'),
        # Pressure tactics
        ('immediate joining',        'Pressure tactic — real companies have proper notice periods'),
        ('urgent hiring',            'Pressure tactic — legitimate companies take time to hire properly'),
        ('direct joining',           'Pressure tactic — skipping normal interview and hiring process'),
        ('no interview',             'Every legitimate company interviews candidates before hiring'),
    ]

    for phrase, reason in suspicious_phrases:
        if phrase in text_lower:
            warning_flags.append(f'⚠️ Found "{phrase}": {reason}')

    return int(prediction), proba, warning_flags, scam_count



# GAUGE CHART

def draw_gauge(probability):
    fig, ax = plt.subplots(figsize=(5, 2.2))
    fig.patch.set_facecolor('#1a1d27')
    ax.set_facecolor('#1a1d27')

    ax.barh(0, 1.0, height=0.5, color='#2d3148', left=0)

    if probability < 0.35:
        bar_color, label = '#27ae60', 'LOW RISK'
    elif probability < 0.65:
        bar_color, label = '#f39c12', 'MEDIUM RISK'
    else:
        bar_color, label = '#e74c3c', 'HIGH RISK'

    ax.barh(0, probability, height=0.5, color=bar_color, left=0)
    ax.text(0.5, 0, f'{probability:.1%}', ha='center', va='center',
            fontsize=18, fontweight='bold', color='white', zorder=5)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.7)
    ax.axis('off')
    ax.set_title(f'Fraud Probability: {label}', fontsize=11,
                 fontweight='bold', color=bar_color, pad=8)
    plt.tight_layout()
    return fig



# SIDEBAR

with st.sidebar:
    st.markdown("## 🛡️ SafeApply")
    st.markdown("*Your First Line of Defense Against Job Scams*")
    st.markdown("---")

    page = st.radio(
        "Navigate:",
        ["🏠 Single Job Check", "📊 Bulk CSV Check", "📈 Model Performance", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    model, tfidf, model_info = load_model_and_vectorizer()

    if model is not None:
        st.success("✅ Model Loaded")
        st.caption(f"**Model:** {model_info.get('best_model_name','XGBoost')}")
        st.caption(f"**F1 Score:** {model_info.get('f1_score','0.8359')}")
        st.caption(f"**ROC-AUC:** {model_info.get('roc_auc','0.9887')}")
        st.caption(f"**Recall:** {model_info.get('recall','0.7803')}")
    else:
        st.error("❌ Model Not Found")
        st.warning("Place `best_model.pkl` and `tfidf_vectorizer.pkl` in the `models/` folder.")


    st.markdown("---")
    st.caption("Dataset: EMSCAD (Kaggle)")
    st.caption("Built with Python · XGBoost · Streamlit")
    st.caption("GitHub: [saloni-78](https://github.com/saloni-78)")



# PAGE 1: SINGLE JOB CHECK

if "🏠 Single Job Check" in page:

    st.markdown('<div class="page-header">🛡️ SafeApply - Trust Before You Apply</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Paste any job posting below to check if it is real or fraudulent</div>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown("#### 📝 Enter Job Details")

        title = st.text_input(
            "Job Title ✱",
            placeholder="e.g., Senior Software Engineer ",
            help="Enter the exact job title from the posting"
        )

        company_profile = st.text_area(
            "Company Profile",
            placeholder="Enter company background, mission, what they do...\n(Leave blank if not provided — missing profile is itself a red flag)",
            height=90,
            help="Missing company profile is a common sign of fake jobs"
        )

        description = st.text_area(
            "Job Description ✱",
            placeholder="Paste the full job description here...\nThe more text you paste, the more accurate the result.",
            height=160,
            help="Most important field — TF-IDF extracts 20,000 language features from this"
        )

        col_r, col_b = st.columns(2)
        with col_r:
            requirements = st.text_area(
                "Requirements",
                placeholder="e.g., 3+ years Python, B.Tech degree...",
                height=80
            )
        with col_b:
            benefits = st.text_area(
                "Benefits",
                placeholder="e.g., Health insurance, stock options...",
                height=80
            )

        st.markdown("**About this posting:**")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            has_logo = st.checkbox(
                "Has company logo?",
                value=False,
                help="Check ONLY if the actual posting shows a verified company logo. "
                     "Leave unchecked if unsure — most fake jobs do not have logos."
            )
        with col_b:
            telecommuting = st.checkbox(
                "Remote / WFH?",
                value=False,
                help="Check if this is advertised as a remote work position"
            )
        with col_c:
            has_questions = st.checkbox(
                "Has screening questions?",
                value=False,
                help="Real companies almost always include screening questions in their applications"
            )

        predict_btn = st.button("🔍 Analyze This Job Posting", type="primary", use_container_width=True)

    with col2:
        st.markdown("#### 📊 Analysis Result")

        if predict_btn:
            if not title or not description:
                st.error("⚠️ Please fill in at least **Job Title** and **Job Description**")
            elif model is None:
                st.error("❌ Model not loaded. Check that model files are in the `models/` folder.")
            else:
                with st.spinner("Analyzing job posting..."):
                    prediction, probability, flags, scam_count = predict_job_posting(
                        title, company_profile, description,
                        requirements, benefits,
                        has_logo, telecommuting, has_questions,
                        model, tfidf
                    )

                # Result box
                if prediction == 1:
                    st.markdown(
                        f'<div class="result-fake">🚨 LIKELY FAKE JOB<br>'
                        f'<span style="font-size:1rem;font-weight:400;">Fraud Probability: {probability:.1%}</span></div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="result-real">✅ LIKELY REAL JOB<br>'
                        f'<span style="font-size:1rem;font-weight:400;">Fraud Probability: {probability:.1%}</span></div>',
                        unsafe_allow_html=True
                    )

                # Gauge
                gauge_fig = draw_gauge(probability)
                st.pyplot(gauge_fig, use_container_width=True)
                plt.close()

                # Detection reason
                if critical_hit := any(kw in (title+' '+description).lower() for kw in [
                    'registration fee','joining fee','training fee','processing fee',
                    'deposit fee','pay to start','fee to apply','upfront fee',
                    'western union','wire transfer','money transfer'
                ]):
                    st.error("🔴 **Layer 1:** Fee/financial fraud keyword detected — instant flag")
                elif scam_count >= 3:
                    st.error(f"🟡 **Layer 2:** {scam_count} scam patterns found — high confidence fake")
                elif scam_count >= 2 and probability >= 0.20:
                    st.warning(f"🟠 **Layer 2+3:** {scam_count} scam patterns + ML flagged — flagged as fake")
                elif prediction == 1:
                    st.warning(f"🟢 **Layer 3:** ML model detected fraud (probability {probability:.1%})")
                else:
                    st.success("🟢 **Layer 3:** ML model found no strong fraud signals")

                # Warning flags
                if flags:
                    st.markdown(f"**🚩 Warning Signs Found ({len(flags)}):**")
                    for flag in flags:
                        st.warning(flag)
                    if scam_count >= 2:
                        st.error(f"⛔ {scam_count} scam patterns detected simultaneously — strong fraud signal")
                else:
                    st.success("✅ No suspicious phrases detected in text")

                # Interpretation
                st.markdown("**📖 Verdict:**")
                if probability < 0.35:
                    st.info("🟢 Low fraud risk — this looks like a legitimate job posting. Still research the company before sharing personal details.")
                elif probability < 0.65:
                    st.warning("🟡 Medium risk — proceed with caution. Research the company independently before applying.")
                else:
                    st.error("🔴 High fraud risk — do NOT share personal information, bank details, or pay any fees.")

        else:
            st.markdown("""
            <div style="background:#1a1d27;border:1px solid #2d3148;border-radius:12px;padding:1.2rem;margin-top:0.5rem;">
            <p style="color:#9e9e9e;margin:0 0 0.8rem 0;">👈 Fill in job details and click <strong>Analyze</strong></p>
            <p style="color:#e8eaf6;font-size:0.9rem;margin:0 0 0.5rem 0;"><strong>What we check:</strong></p>
            <ul style="color:#9e9e9e;font-size:0.85rem;margin:0;padding-left:1.2rem;">
            <li>TF-IDF language patterns (10,000 features + bigrams)</li>
            <li>Text length and word count</li>
            <li>Company logo and screening questions</li>
            <li>35 scam phrase patterns</li>
            <li>11 fee/financial fraud keywords</li>
            <li>Multi-pattern scam detection</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)



# PAGE 2: BULK CSV

elif "📊 Bulk CSV Check" in page:

    st.markdown('<div class="page-header">📊 Bulk Job Checker</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Upload a CSV file to analyze multiple job postings at once</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("#### 📋 Required CSV Format")
    example_df = pd.DataFrame({
        'title':            ['Data Scientist at TechCorp', 'Form Filling Work From Home'],
        'company_profile':  ['We are a product-based tech firm founded in 2015...', ''],
        'description':      ['Looking for ML engineer with 3+ years Python experience...', 'Earn 50000 weekly. No skills needed. WhatsApp us now.'],
        'requirements':     ['Python, SQL, 3+ years experience, B.Tech', ''],
        'benefits':         ['Health insurance, stock options, 25 days leave', ''],
        'has_company_logo': [1, 0],
        'telecommuting':    [0, 1],
        'has_questions':    [1, 0],
    })
    st.dataframe(example_df, use_container_width=True)

    col_dl, _ = st.columns([1, 3])
    with col_dl:
        csv_bytes = example_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Example CSV", data=csv_bytes,
                           file_name="example_jobs.csv", mime="text/csv")

    st.markdown("---")
    st.markdown("#### 📁 Upload Your File")
    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=['csv'],
        label_visibility="collapsed",
        help="CSV must have 'title' and 'description' columns. All others are optional."
    )

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded **{len(input_df):,}** job postings")
            with st.expander("Preview uploaded data"):
                st.dataframe(input_df.head(5), use_container_width=True)

            if model is None:
                st.error("❌ Model not loaded!")
            else:
                if st.button("🚀 Analyze All Jobs", type="primary", use_container_width=True):
                    results_list = []
                    progress_bar = st.progress(0)
                    status_text  = st.empty()

                    for idx, row in input_df.iterrows():
                        progress_bar.progress((idx + 1) / len(input_df))
                        status_text.text(f"Analyzing {idx+1} of {len(input_df)}...")

                        pred, prob, _, _ = predict_job_posting(
                            str(row.get('title', '')),
                            str(row.get('company_profile', '')),
                            str(row.get('description', '')),
                            str(row.get('requirements', '')),
                            str(row.get('benefits', '')),
                            int(row.get('has_company_logo', 0)),
                            int(row.get('telecommuting', 0)),
                            int(row.get('has_questions', 0)),
                            model, tfidf
                        )

                        results_list.append({
                            'Job Title':         str(row.get('title', ''))[:60],
                            'Prediction':        '🚨 FAKE' if pred == 1 else '✅ REAL',
                            'Fraud Probability': f'{prob:.1%}',
                            'Risk Level':        '🔴 HIGH' if prob > 0.65 else '🟡 MEDIUM' if prob > 0.35 else '🟢 LOW'
                        })

                    status_text.text("✅ Analysis complete!")
                    progress_bar.progress(1.0)

                    results_df = pd.DataFrame(results_list)
                    total      = len(results_df)
                    fake_count = (results_df['Prediction'] == '🚨 FAKE').sum()
                    real_count = total - fake_count

                    st.markdown("---")
                    st.markdown("#### 📊 Summary")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Analyzed", total)
                    m2.metric("✅ Real Jobs", real_count)
                    m3.metric("🚨 Suspicious", fake_count)
                    m4.metric("Fraud Rate", f"{fake_count/total:.1%}" if total > 0 else "0%")

                    st.markdown("#### 📋 Detailed Results")
                    st.dataframe(results_df, use_container_width=True)

                    out_csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Download Results CSV", data=out_csv,
                                       file_name="fraud_detection_results.csv", mime="text/csv")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            st.info("Make sure your CSV has 'title' and 'description' columns.")



# PAGE 3: MODEL PERFORMANCE

elif "📈 Model Performance" in page:

    st.markdown('<div class="page-header">📈 Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Complete training details and model comparison</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Dataset overview
    st.markdown("#### 📊 Dataset Overview")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Total Rows",  "17,880")
    d2.metric("Real Jobs",   "17,014")
    d3.metric("Fake Jobs",   "866")
    d4.metric("Train / Test","80% / 20%")

    st.warning(
        "⚠️ **Class Imbalance:** Only 4.8% of jobs are fake (866 out of 17,880). "
        "Solved using **class_weight='balanced'** in all models — "
        "this tells each model to treat fake jobs as ~20x more important during training."
    )

    st.markdown("---")

    # All models results table
    st.markdown("#### 🤖 All 3 Models — Comparison")
    results_data = {
        "Model":     ["Logistic Regression", "Random Forest", "XGBoost ✅ Best"],
        "Accuracy":  ["0.9628", "0.9648", "0.9852"],
        "Precision": ["0.5714", "0.6218", "0.9000"],
        "Recall":    ["0.9249", "0.6936", "0.7803"],
        "F1 Score":  ["0.7064", "0.6557", "0.8359"],
        "ROC-AUC":   ["0.9929", "0.9789", "0.9887"],
    }
    st.dataframe(pd.DataFrame(results_data), use_container_width=True, hide_index=True)
    st.caption("✅ XGBoost selected — highest F1 Score (0.8359) with strong Precision (0.9000) and Recall (0.7803)")

    st.markdown("---")

    # Best model highlight
    st.markdown("#### 🏆 XGBoost — Best Model")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",  "0.9852", help="Overall correct predictions — but misleading for imbalanced data")
    c2.metric("Precision", "0.9000", help="Of all jobs flagged as fake, 90% were actually fake")
    c3.metric("Recall",    "0.7803", delta="Key metric", help="Of all actual fake jobs, 78% were caught")
    c4.metric("F1 Score",  "0.8359", help="Harmonic mean of Precision and Recall — best overall metric")
    c5.metric("ROC-AUC",   "0.9887", help="Area under ROC curve — 1.0 = perfect, 0.5 = random")

    st.markdown("---")

    # Bar chart
    st.markdown("#### 📊 Visual Comparison")
    models_list  = ["LR", "RF", "XGB"]
    metrics_dict = {
        "Accuracy":  [0.9628, 0.9648, 0.9852],
        "Precision": [0.5714, 0.6218, 0.9000],
        "Recall":    [0.9249, 0.6936, 0.7803],
        "F1 Score":  [0.7064, 0.6557, 0.8359],
        "ROC-AUC":   [0.9929, 0.9789, 0.9887],
    }
    colors_bar = ["#3498db", "#e74c3c", "#2ecc71"]
    fig, axes  = plt.subplots(1, 5, figsize=(16, 4))
    fig.patch.set_facecolor('#1a1d27')

    for ax, (metric, vals) in zip(axes, metrics_dict.items()):
        ax.set_facecolor('#1a1d27')
        bars = ax.bar(models_list, vals, color=colors_bar, edgecolor='#2d3148', linewidth=0.8)
        ax.set_title(metric, fontweight="bold", fontsize=10, color='white', pad=6)
        ax.set_ylim(0, 1.18)
        ax.grid(axis="y", alpha=0.2, color='white')
        ax.tick_params(colors='white', labelsize=8)
        for spine in ax.spines.values(): spine.set_edgecolor('#2d3148')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", fontsize=7, fontweight="bold", color='white')
        best_idx = vals.index(max(vals))
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(2.5)

    plt.suptitle("Model Comparison — All Metrics (gold border = best in column)",
                 fontsize=11, fontweight="bold", color='white', y=1.02)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")

    # Why XGBoost
    st.markdown("#### 🔍 Why XGBoost Was Selected")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **Logistic Regression**
        - ✅ Highest Recall: **0.9249**
        - ✅ Highest ROC-AUC: **0.9929**
        - ❌ Lowest Precision: **0.5714**
        - 4 out of every 10 alerts are false alarms
        - **Verdict:** Too many real jobs wrongly flagged
        """)
    with col2:
        st.markdown("""
        **Random Forest**
        - ✅ Middle Precision: **0.6218**
        - ❌ Lowest Recall: **0.6936**
        - ❌ Lowest F1: **0.6557**
        - Misses 30% of actual fake jobs
        - **Verdict:** Worst overall — not suitable
        """)
    with col3:
        st.markdown("""
        **XGBoost ✅ Selected**
        - ✅ Best F1: **0.8359**
        - ✅ Best Precision: **0.9000**
        - ✅ Strong Recall: **0.7803**
        - Best balance across all metrics
        - **Verdict:** Best practical model
        """)

    st.info("💡 **Key principle:** For fraud detection, Recall matters most — missing a fake job is more harmful than a false alarm. XGBoost gives the best practical balance with F1=0.8359.")

    
    st.markdown("---")

    # Training pipeline
    st.markdown("#### 🔧 Training Pipeline")
    st.code("""
Raw CSV (fake_job_postings.csv)
    ↓  pd.read_csv(engine='python', on_bad_lines='skip')

Text Combination
    ↓  title + company_profile + description + requirements + benefits

Text Cleaning
    ↓  lowercase → remove HTML tags → remove URLs → remove special chars

TF-IDF Vectorization
    ↓  max_features=20000, ngram_range=(1,3),min_df=2,max_df=0.95,sublinear_tf=True, fit ONLY on training data
    ↓  20,000 text features 

Feature Engineering
    ↓  text_length, word_count, has_logo, telecommuting, has_questions
    ↓  scipy.sparse.hstack → 20,005 total features

Train-Test Split
    ↓  80% train / 20% test, stratify=y 

Class Imbalance Handling
    ↓  class_weight='balanced' in all models
    ↓  Fake jobs weighted ~20x higher than real jobs
    ↓  Note: SMOTE was NOT used — it does not work well on sparse
    ↓  TF-IDF matrices (20,000 dimensions causes unrealistic interpolation)

Model Training
    ↓  Logistic Regression (class_weight='balanced')
    ↓  Random Forest      (class_weight='balanced', n_estimators=200)
    ↓  XGBoost            (scale_pos_weight=19.6, n_estimators=300)

Model Selection
    ↓  Best model selected by highest F1 Score → XGBoost wins

Save
    →  best_model.pkl + tfidf_vectorizer.pkl (joblib)
    """, language=None)



# PAGE 4: ABOUT

elif "ℹ️ About" in page:

    st.markdown('<div class="page-header">ℹ️ About SafeApply</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">End-to-End Fake Job Detection ML Project</div>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("#### 🎯 Problem")
        st.markdown("""
        Millions of job seekers lose money and personal data to fraudulent job postings every year:
        - 🔴 Steal Aadhaar, PAN, bank details
        - 🔴 Charge illegal registration / training fees
        - 🔴 Trick applicants into money laundering schemes
        - 🔴 Waste time with fake interview calls

        **SafeApply detects these automatically before you apply.**
        """)
    with col2:
        st.markdown("#### 💡 Hybrid Detection System")
        st.markdown("""

        🔴 **Layer 1 — Fee/financial fraud**
        11 critical keywords → instant FAKE (100% certain)

        🟡 **Layer 2a — Multiple scam patterns**
        3+ obvious phrases found → high confidence FAKE

        🟠 **Layer 2b — Combined signal**
        2+ phrases AND ML also suspicious → FAKE

        🟢 **Layer 3 — ML Model (XGBoost)**
        Catches subtle fraud that keywords can't detect
        """)

    st.markdown("---")

    st.markdown("#### 📊 Dataset")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Name",      "EMSCAD")
    d2.metric("Total Rows","17,880")
    d3.metric("Real Jobs", "17,014 (95.2%)")
    d4.metric("Fake Jobs", "866(4.8%)")
    st.caption("Source: Kaggle — University of the Aegean, Greece")

    st.markdown("---")

    st.markdown("#### 🔬 Technical Details")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("""
        | Component | Detail |
        |-----------|--------|
        | Text fields | title + company_profile + description + requirements + benefits |
        | Text cleaning | lowercase, remove HTML/URLs/special chars |
        | TF-IDF | 20,000 features, unigrams + bigrams + trigrams |
        | Extra features | text_length, word_count, has_logo, telecommuting, has_questions |
        | Total features | 20,005 (via scipy.sparse.hstack) |
        """)
    with t2:
        st.markdown("""
        | Component | Detail |
        |-----------|--------|
        | Class imbalance | class_weight='balanced' — fake jobs ~20x higher weight |
        | Why not SMOTE | Does not work on sparse 20,000-dim TF-IDF matrices |
        | Train/Test | 80% / 20%, stratify=y |
        | Best model | XGBoost — F1=0.8359, Recall=0.7803, Precision=0.9000 |
        | App threshold | 0.35 (lower than default 0.5 — better Recall) |
        """)

    st.markdown("---")

    st.markdown("#### 📁 GitHub Repository Structure")
    st.code("""
SafeApply-Fake-Job-Detection/
├── 📓 notebooks/
│   └── fake_job_detection.ipynb   ← Full training notebook (Google Colab)
│
├── 🌐 app/
│   └── app.py                     ← This Streamlit app (4 pages)
│
├── 🤖 models/
│   ├── best_model.pkl             ← Saved XGBoost model (joblib)
│   ├── tfidf_vectorizer.pkl       ← Saved TF-IDF vectorizer
│   └── model_info.json            ← F1, Recall, Precision, ROC-AUC summary
│
├── 📊 assets/
│   ├── class_distribution.png
│   ├── wordclouds.png
│   ├── feature_analysis.png
│   ├── model_comparison.png
│   ├── roc_curves.png
│   └── confusion_matrix.png
│
├── requirements.txt               ← Python dependencies (no version pins)
├── .gitignore
└── README.md                      ← Full documentation with images
    """, language=None)

    st.markdown("---")

    st.markdown("#### 💡 Key Learnings")
    st.markdown("""
    1. **Keyword filters alone are insufficient** — scammers copy real job language. ML finds hidden statistical patterns that keywords cannot detect
    2. **Hybrid systems work best** — rule-based for obvious fraud (fee keywords) + ML for subtle fraud (vague descriptions, short text)
    3. **Accuracy is misleading** for imbalanced data —  Use F1 and Recall instead
    4. **SMOTE does not work on sparse TF-IDF matrices** — 20,000-dimensional sparse vectors make SMOTE create unrealistic synthetic text. Used class_weight='balanced' instead
    5. **Feature order must match exactly** between training notebook and prediction function — any mismatch = completely wrong predictions
    6. **Recall > Precision** for fraud detection — missing a scam is more harmful than a false alarm
    7. **Multiple scam patterns together** = much higher fraud confidence than any single pattern alone
    """)

    st.markdown("---")
    st.caption("Built with 🤍 | Python · Scikit-learn · XGBoost · Streamlit | GitHub: saloni-78")

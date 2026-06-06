import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments

# scipy for combining feature matrices
from scipy.sparse import hstack, csr_matrix


# PAGE CONFIGURATION


st.set_page_config(
    page_title="SafeApply",
    page_icon="🔍",
    layout="wide",          # Use full width of browser
    initial_sidebar_state="expanded"
)


# CUSTOM CSS STYLING


st.markdown("""
    <style>
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1a1a2e;
        text-align: center;
        padding: 1rem 0;
    }

    /* Result boxes */
    .result-fake {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .result-real {
        background: linear-gradient(135deg, #55efc4, #00b894);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        margin: 1rem 0;
    }

    /* Metric card */
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem 0;
    }

    /* Warning box */
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    /* Info box */
    .info-box {
        background: #d4edda;
        border: 1px solid #28a745;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    /* Hide Streamlit menu for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)



# LOAD MODEL - with caching

@st.cache_resource
def load_model_and_vectorizer():
    """
    Load the trained model and TF-IDF vectorizer from disk.
    Returns None if files not found.
    """
    model_paths = [
        '../models/best_model.pkl',
        'models/best_model.pkl',
        'best_model.pkl'
    ]
    tfidf_paths = [
        '../models/tfidf_vectorizer.pkl',
        'models/tfidf_vectorizer.pkl',
        'tfidf_vectorizer.pkl'
    ]
    info_paths = [
        '../models/model_info.json',
        'models/model_info.json',
        'model_info.json'
    ]

    model, tfidf, info = None, None, {}

    for path in model_paths:
        if os.path.exists(path):
            model = joblib.load(path)
            break

    for path in tfidf_paths:
        if os.path.exists(path):
            tfidf = joblib.load(path)
            break

    for path in info_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                info = json.load(f)
            break

    return model, tfidf, info



# TEXT CLEANING FUNCTION



def clean_text(text):
    """Clean text using the same steps as during training."""
    text = str(text)
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)        # Remove HTML tags
    text = re.sub(r'http\S+|www\S+', ' ', text)  # Remove URLs
    text = re.sub(r'[^a-z0-9\s]', ' ', text)  # Remove special chars
    text = re.sub(r'\s+', ' ', text).strip()   # Remove extra spaces
    return text


# PREDICTION FUNCTION


def predict_job_posting(title, description, requirements, benefits,
                         has_logo, telecommuting, has_questions, model, tfidf):
    """
    Make prediction for a single job posting.

    Returns:
    --------
    prediction    : int (0 = Real, 1 = Fake)
    probability   : float (probability of being fake, 0 to 1)
    warning_flags : list of strings (suspicious patterns found)
    """
    # ── Step 1: Combine and clean text ──
    combined = f'{title} {description} {requirements} {benefits}'
    cleaned = clean_text(combined)

    # ── Step 2: TF-IDF features ──
    text_vector = tfidf.transform([cleaned])

    # ── Step 3: All 5 engineered features — MUST match training order exactly ──
    # Order in training: text_length, word_count, has_logo, telecommuting, has_questions
    text_length = len(cleaned)
    word_count  = len(cleaned.split())
    extra = csr_matrix([[
        text_length,
        word_count,
        int(has_logo),
        int(telecommuting),
        int(has_questions)
    ]])

    # ── Step 4: Combine TF-IDF + extra features ──
    X = hstack([text_vector, extra])

    # ── Step 5: Get raw model probability ──
    proba = model.predict_proba(X)[0][1]  # probability of being fake

    # ── Step 6: Rule-based scam detection ──
    # Check text for known scam patterns BEFORE applying threshold
    text_lower = combined.lower()

    # Critical scam keywords — any match = definitely flag as fake
    critical_keywords = [
        'registration fee', 'joining fee', 'training fee',
        'processing fee', 'deposit fee', 'pay to start',
        'fee to apply', 'upfront fee',
        'earn 50000', 'earn 40000', 'earn 30000', 'earn 20000',
        'lakh per month', 'lakhs per month',
        'weekly from home', 'earn weekly', 'earn daily',
        'form filling', 'copy paste job', 'typing job',
        'data entry work from home', 'work from mobile',
        'no skills required', 'no qualification required',
        'western union', 'wire transfer',
    ]
    critical_hit = any(kw in text_lower for kw in critical_keywords)

    # ── Step 7: Apply threshold with rule-based override ──
    # Threshold = 0.35 (lower than default 0.5 — catching more fraud)
    # Also force FAKE if critical keywords found regardless of model score
    if critical_hit:
        prediction = 1
        proba = max(proba, 0.75)  # boost probability display for critical hits
    elif proba >= 0.35:
        prediction = 1
    else:
        prediction = 0

    # Rule-based warning flags (extra intelligence!)
    # These are patterns commonly found in fake jobs
    warning_flags = []
    text_lower = combined.lower()

    suspicious_phrases = [
        # Generic scam patterns
        ('work from home', 'Suspiciously promotes remote work with no skills needed'),
        ('no experience required', 'Legitimate jobs usually require some experience'),
        ('no experience needed', 'Legitimate jobs usually require some experience'),
        ('earn money fast', 'Legitimate companies rarely promise fast earnings'),
        ('guaranteed income', 'No job can guarantee income'),
        ('guaranteed salary', 'No job can guarantee a salary without conditions'),
        ('send your details', 'Real companies use formal application processes'),
        ('whatsapp', 'Professional jobs rarely ask for WhatsApp contact'),
        ('western union', 'Financial fraud indicator - never send money this way'),
        ('wire transfer', 'Financial scam indicator'),
        # Fee-based scams
        ('upfront fee', 'Real employers NEVER charge upfront fees'),
        ('registration fee', 'Real employers NEVER charge registration fees'),
        ('training fee', 'Real employers NEVER charge training fees'),
        ('processing fee', 'Real employers NEVER charge processing fees'),
        ('joining fee', 'Real employers NEVER charge joining fees'),
        ('deposit fee', 'Real employers NEVER charge deposit fees'),
        ('pay to start', 'You should NEVER pay money to start a job'),
        ('fee to apply', 'Legitimate jobs are always free to apply'),
        # Unrealistic earning promises
        ('1 hour', 'Unrealistic - no real job pays well for just 1 hour of work'),
        ('2 hours', 'Unrealistic work hour promise'),
        ('earn from home', 'Vague earning promise - typical scam pattern'),
        ('earn weekly', 'Suspicious weekly earning promise'),
        ('earn daily', 'Suspicious daily earning promise'),
        ('weekly income', 'Vague income promise without job details'),
        ('daily income', 'Vague income promise without job details'),
        # Indian scam patterns
        ('lakh per month', 'Unrealistically high salary promise'),
        ('lakhs per month', 'Unrealistically high salary promise'),
        ('work from mobile', 'Typical mobile-based scam pattern'),
        ('typing job', 'Common low-skill scam job type'),
        ('copy paste', 'Common scam job type - not real employment'),
        ('data entry work from home', 'Extremely common fake job pattern'),
        ('whatsapp me', 'Professional jobs never recruit via WhatsApp'),
        ('immediate joining', 'Pressure tactic - legitimate jobs have proper notice periods'),
        ('urgent hiring', 'Pressure tactic - real companies take time to hire properly'),
    ]

    for phrase, reason in suspicious_phrases:
        if phrase in text_lower:
            warning_flags.append(f'⚠️ Found "{phrase}": {reason}')

    return int(prediction), float(proba), warning_flags



# DRAW PROBABILITY GAUGE CHART


def draw_gauge(probability):
    """
    Draw a colored gauge/meter showing fraud probability.
    Green = safe, Yellow = suspicious, Red = dangerous
    """
    fig, ax = plt.subplots(figsize=(5, 2.5))

    # Draw background bar
    ax.barh(0, 1.0, height=0.5, color='#e0e0e0', left=0)

    # Color based on probability
    if probability < 0.3:
        bar_color = '#2ecc71'   # Green = safe
        label = 'LOW RISK'
    elif probability < 0.6:
        bar_color = '#f39c12'   # Orange = suspicious
        label = 'MEDIUM RISK'
    else:
        bar_color = '#e74c3c'   # Red = high risk
        label = 'HIGH RISK'

    # Draw probability bar
    ax.barh(0, probability, height=0.5, color=bar_color, left=0)

    # Add percentage text
    ax.text(0.5, 0, f'{probability:.1%}', ha='center', va='center',
            fontsize=16, fontweight='bold', color='black', zorder=5)

    # Styling
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.7)
    ax.axis('off')
    ax.set_title(f'Fraud Probability: {label}', fontsize=12, fontweight='bold',
                 color=bar_color)

    plt.tight_layout()
    return fig



# SIDEBAR


with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/find-matching-job--v1.png",
             width=80)
    st.title("🔍 SafeApply - Fake Job Detector")
    st.markdown("---")

    # Navigation
    st.subheader("Navigation")
    page = st.radio(
        "Choose a page:",
        ["🏠 Single Job Check", "📊 Bulk CSV Check", "📈 Model Performance", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Load model and show status
    model, tfidf, model_info = load_model_and_vectorizer()

    if model is not None:
        st.success("✅ Model Loaded")
        if model_info:
            st.caption(f"Model: **{model_info.get('best_model_name', 'Unknown')}**")
            st.caption(f"F1 Score: **{model_info.get('f1_score', 'N/A')}**")
            st.caption(f"ROC-AUC: **{model_info.get('roc_auc', 'N/A')}**")
    else:
        st.error("❌ Model Not Found")
        st.warning(
            "Please train the model first using the notebook, "
            "then place `best_model.pkl` and `tfidf_vectorizer.pkl` "
            "in the `models/` folder."
        )

    st.markdown("---")
    st.caption("Built with 🤍 using Scikit-learn + Streamlit")
    st.caption("Dataset: EMSCAD (Kaggle)")



# PAGE 1: SINGLE JOB CHECK


if "🏠 Single Job Check" in page:

    st.markdown('<h1 class="main-header">🔍 SafeApply - Fake Job Posting Detector</h1>',
                unsafe_allow_html=True)

    st.markdown("---")

    # Two-column layout
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("📝 Enter Job Details")

        # Input fields
        title = st.text_input(
            "Job Title *",
            placeholder="e.g., Senior Software Engineer",
            help="Enter the exact job title as shown in the posting"
        )

        description = st.text_area(
            "Job Description *",
            placeholder="Paste the full job description here...",
            height=150,
            help="The main job description is the most important field for prediction"
        )

        requirements = st.text_area(
            "Requirements (optional)",
            placeholder="e.g., 3+ years Python experience, Bachelor's degree...",
            height=100
        )

        benefits = st.text_area(
            "Benefits (optional)",
            placeholder="e.g., Health insurance, Remote work, Stock options...",
            height=80
        )

        # Checkboxes — all default False (unknown = do not assume)
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            has_logo = st.checkbox(
                "Has company logo?",
                value=False,
                help="Does the posting show a verified company logo? Leave unchecked if unsure."
            )
        with col_b:
            telecommuting = st.checkbox(
                "Remote/Telecommute?",
                value=False,
                help="Is this advertised as a remote work position?"
            )
        with col_c:
            has_questions = st.checkbox(
                "Has screening questions?",
                value=False,
                help="Does the posting include application questions? Real jobs usually do."
            )

        # Predict button
        predict_btn = st.button(
            "🔍 Analyze Job Posting",
            type="primary",
            use_container_width=True
        )

    with col2:
        st.subheader("📊 Analysis Result")

        if predict_btn:
            # Validate input
            if not title or not description:
                st.error("Please fill in at least the Job Title and Description!")
            elif model is None:
                st.error("Model not loaded. Please train the model first.")
            else:
                # Show spinner while processing
                with st.spinner("Analyzing job posting..."):
                    prediction, probability, flags = predict_job_posting(
                        title, description, requirements, benefits,
                        has_logo, telecommuting, has_questions, model, tfidf
                    )

                # Display result — prediction already includes rule-based override
                if prediction == 1:
                    st.markdown(
                        '<div class="result-fake">🚨 LIKELY FAKE JOB!<br>'
                        f'Fraud Probability: {probability:.1%}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="result-real">✅ LIKELY REAL JOB<br>'
                        f'Fraud Probability: {probability:.1%}</div>',
                        unsafe_allow_html=True
                    )

                # Gauge chart
                gauge_fig = draw_gauge(probability)
                st.pyplot(gauge_fig)
                plt.close()

                # Warning flags
                if flags:
                    st.subheader("🚩 Warning Signs Found:")
                    for flag in flags:
                        st.warning(flag)
                else:
                    st.success("✅ No suspicious phrases detected in text.")

                # Confidence interpretation
                st.subheader("📖 How to Interpret:")
                if probability < 0.3:
                    st.info("Low fraud risk. This looks like a legitimate job posting.")
                elif probability < 0.6:
                    st.warning(
                        "Medium risk. Proceed with caution. "
                        "Research the company before applying."
                    )
                else:
                    st.error(
                        "High fraud risk! This has multiple characteristics "
                        "of a fake job. Do NOT share personal/financial information."
                    )

        else:
            # Show placeholder when no prediction yet
            st.info("👈 Fill in the job details and click **Analyze** to see the result.")
            st.markdown("""
            **What we check:**
            - Language patterns in job description
            - Requirements and benefits text
            - Presence of company logo
            - Common fraud indicators
            - 10,000+ text features using TF-IDF
            """)



# PAGE 2: BULK CSV CHECK


elif "📊 Bulk CSV Check" in page:

    st.title("📊 Bulk Job Posting Checker")
    st.markdown(
        "Upload a CSV file with multiple job postings to check them all at once."
    )

    st.markdown("---")

    # Show expected format
    st.subheader("📋 Required CSV Format")
    st.markdown("Your CSV file must have these column names:")

    example_df = pd.DataFrame({
        'title': ['Data Scientist', 'Work from Home Earn Money Fast'],
        'description': ['We are looking for ML engineer...', 'No experience needed! Earn $5000...'],
        'requirements': ['Python, SQL, ML experience', ''],
        'benefits': ['Health insurance, 401k', ''],
        'has_company_logo': [1, 0],
        'telecommuting': [0, 1]
    })
    st.dataframe(example_df, use_container_width=True)

    # Download example CSV
    csv_bytes = example_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Download Example CSV",
        data=csv_bytes,
        file_name="example_jobs.csv",
        mime="text/csv"
    )

    st.markdown("---")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=['csv'],
        help="CSV must have: title, description (required). requirements, benefits, "
             "has_company_logo, telecommuting (optional)"
    )

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded: {len(input_df)} job postings found")
            st.dataframe(input_df.head(), use_container_width=True)

            if model is None:
                st.error("Model not loaded!")
            else:
                if st.button("🚀 Analyze All Jobs", type="primary"):
                    results_list = []

                    # Progress bar — shows percentage completion
                    progress_bar = st.progress(0)
                    status_text = st.empty()  # Placeholder for status text

                    for idx, row in input_df.iterrows():
                        # Update progress
                        progress = (idx + 1) / len(input_df)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing job {idx+1} of {len(input_df)}...")

                        # Get values safely (handle missing columns)
                        title_val = str(row.get('title', ''))
                        desc_val  = str(row.get('description', ''))
                        req_val   = str(row.get('requirements', ''))
                        ben_val   = str(row.get('benefits', ''))
                        logo_val  = int(row.get('has_company_logo', 1))
                        tele_val  = int(row.get('telecommuting', 0))

                        quest_val = int(row.get('has_questions', 0))
                        pred, prob, _ = predict_job_posting(
                            title_val, desc_val, req_val, ben_val,
                            logo_val, tele_val, quest_val, model, tfidf
                        )

                        results_list.append({
                            'job_title': title_val[:60],
                            'prediction': '🚨 FAKE' if pred == 1 else '✅ REAL',
                            'fraud_probability': f'{prob:.1%}',
                            'risk_level': (
                                'HIGH' if prob > 0.6
                                else 'MEDIUM' if prob > 0.3
                                else 'LOW'
                            )
                        })

                    status_text.text("Analysis complete!")
                    progress_bar.progress(1.0)

                    results_df = pd.DataFrame(results_list)

                    # Summary stats
                    total = len(results_df)
                    fake_count = (results_df['prediction'] == '🚨 FAKE').sum()
                    real_count = total - fake_count

                    st.markdown("---")
                    st.subheader("📊 Summary")

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Analyzed", total)
                    m2.metric("Real Jobs", real_count, delta=f"{real_count/total:.0%}")
                    m3.metric("Suspicious Jobs", fake_count,
                              delta=f"{fake_count/total:.0%}",
                              delta_color="inverse")

                    # Results table
                    st.subheader("📋 Detailed Results")
                    st.dataframe(results_df, use_container_width=True)

                    # Download results
                    output_csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "⬇️ Download Results CSV",
                        data=output_csv,
                        file_name="fraud_detection_results.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            st.info("Make sure your CSV has 'title' and 'description' columns.")


# ═══════════════════════════════════════════════════════════
# PAGE 3: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════

elif "📈 Model Performance" in page:

    st.title("📈 Model Performance Dashboard")
    st.markdown("Complete comparison of all three trained models")
    st.markdown("---")

    # ── Dataset Overview ──
    st.subheader("📊 Dataset Overview")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Total Job Postings", "9,812")
    d2.metric("Real Jobs", "9,370 (95.5%)")
    d3.metric("Fake Jobs", "442 (4.5%)")
    d4.metric("Train / Test Split", "80% / 20%")
    st.warning(
        "⚠️ **Class Imbalance:** Only 4.5% of jobs are fake. "
        "Solved using SMOTE — balanced training set to 50% fake / 50% real."
    )

    st.markdown("---")

    # ── All 3 Models Results Table ──
    st.subheader("🤖 All Models — Results Comparison")

    results_data = {
        "Model": ["Logistic Regression", "Random Forest", "XGBoost ✅ Best"],
        "Accuracy":  ["0.9712", "0.9827", "0.9801"],
        "Precision": ["0.6411", "0.9370", "0.7742"],
        "Recall":    ["0.9191", "0.6879", "0.8324"],
        "F1 Score":  ["0.7553", "0.7933", "0.8022"],
        "ROC-AUC":   ["0.9918", "0.9891", "0.9852"],
    }
    results_df = pd.DataFrame(results_data)
    st.dataframe(
        results_df.style
        .highlight_max(subset=["Accuracy","Precision","Recall","F1 Score","ROC-AUC"], color="green")
        .apply(lambda x: ["background-color:green; font-weight: bold"
                          if x.name == 2 else "" for _ in x], axis=1),
        use_container_width=True,
        hide_index=True
    )
    st.caption("✅ Green = best value in column  |  Green row = selected model (XGBoost)")

    st.markdown("---")

    # ── Best Model Metrics ──
    st.subheader("🏆 Best Model — XGBoost")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",  "0.9801", help="Overall correct predictions")
    c2.metric("Precision", "0.7742", help="Of flagged fakes, how many were actually fake")
    c3.metric("Recall",    "0.8324", delta="Most important!", help="Of all actual fakes, how many did we catch")
    c4.metric("F1 Score",  "0.8022", help="Balance between Precision and Recall")
    c5.metric("ROC-AUC",   "0.9852", help="Overall discrimination ability — 1.0 = perfect")

    st.markdown("---")

    # ── Bar Chart Comparison ──
    st.subheader("📊 Visual Comparison — All Models")

    import matplotlib.pyplot as plt
    import numpy as np

    models = ["LR", "RF", "XGBoost"]
    metrics_vals = {
        "Accuracy":  [0.9712, 0.9827, 0.9801],
        "Precision": [0.6411, 0.9370, 0.7742],
        "Recall":    [0.9191, 0.6879, 0.8324],
        "F1 Score":  [0.7553, 0.7933, 0.8022],
        "ROC-AUC":   [0.9918, 0.9891, 0.9852],
    }

    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    for ax, (metric, vals) in zip(axes, metrics_vals.items()):
        bars = ax.bar(models, vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(metric, fontweight="bold", fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
        # Highlight best bar
        best_idx = vals.index(max(vals))
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(2.5)

    plt.suptitle("Model Comparison — All Metrics", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ── Model Analysis ──
    st.subheader("🔍 Why XGBoost Was Selected")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **Logistic Regression**
        - ✅ Highest Recall: **0.9191**
        - ✅ Highest ROC-AUC: **0.9918**
        - ❌ Lowest Precision: **0.6411**
        - ❌ Too many real jobs wrongly flagged as fake
        - **Verdict:** Too many false alarms
        """)

    with col2:
        st.markdown("""
        **Random Forest**
        - ✅ Highest Accuracy: **0.9827**
        - ✅ Highest Precision: **0.9370**
        - ❌ Lowest Recall: **0.6879**
        - ❌ Misses **31%** of actual fake jobs
        - **Verdict:** Lets too many frauds through
        """)

    with col3:
        st.markdown("""
        **XGBoost ✅ Selected**
        - ✅ Highest F1 Score: **0.8022**
        - ✅ Strong Recall: **0.8324**
        - ✅ Good Precision: **0.7742**
        - ✅ Best overall balance
        - **Verdict:** Best practical model
        """)

    st.info(
        "💡 **Key Insight:** For fraud detection, **Recall matters more than Precision**. "
        "Missing a fake job is MORE harmful than a false alarm. "
        "XGBoost gives the best practical balance with F1=0.8022 and Recall=0.8324."
    )

    st.markdown("---")

    # ── Metrics Explanation ──
    st.subheader("📖 Understanding the Metrics")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Accuracy**
        - Overall % of correct predictions
        - ⚠️ *Not reliable for imbalanced data!*
        - Predict all Real → 95.5% accuracy, 0 frauds caught

        **Precision**
        - Of all jobs flagged as fake, what % were actually fake?
        - High precision = fewer false alarms on real jobs
        """)
    with col2:
        st.markdown("""
        **Recall**
        - Of all actual fake jobs, what % did we catch?
        - ⭐ *Most important for fraud detection!*
        - Low recall = many frauds slip through undetected

        **F1 Score**
        - Harmonic mean of Precision and Recall
        - Formula: 2 × (P × R) / (P + R)
        - Best single metric for imbalanced classification
        """)

    st.markdown("---")

    # ── Training Pipeline ──
    st.subheader("🔧 Training Pipeline")
    st.code("""
Raw Text Data (title + description + requirements + benefits)
     ↓
Text Cleaning  (lowercase, remove HTML / URLs / special chars)
     ↓
TF-IDF Vectorization  (10,000 features, unigrams + bigrams)
     +
Feature Engineering  (text_length, word_count, has_logo, telecommuting, has_questions)
     ↓
scipy.sparse.hstack  (combine TF-IDF + numerical features)
     ↓
Train-Test Split  (80% train / 20% test, stratify=y)
     ↓
SMOTE Oversampling  (training only: 4.5% fake → 50% fake)
     ↓
Train 3 Models  (Logistic Regression, Random Forest, XGBoost)
     ↓
Evaluate on Test Set  (F1, Recall, Precision, ROC-AUC)
     ↓
Save Best Model  (joblib → best_model.pkl + tfidf_vectorizer.pkl)
    """, language=None)


# ═══════════════════════════════════════════════════════════
# PAGE 4: ABOUT
# ═══════════════════════════════════════════════════════════

elif "ℹ️ About" in page:

    st.title("ℹ️ About SafeApply")
    st.markdown("Fake Job Posting Detection — End-to-End ML Project")
    st.markdown("---")

    # ── Problem + Solution ──
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Problem Statement")
        st.markdown("""
        Millions of job seekers lose money and personal data
        to fraudulent job postings every year. Scammers:
        - Steal Aadhaar, PAN, bank details
        - Charge illegal registration / training fees
        - Trick applicants into money laundering
        - Waste time with fake interview calls

        **SafeApply detects these scams automatically
        before the job seeker applies.**
        """)

    with col2:
        st.subheader("💡 Our Solution")
        st.markdown("""
        An end-to-end Machine Learning system that:
        - Analyzes job posting text using **TF-IDF NLP**
        - Combines text with **5 engineered features**
        - Handles **4.5% class imbalance** using SMOTE
        - Compares **3 ML models** and selects the best
        - Deploys as a **live Streamlit web app**
        - Flags **37 scam patterns** with rule-based checks
        """)

    st.markdown("---")

    # ── Dataset ──
    st.subheader("📊 Dataset")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Dataset", "EMSCAD")
    d2.metric("Total Rows", "9,812")
    d3.metric("Real Jobs", "9,370 (95.5%)")
    d4.metric("Fake Jobs", "442 (4.5%)")
    st.caption("Source: Kaggle — University of the Aegean, Greece")

    st.markdown("---")

    # ── Technical Approach ──
    st.subheader("🔬 Technical Approach")

    t1, t2 = st.columns(2)
    with t1:
        st.markdown("""
        | Component | Details |
        |-----------|---------|
        | **Text Combination** | title + company_profile + description + requirements + benefits |
        | **Text Cleaning** | lowercase, remove HTML/URLs/special chars |
        | **Text Features** | TF-IDF — 10,000 features, unigrams + bigrams |
        | **Extra Features** | text_length, word_count, has_logo, telecommuting, has_questions |
        | **Feature Matrix** | scipy.sparse.hstack → 10,005 total features |
        """)
    with t2:
        st.markdown("""
        | Component | Details |
        |-----------|---------|
        | **Class Imbalance** | SMOTE — training only, 4.5% → 50% fake |
        | **Train/Test Split** | 80/20, stratify=y |
        | **Models Trained** | Logistic Regression, Random Forest, XGBoost |
        | **Best Model** | XGBoost — F1=0.8022, Recall=0.8324 |
        | **Evaluation** | F1 Score, Recall, Precision, ROC-AUC |
        """)

    st.markdown("---")


    # ── Key Learnings ──
    st.subheader("💡 Key Learnings")
    st.markdown("""
    1. **Accuracy is misleading** for imbalanced data — F1 and Recall are the right metrics
    2. **SMOTE only on training data** — applying on test causes data leakage
    3. **TF-IDF fit only on training data** — prevents test set contamination
    4. **Bigrams are powerful** — "registration fee" means more than "registration" + "fee" separately
    5. **Word clouds showed** fake and real jobs use similar vocab — ML is necessary over keyword rules
    6. **Recall > Precision** for fraud — missing a scam is worse than a false alarm
    """)

    st.markdown("---")


    st.caption("Built with 🤍 using Python · Scikit-learn · XGBoost · Streamlit")

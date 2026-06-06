<div align="center">

# 🛡️ SafeApply — Fake Job Posting Detection

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7-189AB4?style=for-the-badge)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Colab](https://img.shields.io/badge/Google_Colab-Notebook-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com)
[![License](https://img.shields.io/badge/License-MIT-2ecc71?style=for-the-badge)](LICENSE)

<br>

**An end-to-end NLP + Machine Learning system that detects fraudulent job postings
using a hybrid rule-based + ML approach — protecting job seekers before they apply.**

<br>

**[🚀 Live Demo](https://safeapply-fake-job-detection-apaa6u3rgypmi4d2ntg4rv.streamlit.app/)** &nbsp;·&nbsp;
**[📓 Open in Colab](https://colab.research.google.com/github/saloni-78/SafeApply-Fake-Job-Detection/blob/main/notebooks/fake_job_detection.ipynb)** &nbsp;·&nbsp;
**[📊 Dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)**

</div>

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [How It Works — Hybrid System](#-how-it-works--hybrid-system)
- [Project Pipeline](#-project-pipeline)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Feature Engineering](#-feature-engineering)
- [Handling Class Imbalance](#-handling-class-imbalance--smote)
- [Models and Results](#-models--results)
- [Confusion Matrix](#-confusion-matrix)
- [Streamlit Web App](#-streamlit-web-app)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Tech Stack](#-tech-stack)
- [Key Learnings](#-key-learnings)
- [Limitations and Future Work](#-limitations--future-work)

---

## 🎯 Problem Statement

Millions of job seekers encounter **fake job postings** every year. These fraudulent listings:

| Threat | Impact |
|--------|--------|
| 🔴 Steal personal information | Aadhaar, PAN, bank details compromised |
| 🔴 Charge illegal upfront fees | Registration, training, joining fees |
| 🔴 Money laundering schemes | Victims unknowingly used as mules |
| 🔴 Fake interview calls | Time wasted, mental health affected |

**SafeApply** automatically flags suspicious postings using a two-layer hybrid system — before the job seeker applies.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Name** | Real or Fake Job Posting Prediction (EMSCAD) |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) — University of the Aegean, Greece |
| **Total Rows** | 9,812 job postings (after cleaning) |
| **Real Jobs** | 9,370 → **95.5%** |
| **Fake Jobs** | 442 → **4.5%** |
| **Target Column** | `fraudulent` (0 = Real, 1 = Fake) |
| **Text Columns Used** | title, company_profile, description, requirements, benefits |
| **Key Challenge** | Severe class imbalance — only 4.5% fake jobs |

---

## 🧠 How It Works — Hybrid System

SafeApply uses a **two-layer hybrid detection system** — the same approach used by platforms like LinkedIn, Indeed, and Naukri:

```
Job Posting Text
       │
       ▼
┌─────────────────────────────────────────────────┐
│  LAYER 1 — Rule-Based Filter                     │
│                                                  │
│  Checks for 100% certain fraud indicators:       │
│  • Fee fraud: "registration fee", "joining fee"  │
│  • Financial fraud: "western union", "wire       │
│    transfer"                                     │
│                                                  │
│  These have ZERO false positive risk.            │
│  No legitimate employer ever uses these.         │
└─────────────────────────────────────────────────┘
       │                    │
  Found ──→ 🚨 FAKE      Not Found
                              │
                              ▼
┌─────────────────────────────────────────────────┐
│  LAYER 2 — ML Model (XGBoost + TF-IDF)          │
│                                                  │
│  Catches SUBTLE fraud that keywords miss:        │
│  • Vague job descriptions                        │
│  • Suspiciously short requirements               │
│  • Missing company details                       │
│  • Hidden word patterns from 9,000+ examples    │
│  • Unusual text length and word count            │
│                                                  │
│  Threshold = 0.35 (catches more fraud)           │
└─────────────────────────────────────────────────┘
       │                    │
  prob ≥ 0.35 ──→ 🚨 FAKE   prob < 0.35 ──→ ✅ REAL
```

### Why ML is essential — not just keywords

Consider this job posting:
```
Title      : Data Entry Executive
Description: Work from our office. Flexible timings.
             No prior experience needed. Salary 15,000/month.
             Call for interview.
```

- ❌ No suspicious keywords detected
- ❌ No fee mentioned
- ❌ No unrealistic salary
- Looks completely normal to a keyword filter

✅ **ML catches it** — because it learned from 9,000+ examples that:
vague descriptions + no specific skills + missing company profile + low word count = strong fraud signal

> **Rule-based catches obvious scams. ML catches the sophisticated ones that look real.**

---

## 🔧 Project Pipeline

```
Raw CSV Data  (fake_job_postings.csv)
        │
        ▼
┌─────────────────────────────────────────┐
│  1. Data Loading                         │
│     pd.read_csv(engine='python',         │
│     on_bad_lines='skip')                 │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  2. Text Combination                     │
│     title + company_profile +            │
│     description + requirements +         │
│     benefits  →  combined_text           │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  3. Text Cleaning                        │
│     lowercase → remove HTML/URLs →       │
│     remove special characters            │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  4. Feature Engineering                  │
│     TF-IDF (10,000 features, bigrams)    │
│   + text_length, word_count, has_logo,   │
│     telecommuting, has_questions         │
│   → scipy.sparse.hstack → 10,005 total  │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  5. Train-Test Split  (80% / 20%)        │
│     stratify=y  →  preserves 4.5% ratio  │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  6. SMOTE Oversampling                   │
│     Training only: 4.5% → 50% fake       │
│     Test set unchanged (real world dist) │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  7. Train and Compare 3 Models           │
│     ① Logistic Regression (baseline)    │
│     ② Random Forest      (ensemble)     │
│     ③ XGBoost            (best model)   │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  8. Evaluate and Save Best Model         │
│     F1, Recall, Precision, ROC-AUC       │
│     joblib.dump → best_model.pkl         │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  9. Hybrid Streamlit App                 │
│     Rule-based + ML + Warning Flags      │
└─────────────────────────────────────────┘
```

---

## 📈 Exploratory Data Analysis

### Class Distribution

![Class Distribution](assets/class_distribution.png)

> ⚠️ **Key Observation:** Only **4.5% of jobs are fake** — severe class imbalance. A naive model predicting every job as Real gets **95.5% accuracy but catches zero fraud**. This is why we use F1-Score and Recall, and apply SMOTE to balance training data.

---

### Word Cloud — Fake vs Real Jobs

![Word Clouds](assets/wordclouds.png)

**Key Observations:**
- Both fake and real jobs use similar words: *work, project, team, customer, company*
- **Simple keyword matching will NOT work** — scammers deliberately copy real job language
- ML is necessary to find the hidden statistical differences between the two classes

---

### Feature Analysis — Word Count and Company Logo

![Feature Analysis](assets/feature_analysis.png)

**Key Observations:**
- **Word Count (left):** Real jobs have longer, more detailed descriptions — justifies `word_count` as a feature
- **Company Logo (right):** Real jobs more often have verified logos — `has_company_logo` is a useful signal but **not used as a rule** (only as one of 10,005 ML features)

---

## ⚙️ Feature Engineering

### TF-IDF Vectorization

TF-IDF converts text into numerical features with importance weights. Common words like *the, is* get low scores; rare meaningful words get high scores.

| Parameter | Value | Reason |
|-----------|-------|--------|
| `max_features` | 10,000 | Top 10,000 most informative words |
| `ngram_range` | (1, 2) | Single words + bigrams: *"work from"*, *"no experience"* |
| `min_df` | 2 | Ignore words in fewer than 2 documents |
| `sublinear_tf` | True | Log scaling reduces very frequent words |
| Fit on | **Training data only** | Prevents data leakage to test set |

### Engineered Numerical Features

Five features combined with TF-IDF using `scipy.sparse.hstack` → **10,005 total features:**

| Feature | Type | Why It Helps |
|---------|------|-------------|
| `text_length` | Integer | Fake jobs have shorter, vaguer descriptions |
| `word_count` | Integer | Real jobs use more detailed language |
| `has_company_logo` | Binary | Fake jobs less often have verified logos |
| `telecommuting` | Binary | Some scam postings over-promise remote work |
| `has_questions` | Binary | Legitimate jobs usually include screening questions |

> These are used as **inputs to the ML model** — weighted alongside 10,000 TF-IDF features. No single feature overrides the model on its own.

---

## ⚖️ Handling Class Imbalance — SMOTE

**SMOTE** (Synthetic Minority Over-sampling Technique) creates synthetic fake job examples for the training set.

```
Before SMOTE (training set):
  Real jobs :  7,495  (95.5%)
  Fake jobs :    354   (4.5%)   ← model barely sees fake examples

After SMOTE (training set):
  Real jobs :  7,495  (50%)
  Fake jobs :  7,495  (50%)    ← model learns fake patterns properly

Test set (never touched):
  Real jobs :  1,875  (95.5%)
  Fake jobs :     88   (4.5%)  ← real distribution for honest evaluation
```

> ⚠️ **Critical:** SMOTE applied **ONLY on training data**. Applying on test data = data leakage.

---

## 🤖 Models and Results

Three models trained on SMOTE-balanced data, evaluated on original test set:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|:--------:|:---------:|:------:|:--------:|:-------:|
| Logistic Regression | 0.9712 | 0.6411 | **0.9191** | 0.7553 | **0.9918** |
| Random Forest | **0.9827** | **0.9370** | 0.6879 | 0.7933 | 0.9891 |
| **XGBoost ✅** | 0.9801 | 0.7742 | 0.8324 | **0.8022** | 0.9852 |

> 🏆 **XGBoost selected** — highest F1 Score (0.8022) with strong Recall (0.8324)

### Model Comparison Chart

![Model Comparison](assets/model_comparison.png)

### ROC Curves

![ROC Curves](assets/roc_curves.png)

> All three models have ROC-AUC above **0.985** — excellent discrimination ability.

### Why XGBoost Was Selected

| Model | Strength | Weakness | Verdict |
|-------|----------|----------|---------|
| Logistic Regression | Highest Recall 0.9191 | Lowest Precision 0.6411 — too many false alarms | ❌ Flags too many real jobs |
| Random Forest | Highest Precision 0.9370 | Lowest Recall 0.6879 — misses 31% of frauds | ❌ Lets too many frauds through |
| **XGBoost** | Best F1 0.8022 — best balance | Middle precision and recall | ✅ Best practical model |

> 💡 **For fraud detection, Recall matters more than Precision.** Missing a fake job is more harmful than a false alarm.

---

## 📊 Confusion Matrix

![Confusion Matrix](assets/confusion_matrix.png)

| | Predicted Real | Predicted Fake |
|--|:--:|:--:|
| **Actual Real** | ✅ 3,361 correctly identified | ❌ 42 real jobs wrongly flagged |
| **Actual Fake** | ❌ 29 frauds missed | ✅ 144 frauds correctly caught |

- **144 out of 173 fake jobs caught** → 83.2% Recall
- Only **29 fake jobs missed** — acceptable for real-world deployment
- Only **42 false alarms** out of 3,403 real jobs → 98.8% Specificity

---

## 🌐 Streamlit Web App

A 4-page interactive web application with a **hybrid detection engine:**

| Page | Feature |
|------|---------|
| 🏠 **Single Job Check** | Title + description → hybrid prediction + fraud probability gauge + warning flags |
| 📊 **Bulk CSV Upload** | Upload CSV → batch predictions → download results with risk levels |
| 📈 **Model Performance** | Full dashboard — all 3 model results, bar charts, metrics explanation |
| ℹ️ **About** | Project documentation, hybrid system explanation, GitHub structure |

**Detection layers in the app:**
```
Layer 1 — Rule-based (12 critical patterns):
  Fee fraud    : registration fee, joining fee, training fee ...
  Financial    : western union, wire transfer, money transfer

Layer 2 — ML Model (XGBoost, threshold=0.35):
  Catches subtle fraud: vague descriptions, short text,
  missing company details, suspicious word patterns

Layer 3 — Warning flags (35 patterns displayed to user):
  Informational alerts shown even when prediction is Real
  Helps user make their own informed decision
```

---

## 📁 Project Structure

```
SafeApply-Fake-Job-Detection/
│
├── 📓 notebooks/
│   └── fake_job_detection.ipynb    ← Complete Google Colab training notebook
│
├── 🌐 app/
│   └── app.py                      ← Streamlit web app (hybrid detection)
│
├── 🤖 models/
│   ├── best_model.pkl              ← Saved XGBoost model
│   ├── tfidf_vectorizer.pkl        ← Saved TF-IDF vectorizer
│   └── model_info.json             ← Model performance summary
│
├── 📊 assets/
│   ├── class_distribution.png      ← Class imbalance visualization
│   ├── wordclouds.png              ← Fake vs real word clouds
│   ├── feature_analysis.png        ← Word count and logo analysis
│   ├── model_comparison.png        ← All 3 models bar chart
│   ├── roc_curves.png              ← ROC curves for all 3 models
│   └── confusion_matrix.png        ← XGBoost confusion matrix
│
├── requirements.txt                ← Python dependencies
├── .gitignore                      ← Excludes cache and sensitive files
└── README.md                       ← This file
```

---

## ▶️ How to Run

### Option 1 — Live Demo
👉 **[Open SafeApply App](https://safeapply-fake-job-detection-apaa6u3rgypmi4d2ntg4rv.streamlit.app/)**

### Option 2 — Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/saloni-78/SafeApply-Fake-Job-Detection.git
cd SafeApply-Fake-Job-Detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app/app.py
```

### Option 3 — Train from Scratch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saloni-78/SafeApply-Fake-Job-Detection/blob/main/notebooks/fake_job_detection.ipynb)

1. Download `fake_job_postings.csv` from [Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
2. Upload to Google Drive
3. Open notebook in Colab → Run all cells
4. Download model files → place in `models/` folder

---

## 🛠️ Tech Stack

| Category | Library | Purpose |
|----------|---------|---------|
| Data | `pandas`, `numpy` | Data loading and manipulation |
| NLP | `TfidfVectorizer` | TF-IDF with bigrams — 10,000 text features |
| ML | `scikit-learn` | Logistic Regression, Random Forest |
| ML | `xgboost` | Gradient boosting — best model |
| Imbalance | `imbalanced-learn` | SMOTE oversampling |
| Sparse Matrix | `scipy` | hstack TF-IDF + numerical features |
| Visualization | `matplotlib`, `seaborn` | Charts, ROC curves, confusion matrix |
| Visualization | `wordcloud` | Word clouds for EDA |
| Web App | `streamlit` | Interactive 4-page hybrid detection app |
| Model Saving | `joblib` | Save and load model + vectorizer |
| Environment | Google Colab | Cloud notebook for training |

---

## 💡 Key Learnings

1. **Keyword filters alone are insufficient** — scammers copy real job language. ML finds hidden statistical patterns keywords cannot detect
2. **Hybrid systems work best** — rule-based for obvious fraud + ML for subtle fraud — same approach used by LinkedIn, Indeed, Naukri
3. **Accuracy is misleading** for imbalanced data — F1-Score and Recall are the right metrics
4. **SMOTE only on training data** — applying on test data causes data leakage
5. **TF-IDF fit only on training data** — prevents test set contamination
6. **Bigrams are powerful** — *"registration fee"* carries far more signal than individual words
7. **Recall matters more than Precision** — missing a fraud is worse than a false alarm
8. **Random Forest had highest Precision but lowest Recall** — unsuitable for fraud detection
9. **engine='python'** in `pd.read_csv()` handles messy real-world CSV better than C engine
10. **No single feature should override the model** — `has_company_logo` is one of 10,005 features, not a deciding rule

---

## ⚠️ Limitations and Future Work

**Current Limitations:**
- EMSCAD dataset contains mostly Western English job scams — Indian-specific subtle patterns may be underrepresented in ML training data
- ~8,000 rows dropped due to CSV encoding issues — larger clean dataset would improve performance

**Future Improvements:**
- [ ] **SHAP Explainability** — show which specific words drove each prediction
- [ ] **BERT Embeddings** — richer semantic features beyond TF-IDF
- [ ] **Hyperparameter Tuning** with Optuna — Bayesian optimization
- [ ] **MLflow Tracking** — professional experiment logging
- [ ] **Indian Job Portal Dataset** — train on Naukri/LinkedIn India data for better local detection
- [ ] **Confidence threshold tuning** — ROC curve analysis to find optimal threshold
- [ ] **Label Encoding** — add employment_type, industry, required_experience as features

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

## 👤 Author

**Priya · Bhawana ·Saloni **

---

<div align="center">

*Built with 🤍 using Python · Scikit-learn · XGBoost · Streamlit*

</div>

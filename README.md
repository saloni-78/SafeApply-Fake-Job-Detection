<div align="center">

# 🛡️ SafeApply — Fake Job Posting Detection

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7-189AB4?style=for-the-badge)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Colab](https://img.shields.io/badge/Google_Colab-Notebook-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com)
[![License](https://img.shields.io/badge/License-MIT-2ecc71?style=for-the-badge)](LICENSE)

<br>

**An end-to-end NLP + Machine Learning system that detects fraudulent job postings,
protecting job seekers from scams before they apply.**

<br>

**[🚀 Live Demo](https://safeapply-fake-job-detection-apaa6u3rgypmi4d2ntg4rv.streamlit.app/)** &nbsp;·&nbsp;
**[📓 Open in Colab](https://colab.research.google.com/github/saloni-78/SafeApply-Fake-Job-Detection/blob/main/notebooks/fake_job_detection.ipynb)** &nbsp;·&nbsp;
**[📊 Dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)**

</div>

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
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

**SafeApply** uses Machine Learning and NLP to automatically analyze job postings and flag suspicious ones — **before** the job seeker applies.

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
│     lowercase → remove HTML tags →       │
│     remove URLs → remove special chars   │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  4. Feature Engineering                  │
│     TF-IDF (10,000 features, bigrams)    │
│   + text_length, word_count, has_logo,   │
│     telecommuting, has_questions         │
│   → scipy.sparse.hstack combined         │
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
│  9. Streamlit Web App                    │
│     Single predict + Bulk CSV +          │
│     Model Dashboard + About              │
└─────────────────────────────────────────┘
```

---

## 📈 Exploratory Data Analysis

### Class Distribution

![Class Distribution](assets/class_distribution.png)

> ⚠️ **Key Observation:** Only **4.5% of jobs are fake** — severe class imbalance. A naive model predicting every job as Real gets **95.5% accuracy but catches zero fraud**. This is why we use F1-Score and Recall instead of accuracy, and apply SMOTE to balance training data.

---

### Word Cloud — Fake vs Real Jobs

![Word Clouds](assets/wordclouds.png)

**Key Observations:**
- Both fake and real jobs use similar words like *work, project, team, customer* — simple keyword matching will NOT work
- Fake jobs use slightly more vague words: *equipment, provide, perform, position*
- Real jobs use more specific words: *responsible, experience, design, website*
- Scammers **deliberately copy real job language** — this is exactly why ML is necessary over keyword rules

---

### Feature Analysis — Word Count and Company Logo

![Feature Analysis](assets/feature_analysis.png)

**Key Observations:**
- **Word Count (left):** Real jobs (green) have longer, more detailed descriptions. Fake jobs (red) tend to be shorter and vaguer — justifies using `word_count` as a feature
- **Company Logo (right):** Real jobs overwhelmingly have company logos. Fake jobs rarely include verified logos — `has_company_logo` is a strong fraud indicator

---

## ⚙️ Feature Engineering

### TF-IDF Vectorization

TF-IDF (Term Frequency — Inverse Document Frequency) converts job posting text into numerical features. It is an improved Bag of Words that assigns **importance weights** to words — common words get low scores, rare but important words get high scores.

| Parameter | Value | Reason |
|-----------|-------|--------|
| `max_features` | 10,000 | Top 10,000 most informative words |
| `ngram_range` | (1, 2) | Single words + bigrams: *"work from"*, *"no experience"* |
| `min_df` | 2 | Ignore words in fewer than 2 documents |
| `sublinear_tf` | True | Log scaling reduces very frequent words |
| Fit on | **Training data only** | Prevents data leakage to test set |

### Engineered Numerical Features

Five additional features combined with TF-IDF using `scipy.sparse.hstack` → **10,005 total features:**

| Feature | Type | Reasoning |
|---------|------|-----------|
| `text_length` | Integer | Fake jobs have shorter, vaguer descriptions |
| `word_count` | Integer | Real jobs use more detailed language |
| `has_company_logo` | Binary (0/1) | Fake jobs rarely have verified company logos |
| `telecommuting` | Binary (0/1) | Some scam postings over-promise remote work |
| `has_questions` | Binary (0/1) | Legitimate jobs include screening questions |

---

## ⚖️ Handling Class Imbalance — SMOTE

**SMOTE** (Synthetic Minority Over-sampling Technique) creates synthetic fake job examples for the training set.

```
Before SMOTE (training set):
  Real jobs :  7,495  (95.5%)
  Fake jobs :    354   (4.5%)   ← too few for model to learn patterns

After SMOTE (training set):
  Real jobs :  7,495  (50%)
  Fake jobs :  7,495  (50%)    ← balanced! model learns fake patterns properly

Test set (never touched by SMOTE):
  Real jobs :  1,875  (95.5%)
  Fake jobs :     88   (4.5%)  ← real-world distribution preserved
```

> ⚠️ **Critical:** SMOTE applied **ONLY on training data**. Applying on test data = data leakage = falsely good results.

---

## 🤖 Models and Results

Three models trained on SMOTE-balanced data, evaluated on the original test set:

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

> All three models have ROC-AUC above **0.985** — excellent. Curves bend sharply toward the top-left, far above the random baseline (dashed line).

### Why XGBoost Was Selected

| Model | Strength | Weakness | Verdict |
|-------|----------|----------|---------|
| Logistic Regression | Highest Recall 0.9191 | Lowest Precision 0.6411 — too many false alarms | ❌ Real jobs wrongly flagged |
| Random Forest | Highest Precision 0.9370 | Lowest Recall 0.6879 — misses 31% of frauds | ❌ Too many frauds slip through |
| **XGBoost** | Best F1 0.8022 — best balance | Middle precision and recall | ✅ **Best practical model** |

> 💡 **For fraud detection, Recall matters more than Precision.** Missing a fake job is more harmful than a false alarm.

---

## 📊 Confusion Matrix

![Confusion Matrix](assets/confusion_matrix.png)

| | Predicted Real | Predicted Fake |
|--|:--:|:--:|
| **Actual Real** | ✅ 3,361 correctly identified | ❌ 42 real jobs wrongly flagged |
| **Actual Fake** | ❌ 29 frauds missed | ✅ 144 frauds correctly caught |

- **144 out of 173 fake jobs caught** → 83.2% Recall
- Only **29 fake jobs missed** — acceptable for real-world use
- Only **42 false alarms** out of 3,403 real jobs — 98.8% Specificity

---

## 🌐 Streamlit Web App

A 4-page interactive web application:

| Page | Feature |
|------|---------|
| 🏠 **Single Job Check** | Enter title + description → prediction + fraud probability gauge + 37 rule-based warning flags |
| 📊 **Bulk CSV Upload** | Upload CSV → batch predictions → download results with risk levels |
| 📈 **Model Performance** | Full dashboard — all 3 model results, bar charts, metrics explanation |
| ℹ️ **About** | Project documentation, GitHub structure, tech stack, key learnings |

**37 rule-based warning flags including:**
```
Fee patterns   : registration fee, joining fee, training fee, pay to start
Earnings       : earn daily, earn weekly, 1 hour, guaranteed income
Indian scams   : lakh per month, work from mobile, urgent hiring, copy paste
Generic scams  : whatsapp, western union, no experience needed
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
│   └── app.py                      ← Streamlit web application (4 pages)
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
👉 **[Open SafeApply App](https://safeapply-fake-job-detection.streamlit.app)**

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

### Option 3 — Train from Scratch (Google Colab)

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
| NLP | `TfidfVectorizer` (sklearn) | TF-IDF with bigrams — 10,000 text features |
| ML Models | `scikit-learn` | Logistic Regression, Random Forest |
| ML Models | `xgboost` | Gradient boosting — best model |
| Imbalance | `imbalanced-learn` | SMOTE oversampling |
| Sparse Matrix | `scipy` | Combine TF-IDF + numerical features |
| Visualization | `matplotlib`, `seaborn` | Charts, ROC curves, confusion matrix |
| Visualization | `wordcloud` | Word clouds for EDA |
| Web App | `streamlit` | Interactive 4-page deployment |
| Model Saving | `joblib` | Save and load model + vectorizer |
| Environment | Google Colab | Cloud GPU notebook for training |

---

## 💡 Key Learnings

1. **Accuracy is misleading** for imbalanced data — F1-Score and Recall are the right metrics for fraud detection
2. **SMOTE only on training data** — applying on test data causes data leakage and falsely good results
3. **TF-IDF must be fit only on training data** — then used to transform both train and test sets
4. **Bigrams are powerful** — *"registration fee"* carries far more signal than individual words
5. **Word clouds proved ML is necessary** — fake and real jobs use similar vocabulary, ruling out simple keyword detection
6. **Recall matters more than Precision** for fraud — missing a scam is worse than a false alarm
7. **Random Forest had highest Precision but lowest Recall** — unsuitable when missing fraud is costly
9. **Hybrid features work** — TF-IDF sparse matrix combined with 5 numerical features via `scipy.sparse.hstack`

---

## ⚠️ Limitations and Future Work

**Current Limitations:**
- Model trained on EMSCAD (Western English job scams) — Indian-specific patterns handled via 37 rule-based flags instead
- ~8,000 rows dropped due to CSV encoding issues — larger clean dataset would improve performance

**Future Improvements:**
- [ ] SHAP Explainability — show which words drove each individual prediction
- [ ] BERT Embeddings — richer semantic features beyond TF-IDF
- [ ] Hyperparameter Tuning with Optuna — Bayesian optimization
- [ ] MLflow Experiment Tracking — professional logging of all runs
- [ ] Label Encoding — add employment_type, industry, required_experience as features
- [ ] Train on Indian job portal data for better local fraud detection
- [ ] Confidence threshold tuning to maximize Recall

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

## 👤 Author

**Priya**
**Bhawana**
**Saloni**


---

<div align="center">



*Built with 🤍 using Python · Scikit-learn · XGBoost · Streamlit*

</div>

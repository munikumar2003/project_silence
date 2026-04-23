# FairSight — AI Bias & Fairness Auditor

> **Detect hidden discrimination in AI systems before they harm real people.**

FairSight is a full-stack platform that inspects datasets and machine learning models for hidden unfairness across demographic groups. Organisations upload any CSV dataset and any trained model, select protected attributes, and receive a comprehensive fairness audit — with visual dashboards, severity-coded flags, three bias mitigation strategies, and a downloadable professional PDF report.

---

## The Problem

AI systems now make life-changing decisions — loan approvals, hiring, medical diagnoses, criminal sentencing. When these systems are trained on historically biased data, they silently amplify discrimination at scale. A model that denies loans to women or misdiagnoses Black patients more often is not just unfair: it is illegal in most jurisdictions (ECOA, Title VII, GDPR, EU AI Act 2024, Equality Act 2010).

Most organisations have no accessible way to detect this before deployment.

---

## What FairSight Does

```
Upload any CSV dataset + any trained model (.pkl/.joblib)
              ↓
    Step 1 — Data Ingestion & Profiling
    Step 2 — Dataset Bias Detection
    Step 3 — Model Fairness Audit
    Step 4 — Bias Mitigation (3 strategies)
    Step 5 — REST API (Flask, 7 endpoints)
    Step 6 — Professional PDF Report (ReportLab)
              ↓
    Risk Score 0–100 · PASS/FAIL per metric · Actionable recommendations
```

---

## Features

### Dataset Bias Detection
- Disparate Impact Ratio (legal 80% rule)
- Statistical Parity Difference
- Chi-square test for statistical significance
- Per-group positive outcome rates
- Class imbalance scoring
- Severity flags: 🔴 SEVERE / 🟡 MODERATE / 🟢 FAIR

### Model Fairness Audit
- **Upload your own model** (`.pkl`/`.joblib`) — audited AS-IS, no retraining
- **Or choose a built-in algorithm** to train and audit:
  - Random Forest, Logistic Regression, Gradient Boosting, Decision Tree, SVM, KNN, Naïve Bayes
- 6 fairness metrics per demographic group:
  - Demographic Parity Difference
  - Equalized Odds (TPR gap)
  - False Positive Rate Parity
  - Predictive Parity (Precision gap)
  - Accuracy Equality
  - Disparate Impact Ratio
- Composite Risk Score 0–100
- 3-stage encoding resolution for external models (prevents zero-score bug)

### Bias Mitigation
Three evidence-based strategies applied and compared:
1. **Pre-Processing: Reweighting** — higher sample weights for disadvantaged groups
2. **Pre-Processing: Resampling** — oversample underrepresented groups
3. **Post-Processing: Threshold Adjustment** — lower decision threshold per group

Best strategy auto-recommended based on lowest remaining fairness gap.

### Dashboard UI
- Domain selector: Loan, Hiring, Medical, Housing, Criminal Justice, Education
- CSV drag-and-drop upload
- Protected attribute toggles (auto-detected)
- Organisation name field (printed on PDF)
- Model source tabs: train new / upload `.pkl`
- Overview: risk gauge, outcome rate bar chart, attribute severity table
- Data Bias, Model Bias, Mitigation, Report pages
- PDF download linked directly to the API

### PDF Report
7-section professional report:
1. Cover page (organisation name, date, risk score)
2. Executive Summary
3. Dataset Profile
4. Dataset Bias Analysis
5. Model Fairness Analysis
6. Mitigation Results
7. Recommendations (legally grounded)

### Supported Domains
Any binary-classification dataset — not just one vertical:

| Domain | Example target columns |
|--------|----------------------|
| Loan / Credit | `loan_approved`, `credit_granted`, `default` |
| Hiring / Recruitment | `hired`, `selected`, `offered` |
| Medical / Diagnosis | `diagnosed`, `positive`, `readmit` |
| Criminal Justice | `recidivism`, `convicted` |
| Housing | `approved`, `evicted` |
| Education | `admitted`, `passed`, `scholarship` |

---

## Project Structure

```
fairsight/
├── run_audit.py                    # Master pipeline — single source of truth
├── requirements.txt
├── README.md
│
├── backend/
│   ├── api/
│   │   └── app.py                  # Flask REST API (7 endpoints)
│   ├── core/
│   │   ├── data_ingestion.py       # Step 1: DataIngestor
│   │   ├── dataset_bias.py         # Step 2: DatasetBiasDetector
│   │   ├── model_bias.py           # Step 3: ModelBiasDetector
│   │   └── mitigation.py           # Step 4: BiasMitigator
│   └── reports/
│       └── report_generator.py     # Step 6: BiasAuditReport (ReportLab PDF)
│
│
├── sample_data/
│   ├── generate_sample.py          # Generates biased loan dataset
│   └── loan_data.csv               # 1,000-row sample with embedded bias
│
├── uploads/                        # User CSV uploads (created at runtime)
├── models/                         # User model uploads (created at runtime)
└── reports_out/                    # Generated PDFs (created at runtime)
```

---

## Quick Start

### Prerequisites

- Python 3.9+
- pip

### 1. Clone & Install

```bash
git clone https://github.com/munikumar2003/project_silence.git
cd fairsight-bias-auditor
pip install -r requirements.txt
```

### 2. Generate the Sample Dataset

```bash
python generate_sample.py
```

This creates `loan_data.csv` — a 1,000-row loan approval dataset with deliberate gender and race bias built in for demonstration.

### 3. Start the API

```bash
python app.py
```

API runs at **http://localhost:5050**

### 4. Open the Dashboard
Open the **https://github.com/munikumar2003/project_silence/frontend**
Open `dashboard.html` in your browser. The dashboard connects to the API automatically and shows a live status indicator.

---

## CLI Usage

Run the full pipeline from the command line without the web UI:

```bash
# Basic — auto-detect sensitive columns, train Random Forest
python run_audit.py --csv loan_data.csv --target loan_approved

# Specify sensitive attributes and algorithm
python run_audit.py \
  --csv loan_data.csv \
  --target loan_approved \
  --sensitive gender race \
  --algorithm gradient_boosting \
  --org "Acme Bank" \
  --output my_audit_report.pdf

# Audit your own pre-trained model (no retraining)
python run_audit.py \
  --csv your_data.csv \
  --target hired \
  --sensitive gender ethnicity \
  --model your_model.pkl \
  --org "GlobalTech HR"
```

### CLI Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--csv` | ✅ | — | Path to any CSV dataset |
| `--target` | ✅ | — | Target/outcome column name |
| `--sensitive` | ❌ | auto-detect | Protected attribute columns (space-separated) |
| `--algorithm` | ❌ | `random_forest` | Algorithm to train (see list below) |
| `--model` | ❌ | — | Path to your saved `.pkl`/`.joblib` model |
| `--org` | ❌ | `Organization` | Organisation name for the PDF report |
| `--output` | ❌ | `bias_audit_report.pdf` | Output PDF path |

### Supported Algorithms

| Key | Algorithm |
|-----|-----------|
| `random_forest` | Random Forest |
| `logistic_regression` | Logistic Regression |
| `gradient_boosting` | Gradient Boosting |
| `decision_tree` | Decision Tree |
| `svm` | Support Vector Machine |
| `knn` | K-Nearest Neighbours |
| `naive_bayes` | Naïve Bayes |

---

## REST API Reference

Base URL: `http://localhost:8080`

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/algorithms` | List all supported algorithm keys and labels |
| `GET` | `/api/sample` | Load the built-in biased loan dataset |
| `POST` | `/api/upload` | Upload a CSV dataset |
| `POST` | `/api/upload_model` | Upload a pre-trained `.pkl`/`.joblib` model |
| `POST` | `/api/audit` | Run the full bias audit pipeline (Steps 1–3) |
| `POST` | `/api/mitigate` | Run all 3 mitigation strategies (Step 4) |
| `GET` | `/api/report/<sid>` | Generate and download the PDF report |

### Example: Full Audit Flow

```bash
# 1. Load sample dataset
curl http://localhost:8080/api/sample
# Returns: { "session_id": "sample_ab12", "columns": [...], ... }

# 2. Run audit
curl -X POST http://localhost:8080/api/audit \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "sample_ab12",
    "target_col": "loan_approved",
    "sensitive_cols": ["gender", "race"],
    "algorithm": "random_forest",
    "org_name": "Acme Bank"
  }'

# 3. Run mitigation
curl -X POST http://localhost:8080/api/mitigate \
  -H "Content-Type: application/json" \
  -d '{ "session_id": "sample_ab12" }'

# 4. Download PDF report
curl http://localhost:8080/api/report/sample_ab12 -o audit_report.pdf
```

### Example: Upload Your Own CSV and Model

```bash
# Upload CSV
SESSION=$(curl -s -X POST http://localhost:8080/api/upload \
  -F "file=@your_data.csv" | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")

# Upload model
curl -X POST http://localhost:8080/api/upload_model \
  -F "session_id=$SESSION" \
  -F "file=@your_model.pkl"

# Audit
curl -X POST http://localhost:8080/api/audit \
  -H "Content-Type: application/json" \
  -d "{
    \"session_id\": \"$SESSION\",
    \"target_col\": \"hired\",
    \"sensitive_cols\": [\"gender\", \"ethnicity\"],
    \"org_name\": \"GlobalTech HR\"
  }"
```

---

## Fairness Metrics Explained

| Metric | Threshold | What it measures |
|--------|-----------|-----------------|
| **Disparate Impact Ratio** | ≥ 0.80 | Ratio of positive outcome rates: worst group ÷ best group. Below 0.80 is legally problematic under the EEOC 80% rule. |
| **Demographic Parity Diff** | ≤ 10% | Gap in positive prediction rates between most and least favoured group. |
| **Equalized Odds (TPR gap)** | ≤ 10% | Gap in true positive rates — is the model equally good at identifying qualified individuals across groups? |
| **FPR Parity** | ≤ 10% | Gap in false positive rates — does the model make equally many false-positive errors across groups? |
| **Predictive Parity** | ≤ 10% | Gap in precision (positive predictive value) across groups. |
| **Accuracy Equality** | ≤ 5% | Gap in overall accuracy — is the model equally reliable for all demographics? |

### Risk Score

The composite Risk Score (0–100) is computed as:

```
per_attribute_score = (1 - Disparate_Impact) × 50 + Demographic_Parity_Diff × 100
Risk_Score = mean(per_attribute_scores)

LOW    →  0–30
MEDIUM → 31–60
HIGH   → 61–100
```

---

## How External Model Auditing Works

When you upload a `.pkl`/`.joblib` model, FairSight uses a 3-stage encoding resolution to obtain valid predictions without retraining:

1. **Attempt 1 — Raw data**: Feed the original (un-encoded) data directly. Works for sklearn `Pipeline` objects that include their own preprocessing.
2. **Attempt 2 — Numeric encoding**: Encode features numerically (with and without scaling). If accuracy is suspiciously low, try flipping the predictions — this catches label encoding mismatches where the user's encoder and ours assigned 0/1 in opposite order.
3. **Attempt 3 — Best effort**: If accuracy remains low, audit fairness anyway with a warning that the model may have unusual preprocessing requirements.

> **Tip:** For best results, save your model as a full `sklearn.pipeline.Pipeline` that includes all preprocessing steps:
> ```python
> from sklearn.pipeline import Pipeline
> from sklearn.preprocessing import StandardScaler
> from sklearn.ensemble import RandomForestClassifier
>
> pipeline = Pipeline([
>     ('scaler', StandardScaler()),
>     ('model', RandomForestClassifier())
> ])
> pipeline.fit(X_train, y_train)
>
> import joblib
> joblib.dump(pipeline, 'my_model.pkl')
> ```

---

## Sample Output

Running the pipeline on the built-in loan dataset (1,000 records, embedded gender and race bias):

```
════════════════════════════════════════════════════════════
  AI BIAS AUDITOR  |  Acme Bank
════════════════════════════════════════════════════════════

📂  STEP 1 — Data Ingestion & Profiling
[✔] Loaded dataset: 1,000 rows × 8 columns
[✔] Sensitive columns detected: ['gender', 'race', 'age']
[✔] Target column set to: 'loan_approved'

🔍  STEP 2 — Dataset Bias Detection
  🟡 [MODERATE] gender  |  DI=0.773  gap=11.5%  worst='Female'
  🟡 [MODERATE] race    |  DI=0.728  gap=13.8%  worst='Black'

🤖  STEP 3 — Model Fairness Audit
[✔] Model trained (Random Forest). Accuracy: 0.587
  🔴 [SEVERE] gender    |  DI=0.571  gap=23.7%  worst='Female'
  🔴 [SEVERE] race      |  DI=0.609  gap=20.9%  worst='Black'

📊  RISK: MEDIUM (42.8/100)  |  14 flags raised
```

The model amplified the raw data bias — the gender gap grew from 11.5% (in data) to 23.7% (in model predictions).

---

## Technology Stack

### Backend
| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.9+ | Core language |
| Flask | 3.0 | REST API |
| Flask-CORS | 4.0 | Cross-origin support |
| scikit-learn | 1.3+ | ML algorithms, metrics, preprocessing |
| pandas | 2.0+ | Data loading and manipulation |
| numpy | 1.26+ | Array operations |
| scipy | 1.11+ | Chi-square statistical testing |
| joblib | 1.3+ | Model serialisation |
| ReportLab | 4.0 | PDF generation |

### Frontend
| Technology | Purpose |
|-----------|---------|
| HTML5 / CSS3 / Vanilla JS | Single-file dashboard, no build step |
| Chart.js 4.4 | Bar charts, canvas risk gauge |
| Google Fonts | DM Serif Display, DM Mono, Outfit |
| Fetch API | All REST API calls |

---

## Roadmap

- [ ] **Google Gemini integration** — plain-English explanations of every bias flag, jurisdiction-specific legal guidance, conversational Q&A
- [ ] **Google Cloud Run deployment** — containerised Flask API, Cloud Storage for files, Firebase Auth
- [ ] **CI/CD integration** — GitHub Actions hook to fail builds when Disparate Impact drops below 0.80
- [ ] **Bias drift monitoring** — automated quarterly re-audits via Cloud Scheduler
- [ ] **Multi-class & regression support** — extend beyond binary classification
- [ ] **Vertex AI integration** — large-scale model auditing
- [ ] **SHAP explanations** — individual-level fairness via feature importance

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---


## Acknowledgements

- [IBM AIF360](https://github.com/Trusted-AI/AIF360) and [Fairlearn](https://fairlearn.org/) — foundational fairness metric definitions
- [EEOC Uniform Guidelines](https://www.eeoc.gov/laws/guidance/questions-and-answers-clarify-and-provide-common-interpretation-uniform-guidelines) — 80% rule threshold
- Built for [Google Solution Challenge 2026](https://developers.google.com/community/gdsc-solution-challenge)

---

*FairSight — Because fairness is not optional when AI makes decisions about people.*

"""
run_audit.py — Master Pipeline
================================
Single source of truth for both the CLI and the Flask API.

Inputs
------
  csv_path        : any CSV dataset (loan, hiring, medical, etc.)
  target_col      : binary outcome column
  sensitive_cols  : demographic columns to audit
  algorithm       : algorithm key to train (ignored when model_path given)
  model_path      : user's saved .pkl/.joblib — audited AS-IS if provided
  org_name        : written on the PDF cover page
  generate_pdf    : True for CLI, False when called by API
"""

import sys, os, argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from sklearn.metrics import accuracy_score

from data_ingestion import DataIngestor
from dataset_bias   import DatasetBiasDetector
from model_bias     import (ModelBiasDetector,
                                          ALGORITHM_DISPLAY_NAMES,
                                          SUPPORTED_ALGORITHMS)
from mitigation     import BiasMitigator


# ── Shared serialiser ─────────────────────────────────────────────────
def serialize(obj):
    if isinstance(obj, dict):  return {k: serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [serialize(i) for i in obj]
    if isinstance(obj, bool):  return bool(obj)
    if hasattr(obj, "item"):   return obj.item()
    if isinstance(obj, float) and obj != obj: return None
    return obj


# ── Helpers ───────────────────────────────────────────────────────────
def _risk_score(model_bias: dict) -> dict:
    scores = []
    for res in model_bias.values():
        if "error" in res:
            continue
        di = res.get("disparate_impact_ratio", 1.0) or 1.0
        dp = res.get("demographic_parity_difference", 0.0) or 0.0
        scores.append(min((1 - di) * 50 + dp * 100, 100))
    overall = round(sum(scores) / len(scores), 1) if scores else 0.0
    level   = "HIGH" if overall > 60 else "MEDIUM" if overall > 30 else "LOW"
    return {"score": overall, "level": level}


def _summary(dataset_bias, model_bias, risk, org_name, algo_label) -> dict:
    sev  = [c for c, r in model_bias.items() if r.get("severity") == "SEVERE"]
    mod  = [c for c, r in model_bias.items() if r.get("severity") == "MODERATE"]
    total = (sum(len(r.get("flags", [])) for r in model_bias.values()) +
             sum(len(r.get("flags", [])) for r in dataset_bias.values()))
    rec = ("URGENT: Severe bias detected. Do not deploy without mitigation."
           if risk["level"] == "HIGH" else
           "WARNING: Moderate bias detected. Apply mitigation before deployment."
           if risk["level"] == "MEDIUM" else
           "System appears relatively fair. Continue monitoring for drift.")
    return {
        "risk_level":          risk["level"],
        "risk_score":          risk["score"],
        "total_flags":         total,
        "severe_attributes":   sev,
        "moderate_attributes": mod,
        "recommendation":      rec,
        "org_name":            org_name,
        "algorithm":           algo_label,
    }


# ══════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════
def run_pipeline(
    csv_path:       str,
    target_col:     str,
    sensitive_cols: list = None,
    algorithm:      str  = "random_forest",
    model_path:     str  = None,
    org_name:       str  = "Organization",
    output_pdf:     str  = None,
    generate_pdf:   bool = True,
) -> dict:

    print("\n" + "═" * 60)
    print(f"  AI BIAS AUDITOR  |  {org_name}")
    print("═" * 60)

    # ── STEP 1: Ingest ────────────────────────────────────────────
    print("\n📂  STEP 1 — Data Ingestion & Profiling")
    ingestor = DataIngestor()
    ingestor.load_csv(csv_path)
    ingestor.detect_sensitive_columns(user_defined=sensitive_cols or [])
    ingestor.set_target_column(target_col)
    profile  = ingestor.profile_dataset()
    df       = ingestor.get_clean_df()

    if sensitive_cols:
        use_cols = [c for c in sensitive_cols if c in df.columns]
    else:
        auto     = ingestor.sensitive_cols
        use_cols = [c for c in auto if c != "age"] or auto

    mode_str = f"External — {model_path}" if model_path else f"Train — {algorithm}"
    print(f"   Rows      : {profile['shape']['rows']:,}")
    print(f"   Target    : {target_col}")
    print(f"   Sensitive : {use_cols}")
    print(f"   Model     : {mode_str}")

    # ── STEP 2: Dataset Bias ──────────────────────────────────────
    print("\n🔍  STEP 2 — Dataset Bias Detection")
    ds_detector  = DatasetBiasDetector(df=df, sensitive_cols=use_cols, target_col=target_col)
    dataset_bias = ds_detector.run_all_checks()

    # ── STEP 3: Model Bias ────────────────────────────────────────
    print("\n🤖  STEP 3 — Model Fairness Audit")
    md = ModelBiasDetector(
        df             = df,
        sensitive_cols = use_cols,
        target_col     = target_col,
        model_path     = model_path,
        algorithm      = algorithm,
    )

    if model_path:
        # Audit the user's OWN model — no retraining
        md.load_and_audit_external_model()
        algo_label   = type(md.model).__name__
        source_label = f"External — {algo_label}"
    else:
        # Train from chosen algorithm
        md.train_model()
        algo_label   = ALGORITHM_DISPLAY_NAMES.get(algorithm, algorithm)
        source_label = f"Trained — {algo_label}"

    overall_acc = accuracy_score(md.y_test, md.y_pred)
    model_bias  = md.compute_fairness_metrics()

    model_info = {
        "algorithm":    algo_label,
        "source":       source_label,
        "accuracy":     round(float(overall_acc), 4),
        "mode":         "external" if model_path else "trained",
        "model_path":   model_path or "",
    }

    # ── Risk + Summary ────────────────────────────────────────────
    risk    = _risk_score(model_bias)
    summary = _summary(dataset_bias, model_bias, risk, org_name, algo_label)

    print(f"\n📊  RISK: {risk['level']} ({risk['score']}/100) | "
          f"{summary['total_flags']} flags | model acc={overall_acc:.3f}")

    result = {
        "dataset_profile": serialize(profile),
        "dataset_bias":    serialize(dataset_bias),
        "model_bias":      serialize(model_bias),
        "model_info":      model_info,
        "risk_score":      risk,
        "summary":         summary,
    }

    # ── STEP 4 + 6 (PDF) — CLI or explicit request ───────────────
    if generate_pdf:
        print("\n🛠   STEP 4 — Bias Mitigation")
        mit = BiasMitigator(df=df, sensitive_cols=use_cols, target_col=target_col)
        r1, r2, r3 = mit.mitigate_reweighting(), mit.mitigate_resampling(), mit.mitigate_threshold_adjustment()
        best = mit.get_best_strategy()
        mitigation = {
            "mitigation_results": {
                "reweighting":          serialize(r1),
                "resampling":           serialize(r2),
                "threshold_adjustment": serialize(r3),
            },
            "recommended_strategy": best,
        }
        result["mitigation_results"] = mitigation

        print("\n📄  STEP 6 — Generating PDF Report")
        pdf_path = output_pdf or os.path.join(ROOT, "bias_audit_report.pdf")
        from report_generator import BiasAuditReport
        BiasAuditReport(result, mitigation, org_name, pdf_path).generate()
        print(f"\n{'═'*60}\n  ✔ DONE  |  PDF → {pdf_path}\n{'═'*60}\n")

    return result


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="AI Bias Auditor")
    p.add_argument("--csv",       required=True)
    p.add_argument("--target",    required=True)
    p.add_argument("--sensitive", nargs="+", default=None)
    p.add_argument("--algorithm", default="random_forest",
                   choices=list(SUPPORTED_ALGORITHMS.keys()))
    p.add_argument("--model",     default=None, help=".pkl/.joblib path")
    p.add_argument("--org",       default="Organization")
    p.add_argument("--output",    default=None)
    args = p.parse_args()

    run_pipeline(
        csv_path       = args.csv,
        target_col     = args.target,
        sensitive_cols = args.sensitive,
        algorithm      = args.algorithm,
        model_path     = args.model,
        org_name       = args.org,
        output_pdf     = args.output,
        generate_pdf   = True,
    )
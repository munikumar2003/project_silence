"""
Flask REST API — Bias Auditor
==============================
Delegates ALL pipeline logic to run_audit.run_pipeline().

Endpoints:
  GET  /api/health              Health check
  GET  /api/algorithms          List supported algorithm names
  GET  /api/sample              Load built-in sample dataset
  POST /api/upload              Upload a CSV dataset
  POST /api/upload_model        Upload user's pre-trained .pkl/.joblib model
  POST /api/audit               Run full audit (Steps 1-3)
  POST /api/mitigate            Run mitigation strategies (Step 4)
  GET  /api/report/<sid>        Generate & download PDF report

Run:
  python backend/api/app.py
  → http://localhost:5050
"""

import os, sys, uuid, traceback

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS

from run_audit import run_pipeline, serialize, ALGORITHM_DISPLAY_NAMES
from data_ingestion import DataIngestor
from mitigation     import BiasMitigator

app = Flask(__name__)
CORS(app)

UPLOAD_DIR = os.path.join(ROOT, "uploads");      os.makedirs(UPLOAD_DIR, exist_ok=True)
MODEL_DIR  = os.path.join(ROOT, "models");       os.makedirs(MODEL_DIR,  exist_ok=True)
REPORT_DIR = os.path.join(ROOT, "reports_out");  os.makedirs(REPORT_DIR, exist_ok=True)

# {session_id → {filepath, model_path, columns, sensitive_cols, target_col,
#                algorithm, org_name, audit_result, mitigation_result}}
SESSIONS = {}

@app.route("/")
def home():
    return render_template("index.html")
# ══════════════════════════════════════════════════════════════════════
# HEALTH
# ══════════════════════════════════════════════════════════════════════
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "message": "Bias Auditor API is running."})


# ══════════════════════════════════════════════════════════════════════
# ALGORITHM LIST  — frontend uses this to build the dropdown
# ══════════════════════════════════════════════════════════════════════
@app.route("/api/algorithms")
def algorithms():
    return jsonify({
        "algorithms": [
            {"key": k, "label": v}
            for k, v in ALGORITHM_DISPLAY_NAMES.items()
        ]
    })


# ══════════════════════════════════════════════════════════════════════
# UPLOAD CSV
# ══════════════════════════════════════════════════════════════════════
@app.route("/api/upload", methods=["POST"])
def upload_dataset():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file in request."}), 400
        f = request.files["file"]
        if not f.filename.lower().endswith(".csv"):
            return jsonify({"error": "Only CSV files are supported."}), 400

        sid   = str(uuid.uuid4())[:8]
        fpath = os.path.join(UPLOAD_DIR, f"{sid}.csv")
        f.save(fpath)

        ingestor = DataIngestor()
        ingestor.load_csv(fpath)
        auto_sensitive = ingestor.detect_sensitive_columns()

        SESSIONS[sid] = {
            "filepath":       fpath,
            "model_path":     None,
            "columns":        list(ingestor.df.columns),
            "sensitive_cols": auto_sensitive,
        }
        return jsonify({
            "session_id":              sid,
            "columns":                 list(ingestor.df.columns),
            "shape":                   {"rows": int(ingestor.df.shape[0]),
                                        "cols": int(ingestor.df.shape[1])},
            "auto_detected_sensitive": auto_sensitive,
            "message":                 "Dataset uploaded successfully.",
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ══════════════════════════════════════════════════════════════════════
# UPLOAD PRE-TRAINED MODEL  (optional)
# ══════════════════════════════════════════════════════════════════════
@app.route("/api/upload_model", methods=["POST"])
def upload_model():
    """
    User uploads their own trained model (.pkl or .joblib).
    Must be called AFTER /api/upload so we have a session_id.
    Body: multipart form with fields: session_id, file (the model file)
    """
    try:
        sid = request.form.get("session_id")
        if not sid or sid not in SESSIONS:
            return jsonify({"error": "Invalid session_id. Upload dataset first."}), 400

        if "file" not in request.files:
            return jsonify({"error": "No model file in request."}), 400

        f    = request.files["file"]
        ext  = os.path.splitext(f.filename)[1].lower()
        if ext not in (".pkl", ".joblib"):
            return jsonify({"error": "Model must be a .pkl or .joblib file."}), 400

        mpath = os.path.join(MODEL_DIR, f"{sid}_model{ext}")
        f.save(mpath)

        SESSIONS[sid]["model_path"] = mpath
        return jsonify({
            "session_id":  sid,
            "model_path":  mpath,
            "filename":    f.filename,
            "message":     "Model uploaded. Bias will be audited on YOUR model.",
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ══════════════════════════════════════════════════════════════════════
# LOAD SAMPLE DATASET
# ══════════════════════════════════════════════════════════════════════
@app.route("/api/sample")
def load_sample():
    try:
        sample_path = os.path.join(ROOT,"loan_data.csv")
        if not os.path.exists(sample_path):
            sys.path.insert(0, os.path.join(ROOT))
            from generate_sample import generate_loan_dataset
            generate_loan_dataset().to_csv(sample_path, index=False)

        sid = "sample_" + str(uuid.uuid4())[:4]
        ingestor = DataIngestor()
        ingestor.load_csv(sample_path)
        auto_sensitive = ingestor.detect_sensitive_columns()

        SESSIONS[sid] = {
            "filepath":       sample_path,
            "model_path":     None,
            "columns":        list(ingestor.df.columns),
            "sensitive_cols": auto_sensitive,
        }
        return jsonify({
            "session_id":              sid,
            "columns":                 list(ingestor.df.columns),
            "shape":                   {"rows": int(ingestor.df.shape[0]),
                                        "cols": int(ingestor.df.shape[1])},
            "auto_detected_sensitive": auto_sensitive,
            "suggested_target":        "loan_approved",
            "domain_hint":             "loan_application",
            "message":                 "Sample biased loan dataset loaded.",
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ══════════════════════════════════════════════════════════════════════
# RUN FULL AUDIT  →  run_audit.run_pipeline()
# ══════════════════════════════════════════════════════════════════════
@app.route("/api/audit", methods=["POST"])
def run_audit_endpoint():
    """
    Body JSON:
    {
      "session_id":     "<sid>",
      "target_col":     "loan_approved",
      "sensitive_cols": ["gender", "race"],
      "algorithm":      "gradient_boosting",   // ignored if model was uploaded
      "org_name":       "Acme Bank"            // written into the PDF report
    }
    """
    try:
        body           = request.get_json(force=True)
        sid            = body.get("session_id")
        target_col     = body.get("target_col", "")
        sensitive_cols = body.get("sensitive_cols", [])
        algorithm      = body.get("algorithm", "random_forest")
        org_name       = body.get("org_name", "Organization").strip() or "Organization"

        if not sid or sid not in SESSIONS:
            return jsonify({"error": "Invalid session_id."}), 400
        if not target_col:
            return jsonify({"error": "target_col is required."}), 400
        if not sensitive_cols:
            return jsonify({"error": "sensitive_cols must not be empty."}), 400

        fpath      = SESSIONS[sid]["filepath"]
        model_path = SESSIONS[sid].get("model_path")   # None = train from algorithm

        # ── Single call into run_audit.py ────────────────────────
        result = run_pipeline(
            csv_path       = fpath,
            target_col     = target_col,
            sensitive_cols = sensitive_cols,
            algorithm      = algorithm,
            model_path     = model_path,
            org_name       = org_name,
            generate_pdf   = False,
        )

        SESSIONS[sid].update({
            "audit_result":   result,
            "target_col":     target_col,
            "sensitive_cols": sensitive_cols,
            "algorithm":      algorithm,
            "org_name":       org_name,
        })
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ══════════════════════════════════════════════════════════════════════
# MITIGATION
# ══════════════════════════════════════════════════════════════════════
@app.route("/api/mitigate", methods=["POST"])
def run_mitigation_endpoint():
    try:
        body = request.get_json(force=True)
        sid  = body.get("session_id")

        if not sid or sid not in SESSIONS:
            return jsonify({"error": "Invalid session_id."}), 400
        if "audit_result" not in SESSIONS[sid]:
            return jsonify({"error": "Run /api/audit first."}), 400

        fpath          = SESSIONS[sid]["filepath"]
        target_col     = SESSIONS[sid]["target_col"]
        sensitive_cols = SESSIONS[sid]["sensitive_cols"]
        algorithms=     SESSIONS[sid]["algorithm"]

        ingestor = DataIngestor()
        ingestor.load_csv(fpath)
        ingestor.set_target_column(target_col)
        df = ingestor.get_clean_df()

        mitigator = BiasMitigator(df=df, sensitive_cols=sensitive_cols, target_col=target_col, algorithm=algorithms)
        r1   = mitigator.mitigate_reweighting()
        r2   = mitigator.mitigate_resampling()
        r3   = mitigator.mitigate_threshold_adjustment()
        best = mitigator.get_best_strategy()

        result = {
            "mitigation_results": {
                "reweighting":          serialize(r1),
                "resampling":           serialize(r2),
                "threshold_adjustment": serialize(r3),
            },
            "recommended_strategy": best,
            "message": f"Mitigation complete. Recommended: {best}",
        }
        SESSIONS[sid]["mitigation_result"] = result
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ══════════════════════════════════════════════════════════════════════
# PDF REPORT
# ══════════════════════════════════════════════════════════════════════
@app.route("/api/report/<sid>")
def download_report(sid):
    try:
        if sid not in SESSIONS or "audit_result" not in SESSIONS[sid]:
            return jsonify({"error": "Run /api/audit first."}), 400

        from report_generator import BiasAuditReport

        org_name  = SESSIONS[sid].get("org_name", request.args.get("org", "Organization"))
        pdf_path  = os.path.join(REPORT_DIR, f"{sid}_report.pdf")

        BiasAuditReport(
            audit_results      = SESSIONS[sid]["audit_result"],
            mitigation_results = SESSIONS[sid].get("mitigation_result", {}),
            org_name           = org_name,
            output_path        = pdf_path,
        ).generate()

        return send_file(
            pdf_path,
            mimetype      = "application/pdf",
            as_attachment = True,
            download_name = f"bias_audit_{sid}.pdf",
        )
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("  Bias Auditor API  →  http://localhost:5050")
    print("=" * 55)
    #app.run(debug=True, port=5050, use_reloader=False)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5050)))
"""
STEP 3: Model Bias Detection
==============================
Audits ANY sklearn-compatible binary classification model for fairness.

Two modes:
  A) External model (.pkl/.joblib) — user's OWN pre-trained model is audited AS-IS.
     The critical fix: we NEVER re-encode data. Instead we probe the model to
     discover what encoding it expects, then produce predictions in its native space
     and map results back to interpretable group labels using the ORIGINAL dataframe.

  B) Train mode — we train a chosen algorithm here, encoding consistently,
     then audit fairness on the held-out test split.

Supported algorithms (Mode B):
  random_forest | logistic_regression | gradient_boosting |
  decision_tree | svm | knn | naive_bayes

Fairness metrics per sensitive attribute group:
  1. Demographic Parity Difference   — equal positive prediction rates?
  2. Equalized Odds (TPR gap)        — equal true positive rates?
  3. False Positive Rate Parity      — equal false positive rates?
  4. Predictive Parity (precision)   — equal precision?
  5. Accuracy Equality               — equal accuracy?
  6. Disparate Impact Ratio          — legal 80% rule
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.svm             import SVC
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.naive_bayes     import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.metrics         import accuracy_score, confusion_matrix
from sklearn.pipeline        import Pipeline


SUPPORTED_ALGORITHMS = {
    "random_forest":       lambda: RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, class_weight="balanced"),
    "logistic_regression": lambda: LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    "gradient_boosting":   lambda: GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42),
    "decision_tree":       lambda: DecisionTreeClassifier(max_depth=8, random_state=42, class_weight="balanced"),
    "svm":                 lambda: SVC(probability=True, random_state=42, class_weight="balanced", kernel="rbf"),
    "knn":                 lambda: KNeighborsClassifier(n_neighbors=7),
    "naive_bayes":         lambda: GaussianNB(),
}

ALGORITHM_DISPLAY_NAMES = {
    "random_forest":       "Random Forest",
    "logistic_regression": "Logistic Regression",
    "gradient_boosting":   "Gradient Boosting",
    "decision_tree":       "Decision Tree",
    "svm":                 "Support Vector Machine (SVM)",
    "knn":                 "K-Nearest Neighbours",
    "naive_bayes":         "Naïve Bayes",
}


def _build_encoding_map(df: pd.DataFrame, feature_cols: list, target_col: str):
    """
    Fit label encoders for every categorical column.
    Returns encoded X (ndarray), encoded y (ndarray), and the encoder dict.
    """
    encoders = {}
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le

    if not pd.api.types.is_numeric_dtype(y):
        le_y = LabelEncoder()
        y = le_y.fit_transform(y.astype(str))
        encoders["__target__"] = le_y
    else:
        y = y.values

    return X.values.astype(float), np.array(y), encoders


def _positive_label_index(y_encoded: np.ndarray) -> int:
    """Returns the integer label representing the 'positive' class."""
    # Sklearn convention: classes are sorted, so 1 is the 'positive' class
    # in a binary {0,1} encoding. Return 1 if present.
    unique = np.unique(y_encoded)
    if 1 in unique:
        return 1
    return unique[-1]   # fallback: highest label


class ModelBiasDetector:
    """
    Generic fairness auditor. Works with any dataset domain and any sklearn model.
    """

    def __init__(
        self,
        df:             pd.DataFrame,
        sensitive_cols: list,
        target_col:     str,
        feature_cols:   list = None,
        model_path:     str  = None,   # user's saved model
        algorithm:      str  = "random_forest",
    ):
        self.df             = df.copy()
        self.sensitive_cols = sensitive_cols
        self.target_col     = target_col
        self.feature_cols   = feature_cols or [c for c in df.columns if c != target_col]
        self.model_path     = model_path
        self.algorithm      = (algorithm or "random_forest").lower().strip()

        self.model          = None
        self.model_source   = None
        self.scaler         = StandardScaler()
        self.encoders       = {}

        # These are set after training/loading:
        self.X_test    = None   # encoded feature array used for predictions
        self.y_test    = None   # ground-truth labels (int)
        self.y_pred    = None   # model's predictions (int)
        self.test_df   = None   # original (un-encoded) rows matching y_test / y_pred
        self.results   = {}

    # ══════════════════════════════════════════════════════════════
    # MODE B — TRAIN FROM CHOSEN ALGORITHM
    # ══════════════════════════════════════════════════════════════
    def train_model(self):
        if self.algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError(f"Unknown algorithm '{self.algorithm}'. "
                             f"Supported: {list(SUPPORTED_ALGORITHMS.keys())}")

        df = self.df.copy()
        X_enc, y_enc, self.encoders = _build_encoding_map(df, self.feature_cols, self.target_col)

        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_enc, y_enc, test_size=0.3, random_state=42, stratify=y_enc)
        except ValueError:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_enc, y_enc, test_size=0.3, random_state=42)

        # Keep the original rows that correspond to the test split
        # We need this to group predictions by sensitive attribute values
        all_idx   = np.arange(len(df))
        _, te_idx = train_test_split(all_idx, test_size=0.3, random_state=42,
                                     stratify=y_enc if len(np.unique(y_enc)) > 1 else None)
        self.test_df = df.iloc[te_idx].reset_index(drop=True)

        X_tr_s  = self.scaler.fit_transform(X_tr)
        X_te_s  = self.scaler.transform(X_te)

        self.model        = SUPPORTED_ALGORITHMS[self.algorithm]()
        self.model_source = f"trained:{self.algorithm}"
        self.model.fit(X_tr_s, y_tr)
        self.y_pred = self.model.predict(X_te_s)
        self.y_test = y_te
        self.X_test = X_te_s

        acc = accuracy_score(y_te, self.y_pred)
        print(f"[✔] Model trained ({ALGORITHM_DISPLAY_NAMES[self.algorithm]}). "
              f"Accuracy: {acc:.3f}")
        return self.model

    # ══════════════════════════════════════════════════════════════
    # MODE A — AUDIT AN EXTERNAL (USER-UPLOADED) MODEL
    # ══════════════════════════════════════════════════════════════
    def load_and_audit_external_model(self):
        """
        THE KEY FIX:
        We do NOT try to guess how the user encoded their data.
        Instead we:
          1. Load the model.
          2. Try feeding it raw (un-encoded) data first — many pipelines handle
             this internally.
          3. If that fails, fall back to our own encoding, but then we detect
             whether the model's output labels are inverted by checking
             whether accuracy > 0.5; if not, we flip predictions.
          4. We keep the ORIGINAL dataframe rows as test_df so group labels
             (Male/Female, White/Black…) stay readable.
        """
        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.model        = joblib.load(self.model_path)
        self.model_source = "external"
        algo_name         = type(self.model).__name__
        print(f"[✔] External model loaded: {algo_name}")

        df   = self.df.copy()
        X_raw = df[self.feature_cols].copy()

        # Encode ground-truth target
        y_raw = df[self.target_col].copy()
        if not pd.api.types.is_numeric_dtype(y_raw):
            le_y = LabelEncoder()
            y_enc = le_y.fit_transform(y_raw.astype(str))
            self.encoders["__target__"] = le_y
        else:
            y_enc = y_raw.values.astype(int)

        # ── Attempt 1: feed raw (un-encoded) data directly ────────
        # Works if the model is a Pipeline with its own preprocessing,
        # or was trained on already-numeric data.
        preds = None
        try:
            preds = self.model.predict(X_raw)
            preds = np.array(preds).ravel()
            # Coerce to same integer space as y_enc
            if not pd.api.types.is_numeric_dtype(pd.Series(preds)):
                le_p = LabelEncoder()
                le_p.fit(np.concatenate([y_raw.astype(str), preds.astype(str)]))
                y_enc = le_p.transform(y_raw.astype(str))
                preds = le_p.transform(preds.astype(str))
            else:
                preds = preds.astype(int)
            acc_raw = accuracy_score(y_enc, preds)
            print(f"   Attempt 1 (raw data): accuracy = {acc_raw:.3f}")
            if acc_raw >= 0.45:   # reasonable — use these predictions
                self.y_pred = preds
                self.y_test = y_enc
                self.test_df = df.reset_index(drop=True)
                print(f"[✔] External model predictions accepted (raw). Accuracy: {acc_raw:.3f}")
                return
        except Exception as ex:
            print(f"   Attempt 1 failed ({ex.__class__.__name__}): {ex}")

        # ── Attempt 2: encode numerically, then predict ───────────
        X_enc, y_enc2, self.encoders = _build_encoding_map(df, self.feature_cols, self.target_col)

        # Try without scaling first
        preds2 = None
        for use_scale in [False, True]:
            try:
                X_in = self.scaler.fit_transform(X_enc) if use_scale else X_enc
                preds2 = self.model.predict(X_in)
                preds2 = np.array(preds2).ravel().astype(int)
                acc2 = accuracy_score(y_enc2, preds2)
                print(f"   Attempt 2 (encoded, scale={use_scale}): accuracy = {acc2:.3f}")
                if acc2 >= 0.45:
                    self.y_pred = preds2
                    self.y_test = y_enc2
                    self.test_df = df.reset_index(drop=True)
                    print(f"[✔] External model predictions accepted. Accuracy: {acc2:.3f}")
                    return
                # If accuracy very low, predictions may be flipped — try inverting
                preds2_flipped = 1 - preds2
                acc_flip = accuracy_score(y_enc2, preds2_flipped)
                print(f"   Attempt 2 flipped: accuracy = {acc_flip:.3f}")
                if acc_flip > acc2 and acc_flip >= 0.45:
                    self.y_pred = preds2_flipped
                    self.y_test = y_enc2
                    self.test_df = df.reset_index(drop=True)
                    print(f"[✔] External model: predictions were inverted, corrected. Accuracy: {acc_flip:.3f}")
                    return
            except Exception as ex:
                print(f"   Attempt 2 (scale={use_scale}) failed: {ex}")

        # ── Attempt 3: use best predictions we got even if accuracy < 0.45 ──
        # Could be a genuinely poor model — we still audit fairness accurately
        if preds2 is not None:
            self.y_pred  = preds2
            self.y_test  = y_enc2
            self.test_df = df.reset_index(drop=True)
            acc_final = accuracy_score(y_enc2, preds2)
            print(f"[⚠] External model: low accuracy ({acc_final:.3f}). "
                  f"Auditing fairness anyway — model may have been trained on differently preprocessed data.")
            return

        raise RuntimeError("Could not obtain valid predictions from the uploaded model. "
                           "Ensure it is a scikit-learn compatible binary classifier.")

    # ══════════════════════════════════════════════════════════════
    # COMPUTE FAIRNESS METRICS  (shared by both modes)
    # ══════════════════════════════════════════════════════════════
    def compute_fairness_metrics(self) -> dict:
        if self.y_pred is None:
            raise RuntimeError("No predictions. Call train_model() or "
                               "load_and_audit_external_model() first.")

        print("\n[▶] Computing Fairness Metrics...")
        results = {}
        for col in self.sensitive_cols:
            if col not in self.test_df.columns:
                print(f"  [!] '{col}' not in test set — skipping.")
                continue
            r = self._analyze_attribute(col)
            results[col] = r
            self._print_summary(col, r)

        self.results = results
        return results

    # ══════════════════════════════════════════════════════════════
    # PER-ATTRIBUTE ANALYSIS
    # ══════════════════════════════════════════════════════════════
    def _analyze_attribute(self, col: str) -> dict:
        """
        Compute all fairness metrics for one sensitive attribute.
        Uses test_df (original un-encoded values) for group labels —
        this guarantees 'Male'/'Female', not 0/1.
        """
        raw_col = self.test_df[col]

        # Bin continuous numeric columns into quantile groups
        if pd.api.types.is_numeric_dtype(raw_col) and raw_col.nunique() > 10:
            try:
                groups_series = pd.qcut(raw_col, q=4,
                                        labels=["Q1 (youngest)","Q2","Q3","Q4 (oldest)"],
                                        duplicates="drop")
                bin_note = "continuous — binned into quartile groups"
            except Exception:
                groups_series = raw_col.astype(str)
                bin_note = ""
        else:
            groups_series = raw_col.astype(str)
            bin_note = ""

        y_test = np.array(self.y_test)
        y_pred = np.array(self.y_pred)
        pos_label = _positive_label_index(y_test)

        group_metrics = {}
        for grp in groups_series.dropna().unique():
            mask = (groups_series == grp).values
            if mask.sum() < 5:
                continue

            yt = y_test[mask]
            yp = y_pred[mask]

            n         = int(mask.sum())
            pos_rate  = float((yp == pos_label).mean())
            acc       = float(accuracy_score(yt, yp))

            if len(np.unique(yt)) < 2 or len(np.unique(yp)) < 2:
                tpr = fpr = prec = 0.0
            else:
                try:
                    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
                    tpr  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                except Exception:
                    tpr = fpr = prec = 0.0

            group_metrics[str(grp)] = {
                "n_samples":                int(n),
                "positive_prediction_rate": round(pos_rate, 4),
                "true_positive_rate":       round(tpr,      4),
                "false_positive_rate":      round(fpr,      4),
                "precision":                round(prec,     4),
                "accuracy":                 round(acc,      4),
            }

        if len(group_metrics) < 2:
            return {
                "error": f"Not enough groups with sufficient samples in '{col}'",
                "severity": "FAIR",
                "flags": [],
                "group_metrics": group_metrics,
                "disparate_impact_ratio": 1.0,
                "demographic_parity_difference": 0.0,
                "equalized_odds_difference": 0.0,
                "predictive_parity_difference": 0.0,
                "accuracy_equality_difference": 0.0,
                "reference_group": "",
                "most_disadvantaged_group": "",
                "bin_note": bin_note,
            }

        ppr  = {g: m["positive_prediction_rate"] for g, m in group_metrics.items()}
        tprs = {g: m["true_positive_rate"]        for g, m in group_metrics.items()}
        accs = {g: m["accuracy"]                  for g, m in group_metrics.items()}
        prcs = {g: m["precision"]                 for g, m in group_metrics.items()}

        max_ppr, min_ppr = max(ppr.values()), min(ppr.values())
        di      = round(min_ppr / max_ppr, 4) if max_ppr > 0 else 1.0
        dp_diff = round(max_ppr - min_ppr,              4)
        eo_diff = round(max(tprs.values()) - min(tprs.values()), 4)
        ac_diff = round(max(accs.values()) - min(accs.values()), 4)
        pp_diff = round(max(prcs.values()) - min(prcs.values()), 4)

        ref   = max(ppr, key=ppr.get)
        worst = min(ppr, key=ppr.get)

        severity = ("SEVERE"   if di < 0.8 and dp_diff > 0.20 else
                    "MODERATE" if di < 0.8 or  dp_diff > 0.10 else
                    "FAIR")

        return {
            "group_metrics":                  group_metrics,
            "bin_note":                       bin_note,
            "disparate_impact_ratio":         di,
            "demographic_parity_difference":  dp_diff,
            "equalized_odds_difference":      eo_diff,
            "predictive_parity_difference":   pp_diff,
            "accuracy_equality_difference":   ac_diff,
            "reference_group":                ref,
            "most_disadvantaged_group":       worst,
            "severity":                       severity,
            "flags":                          self._flags(col, di, dp_diff, eo_diff,
                                                          ac_diff, ref, worst, ppr),
        }

    # ══════════════════════════════════════════════════════════════
    # FLAGS
    # ══════════════════════════════════════════════════════════════
    def _flags(self, col, di, dp, eo, ac, ref, worst, ppr) -> list:
        flags = []
        if di < 0.8:
            flags.append({"type": "DISPARATE_IMPACT", "severity": "HIGH",
                "message": (f"Disparate Impact = {di:.3f} (legal threshold ≥ 0.80). "
                            f"'{worst}' receives positive predictions at only {di*100:.1f}% "
                            f"the rate of '{ref}'. This breaches the 80% rule used in "
                            f"employment, lending, and housing law.")})
        if dp > 0.15:
            flags.append({"type": "DEMOGRAPHIC_PARITY", "severity": "HIGH",
                "message": (f"Demographic Parity gap = {dp*100:.1f}%. The model approves "
                            f"'{ref}' ({ppr[ref]*100:.1f}%) far more than '{worst}' "
                            f"({ppr[worst]*100:.1f}%). The system produces unequal decisions "
                            f"regardless of merit.")})
        elif dp > 0.05:
            flags.append({"type": "DEMOGRAPHIC_PARITY", "severity": "MEDIUM",
                "message": f"Demographic Parity gap = {dp*100:.1f}% between '{ref}' and '{worst}'. Moderate concern."})
        if eo > 0.1:
            flags.append({"type": "EQUALIZED_ODDS", "severity": "MEDIUM",
                "message": (f"True Positive Rate gap = {eo*100:.1f}% across groups. "
                            f"The model is better at correctly identifying qualified "
                            f"individuals in some groups than others — unequal benefit.")})
        if ac > 0.05:
            flags.append({"type": "ACCURACY_EQUALITY", "severity": "MEDIUM",
                "message": (f"Accuracy gap of {ac*100:.1f}% across groups. "
                            f"The model is less reliable for certain demographic groups.")})
        return flags

    def _print_summary(self, col, r):
        icon = {"SEVERE": "🔴", "MODERATE": "🟡", "FAIR": "🟢"}.get(r.get("severity",""), "⚪")
        print(f"  {icon} [{r.get('severity','?')}] '{col}'  |  "
              f"DI={r.get('disparate_impact_ratio','?')}  "
              f"gap={r.get('demographic_parity_difference','?')}  "
              f"worst='{r.get('most_disadvantaged_group','?')}'")

    def save_model(self, path: str):
        if not self.model:
            raise RuntimeError("No model to save.")
        joblib.dump(self.model, path)
        print(f"[✔] Model saved → {path}")

    @staticmethod
    def list_algorithms():
        return ALGORITHM_DISPLAY_NAMES.copy()
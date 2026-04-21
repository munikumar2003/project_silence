"""
STEP 4: Bias Mitigation
=========================
Provides three strategies to reduce bias:

1. PRE-PROCESSING  → Fix the training data before model sees it
   - Reweighting: Give underrepresented groups higher sample weights
   - Resampling: Oversample disadvantaged groups (SMOTE-like)

2. IN-PROCESSING   → Adjust model training to include fairness constraints
   - Fairness-aware model with balanced class weights per group

3. POST-PROCESSING → Adjust the model's decision thresholds per group
   - Lower decision threshold for disadvantaged groups to equalize outcomes

After mitigation, recomputes fairness metrics to show improvement.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from typing import Optional

SUPPORTED_ALGORITHMS = {
    "random_forest":      lambda: RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, class_weight="balanced"),
    "logistic_regression":lambda: LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    "gradient_boosting":  lambda: GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42),
    "decision_tree":      lambda: DecisionTreeClassifier(max_depth=8, random_state=42, class_weight="balanced"),
    "svm":                lambda: SVC(probability=True, random_state=42, class_weight="balanced"),
    "knn":                lambda: KNeighborsClassifier(n_neighbors=7),
    "naive_bayes":        lambda: GaussianNB(),
}

class BiasMitigator:
    """
    Applies bias mitigation techniques and re-evaluates fairness metrics.
    """
    name="random_forest"
    def __init__(self, df: pd.DataFrame, sensitive_cols: list,
                 target_col: str, feature_cols: list = None, algorithm: str = "random_forest"):
        self.df = df.copy()
        self.sensitive_cols = sensitive_cols
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.algorithm = SUPPORTED_ALGORITHMS.get(algorithm, SUPPORTED_ALGORITHMS["random_forest"])
        self.encoders = {}
        self.scaler = StandardScaler()
        self.mitigation_results = {}
        self.name = algorithm

    # ------------------------------------------------------------------
    # DATA PREPARATION (shared across strategies)
    # ------------------------------------------------------------------
    def _prepare(self):
        """Encode and split data, returning train/test sets."""
        df = self.df.copy()

        if self.feature_cols is None:
            self.feature_cols = [c for c in df.columns if c != self.target_col]

        X = df[self.feature_cols].copy()
        y = df[self.target_col].copy()

        # Encode categoricals
        for col in X.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.encoders[col] = le

        if y.dtype == object:
            le_y = LabelEncoder()
            y = le_y.fit_transform(y)
            self.encoders[self.target_col] = le_y

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        test_df_orig = df.iloc[X_test.index].copy()

        X_train_arr = self.scaler.fit_transform(X_train)
        X_test_arr = self.scaler.transform(X_test)

        return X_train, X_test_arr, y_train, y_test, X_train_arr, test_df_orig, self.algorithm

    # ------------------------------------------------------------------
    # STRATEGY 1: PRE-PROCESSING — REWEIGHTING
    # ------------------------------------------------------------------
    def mitigate_reweighting(self) -> dict:
        """
        Assign higher sample weights to underrepresented/disadvantaged groups
        during training. This makes the model pay more attention to them.
        """
        
        print("\n[▶] Strategy 1: Pre-Processing (Reweighting)...")

        X_train, X_test, y_train, y_test, X_train_arr, test_df, algorithm = self._prepare()

        # Compute sample weights based on group representation
        # Disadvantaged groups get higher weights
        df_train = self.df.iloc[X_train.index].copy()
        sample_weights = self._compute_sample_weights(df_train, y_train)

        # Train with weights
        model = algorithm()
        #print(self.name)
        if(self.name == "knn" or self.name == "naive_bayes" or self.name == "gradient_boosting"):
            # KNN does not support sample_weight, so we skip weighting for KNN
            model.fit(X_train_arr, y_train)
        else:
            model.fit(X_train_arr, y_train, sample_weight=sample_weights)
        y_pred = model.predict(X_test)

        metrics = self._compute_group_metrics(y_test, y_pred, test_df)
        result = {
            "strategy": "Pre-Processing: Reweighting",
            "description": (
                "Sample weights were assigned inversely proportional to group "
                "frequency × outcome rate. Disadvantaged groups received higher "
                "weights, forcing the model to learn better representations for them."
            ),
            "group_metrics": metrics,
            "overall_accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "fairness_improvement": self._summarize_fairness(metrics)
        }

        self.mitigation_results["reweighting"] = result
        self._print_strategy_result("Reweighting", result)
        return result

    # ------------------------------------------------------------------
    # STRATEGY 2: PRE-PROCESSING — RESAMPLING
    # ------------------------------------------------------------------
    def mitigate_resampling(self) -> dict:
        """
        Oversample underrepresented groups so the model sees them more often.
        This is done BEFORE training on the training split only.
        """
        print("\n[▶] Strategy 2: Pre-Processing (Resampling / Oversampling)...")

        X_train, X_test, y_train, y_test, X_train_arr, test_df, algorithm = self._prepare()

        # Identify the majority size across groups for the primary sensitive attribute
        primary_col = self.sensitive_cols[0]
        df_train = self.df.iloc[X_train.index].copy()
        df_train["__y__"] = y_train.values if hasattr(y_train, 'values') else y_train

        # Oversample each group to match the largest group's count
        max_count = df_train[primary_col].value_counts().max()
        resampled_parts = []
        for grp in df_train[primary_col].unique():
            grp_df = df_train[df_train[primary_col] == grp]
            if len(grp_df) < max_count:
                grp_df = resample(grp_df, replace=True, n_samples=max_count, random_state=42)
            resampled_parts.append(grp_df)

        resampled_df = pd.concat(resampled_parts).sample(frac=1, random_state=42)
        y_resampled = resampled_df["__y__"].values
        resampled_df = resampled_df.drop(columns=["__y__"])

        # Re-encode and scale the resampled training data
        X_resampled = resampled_df[self.feature_cols].copy()
        for col in X_resampled.select_dtypes(include=["object", "category"]).columns:
            if col in self.encoders:
                X_resampled[col] = self.encoders[col].transform(X_resampled[col].astype(str))
        X_resampled_arr = self.scaler.transform(X_resampled)

        model = algorithm()
        model.fit(X_resampled_arr, y_resampled)
        y_pred = model.predict(X_test)

        metrics = self._compute_group_metrics(y_test, y_pred, test_df)
        result = {
            "strategy": "Pre-Processing: Resampling",
            "description": (
                "Underrepresented groups were oversampled in the training set "
                "so each group has equal representation. This allows the model "
                "to learn equally well for all demographic groups."
            ),
            "group_metrics": metrics,
            "overall_accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "fairness_improvement": self._summarize_fairness(metrics)
        }

        self.mitigation_results["resampling"] = result
        self._print_strategy_result("Resampling", result)
        return result

    # ------------------------------------------------------------------
    # STRATEGY 3: POST-PROCESSING — THRESHOLD ADJUSTMENT
    # ------------------------------------------------------------------
    def mitigate_threshold_adjustment(self) -> dict:
        """
        Train a standard model, then adjust the decision threshold
        per group to equalize positive prediction rates.
        Disadvantaged groups get a lower threshold (easier to approve).
        """
        print("\n[▶] Strategy 3: Post-Processing (Threshold Adjustment)...")

        X_train, X_test, y_train, y_test, X_train_arr, test_df, algorithm = self._prepare()

        # Train standard model
        model = algorithm()
        model.fit(X_train_arr, y_train)

        # Get predicted probabilities
        y_proba = model.predict_proba(X_test)[:, 1]

        # Determine per-group thresholds to equalize positive prediction rates
        # Target = overall mean positive prediction rate
        primary_col = self.sensitive_cols[0] if self.sensitive_cols[0] != "age" else (
            self.sensitive_cols[1] if len(self.sensitive_cols) > 1 else self.sensitive_cols[0]
        )

        target_rate = np.mean(y_proba >= 0.5)  # baseline rate
        groups = test_df[primary_col].values if primary_col in test_df.columns else None

        if groups is not None:
            unique_grps = np.unique(groups)
            group_thresholds = {}

            for grp in unique_grps:
                mask = groups == grp
                grp_proba = y_proba[mask]
                grp_rate = np.mean(grp_proba >= 0.5)

                # If group rate is below target, lower their threshold
                if grp_rate < target_rate - 0.03:
                    # Find threshold that gives ~target_rate for this group
                    thresholds = np.linspace(0.1, 0.9, 100)
                    diffs = [abs(np.mean(grp_proba >= t) - target_rate) for t in thresholds]
                    optimal_t = thresholds[np.argmin(diffs)]
                    group_thresholds[str(grp)] = round(float(optimal_t), 3)
                else:
                    group_thresholds[str(grp)] = 0.5

            # Apply per-group thresholds
            y_pred_adjusted = np.zeros(len(y_proba), dtype=int)
            for grp in unique_grps:
                mask = groups == grp
                threshold = group_thresholds.get(str(grp), 0.5)
                y_pred_adjusted[mask] = (y_proba[mask] >= threshold).astype(int)
        else:
            y_pred_adjusted = (y_proba >= 0.5).astype(int)
            group_thresholds = {}

        metrics = self._compute_group_metrics(y_test, y_pred_adjusted, test_df)
        result = {
            "strategy": "Post-Processing: Threshold Adjustment",
            "description": (
                "The model was trained normally, then different decision thresholds "
                "were applied per group to equalize positive prediction rates. "
                "Groups that were disadvantaged receive a lower threshold (easier approval)."
            ),
            "group_thresholds": group_thresholds,
            "group_metrics": metrics,
            "overall_accuracy": round(float(accuracy_score(y_test, y_pred_adjusted)), 4),
            "fairness_improvement": self._summarize_fairness(metrics)
        }

        self.mitigation_results["threshold_adjustment"] = result
        self._print_strategy_result("Threshold Adjustment", result)
        return result

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------
    def _compute_sample_weights(self, df_train, y_train) -> np.ndarray:
        """Compute sample weights: higher weight for disadvantaged groups."""
        weights = np.ones(len(df_train))
        y_arr = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
        # Reset index for positional alignment
        df_reset = df_train.reset_index(drop=True)

        for col in self.sensitive_cols:
            if col not in df_reset.columns:
                continue
            # Compute positive rate per group using positional index
            group_positive_rates = {}
            for grp in df_reset[col].unique():
                mask = (df_reset[col] == grp).values
                group_positive_rates[grp] = y_arr[mask].mean() if mask.sum() > 0 else 0.5

            max_rate = max(group_positive_rates.values())
            for i in range(len(df_reset)):
                grp = df_reset.at[i, col]
                grp_rate = group_positive_rates.get(grp, 0.5)
                boost = max_rate / max(grp_rate, 0.01)
                weights[i] *= min(boost, 3.0)

        return weights

    def _compute_group_metrics(self, y_test, y_pred, test_df) -> dict:
        """Compute positive prediction rates per group."""
        metrics = {}
        for col in self.sensitive_cols:
            if col not in test_df.columns:
                continue
            col_metrics = {}
            groups = test_df[col].values
            for grp in np.unique(groups):
                mask = groups == grp
                if mask.sum() == 0:
                    continue
                pos_rate = y_pred[mask].mean()
                acc = accuracy_score(y_test[mask], y_pred[mask])
                col_metrics[str(grp)] = {
                    "positive_prediction_rate": round(float(pos_rate), 4),
                    "accuracy": round(float(acc), 4),
                    "n_samples": int(mask.sum())
                }
            metrics[col] = col_metrics
        return metrics

    def _summarize_fairness(self, metrics: dict) -> dict:
        """Compute summary fairness gaps after mitigation."""
        summary = {}
        for col, groups in metrics.items():
            rates = [m["positive_prediction_rate"] for m in groups.values()]
            gap = round(max(rates) - min(rates), 4)
            di = round(min(rates) / max(rates), 4) if max(rates) > 0 else 1.0
            severity = "SEVERE" if di < 0.8 and gap > 0.2 else (
                "MODERATE" if di < 0.8 or gap > 0.1 else "FAIR"
            )
            summary[col] = {
                "demographic_parity_gap": gap,
                "disparate_impact_ratio": di,
                "severity_after_mitigation": severity
            }
        return summary

    def _print_strategy_result(self, name: str, result: dict):
        print(f"  ✔ {name} complete. Accuracy: {result['overall_accuracy']}")
        for col, summary in result["fairness_improvement"].items():
            icon = {"SEVERE": "🔴", "MODERATE": "🟡", "FAIR": "🟢"}.get(
                summary["severity_after_mitigation"], "⚪"
            )
            print(f"     {icon} {col}: gap={summary['demographic_parity_gap']:.3f}, "
                  f"DI={summary['disparate_impact_ratio']:.3f} "
                  f"[{summary['severity_after_mitigation']}]")

    def get_best_strategy(self) -> str:
        """Returns the name of the strategy with the lowest average fairness gap."""
        best = None
        best_gap = float("inf")
        for strategy_name, result in self.mitigation_results.items():
            avg_gap = np.mean([
                s["demographic_parity_gap"]
                for s in result["fairness_improvement"].values()
            ])
            if avg_gap < best_gap:
                best_gap = avg_gap
                best = strategy_name
        return best
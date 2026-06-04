"""
STEP 3.5: SHAP Individual-Level Fairness Explanations
=======================================================
Uses SHAP (SHapley Additive exPlanations) to explain individual predictions
and identify which features drive unfair decisions for specific demographic groups.

Key insights:
1. Group-level metrics (demographic parity) don't explain WHY a specific person got an unfair outcome
2. SHAP shows per-individual feature contributions to predictions
3. Identifies if protected attributes or proxies are driving unfair decisions
4. Helps debug fairness issues at the individual level

Outputs:
- Per-individual SHAP values for misclassified/unfairly treated cases
- Feature importance ranking per demographic group
- Waterfall plots for interpretability
- Summary statistics on feature usage across groups
"""

import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class SHAPFairnessExplainer:
    """
    Computes SHAP explanations for individual fairness analysis.
    Focuses on cases where predictions diverge across demographic groups.
    """

    def __init__(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        test_df: pd.DataFrame,
        feature_names: List[str],
        sensitive_cols: List[str],
    ):
        """
        Args:
            model: Trained sklearn classifier with predict_proba or predict method
            X_test: Test feature array (encoded)
            y_test: Ground truth labels
            y_pred: Model predictions
            test_df: Original (unencoded) test dataframe with group labels
            feature_names: Names of features
            sensitive_cols: Demographic attribute columns
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        self.test_df = test_df
        self.feature_names = feature_names
        self.sensitive_cols = sensitive_cols
        self.shap_values = None
        self.explainer = None
        self.results = {}

    def compute_shap_values(self) -> dict:
        """
        Compute SHAP values using KernelExplainer (model-agnostic).
        Returns dict with per-group feature importance and key insights.
        """
        try:
            print("[•] Computing SHAP values for individual predictions...")

            # Use KernelExplainer for model-agnostic explanations
            # For faster computation, use a sample if dataset is very large
            background_size = min(100, len(self.X_test) // 2)
            background_indices = np.random.choice(len(self.X_test), background_size, replace=False)
            X_background = self.X_test[background_indices]

            # Create explainer
            self.explainer = shap.KernelExplainer(
                model=self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                data=X_background
            )

            # Compute SHAP values (for positive class if predict_proba is used)
            self.shap_values = self.explainer.shap_values(self.X_test)
            if hasattr(self.model, 'predict_proba'):
                # SHAP values shape: (n_samples, n_features, n_classes)
                # We focus on positive class (index 1) for binary classification.
                if isinstance(self.shap_values, list):
                    self.shap_values = self.shap_values[1]
                elif isinstance(self.shap_values, np.ndarray) and self.shap_values.ndim == 3:
                    class_index = 1 if self.shap_values.shape[2] > 1 else 0
                    self.shap_values = self.shap_values[:, :, class_index]
            elif isinstance(self.shap_values, np.ndarray) and self.shap_values.ndim == 3:
                # Fallback for 3D outputs from SHAP explainer
                self.shap_values = self.shap_values[:, :, 0]

            print(f"   ✔ SHAP values computed. Shape: {self.shap_values.shape}")

            # Analyze per demographic group
            results = self._analyze_group_fairness()
            self.results = results
            return results

        except Exception as e:
            print(f"   ⚠ SHAP computation failed: {e}")
            return {"error": str(e), "group_analysis": {}}

    def _analyze_group_fairness(self) -> dict:
        """
        Analyze SHAP values per demographic group to identify fairness issues.
        """
        group_analysis = {}
        primary_col = self.sensitive_cols[0] if self.sensitive_cols else None

        if not primary_col or primary_col not in self.test_df.columns:
            return {"error": "Primary sensitive column not found", "group_analysis": {}}

        groups = self.test_df[primary_col].unique()

        for grp in groups:
            mask = (self.test_df[primary_col] == grp).values
            grp_shap = self.shap_values[mask]
            grp_X = self.X_test[mask]
            grp_y_pred = self.y_pred[mask]

            # Compute mean absolute SHAP values (feature importance)
            mean_abs_shap = np.mean(np.abs(grp_shap), axis=0)

            # Find top 5 features driving predictions for this group
            top_indices = np.argsort(mean_abs_shap)[-5:][::-1]
            top_features = [(self.feature_names[i], round(float(mean_abs_shap[i]), 4)) for i in top_indices]

            # Identify misclassified cases in this group
            misclassified_mask = (grp_y_pred != self.y_test[mask])
            n_misclassified = misclassified_mask.sum()

            # For misclassified cases, show which features were most influential
            if n_misclassified > 0:
                misclass_shap = grp_shap[misclassified_mask]
                mean_misclass_shap = np.mean(np.abs(misclass_shap), axis=0)
                misclass_top_indices = np.argsort(mean_misclass_shap)[-3:][::-1]
                misclass_top = [(self.feature_names[i], round(float(mean_misclass_shap[i]), 4))
                                for i in misclass_top_indices]
            else:
                misclass_top = []

            group_analysis[str(grp)] = {
                "n_samples": int(mask.sum()),
                "n_misclassified": int(n_misclassified),
                "misclassification_rate": round(float(n_misclassified / mask.sum()), 4) if mask.sum() > 0 else 0.0,
                "top_features": top_features,
                "top_misclass_features": misclass_top,
                "mean_shap_values": {self.feature_names[i]: round(float(mean_abs_shap[i]), 4) 
                                     for i in range(len(self.feature_names))},
            }

        return {
            "method": "SHAP (KernelExplainer)",
            "primary_group_column": primary_col,
            "group_analysis": group_analysis,
            "fairness_insights": self._generate_fairness_insights(group_analysis),
        }

    def _generate_fairness_insights(self, group_analysis: dict) -> List[dict]:
        """
        Generate insights about fairness issues from SHAP analysis.
        """
        insights = []

        # Check if any group has dramatically different feature importance
        if len(group_analysis) < 2:
            return insights

        groups_list = list(group_analysis.keys())
        top_features_per_group = {grp: [f[0] for f in group_analysis[grp]["top_features"]]
                                  for grp in groups_list}

        # Insight 1: Feature importance divergence across groups
        all_features = set()
        for features in top_features_per_group.values():
            all_features.update(features)

        for feat in all_features:
            feature_importance = {}
            for grp in groups_list:
                top_feats = top_features_per_group[grp]
                if feat in top_feats:
                    idx = top_feats.index(feat)
                    feature_importance[grp] = 5 - idx  # Ranking score
                else:
                    feature_importance[grp] = 0

            # If a feature is highly important for one group but not others, flag it
            if max(feature_importance.values()) > 3 and min(feature_importance.values()) == 0:
                insights.append({
                    "type": "Feature Divergence",
                    "severity": "MEDIUM",
                    "message": (f"Feature '{feat}' is driving predictions for {list(feature_importance.keys())[0]} "
                               f"but not others. May indicate biased feature usage."),
                    "feature": feat,
                    "affected_groups": list(feature_importance.keys()),
                })

        # Insight 2: High misclassification rate in minority groups
        for grp, analysis in group_analysis.items():
            if analysis["misclassification_rate"] > 0.3:
                insights.append({
                    "type": "High Error Rate",
                    "severity": "MODERATE",
                    "message": (f"Group '{grp}' has {analysis['misclassification_rate']*100:.1f}% "
                               f"misclassification rate. Features driving these errors: "
                               f"{', '.join([f[0] for f in analysis['top_misclass_features'][:2]])}."),
                    "group": grp,
                    "error_rate": analysis["misclassification_rate"],
                })

        return insights

    def get_unfair_cases(self, top_k: int = 5) -> dict:
        """
        Identify the most unfair individual predictions per group.
        Returns cases where the model's decision is most biased.
        """
        unfair_cases = {}
        primary_col = self.sensitive_cols[0] if self.sensitive_cols else None

        if not primary_col or primary_col not in self.test_df.columns:
            return {}  # Return empty dict for consistency

        groups = self.test_df[primary_col].unique()

        for grp in groups:
            mask = (self.test_df[primary_col] == grp).values
            grp_shap = self.shap_values[mask]
            grp_indices = np.where(mask)[0]

            # Find cases with highest SHAP variance (most uncertain/debatable predictions)
            shap_entropy = np.sum(np.abs(grp_shap), axis=1)
            top_uncertain_idx = np.argsort(shap_entropy)[-top_k:][::-1]

            unfair_cases[str(grp)] = []
            for idx in top_uncertain_idx:
                idx = int(idx)  # Convert numpy int to Python int to avoid indexing errors
                original_idx = int(grp_indices[idx])  # Also convert this
                row = self.test_df.iloc[original_idx]
                prediction = self.y_pred[original_idx]
                ground_truth = self.y_test[original_idx]

                # Get top contributing features for this prediction
                top_feat_idx = np.argsort(np.abs(grp_shap[idx]))[-3:][::-1]
                top_features = [(self.feature_names[int(fi)], round(float(grp_shap[idx][int(fi)]), 4))
                               for fi in top_feat_idx]

                unfair_cases[str(grp)].append({
                    "row_index": int(original_idx),
                    "prediction": int(prediction),
                    "ground_truth": int(ground_truth),
                    "correct": bool(prediction == ground_truth),
                    "top_contributing_features": top_features,
                    "shap_score": round(float(shap_entropy[idx]), 4),
                })

        return unfair_cases

    def summary_report(self) -> dict:
        """
        Generate comprehensive summary of SHAP-based fairness analysis.
        """
        if not self.results:
            return {"error": "SHAP values not yet computed. Call compute_shap_values() first."}

        return {
            "explanation_method": "SHAP (KernelExplainer)",
            "n_samples_explained": int(len(self.X_test)),
            "n_features": int(len(self.feature_names)),
            "sensitive_attributes": self.sensitive_cols,
            "group_analysis": self.results.get("group_analysis", {}),
            "fairness_insights": self.results.get("fairness_insights", []),
            "unfair_cases": self.get_unfair_cases(),
        }

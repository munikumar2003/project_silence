"""
STEP 2: Dataset Bias Detection
================================
Inspects the RAW DATA (before any model) for inherent bias.

Metrics computed:
1. Class Imbalance Score    - Are sensitive groups evenly represented?
2. Label Bias Score         - Do certain groups get favorable outcomes more in historical data?
3. Disparate Impact Ratio   - (Most favorable group) / (Least favorable group) outcome rate
4. Statistical Parity Diff  - Max difference in positive outcome rates across groups
5. Flagging                 - Severity levels: SEVERE / MODERATE / FAIR
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional


# Thresholds for flagging
DISPARATE_IMPACT_THRESHOLD = 0.8      # < 0.8 = legally problematic (80% rule)
STAT_PARITY_DIFF_THRESHOLD = 0.1     # > 10% difference = moderate concern
STAT_PARITY_DIFF_SEVERE = 0.2        # > 20% difference = severe
IMBALANCE_THRESHOLD = 0.3            # group < 30% of dominant = imbalanced


class DatasetBiasDetector:
    """
    Detects bias in a raw dataset across sensitive attribute groups.
    Works entirely on the data BEFORE any model is trained.
    """

    def __init__(self, df: pd.DataFrame, sensitive_cols: list, target_col: str):
        self.df = df.copy()
        self.sensitive_cols = sensitive_cols
        self.target_col = target_col
        self.results = {}

    # ------------------------------------------------------------------
    # MAIN ENTRY POINT
    # ------------------------------------------------------------------
    def run_all_checks(self) -> dict:
        """Run all dataset bias checks and return full results."""
        print("\n[▶] Running Dataset Bias Detection...")

        all_results = {}
        for col in self.sensitive_cols:
            if col == "age":
                # For age (continuous), bin it first
                df_work = self.df.copy()
                df_work["age_group"] = pd.cut(
                    df_work["age"],
                    bins=[0, 30, 45, 60, 100],
                    labels=["18-30", "31-45", "46-60", "60+"]
                )
                col_result = self._analyze_attribute(df_work, "age_group")
            else:
                col_result = self._analyze_attribute(self.df, col)

            all_results[col] = col_result
            self._print_summary(col, col_result)

        self.results = all_results
        return all_results

    # ------------------------------------------------------------------
    # PER-ATTRIBUTE ANALYSIS
    # ------------------------------------------------------------------
    def _analyze_attribute(self, df: pd.DataFrame, col: str) -> dict:
        """Full bias analysis for a single sensitive attribute."""
        result = {}

        # 1. Group sizes and representation
        group_counts = df[col].value_counts()
        total = len(df)
        group_sizes = {str(k): int(v) for k, v in group_counts.items()}
        group_pcts = {str(k): round(float(v / total * 100), 2) for k, v in group_counts.items()}

        # 2. Positive outcome rates per group
        positive_label = self._get_positive_label(df)
        outcome_rates = {}
        for grp in group_counts.index:
            subset = df[df[col] == grp]
            rate = (subset[self.target_col] == positive_label).sum() / len(subset)
            outcome_rates[str(grp)] = round(float(rate), 4)

        # 3. Class Imbalance Score
        max_count = group_counts.max()
        imbalance_flags = {
            str(k): v < max_count * IMBALANCE_THRESHOLD
            for k, v in group_counts.items()
        }
        imbalance_score = round(float(group_counts.min() / group_counts.max()), 4)

        # 4. Disparate Impact Ratio
        rates = list(outcome_rates.values())
        if max(rates) > 0:
            disparate_impact = round(min(rates) / max(rates), 4)
        else:
            disparate_impact = 1.0

        # 5. Statistical Parity Difference
        stat_parity_diff = round(max(rates) - min(rates), 4)

        # 6. Chi-Square Test (statistical significance)
        contingency = pd.crosstab(df[col], df[self.target_col])
        chi2, p_value, _, _ = stats.chi2_contingency(contingency)
        statistically_significant = bool(p_value < 0.05)

        # 7. Identify most / least advantaged groups
        best_group = max(outcome_rates, key=outcome_rates.get)
        worst_group = min(outcome_rates, key=outcome_rates.get)

        # 8. Severity flag
        severity = self._compute_severity(disparate_impact, stat_parity_diff)

        result = {
            "group_sizes": group_sizes,
            "group_percentages": group_pcts,
            "outcome_rates": outcome_rates,
            "imbalance_score": imbalance_score,
            "imbalanced_groups": imbalance_flags,
            "disparate_impact_ratio": disparate_impact,
            "statistical_parity_difference": stat_parity_diff,
            "chi2_p_value": round(float(p_value), 6),
            "statistically_significant": statistically_significant,
            "most_advantaged_group": best_group,
            "least_advantaged_group": worst_group,
            "severity": severity,
            "flags": self._generate_flags(
                col, disparate_impact, stat_parity_diff,
                imbalance_score, statistically_significant,
                best_group, worst_group, outcome_rates
            )
        }
        return result

    # ------------------------------------------------------------------
    # SEVERITY CLASSIFICATION
    # ------------------------------------------------------------------
    def _compute_severity(self, disparate_impact: float, stat_parity_diff: float) -> str:
        """
        Classify overall bias severity for this attribute.
        SEVERE   → Disparate Impact < 0.8 AND parity diff > 20%
        MODERATE → Disparate Impact < 0.8 OR parity diff > 10%
        FAIR     → All metrics within acceptable range
        """
        if disparate_impact < DISPARATE_IMPACT_THRESHOLD and stat_parity_diff > STAT_PARITY_DIFF_SEVERE:
            return "SEVERE"
        elif disparate_impact < DISPARATE_IMPACT_THRESHOLD or stat_parity_diff > STAT_PARITY_DIFF_THRESHOLD:
            return "MODERATE"
        else:
            return "FAIR"

    # ------------------------------------------------------------------
    # FLAG GENERATION
    # ------------------------------------------------------------------
    def _generate_flags(self, col, disparate_impact, stat_parity_diff,
                        imbalance_score, significant, best_group, worst_group, rates) -> list:
        """Generate human-readable warning flags."""
        flags = []

        if disparate_impact < DISPARATE_IMPACT_THRESHOLD:
            flags.append({
                "type": "DISPARATE_IMPACT",
                "severity": "HIGH",
                "message": (
                    f"Disparate Impact Ratio = {disparate_impact:.3f} "
                    f"(threshold: {DISPARATE_IMPACT_THRESHOLD}). "
                    f"Group '{worst_group}' receives positive outcomes at only "
                    f"{disparate_impact*100:.1f}% the rate of '{best_group}'. "
                    f"This may violate the 80% rule in employment/lending law."
                )
            })

        if stat_parity_diff > STAT_PARITY_DIFF_SEVERE:
            flags.append({
                "type": "STATISTICAL_PARITY",
                "severity": "HIGH",
                "message": (
                    f"Outcome gap of {stat_parity_diff*100:.1f}% between "
                    f"'{best_group}' ({rates[best_group]*100:.1f}%) and "
                    f"'{worst_group}' ({rates[worst_group]*100:.1f}%). "
                    f"This is a severe disparity in historical data."
                )
            })
        elif stat_parity_diff > STAT_PARITY_DIFF_THRESHOLD:
            flags.append({
                "type": "STATISTICAL_PARITY",
                "severity": "MEDIUM",
                "message": (
                    f"Outcome gap of {stat_parity_diff*100:.1f}% between "
                    f"'{best_group}' and '{worst_group}'. Moderate concern."
                )
            })

        if imbalance_score < IMBALANCE_THRESHOLD:
            flags.append({
                "type": "CLASS_IMBALANCE",
                "severity": "MEDIUM",
                "message": (
                    f"Some groups in '{col}' are significantly underrepresented. "
                    f"Minority-to-majority ratio = {imbalance_score:.2f}. "
                    f"This can cause the model to learn less reliable patterns for small groups."
                )
            })

        if significant:
            flags.append({
                "type": "STATISTICAL_SIGNIFICANCE",
                "severity": "INFO",
                "message": (
                    f"Chi-square test confirms that outcome differences across '{col}' groups "
                    f"are statistically significant (not due to chance). Bias is real in this data."
                )
            })

        return flags

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------
    def _get_positive_label(self, df=None):
        if df is None:
            df = self.df
        target_vals = df[self.target_col].unique()
        for val in target_vals:
            if str(val).lower() in ["1", "yes", "true", "approved", "hired", "positive"]:
                return val
        return target_vals[0]

    def _print_summary(self, col: str, result: dict):
        severity_icon = {"SEVERE": "🔴", "MODERATE": "🟡", "FAIR": "🟢"}.get(result["severity"], "⚪")
        print(f"\n  {severity_icon} [{result['severity']}] Attribute: '{col}'")
        print(f"     Disparate Impact Ratio : {result['disparate_impact_ratio']}")
        print(f"     Stat. Parity Difference: {result['statistical_parity_difference']}")
        print(f"     Most advantaged group  : {result['most_advantaged_group']} "
              f"({result['outcome_rates'][result['most_advantaged_group']]*100:.1f}%)")
        print(f"     Least advantaged group : {result['least_advantaged_group']} "
              f"({result['outcome_rates'][result['least_advantaged_group']]*100:.1f}%)")
        print(f"     Flags: {len(result['flags'])} issue(s) found")
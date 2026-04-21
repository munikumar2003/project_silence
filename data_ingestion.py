"""
STEP 1: Data Ingestion & Profiling
===================================
Responsible for:
- Loading datasets (CSV)
- Detecting sensitive/protected attributes
- Profiling the dataset (distributions, missing values, class balance)
- Returning a structured summary for downstream bias analysis
"""

import pandas as pd
import numpy as np
from typing import Optional


# Common protected/sensitive attribute names to auto-detect
SENSITIVE_KEYWORDS = [
    "gender", "sex", "race", "ethnicity", "religion", "nationality",
    "age", "disability", "marital", "pregnancy", "color", "caste",
    "tribe", "origin", "sexual_orientation", "income_group"
]


class DataIngestor:
    """
    Loads a dataset and produces a full profile including:
    - Shape, dtypes, missing values
    - Auto-detected sensitive columns
    - Target column distribution
    - Per-group outcome rates
    """

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.sensitive_cols: list = []
        self.target_col: Optional[str] = None
        self.profile: dict = {}

    # ------------------------------------------------------------------
    # 1. LOAD DATA
    # ------------------------------------------------------------------
    def load_csv(self, filepath: str) -> pd.DataFrame:
        """Load a CSV file into a DataFrame."""
        try:
            self.df = pd.read_csv(filepath)
            print(f"[✔] Loaded dataset: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
            return self.df
        except Exception as e:
            raise ValueError(f"Failed to load CSV: {e}")

    def load_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Accept a pre-loaded DataFrame directly."""
        self.df = df.copy()
        return self.df

    # ------------------------------------------------------------------
    # 2. AUTO-DETECT SENSITIVE COLUMNS
    # ------------------------------------------------------------------
    def detect_sensitive_columns(self, user_defined: list = None) -> list:
        """
        Auto-detects columns that likely represent protected attributes
        by matching column names against known sensitive keywords.
        User can also pass their own list to override/extend.
        """
        if self.df is None:
            raise RuntimeError("No data loaded. Call load_csv() first.")

        auto_detected = []
        for col in self.df.columns:
            col_lower = col.lower().replace(" ", "_")
            if any(keyword in col_lower for keyword in SENSITIVE_KEYWORDS):
                auto_detected.append(col)

        # Merge with user-defined if provided
        if user_defined:
            combined = list(set(auto_detected + user_defined))
        else:
            combined = auto_detected

        self.sensitive_cols = combined
        print(f"[✔] Sensitive columns detected: {self.sensitive_cols}")
        return self.sensitive_cols

    # ------------------------------------------------------------------
    # 3. SET TARGET COLUMN
    # ------------------------------------------------------------------
    def set_target_column(self, target_col: str):
        """
        Define which column is the outcome/label we want to audit
        (e.g., 'loan_approved', 'hired', 'diagnosis').
        """
        if target_col not in self.df.columns:
            raise ValueError(f"Column '{target_col}' not found in dataset.")
        self.target_col = target_col
        print(f"[✔] Target column set to: '{self.target_col}'")

    # ------------------------------------------------------------------
    # 4. PROFILE THE DATASET
    # ------------------------------------------------------------------
    def profile_dataset(self) -> dict:
        """
        Generates a complete profile of the dataset including:
        - Basic info (shape, dtypes, missing values)
        - Target column distribution
        - Sensitive attribute value distributions
        - Per-group outcome rates (for bias preview)
        """
        if self.df is None:
            raise RuntimeError("No data loaded.")

        df = self.df
        profile = {}

        # --- 4a. Basic Info ---
        profile["shape"] = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}
        profile["column_names"] = list(df.columns)
        profile["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}

        # --- 4b. Missing Values ---
        missing = df.isnull().sum()
        profile["missing_values"] = {
            col: {
                "count": int(missing[col]),
                "percent": round(float(missing[col] / len(df) * 100), 2)
            }
            for col in df.columns if missing[col] > 0
        }

        # --- 4c. Target Distribution ---
        if self.target_col:
            target_counts = df[self.target_col].value_counts()
            profile["target_distribution"] = {
                "column": self.target_col,
                "counts": {str(k): int(v) for k, v in target_counts.items()},
                "percentages": {
                    str(k): round(float(v / len(df) * 100), 2)
                    for k, v in target_counts.items()
                }
            }

        # --- 4d. Sensitive Attribute Distributions ---
        profile["sensitive_attributes"] = {}
        for col in self.sensitive_cols:
            col_counts = df[col].value_counts()
            profile["sensitive_attributes"][col] = {
                "unique_values": int(df[col].nunique()),
                "counts": {str(k): int(v) for k, v in col_counts.items()},
                "percentages": {
                    str(k): round(float(v / len(df) * 100), 2)
                    for k, v in col_counts.items()
                }
            }

        # --- 4e. Per-Group Outcome Rates (Bias Preview) ---
        if self.target_col and self.sensitive_cols:
            profile["group_outcome_rates"] = {}
            target_positive = self._get_positive_label()

            for col in self.sensitive_cols:
                rates = {}
                grouped = df.groupby(col)[self.target_col]
                for group_val, group_data in grouped:
                    positive_rate = (group_data == target_positive).sum() / len(group_data)
                    rates[str(group_val)] = round(float(positive_rate * 100), 2)
                profile["group_outcome_rates"][col] = rates

        self.profile = profile
        print("[✔] Dataset profiling complete.")
        return profile

    # ------------------------------------------------------------------
    # HELPER
    # ------------------------------------------------------------------
    def _get_positive_label(self):
        """
        Infers the 'positive' label from the target column.
        For binary: returns the majority label (commonly 1 or 'Yes').
        """
        target_vals = self.df[self.target_col].unique()
        # Prefer 1, True, 'yes', 'approved', 'hired' as positive
        for val in target_vals:
            if str(val).lower() in ["1", "yes", "true", "approved", "hired", "positive"]:
                return val
        return target_vals[0]  # fallback: first value

    def get_clean_df(self) -> pd.DataFrame:
        """Returns a copy of the loaded DataFrame."""
        return self.df.copy()

    def summary(self) -> dict:
        """Returns the full ingestion summary."""
        return {
            "shape": self.df.shape if self.df is not None else None,
            "sensitive_columns": self.sensitive_cols,
            "target_column": self.target_col,
            "profile": self.profile
        }
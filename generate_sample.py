"""
Sample Dataset Generator
=========================
Creates a realistic biased loan approval dataset for testing the bias auditor.
The dataset deliberately encodes bias against:
- Women (lower approval rates)
- Certain racial groups
- Older applicants

This simulates what real historical data often looks like.
"""

import pandas as pd
import numpy as np

def generate_loan_dataset(n_samples=1000, random_seed=42):
    np.random.seed(random_seed)

    # --- Demographics ---
    gender = np.random.choice(["Male", "Female"], size=n_samples, p=[0.55, 0.45])
    race = np.random.choice(
        ["White", "Black", "Hispanic", "Asian"],
        size=n_samples,
        p=[0.50, 0.20, 0.18, 0.12]
    )
    age = np.random.randint(22, 65, size=n_samples)

    # --- Financial Features ---
    income = np.random.normal(55000, 18000, size=n_samples).clip(15000, 150000).astype(int)
    credit_score = np.random.normal(670, 80, size=n_samples).clip(300, 850).astype(int)
    debt_ratio = np.random.uniform(0.1, 0.6, size=n_samples).round(2)
    years_employed = np.random.randint(0, 30, size=n_samples)

    # --- Biased Approval Logic ---
    # Base approval probability from financial features
    base_prob = (
        (credit_score - 300) / 550 * 0.5 +
        (income - 15000) / 135000 * 0.3 +
        (1 - debt_ratio) * 0.2
    )

    # Inject BIAS: disadvantage certain groups (simulating historical discrimination)
    gender_penalty = np.where(gender == "Female", -0.12, 0.0)
    race_penalty = np.where(race == "Black", -0.14,
                   np.where(race == "Hispanic", -0.10, 0.0))
    age_penalty = np.where(age > 55, -0.08, 0.0)

    final_prob = (base_prob + gender_penalty + race_penalty + age_penalty).clip(0, 1)
    approved = (np.random.uniform(0, 1, n_samples) < final_prob).astype(int)

    df = pd.DataFrame({
        "age": age,
        "gender": gender,
        "race": race,
        "income": income,
        "credit_score": credit_score,
        "debt_ratio": debt_ratio,
        "years_employed": years_employed,
        "loan_approved": approved
    })

    return df


if __name__ == "__main__":
    df = generate_loan_dataset()
    df.to_csv("loan_data.csv", index=False)
    print(f"[✔] Sample dataset saved: {df.shape[0]} rows × {df.shape[1]} columns")
    print("\nApproval rate by gender:")
    print(df.groupby("gender")["loan_approved"].mean().round(3))
    print("\nApproval rate by race:")
    print(df.groupby("race")["loan_approved"].mean().round(3))